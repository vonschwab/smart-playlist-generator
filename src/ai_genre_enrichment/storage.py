from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from .models import AUTHORITATIVE_SOURCE_TYPES, RESPONSE_SCHEMA_VERSION

VALID_STATUSES = {"pending", "complete", "failed", "skipped", "needs_review"}
AUTO_APPLY_MIN_CONFIDENCE = 0.85
BROAD_AUTO_APPLY_BLOCKLIST = {
    "alternative",
    "alternative rock",
    "electronic",
    "experimental",
    "folk",
    "hip hop",
    "indie rock",
    "jazz",
    "pop",
    "rock",
}
NON_GENRE_AUTO_APPLY_BLOCKLIST = {
    "improvisation",
    "meditation",
    "oakland",
    "saxophone",
}
REVIEW_ONLY_REASON_HINTS = (
    "album title",
    "track title",
    "track names",
    "suggest",
    "suggests",
    "typical",
    "expect",
    "possible",
    "potential",
    "commonly associated",
)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_release_checks_cache_identity(conn: sqlite3.Connection) -> None:
    full_identity = [
        "release_key",
        "input_hash",
        "prompt_version",
        "taxonomy_version",
        "model",
        "web_mode",
        "source_evidence_hash",
        "response_schema_version",
    ]
    unique_indexes = []
    for index_row in conn.execute("PRAGMA index_list(ai_genre_release_checks)"):
        if not index_row["unique"]:
            continue
        index_name = index_row["name"]
        unique_indexes.append([row["name"] for row in conn.execute(f"PRAGMA index_info({index_name})")])
    if full_identity in unique_indexes:
        return

    conn.executescript(
        """
        ALTER TABLE ai_genre_release_checks RENAME TO ai_genre_release_checks_old;

        CREATE TABLE ai_genre_release_checks (
            check_id INTEGER PRIMARY KEY AUTOINCREMENT,
            release_key TEXT NOT NULL,
            normalized_artist TEXT NOT NULL,
            normalized_album TEXT NOT NULL,
            album_id TEXT,
            mbid TEXT,
            discogs_id TEXT,
            identifiers_json TEXT NOT NULL DEFAULT '{}',
            input_hash TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            taxonomy_version TEXT NOT NULL,
            model TEXT NOT NULL,
            web_mode TEXT NOT NULL DEFAULT 'off',
            source_evidence_hash TEXT NOT NULL DEFAULT 'none',
            response_schema_version TEXT NOT NULL DEFAULT 'ai-genre-response-v1',
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'complete', 'failed', 'skipped', 'needs_review')
            ),
            checked_at TEXT NOT NULL,
            response_json TEXT,
            error_message TEXT,
            overall_confidence REAL,
            evidence_quality TEXT,
            auto_apply_eligible INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            estimated_cost_usd REAL,
            UNIQUE (
                release_key, input_hash, prompt_version, taxonomy_version,
                model, web_mode, source_evidence_hash, response_schema_version
            )
        );

        INSERT INTO ai_genre_release_checks (
            check_id, release_key, normalized_artist, normalized_album, album_id,
            mbid, discogs_id, identifiers_json, input_hash, prompt_version,
            taxonomy_version, model, web_mode, source_evidence_hash,
            response_schema_version, status, checked_at, response_json,
            error_message, overall_confidence, evidence_quality,
            auto_apply_eligible, input_tokens, output_tokens, total_tokens,
            estimated_cost_usd
        )
        SELECT
            check_id, release_key, normalized_artist, normalized_album, album_id,
            mbid, discogs_id, identifiers_json, input_hash, prompt_version,
            taxonomy_version, model, web_mode, source_evidence_hash,
            response_schema_version, status, checked_at, response_json,
            error_message, overall_confidence, evidence_quality,
            auto_apply_eligible, input_tokens, output_tokens, total_tokens,
            estimated_cost_usd
        FROM ai_genre_release_checks_old;

        DROP TABLE ai_genre_release_checks_old;

        CREATE INDEX IF NOT EXISTS idx_ai_genre_checks_release
            ON ai_genre_release_checks (release_key, status);
        """
    )


class SidecarStore:
    """SQLite sidecar store for AI genre checks and recommendations."""

    def __init__(self, db_path: str | Path = "data/ai_genre_enrichment.db") -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ai_genre_release_checks (
                    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_key TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    album_id TEXT,
                    mbid TEXT,
                    discogs_id TEXT,
                    identifiers_json TEXT NOT NULL DEFAULT '{}',
                    input_hash TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    web_mode TEXT NOT NULL DEFAULT 'off',
                    source_evidence_hash TEXT NOT NULL DEFAULT 'none',
                    response_schema_version TEXT NOT NULL DEFAULT 'ai-genre-response-v1',
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'complete', 'failed', 'skipped', 'needs_review')
                    ),
                    checked_at TEXT NOT NULL,
                    response_json TEXT,
                    error_message TEXT,
                    overall_confidence REAL,
                    evidence_quality TEXT,
                    auto_apply_eligible INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    UNIQUE (
                        release_key, input_hash, prompt_version, taxonomy_version,
                        model, web_mode, source_evidence_hash, response_schema_version
                    )
                );

                CREATE TABLE IF NOT EXISTS ai_genre_suggestions (
                    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_id INTEGER NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    genre TEXT,
                    descriptor_tag TEXT,
                    confidence REAL,
                    reason TEXT,
                    prune_type TEXT,
                    recommendation_basis TEXT,
                    supporting_source_indexes_json TEXT,
                    descriptor_or_genre TEXT,
                    auto_apply_eligible INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (check_id) REFERENCES ai_genre_release_checks(check_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ai_genre_run_log (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    command TEXT NOT NULL,
                    status TEXT NOT NULL,
                    releases_seen INTEGER NOT NULL DEFAULT 0,
                    releases_called INTEGER NOT NULL DEFAULT 0,
                    releases_skipped INTEGER NOT NULL DEFAULT 0,
                    releases_failed INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    skipped_well_tagged INTEGER NOT NULL DEFAULT 0,
                    no_web_checks INTEGER NOT NULL DEFAULT 0,
                    authoritative_source_checks INTEGER NOT NULL DEFAULT 0,
                    needs_review INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ai_genre_source_pages (
                    source_page_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_key TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    album_id TEXT,
                    source_url TEXT NOT NULL,
                    source_domain TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    identity_status TEXT NOT NULL,
                    identity_confidence REAL NOT NULL DEFAULT 0,
                    fetched_at TEXT,
                    extraction_status TEXT NOT NULL DEFAULT 'pending',
                    extraction_hash TEXT NOT NULL DEFAULT 'none',
                    evidence_summary TEXT,
                    UNIQUE (release_key, source_url)
                );

                CREATE TABLE IF NOT EXISTS ai_genre_source_tags (
                    source_tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_page_id INTEGER NOT NULL,
                    raw_tag TEXT NOT NULL,
                    normalized_tag TEXT NOT NULL,
                    tag_position INTEGER,
                    extracted_at TEXT NOT NULL,
                    FOREIGN KEY (source_page_id)
                        REFERENCES ai_genre_source_pages(source_page_id) ON DELETE CASCADE,
                    UNIQUE (source_page_id, normalized_tag)
                );

                CREATE TABLE IF NOT EXISTS ai_genre_tag_classifications (
                    classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_tag_id INTEGER NOT NULL,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    classifier TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    classified_at TEXT NOT NULL,
                    FOREIGN KEY (source_tag_id)
                        REFERENCES ai_genre_source_tags(source_tag_id) ON DELETE CASCADE,
                    UNIQUE (source_tag_id, classifier)
                );

                CREATE TABLE IF NOT EXISTS enriched_genres (
                    enriched_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_key TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    album_id TEXT,
                    genre TEXT NOT NULL,
                    basis TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    source_tag_id INTEGER,
                    source_page_id INTEGER,
                    source_ref TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    added_at TEXT NOT NULL,
                    FOREIGN KEY (source_tag_id) REFERENCES ai_genre_source_tags(source_tag_id),
                    FOREIGN KEY (source_page_id) REFERENCES ai_genre_source_pages(source_page_id),
                    UNIQUE (release_key, genre, basis, source_ref)
                );

                CREATE TABLE IF NOT EXISTS enriched_genre_signatures (
                    release_key TEXT PRIMARY KEY,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    album_id TEXT,
                    signature_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_genre_review_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_tag_id INTEGER,
                    release_key TEXT NOT NULL,
                    raw_tag TEXT NOT NULL,
                    normalized_tag TEXT NOT NULL,
                    original_classification TEXT NOT NULL,
                    reviewed_classification TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT 'human',
                    decided_at TEXT NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (source_tag_id) REFERENCES ai_genre_source_tags(source_tag_id),
                    UNIQUE (source_tag_id, reviewer)
                );

                CREATE TABLE IF NOT EXISTS ai_tag_adjudication_cache (
                    normalized_tag TEXT PRIMARY KEY,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    classifier TEXT NOT NULL DEFAULT 'ai',
                    times_seen INTEGER NOT NULL DEFAULT 1,
                    decided_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_review_decisions_tag
                    ON ai_genre_review_decisions (source_tag_id);
                CREATE INDEX IF NOT EXISTS idx_review_decisions_release
                    ON ai_genre_review_decisions (release_key);

                CREATE INDEX IF NOT EXISTS idx_ai_genre_checks_release
                    ON ai_genre_release_checks (release_key, status);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_suggestions_type
                    ON ai_genre_suggestions (suggestion_type, genre);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_source_pages_release
                    ON ai_genre_source_pages (release_key);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_source_tags_page
                    ON ai_genre_source_tags (source_page_id);
                CREATE INDEX IF NOT EXISTS idx_enriched_genres_release
                    ON enriched_genres (release_key);
                CREATE INDEX IF NOT EXISTS idx_enriched_genres_genre
                    ON enriched_genres (genre);
                """
            )
            _ensure_column(conn, "ai_genre_release_checks", "web_mode", "TEXT NOT NULL DEFAULT 'off'")
            _ensure_column(conn, "ai_genre_release_checks", "source_evidence_hash", "TEXT NOT NULL DEFAULT 'none'")
            _ensure_column(
                conn,
                "ai_genre_release_checks",
                "response_schema_version",
                "TEXT NOT NULL DEFAULT 'ai-genre-response-v1'",
            )
            _ensure_column(conn, "ai_genre_suggestions", "recommendation_basis", "TEXT")
            _ensure_column(conn, "ai_genre_suggestions", "supporting_source_indexes_json", "TEXT")
            _ensure_column(conn, "ai_genre_suggestions", "descriptor_or_genre", "TEXT")
            _ensure_column(conn, "ai_genre_run_log", "cache_hits", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "ai_genre_run_log", "skipped_well_tagged", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "ai_genre_run_log", "no_web_checks", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "ai_genre_run_log", "authoritative_source_checks", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "ai_genre_run_log", "needs_review", "INTEGER NOT NULL DEFAULT 0")
            _migrate_release_checks_cache_identity(conn)

    def has_complete_check(
        self,
        release_key: str,
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str = "off",
        source_evidence_hash: str = "none",
        response_schema_version: str = RESPONSE_SCHEMA_VERSION,
    ) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM ai_genre_release_checks
                WHERE release_key = ?
                  AND input_hash = ?
                  AND prompt_version = ?
                  AND taxonomy_version = ?
                  AND model = ?
                  AND web_mode = ?
                  AND source_evidence_hash = ?
                  AND response_schema_version = ?
                  AND status IN ('complete', 'needs_review')
                LIMIT 1
                """,
                (
                    release_key,
                    input_hash,
                    prompt_version,
                    taxonomy_version,
                    model,
                    web_mode,
                    source_evidence_hash,
                    response_schema_version,
                ),
            ).fetchone()
        return row is not None

    def record_pending_check(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        identifiers: dict[str, Any],
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str = "off",
        source_evidence_hash: str = "none",
        response_schema_version: str = RESPONSE_SCHEMA_VERSION,
    ) -> int:
        return self._upsert_check(
            release_key=release_key,
            normalized_artist=normalized_artist,
            normalized_album=normalized_album,
            album_id=album_id,
            identifiers=identifiers,
            input_hash=input_hash,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
            model=model,
            web_mode=web_mode,
            source_evidence_hash=source_evidence_hash,
            response_schema_version=response_schema_version,
            status="pending",
            response_json=None,
            error_message=None,
            overall_confidence=None,
            evidence_quality=None,
            auto_apply_eligible=False,
            token_usage={},
            estimated_cost_usd=None,
        )

    def record_complete_check(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        identifiers: dict[str, Any],
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str = "off",
        source_evidence_hash: str = "none",
        response_schema_version: str = RESPONSE_SCHEMA_VERSION,
        response_json: dict[str, Any],
        overall_confidence: float | None,
        evidence_quality: str | None,
        auto_apply_eligible: bool,
        token_usage: dict[str, int] | None = None,
        estimated_cost_usd: float | None = None,
    ) -> int:
        status = "needs_review" if response_json.get("should_escalate") else "complete"
        check_id = self._upsert_check(
            release_key=release_key,
            normalized_artist=normalized_artist,
            normalized_album=normalized_album,
            album_id=album_id,
            identifiers=identifiers,
            input_hash=input_hash,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
            model=model,
            web_mode=web_mode,
            source_evidence_hash=source_evidence_hash,
            response_schema_version=response_schema_version,
            status=status,
            response_json=response_json,
            error_message=None,
            overall_confidence=overall_confidence,
            evidence_quality=evidence_quality,
            auto_apply_eligible=auto_apply_eligible,
            token_usage=token_usage or {},
            estimated_cost_usd=estimated_cost_usd,
        )
        self._replace_suggestions(check_id, response_json)
        return check_id

    def record_failed_check(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        identifiers: dict[str, Any],
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str = "off",
        source_evidence_hash: str = "none",
        response_schema_version: str = RESPONSE_SCHEMA_VERSION,
        error_message: str,
    ) -> int:
        return self._upsert_check(
            release_key=release_key,
            normalized_artist=normalized_artist,
            normalized_album=normalized_album,
            album_id=album_id,
            identifiers=identifiers,
            input_hash=input_hash,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
            model=model,
            web_mode=web_mode,
            source_evidence_hash=source_evidence_hash,
            response_schema_version=response_schema_version,
            status="failed",
            response_json=None,
            error_message=error_message,
            overall_confidence=None,
            evidence_quality=None,
            auto_apply_eligible=False,
            token_usage={},
            estimated_cost_usd=None,
        )

    def record_skipped_check(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        identifiers: dict[str, Any],
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str = "off",
        source_evidence_hash: str = "none",
        response_schema_version: str = RESPONSE_SCHEMA_VERSION,
        reason: str,
    ) -> int:
        return self._upsert_check(
            release_key=release_key,
            normalized_artist=normalized_artist,
            normalized_album=normalized_album,
            album_id=album_id,
            identifiers=identifiers,
            input_hash=input_hash,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
            model=model,
            web_mode=web_mode,
            source_evidence_hash=source_evidence_hash,
            response_schema_version=response_schema_version,
            status="skipped",
            response_json={"reason": reason},
            error_message=None,
            overall_confidence=None,
            evidence_quality=None,
            auto_apply_eligible=False,
            token_usage={},
            estimated_cost_usd=None,
        )

    def report(self, low_confidence_limit: int = 5) -> dict[str, Any]:
        with self.connect() as conn:
            status_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM ai_genre_release_checks GROUP BY status"
                )
            }
            token_row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM ai_genre_release_checks
                """
            ).fetchone()
            additions = Counter(
                {
                    row["genre"]: row["count"]
                    for row in conn.execute(
                        """
                        SELECT genre, COUNT(*) AS count
                        FROM ai_genre_suggestions
                        WHERE suggestion_type = 'add' AND genre IS NOT NULL
                        GROUP BY genre
                        ORDER BY count DESC, genre
                        LIMIT 20
                        """
                    )
                }
            )
            prunes = Counter(
                {
                    row["genre"]: row["count"]
                    for row in conn.execute(
                        """
                        SELECT genre, COUNT(*) AS count
                        FROM ai_genre_suggestions
                        WHERE suggestion_type = 'prune' AND genre IS NOT NULL
                        GROUP BY genre
                        ORDER BY count DESC, genre
                        LIMIT 20
                        """
                    )
                }
            )
            low_confidence = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT release_key, overall_confidence, evidence_quality
                    FROM ai_genre_release_checks
                    WHERE overall_confidence IS NOT NULL
                      AND overall_confidence < 0.5
                    ORDER BY overall_confidence ASC, checked_at DESC
                    LIMIT ?
                    """,
                    (low_confidence_limit,),
                )
            ]
            classification_counts = {
                row["classification"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT classification, COUNT(*) AS count
                    FROM ai_genre_tag_classifications
                    GROUP BY classification
                    ORDER BY classification
                    """
                )
            }
            review_only_tag_gaps = [
                {
                    "tag": row["normalized_tag"],
                    "count": row["count"],
                    "source_types": sorted(row["source_types"].split("|")) if row["source_types"] else [],
                    "example_releases": row["example_releases"].split("|") if row["example_releases"] else [],
                }
                for row in conn.execute(
                    """
                    WITH review_only AS (
                        SELECT
                            t.normalized_tag,
                            p.release_key,
                            p.source_type
                        FROM ai_genre_tag_classifications c
                        JOIN ai_genre_source_tags t ON t.source_tag_id = c.source_tag_id
                        JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                        WHERE c.classification = 'review_only'
                    ),
                    ranked_examples AS (
                        SELECT
                            normalized_tag,
                            release_key,
                            ROW_NUMBER() OVER (
                                PARTITION BY normalized_tag
                                ORDER BY release_key
                            ) AS release_rank
                        FROM (
                            SELECT DISTINCT normalized_tag, release_key
                            FROM review_only
                        )
                    )
                    SELECT
                        r.normalized_tag,
                        COUNT(*) AS count,
                        GROUP_CONCAT(DISTINCT r.source_type) AS source_types,
                        (
                            SELECT GROUP_CONCAT(re.release_key, '|')
                            FROM ranked_examples re
                            WHERE re.normalized_tag = r.normalized_tag
                              AND re.release_rank <= 3
                            ORDER BY re.release_rank
                        ) AS example_releases
                    FROM review_only r
                    GROUP BY r.normalized_tag
                    ORDER BY count DESC, r.normalized_tag
                    LIMIT 50
                    """
                )
            ]
            run_totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS runs,
                    COALESCE(SUM(releases_seen), 0) AS releases_seen,
                    COALESCE(SUM(releases_called), 0) AS releases_called,
                    COALESCE(SUM(releases_skipped), 0) AS releases_skipped,
                    COALESCE(SUM(releases_failed), 0) AS releases_failed,
                    COALESCE(SUM(cache_hits), 0) AS cache_hits,
                    COALESCE(SUM(skipped_well_tagged), 0) AS skipped_well_tagged,
                    COALESCE(SUM(no_web_checks), 0) AS no_web_checks,
                    COALESCE(SUM(authoritative_source_checks), 0) AS authoritative_source_checks,
                    COALESCE(SUM(needs_review), 0) AS needs_review
                FROM ai_genre_run_log
                """
            ).fetchone()
            source_domains = Counter()
            for row in conn.execute(
                "SELECT response_json FROM ai_genre_release_checks WHERE response_json IS NOT NULL"
            ):
                try:
                    payload = json.loads(row["response_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                for source in payload.get("source_evidence", []):
                    domain = _domain_from_url(source.get("source_url"))
                    if domain:
                        source_domains[domain] += 1
        return {
            "status_counts": status_counts,
            "token_usage": dict(token_row),
            "run_totals": dict(run_totals),
            "top_additions": dict(additions),
            "top_prunes": dict(prunes),
            "low_confidence_examples": low_confidence,
            "classification_counts": classification_counts,
            "review_only_tag_gaps": review_only_tag_gaps,
            "source_domains_used": dict(source_domains.most_common(20)),
        }

    def record_run_log(
        self,
        *,
        command: str,
        status: str,
        releases_seen: int,
        releases_called: int,
        releases_skipped: int,
        releases_failed: int,
        cache_hits: int = 0,
        skipped_well_tagged: int = 0,
        no_web_checks: int = 0,
        authoritative_source_checks: int = 0,
        needs_review: int = 0,
        token_usage: dict[str, int],
        estimated_cost_usd: float = 0.0,
    ) -> None:
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_run_log (
                    started_at, completed_at, command, status, releases_seen,
                    releases_called, releases_skipped, releases_failed,
                    cache_hits, skipped_well_tagged, no_web_checks,
                    authoritative_source_checks, needs_review,
                    input_tokens, output_tokens, total_tokens, estimated_cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    now,
                    command,
                    status,
                    releases_seen,
                    releases_called,
                    releases_skipped,
                    releases_failed,
                    cache_hits,
                    skipped_well_tagged,
                    no_web_checks,
                    authoritative_source_checks,
                    needs_review,
                    token_usage.get("input_tokens", 0),
                    token_usage.get("output_tokens", 0),
                    token_usage.get("total_tokens", 0),
                    estimated_cost_usd,
                ),
            )

    def upsert_source_page(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        source_url: str,
        source_type: str,
        identity_status: str,
        identity_confidence: float,
        evidence_summary: str,
    ) -> int:
        """Insert or update one confirmed source page and return source_page_id."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_source_pages (
                    release_key, normalized_artist, normalized_album, album_id,
                    source_url, source_domain, source_type, identity_status,
                    identity_confidence, fetched_at, extraction_status,
                    extraction_hash, evidence_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'none', ?)
                ON CONFLICT(release_key, source_url)
                DO UPDATE SET
                    normalized_artist = excluded.normalized_artist,
                    normalized_album = excluded.normalized_album,
                    album_id = excluded.album_id,
                    source_domain = excluded.source_domain,
                    source_type = excluded.source_type,
                    identity_status = excluded.identity_status,
                    identity_confidence = excluded.identity_confidence,
                    fetched_at = excluded.fetched_at,
                    evidence_summary = excluded.evidence_summary
                """,
                (
                    release_key,
                    normalized_artist,
                    normalized_album,
                    album_id,
                    source_url,
                    _domain_from_url(source_url) or "",
                    source_type,
                    identity_status,
                    identity_confidence,
                    _now_iso(),
                    evidence_summary,
                ),
            )
            row = conn.execute(
                """
                SELECT source_page_id
                FROM ai_genre_source_pages
                WHERE release_key = ? AND source_url = ?
                """,
                (release_key, source_url),
            ).fetchone()
            return int(row["source_page_id"])

    def replace_source_tags(self, source_page_id: int, raw_tags: list[str]) -> None:
        """Replace extracted source tags for one source page."""
        from .tag_classification import normalize_source_tag

        seen: set[str] = set()
        rows: list[tuple[int, str, str, int, str]] = []
        for position, raw_tag in enumerate(raw_tags):
            normalized = normalize_source_tag(raw_tag)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows.append((source_page_id, raw_tag, normalized, position, _now_iso()))

        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM ai_genre_tag_classifications
                WHERE source_tag_id IN (
                    SELECT source_tag_id
                    FROM ai_genre_source_tags
                    WHERE source_page_id = ?
                )
                """,
                (source_page_id,),
            )
            conn.execute("DELETE FROM ai_genre_source_tags WHERE source_page_id = ?", (source_page_id,))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO ai_genre_source_tags (
                        source_page_id, raw_tag, normalized_tag, tag_position, extracted_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.execute(
                """
                UPDATE ai_genre_source_pages
                SET extraction_status = ?, extraction_hash = ?, fetched_at = ?
                WHERE source_page_id = ?
                """,
                ("complete", _hash_tags([row[2] for row in rows]), _now_iso(), source_page_id),
            )

    def mark_source_page_extraction_failed(self, source_page_id: int, error_message: str) -> None:
        """Record a failed deterministic extraction and clear stale tags for one source page."""
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM ai_genre_tag_classifications
                WHERE source_tag_id IN (
                    SELECT source_tag_id
                    FROM ai_genre_source_tags
                    WHERE source_page_id = ?
                )
                """,
                (source_page_id,),
            )
            conn.execute("DELETE FROM ai_genre_source_tags WHERE source_page_id = ?", (source_page_id,))
            conn.execute(
                """
                UPDATE ai_genre_source_pages
                SET extraction_status = ?,
                    extraction_hash = ?,
                    fetched_at = ?,
                    evidence_summary = ?
                WHERE source_page_id = ?
                """,
                (
                    "failed",
                    "none",
                    _now_iso(),
                    f"User-supplied source URL. Extraction failed: {error_message}",
                    source_page_id,
                ),
            )

    def classify_source_tags(
        self,
        source_page_id: int,
        *,
        adjudicate: bool = False,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Run deterministic source-tag classification for one source page.

        Unknown tags (review_only) are checked against the adjudication cache first.
        If adjudicate=True, any remaining unknowns are sent to the AI in a single batch.
        """
        from .tag_classification import classify_source_tag

        with self.connect() as conn:
            tags = list(
                conn.execute(
                    """
                    SELECT source_tag_id, raw_tag
                    FROM ai_genre_source_tags
                    WHERE source_page_id = ?
                    ORDER BY tag_position, source_tag_id
                    """,
                    (source_page_id,),
                )
            )
            conn.execute(
                """
                DELETE FROM ai_genre_tag_classifications
                WHERE source_tag_id IN (
                    SELECT source_tag_id
                    FROM ai_genre_source_tags
                    WHERE source_page_id = ?
                )
                """,
                (source_page_id,),
            )
            rows: list[tuple] = []
            seen_normalized_tags: set[str] = set()
            deleted_source_tag_ids: set[int] = set()
            # (source_tag_id, raw_tag, normalized_tag) for tags that need AI adjudication
            review_only_batch: list[tuple[int, str, str]] = []
            # Cache entries to increment after this connection closes (to avoid nested lock)
            # Each entry: (normalized_tag, classification, confidence, classifier)
            cache_increments: list[tuple[str, str, float, str]] = []

            for tag in tags:
                source_tag_id = int(tag["source_tag_id"])
                if source_tag_id in deleted_source_tag_ids:
                    continue
                classification = classify_source_tag(tag["raw_tag"])
                if classification.normalized_tag in seen_normalized_tags:
                    conn.execute("DELETE FROM ai_genre_source_tags WHERE source_tag_id = ?", (source_tag_id,))
                    continue
                duplicate_rows = list(
                    conn.execute(
                        """
                        SELECT source_tag_id
                        FROM ai_genre_source_tags
                        WHERE source_page_id = ?
                          AND normalized_tag = ?
                          AND source_tag_id != ?
                        """,
                        (source_page_id, classification.normalized_tag, source_tag_id),
                    )
                )
                for duplicate in duplicate_rows:
                    duplicate_id = int(duplicate["source_tag_id"])
                    deleted_source_tag_ids.add(duplicate_id)
                    conn.execute("DELETE FROM ai_genre_source_tags WHERE source_tag_id = ?", (duplicate_id,))
                conn.execute(
                    """
                    UPDATE ai_genre_source_tags
                    SET normalized_tag = ?
                    WHERE source_tag_id = ?
                    """,
                    (classification.normalized_tag, source_tag_id),
                )
                seen_normalized_tags.add(classification.normalized_tag)

                if classification.classification == "review_only":
                    # lookup_cached_adjudication opens a separate read-only connection;
                    # safe because it only reads ai_tag_adjudication_cache, not the tables
                    # being written by the outer connection.
                    cached = self.lookup_cached_adjudication(classification.normalized_tag)
                    if cached is not None:
                        rows.append((
                            source_tag_id,
                            cached["classification"],
                            cached["confidence"],
                            "cached_ai",
                            f"Cached AI adjudication (seen {cached['times_seen']}x).",
                            _now_iso(),
                        ))
                        # Queue cache increment — will write after this connection closes
                        cache_increments.append((
                            classification.normalized_tag,
                            cached["classification"],
                            cached["confidence"],
                            cached["classifier"],
                        ))
                        continue
                    review_only_batch.append((source_tag_id, tag["raw_tag"], classification.normalized_tag))
                    continue

                rows.append(
                    (
                        source_tag_id,
                        classification.classification,
                        classification.confidence,
                        "deterministic",
                        classification.reason,
                        _now_iso(),
                    )
                )

            if rows:
                conn.executemany(
                    """
                    INSERT INTO ai_genre_tag_classifications (
                        source_tag_id, classification, confidence, classifier, reason, classified_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        # --- outside the connection block to avoid nested lock ---

        # Flush cache increments for cache-hit tags
        for norm_tag, cls, conf, classifier in cache_increments:
            self.cache_adjudication(
                normalized_tag=norm_tag,
                classification=cls,
                confidence=conf,
                classifier=classifier,
            )

        if not review_only_batch:
            return

        ai_rows: list[tuple] = []

        if adjudicate:
            from .tag_adjudicator import adjudicate_tags

            ai_input = [(raw, norm) for _, raw, norm in review_only_batch]
            ai_results = adjudicate_tags(ai_input, model=model)

            ai_cache_writes: list[tuple[str, str, float]] = []
            for source_tag_id, raw_tag, normalized_tag in review_only_batch:
                ai_result = ai_results.get(normalized_tag)
                if ai_result is not None:
                    # Only cache definitive classifications; review_only means AI was also
                    # uncertain, so re-run adjudication on future encounters.
                    if ai_result["classification"] != "review_only":
                        ai_cache_writes.append((
                            normalized_tag,
                            ai_result["classification"],
                            ai_result["confidence"],
                        ))
                    ai_rows.append((
                        source_tag_id,
                        ai_result["classification"],
                        ai_result["confidence"],
                        "ai",
                        ai_result.get("reason", "AI adjudication."),
                        _now_iso(),
                    ))
                else:
                    ai_rows.append((
                        source_tag_id,
                        "review_only",
                        0.5,
                        "deterministic",
                        "Unknown source tag requires adjudication before use.",
                        _now_iso(),
                    ))

            # Cache AI results before writing classification rows
            for norm_tag, cls, conf in ai_cache_writes:
                self.cache_adjudication(
                    normalized_tag=norm_tag,
                    classification=cls,
                    confidence=conf,
                    classifier="ai",
                )
        else:
            ai_rows = [
                (
                    source_tag_id,
                    "review_only",
                    0.5,
                    "deterministic",
                    "Unknown source tag requires adjudication before use.",
                    _now_iso(),
                )
                for source_tag_id, _, _ in review_only_batch
            ]

        if ai_rows:
            with self.connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO ai_genre_tag_classifications (
                        source_tag_id, classification, confidence, classifier, reason, classified_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ai_rows,
                )

    def rebuild_enriched_genres_for_release(self, release_key: str) -> None:
        """Rebuild accepted enriched_genres and release signature for one release."""
        with self.connect() as conn:
            source_rows = list(
                conn.execute(
                    """
                    SELECT
                        p.release_key,
                        p.normalized_artist,
                        p.normalized_album,
                        p.album_id,
                        p.source_url,
                        p.source_type,
                        p.identity_status,
                        t.source_tag_id,
                        t.source_page_id,
                        t.normalized_tag,
                        c.classification,
                        c.confidence,
                        d.reviewed_classification
                    FROM ai_genre_tag_classifications c
                    JOIN ai_genre_source_tags t ON t.source_tag_id = c.source_tag_id
                    JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                    LEFT JOIN ai_genre_review_decisions d ON d.source_tag_id = t.source_tag_id AND d.reviewer = 'human'
                    WHERE p.release_key = ?
                      AND (
                          (c.classification = 'genre_style' AND d.reviewed_classification IS NULL)
                          OR d.reviewed_classification = 'genre_style'
                      )
                      AND p.source_type IN (
                          'official_release',
                          'official_artist',
                          'official_label',
                          'bandcamp_release',
                          'label_catalog',
                          'press_release',
                          'liner_notes',
                          'official_distributor',
                          'local_metadata',
                          'lastfm_tags'
                      )
                      AND p.identity_status IN ('confirmed', 'probable')
                    ORDER BY t.normalized_tag, t.source_tag_id
                    """,
                    (release_key,),
                )
            )
            metadata_row = conn.execute(
                """
                SELECT release_key, normalized_artist, normalized_album, album_id
                FROM ai_genre_source_pages
                WHERE release_key = ?
                ORDER BY identity_confidence DESC, source_page_id
                LIMIT 1
                """,
                (release_key,),
            ).fetchone()
            conn.execute("DELETE FROM enriched_genres WHERE release_key = ?", (release_key,))
            conn.execute("DELETE FROM enriched_genre_signatures WHERE release_key = ?", (release_key,))

            now = _now_iso()
            rows = [
                (
                    row["release_key"],
                    row["normalized_artist"],
                    row["normalized_album"],
                    row["album_id"],
                    row["normalized_tag"],
                    (
                        "lastfm_tags" if row["source_type"] == "lastfm_tags"
                        else "local_metadata" if row["source_type"] == "local_metadata"
                        else "authoritative_source"
                    ),
                    # Use 0.90 for human-reviewed genre_style (promoted from review_only or other)
                    # Keep original confidence for auto-classified genre_style
                    0.90 if row["reviewed_classification"] == "genre_style" and row["classification"] != "genre_style" else row["confidence"],
                    row["source_tag_id"],
                    row["source_page_id"],
                    f"source_tag:{row['source_tag_id']}",
                    "accepted",
                    now,
                )
                for row in source_rows
            ]
            if rows:
                conn.executemany(
                    """
                    INSERT INTO enriched_genres (
                        release_key, normalized_artist, normalized_album, album_id,
                        genre, basis, confidence, source_tag_id, source_page_id,
                        source_ref, status, added_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            if metadata_row and rows:
                signature = {
                    "genres": sorted({row["normalized_tag"] for row in source_rows}),
                    "sources": _signature_sources(source_rows),
                }
                conn.execute(
                    """
                    INSERT INTO enriched_genre_signatures (
                        release_key, normalized_artist, normalized_album, album_id,
                        signature_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata_row["release_key"],
                        metadata_row["normalized_artist"],
                        metadata_row["normalized_album"],
                        metadata_row["album_id"],
                        json.dumps(signature, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )

    def get_review_queue(
        self,
        *,
        release_key: str | None = None,
        classification: str | None = None,
        source_type: str | None = None,
        max_confidence: float = 0.80,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return tags needing human review, ordered by confidence ascending."""
        with self.connect() as conn:
            clauses = [
                "d.decision_id IS NULL",
                "(c.classification = 'review_only' OR c.confidence < ?)",
            ]
            params: list[Any] = [max_confidence]
            if release_key:
                clauses.append("p.release_key = ?")
                params.append(release_key)
            if classification:
                clauses.append("c.classification = ?")
                params.append(classification)
            if source_type:
                clauses.append("p.source_type = ?")
                params.append(source_type)
            where = " AND ".join(clauses)
            limit_clause = f" LIMIT {int(limit)}" if limit else ""
            rows = list(conn.execute(
                f"""
                SELECT t.source_tag_id, p.release_key, p.normalized_artist, p.normalized_album,
                       t.raw_tag, t.normalized_tag, c.classification, c.confidence,
                       p.source_url, p.source_type
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
                LEFT JOIN ai_genre_review_decisions d ON d.source_tag_id = t.source_tag_id
                WHERE {where}
                ORDER BY c.confidence ASC, p.release_key, t.tag_position
                {limit_clause}
                """,
                params,
            ))
            return [dict(row) for row in rows]

    def record_review_decision(
        self,
        *,
        source_tag_id: int,
        release_key: str,
        raw_tag: str,
        normalized_tag: str,
        original_classification: str,
        reviewed_classification: str,
        reviewer: str = "human",
        notes: str | None = None,
    ) -> int:
        """Record a human review decision for one source tag."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_review_decisions (
                    source_tag_id, release_key, raw_tag, normalized_tag,
                    original_classification, reviewed_classification,
                    reviewer, decided_at, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_tag_id, reviewer)
                DO UPDATE SET
                    reviewed_classification = excluded.reviewed_classification,
                    decided_at = excluded.decided_at,
                    notes = excluded.notes
                """,
                (
                    source_tag_id,
                    release_key,
                    raw_tag,
                    normalized_tag,
                    original_classification,
                    reviewed_classification,
                    reviewer,
                    _now_iso(),
                    notes,
                ),
            )
            row = conn.execute(
                "SELECT decision_id FROM ai_genre_review_decisions WHERE source_tag_id = ? AND reviewer = ?",
                (source_tag_id, reviewer),
            ).fetchone()
            return int(row["decision_id"])

    def undo_review_decision(self, source_tag_id: int, reviewer: str = "human") -> bool:
        """Remove a review decision, returning the tag to the review queue."""
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM ai_genre_review_decisions WHERE source_tag_id = ? AND reviewer = ?",
                (source_tag_id, reviewer),
            )
            return cursor.rowcount > 0

    def get_review_context(self, release_key: str) -> list[dict[str, Any]]:
        """Return all classified tags for a release (for showing context during review)."""
        with self.connect() as conn:
            rows = list(conn.execute(
                """
                SELECT t.normalized_tag, c.classification, MAX(c.confidence) AS confidence
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
                WHERE p.release_key = ?
                GROUP BY t.normalized_tag, c.classification
                ORDER BY c.classification, t.normalized_tag
                """,
                (release_key,),
            ))
            return [dict(row) for row in rows]

    def get_graduated_terms(self) -> dict[str, set[str]]:
        """Return all human-reviewed terms grouped by their reviewed classification."""
        with self.connect() as conn:
            rows = list(conn.execute(
                """
                SELECT DISTINCT normalized_tag, reviewed_classification
                FROM ai_genre_review_decisions
                WHERE reviewed_classification != 'rejected'
                  AND reviewer = 'human'
                """
            ))
            result: dict[str, set[str]] = {}
            for row in rows:
                classification = row["reviewed_classification"]
                result.setdefault(classification, set()).add(row["normalized_tag"])
            return result

    def lookup_cached_adjudication(self, normalized_tag: str) -> dict[str, Any] | None:
        """Look up a cached AI adjudication result by normalized tag."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT classification, confidence, classifier, times_seen FROM ai_tag_adjudication_cache WHERE normalized_tag = ?",
                (normalized_tag,),
            ).fetchone()
            return dict(row) if row else None

    def cache_adjudication(
        self,
        *,
        normalized_tag: str,
        classification: str,
        confidence: float,
        classifier: str = "ai",
    ) -> None:
        """Cache an AI adjudication result, incrementing times_seen on conflict."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_tag_adjudication_cache (normalized_tag, classification, confidence, classifier, times_seen, decided_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(normalized_tag)
                DO UPDATE SET
                    times_seen = ai_tag_adjudication_cache.times_seen + 1,
                    decided_at = excluded.decided_at
                """,
                (normalized_tag, classification, confidence, classifier, _now_iso()),
            )

    def get_ai_graduated_terms(self, min_times_seen: int = 3) -> dict[str, set[str]]:
        """Return AI-adjudicated terms grouped by classification, filtered by frequency."""
        with self.connect() as conn:
            rows = list(conn.execute(
                """
                SELECT normalized_tag, classification
                FROM ai_tag_adjudication_cache
                WHERE times_seen >= ?
                  AND classification NOT IN ('review_only', 'rejected')
                """,
                (min_times_seen,),
            ))
            result: dict[str, set[str]] = {}
            for row in rows:
                result.setdefault(row["classification"], set()).add(row["normalized_tag"])
            return result

    def _upsert_check(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        identifiers: dict[str, Any],
        input_hash: str,
        prompt_version: str,
        taxonomy_version: str,
        model: str,
        web_mode: str,
        source_evidence_hash: str,
        response_schema_version: str,
        status: str,
        response_json: dict[str, Any] | None,
        error_message: str | None,
        overall_confidence: float | None,
        evidence_quality: str | None,
        auto_apply_eligible: bool,
        token_usage: dict[str, int],
        estimated_cost_usd: float | None,
    ) -> int:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid check status: {status}")
        mbid = identifiers.get("musicbrainz_release_mbid") or identifiers.get("mbid")
        discogs_id = identifiers.get("discogs_release_id") or identifiers.get("discogs_id")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_release_checks (
                    release_key, normalized_artist, normalized_album, album_id,
                    mbid, discogs_id, identifiers_json, input_hash, prompt_version,
                    taxonomy_version, model, web_mode, source_evidence_hash,
                    response_schema_version, status, checked_at, response_json,
                    error_message, overall_confidence, evidence_quality,
                    auto_apply_eligible, input_tokens, output_tokens, total_tokens,
                    estimated_cost_usd
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    release_key, input_hash, prompt_version, taxonomy_version,
                    model, web_mode, source_evidence_hash, response_schema_version
                )
                DO UPDATE SET
                    normalized_artist = excluded.normalized_artist,
                    normalized_album = excluded.normalized_album,
                    album_id = excluded.album_id,
                    mbid = excluded.mbid,
                    discogs_id = excluded.discogs_id,
                    identifiers_json = excluded.identifiers_json,
                    web_mode = excluded.web_mode,
                    source_evidence_hash = excluded.source_evidence_hash,
                    response_schema_version = excluded.response_schema_version,
                    status = excluded.status,
                    checked_at = excluded.checked_at,
                    response_json = excluded.response_json,
                    error_message = excluded.error_message,
                    overall_confidence = excluded.overall_confidence,
                    evidence_quality = excluded.evidence_quality,
                    auto_apply_eligible = excluded.auto_apply_eligible,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    total_tokens = excluded.total_tokens,
                    estimated_cost_usd = excluded.estimated_cost_usd
                """,
                (
                    release_key,
                    normalized_artist,
                    normalized_album,
                    album_id,
                    mbid,
                    discogs_id,
                    json.dumps(identifiers, sort_keys=True),
                    input_hash,
                    prompt_version,
                    taxonomy_version,
                    model,
                    web_mode,
                    source_evidence_hash,
                    response_schema_version,
                    status,
                    _now_iso(),
                    json.dumps(response_json, sort_keys=True) if response_json is not None else None,
                    error_message,
                    overall_confidence,
                    evidence_quality,
                    1 if auto_apply_eligible else 0,
                    token_usage.get("input_tokens"),
                    token_usage.get("output_tokens"),
                    token_usage.get("total_tokens"),
                    estimated_cost_usd,
                ),
            )
            row = conn.execute(
                """
                SELECT check_id
                FROM ai_genre_release_checks
                WHERE release_key = ?
                  AND input_hash = ?
                  AND prompt_version = ?
                  AND taxonomy_version = ?
                  AND model = ?
                  AND web_mode = ?
                  AND source_evidence_hash = ?
                  AND response_schema_version = ?
                """,
                (
                    release_key,
                    input_hash,
                    prompt_version,
                    taxonomy_version,
                    model,
                    web_mode,
                    source_evidence_hash,
                    response_schema_version,
                ),
            ).fetchone()
            return int(row["check_id"])

    def _replace_suggestions(self, check_id: int, response_json: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM ai_genre_suggestions WHERE check_id = ?", (check_id,))
            rows = []
            for item in response_json.get("existing_genres_to_keep", []):
                rows.append(_suggestion_row(check_id, "keep", item, genre_key="genre"))
            for item in response_json.get("existing_genres_to_prune", []):
                rows.append(_suggestion_row(check_id, "prune", item, genre_key="genre"))
            for item in response_json.get("new_genres_to_add", []):
                auto_apply = _is_conservative_auto_apply_candidate(
                    item,
                    overall_confidence=response_json.get("release_level_confidence"),
                    evidence_quality=response_json.get("evidence_quality"),
                    should_escalate=bool(response_json.get("should_escalate")),
                    source_evidence=response_json.get("source_evidence", []),
                )
                rows.append(_suggestion_row(check_id, "add", item, genre_key="genre", auto_apply=auto_apply))
            for item in response_json.get("descriptor_tags", []):
                rows.append(_suggestion_row(check_id, "descriptor", item, genre_key="tag"))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO ai_genre_suggestions (
                        check_id, suggestion_type, genre, descriptor_tag, confidence,
                        reason, prune_type, recommendation_basis,
                        supporting_source_indexes_json, descriptor_or_genre, auto_apply_eligible
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )


def _is_conservative_auto_apply_candidate(
    item: dict[str, Any],
    *,
    overall_confidence: float | None,
    evidence_quality: str | None,
    should_escalate: bool,
    source_evidence: list[dict[str, Any]],
) -> bool:
    if should_escalate or evidence_quality != "high":
        return False
    if overall_confidence is None or float(overall_confidence) < AUTO_APPLY_MIN_CONFIDENCE:
        return False
    if not item.get("auto_apply_eligible"):
        return False
    if float(item.get("confidence") or 0.0) < AUTO_APPLY_MIN_CONFIDENCE:
        return False
    if item.get("recommendation_basis") not in {"authoritative_source", "hybrid"}:
        return False
    genre = str(item.get("genre") or "").casefold()
    if genre in BROAD_AUTO_APPLY_BLOCKLIST or genre in NON_GENRE_AUTO_APPLY_BLOCKLIST:
        return False
    indexes = item.get("supporting_source_indexes") or []
    if not any(
        isinstance(index, int)
        and 0 <= index < len(source_evidence)
        and source_evidence[index].get("source_type") in AUTHORITATIVE_SOURCE_TYPES
        and source_evidence[index].get("reliability") in {"high", "medium"}
        and source_evidence[index].get("release_specific") is True
        for index in indexes
    ):
        return False
    reason = str(item.get("reason") or "").casefold()
    return not any(hint in reason for hint in REVIEW_ONLY_REASON_HINTS)


def _hash_tags(normalized_tags: list[str]) -> str:
    blob = json.dumps(normalized_tags, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _signature_sources(source_rows: list[sqlite3.Row]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for row in source_rows:
        key = (row["source_type"], row["source_url"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_type": row["source_type"], "source_url": row["source_url"]})
    return sorted(sources, key=lambda item: (item["source_type"], item["source_url"]))


def _suggestion_row(
    check_id: int,
    suggestion_type: str,
    item: dict[str, Any],
    *,
    genre_key: str,
    auto_apply: bool = False,
) -> tuple[Any, ...]:
    is_descriptor = suggestion_type == "descriptor"
    return (
        check_id,
        suggestion_type,
        None if is_descriptor else item.get(genre_key),
        item.get(genre_key) if is_descriptor else None,
        item.get("confidence"),
        item.get("reason"),
        item.get("prune_type"),
        item.get("recommendation_basis"),
        json.dumps(item.get("supporting_source_indexes", []), sort_keys=True),
        item.get("descriptor_or_genre"),
        1 if auto_apply else 0,
    )


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(str(url))
    return parsed.netloc.lower().removeprefix("www.") or None
