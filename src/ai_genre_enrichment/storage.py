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
from .policy import (
    LEGACY_POLICY_VERSION,
    STABILIZED_POLICY_VERSION,
    can_seed_signature,
    canonical_source_type,
    evidence_basis,
)

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
# Strings that are never valid genres: always pruned, never added or stored.
ALWAYS_PRUNE_GENRES: frozenset[str] = frozenset({
    "electronicnic",
    "empty",
})
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
        # Wait up to 30s for a lock instead of failing instantly, so concurrent
        # collection passes (e.g. Last.fm + Bandcamp in separate terminals)
        # don't error on write contention.
        conn.execute("PRAGMA busy_timeout = 30000")
        # WAL so readers never block on a writer (persistent once set). The GUI
        # review panel reads this DB inline on the worker's reader thread WHILE
        # a scan/enrich writes it — under the default delete journal those reads
        # wedged behind commit locks (2026-06-12 review-queue timeout incident).
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def connect_readonly(self, *, busy_timeout_ms: int = 2000) -> sqlite3.Connection:
        """Read-only connection for inline/latency-sensitive readers.

        mode=ro + WAL means these never block on writers and can never wedge
        the caller's thread for the full 30s write busy_timeout. Raises
        sqlite3.OperationalError if the DB file does not exist.
        """
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
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

                CREATE TABLE IF NOT EXISTS ai_genre_source_attempts (
                    release_key TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    attempted_at TEXT NOT NULL,
                    PRIMARY KEY (release_key, source_type)
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
                    enrichment_policy_version TEXT,
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
                    enrichment_policy_version TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_genre_user_overrides (
                    release_key TEXT PRIMARY KEY,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    genres_add_json TEXT NOT NULL DEFAULT '[]',
                    genres_remove_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_genre_review_queue (
                    release_key       TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album  TEXT NOT NULL,
                    term              TEXT NOT NULL,
                    confidence        REAL,
                    basis             TEXT NOT NULL DEFAULT 'hybrid_fusion',
                    sources_json      TEXT NOT NULL DEFAULT '[]',
                    reason            TEXT NOT NULL DEFAULT '',
                    status            TEXT NOT NULL DEFAULT 'pending' CHECK (
                        status IN ('pending', 'accepted', 'rejected')
                    ),
                    scanned_at        TEXT NOT NULL,
                    decided_at        TEXT,
                    PRIMARY KEY (release_key, term)
                );

                CREATE INDEX IF NOT EXISTS idx_review_queue_status
                    ON ai_genre_review_queue (status, release_key);

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

                CREATE TABLE IF NOT EXISTS ai_genre_model_priors (
                    prior_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_key TEXT NOT NULL,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    album_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    enrichment_policy_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    warnings_json TEXT,
                    error_message TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (
                        release_key, provider, model, prompt_version, taxonomy_version,
                        schema_version, enrichment_policy_version, input_hash
                    )
                );

                CREATE TABLE IF NOT EXISTS ai_genre_model_prior_terms (
                    prior_term_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prior_id INTEGER NOT NULL,
                    release_key TEXT NOT NULL,
                    raw_term TEXT NOT NULL,
                    normalized_term TEXT NOT NULL,
                    canonical_slug TEXT,
                    confidence REAL NOT NULL,
                    specificity TEXT NOT NULL,
                    taxonomy_role TEXT NOT NULL,
                    mapping_status TEXT NOT NULL,
                    accepted_for_shadow INTEGER NOT NULL DEFAULT 0,
                    auto_apply_eligible INTEGER NOT NULL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (prior_id) REFERENCES ai_genre_model_priors(prior_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ai_genre_model_priors_release
                    ON ai_genre_model_priors (release_key);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_model_priors_provider_model
                    ON ai_genre_model_priors (provider, model);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_release
                    ON ai_genre_model_prior_terms (release_key);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_normalized
                    ON ai_genre_model_prior_terms (normalized_term);
                CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_mapping
                    ON ai_genre_model_prior_terms (mapping_status, accepted_for_shadow);

                CREATE TABLE IF NOT EXISTS genre_graph_canonical_genres (
                    genre_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    specificity_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS genre_graph_aliases (
                    alias TEXT PRIMARY KEY,
                    canonical_genre_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (canonical_genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id)
                );

                CREATE TABLE IF NOT EXISTS genre_graph_edges (
                    source_genre_id TEXT NOT NULL,
                    target_genre_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT,
                    PRIMARY KEY (source_genre_id, target_genre_id, edge_type),
                    FOREIGN KEY (source_genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id),
                    FOREIGN KEY (target_genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id)
                );

                CREATE TABLE IF NOT EXISTS genre_graph_canonical_facets (
                    facet_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    facet_type TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS genre_graph_bridge_rules (
                    source_genre_id TEXT NOT NULL,
                    target_genre_id TEXT NOT NULL,
                    required_family_min REAL NOT NULL,
                    required_facet_overlap REAL NOT NULL,
                    required_sonic_similarity REAL NOT NULL,
                    required_transition_quality REAL NOT NULL,
                    mode_allowed TEXT NOT NULL,
                    notes TEXT,
                    PRIMARY KEY (source_genre_id, target_genre_id),
                    FOREIGN KEY (source_genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id),
                    FOREIGN KEY (target_genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id)
                );

                CREATE TABLE IF NOT EXISTS genre_graph_rejected_terms (
                    term TEXT PRIMARY KEY,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS genre_graph_release_genre_assignments (
                    release_id TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    album TEXT NOT NULL,
                    genre_id TEXT NOT NULL,
                    assignment_layer TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_reliability REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    rejected_by_user INTEGER NOT NULL DEFAULT 0,
                    provenance_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (release_id, genre_id, assignment_layer),
                    FOREIGN KEY (genre_id)
                        REFERENCES genre_graph_canonical_genres(genre_id)
                );

                CREATE TABLE IF NOT EXISTS genre_graph_release_facet_assignments (
                    release_id TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    album TEXT NOT NULL,
                    facet_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (release_id, facet_id, source),
                    FOREIGN KEY (facet_id)
                        REFERENCES genre_graph_canonical_facets(facet_id)
                );

                CREATE TABLE IF NOT EXISTS genre_graph_release_materialization (
                    release_id TEXT PRIMARY KEY,
                    evidence_fingerprint TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
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
            _ensure_column(conn, "enriched_genres", "enrichment_policy_version", "TEXT")
            _ensure_column(conn, "enriched_genre_signatures", "enrichment_policy_version", "TEXT")
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
            signature_policy_counts = {
                row["policy_version"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT COALESCE(enrichment_policy_version, ?) AS policy_version,
                           COUNT(*) AS count
                    FROM enriched_genre_signatures
                    GROUP BY COALESCE(enrichment_policy_version, ?)
                    ORDER BY policy_version
                    """,
                    (LEGACY_POLICY_VERSION, LEGACY_POLICY_VERSION),
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
            "signature_policy_counts": signature_policy_counts,
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

    def release_keys_with_source_type(self, source_type: str) -> set[str]:
        """Return release_keys that already have a source page of the given type.

        Used by collection passes to skip releases already scraped from a given
        source (e.g. ``lastfm_tags``) so repeated runs don't duplicate effort.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT release_key FROM ai_genre_source_pages "
                "WHERE source_type = ?",
                (source_type,),
            )
            return {row[0] for row in rows}

    def all_collected_tags(self) -> list[sqlite3.Row]:
        """Every collected source tag joined to its release, for growth analysis.

        Returns rows with: release_key, normalized_artist, normalized_album,
        normalized_tag. One row per (page, tag); callers aggregate.
        """
        with self.connect() as conn:
            return list(conn.execute(
                """
                SELECT p.release_key, p.normalized_artist, p.normalized_album,
                       t.normalized_tag
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                WHERE t.normalized_tag IS NOT NULL AND t.normalized_tag != ''
                """
            ).fetchall())

    def record_source_attempt(
        self, release_key: str, source_type: str, status: str, detail: str | None = None
    ) -> None:
        """Record that a collection attempt was made for a release/source.

        Persists negative results too (``status='miss'``), so expensive passes
        (e.g. the Bandcamp LLM locator) never re-pay for a release already tried.
        Upserts on (release_key, source_type) so the latest attempt wins.
        """
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_source_attempts
                    (release_key, source_type, status, detail, attempted_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(release_key, source_type) DO UPDATE SET
                    status = excluded.status,
                    detail = excluded.detail,
                    attempted_at = excluded.attempted_at
                """,
                (release_key, source_type, status, detail, _now_iso()),
            )

    def release_keys_attempted(
        self,
        source_type: str,
        *,
        status: str | None = None,
        newer_than_iso: str | None = None,
    ) -> set[str]:
        """Return release_keys with a prior attempt of the given source_type.

        Includes misses — that's the point: an attempted-but-not-found release
        should be skipped on reruns, not retried.

        ``status`` filters to a single attempt status (e.g. ``"miss"``).
        ``newer_than_iso`` filters to attempts whose ``attempted_at`` is at or
        after the given ISO-8601 cutoff — used for TTL-bounded rechecks, so a
        miss eventually ages out and is retried (e.g. a Last.fm album that gets
        tagged later). ``attempted_at`` is stored via ``_now_iso`` (UTC, second
        precision, fixed ``+00:00`` offset) so lexical ``>=`` is chronological.
        """
        sql = "SELECT release_key FROM ai_genre_source_attempts WHERE source_type = ?"
        params: list[object] = [source_type]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if newer_than_iso is not None:
            sql += " AND attempted_at >= ?"
            params.append(newer_than_iso)
        with self.connect() as conn:
            return {row[0] for row in conn.execute(sql, params)}

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
        model: str | None = None,
    ) -> bool:
        """Run deterministic source-tag classification for one source page.

        Unknown tags (review_only) are checked against the adjudication cache first.
        If adjudicate=True, any remaining unknowns are sent to the AI in a single batch.

        Returns True if adjudicate_tags was called (uncached AI work was done).
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
            return False

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

        return adjudicate

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
                          'bandcamp_tags',
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

            from .genre_vocabulary import GenreVocabulary
            _vocab = GenreVocabulary()

            now = _now_iso()
            expanded_rows = []
            for row in source_rows:
                basis = evidence_basis(row["source_type"])
                confidence = (
                    0.90 if row["reviewed_classification"] == "genre_style" and row["classification"] != "genre_style"
                    else row["confidence"]
                )
                canonical = _vocab.resolve_alias(row["normalized_tag"])
                decomposed = _vocab.decompose_tag(canonical)
                genres = [g for g in (decomposed if decomposed else [canonical]) if g not in ALWAYS_PRUNE_GENRES]
                for genre in genres:
                    expanded_rows.append((row, genre, basis, confidence))

            seed_genres = {
                genre
                for row, genre, _, _ in expanded_rows
                if can_seed_signature(row["source_type"])
            }
            rows = []
            expanded_genres: set[str] = set()
            for row, genre, basis, confidence in expanded_rows:
                if not can_seed_signature(row["source_type"]) and genre not in seed_genres:
                    continue
                rows.append((
                    row["release_key"],
                    row["normalized_artist"],
                    row["normalized_album"],
                    row["album_id"],
                    genre,
                    basis,
                    confidence,
                    row["source_tag_id"],
                    row["source_page_id"],
                    f"source_tag:{row['source_tag_id']}",
                    "accepted",
                    STABILIZED_POLICY_VERSION,
                    now,
                ))
                expanded_genres.add(genre)
            if rows:
                conn.executemany(
                    """
                    INSERT INTO enriched_genres (
                        release_key, normalized_artist, normalized_album, album_id,
                        genre, basis, confidence, source_tag_id, source_page_id,
                        source_ref, status, enrichment_policy_version, added_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            if metadata_row and rows:
                override_row = conn.execute(
                    "SELECT genres_add_json, genres_remove_json "
                    "FROM ai_genre_user_overrides WHERE release_key = ?",
                    (release_key,),
                ).fetchone()
                final_genres: set[str] = set(expanded_genres)
                if override_row:
                    final_genres -= set(json.loads(override_row["genres_remove_json"]))
                    final_genres |= set(json.loads(override_row["genres_add_json"]))
                signature = {
                    "genres": sorted(final_genres),
                    "sources": _signature_sources(source_rows),
                }
                conn.execute(
                    """
                    INSERT INTO enriched_genre_signatures (
                        release_key, normalized_artist, normalized_album, album_id,
                        signature_json, enrichment_policy_version, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata_row["release_key"],
                        metadata_row["normalized_artist"],
                        metadata_row["normalized_album"],
                        metadata_row["album_id"],
                        json.dumps(signature, ensure_ascii=False, sort_keys=True),
                        STABILIZED_POLICY_VERSION,
                        now,
                    ),
                )

    def replace_hybrid_enriched_genres_for_release(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        accepted_genres: list[dict[str, Any]],
    ) -> int:
        """Replace one release's sidecar enriched genres from a fused hybrid report."""
        now = _now_iso()
        rows: list[tuple[Any, ...]] = []
        signature_sources: list[dict[str, str]] = []
        final_genres: set[str] = set()

        for item in accepted_genres:
            genre = str(item.get("term") or "").strip().casefold()
            if not genre or genre in ALWAYS_PRUNE_GENRES:
                continue
            basis = str(item.get("basis") or "hybrid_evidence")
            confidence = float(item.get("confidence") or 0.0)
            source_ref = f"hybrid:{basis}:{genre}"
            rows.append((
                release_key,
                normalized_artist,
                normalized_album,
                album_id,
                genre,
                basis,
                confidence,
                None,
                None,
                source_ref,
                "accepted",
                STABILIZED_POLICY_VERSION,
                now,
            ))
            final_genres.add(genre)
            signature_sources.append({
                "source_type": "hybrid_evidence",
                "source_url": f"hybrid://{basis.replace('+', '/')}/{genre.replace(' ', '%20')}",
            })

        with self.connect() as conn:
            conn.execute("DELETE FROM enriched_genres WHERE release_key = ?", (release_key,))
            conn.execute("DELETE FROM enriched_genre_signatures WHERE release_key = ?", (release_key,))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO enriched_genres (
                        release_key, normalized_artist, normalized_album, album_id,
                        genre, basis, confidence, source_tag_id, source_page_id,
                        source_ref, status, enrichment_policy_version, added_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                override_row = conn.execute(
                    "SELECT genres_add_json, genres_remove_json "
                    "FROM ai_genre_user_overrides WHERE release_key = ?",
                    (release_key,),
                ).fetchone()
                if override_row:
                    final_genres -= set(json.loads(override_row["genres_remove_json"]))
                    final_genres |= set(json.loads(override_row["genres_add_json"]))
                signature = {
                    "genres": sorted(final_genres),
                    "sources": sorted(
                        signature_sources,
                        key=lambda item: (item["source_type"], item["source_url"]),
                    ),
                }
                conn.execute(
                    """
                    INSERT INTO enriched_genre_signatures (
                        release_key, normalized_artist, normalized_album, album_id,
                        signature_json, enrichment_policy_version, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        release_key,
                        normalized_artist,
                        normalized_album,
                        album_id,
                        json.dumps(signature, ensure_ascii=False, sort_keys=True),
                        STABILIZED_POLICY_VERSION,
                        now,
                    ),
                )
        return len(rows)

    def mark_check_complete(self, check_id: int) -> None:
        """Mark a release check as reviewed so it leaves the escalation queue."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE ai_genre_release_checks SET status = 'complete' WHERE check_id = ?",
                (check_id,),
            )
            conn.commit()

    def get_escalated_queue(
        self,
        *,
        release_key: str | None = None,
        artist: str | None = None,
        album: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return one row per actionable suggestion (add/prune) for escalated releases.

        Rows are grouped by release (consecutive) so a caller can flush per release.
        Each row carries the parent check's identity, confidence, and response_json.
        """
        with self.connect() as conn:
            clauses = ["c.status = 'needs_review'", "s.suggestion_type IN ('add', 'prune')"]
            params: list[Any] = []
            if release_key:
                clauses.append("c.release_key = ?")
                params.append(release_key)
            if artist:
                clauses.append("c.normalized_artist = ?")
                params.append(artist)
            if album:
                clauses.append("c.normalized_album = ?")
                params.append(album)
            where = " AND ".join(clauses)
            rows = list(conn.execute(
                f"""
                SELECT
                    c.check_id,
                    c.release_key,
                    c.normalized_artist,
                    c.normalized_album,
                    c.overall_confidence,
                    c.evidence_quality,
                    c.response_json,
                    s.suggestion_id,
                    s.suggestion_type,
                    s.genre,
                    s.confidence AS suggestion_confidence,
                    s.reason,
                    s.recommendation_basis
                FROM ai_genre_release_checks c
                JOIN ai_genre_suggestions s ON s.check_id = c.check_id
                WHERE {where}
                ORDER BY c.release_key, s.suggestion_type, s.suggestion_id
                """,
                params,
            ))
            return [dict(row) for row in rows]

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
        """Cache an AI adjudication result, incrementing times_seen on conflict.

        First-answer-wins: on conflict, only times_seen and decided_at are updated,
        not classification/confidence. Re-run with graduate-ai to promote a new decision
        after a model upgrade.
        """
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

    def find_model_prior(
        self,
        *,
        release_key: str,
        provider: str,
        model: str,
        prompt_version: str,
        taxonomy_version: str,
        schema_version: str,
        enrichment_policy_version: str,
        input_hash: str,
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM ai_genre_model_priors
                WHERE release_key = ? AND provider = ? AND model = ? AND prompt_version = ?
                  AND taxonomy_version = ? AND schema_version = ?
                  AND enrichment_policy_version = ? AND input_hash = ?
                """,
                (
                    release_key, provider, model, prompt_version, taxonomy_version,
                    schema_version, enrichment_policy_version, input_hash,
                ),
            ).fetchone()
            return dict(row) if row else None

    def hybrid_source_terms_for_release(self, release_key: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.source_type,
                    p.source_domain,
                    p.normalized_artist,
                    p.identity_confidence,
                    t.normalized_tag AS term,
                    t.normalized_tag AS canonical_slug,
                    COALESCE(d.reviewed_classification, c.classification) AS mapping_status,
                    c.confidence,
                    CASE
                        WHEN d.reviewed_classification IS NOT NULL THEN 'human'
                        ELSE c.classifier
                    END AS classifier
                FROM ai_genre_tag_classifications c
                JOIN ai_genre_source_tags t ON t.source_tag_id = c.source_tag_id
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                LEFT JOIN ai_genre_review_decisions d ON d.source_tag_id = t.source_tag_id AND d.reviewer = 'human'
                WHERE p.release_key = ?
                  AND p.identity_status IN ('confirmed', 'probable')
                  AND (d.reviewed_classification IS NULL OR d.reviewed_classification != 'rejected')
                ORDER BY p.source_type, t.normalized_tag, t.source_tag_id
                """,
                (release_key,),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "source_type": canonical_source_type(row["source_type"]),
                }
                for row in rows
            ]

    def bandcamp_domain_artist_counts(self) -> dict[str, int]:
        """Distinct-artist count per bandcamp domain (label-storefront signal).

        A domain hosting releases by multiple artists in this store is a label
        storefront; its tags describe the catalog, not any one release.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT source_domain, COUNT(DISTINCT normalized_artist) AS n
                FROM ai_genre_source_pages
                WHERE source_domain LIKE '%bandcamp.com%'
                GROUP BY source_domain
                """
            ).fetchall()
            return {str(row["source_domain"]): int(row["n"]) for row in rows}

    def latest_model_prior_terms_for_release(self, release_key: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*
                FROM ai_genre_model_prior_terms t
                JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
                WHERE t.release_key = ?
                  AND p.status = 'complete'
                  AND p.updated_at = (
                      SELECT MAX(p2.updated_at)
                      FROM ai_genre_model_priors p2
                      WHERE p2.release_key = p.release_key
                        AND p2.status = 'complete'
                  )
                ORDER BY t.normalized_term
                """,
                (release_key,),
            ).fetchall()
            return [dict(row) for row in rows]

    def accepted_enriched_genres_for_release(self, release_key: str) -> list[dict[str, Any]]:
        """Return accepted genres from the enriched_genres table."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT genre FROM enriched_genres WHERE release_key = ? AND status = 'accepted' ORDER BY genre",
                (release_key,),
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_check_suggestions_for_release(self, release_key: str) -> list[dict[str, Any]]:
        """Return keep/add suggestions from the most recent complete AI enrichment check."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.genre, s.confidence, s.recommendation_basis
                FROM ai_genre_suggestions s
                JOIN ai_genre_release_checks c ON c.check_id = s.check_id
                WHERE c.release_key = ?
                  AND c.status = 'complete'
                  AND s.suggestion_type IN ('keep', 'add')
                  AND s.genre IS NOT NULL
                  AND c.checked_at = (
                      SELECT MAX(c2.checked_at)
                      FROM ai_genre_release_checks c2
                      WHERE c2.release_key = c.release_key AND c2.status = 'complete'
                  )
                ORDER BY s.suggestion_id
                """,
                (release_key,),
            ).fetchall()
            return [dict(row) for row in rows]

    def record_model_prior(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        album_id: str | None,
        provider: str,
        model: str,
        prompt_version: str,
        taxonomy_version: str,
        schema_version: str,
        enrichment_policy_version: str,
        input_hash: str,
        status: str,
        response_json: dict[str, Any] | None,
        warnings: list[str],
        error_message: str | None,
        token_usage: dict[str, int],
        estimated_cost_usd: float | None,
        mapped_terms: list[dict[str, Any]],
    ) -> int:
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_model_priors (
                    release_key, normalized_artist, normalized_album, album_id,
                    provider, model, prompt_version, taxonomy_version, schema_version,
                    enrichment_policy_version, input_hash, status, response_json,
                    warnings_json, error_message, input_tokens, output_tokens,
                    total_tokens, estimated_cost_usd, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    release_key, provider, model, prompt_version, taxonomy_version,
                    schema_version, enrichment_policy_version, input_hash
                ) DO UPDATE SET
                    status = excluded.status,
                    response_json = excluded.response_json,
                    warnings_json = excluded.warnings_json,
                    error_message = excluded.error_message,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    total_tokens = excluded.total_tokens,
                    estimated_cost_usd = excluded.estimated_cost_usd,
                    updated_at = excluded.updated_at
                """,
                (
                    release_key, normalized_artist, normalized_album, album_id,
                    provider, model, prompt_version, taxonomy_version, schema_version,
                    enrichment_policy_version, input_hash, status,
                    json.dumps(response_json, sort_keys=True) if response_json is not None else None,
                    json.dumps(warnings, sort_keys=True), error_message,
                    token_usage.get("input_tokens"), token_usage.get("output_tokens"),
                    token_usage.get("total_tokens"), estimated_cost_usd, now, now,
                ),
            )
            prior_id = int(conn.execute(
                """
                SELECT prior_id FROM ai_genre_model_priors
                WHERE release_key = ? AND provider = ? AND model = ? AND prompt_version = ?
                  AND taxonomy_version = ? AND schema_version = ?
                  AND enrichment_policy_version = ? AND input_hash = ?
                """,
                (
                    release_key, provider, model, prompt_version, taxonomy_version,
                    schema_version, enrichment_policy_version, input_hash,
                ),
            ).fetchone()["prior_id"])
            conn.execute(
                "DELETE FROM ai_genre_model_prior_terms WHERE prior_id = ?",
                (prior_id,),
            )
            conn.executemany(
                """
                INSERT INTO ai_genre_model_prior_terms (
                    prior_id, release_key, raw_term, normalized_term, canonical_slug,
                    confidence, specificity, taxonomy_role, mapping_status,
                    accepted_for_shadow, auto_apply_eligible, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                [
                    (
                        prior_id, release_key, term["raw_term"], term["normalized_term"],
                        term["canonical_slug"], term["confidence"], term["specificity"],
                        term["taxonomy_role"], term["mapping_status"],
                        term["accepted_for_shadow"], term["notes"], now,
                    )
                    for term in mapped_terms
                ],
            )
            return prior_id

    def model_prior_report(self) -> dict[str, Any]:
        with self.connect() as conn:
            status_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM ai_genre_model_priors GROUP BY status ORDER BY status"
                )
            }
            mapping_status_counts = {
                row["mapping_status"]: row["count"]
                for row in conn.execute(
                    "SELECT mapping_status, COUNT(*) AS count FROM ai_genre_model_prior_terms "
                    "GROUP BY mapping_status ORDER BY mapping_status"
                )
            }
            accepted = conn.execute(
                "SELECT COUNT(*) FROM ai_genre_model_prior_terms WHERE accepted_for_shadow = 1"
            ).fetchone()[0]
            return {
                "status_counts": status_counts,
                "mapping_status_counts": mapping_status_counts,
                "accepted_for_shadow": accepted,
            }

    def upsert_layered_taxonomy(self, taxonomy: Any) -> dict[str, int]:
        """Persist a loaded layered taxonomy into the sidecar graph tables."""
        with self.connect() as conn:
            # The reviewed taxonomy can rename ids while keeping canonical names.
            # Replace the sidecar graph registry as a unit to avoid merging stale
            # seed versions into the active taxonomy.
            for table in (
                "genre_graph_aliases",
                "genre_graph_edges",
                "genre_graph_bridge_rules",
                "genre_graph_rejected_terms",
                "genre_graph_canonical_facets",
                "genre_graph_canonical_genres",
            ):
                conn.execute(f"DELETE FROM {table}")
            conn.executemany(
                """
                INSERT INTO genre_graph_canonical_genres (
                    genre_id, name, kind, specificity_score, status, taxonomy_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(genre_id) DO UPDATE SET
                    name = excluded.name,
                    kind = excluded.kind,
                    specificity_score = excluded.specificity_score,
                    status = excluded.status,
                    taxonomy_version = excluded.taxonomy_version
                """,
                [
                    (
                        genre.genre_id,
                        genre.name,
                        genre.kind,
                        genre.specificity_score,
                        genre.status,
                        genre.taxonomy_version,
                    )
                    for genre in taxonomy.genres
                ],
            )
            conn.executemany(
                """
                INSERT INTO genre_graph_aliases (
                    alias, canonical_genre_id, source, confidence
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    canonical_genre_id = excluded.canonical_genre_id,
                    source = excluded.source,
                    confidence = excluded.confidence
                """,
                [
                    (alias.alias, alias.canonical_genre_id, alias.source, alias.confidence)
                    for alias in taxonomy.aliases
                    if getattr(alias, "target_kind", "genre") == "genre"
                ],
            )
            conn.executemany(
                """
                INSERT INTO genre_graph_edges (
                    source_genre_id, target_genre_id, edge_type,
                    weight, confidence, source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_genre_id, target_genre_id, edge_type) DO UPDATE SET
                    weight = excluded.weight,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    notes = excluded.notes
                """,
                [
                    (
                        edge.source_genre_id,
                        edge.target_genre_id,
                        edge.edge_type,
                        edge.weight,
                        edge.confidence,
                        edge.source,
                        edge.notes,
                    )
                    for edge in taxonomy.edges
                ],
            )
            conn.executemany(
                """
                INSERT INTO genre_graph_canonical_facets (
                    facet_id, name, facet_type, status
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(facet_id) DO UPDATE SET
                    name = excluded.name,
                    facet_type = excluded.facet_type,
                    status = excluded.status
                """,
                [
                    (facet.facet_id, facet.name, facet.facet_type, facet.status)
                    for facet in taxonomy.facets
                ],
            )
            conn.executemany(
                """
                INSERT INTO genre_graph_bridge_rules (
                    source_genre_id, target_genre_id, required_family_min,
                    required_facet_overlap, required_sonic_similarity,
                    required_transition_quality, mode_allowed, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_genre_id, target_genre_id) DO UPDATE SET
                    required_family_min = excluded.required_family_min,
                    required_facet_overlap = excluded.required_facet_overlap,
                    required_sonic_similarity = excluded.required_sonic_similarity,
                    required_transition_quality = excluded.required_transition_quality,
                    mode_allowed = excluded.mode_allowed,
                    notes = excluded.notes
                """,
                [
                    (
                        rule.source_genre_id,
                        rule.target_genre_id,
                        rule.required_family_min,
                        rule.required_facet_overlap,
                        rule.required_sonic_similarity,
                        rule.required_transition_quality,
                        ",".join(rule.mode_allowed),
                        rule.notes,
                    )
                    for rule in taxonomy.bridge_rules
                ],
            )
            conn.executemany(
                """
                INSERT INTO genre_graph_rejected_terms (
                    term, reason
                ) VALUES (?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    reason = excluded.reason
                """,
                [
                    (term.term, term.reason)
                    for term in taxonomy.rejected_terms
                ],
            )
            return {
                "genre_count": len(taxonomy.genres),
                "alias_count": len([alias for alias in taxonomy.aliases if getattr(alias, "target_kind", "genre") == "genre"]),
                "edge_count": len(taxonomy.edges),
                "facet_count": len(taxonomy.facets),
                "bridge_rule_count": len(taxonomy.bridge_rules),
                "rejected_term_count": len(taxonomy.rejected_terms),
            }

    def layered_taxonomy_report(self) -> dict[str, Any]:
        with self.connect() as conn:
            taxonomy_version = conn.execute(
                """
                SELECT taxonomy_version
                FROM genre_graph_canonical_genres
                GROUP BY taxonomy_version
                ORDER BY COUNT(*) DESC, taxonomy_version
                LIMIT 1
                """
            ).fetchone()
            genre_counts_by_kind = {
                row["kind"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT kind, COUNT(*) AS count
                    FROM genre_graph_canonical_genres
                    GROUP BY kind
                    ORDER BY kind
                    """
                )
            }
            facet_counts_by_type = {
                row["facet_type"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT facet_type, COUNT(*) AS count
                    FROM genre_graph_canonical_facets
                    GROUP BY facet_type
                    ORDER BY facet_type
                    """
                )
            }
            edge_counts_by_type = {
                row["edge_type"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT edge_type, COUNT(*) AS count
                    FROM genre_graph_edges
                    GROUP BY edge_type
                    ORDER BY edge_type
                    """
                )
            }
            alias_count = int(conn.execute("SELECT COUNT(*) FROM genre_graph_aliases").fetchone()[0])
            bridge_rule_count = int(conn.execute("SELECT COUNT(*) FROM genre_graph_bridge_rules").fetchone()[0])
            rejected_term_count = int(conn.execute("SELECT COUNT(*) FROM genre_graph_rejected_terms").fetchone()[0])
            review_count = int(conn.execute(
                "SELECT COUNT(*) FROM genre_graph_canonical_genres WHERE status = 'review'"
            ).fetchone()[0])
            deprecated_count = int(conn.execute(
                "SELECT COUNT(*) FROM genre_graph_canonical_genres WHERE status = 'deprecated'"
            ).fetchone()[0])
        return {
            "taxonomy_version": taxonomy_version["taxonomy_version"] if taxonomy_version else None,
            "genre_counts_by_kind": genre_counts_by_kind,
            "facet_counts_by_type": facet_counts_by_type,
            "edge_counts_by_type": edge_counts_by_type,
            "alias_count": alias_count,
            "bridge_rule_count": bridge_rule_count,
            "rejected_term_count": rejected_term_count,
            "review_count": review_count,
            "deprecated_count": deprecated_count,
        }

    def has_genre_assignments(self, release_id: str) -> bool:
        """True if the release already has any materialized graph genre rows."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM genre_graph_release_genre_assignments "
                "WHERE release_id = ? LIMIT 1",
                (release_id,),
            ).fetchone()
        return row is not None

    def materialization_fingerprint(self, release_id: str) -> str | None:
        """Evidence fingerprint recorded at the release's last materialization.

        None when the release was never fingerprinted (drives the incremental
        enrich guard's bootstrap-adopt: existing assignments are kept as-is).
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT evidence_fingerprint FROM genre_graph_release_materialization "
                "WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def set_materialization_fingerprint(self, release_id: str, fingerprint: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO genre_graph_release_materialization "
                "(release_id, evidence_fingerprint, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(release_id) DO UPDATE SET "
                "evidence_fingerprint = excluded.evidence_fingerprint, "
                "updated_at = excluded.updated_at",
                (release_id, fingerprint, _now_iso()),
            )

    def replace_layered_assignments_for_release(
        self,
        *,
        release_id: str,
        artist: str,
        album: str,
        genre_assignments: list[dict[str, Any]],
        facet_assignments: list[dict[str, Any]],
    ) -> dict[str, int]:
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM genre_graph_release_genre_assignments WHERE release_id = ?",
                (release_id,),
            )
            conn.execute(
                "DELETE FROM genre_graph_release_facet_assignments WHERE release_id = ?",
                (release_id,),
            )
            if genre_assignments:
                conn.executemany(
                    """
                    INSERT INTO genre_graph_release_genre_assignments (
                        release_id, artist, album, genre_id, assignment_layer,
                        confidence, source_reliability, evidence_count,
                        rejected_by_user, provenance_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            release_id,
                            artist,
                            album,
                            row["genre_id"],
                            row["assignment_layer"],
                            row["confidence"],
                            row["source_reliability"],
                            row["evidence_count"],
                            1 if row.get("rejected_by_user") else 0,
                            json.dumps(row.get("provenance", {}), sort_keys=True),
                            now,
                        )
                        for row in genre_assignments
                    ],
                )
            if facet_assignments:
                conn.executemany(
                    """
                    INSERT INTO genre_graph_release_facet_assignments (
                        release_id, artist, album, facet_id, confidence,
                        source, provenance_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            release_id,
                            artist,
                            album,
                            row["facet_id"],
                            row["confidence"],
                            row["source"],
                            json.dumps(row.get("provenance", {}), sort_keys=True),
                            now,
                        )
                        for row in facet_assignments
                    ],
                )
            return {
                "genre_assignment_count": len(genre_assignments),
                "facet_assignment_count": len(facet_assignments),
            }

    def layered_release_summary(self, release_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            genre_rows = [
                {
                    "genre_id": row["genre_id"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "assignment_layer": row["assignment_layer"],
                    "confidence": float(row["confidence"]),
                    "source_reliability": float(row["source_reliability"]),
                    "evidence_count": int(row["evidence_count"]),
                    "rejected_by_user": bool(row["rejected_by_user"]),
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
                for row in conn.execute(
                    """
                    SELECT
                        r.genre_id, c.name, c.kind, r.assignment_layer,
                        r.confidence, r.source_reliability, r.evidence_count,
                        r.rejected_by_user, r.provenance_json
                    FROM genre_graph_release_genre_assignments r
                    LEFT JOIN genre_graph_canonical_genres c
                      ON c.genre_id = r.genre_id
                    WHERE r.release_id = ?
                    ORDER BY
                        CASE r.assignment_layer
                            WHEN 'human' THEN 0
                            WHEN 'observed_leaf' THEN 1
                            WHEN 'model_prior' THEN 2
                            WHEN 'inferred_parent' THEN 3
                            WHEN 'inferred_family' THEN 4
                            ELSE 9
                        END,
                        c.name,
                        r.genre_id
                    """,
                    (release_id,),
                )
            ]
            facet_rows = [
                {
                    "facet_id": row["facet_id"],
                    "name": row["name"],
                    "facet_type": row["facet_type"],
                    "confidence": float(row["confidence"]),
                    "source": row["source"],
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
                for row in conn.execute(
                    """
                    SELECT
                        r.facet_id, f.name, f.facet_type, r.confidence,
                        r.source, r.provenance_json
                    FROM genre_graph_release_facet_assignments r
                    LEFT JOIN genre_graph_canonical_facets f
                      ON f.facet_id = r.facet_id
                    WHERE r.release_id = ?
                    ORDER BY f.facet_type, f.name, r.facet_id
                    """,
                    (release_id,),
                )
            ]

        genres_by_layer: dict[str, list[dict[str, Any]]] = {}
        for row in genre_rows:
            genres_by_layer.setdefault(str(row["assignment_layer"]), []).append(row)
        return {
            "release_id": release_id,
            "genres_by_layer": genres_by_layer,
            "facets": facet_rows,
            "genre_assignment_count": len(genre_rows),
            "facet_assignment_count": len(facet_rows),
        }

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
                genre = str(item.get("genre") or "").casefold()
                if genre in ALWAYS_PRUNE_GENRES:
                    rows.append(_suggestion_row(check_id, "prune", item, genre_key="genre"))
                else:
                    rows.append(_suggestion_row(check_id, "keep", item, genre_key="genre"))
            for item in response_json.get("existing_genres_to_prune", []):
                rows.append(_suggestion_row(check_id, "prune", item, genre_key="genre"))
            for item in response_json.get("new_genres_to_add", []):
                if str(item.get("genre") or "").casefold() in ALWAYS_PRUNE_GENRES:
                    continue
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

    # -------------------------------------------------------------------------
    # User override CRUD
    # -------------------------------------------------------------------------

    def set_user_override(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        genres_add: list[str],
        genres_remove: list[str],
    ) -> None:
        """Upsert the manual genre override for a release. Replaces prior entry."""
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_user_overrides (
                    release_key, normalized_artist, normalized_album,
                    genres_add_json, genres_remove_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_key) DO UPDATE SET
                    genres_add_json = excluded.genres_add_json,
                    genres_remove_json = excluded.genres_remove_json,
                    updated_at = excluded.updated_at
                """,
                (
                    release_key, normalized_artist, normalized_album,
                    json.dumps(sorted({g.casefold() for g in genres_add if g.casefold() not in ALWAYS_PRUNE_GENRES})),
                    json.dumps(sorted({g.casefold() for g in genres_remove})),
                    now,
                ),
            )
            conn.commit()

    def get_user_override(self, release_key: str) -> dict | None:
        """Return the override dict or None when no override exists."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT genres_add_json, genres_remove_json, updated_at "
                "FROM ai_genre_user_overrides WHERE release_key = ?",
                (release_key,),
            ).fetchone()
        if not row:
            return None
        return {
            "genres_add": json.loads(row["genres_add_json"]),
            "genres_remove": json.loads(row["genres_remove_json"]),
            "updated_at": row["updated_at"],
        }

    def delete_user_override(self, release_key: str) -> None:
        """Remove the manual override for a release."""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM ai_genre_user_overrides WHERE release_key = ?",
                (release_key,),
            )
            conn.commit()

    def list_review_scan_releases(self) -> list[dict[str, Any]]:
        """Distinct releases known to the evidence layer, for the review scan."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT release_key,
                       MIN(normalized_artist) AS normalized_artist,
                       MIN(normalized_album) AS normalized_album
                FROM ai_genre_source_pages
                GROUP BY release_key
                ORDER BY release_key
                """
            )
            return [dict(row) for row in rows]

    def sync_review_queue_for_release(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        terms: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Reconcile pending queue rows for one release against a fresh scan.

        Inserts new terms, refreshes still-pending ones, prunes pending rows
        whose term no longer appears. Rows with a decided status are never
        touched, so rescans cannot resurrect settled questions.
        """
        now = _now_iso()
        term_names = {t["term"] for t in terms}
        inserted = updated = pruned = 0
        with self.connect() as conn:
            existing = {
                row["term"]: row["status"]
                for row in conn.execute(
                    "SELECT term, status FROM ai_genre_review_queue WHERE release_key = ?",
                    (release_key,),
                )
            }
            for term, status in existing.items():
                if status == "pending" and term not in term_names:
                    conn.execute(
                        "DELETE FROM ai_genre_review_queue WHERE release_key = ? AND term = ?",
                        (release_key, term),
                    )
                    pruned += 1
            for t in terms:
                status = existing.get(t["term"])
                if status is None:
                    conn.execute(
                        """
                        INSERT INTO ai_genre_review_queue (
                            release_key, normalized_artist, normalized_album, term,
                            confidence, basis, sources_json, reason, status, scanned_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (
                            release_key, normalized_artist, normalized_album, t["term"],
                            t.get("confidence"), t.get("basis") or "hybrid_fusion",
                            json.dumps(list(t.get("sources") or [])),
                            t.get("reason") or "", now,
                        ),
                    )
                    inserted += 1
                elif status == "pending":
                    conn.execute(
                        """
                        UPDATE ai_genre_review_queue
                        SET confidence = ?, basis = ?, sources_json = ?, reason = ?, scanned_at = ?
                        WHERE release_key = ? AND term = ? AND status = 'pending'
                        """,
                        (
                            t.get("confidence"), t.get("basis") or "hybrid_fusion",
                            json.dumps(list(t.get("sources") or [])),
                            t.get("reason") or "", now, release_key, t["term"],
                        ),
                    )
                    updated += 1
            conn.commit()
        return {"inserted": inserted, "updated": updated, "pruned": pruned}

    def get_review_queue_page(
        self,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
        readonly: bool = False,
    ) -> dict[str, Any]:
        """Releases with pending review terms, most-pending first.

        Header counts (pending_releases / pending_terms) always describe the
        whole queue, not the filtered page.

        ``readonly=True`` is for inline/latency-sensitive callers (the GUI
        review panel polling from the worker's reader thread): uses a read-only
        connection that cannot block on writers, and returns an EMPTY page when
        the DB or queue table doesn't exist yet (fresh install, scan never run)
        instead of creating schema or raising.
        """
        if readonly:
            try:
                conn = self.connect_readonly()
            except sqlite3.OperationalError:
                return {"releases": [], "pending_releases": 0, "pending_terms": 0}
            try:
                return self._review_queue_page(conn, search=search, limit=limit, offset=offset)
            except sqlite3.OperationalError:
                # e.g. "no such table: ai_genre_review_queue" before first scan
                return {"releases": [], "pending_releases": 0, "pending_terms": 0}
            finally:
                conn.close()
        with self.connect() as conn:
            return self._review_queue_page(conn, search=search, limit=limit, offset=offset)

    def _review_queue_page(
        self,
        conn: sqlite3.Connection,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        with conn:
            counts = conn.execute(
                "SELECT COUNT(DISTINCT release_key) AS pr, COUNT(*) AS pt "
                "FROM ai_genre_review_queue WHERE status = 'pending'"
            ).fetchone()
            decided = conn.execute(
                "SELECT COUNT(DISTINCT release_key) AS dr, COUNT(*) AS dt "
                "FROM ai_genre_review_queue WHERE status != 'pending'"
            ).fetchone()
            where = ""
            params: list[Any] = []
            if search:
                where = "WHERE (normalized_artist LIKE ? OR normalized_album LIKE ?)"
                pattern = f"%{search.strip().casefold()}%"
                params = [pattern, pattern]
            release_rows = list(conn.execute(
                f"""
                SELECT release_key, normalized_artist, normalized_album,
                       SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count
                FROM ai_genre_review_queue
                {where}
                GROUP BY release_key, normalized_artist, normalized_album
                HAVING pending_count > 0
                ORDER BY pending_count DESC, release_key
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ))
            releases = []
            for rel in release_rows:
                terms = [
                    {
                        "term": row["term"],
                        "confidence": row["confidence"],
                        "basis": row["basis"],
                        "sources": json.loads(row["sources_json"]),
                        "reason": row["reason"],
                        "status": row["status"],
                    }
                    for row in conn.execute(
                        "SELECT term, confidence, basis, sources_json, reason, status "
                        "FROM ai_genre_review_queue WHERE release_key = ? "
                        "ORDER BY confidence DESC, term",
                        (rel["release_key"],),
                    )
                ]
                releases.append({
                    "release_key": rel["release_key"],
                    "artist": rel["normalized_artist"],
                    "album": rel["normalized_album"],
                    "pending": [t for t in terms if t["status"] == "pending"],
                    "decided": [t for t in terms if t["status"] != "pending"],
                })
        return {
            "releases": releases,
            "pending_releases": int(counts["pr"]),
            "pending_terms": int(counts["pt"]),
            "decided_releases": int(decided["dr"]),
            "decided_terms": int(decided["dt"]),
        }

    def get_completed_review_page(
        self,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
        readonly: bool = False,
    ) -> dict[str, Any]:
        """Releases with at least one *decided* review term, newest decision first.

        The 'Completed' counterpart to ``get_review_queue_page``: surfaces work
        the user already settled — including fully-decided releases that drop off
        the pending queue — so saved progress is browsable and revertible.
        Header counts (decided_releases / decided_terms) describe the whole queue,
        not the filtered page. ``readonly=True`` behaves exactly as on the pending
        page (read-only connection, empty page when the DB/table is absent).
        """
        if readonly:
            try:
                conn = self.connect_readonly()
            except sqlite3.OperationalError:
                return {"releases": [], "decided_releases": 0, "decided_terms": 0}
            try:
                return self._completed_review_page(conn, search=search, limit=limit, offset=offset)
            except sqlite3.OperationalError:
                return {"releases": [], "decided_releases": 0, "decided_terms": 0}
            finally:
                conn.close()
        with self.connect() as conn:
            return self._completed_review_page(conn, search=search, limit=limit, offset=offset)

    def _completed_review_page(
        self,
        conn: sqlite3.Connection,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        with conn:
            counts = conn.execute(
                "SELECT COUNT(DISTINCT release_key) AS dr, COUNT(*) AS dt "
                "FROM ai_genre_review_queue WHERE status != 'pending'"
            ).fetchone()
            where = ""
            params: list[Any] = []
            if search:
                where = "WHERE (normalized_artist LIKE ? OR normalized_album LIKE ?)"
                pattern = f"%{search.strip().casefold()}%"
                params = [pattern, pattern]
            release_rows = list(conn.execute(
                f"""
                SELECT release_key, normalized_artist, normalized_album,
                       SUM(CASE WHEN status != 'pending' THEN 1 ELSE 0 END) AS decided_count,
                       MAX(decided_at) AS last_decided
                FROM ai_genre_review_queue
                {where}
                GROUP BY release_key, normalized_artist, normalized_album
                HAVING decided_count > 0
                ORDER BY last_decided DESC, release_key
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ))
            releases = []
            for rel in release_rows:
                terms = [
                    {
                        "term": row["term"],
                        "confidence": row["confidence"],
                        "basis": row["basis"],
                        "sources": json.loads(row["sources_json"]),
                        "reason": row["reason"],
                        "status": row["status"],
                    }
                    for row in conn.execute(
                        "SELECT term, confidence, basis, sources_json, reason, status "
                        "FROM ai_genre_review_queue WHERE release_key = ? "
                        "ORDER BY decided_at DESC, confidence DESC, term",
                        (rel["release_key"],),
                    )
                ]
                releases.append({
                    "release_key": rel["release_key"],
                    "artist": rel["normalized_artist"],
                    "album": rel["normalized_album"],
                    "pending": [t for t in terms if t["status"] == "pending"],
                    "decided": [t for t in terms if t["status"] != "pending"],
                })
        return {
            "releases": releases,
            "decided_releases": int(counts["dr"]),
            "decided_terms": int(counts["dt"]),
        }

    def set_review_queue_status(
        self, *, release_key: str, term: str, status: str
    ) -> None:
        """Set a queue row's status. 'pending' clears decided_at (revert)."""
        if status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid review queue status: {status}")
        decided_at = None if status == "pending" else _now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE ai_genre_review_queue SET status = ?, decided_at = ? "
                "WHERE release_key = ? AND term = ?",
                (status, decided_at, release_key, term),
            )
            if cur.rowcount == 0:
                raise ValueError(f"no review queue row for {release_key!r} / {term!r}")
            conn.commit()


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
        if not can_seed_signature(row["source_type"]):
            continue
        source_type = canonical_source_type(row["source_type"])
        key = (source_type, row["source_url"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_type": source_type, "source_url": row["source_url"]})
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
