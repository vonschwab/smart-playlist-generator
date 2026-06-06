"""Publish authoritative layered genres from the enrichment sidecar into metadata.db.

See docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict  # noqa: F401 — used in Task 8: PublishStats
from datetime import datetime, timezone

from src.ai_genre_enrichment.layered_assignment import classify_layered_term
from src.ai_genre_enrichment.normalization import (
    normalize_release_artist,
    normalize_release_name,
)

try:
    from src.genre.normalize import normalize_and_split_genre
    _NORMALIZE_AVAILABLE = True
except Exception:  # pragma: no cover - normalization optional
    _NORMALIZE_AVAILABLE = False

_WEIGHT_TRACK = 1.0
_WEIGHT_ALBUM = 0.8
_WEIGHT_ARTIST = 0.5


def _split(raw: str) -> list[str]:
    if not raw or raw == "__EMPTY__":
        return []
    if _NORMALIZE_AVAILABLE:
        return [t for t in normalize_and_split_genre(raw) if t]
    token = raw.strip().casefold()
    return [token] if token else []

# Taxonomy + authority DDL mirrors src/ai_genre_enrichment/storage.py so the
# published tables are schema-faithful copies. Authority tables add `album_id`.
_PUBLISHED_DDL = """
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
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_edges (
    source_genre_id TEXT NOT NULL,
    target_genre_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id, edge_type)
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
    PRIMARY KEY (source_genre_id, target_genre_id)
);
CREATE TABLE IF NOT EXISTS genre_graph_rejected_terms (
    term TEXT PRIMARY KEY,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_taxonomy_meta (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
    version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    published_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_release_genre_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
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
    PRIMARY KEY (release_id, genre_id, assignment_layer)
);
CREATE TABLE IF NOT EXISTS genre_graph_release_facet_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
    artist TEXT NOT NULL,
    album TEXT NOT NULL,
    facet_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (release_id, facet_id, source)
);
CREATE TABLE IF NOT EXISTS release_effective_genres (
    album_id TEXT NOT NULL,
    release_key TEXT,
    genre_id TEXT NOT NULL,
    assignment_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (album_id, genre_id, assignment_layer)
);
CREATE INDEX IF NOT EXISTS idx_release_effective_genres_album
    ON release_effective_genres (album_id);
"""

# Tables this sub-project owns. Order matters for DROP (children first is N/A
# since FKs are not enforced, but keep a stable list for unpublish()).
# Note: idx_release_effective_genres_album is intentionally absent — SQLite
# drops indexes automatically when their parent table is dropped.
PUBLISHED_TABLES = [
    "release_effective_genres",
    "genre_graph_release_genre_assignments",
    "genre_graph_release_facet_assignments",
    "genre_graph_taxonomy_meta",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
    "genre_graph_canonical_facets",
    "genre_graph_canonical_genres",
]


def create_published_schema(conn: sqlite3.Connection) -> None:
    """Create all published genre tables in metadata.db (idempotent).

    Executes statements individually (NOT executescript, which would implicitly
    COMMIT and break the publish() dry-run rollback). The DDL contains no
    embedded semicolons, so a simple split is safe.
    """
    for stmt in _PUBLISHED_DDL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TAXONOMY_COPY_TABLES = [
    "genre_graph_canonical_genres",
    "genre_graph_canonical_facets",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
]


def _taxonomy_fingerprint(conn: sqlite3.Connection) -> str:
    """Stable hash of the published taxonomy (genres + edges)."""
    genres = conn.execute(
        "SELECT genre_id, name, kind, specificity_score, status "
        "FROM genre_graph_canonical_genres ORDER BY genre_id"
    ).fetchall()
    edges = conn.execute(
        "SELECT source_genre_id, target_genre_id, edge_type, weight "
        "FROM genre_graph_edges ORDER BY source_genre_id, target_genre_id, edge_type"
    ).fetchall()
    payload = json.dumps(
        {"genres": [tuple(r) for r in genres], "edges": [tuple(r) for r in edges]},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_taxonomy(conn: sqlite3.Connection) -> None:
    """Copy taxonomy tables from attached `side` DB; rewrite taxonomy_meta.

    Requires the sidecar attached as schema `side`.
    """
    for table in _TAXONOMY_COPY_TABLES:
        conn.execute(f"DELETE FROM {table}")
        conn.execute(f"INSERT INTO {table} SELECT * FROM side.{table}")
    version_row = conn.execute(
        "SELECT taxonomy_version FROM genre_graph_canonical_genres LIMIT 1"
    ).fetchone()
    version = version_row[0] if version_row else "unknown"
    conn.execute("DELETE FROM genre_graph_taxonomy_meta")
    conn.execute(
        "INSERT INTO genre_graph_taxonomy_meta (version, fingerprint, published_at) "
        "VALUES (?, ?, ?)",
        (version, _taxonomy_fingerprint(conn), _now_iso()),
    )


def resolve_release_key_to_album_id(
    conn: sqlite3.Connection,
) -> tuple[dict[str, str], int]:
    """Build release_key -> album_id. Signatures win; albums recompute fills gaps.

    Returns (mapping, collision_count). Requires sidecar attached as `side`.
    """
    mapping: dict[str, str] = {}

    # 1) exact from signatures
    for row in conn.execute(
        "SELECT release_key, album_id FROM side.enriched_genre_signatures "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        mapping[row[0]] = row[1]

    # 2) recompute from albums for keys not already mapped
    collisions = 0
    computed: dict[str, str] = {}
    for album_id, title, artist in conn.execute(
        "SELECT album_id, title, artist FROM albums "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        key = f"{normalize_release_artist(artist)}::{normalize_release_name(title)}"
        if not key or key == "::":
            continue
        if key in computed and computed[key] != album_id:
            collisions += 1
            # deterministic: keep the lexicographically smaller album_id
            computed[key] = min(computed[key], album_id)
        else:
            computed[key] = album_id
    for key, album_id in computed.items():
        mapping.setdefault(key, album_id)

    return mapping, collisions


def populate_authority(conn: sqlite3.Connection, key_to_album: dict[str, str]) -> None:
    """Copy graph genre + facet assignments into metadata.db, stamping album_id.

    Requires sidecar attached as `side`. Pure graph (no overrides here).
    """
    conn.execute("DELETE FROM genre_graph_release_genre_assignments")
    conn.execute("DELETE FROM genre_graph_release_facet_assignments")

    for row in conn.execute(
        "SELECT release_id, artist, album, genre_id, assignment_layer, confidence, "
        "source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at "
        "FROM side.genre_graph_release_genre_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_genre_assignments "
            "(release_id, album_id, artist, album, genre_id, assignment_layer, confidence, "
            " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5],
             row[6], row[7], row[8], row[9], row[10]),
        )

    for row in conn.execute(
        "SELECT release_id, artist, album, facet_id, confidence, source, "
        "provenance_json, updated_at FROM side.genre_graph_release_facet_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_facet_assignments "
            "(release_id, album_id, artist, album, facet_id, confidence, source, "
            " provenance_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5], row[6], row[7]),
        )


def legacy_genres_by_album(conn: sqlite3.Connection) -> dict[str, list[tuple[str, float]]]:
    """Album-grain legacy genres: track(1.0)+album(0.8)+artist(0.5), max weight/token.

    Mirrors the weighting scheme from ``load_genres_for_tracks`` and normalises
    raw genre strings via ``normalize_and_split_genre``.  Returns a mapping of
    ``album_id`` → sorted list of ``(token, weight)`` pairs.  Albums whose only
    genre data is the ``__EMPTY__`` sentinel are excluded from the result.
    """
    acc: dict[str, dict[str, float]] = defaultdict(dict)

    def add(album_id: str, raw: str, base_weight: float) -> None:
        tokens = _split(raw)
        if not tokens:
            return
        per = base_weight / len(tokens)
        for tok in tokens:
            if per > acc[album_id].get(tok, 0.0):
                acc[album_id][tok] = per

    for album_id, genre in conn.execute(
        "SELECT t.album_id, tg.genre FROM tracks t "
        "JOIN track_genres tg ON tg.track_id = t.track_id "
        "WHERE t.album_id IS NOT NULL AND t.album_id != ''"
    ):
        add(album_id, genre, _WEIGHT_TRACK)

    for album_id, genre in conn.execute(
        "SELECT album_id, genre FROM album_genres "
        "WHERE album_id IS NOT NULL AND album_id != '' AND genre != '__EMPTY__'"
    ):
        add(album_id, genre, _WEIGHT_ALBUM)

    for album_id, genre in conn.execute(
        "SELECT a.album_id, ag.genre FROM albums a "
        "JOIN artist_genres ag ON ag.artist = a.artist "
        "WHERE a.album_id IS NOT NULL AND a.album_id != '' AND ag.genre != '__EMPTY__'"
    ):
        add(album_id, genre, _WEIGHT_ARTIST)

    return {aid: sorted(toks.items()) for aid, toks in acc.items() if toks}


def _term_to_genre_id(taxonomy, term: str) -> str | None:
    """Return a graph genre_id for a term, or None if unmappable/non-genre."""
    classification = classify_layered_term(taxonomy, term)
    if classification.term_kind in {"reject", "review", "facet", "alias"}:
        if classification.term_kind == "alias" and classification.canonical_id:
            if taxonomy.genre_by_id(classification.canonical_id) is not None:
                return classification.canonical_id
        return None
    return classification.canonical_id


def classify_override_terms(
    taxonomy, add: list[str], remove: list[str]
) -> tuple[list[str], list[str]]:
    """Map override add/remove names to graph genre_ids (unmappable skipped)."""
    add_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in add) if gid]
    remove_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in remove) if gid]
    return list(dict.fromkeys(add_ids)), list(dict.fromkeys(remove_ids))
