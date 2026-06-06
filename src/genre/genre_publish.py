"""Publish authoritative layered genres from the enrichment sidecar into metadata.db.

See docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

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
