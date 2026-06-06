import json
import sqlite3
import pytest
from src.genre import genre_publish


def _meta_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_create_published_schema_creates_all_tables():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {
        "genre_graph_canonical_genres",
        "genre_graph_canonical_facets",
        "genre_graph_edges",
        "genre_graph_aliases",
        "genre_graph_bridge_rules",
        "genre_graph_rejected_terms",
        "genre_graph_taxonomy_meta",
        "genre_graph_release_genre_assignments",
        "genre_graph_release_facet_assignments",
        "release_effective_genres",
    }
    assert expected <= names


def test_create_published_schema_is_idempotent():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    genre_publish.create_published_schema(conn)  # must not raise
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info('release_effective_genres')"
    ).fetchall()}
    assert {"album_id", "release_key", "genre_id", "assignment_layer",
            "confidence", "source"} <= cols
