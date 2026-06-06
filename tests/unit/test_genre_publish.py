import sqlite3
from src.genre import genre_publish
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


def _meta_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_sidecar(tmp_path):
    """A sidecar with the real taxonomy loaded, nothing else."""
    side = tmp_path / "sidecar.db"
    store = SidecarStore(str(side))
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())
    return side


def _attach(meta_conn, side_path):
    # Plain path (no file: URI) to avoid ATTACH URI-mode pitfalls. The sidecar is
    # only ever read (SELECT) by the publish code, so writable-attach is harmless.
    meta_conn.execute("ATTACH DATABASE ? AS side", (str(side_path),))


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
    assert expected == names


def test_create_published_schema_is_idempotent():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    genre_publish.create_published_schema(conn)  # must not raise
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info('release_effective_genres')"
    ).fetchall()}
    assert {"album_id", "release_key", "genre_id", "assignment_layer",
            "confidence", "source"} <= cols


def test_copy_taxonomy_copies_rows_and_writes_meta(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)

    side_genres = conn.execute(
        "SELECT COUNT(*) FROM side.genre_graph_canonical_genres"
    ).fetchone()[0]
    main_genres = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_canonical_genres"
    ).fetchone()[0]
    assert main_genres == side_genres > 0

    meta = conn.execute(
        "SELECT version, fingerprint, published_at FROM genre_graph_taxonomy_meta"
    ).fetchall()
    assert len(meta) == 1
    assert meta[0]["version"]  # non-empty
    assert len(meta[0]["fingerprint"]) == 64


def test_copy_taxonomy_is_idempotent(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    genre_publish.copy_taxonomy(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM genre_graph_taxonomy_meta"
    ).fetchone()[0] == 1
