import json
import sqlite3

from src.genre import genre_publish


def _schema(conn):
    conn.execute(
        "CREATE TABLE release_effective_genres ("
        "album_id TEXT NOT NULL, release_key TEXT, genre_id TEXT NOT NULL, "
        "assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, "
        "PRIMARY KEY (album_id, genre_id, assignment_layer))"
    )
    conn.execute(
        "CREATE TABLE genre_graph_release_genre_assignments ("
        "album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL)"
    )


def test_materialize_orphan_album_user_only():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _schema(conn)
    # Orphan: no graph rows, no legacy; user adds two genres.
    genre_publish.materialize_album_genres(
        conn, "ALB1",
        graph_album_ids=set(), legacy={},
        overrides={"ALB1": (["dream_pop", "shoegaze"], set())},
        album_to_key={"ALB1": "the radio dept::pet grief"},
    )
    rows = conn.execute(
        "SELECT genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres WHERE album_id='ALB1' ORDER BY genre_id"
    ).fetchall()
    assert [(r["genre_id"], r["assignment_layer"], r["confidence"], r["source"]) for r in rows] == [
        ("dream_pop", "observed_leaf", 1.0, "user"),
        ("shoegaze", "observed_leaf", 1.0, "user"),
    ]


def test_materialize_graph_minus_remove_plus_add():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _schema(conn)
    conn.executemany(
        "INSERT INTO genre_graph_release_genre_assignments VALUES (?,?,?,?)",
        [("ALB2", "cool_jazz", "observed_leaf", 0.9),
         ("ALB2", "jazz", "inferred_family", 0.9)],
    )
    genre_publish.materialize_album_genres(
        conn, "ALB2",
        graph_album_ids={"ALB2"}, legacy={},
        overrides={"ALB2": (["post_bop"], {"cool_jazz"})},
        album_to_key={"ALB2": "k"},
    )
    got = {(r["genre_id"], r["source"]) for r in conn.execute(
        "SELECT genre_id, source FROM release_effective_genres WHERE album_id='ALB2'")}
    assert got == {("jazz", "graph"), ("post_bop", "user")}


def _publish_db(tmp_path):
    """metadata.db + sidecar with the tables resolve/build_resolved_table read.

    Orphan: tracks reference album_id ORPH1, but no `albums` row exists for it.
    """
    meta_path = tmp_path / "m.db"
    side_path = tmp_path / "s.db"

    meta = sqlite3.connect(meta_path)
    meta.row_factory = sqlite3.Row
    meta.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE release_effective_genres ("
        " album_id TEXT NOT NULL, release_key TEXT, genre_id TEXT NOT NULL, "
        " assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, "
        " PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    meta.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    meta.commit()

    sconn = sqlite3.connect(side_path)
    sconn.executescript(
        "CREATE TABLE enriched_genre_signatures (release_key TEXT, album_id TEXT);"
        "CREATE TABLE ai_genre_user_overrides ("
        " release_key TEXT, normalized_artist TEXT, normalized_album TEXT, "
        " genres_add_json TEXT, genres_remove_json TEXT, updated_at TEXT);"
    )
    sconn.execute(
        "INSERT INTO ai_genre_user_overrides VALUES "
        "('the radio dept::pet grief','the radio dept','pet grief',?,'[]','t')",
        (json.dumps(["dream pop"]),),
    )
    sconn.commit()
    sconn.close()

    meta.execute("ATTACH DATABASE ? AS side", (str(side_path),))
    return meta


def test_orphan_release_key_maps_from_tracks(tmp_path):
    """resolve_release_key_to_album_id derives album_id from tracks for orphans."""
    meta = _publish_db(tmp_path)
    mapping, _collisions = genre_publish.resolve_release_key_to_album_id(meta)
    # double-space artist normalizes the same as single-space → release_key matches
    assert mapping.get("the radio dept::pet grief") == "ORPH1"


def test_orphan_override_survives_build_resolved_table(tmp_path):
    """A user override on an album with tracks but no albums row publishes."""
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    meta = _publish_db(tmp_path)
    taxonomy = load_default_layered_taxonomy()
    expected_id = genre_publish._term_to_genre_id(taxonomy, "dream pop")
    assert expected_id, "'dream pop' must resolve to a canonical genre_id"

    mapping, _ = genre_publish.resolve_release_key_to_album_id(meta)
    genre_publish.build_resolved_table(meta, mapping, taxonomy)

    got = {
        (r["genre_id"], r["source"])
        for r in meta.execute(
            "SELECT genre_id, source FROM release_effective_genres WHERE album_id='ORPH1'"
        )
    }
    assert (expected_id, "user") in got, f"orphan override lost: {got}"
