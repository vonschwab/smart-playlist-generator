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
