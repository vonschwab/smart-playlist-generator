import sqlite3

from src.genre import authority


def _canon(conn):
    conn.execute(
        "CREATE TABLE genre_graph_canonical_genres ("
        "genre_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        "specificity_score REAL NOT NULL, status TEXT NOT NULL, taxonomy_version TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        [("dream_pop", "Dream Pop", "genre", 0.8, "active", "v1"),
         ("dreamo", "Dreamo", "genre", 0.7, "active", "v1"),
         ("shoegaze", "Shoegaze", "genre", 0.9, "active", "v1"),
         ("old_thing", "Old Dream", "genre", 0.5, "deprecated", "v1")],
    )


def test_canonical_genre_search_matches_active_by_name():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _canon(conn)
    out = authority.canonical_genre_search(conn, "dream", limit=10)
    names = [n for _, n in out]
    assert "Dream Pop" in names and "Dreamo" in names
    assert "Old Dream" not in names  # deprecated excluded
    assert ("shoegaze", "Shoegaze") not in out


def test_canonical_genre_search_empty_query():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _canon(conn)
    assert authority.canonical_genre_search(conn, "  ", limit=10) == []
