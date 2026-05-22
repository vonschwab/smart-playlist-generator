import logging
import sqlite3

from src.blacklist_db import ensure_blacklist_schema


def test_blacklist_schema_adds_column_and_index(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT)")
    cur.execute("INSERT INTO tracks (track_id, title) VALUES (?, ?)", ("1", "t"))
    conn.commit()

    ensure_blacklist_schema(conn, logger=logging.getLogger("test"))

    cols = {row[1] for row in cur.execute("PRAGMA table_info(tracks)").fetchall()}
    assert "is_blacklisted" in cols

    cur.execute("SELECT is_blacklisted FROM tracks WHERE track_id = ?", ("1",))
    row = cur.fetchone()
    assert row[0] == 0

    index_names = [row[1] for row in cur.execute("PRAGMA index_list(tracks)").fetchall()]
    assert "idx_tracks_is_blacklisted" in index_names
    tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "blacklisted_artists" in tables
    assert "blacklisted_albums" in tables

    ensure_blacklist_schema(conn, logger=logging.getLogger("test"))
    conn.close()
