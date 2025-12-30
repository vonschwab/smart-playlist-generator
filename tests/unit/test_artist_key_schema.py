import logging
import sqlite3

from src.artist_key_db import ensure_artist_key_schema
from src.string_utils import normalize_artist_key


def test_artist_key_schema_backfill_and_query(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT)")
    cur.executemany(
        "INSERT INTO tracks (track_id, artist) VALUES (?, ?)",
        [
            ("1", "João Gilberto"),
            ("2", "Joao Gilberto"),
            ("3", "Luiz Bonfá"),
        ],
    )
    conn.commit()

    ensure_artist_key_schema(conn, logger=logging.getLogger("test"))

    cols = {row[1] for row in cur.execute("PRAGMA table_info(tracks)").fetchall()}
    assert "artist_key" in cols

    key = normalize_artist_key("Joao Gilberto")
    cur.execute(
        "SELECT track_id FROM tracks WHERE artist_key = ? ORDER BY track_id",
        (key,),
    )
    rows = [row[0] for row in cur.fetchall()]
    assert rows == ["1", "2"]

    index_names = [row[1] for row in cur.execute("PRAGMA index_list(tracks)").fetchall()]
    assert "idx_tracks_artist_key" in index_names

    # Idempotent call
    ensure_artist_key_schema(conn, logger=logging.getLogger("test"))
    conn.close()
