import sqlite3
from pathlib import Path

import pytest

from scripts.scan_library import LibraryScanner


def _make_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "library:\n"
        f"  music_directory: {tmp_path}\n"
        f"  database_path: {tmp_path / 'test.db'}\n"
        "openai:\n"
        "  api_key: dummy\n"
    )
    return cfg_path


def _init_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            duration_ms INTEGER,
            file_mtime_ns INTEGER,
            file_size_bytes INTEGER,
            tags_fingerprint TEXT,
            file_path TEXT,
            file_modified INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE albums (
            album_id TEXT PRIMARY KEY,
            artist TEXT,
            title TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_album_id_normalization_is_stable(tmp_path: Path):
    cfg = _make_config(tmp_path)
    db_path = _init_db(tmp_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    aid1 = scanner._compute_album_id("Radiohead", "OK Computer")
    aid2 = scanner._compute_album_id(" radiohead ", "  ok   computer")
    assert aid1 == aid2

    aid3 = scanner._compute_album_id("Beyoncé", "Lemonade")
    aid4 = scanner._compute_album_id("Beyonce", "LEMONADE")
    assert aid3 == aid4

    assert scanner._compute_album_id(None, None) is None
    scanner.close()


def test_upsert_track_sets_album_id_and_album_row(tmp_path: Path):
    cfg = _make_config(tmp_path)
    db_path = _init_db(tmp_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    metadata = {
        "artist": "Radiohead",
        "title": "Karma Police",
        "album": "OK Computer",
        "duration": 245,
        "file_path": str(tmp_path / "karma_police.mp3"),
        "file_modified": 123,
    }

    scanner.upsert_track(metadata)
    cursor = scanner.conn.cursor()
    row = cursor.execute("SELECT album_id FROM tracks WHERE track_id IS NOT NULL").fetchone()
    assert row is not None
    album_id = row[0]
    assert album_id

    album_row = cursor.execute(
        "SELECT artist, title FROM albums WHERE album_id = ?",
        (album_id,),
    ).fetchone()
    assert album_row is not None
    assert album_row[0] == "Radiohead"
    assert album_row[1] == "OK Computer"
    scanner.close()


def test_backfill_updates_missing_album_ids_only(tmp_path: Path):
    cfg = _make_config(tmp_path)
    db_path = _init_db(tmp_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    cur = scanner.conn.cursor()
    cur.execute(
        "INSERT INTO tracks (track_id, artist, title, album, album_id, file_path, file_modified) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("t1", "Artist One", "Song A", "Album X", None, "/tmp/a.mp3", 1),
    )
    cur.execute(
        "INSERT INTO tracks (track_id, artist, title, album, album_id, file_path, file_modified) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("t2", "Artist Two", "Song B", "Album Y", "preserve123", "/tmp/b.mp3", 1),
    )
    scanner.conn.commit()

    stats = scanner.backfill_album_ids()
    assert stats["updated"] == 1

    refreshed = scanner.conn.cursor()
    backfilled = refreshed.execute("SELECT album_id FROM tracks WHERE track_id='t1'").fetchone()[0]
    preserved = refreshed.execute("SELECT album_id FROM tracks WHERE track_id='t2'").fetchone()[0]
    assert backfilled
    assert preserved == "preserve123"

    albums_count = refreshed.execute(
        "SELECT COUNT(*) FROM albums WHERE album_id = ?", (backfilled,)
    ).fetchone()[0]
    assert albums_count == 1
    scanner.close()
