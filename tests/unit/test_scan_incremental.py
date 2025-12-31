import os
import sqlite3
import time
from pathlib import Path

from scripts.scan_library import LibraryScanner


def _config(tmp_path: Path, db_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "library:\n"
        f"  music_directory: {tmp_path}\n"
        f"  database_path: {db_path}\n"
        "openai:\n"
        "  api_key: dummy\n"
    )
    return cfg


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
            file_modified INTEGER,
            last_updated TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE track_genres (
            track_id TEXT,
            genre TEXT,
            source TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_unchanged_file_skips_update(tmp_path: Path):
    db_path = _init_db(tmp_path)
    cfg = _config(tmp_path, db_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    fpath = tmp_path / "song.mp3"
    fpath.write_text("v1")
    stat = fpath.stat()
    metadata = {
        "artist": "Artist",
        "title": "Title",
        "album": "Album",
        "duration": 10,
        "file_path": str(fpath),
        "file_modified": int(stat.st_mtime),
        "file_mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
    }
    track_id, _ = scanner.upsert_track(metadata)
    scanner.conn.execute(
        "UPDATE tracks SET last_updated = 'yesterday' WHERE track_id = ?",
        (track_id,),
    )
    scanner.conn.commit()

    # Run again with identical metadata and preserved mtime/size
    track_id2, is_new = scanner.upsert_track(metadata)
    assert track_id2 == track_id
    assert is_new is False
    last_updated = scanner.conn.execute(
        "SELECT last_updated FROM tracks WHERE track_id = ?",
        (track_id,),
    ).fetchone()[0]
    assert last_updated == "yesterday"

    # Genres should not be churned
    count_genres = scanner.conn.execute("SELECT COUNT(*) FROM track_genres").fetchone()[0]
    assert count_genres == 0
    scanner.close()


def test_tag_change_with_same_mtime_triggers_update(tmp_path: Path):
    db_path = _init_db(tmp_path)
    cfg = _config(tmp_path, db_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    fpath = tmp_path / "song2.mp3"
    fpath.write_text("v1")
    stat = fpath.stat()
    base_meta = {
        "artist": "Artist",
        "title": "Title",
        "album": "Album",
        "duration": 10,
        "file_path": str(fpath),
        "file_modified": int(stat.st_mtime),
        "file_mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
    }
    track_id, _ = scanner.upsert_track(base_meta)

    # Simulate tag change without touching mtime/size
    changed = dict(base_meta)
    changed["album"] = "Album v2"
    track_id2, is_new = scanner.upsert_track(changed)

    assert track_id2 == track_id
    assert is_new is False
    row = scanner.conn.execute(
        "SELECT album FROM tracks WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    assert row[0] == "Album v2"
    scanner.close()


def test_mtime_or_size_change_triggers_update(tmp_path: Path):
    db_path = _init_db(tmp_path)
    cfg = _config(tmp_path, db_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    fpath = tmp_path / "song3.mp3"
    fpath.write_text("v1")
    stat = fpath.stat()
    meta = {
        "artist": "Artist",
        "title": "Title",
        "album": "Album",
        "duration": 10,
        "file_path": str(fpath),
        "file_modified": int(stat.st_mtime),
        "file_mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
    }
    track_id, _ = scanner.upsert_track(meta)

    time.sleep(0.01)
    fpath.write_text("v2")  # changes size and mtime
    stat2 = fpath.stat()
    meta2 = dict(meta)
    meta2["file_modified"] = int(stat2.st_mtime)
    meta2["file_mtime_ns"] = stat2.st_mtime_ns
    meta2["file_size"] = stat2.st_size

    track_id2, is_new = scanner.upsert_track(meta2)
    assert track_id2 == track_id  # path-based track_id is stable
    assert is_new is False
    stored_size = scanner.conn.execute(
        "SELECT file_size_bytes FROM tracks WHERE track_id = ?", (track_id,),
    ).fetchone()[0]
    assert stored_size == stat2.st_size
    scanner.close()
