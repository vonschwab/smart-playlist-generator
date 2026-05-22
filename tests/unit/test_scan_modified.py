import os
import sqlite3
from pathlib import Path

import pytest

from scripts.scan_library import LibraryScanner


def _make_config(tmp_path: Path, db_path: Path, music_dir: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "library:\n"
        f"  database_path: {db_path}\n"
        f"  music_directory: {music_dir}\n"
        "openai:\n"
        "  api_key: test-key\n",
        encoding="utf-8",
    )
    return config_path


def _make_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            artist_key TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            album_artist TEXT,
            file_path TEXT,
            file_modified INTEGER,
            file_mtime_ns INTEGER,
            file_size_bytes INTEGER,
            tags_fingerprint TEXT,
            duration_ms INTEGER,
            sonic_features TEXT
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS artist_keys (artist_key TEXT PRIMARY KEY, artist TEXT)")
    conn.commit()
    return conn


def test_quick_scan_skips_unchanged(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    audio_file = music_dir / "song.mp3"
    audio_file.write_bytes(b"hello")
    stat = audio_file.stat()

    db_path = tmp_path / "db.sqlite"
    conn = _make_db(db_path)
    conn.execute(
        """
        INSERT INTO tracks (track_id, artist, artist_key, title, album, album_id, album_artist,
                            file_path, file_modified, file_mtime_ns, file_size_bytes, tags_fingerprint,
                            duration_ms, sonic_features)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "t1",
            "Artist",
            "artist",
            "Title",
            "Album",
            "alb1",
            "Artist",
            str(audio_file),
            int(stat.st_mtime),
            stat.st_mtime_ns,
            stat.st_size,
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    config_path = _make_config(tmp_path, db_path, music_dir)
    scanner = LibraryScanner(config_path=str(config_path), db_path=str(db_path))
    files = scanner.scan_files(quick=True)
    scanner.close()
    assert files == []


@pytest.mark.skipif(os.name != "nt", reason="Path case normalization applies to Windows")
def test_quick_scan_matches_case_insensitive(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    audio_file = music_dir / "Track.FLAC"
    audio_file.write_bytes(b"data")
    stat = audio_file.stat()

    db_path = tmp_path / "db.sqlite"
    conn = _make_db(db_path)
    conn.execute(
        "INSERT INTO tracks (track_id, artist, artist_key, title, album, album_id, album_artist, "
        "file_path, file_modified, file_mtime_ns, file_size_bytes, tags_fingerprint, duration_ms, sonic_features) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "t2",
            "Artist",
            "artist",
            "Title",
            "Album",
            "alb2",
            "Artist",
            str(audio_file).upper(),
            int(stat.st_mtime),
            stat.st_mtime_ns,
            stat.st_size,
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    config_path = _make_config(tmp_path, db_path, music_dir)
    scanner = LibraryScanner(config_path=str(config_path), db_path=str(db_path))
    files = scanner.scan_files(quick=True)
    scanner.close()
    assert files == []


def test_quick_scan_uses_file_modified_fallback(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    audio_file = music_dir / "fallback.ogg"
    audio_file.write_bytes(b"data2")
    stat = audio_file.stat()

    db_path = tmp_path / "db.sqlite"
    conn = _make_db(db_path)
    # Leave fingerprint columns NULL, rely on file_modified seconds
    conn.execute(
        "INSERT INTO tracks (track_id, artist, artist_key, title, album, album_id, album_artist, "
        "file_path, file_modified, file_mtime_ns, file_size_bytes, tags_fingerprint, duration_ms, sonic_features) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "t3",
            "Artist",
            "artist",
            "Title",
            "Album",
            "alb3",
            "Artist",
            str(audio_file),
            int(stat.st_mtime),
            None,
            None,
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    config_path = _make_config(tmp_path, db_path, music_dir)
    scanner = LibraryScanner(config_path=str(config_path), db_path=str(db_path))
    files = scanner.scan_files(quick=True)
    scanner.close()
    assert files == []
