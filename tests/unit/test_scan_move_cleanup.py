import sqlite3
from pathlib import Path

from scripts.scan_library import LibraryScanner


def _setup_db(tmp_path: Path) -> Path:
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
            sonic_features TEXT,
            musicbrainz_id TEXT,
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
    conn.execute(
        """
        CREATE TABLE track_effective_genres (
            track_id TEXT,
            genre TEXT,
            source TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


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


def test_cleanup_guard_skips_deletions_when_empty_discovery(tmp_path: Path, monkeypatch):
    db_path = _setup_db(tmp_path)
    cfg = _config(tmp_path, db_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    # Seed a track and genre row
    cur = scanner.conn.cursor()
    cur.execute(
        "INSERT INTO tracks (track_id, artist, title, album, album_id, file_path, file_modified) "
        "VALUES ('t1','A','Song','Album','aid','/missing.mp3',1)"
    )
    cur.execute(
        "INSERT INTO track_genres (track_id, genre, source) VALUES ('t1','rock','file')"
    )
    scanner.conn.commit()

    # Simulate empty discovery and missing library dir
    monkeypatch.setattr(scanner, "scan_files", lambda quick=False: [])
    scanner.music_dir = tmp_path / "missing_dir"

    scanner.run(quick=True, cleanup=True)

    remaining = scanner.conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    assert remaining == 1
    genre_remaining = scanner.conn.execute("SELECT COUNT(*) FROM track_genres").fetchone()[0]
    assert genre_remaining == 1
    scanner.close()


def test_move_detection_updates_existing_row_and_preserves_metadata(tmp_path: Path):
    db_path = _setup_db(tmp_path)
    cfg = _config(tmp_path, db_path)
    scanner = LibraryScanner(config_path=cfg, db_path=db_path)

    old_path = tmp_path / "old.mp3"
    new_path = tmp_path / "new.mp3"

    old_track_id = scanner.generate_track_id(str(old_path), "Artist", "Title")
    cur = scanner.conn.cursor()
    cur.execute(
        "INSERT INTO tracks (track_id, artist, title, album, album_id, sonic_features, file_path, file_modified) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (old_track_id, "Artist", "Title", "Album", "aid123", '{"full":{}}', str(old_path), 1),
    )
    cur.execute(
        "INSERT INTO track_genres (track_id, genre, source) VALUES (?, ?, 'file')",
        (old_track_id, "rock"),
    )
    cur.execute(
        "INSERT INTO track_effective_genres (track_id, genre, source) VALUES (?, ?, 'calc')",
        (old_track_id, "alt"),
    )
    scanner.conn.commit()

    metadata = {
        "artist": "Artist",
        "title": "Title",
        "album": "Album",
        "duration": 200,
        "file_path": str(new_path),
        "file_modified": 2,
    }

    new_track_id, is_new = scanner.upsert_track(metadata)
    assert is_new is False
    assert new_track_id != old_track_id

    row = scanner.conn.execute(
        "SELECT track_id, file_path, sonic_features FROM tracks WHERE track_id = ?",
        (new_track_id,),
    ).fetchone()
    assert row is not None
    assert row[1] == str(new_path)
    assert row[2] == '{"full":{}}'  # preserved

    genres = scanner.conn.execute("SELECT track_id FROM track_genres").fetchall()
    assert len(genres) == 1
    assert genres[0][0] == new_track_id
    eff = scanner.conn.execute("SELECT track_id FROM track_effective_genres").fetchall()
    assert len(eff) == 1
    assert eff[0][0] == new_track_id

    # Audit should show no orphaned track_id refs
    from tools.audit_library_db import audit_database
    report = audit_database(db_path)
    assert all(v == 0 for v in report["orphans"].values())
    scanner.close()
