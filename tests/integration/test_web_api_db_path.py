"""Regression coverage: src/playlist_web/app.py must resolve DB_PATH /
SIDECAR_DB_PATH from the config passed to create_app(), not from a
ROOT-anchored module constant.

Root cause (2026-07-16): DB_PATH/SIDECAR_DB_PATH were hardcoded to
``ROOT / "data" / "metadata.db"``. In a satellite clone, that file is a
0-byte stub, so every DB-backed endpoint silently degraded to an empty
result (the sqlite errors were swallowed). This mirrors the 2026-07-07
CLI/worker fix (`resolve_database_path(config)`), extended to app.py.
"""
import logging
import sqlite3

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app
from tests.fixtures.fake_worker import FAKE_WORKER_CMD as FAKE


def _seed_db(db_path):
    """Minimal schema for resolved_genres_for_artist() (src/genre/authority.py):
    tracks -> release_effective_genres -> genre_graph_canonical_genres."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE tracks (
                track_id TEXT PRIMARY KEY, artist TEXT, album_id TEXT
            );
            CREATE TABLE release_effective_genres (
                album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL
            );
            CREATE TABLE genre_graph_canonical_genres (
                genre_id TEXT PRIMARY KEY, name TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO tracks (track_id, artist, album_id) VALUES (?, ?, ?)",
            ("t1", "Swirlies", "a1"),
        )
        conn.execute(
            "INSERT INTO release_effective_genres "
            "(album_id, genre_id, assignment_layer, confidence) VALUES (?, ?, ?, ?)",
            ("a1", "shoegaze", "observed_leaf", 0.9),
        )
        conn.execute(
            "INSERT INTO genre_graph_canonical_genres (genre_id, name) VALUES (?, ?)",
            ("shoegaze", "shoegaze"),
        )
        conn.commit()
    finally:
        conn.close()


def _write_config(tmp_path, db_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"library:\n  database_path: {db_path.as_posix()}\n",
        encoding="utf-8",
    )
    return cfg


def test_genres_for_artist_reads_db_from_config(tmp_path):
    """The endpoint must read the DB the passed-in config points at, not the
    ROOT-anchored module constant (which is a 0-byte stub in a satellite)."""
    db_path = tmp_path / "metadata.db"
    _seed_db(db_path)
    cfg_path = _write_config(tmp_path, db_path)

    app = create_app(worker_cmd=FAKE, config_path=str(cfg_path))
    with TestClient(app) as client:
        r = client.get("/api/genres/for_artist", params={"artist": "Swirlies"})
        assert r.status_code == 200
        names = [g["name"] for g in r.json()["genres"]]
        assert names == ["shoegaze"]


def test_missing_db_logs_warning_and_returns_graceful_empty(tmp_path, caplog):
    """A configured DB that exists but has no schema (satellite-stub shape)
    must log a loud warning naming the DB path — never swallow silently —
    while still returning the documented empty response shape."""
    db_path = tmp_path / "metadata.db"
    sqlite3.connect(str(db_path)).close()  # 0-byte-equivalent: exists, no tables
    cfg_path = _write_config(tmp_path, db_path)

    app = create_app(worker_cmd=FAKE, config_path=str(cfg_path))
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING, logger="src.playlist_web.app"):
            r = client.get("/api/artists/search", params={"q": "sw"})
        assert r.status_code == 200
        assert r.json() == {"items": []}
        assert any(str(db_path) in rec.message for rec in caplog.records), (
            "expected a warning naming the DB path; got: "
            f"{[rec.message for rec in caplog.records]}"
        )
