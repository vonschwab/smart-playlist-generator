"""Drives the REAL worker subprocess through WorkerBridge via asyncio
(Proactor loop) — NOT TestClient, which can't faithfully read real-worker
stdout on Windows (see the web-gui skill)."""
import asyncio
import sqlite3
import sys

import pytest

from src.playlist_web.worker_bridge import WorkerBridge

WORKER_CMD = [sys.executable, "-m", "src.playlist_gui.worker"]


async def _noop_event(_ev) -> None:
    return None


@pytest.mark.integration
@pytest.mark.slow
def test_edit_genres_writes_authority(tmp_path):
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE genre_graph_canonical_genres (genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT, "
        " specificity_score REAL, status TEXT, taxonomy_version TEXT);"
        "CREATE TABLE release_effective_genres (album_id TEXT NOT NULL, release_key TEXT, "
        " genre_id TEXT NOT NULL, assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, "
        " source TEXT NOT NULL, PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    conn.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    conn.commit()
    conn.close()

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"library:\n  database_path: {meta.as_posix()}\n"
        "playlists:\n  ds_pipeline:\n    genre_source: graph\n"
    )

    async def run():
        bridge = WorkerBridge(WORKER_CMD, _noop_event)
        await bridge.start()
        try:
            return await bridge.command({
                "cmd": "edit_genres", "base_config_path": str(cfg),
                "artist": "The  Radio Dept.", "album": "Pet Grief",
                "genres": ["dream pop", "shoegaze"],
            }, timeout=120.0)
        finally:
            await bridge.stop()

    result = asyncio.run(run())
    assert result.get("no_change") is False
    assert len(result.get("added", [])) == 2

    c = sqlite3.connect(meta)
    rows = c.execute(
        "SELECT genre_id FROM release_effective_genres "
        "WHERE album_id='ORPH1' AND source='user'"
    ).fetchall()
    c.close()
    assert len(rows) == 2
