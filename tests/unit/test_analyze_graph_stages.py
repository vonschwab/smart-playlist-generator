"""Unit tests for the analyze_library graph stages (lastfm, enrich, publish)."""
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.analyze_library as al
from src.ai_genre_enrichment.storage import SidecarStore


def _metadata_db(tmp_path: Path) -> str:
    """Minimal metadata.db: one album with one track and an artist genre."""
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
            album_id TEXT, title TEXT, file_path TEXT);
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        INSERT INTO tracks VALUES ('t1','Slowdive','Souvlaki','alb1','Alison','/x/a.flac');
        INSERT INTO albums VALUES ('alb1','Souvlaki','Slowdive');
        INSERT INTO artist_genres VALUES ('Slowdive','shoegaze','musicbrainz_artist');
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _ctx(tmp_path: Path, db_path: str, sidecar: str, **arg_overrides):
    """Build a minimal stage ctx with a live metadata.db connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    defaults = dict(
        force=False, limit=None, dry_run=False, progress=False, verbose=False,
        progress_interval=15.0, progress_every=500, max_tracks=0, model=None,
        enrich_chunk_size=50, lastfm_api_key="FAKEKEY",
    )
    defaults.update(arg_overrides)
    args = Namespace(**defaults)
    return {
        "config_path": str(tmp_path / "config.yaml"),
        "db_path": db_path,
        "out_dir": tmp_path,
        "args": args,
        "conn": conn,
        "config_hash": "test",
        "library_root": str(tmp_path),
        "genres_dirty": False, "sonic_dirty": False,
        "artifacts_dirty": False, "force_stage": False,
    }


def test_stage_lastfm_fetches_stores_and_classifies(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))

    captured = {}

    def fake_fetch(artist, album, api_key, limit=20):
        captured["args"] = (artist, album, api_key, limit)
        return ["shoegaze", "dream pop", "ambient"]

    monkeypatch.setattr(al, "fetch_lastfm_tags", fake_fetch)

    ctx = _ctx(tmp_path, db_path, sidecar)
    result = al.stage_lastfm(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["extracted"] == 1
    # tags landed in the sidecar as a lastfm_tags source page
    store = SidecarStore(sidecar)
    assert "slowdive::souvlaki" in store.release_keys_with_source_type("lastfm_tags")


def test_stage_lastfm_missing_key_raises(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    ctx = _ctx(tmp_path, db_path, sidecar, lastfm_api_key=None)
    # also ensure config lookup can't supply a key
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: None)
    with pytest.raises(RuntimeError, match="Last.fm API key"):
        al.stage_lastfm(ctx)
    ctx["conn"].close()
