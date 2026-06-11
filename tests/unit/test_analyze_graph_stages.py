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


def _ctx(tmp_path: Path, db_path: str, **arg_overrides):
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

    ctx = _ctx(tmp_path, db_path)
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
    ctx = _ctx(tmp_path, db_path, lastfm_api_key=None)
    # also ensure config lookup can't supply a key
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: None)
    with pytest.raises(RuntimeError, match="Last.fm API key"):
        al.stage_lastfm(ctx)
    ctx["conn"].close()


def _seed_sidecar_with_pages(sidecar: str):
    """One release with a lastfm source page carrying a known + an unknown tag."""
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="alb1",
        source_url="lastfm://artist/slowdive/album/souvlaki",
        source_type="lastfm_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="seed",
    )
    # 'shoegaze' classifies deterministically; 'zzz unknown thing' is review_only.
    store.replace_source_tags(page_id, ["shoegaze", "zzz unknown thing"])
    return store


class _RecordingAdjudicator:
    """Stand-in for adjudicate_tags: records calls, returns canned classifications."""
    def __init__(self):
        self.calls = []

    def __call__(self, tags, *, model=None, dry_run=False, client=None):
        self.calls.append([norm for _, norm in tags])
        return {
            norm: {"classification": "genre_style", "confidence": 0.8, "reason": "ok"}
            for _, norm in tags
        }


def test_stage_enrich_dedupes_adjudicates_and_materializes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    rec = _RecordingAdjudicator()
    monkeypatch.setattr(al, "adjudicate_tags", rec)

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["releases_enriched"] == 1
    # exactly one distinct unknown tag adjudicated, in a single chunk
    assert rec.calls == [["zzz unknown thing"]]
    assert result["tags_adjudicated"] == 1


def test_stage_enrich_no_pending_skips(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()  # empty sidecar, no source pages
    monkeypatch.setattr(al, "adjudicate_tags", _RecordingAdjudicator())

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()
    assert result["skipped"] is True
    assert result.get("releases_enriched", 0) == 0


def test_stage_enrich_propagates_adjudication_failure(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    def boom(tags, *, model=None, dry_run=False, client=None):
        raise RuntimeError("Claude Code request failed after retries: rate window")

    monkeypatch.setattr(al, "adjudicate_tags", boom)
    ctx = _ctx(tmp_path, db_path)
    with pytest.raises(RuntimeError, match="failed after retries"):
        al.stage_enrich(ctx)
    ctx["conn"].close()


def _published_table_exists(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='release_effective_genres'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def test_stage_publish_first_run_backs_up_and_publishes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path)
    result = al.stage_publish(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["backed_up"] is True
    # a timestamped backup was created next to metadata.db
    backups = list(Path(db_path).parent.glob("metadata.db.bak.*"))
    assert backups, "expected a first-publish backup"
    # release_effective_genres now exists
    assert _published_table_exists(db_path)
    assert result["validation_ok"] is True


def test_stage_publish_second_run_no_backup(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    al.stage_publish(_ctx(tmp_path, db_path))  # first publish (backs up)
    before = set(Path(db_path).parent.glob("metadata.db.bak.*"))
    ctx2 = _ctx(tmp_path, db_path)
    result = al.stage_publish(ctx2)
    ctx2["conn"].close()
    after = set(Path(db_path).parent.glob("metadata.db.bak.*"))

    assert result["backed_up"] is False
    assert before == after, "second publish must not create a new backup"


def test_stage_publish_dry_run_rolls_back(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path, dry_run=True)
    result = al.stage_publish(ctx)
    ctx["conn"].close()
    assert result["dry_run"] is True
    # dry-run rolls back the publish transaction → no published table persists
    assert not _published_table_exists(db_path)
