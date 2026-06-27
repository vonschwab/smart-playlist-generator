"""stage_lastfm must persist 'miss' results so empty releases aren't re-fetched
on every run (TTL-bounded recheck). Mirrors the bandcamp negative-cache pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

import scripts.analyze_library as al
from src.ai_genre_enrichment.storage import SidecarStore


def _rel(release_key: str) -> SimpleNamespace:
    artist, album = release_key.split("::")
    return SimpleNamespace(
        release_key=release_key,
        normalized_artist=artist,
        normalized_album=album,
        album_id=None,
    )


def _args() -> SimpleNamespace:
    # progress=False -> no ProgressLogger; the other progress knobs go unused.
    return SimpleNamespace(limit=None, force=False, progress=False)


def _ctx(tmp_path) -> dict:
    # Non-existent config -> stage falls back to the default recheck window (30d).
    return {
        "args": _args(),
        "config_path": str(tmp_path / "nope.yaml"),
        "db_path": str(tmp_path / "meta.db"),
    }


def test_empty_result_records_miss_and_skips_within_ttl(tmp_path, monkeypatch):
    db = tmp_path / "enr.db"
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", db)
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: "k")
    monkeypatch.setattr(al, "discover_releases", lambda *a, **k: [_rel("acetone::york blvd")])
    monkeypatch.setattr(al, "fetch_lastfm_tags", lambda **k: [])  # Last.fm has no tags
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    res1 = al.stage_lastfm(_ctx(tmp_path))
    assert res1["empty"] == 1

    # The miss must be recorded in the attempt ledger.
    store = SidecarStore(str(db))
    store.initialize()
    assert "acetone::york blvd" in store.release_keys_attempted("lastfm_tags", status="miss")

    # Second run within the TTL: the recent miss is skipped, NOT re-fetched.
    def boom(**k):
        raise AssertionError("a miss within the recheck window must not be re-fetched")

    monkeypatch.setattr(al, "fetch_lastfm_tags", boom)
    res2 = al.stage_lastfm(_ctx(tmp_path))
    assert res2.get("skipped") is True


def test_miss_older_than_ttl_is_refetched_and_timestamp_refreshes(tmp_path, monkeypatch):
    db = tmp_path / "enr.db"
    store = SidecarStore(str(db))
    store.initialize()
    # Pre-seed an ancient miss (well outside any reasonable recheck window).
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO ai_genre_source_attempts "
            "(release_key, source_type, status, detail, attempted_at) VALUES (?,?,?,?,?)",
            ("old::album", "lastfm_tags", "miss", None, "2000-01-01T00:00:00+00:00"),
        )

    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", db)
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: "k")
    monkeypatch.setattr(al, "discover_releases", lambda *a, **k: [_rel("old::album")])
    seen = {"n": 0}

    def fake_fetch(**k):
        seen["n"] += 1
        return []  # still empty

    monkeypatch.setattr(al, "fetch_lastfm_tags", fake_fetch)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    al.stage_lastfm(_ctx(tmp_path))
    assert seen["n"] == 1  # the stale miss WAS re-fetched

    # Its timestamp must be refreshed so it skips again for the next window.
    refreshed = store.release_keys_attempted(
        "lastfm_tags", status="miss", newer_than_iso="2020-01-01T00:00:00+00:00"
    )
    assert "old::album" in refreshed
