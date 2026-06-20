# tests/unit/test_adjudication_runner.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.adjudication_runner import build_todo, run_adjudication
from src.ai_genre_enrichment.adjudication_store import AdjudicationStore


class FakeClient:
    """Mimics ClaudeCodeEnrichmentClient.call_structured_session for tests."""
    def __init__(self, responses):
        self._responses = responses  # album_id -> parsed dict (or Exception)

    def call_structured_session(self, items, *, response_format, validator, instructions,
                                on_result, reset_every):
        for album_id, _prompt in items:
            r = self._responses.get(album_id)
            if isinstance(r, Exception):
                on_result(album_id, None, str(r), {})
            else:
                on_result(album_id, r, None, {"total_tokens": 10})


class FakeAdapter:
    def canonicalize_tag(self, tag):
        return tag
    def node(self, name):
        return None


def _meta(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO albums VALUES ('a2','Nowhere','Ride',1990,NULL);
        """
    )
    conn.commit()
    return conn


def _resp(genres, escalate=False):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "overall_confidence": 0.9}


def test_run_adjudication_checkpoints_each_album(tmp_path):
    conn = _meta(tmp_path)
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    assert len(todo) == 2
    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["shoegaze", "dream pop"])})
    summary = run_adjudication(store, todo, model="sonnet", instructions="x",
                               prompt_version="pv", adapter=FakeAdapter(), client=client)
    assert summary.adjudicated == 2
    assert summary.paused is False
    assert store.complete_album_ids("pv") == {"a1", "a2"}


def test_run_adjudication_skips_already_done(tmp_path):
    conn = _meta(tmp_path)
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["shoegaze"])})
    run_adjudication(store, todo, model="sonnet", instructions="x",
                     prompt_version="pv", adapter=FakeAdapter(), client=client)
    todo2 = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    assert todo2 == []  # both already complete -> nothing to do


def test_run_adjudication_pauses_after_fail_streak(tmp_path):
    conn = _meta(tmp_path)
    # 8 albums that all fail -> pause (use b-prefix to avoid PK collision with a1/a2 in _meta)
    ids = [f"b{i}" for i in range(8)]
    conn.executemany("INSERT INTO albums VALUES (?,?,?,?,?)",
                     [(i, f"T{i}", f"Art{i}", 2000, None) for i in ids])
    conn.commit()
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ids, prompt_version="pv")
    client = FakeClient({i: RuntimeError("rate limit") for i in ids})
    summary = run_adjudication(store, todo, model="sonnet", instructions="x",
                               prompt_version="pv", adapter=FakeAdapter(), client=client)
    assert summary.paused is True
    assert summary.failed >= 8
