# tests/unit/test_analyze_adjudicate_stages.py
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import scripts.analyze_library as al
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.storage import SidecarStore


class FakeClient:
    def __init__(self, responses):
        self._responses = responses
    def call_structured_session(self, items, *, response_format, validator, instructions,
                                on_result, reset_every):
        for album_id, _ in items:
            on_result(album_id, self._responses[album_id], None, {"total_tokens": 5})


def _metadata_db(tmp_path: Path) -> str:
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO albums VALUES ('a2','X','Y',2000,NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _resp(genres, escalate=False):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "escalate_reason": "sparse",
            "dropped_file_tags": [], "overall_confidence": 0.9}


def test_adjudicate_then_apply_materializes_and_queues(tmp_path, monkeypatch):
    db = _metadata_db(tmp_path)
    side = tmp_path / "ai_genre_enrichment.db"
    SidecarStore(str(side)).initialize()
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", side)

    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["dream pop"], escalate=True)})
    args = Namespace(adjudicate_model="sonnet", adjudicate_client=client, limit=None)
    ctx = {"args": args, "db_path": db}

    out_adj = al.stage_adjudicate(ctx)
    assert out_adj["adjudicated"] == 2 and not out_adj.get("paused")

    out_apply = al.stage_apply(ctx)
    assert out_apply["materialized"] == 1
    assert out_apply["escalated"] == 1

    q = EscalationQueue(side)
    assert [p["album_id"] for p in q.list_pending()] == ["a2"]
