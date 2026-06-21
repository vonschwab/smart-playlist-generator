# tests/unit/test_adjudication_apply.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.adjudication_apply import apply_adjudications, best_results
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


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
        INSERT INTO albums VALUES ('a2','X','Y',2000,NULL);
        """
    )
    conn.commit()
    return conn


def _resp(genres, escalate=False, reason=""):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "escalate_reason": reason,
            "dropped_file_tags": [], "overall_confidence": 0.9}


def test_best_results_prefers_thorough():
    rows = [("a1", "std", _resp(["x"])), ("a1", "tho", _resp(["x", "y"]))]
    best = best_results(rows, thorough_pv="tho")
    assert len(best["a1"]["genres"]) == 2


def test_apply_materializes_nonescalated_and_enqueues_escalated(tmp_path):
    conn = _meta(tmp_path)
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    queue = EscalationQueue(side)
    rows = [
        ("a1", "std", _resp(["shoegaze"])),                       # non-escalated -> materialize
        ("a2", "std", _resp(["dream pop"], escalate=True, reason="sparse")),  # -> queue
    ]
    summary = apply_adjudications(
        rows=rows, thorough_pv="tho", std_pv="std", meta_conn=conn, id2name={},
        taxonomy=load_default_layered_taxonomy(), adapter=FakeAdapter(),
        sidecar_store=store, queue=queue,
    )
    assert summary.materialized == 1
    assert summary.escalated == 1
    # a1 materialized, a2 NOT materialized
    c = sqlite3.connect(side)
    a1 = c.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
                   "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    a2 = c.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
                   "WHERE release_id='y::x'").fetchone()[0]
    c.close()
    assert a1 >= 1 and a2 == 0
    pending = queue.list_pending()
    assert [p["album_id"] for p in pending] == ["a2"]
