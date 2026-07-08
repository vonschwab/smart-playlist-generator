# tests/unit/test_adjudication_apply.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.adjudication_apply import (
    apply_adjudications,
    best_results,
    prune_orphaned_adjudications,
)
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
    assert a1 >= 1 and a2 >= 1  # a2 now provisionally materialized (still queued below)
    pending = queue.list_pending()
    assert [p["album_id"] for p in pending] == ["a2"]


def test_apply_skips_orphaned_album_without_crashing(tmp_path):
    """An adjudication whose album_id has no matching `albums` row (orphaned metadata,
    e.g. the album was deleted by scan cleanup after adjudication) must be skipped with
    a warning — not crash the whole apply stage on a NULL-artist insert."""
    conn = _meta(tmp_path)  # albums a1, a2
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    queue = EscalationQueue(side)
    rows = [
        ("a1", "std", _resp(["shoegaze"])),        # live -> materialize
        ("ghost", "std", _resp(["hauntology"])),   # orphaned album_id -> skip, don't crash
    ]
    summary = apply_adjudications(
        rows=rows, thorough_pv="tho", std_pv="std", meta_conn=conn, id2name={},
        taxonomy=load_default_layered_taxonomy(), adapter=FakeAdapter(),
        sidecar_store=store, queue=queue,
    )
    assert summary.materialized == 1        # only a1
    assert summary.skipped_orphan == 1      # ghost skipped, not materialized
    c = sqlite3.connect(side)
    null_artist = c.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE artist IS NULL OR artist=''").fetchone()[0]
    c.close()
    assert null_artist == 0  # never wrote a NULL-artist row


def test_prune_orphaned_adjudications_removes_dead_albums(tmp_path):
    """Deleting an album must cascade into the adjudications cache: a cached
    adjudication whose album_id no longer exists in `albums` is removed, live ones kept."""
    meta = _meta(tmp_path)  # albums a1, a2
    enr = sqlite3.connect(tmp_path / "enr.db")
    enr.executescript(
        "CREATE TABLE adjudications (album_id TEXT NOT NULL, prompt_version TEXT NOT NULL, "
        "status TEXT, response_json TEXT);"
        "INSERT INTO adjudications VALUES ('a1','std','complete','{}');"
        "INSERT INTO adjudications VALUES ('ghost','v1','complete','{}');"
        "INSERT INTO adjudications VALUES ('ghost','v2','complete','{}');"  # 2 rows, 1 dead album
    )
    enr.commit()
    removed = prune_orphaned_adjudications(enr, meta)
    assert removed == 1  # one dead album (regardless of its prompt_version row count)
    remaining = {r[0] for r in enr.execute("SELECT DISTINCT album_id FROM adjudications")}
    enr.close()
    assert remaining == {"a1"}  # ghost gone, live a1 kept


def test_apply_provisionally_materializes_escalated_with_genres(tmp_path):
    """An escalated album with proposed genres but no prior assignment is materialized
    PROVISIONALLY at reduced confidence (never worse than legacy) and stays queued for
    human confirmation. Bucket 1 of the 2026-06-25 un-enriched-albums audit."""
    conn = _meta(tmp_path)
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    queue = EscalationQueue(side)
    rows = [("a2", "std", _resp(["dream pop"], escalate=True, reason="sparse"))]  # conf 0.9
    summary = apply_adjudications(
        rows=rows, thorough_pv="tho", std_pv="std", meta_conn=conn, id2name={},
        taxonomy=load_default_layered_taxonomy(), adapter=FakeAdapter(),
        sidecar_store=store, queue=queue,
    )
    c = sqlite3.connect(side)
    confs = [r[0] for r in c.execute(
        "SELECT confidence FROM genre_graph_release_genre_assignments "
        "WHERE release_id='y::x' AND assignment_layer='observed_leaf'").fetchall()]
    c.close()
    assert confs, "escalated-with-genres should be provisionally materialized"
    assert all(con < 0.9 for con in confs), "provisional confidence must be reduced below 0.9"
    assert [p["album_id"] for p in queue.list_pending()] == ["a2"]  # still reviewable
    assert summary.provisional == 1


def test_apply_does_not_clobber_prior_assignment_on_escalation(tmp_path):
    """Provisional fill only happens when the release has NO prior assignment — an
    escalation must never overwrite an existing (confirmed) materialization."""
    conn = _meta(tmp_path)
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    queue = EscalationQueue(side)
    # First: non-escalated materialize for a2 (release y::x) -> dream pop.
    apply_adjudications(
        rows=[("a2", "std", _resp(["dream pop"]))], thorough_pv="tho", std_pv="std",
        meta_conn=conn, id2name={}, taxonomy=load_default_layered_taxonomy(),
        adapter=FakeAdapter(), sidecar_store=store, queue=queue)
    c = sqlite3.connect(side)
    before = c.execute("SELECT genre_id, confidence FROM genre_graph_release_genre_assignments "
                       "WHERE release_id='y::x' ORDER BY genre_id").fetchall()
    c.close()
    # Then: same release escalates with a different genre -> must NOT clobber `before`.
    apply_adjudications(
        rows=[("a2", "std", _resp(["shoegaze"], escalate=True, reason="sparse"))],
        thorough_pv="tho", std_pv="std", meta_conn=conn, id2name={},
        taxonomy=load_default_layered_taxonomy(), adapter=FakeAdapter(),
        sidecar_store=store, queue=queue)
    c = sqlite3.connect(side)
    after = c.execute("SELECT genre_id, confidence FROM genre_graph_release_genre_assignments "
                      "WHERE release_id='y::x' ORDER BY genre_id").fetchall()
    c.close()
    assert after == before  # prior preserved, provisional skipped
