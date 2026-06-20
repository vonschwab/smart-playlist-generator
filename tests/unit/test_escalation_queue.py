from __future__ import annotations

from src.ai_genre_enrichment.escalation_queue import EscalationQueue


def _enq(q, album_id="a1", proposed=("shoegaze",), input_hash="h1"):
    q.enqueue(
        album_id=album_id, release_key="slowdive::souvlaki", artist="Slowdive",
        album="Souvlaki", prior_observed_leaf=["indie rock"],
        proposed_genres=[{"term": g, "confidence": 0.9} for g in proposed],
        escalate_reason="sparse", dropped_file_tags=[], prompt_version="pv",
        model="sonnet", input_hash=input_hash,
    )


def test_enqueue_and_list_pending(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q)
    pending = q.list_pending()
    assert len(pending) == 1
    row = pending[0]
    assert row["album_id"] == "a1"
    assert row["prior_observed_leaf"] == ["indie rock"]
    assert row["proposed_genres"][0]["term"] == "shoegaze"
    assert row["status"] == "pending"


def test_decided_row_not_reopened_when_proposal_unchanged(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, input_hash="h1")
    q._mark(  # test helper: force a decided state without materializing
        "a1", status="rejected", decision_genres=None)
    _enq(q, input_hash="h1")  # same proposal
    assert q.get("a1")["status"] == "rejected"
    assert q.list_pending() == []


def test_changed_proposal_reopens_decided_row(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, input_hash="h1")
    q._mark("a1", status="rejected", decision_genres=None)
    _enq(q, proposed=("dream pop",), input_hash="h2")  # changed proposal
    assert q.get("a1")["status"] == "pending"


import sqlite3 as _sqlite3

from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


def test_record_decision_accept_materializes_and_marks(tmp_path):
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    q = EscalationQueue(side)
    _enq(q, proposed=("shoegaze",))
    taxonomy = load_default_layered_taxonomy()
    q.record_decision("a1", "accept", sidecar_store=store, taxonomy=taxonomy)
    assert q.get("a1")["status"] == "accepted"
    # an observed_leaf row was written for the release
    conn = _sqlite3.connect(side)
    n = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    conn.close()
    assert n >= 1


def test_record_decision_reject_does_not_materialize(tmp_path):
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    q = EscalationQueue(side)
    _enq(q)
    taxonomy = load_default_layered_taxonomy()
    q.record_decision("a1", "reject", sidecar_store=store, taxonomy=taxonomy)
    assert q.get("a1")["status"] == "rejected"
    conn = _sqlite3.connect(side)
    n = conn.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments").fetchone()[0]
    conn.close()
    assert n == 0
