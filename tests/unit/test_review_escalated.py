from __future__ import annotations

from scripts.research.review_escalated import ReviewDecisionStore, parse_decision


def test_parse_decision_variants():
    assert parse_decision("accept") == ("accept", [])
    assert parse_decision("reject") == ("reject", [])
    assert parse_decision("skip") == ("skip", [])
    assert parse_decision("edit shoegaze, dream pop ,  noise pop") == (
        "edit", ["shoegaze", "dream pop", "noise pop"])


def test_decision_store_roundtrip_and_resume(tmp_path):
    s = ReviewDecisionStore(tmp_path / "d.db")
    assert s.decided_ids() == set()
    s.save("a1", "accept", [])
    s.save("a2", "edit", ["funk", "soul"])
    assert s.get("a2") == {"decision": "edit", "genres": ["funk", "soul"]}
    assert s.decided_ids() == {"a1", "a2"}


def test_review_apply_materializes_via_queue(tmp_path, monkeypatch):
    import sqlite3
    import scripts.research.review_escalated as re_mod
    from src.ai_genre_enrichment.escalation_queue import EscalationQueue
    from src.ai_genre_enrichment.storage import SidecarStore

    side = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(str(side)); store.initialize()
    q = EscalationQueue(side)
    q.enqueue(album_id="a1", release_key="slowdive::souvlaki", artist="Slowdive",
              album="Souvlaki", prior_observed_leaf=["indie rock"],
              proposed_genres=[{"term": "shoegaze", "confidence": 0.9}],
              escalate_reason="sparse", dropped_file_tags=[], prompt_version="pv",
              model="sonnet", input_hash="h1")
    # mark a decision directly, then run --apply path
    q.record_decision  # ensure attribute exists
    # Point the CLI's sidecar resolver at our temp DB.
    monkeypatch.setattr(re_mod, "_sidecar_path", lambda: str(side))

    # Decide accept, then apply.
    re_mod.apply_decisions(sidecar_path=str(side), decisions={"a1": ("accept", [])})
    n = sqlite3.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    assert n >= 1
    assert q.get("a1")["status"] == "accepted"
