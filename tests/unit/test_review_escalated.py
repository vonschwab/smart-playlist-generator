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
