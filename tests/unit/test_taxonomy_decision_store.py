import json

from src.ai_genre_enrichment.taxonomy_decision_store import (
    TaxonomyDecisionStore, list_decisions,
)


def _store(tmp_path):
    return TaxonomyDecisionStore(tmp_path / "ai_genre_enrichment.db")


def test_record_and_list_pending(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(
            term="vapor wave", raw_term="Vapor Wave", verdict="add",
            proposal_json=json.dumps({"name": "vaporwave", "kind": "genre"}),
            claude_json=json.dumps({"verdict": "add"}), human_edited=0,
        )
        pending = store.list_pending()
        assert [r["term"] for r in pending] == ["vapor wave"]
        assert pending[0]["verdict"] == "add"
        assert pending[0]["proposal"] == {"name": "vaporwave", "kind": "genre"}
        assert pending[0]["status"] == "pending"
        assert store.decided_terms() == {"vapor wave"}
    finally:
        store.close()


def test_revert_removes_from_pending(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="reject",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.revert("t")
        assert store.list_pending() == []
        assert store.get("t") is None
    finally:
        store.close()


def test_record_decision_upserts(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="add",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.record_decision(term="t", raw_term="t", verdict="reject",
                              proposal_json="{}", claude_json="{}", human_edited=1)
        rows = store.list_pending()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "reject"
        assert rows[0]["human_edited"] == 1
    finally:
        store.close()


def test_mark_applied_moves_to_applied(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="add",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.mark_applied(["t"], batch_version="0.9.0-gui-20260626-grown")
        assert store.list_pending() == []
        applied = store.list_applied()
        assert applied[0]["status"] == "applied"
        assert applied[0]["batch_version"] == "0.9.0-gui-20260626-grown"
        assert applied[0]["applied_at"] is not None
    finally:
        store.close()


def test_list_decisions_readonly_missing_table_is_empty(tmp_path):
    # mode=ro read on a fresh DB with no table must NOT raise.
    (tmp_path / "ai_genre_enrichment.db").write_bytes(b"")  # empty file
    assert list_decisions(tmp_path / "ai_genre_enrichment.db", status="pending") == []
