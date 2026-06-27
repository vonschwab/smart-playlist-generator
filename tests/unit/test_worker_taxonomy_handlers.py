"""Worker-handler tests for the taxonomy term adjudication commands.

The read/apply handlers delegate to the (separately unit-tested) Phase 1-3
modules and are exercised end-to-end via the fake-worker integration test; here
we anchor the self-contained decision handler (record/revert) through the worker.
"""
from src.playlist_gui import worker
from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore


def test_record_taxonomy_decision_writes_and_emits(tmp_path, monkeypatch):
    db = tmp_path / "ai_genre_enrichment.db"
    monkeypatch.setattr(worker, "SIDECAR_DB_PATH", str(db))
    events: list[dict] = []
    monkeypatch.setattr(worker, "emit_event", lambda e: events.append(e))

    worker.handle_record_taxonomy_decision({
        "request_id": "r1", "term": "vaporwave", "raw_term": "Vaporwave",
        "verdict": "add", "proposal": {"name": "vaporwave", "kind": "genre"},
        "claude": {"verdict": "add"}, "human_edited": False})

    types = [e["type"] for e in events]
    assert "result" in types and "done" in types
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    # untracked handlers must set job_id=None explicitly (reader-thread discipline)
    assert all(e.get("job_id") is None for e in events)

    store = TaxonomyDecisionStore(db)
    try:
        pending = store.list_pending()
        assert [r["term"] for r in pending] == ["vaporwave"]
        assert pending[0]["verdict"] == "add"
        assert pending[0]["proposal"] == {"name": "vaporwave", "kind": "genre"}
    finally:
        store.close()


def test_record_taxonomy_decision_revert(tmp_path, monkeypatch):
    db = tmp_path / "ai_genre_enrichment.db"
    monkeypatch.setattr(worker, "SIDECAR_DB_PATH", str(db))
    monkeypatch.setattr(worker, "emit_event", lambda e: None)

    worker.handle_record_taxonomy_decision({
        "request_id": "r1", "term": "x", "raw_term": "x", "verdict": "reject",
        "proposal": {"reject_reason": "user_list"}, "claude": {}, "human_edited": False})
    worker.handle_record_taxonomy_decision({
        "request_id": "r2", "term": "x", "verdict": "revert"})

    store = TaxonomyDecisionStore(db)
    try:
        assert store.list_pending() == []
    finally:
        store.close()
