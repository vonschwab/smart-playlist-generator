# tests/unit/test_worker_review_queue.py
"""Worker handler round-trips for the genre review queue commands."""
import json

from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import (
    _worker_state,
    handle_apply_genre_review_decision,
    handle_get_genre_review_queue,
)


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


def _seed(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    store.sync_review_queue_for_release(
        release_key="acetone::cindy", normalized_artist="acetone",
        normalized_album="cindy",
        terms=[{"term": "slowcore", "confidence": 0.4, "basis": "hybrid_fusion",
                "sources": ["lastfm_tags"], "reason": "uncertain"}],
    )
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))
    return store


def test_get_queue_emits_result_and_done(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_get_genre_review_queue({"cmd": "get_genre_review_queue", "request_id": "r1"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "genre_review_queue"
    assert result["pending_terms"] == 1
    assert result["releases"][0]["release_key"] == "acetone::cindy"
    assert done["ok"] is True


def test_apply_decision_emits_result_and_updates_db(tmp_path, monkeypatch, capsys):
    store = _seed(tmp_path, monkeypatch)
    handle_apply_genre_review_decision({
        "cmd": "apply_genre_review_decision", "request_id": "r2",
        "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
    })
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "genre_review_decision"
    assert result["status"] == "accepted"
    assert done["ok"] is True
    assert store.get_user_override("acetone::cindy")["genres_add"] == ["slowcore"]


def test_apply_decision_bad_input_emits_failed_done(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_apply_genre_review_decision({
        "cmd": "apply_genre_review_decision", "request_id": "r3",
        "release_key": "acetone::cindy", "term": "slowcore", "decision": "maybe",
    })
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is False


def test_inline_handlers_do_not_inherit_active_job_id(tmp_path, monkeypatch, capsys):
    """When these untracked handlers run inline while a scan job is active,
    their events must NOT carry the scan's job_id — otherwise JobRegistry would
    overwrite the scan job's tool_result and mark it done early.
    """
    _seed(tmp_path, monkeypatch)
    # Simulate a scan running on the worker thread: _worker_state holds its id.
    _worker_state.start_request("scan-req", "scan_genre_review", "scan-job-123")
    try:
        handle_get_genre_review_queue({"cmd": "get_genre_review_queue", "request_id": "q1"})
        handle_apply_genre_review_decision({
            "cmd": "apply_genre_review_decision", "request_id": "q2",
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
        })
        events = _events(capsys)
    finally:
        _worker_state.end_request()

    # Every emitted event addresses the quick command's own request_id and
    # explicitly nulls job_id — never the scan's job_id.
    assert events, "handlers emitted no events"
    assert {e.get("request_id") for e in events} <= {"q1", "q2"}
    for e in events:
        assert e.get("job_id") is None, f"leaked job_id on {e}"
