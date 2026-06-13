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


def test_get_completed_emits_result_and_done(tmp_path, monkeypatch, capsys):
    from src.playlist_gui.worker import handle_get_genre_review_completed

    store = _seed(tmp_path, monkeypatch)
    store.set_review_queue_status(
        release_key="acetone::cindy", term="slowcore", status="accepted"
    )
    handle_get_genre_review_completed(
        {"cmd": "get_genre_review_completed", "request_id": "rc"}
    )
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "genre_review_completed"
    assert result["decided_terms"] == 1
    assert result["releases"][0]["release_key"] == "acetone::cindy"
    assert result["releases"][0]["decided"][0]["status"] == "accepted"
    assert done["ok"] is True


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


# ── Review-during-scan regression (2026-06-12 timeout incident) ──────────────
# The review panel polls get_genre_review_queue inline on the worker's READER
# thread while a long tracked job (scan/enrich) writes the sidecar. With a
# delete-journal DB + a write-capable connection doing DDL (initialize), the
# inline read wedged the reader for busy_timeout=30s per poll: every untracked
# command behind it (including cancel) starved and the bridge timed out (500).

import threading


def test_sidecar_connect_enables_wal(tmp_path):
    """WAL is the enabling condition for 'review reads work during scan writes'."""
    store = SidecarStore(tmp_path / "side.db")
    store.initialize()
    conn = store.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_get_queue_succeeds_while_writer_holds_lock(tmp_path, monkeypatch, capsys):
    """A held write transaction must not block or fail the inline queue read."""
    store = _seed(tmp_path, monkeypatch)
    writer = store.connect()
    # Simulate the scan mid-commit: EXCLUSIVE blocks readers in delete-journal
    # mode (WAL readers are unaffected — that's the point of the fix).
    writer.execute("BEGIN EXCLUSIVE")
    try:
        t = threading.Thread(
            target=handle_get_genre_review_queue,
            args=({"cmd": "get_genre_review_queue", "request_id": "r9"},),
            daemon=True,
        )
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "inline queue read blocked behind the scan's write lock"
        events = _events(capsys)
        done = next(e for e in events if e["type"] == "done")
        assert done["ok"] is True
        result = next(e for e in events if e["type"] == "result")
        assert result["pending_terms"] == 1
    finally:
        writer.rollback()
        writer.close()


def test_get_queue_missing_db_returns_empty_page(tmp_path, monkeypatch, capsys):
    """Fresh install (no sidecar yet): empty page, ok=True — never an error."""
    monkeypatch.setattr(
        "src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp_path / "nope.db")
    )
    handle_get_genre_review_queue({"cmd": "get_genre_review_queue", "request_id": "r2"})
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    result = next(e for e in events if e["type"] == "result")
    assert result["pending_terms"] == 0
    assert result["releases"] == []


def test_untracked_dispatch_emits_done_on_handler_crash(monkeypatch, capsys):
    """A crashing untracked handler must still produce done(ok=false) — a missing
    done leaves the bridge future waiting until TimeoutError (500)."""
    import src.playlist_gui.worker as worker

    def _boom(cmd_data):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(worker.UNTRACKED_COMMAND_HANDLERS, "get_genre_review_queue", _boom)
    worker.process_command(
        json.dumps({"cmd": "get_genre_review_queue", "request_id": "r3"})
    )
    events = _events(capsys)
    done = next((e for e in events if e["type"] == "done"), None)
    assert done is not None, "no done event emitted for crashed untracked handler"
    assert done["ok"] is False
    assert done["request_id"] == "r3"
