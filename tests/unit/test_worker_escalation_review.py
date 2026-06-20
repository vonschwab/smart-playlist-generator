# tests/unit/test_worker_escalation_review.py
"""Worker handler round-trips for the album-grain escalation review commands."""
import json

from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import (
    handle_apply_escalation_decision,
    handle_get_escalation_queue,
)


def _events(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]


def _seed(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar)); store.initialize()
    q = EscalationQueue(sidecar)
    q.enqueue(album_id="a1", release_key="slowdive::souvlaki", artist="Slowdive",
              album="Souvlaki", prior_observed_leaf=["indie rock"],
              proposed_genres=[{"term": "shoegaze", "confidence": 0.9}],
              escalate_reason="sparse", dropped_file_tags=["dream pop"],
              prompt_version="pv", model="sonnet", input_hash="h1")
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))
    return sidecar


def test_get_escalation_queue_emits_result(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    handle_get_escalation_queue({"cmd": "get_escalation_queue", "request_id": "r1"})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "escalation_queue"
    assert result["pending_albums"] == 1
    assert result["escalations"][0]["album_id"] == "a1"
    assert result["escalations"][0]["dropped_file_tags"] == ["dream pop"]
    assert result["job_id"] is None
    assert done["ok"] is True


def test_apply_escalation_decision_accept_materializes(tmp_path, monkeypatch, capsys):
    sidecar = _seed(tmp_path, monkeypatch)
    handle_apply_escalation_decision({
        "cmd": "apply_escalation_decision", "request_id": "r2",
        "album_id": "a1", "decision": "accept",
    })
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is True
    import sqlite3
    n = sqlite3.connect(sidecar).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'").fetchone()[0]
    assert n >= 1
    assert EscalationQueue(sidecar).get("a1")["status"] == "accepted"
