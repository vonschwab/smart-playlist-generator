# tests/integration/test_web_review_api.py
"""Integration tests for /api/review/* endpoints."""
import sys
import time

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def _wait_done(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("success", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    return client.get(f"/api/jobs/{job_id}").json()


def test_scan_creates_job_and_result_lands_in_tool_result():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/scan")
        assert resp.status_code == 200
        job = _wait_done(client, resp.json()["job_id"])
        assert job["status"] == "success"
        assert job["tool_result"]["result_type"] == "scan_genre_review"
        assert job["tool_result"]["pending_terms"] == 3


def test_queue_returns_releases_and_counts():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/review/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending_terms"] == 2
        assert data["releases"][0]["release_key"] == "acetone::cindy"
        assert len(data["releases"][0]["pending"]) == 2


def test_completed_endpoint_returns_decided_releases():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/review/completed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["decided_terms"] == 1
        assert data["releases"][0]["release_key"] == "acetone::cindy"
        assert data["releases"][0]["decided"][0]["status"] == "accepted"


def test_decision_round_trip():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/decision", json={
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


def test_decision_invalid_decision_is_422():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/review/decision", json={
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "maybe",
        })
        assert resp.status_code == 422


def test_queue_and_decision_work_while_a_tracked_job_is_busy():
    """A long scan/enrich holding the bridge must not block the review panel.

    Simulate an in-flight tracked job by pinning the bridge's active request id;
    the queue read and decision apply are untracked and must still succeed.
    """
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        app.state.bridge._active_request_id = "scan-in-flight"

        q = client.get("/api/review/queue")
        assert q.status_code == 200
        assert q.json()["pending_terms"] == 2

        d = client.post("/api/review/decision", json={
            "release_key": "acetone::cindy", "term": "slowcore", "decision": "accept",
        })
        assert d.status_code == 200
        assert d.json()["status"] == "accepted"

        # The tracked job's id is untouched by the untracked traffic.
        assert app.state.bridge._active_request_id == "scan-in-flight"
        app.state.bridge._active_request_id = None


# ── Graceful degradation: worker stall/death must not be a bare 500 ──────────
# (2026-06-12) The review_queue endpoint let asyncio.TimeoutError and the
# "worker not running" RuntimeError propagate uncaught -> HTTP 500 with a scary
# uvicorn traceback and no signal. They must map to clean 5xx with a message.
from src.playlist_web.worker_bridge import WorkerTimeout, WorkerUnavailable


def test_review_queue_worker_timeout_is_504_not_500(monkeypatch):
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        async def _timeout(*a, **k):
            raise WorkerTimeout("Worker did not respond within 60s")
        monkeypatch.setattr(app.state.bridge, "command", _timeout)
        resp = client.get("/api/review/queue")
        assert resp.status_code == 504
        assert "detail" in resp.json()


def test_review_queue_worker_unavailable_is_503(monkeypatch):
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        async def _down(*a, **k):
            raise WorkerUnavailable("Worker not running")
        monkeypatch.setattr(app.state.bridge, "command", _down)
        resp = client.get("/api/review/queue")
        assert resp.status_code == 503
        assert "detail" in resp.json()
