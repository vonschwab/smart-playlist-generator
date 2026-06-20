"""Album-grain escalation review API round-trips against the fake worker."""
from fastapi.testclient import TestClient

from src.playlist_web.app import create_app
from tests.fixtures.fake_worker import FAKE_WORKER_CMD


def _client():
    return TestClient(create_app(worker_cmd=FAKE_WORKER_CMD))


def test_queue_returns_escalations():
    with _client() as c:
        r = c.get("/api/review/queue")
        assert r.status_code == 200
        body = r.json()
        assert "escalations" in body and "pending_albums" in body


def test_decision_round_trip():
    with _client() as c:
        r = c.post("/api/review/decision",
                   json={"album_id": "a1", "decision": "accept"})
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_publish_returns_job_id():
    with _client() as c:
        r = c.post("/api/review/publish")
        assert r.status_code == 200
        assert "job_id" in r.json()
