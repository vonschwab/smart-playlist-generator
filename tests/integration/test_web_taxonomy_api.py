"""Taxonomy term adjudication API round-trips against the fake worker.

Mirrors test_web_review_api.py one layer up: vocabulary-level queue/adjudicate/
decision (untracked) + the tracked Apply job.
"""
import time

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app
from tests.fixtures.fake_worker import FAKE_WORKER_CMD


def _client():
    return TestClient(create_app(worker_cmd=FAKE_WORKER_CMD))


def test_queue_returns_terms():
    with _client() as c:
        r = c.get("/api/taxonomy/queue")
        assert r.status_code == 200
        body = r.json()
        assert "terms" in body and "untriaged_terms" in body
        assert body["terms"][0]["term"] == "vaporwave"
        assert body["terms"][0]["album_frequency"] == 7


def test_completed_returns_terms():
    with _client() as c:
        r = c.get("/api/taxonomy/completed")
        assert r.status_code == 200
        body = r.json()
        assert "terms" in body and "decided_terms" in body
        assert body["terms"][0]["decision"]["verdict"] == "add"


def test_adjudicate_returns_verdict():
    with _client() as c:
        r = c.post("/api/taxonomy/adjudicate", json={"term": "vaporwave"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["verdict"] == "add"
        assert body["proposal"]["kind"] == "genre"


def test_adjudicate_requires_term():
    with _client() as c:
        r = c.post("/api/taxonomy/adjudicate", json={"term": "  "})
        assert r.status_code == 422


def test_decision_round_trip():
    with _client() as c:
        r = c.post("/api/taxonomy/decision",
                   json={"term": "vaporwave", "raw_term": "Vaporwave",
                         "verdict": "add", "proposal": {"name": "vaporwave"}})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["status"] == "add"


def test_apply_returns_job_and_succeeds():
    with _client() as c:
        r = c.post("/api/taxonomy/apply")
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        for _ in range(100):
            job = c.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("success", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert job["status"] == "success"
        # tracked result lands in tool_result (generic result branch)
        assert job["tool_result"]["added"] == 1
        assert job["tool_result"]["applied_terms"] == ["vaporwave"]
