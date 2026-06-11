# tests/integration/test_web_tools_api.py
"""Integration tests for /api/tools/analyze and /api/tools/enrich endpoints."""
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


def test_analyze_creates_job_and_succeeds():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/tools/analyze", json={})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        job = _wait_done(client, job_id)
        assert job["status"] == "success"
        assert job["tool_result"]["result_type"] == "analyze_library"
        stages = job["tool_result"]["stages"]
        assert isinstance(stages, list) and len(stages) == 11


def test_analyze_with_stage_subset():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/tools/analyze",
                           json={"stages": ["lastfm", "enrich", "publish"]})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        job = _wait_done(client, job_id)
        assert job["status"] == "success"
        stages = job["tool_result"]["stages"]
        assert len(stages) == 3


def test_enrich_creates_job_and_succeeds():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/tools/enrich", json={"scope": "all_unenriched"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        job = _wait_done(client, job_id)
        assert job["status"] == "success"
        assert job["tool_result"]["result_type"] == "enrich_genres"
        assert job["tool_result"]["releases"] == 5


def test_analyze_job_appears_in_jobs_list():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/tools/analyze", json={"dry_run": True})
        job_id = resp.json()["job_id"]
        _wait_done(client, job_id)
        jobs = client.get("/api/jobs").json()
        ids = [j["job_id"] for j in jobs]
        assert job_id in ids
