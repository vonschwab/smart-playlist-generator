# tests/integration/test_web_api.py
import sys
import time

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_health_ok():
    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "worker_running" in body


def test_setup_status_endpoint_no_config(tmp_path):
    app = create_app(config_path=str(tmp_path / "config.yaml"))
    client = TestClient(app)
    r = client.get("/api/setup/status")
    assert r.status_code == 200
    assert r.json()["state"] == "needs_setup"


def test_generate_runs_to_success_via_fake_worker():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:  # triggers startup (bridge.start)
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        deadline = time.time() + 5
        job = None
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("success", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert job["status"] == "success"
        assert job["playlist"]["track_count"] == 2
        assert job["playlist"]["metrics"]["distinct_artists"] == 2


def test_generate_rejects_invalid_request():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "", "tracks": 5})
        assert resp.status_code == 422
        assert "artist" in resp.json()["detail"].lower()


def test_jobs_list_returns_recent():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
        time.sleep(0.3)
        jobs = client.get("/api/jobs").json()
        assert isinstance(jobs, list) and len(jobs) >= 1


def test_websocket_streams_generation_events():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 2})
            saw_done = False
            for _ in range(50):
                msg = ws.receive_json()
                if msg.get("type") == "done":
                    saw_done = True
                    break
            assert saw_done
