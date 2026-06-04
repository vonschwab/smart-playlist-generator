# tests/integration/test_web_api.py
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def test_health_ok():
    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "worker_running" in body
