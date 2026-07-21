"""Setup API: status carries checks; test/{service} delegates; config writes/409s."""
from fastapi.testclient import TestClient

from src.playlist_web.app import create_app


def _client(tmp_path):
    return TestClient(create_app(config_path=str(tmp_path / "config.yaml")))


def test_status_includes_checks(tmp_path):
    r = _client(tmp_path).get("/api/setup/status")
    body = r.json()
    assert r.status_code == 200
    assert isinstance(body.get("checks"), list)
    assert all({"id", "status", "summary"} <= set(c) for c in body["checks"])


def test_test_service_not_configured_is_pass(tmp_path):
    r = _client(tmp_path).post("/api/setup/test/lastfm", json={"config": {"lastfm": {}}})
    assert r.status_code == 200
    assert r.json()["status"] == "pass"  # not configured -> pass (optional)


def test_config_writes_then_409_without_reconfigure(tmp_path):
    c = _client(tmp_path)
    music = tmp_path / "music"; music.mkdir()
    body = {"music_directory": str(music), "ai_genre_provider": "zero_touch"}
    r1 = c.post("/api/setup/config", json=body)
    assert r1.status_code == 200 and r1.json()["ok"] is True
    r2 = c.post("/api/setup/config", json=body)
    assert r2.status_code == 409
    r3 = c.post("/api/setup/config", json={**body, "reconfigure": True})
    assert r3.status_code == 200
