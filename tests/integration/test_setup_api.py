"""Setup API: status carries checks; test/{service} delegates; config writes/409s."""
import sqlite3

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app


def _client(tmp_path):
    return TestClient(create_app(config_path=str(tmp_path / "config.yaml")))


def _make_ready_home(tmp_path):
    """Mirrors tests/unit/test_setup_state.py::test_ready — a fully configured,
    already-analyzed home (state == "ready")."""
    music = tmp_path / "music"
    music.mkdir()
    dbdir = tmp_path / "data"
    dbdir.mkdir()
    db = dbdir / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO tracks VALUES ('t1')")
    conn.commit()
    conn.close()
    (tmp_path / "config.yaml").write_text(
        f"library:\n  music_directory: {music.as_posix()}\n  database_path: data/metadata.db\n",
        encoding="utf-8")


def test_status_includes_checks(tmp_path):
    r = _client(tmp_path).get("/api/setup/status")
    body = r.json()
    assert r.status_code == 200
    assert body["state"] == "needs_setup"
    assert isinstance(body.get("checks"), list)
    assert len(body["checks"]) > 0
    assert all({"id", "status", "summary"} <= set(c) for c in body["checks"])


def test_ready_status_has_empty_checks(tmp_path):
    """I1: an already-configured/analyzed user must not pay for the full
    doctor-style health sweep (incl. a metadata.db integrity_check) on every
    app load — checks are only ever rendered by the needs_setup Environment
    step."""
    _make_ready_home(tmp_path)
    r = _client(tmp_path).get("/api/setup/status")
    body = r.json()
    assert r.status_code == 200
    assert body["state"] == "ready"
    assert body["checks"] == []


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


def test_config_with_empty_music_directory_is_400(tmp_path):
    """C1: the server is the authoritative guard against an empty
    music_directory reaching disk, independent of any client-side guard."""
    r = _client(tmp_path).post("/api/setup/config", json={"music_directory": ""})
    assert r.status_code == 400
    assert "music_directory" in r.json()["detail"]


def test_config_with_missing_music_directory_field_is_422(tmp_path):
    """Pydantic's own required-field validation still applies when the key is
    absent entirely (as opposed to present-but-empty, covered above)."""
    r = _client(tmp_path).post("/api/setup/config", json={})
    assert r.status_code == 422
