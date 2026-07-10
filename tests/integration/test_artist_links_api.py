import sys
import pytest
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.mark.integration
def test_list_artist_links_returns_groups():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/artists/links")
        assert resp.status_code == 200
        assert resp.json()["groups"] == [{"type": "sibling", "members": ["Smog", "Bill Callahan"]}]


@pytest.mark.integration
def test_save_artist_links_ok():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/artists/links/save", json={
            "groups": [{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}]})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["count"] == 1


@pytest.mark.integration
def test_save_artist_links_rejects_invalid_with_422():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.post("/api/artists/links/save", json={
            "groups": [{"type": "alias", "members": ["OnlyOne"]}]})
        assert resp.status_code == 422
