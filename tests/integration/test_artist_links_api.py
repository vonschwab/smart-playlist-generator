import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]

# The app resolves its DB from a fixed repo-relative path (DB_PATH in app.py);
# skip the search test rather than hard-fail where that DB isn't present
# (mirrors _requires_artifact / _req in tests/integration/test_*_integration.py).
_DB_PATH = Path("data/metadata.db")
_requires_db = pytest.mark.skipif(not _DB_PATH.exists(), reason="metadata.db required")


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


@pytest.mark.integration
@_requires_db
def test_artists_search_returns_items():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        resp = client.get("/api/artists/search", params={"q": "", "limit": 5})
        assert resp.status_code == 200
        assert isinstance(resp.json()["items"], list)
