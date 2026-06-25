"""API-layer tests for genre editing using the fake worker + TestClient.

TestClient faithfully drives the FAKE worker stub (the Windows real-worker
trap only applies to the real subprocess — see the web-gui skill)."""
from fastapi.testclient import TestClient

from src.playlist_web.app import create_app
from tests.fixtures.fake_worker import FAKE_WORKER_CMD as FAKE


def test_edit_genres_passes_through_resolved():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        r = client.post("/api/edit_genres", json={
            "artist": "The Radio Dept.", "album": "Pet Grief",
            "genres": ["dream pop"], "base_genres": []})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["resolved"] == ["dream pop"]
        assert body["no_change"] is False
        assert body["unknown"] == []


def test_refresh_genre_artifact_returns_job():
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        r = client.post("/api/refresh_genre_artifact")
        assert r.status_code == 200
        assert "job_id" in r.json()


def test_genres_search_endpoint_shape():
    """The endpoint returns a JSON object with an items list (empty if no DB)."""
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as client:
        r = client.get("/api/genres/search", params={"q": "dream"})
        assert r.status_code == 200
        assert "items" in r.json()
