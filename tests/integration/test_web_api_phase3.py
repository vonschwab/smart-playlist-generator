import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_playlist_out_maps_transition_score_and_percentiles():
    from src.playlist_web.schemas import PlaylistOut

    raw = {
        "name": "X", "track_count": 2,
        "tracks": [
            {"position": 0, "rating_key": "k0", "artist": "A", "title": "T0",
             "album": "Al", "duration_ms": 1, "file_path": "/0", "genres": [],
             "transition_score": 0.62},
            {"position": 1, "rating_key": "k1", "artist": "B", "title": "T1",
             "album": "Al", "duration_ms": 1, "file_path": "/1", "genres": [],
             "transition_score": None},
        ],
        "metrics": {"mean_transition": 0.7, "min_transition": 0.5,
                    "p10_transition": 0.55, "p90_transition": 0.8, "distinct_artists": 2},
    }
    pl = PlaylistOut.from_worker(raw)
    assert pl.tracks[0].transition_score == 0.62
    assert pl.tracks[1].transition_score is None
    assert pl.metrics.p10_transition == 0.55
    assert pl.metrics.p90_transition == 0.8


from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def test_get_blacklist_groups_entries():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.get("/api/blacklist")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["artists"][0]["display_name"] == "Nick Drake"
        assert body["artists"][0]["scope"] == "artist"
        assert body["albums"][0]["album"] == "Pink Moon"
        assert body["albums"][0]["artist"] == "Nick Drake"
        assert body["albums"][0]["scope"] == "album"
        assert body["tracks"][0]["track_id"] == "t1"
        assert body["tracks"][0]["scope"] == "track"


def test_post_blacklist_artist_ok():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist/artist", json={"artist": "Coldplay"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


def test_generate_records_created_at_and_params():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 5})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["created_at"] is not None
        assert detail["request_params"]["artist"] == "Acetone"
        assert detail["request_params"]["tracks"] == 5


def test_cancel_unknown_job_404():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/jobs/does-not-exist/cancel")
        assert resp.status_code == 404


def test_cancel_completed_job_409():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        # Generate against the fake worker; it completes near-instantly.
        job_id = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone"}).json()["job_id"]
        # Poll until the job is no longer running.
        import time as _t
        for _ in range(50):
            status = client.get(f"/api/jobs/{job_id}").json()["status"]
            if status != "running":
                break
            _t.sleep(0.05)
        resp = client.post(f"/api/jobs/{job_id}/cancel")
        assert resp.status_code == 409
