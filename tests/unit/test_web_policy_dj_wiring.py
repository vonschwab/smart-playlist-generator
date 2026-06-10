"""Web layer must resolve seed artists so policy can enable DJ bridging.

Regression for the 2026-06-10 audit finding: the web app called
`derive_runtime_config(ui)` without `seed_artist_keys` and without
`seed_track_ids` on the UI state, so the policy's "Phase 1 limitation"
branch force-disabled DJ bridging on every web GUI run regardless of
config.yaml. Direct worker replays of the same request ran with dj_union;
GUI runs silently got baseline.
"""
import sys

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]

SEEDS_BODY = {
    "mode": "seeds",
    "tracks": 10,
    "seed_tracks": ["Song A", "Song B", "Song C"],
    "seed_track_ids": ["t1", "t2", "t3"],
}


def _capture_app(resolver):
    app = create_app(worker_cmd=FAKE, seed_artist_resolver=resolver)
    captured: dict = {}

    async def fake_submit(cmd: dict) -> None:
        captured.update(cmd)

    app.state.bridge.submit = fake_submit
    return app, captured


def _dj_overrides(captured: dict) -> dict:
    return (
        captured["overrides"]["playlists"]["ds_pipeline"]["pier_bridge"]["dj_bridging"]
    )


def test_dj_bridging_enabled_for_multi_artist_seeds():
    app, captured = _capture_app(lambda ids: [f"artist-{t}" for t in ids])
    with TestClient(app) as client:
        resp = client.post("/api/generate", json=SEEDS_BODY)
        assert resp.status_code == 200
    assert _dj_overrides(captured)["enabled"] is True


def test_dj_bridging_disabled_for_single_artist_seeds():
    app, captured = _capture_app(lambda ids: ["same-artist" for _ in ids])
    with TestClient(app) as client:
        resp = client.post("/api/generate", json=SEEDS_BODY)
        assert resp.status_code == 200
    assert _dj_overrides(captured)["enabled"] is False


def test_dj_union_pooling_when_genre_mode_discover():
    app, captured = _capture_app(lambda ids: [f"artist-{t}" for t in ids])
    with TestClient(app) as client:
        body = dict(SEEDS_BODY, genre_mode="discover")
        resp = client.post("/api/generate", json=body)
        assert resp.status_code == 200
    dj = _dj_overrides(captured)
    assert dj["enabled"] is True
    assert dj["pooling"]["strategy"] == "dj_union"
    assert dj["pooling"]["k_genre"] == 80


def test_artist_mode_does_not_call_resolver():
    calls: list = []

    def resolver(ids):
        calls.append(ids)
        return []

    app, captured = _capture_app(resolver)
    with TestClient(app) as client:
        resp = client.post(
            "/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 5}
        )
        assert resp.status_code == 200
    assert calls == []
    assert _dj_overrides(captured)["enabled"] is False
