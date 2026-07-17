import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_phase2_schemas_importable():
    from src.playlist_web.schemas import (
        CandidateOut,
        ReplaceSuggestionsRequest,
        ReplaceSuggestionsResponse,
        BlacklistRequest,
        EditGenresRequest,
        PlexExportRequest,
    )

    cand = CandidateOut(track_id="k1", title="T", artist="A", album="Al", genres=["x"], fit_score=0.7)
    assert cand.fit_score == 0.7

    bl = BlacklistRequest(scope="album", value="Leisure", artist="Marbled Eye")
    assert bl.artist == "Marbled Eye"

    req = ReplaceSuggestionsRequest(position=5)
    assert req.position == 5
    assert req.top_k == 10

    eg = EditGenresRequest(artist="Acetone", album="York Blvd", genres=["slowcore"])
    assert eg.genres == ["slowcore"]

    pe = PlexExportRequest(title="My Mix")
    assert pe.title == "My Mix"

    cands = ReplaceSuggestionsResponse.from_worker_candidates(
        position=3,
        raw=[{"rating_key": "k9", "title": "Song", "artist": "Band", "album": "LP",
              "genres": ["slowcore"], "mean_t": 0.66}],
    )
    assert cands.candidates[0].track_id == "k9"
    assert cands.candidates[0].fit_score == 0.66


import tempfile
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def _make_audio_db(tmp: Path) -> Path:
    """Create a tiny metadata.db with one track pointing at a real bytes file."""
    audio = tmp / "song.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 1000)  # 1003 bytes of fake audio
    db = tmp / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, file_path TEXT)")
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("k0", str(audio)))
    conn.commit()
    conn.close()
    return db


def _config_for_db(tmp: Path, db: Path) -> str:
    # create_app() resolves DB_PATH from config (2026-07-16 fix), so point a
    # temp config at the temp DB rather than monkeypatching the module
    # constant directly — create_app() rebinds it from config_path on every
    # call and would otherwise clobber a pre-call monkeypatch.
    cfg = tmp / "config.yaml"
    cfg.write_text(f"library:\n  database_path: {db.as_posix()}\n", encoding="utf-8")
    return str(cfg)


def test_audio_full_request():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        with TestClient(create_app(worker_cmd=FAKE, config_path=_config_for_db(tmp, db))) as client:
            resp = client.get("/api/audio/k0")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"
            assert resp.headers["accept-ranges"] == "bytes"
            assert len(resp.content) == 1003


def test_audio_range_request():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        with TestClient(create_app(worker_cmd=FAKE, config_path=_config_for_db(tmp, db))) as client:
            resp = client.get("/api/audio/k0", headers={"Range": "bytes=0-99"})
            assert resp.status_code == 206
            assert resp.headers["content-range"] == "bytes 0-99/1003"
            assert len(resp.content) == 100


def test_audio_unknown_track_404():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        with TestClient(create_app(worker_cmd=FAKE, config_path=_config_for_db(tmp, db))) as client:
            resp = client.get("/api/audio/nope")
            assert resp.status_code == 404


def test_replace_suggestions_returns_candidates():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/replace_suggestions", json={"job_id": "j1", "position": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["position"] == 3
        assert body["candidates"][0]["track_id"] == "k9"
        assert body["candidates"][0]["fit_score"] == 0.74


def test_replace_suggestions_pier_rejected():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/replace_suggestions", json={"job_id": "j1", "position": 0})
        assert resp.status_code == 422


def test_blacklist_track():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"track_ids": ["k0", "k1"], "enabled": True})
        assert resp.status_code == 200
        assert resp.json()["updated"] == 2


def test_blacklist_album_scope():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"scope": "album", "value": "Leisure", "artist": "Marbled Eye"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


def test_blacklist_album_scope_requires_artist():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist", json={"scope": "album", "value": "Leisure"})
        assert resp.status_code == 422


def test_edit_genres():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/edit_genres", json={
            "artist": "Marbled Eye", "album": "Leisure", "genres": ["post-punk", "dream pop"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "post-punk" in body["resolved"]


def test_edit_genres_requires_artist_album():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/edit_genres", json={"artist": "", "album": "", "genres": ["x"]})
        assert resp.status_code == 422


def test_plex_export_unconfigured_returns_503():
    # Pass a config path with no plex section so the result is deterministic
    # regardless of the developer's real config.yaml.
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "config.yaml"
        cfg.write_text("library:\n  database_path: data/metadata.db\n", encoding="utf-8")
        with TestClient(create_app(worker_cmd=FAKE, config_path=str(cfg))) as client:
            resp = client.post("/api/export/plex", json={
                "title": "My Playlist",
                "tracks": [{"rating_key": "k0", "title": "Sundown", "artist": "Acetone", "file_path": "/0.flac"}],
            })
            assert resp.status_code == 503
            assert "plex" in resp.json()["detail"].lower()
