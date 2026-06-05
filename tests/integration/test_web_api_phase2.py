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


def test_audio_full_request(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/k0")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"
            assert resp.headers["accept-ranges"] == "bytes"
            assert len(resp.content) == 1003


def test_audio_range_request(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/k0", headers={"Range": "bytes=0-99"})
            assert resp.status_code == 206
            assert resp.headers["content-range"] == "bytes 0-99/1003"
            assert len(resp.content) == 100


def test_audio_unknown_track_404(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _make_audio_db(tmp)
        import src.playlist_web.app as appmod
        monkeypatch.setattr(appmod, "DB_PATH", db)
        with TestClient(create_app(worker_cmd=FAKE)) as client:
            resp = client.get("/api/audio/nope")
            assert resp.status_code == 404
