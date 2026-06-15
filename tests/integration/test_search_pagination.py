import sqlite3
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def _make_db(tmp: Path) -> Path:
    db = tmp / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE artists (artist_name TEXT PRIMARY KEY);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
                             album TEXT, duration_ms INTEGER, file_path TEXT);
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT, source TEXT,
                                             priority INTEGER, weight REAL);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        """
    )
    for name in ["Beach House", "Beck", "Beirut", "Bell Orchestre", "Belle & Sebastian"]:
        conn.execute("INSERT INTO artists VALUES (?)", (name,))
    for i in range(5):
        conn.execute(
            "INSERT INTO tracks VALUES (?,?,?,?,?,?)",
            (f"t{i}", f"Song {i}", "Beach House", "Bloom", 200000, f"/m/{i}.flac"),
        )
    conn.execute("INSERT INTO track_effective_genres VALUES ('t0','dream pop','x',1,1.0)")
    conn.execute("INSERT INTO track_effective_genres VALUES ('t0','shoegaze','x',2,1.0)")
    conn.execute("INSERT INTO track_genres VALUES ('t1','indie','x',0.9)")
    conn.commit()
    conn.close()
    return db


def _client(monkeypatch, db: Path) -> TestClient:
    import src.playlist_web.app as appmod
    monkeypatch.setattr(appmod, "DB_PATH", db)
    return TestClient(create_app(worker_cmd=FAKE))


def test_track_search_paginates(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            r0 = client.get("/api/tracks/search", params={"q": "beach", "offset": 0, "limit": 3}).json()
            assert [it["track_id"] for it in r0["items"]] == ["t0", "t1", "t2"]
            assert r0["has_more"] is True
            r1 = client.get("/api/tracks/search", params={"q": "beach", "offset": 3, "limit": 3}).json()
            assert [it["track_id"] for it in r1["items"]] == ["t3", "t4"]
            assert r1["has_more"] is False


def test_track_search_batches_genres_with_fallback(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            items = client.get("/api/tracks/search", params={"q": "beach", "limit": 25}).json()["items"]
            by_id = {it["track_id"]: it["genres"] for it in items}
            assert by_id["t0"] == ["dream pop", "shoegaze"]
            assert by_id["t1"] == ["indie"]
            assert by_id["t2"] == []


def test_track_search_empty_query(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            assert client.get("/api/tracks/search", params={"q": ""}).json() == {"items": [], "has_more": False}


import pytest


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 300}, {"offset": -1}])
def test_track_search_rejects_out_of_range_params(monkeypatch, params):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            resp = client.get("/api/tracks/search", params={"q": "beach", **params})
            assert resp.status_code == 422


def test_autocomplete_paginates(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            r0 = client.get("/api/autocomplete", params={"q": "be", "offset": 0, "limit": 2}).json()
            assert r0["items"] == ["Beach House", "Beck"]  # alphabetical
            assert r0["has_more"] is True
            r2 = client.get("/api/autocomplete", params={"q": "be", "offset": 4, "limit": 2}).json()
            assert r2["items"] == ["Belle & Sebastian"]
            assert r2["has_more"] is False


def test_autocomplete_empty_query(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            assert client.get("/api/autocomplete", params={"q": ""}).json() == {"items": [], "has_more": False}


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 300}, {"offset": -1}])
def test_autocomplete_rejects_out_of_range_params(monkeypatch, params):
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(Path(d))
        with _client(monkeypatch, db) as client:
            resp = client.get("/api/autocomplete", params={"q": "be", **params})
            assert resp.status_code == 422
