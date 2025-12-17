import sqlite3
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import CONFIG_PATH, app
from api.main import _sanitize_filename

client = TestClient(app)


def test_search_tracks_basic():
    resp = client.get("/api/search/tracks", params={"q": "a"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_artists_basic():
    resp = client.get("/api/search/artists", params={"q": "a"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def _sample_artist(min_tracks: int = 1) -> str:
    conn = sqlite3.connect("data/metadata.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT artist, COUNT(*) AS c FROM tracks GROUP BY artist HAVING c >= ? LIMIT 1",
        (min_tracks,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise AssertionError("No artist found in test database")
    return row[0]


def test_resolve_artist_seed_fallback():
    artist = _sample_artist()
    with patch("api.main.fetch_artist_top_tracks", return_value=[]):
        resp = client.get("/api/seed/artist", params={"artist_name": artist, "random_seed": 123})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("primary_seed_track")


def test_generate_artist_seed_deterministic():
    artist = _sample_artist()
    payload = {
        "seed": {"type": "artist", "artist_name": artist},
        "mode": "narrow",
        "length": 5,
        "random_seed": 42,
    }
    with patch("api.main.fetch_artist_top_tracks", return_value=[]):
        resp1 = client.post("/api/playlist/generate", json=payload)
        resp2 = client.post("/api/playlist/generate", json=payload)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    t1 = [t["track_id"] for t in resp1.json().get("tracks", [])]
    t2 = [t["track_id"] for t in resp2.json().get("tracks", [])]
    assert t1 == t2


def test_export_m3u_writes_file(tmp_path):
    conn = sqlite3.connect("data/metadata.db")
    cur = conn.cursor()
    cur.execute("SELECT track_id FROM tracks WHERE file_path IS NOT NULL LIMIT 2")
    rows = cur.fetchall()
    conn.close()
    assert rows
    track_ids = [r[0] for r in rows]

    payload = {
        "track_ids": track_ids,
        "output_dir": str(tmp_path),
        "filename": "Auto - Test.m3u",
    }
    resp = client.post("/api/playlist/export", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"]
    assert data["path"]
    path = Path(data["path"])
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert content.strip() != ""


def test_export_rejects_bad_filename():
    with patch("api.main._resolve_paths", return_value=["/tmp/a.mp3"]):
        resp = client.post(
            "/api/playlist/export",
            json={"track_ids": ["x"], "output_dir": "E:\\\\TEST", "filename": "../bad.m3u"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "Invalid filename" in data["error"]


def test_get_settings_masks_api_key():
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lastfm"]["api_key"] == "****" or data["lastfm"]["api_key"] == ""


def test_put_settings_persists_and_roundtrips(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "library:\n  database_path: data/metadata.db\nopenai:\n  api_key: test\nlastfm:\n  api_key: original\n  username: original_user\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PLAYLIST_CONFIG_PATH", str(cfg_path))
    from importlib import reload
    import api.main as main_mod

    reload(main_mod)
    client_local = TestClient(main_mod.app)
    payload = {
        "export": {"default_output_dir": "E:\\\\PLAY"},
        "lastfm": {"username": "new_user", "api_key": "****"},
        "generation": {
            "default_mode": "dynamic",
            "default_length": 40,
            "deterministic_by_default": False,
            "additional_seed_count": 3,
        },
        "advanced": {},
    }
    put_resp = client_local.put("/api/settings", json=payload)
    assert put_resp.status_code == 200
    got = client_local.get("/api/settings").json()
    assert got["export"]["default_output_dir"] == "E:\\\\PLAY"
    assert got["generation"]["default_length"] == 40
    assert got["lastfm"]["username"] == "new_user"
    # api_key should remain masked but preserved
    assert got["lastfm"]["api_key"] == "****"


def test_put_settings_invalid_rejected():
    payload = {
        "export": {"default_output_dir": "E:\\\\PLAY"},
        "lastfm": {"username": "u", "api_key": ""},
        "generation": {
            "default_mode": "bad",
            "default_length": 2,
            "deterministic_by_default": True,
            "additional_seed_count": 10,
        },
        "advanced": {},
    }
    resp = client.put("/api/settings", json=payload)
    assert resp.status_code in (400, 422)
