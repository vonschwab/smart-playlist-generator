"""Folder browse: lists subdirs + audio counts; bad path 400; permission-denied graceful."""
from src.setup.browse import list_directory


def test_lists_subdirs_and_counts_audio(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.mp3").write_bytes(b"x")
    (tmp_path / "sub" / "b.flac").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    res = list_directory(str(tmp_path))
    assert res["path"] == str(tmp_path.resolve())
    names = {e["name"]: e for e in res["entries"]}
    assert "sub" in names
    assert names["sub"]["audio_count"] == 2  # only audio counted
    assert res["is_music_dir"] is False  # tmp_path itself has no audio


def test_is_music_dir_true_when_folder_has_audio(tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    assert list_directory(str(tmp_path))["is_music_dir"] is True


def test_parent_is_reported(tmp_path):
    (tmp_path / "child").mkdir()
    res = list_directory(str(tmp_path / "child"))
    assert res["parent"] == str(tmp_path.resolve())


def test_bad_path_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        list_directory(str(tmp_path / "does-not-exist"))


def test_browse_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from src.playlist_web.app import create_app
    (tmp_path / "music").mkdir()
    app = create_app(config_path=str(tmp_path / "config.yaml"))
    client = TestClient(app)
    r = client.get("/api/setup/browse", params={"path": str(tmp_path)})
    assert r.status_code == 200
    assert any(e["name"] == "music" for e in r.json()["entries"])
    bad = client.get("/api/setup/browse", params={"path": str(tmp_path / "nope")})
    assert bad.status_code == 400
