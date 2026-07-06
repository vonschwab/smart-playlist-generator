"""Unit tests for the shared workspace-detection helper (canonical vs satellite)."""

import importlib.util
import pathlib

_MOD = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "workspace_identity.py"
)
_spec = importlib.util.spec_from_file_location("workspace_identity", _MOD)
assert _spec is not None and _spec.loader is not None, f"helper not found at {_MOD}"
wsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsi)


def _repo_with_origin(tmp_path, url):
    git = tmp_path / ".git"
    git.mkdir()
    body = "[core]\n\trepositoryformatversion = 0\n"
    if url is not None:
        body += f'[remote "origin"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    (git / "config").write_text(body, encoding="utf-8")
    return tmp_path


def test_github_https_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, "https://github.com/vonschwab/playlist-generator.git")
    assert wsi.is_satellite(repo) is False


def test_github_ssh_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, "git@github.com:vonschwab/playlist-generator.git")
    assert wsi.is_satellite(repo) is False


def test_windows_path_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_backslash_path_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_file_url_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "file:///C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_no_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, None)
    assert wsi.is_satellite(repo) is False


def test_missing_git_dir_is_canonical(tmp_path):
    assert wsi.is_satellite(tmp_path) is False


def test_origin_url_extraction(tmp_path):
    repo = _repo_with_origin(tmp_path, "https://github.com/x/y.git")
    assert wsi.origin_url(repo) == "https://github.com/x/y.git"


def test_this_repo_is_canonical():
    # The canonical checkout's origin is GitHub — detection must say canonical.
    assert wsi.is_satellite(pathlib.Path(__file__).resolve().parents[1]) is False
