"""Unit tests for resolve_database_path: config -> absolute DB path (repo-root, not cwd)."""

import os
from pathlib import Path

from src.config_loader import Config, resolve_database_path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg_object(tmp_path, db_value):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"library:\n  music_directory: E:\\\\MUSIC\n  database_path: {db_value}\n",
        encoding="utf-8",
    )
    return Config(str(cfg_file))


def test_absolute_path_returned_as_is(tmp_path):
    abs_db = (tmp_path / "elsewhere" / "metadata.db")
    result = resolve_database_path(_cfg_object(tmp_path, abs_db.as_posix()))
    assert Path(result) == abs_db.resolve()


def test_relative_resolves_against_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd != repo root
    result = resolve_database_path(_cfg_object(tmp_path, "data/metadata.db"))
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()
    assert os.path.isabs(result)


def test_dict_input_absolute(tmp_path):
    abs_db = (tmp_path / "d" / "metadata.db")
    result = resolve_database_path({"library": {"database_path": abs_db.as_posix()}})
    assert Path(result) == abs_db.resolve()


def test_dict_input_relative_resolves_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve_database_path({"library": {"database_path": "data/metadata.db"}})
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_missing_library_falls_back_to_default_absolute(tmp_path):
    assert Path(resolve_database_path({})) == (_REPO_ROOT / "data" / "metadata.db").resolve()
    assert Path(resolve_database_path({"library": {}})) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_none_input_falls_back_to_default_absolute(tmp_path):
    assert Path(resolve_database_path(None)) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_canonical_relative_config_resolves_to_repo_db(tmp_path, monkeypatch):
    # Regression: the real canonical pattern (relative 'data/metadata.db') must
    # resolve to the repo's real DB path regardless of cwd.
    monkeypatch.chdir(tmp_path)
    result = resolve_database_path({"library": {"database_path": "data/metadata.db"}})
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()
