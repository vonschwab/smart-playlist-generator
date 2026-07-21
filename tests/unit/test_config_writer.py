# tests/unit/test_config_writer.py
"""Config writer: copies the commented template, patches user keys, atomic, never-clobber."""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from src.mixarc.paths import MixarcHome
from src.setup.config_writer import ConfigExistsError, write_config


def _home(tmp_path):
    return MixarcHome(config_path=tmp_path / "config.yaml", anchor_dir=tmp_path, source="platformdirs")


def _load(path):
    y = YAML()
    return y.load(Path(path).read_text(encoding="utf-8"))


def test_writes_music_dir_and_provider(tmp_path):
    home = _home(tmp_path)
    p = write_config(home, {"music_directory": "/music", "ai_genre_provider": "zero_touch"})
    data = _load(p)
    assert data["library"]["music_directory"] == "/music"
    assert data["ai_genre"]["provider"] == "zero_touch"


def test_preserves_template_comments(tmp_path):
    """A comment from config.example.yaml must survive into the written config."""
    home = _home(tmp_path)
    p = write_config(home, {"music_directory": "/music"})
    text = Path(p).read_text(encoding="utf-8")
    assert "#" in text  # comments preserved (dump would strip them)


def test_optional_services_patched_only_when_present(tmp_path):
    home = _home(tmp_path)
    p = write_config(home, {"music_directory": "/m", "lastfm": {"api_key": "k", "username": "u"}})
    data = _load(p)
    assert data["lastfm"]["api_key"] == "k" and data["lastfm"]["username"] == "u"


def test_refuses_clobber_without_reconfigure(tmp_path):
    home = _home(tmp_path)
    write_config(home, {"music_directory": "/m"})
    with pytest.raises(ConfigExistsError):
        write_config(home, {"music_directory": "/other"})


def test_reconfigure_overwrites(tmp_path):
    home = _home(tmp_path)
    write_config(home, {"music_directory": "/m"})
    p = write_config(home, {"music_directory": "/new"}, reconfigure=True)
    assert _load(p)["library"]["music_directory"] == "/new"
