"""Tests for PresetManager with UIState storage."""

import json
from pathlib import Path

import pytest

from src.playlist_gui.config.presets import PresetManager
from src.playlist_gui.ui_state import UIStateModel


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """PresetManager with isolated temp directory."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    return PresetManager()


def test_save_and_load_preset(manager):
    state = UIStateModel(mode="genre", genre_query="ambient", cohesion_mode="strict")
    manager.save_preset("My Ambient", state)
    loaded = manager.load_preset("My Ambient")
    assert loaded is not None
    assert loaded == state


def test_load_nonexistent_returns_none(manager):
    assert manager.load_preset("does not exist") is None


def test_list_presets_excludes_session_file(manager):
    state = UIStateModel()
    manager.save_preset("Preset One", state)
    manager.save_session(state)
    presets = manager.list_presets()
    names = [p["name"] for p in presets]
    assert "Preset One" in names
    assert "_session" not in names


def test_delete_preset(manager):
    state = UIStateModel(mode="seeds")
    manager.save_preset("Temporary", state)
    assert manager.preset_exists("Temporary")
    manager.delete_preset("Temporary")
    assert not manager.preset_exists("Temporary")


def test_save_session_and_load_session(manager):
    state = UIStateModel(
        mode="artist",
        cohesion_mode="discover",
        track_count=40,
        artist_queries=["Boards of Canada"],
    )
    manager.save_session(state)
    loaded = manager.load_session()
    assert loaded == state


def test_load_session_returns_none_when_missing(manager):
    assert manager.load_session() is None


def test_load_session_returns_none_on_corrupt_file(manager):
    session_path = manager.presets_dir / "_session.json"
    session_path.write_text("not valid json {{{", encoding="utf-8")
    assert manager.load_session() is None


def test_load_preset_handles_missing_fields(manager):
    """Old preset file missing new fields gets defaults."""
    import yaml

    path = manager._get_preset_path("Old Preset")
    data = {
        "name": "Old Preset",
        "version": 1,
        "state": {"mode": "artist", "genre_mode": "narrow"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    loaded = manager.load_preset("Old Preset")
    assert loaded is not None
    assert loaded.mode == "artist"
    assert loaded.genre_mode == "narrow"
    assert loaded.cohesion_mode == "dynamic"  # default
    assert loaded.track_count == 30  # default


def test_load_preset_full_returns_metadata(manager):
    state = UIStateModel(cohesion_mode="narrow")
    manager.save_preset("With Meta", state, description="A test preset")
    full = manager.load_preset_full("With Meta")
    assert full is not None
    assert full["name"] == "With Meta"
    assert full["description"] == "A test preset"
    assert full["version"] == 1
    assert full["state"]["cohesion_mode"] == "narrow"
