"""Tests for GUI generation request validation."""

from dataclasses import asdict

from src.playlist_gui.main_window import _validate_generation_request
from src.playlist_gui.main_window import MainWindow
from src.playlist_gui.ui_state import UIStateModel


def test_artist_mode_requires_artist_input():
    state = UIStateModel(mode="artist", artist_queries=[""])

    message = _validate_generation_request(state)

    assert message == "Enter an artist before generating."


def test_genre_mode_requires_genre_input():
    state = UIStateModel(mode="genre", genre_query=" ")

    message = _validate_generation_request(state)

    assert message == "Enter a genre before generating."


def test_seeds_mode_requires_at_least_one_seed():
    state = UIStateModel(mode="seeds")

    message = _validate_generation_request(state, seed_tracks=[], seed_track_ids=[])

    assert message == "Add at least one seed track before generating."


def test_seeds_mode_accepts_display_seed_or_seed_id():
    state = UIStateModel(mode="seeds")

    assert _validate_generation_request(state, seed_tracks=["Artist - Title"], seed_track_ids=[]) is None
    assert _validate_generation_request(state, seed_tracks=[], seed_track_ids=["123"]) is None


def test_seeds_mode_accepts_seed_id_from_ui_state():
    state = UIStateModel(mode="seeds", seed_track_ids=["123"])

    assert _validate_generation_request(state) is None


def test_history_mode_has_no_required_text_input():
    state = UIStateModel(mode="history")

    assert _validate_generation_request(state) is None


class _FakeConfigModel:
    def get_overrides(self):
        return {}


class _FakeGeneratePanel:
    def __init__(self):
        self.validation_messages = []
        self.cleared = 0

    def get_seed_track_ids(self):
        return []

    def get_seed_display_strings(self):
        return []

    def show_validation_message(self, message):
        self.validation_messages.append(message)

    def clear_validation_message(self):
        self.cleared += 1


def test_main_window_surfaces_generation_validation_inline(monkeypatch):
    import src.playlist_gui.main_window as main_window_mod

    warnings = []
    monkeypatch.setattr(
        main_window_mod.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    panel = _FakeGeneratePanel()
    window = MainWindow.__new__(MainWindow)
    window._is_generating = False
    window._config_model = _FakeConfigModel()
    window._generate_panel = panel

    MainWindow._on_generate_v2(
        window,
        asdict(UIStateModel(mode="artist", artist_queries=[""])),
    )

    assert panel.validation_messages == ["Enter an artist before generating."]
    assert warnings == []
