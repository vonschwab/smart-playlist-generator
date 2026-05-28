"""Tests for programmatic UI state restoration."""

import pytest

from src.playlist_gui.config.presets import PresetManager
from src.playlist_gui.ui_state import UIStateModel
from src.playlist_gui.widgets.generate_panel import GeneratePanel


def test_artist_panel_set_presence(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_presence("very_high")
    assert panel._artist_panel.get_presence() == "very_high"

    panel._artist_panel.set_presence("very_low")
    assert panel._artist_panel.get_presence() == "very_low"


def test_artist_panel_set_variety(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_variety("sprawling")
    assert panel._artist_panel.get_variety() == "sprawling"

    panel._artist_panel.set_variety("focused")
    assert panel._artist_panel.get_variety() == "focused"


def test_seeds_panel_set_auto_order(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._seeds_panel.set_auto_order(False)
    assert panel._seeds_panel.get_auto_order() is False

    panel._seeds_panel.set_auto_order(True)
    assert panel._seeds_panel.get_auto_order() is True


from src.playlist_gui.ui_state import UIStateModel


def test_apply_ui_state_round_trip_defaults(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    original = UIStateModel()
    panel.apply_ui_state(original)
    restored = panel.build_ui_state()

    assert restored.mode == original.mode
    assert restored.cohesion_mode == original.cohesion_mode
    assert restored.genre_mode == original.genre_mode
    assert restored.sonic_mode == original.sonic_mode
    assert restored.pace_mode == original.pace_mode
    assert restored.track_count == original.track_count
    assert restored.diversity_gamma == original.diversity_gamma
    assert restored.artist_diversity_mode == original.artist_diversity_mode
    assert restored.recency_enabled == original.recency_enabled
    assert restored.recency_days == original.recency_days
    assert restored.recency_plays_threshold == original.recency_plays_threshold
    assert restored.exclude_seed_tracks_from_recency == original.exclude_seed_tracks_from_recency
    assert restored.artist_spacing == original.artist_spacing


def test_apply_ui_state_round_trip_custom_artist_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(
        mode="artist",
        cohesion_mode="strict",
        genre_mode="dynamic",
        sonic_mode="off",
        pace_mode="strict",
        track_count=50,
        diversity_gamma=0.08,
        artist_diversity_mode="one_per_artist",
        recency_enabled=False,
        recency_days=30,
        recency_plays_threshold=3,
        exclude_seed_tracks_from_recency=True,
        artist_spacing="very_strong",
        artist_queries=["Slowdive"],
        artist_presence="high",
        artist_variety="sprawling",
        include_collaborations=True,
    )
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "artist"
    assert restored.cohesion_mode == "strict"
    assert restored.genre_mode == "dynamic"
    assert restored.sonic_mode == "off"
    assert restored.pace_mode == "strict"
    assert restored.track_count == 50
    assert restored.diversity_gamma == 0.08
    assert restored.artist_diversity_mode == "one_per_artist"
    assert restored.recency_enabled is False
    assert restored.recency_days == 30
    assert restored.recency_plays_threshold == 3
    assert restored.exclude_seed_tracks_from_recency is True
    assert restored.artist_spacing == "very_strong"
    assert restored.artist_queries == ["Slowdive"]
    assert restored.artist_presence == "high"
    assert restored.artist_variety == "sprawling"
    assert restored.include_collaborations is True


def test_apply_ui_state_round_trip_genre_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(mode="genre", genre_query="shoegaze")
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "genre"
    assert restored.genre_query == "shoegaze"
    assert restored.exclude_seed_tracks_from_recency is False


def test_apply_ui_state_round_trip_seeds_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(
        mode="seeds",
        seed_auto_order=False,
    )
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "seeds"
    assert restored.seed_auto_order is False


from src.playlist_gui.config.presets import PresetManager


def test_preset_save_load_round_trip_through_panel(qtbot, tmp_path, monkeypatch):
    """Full workflow: build state from panel → save preset → load preset → apply → verify."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    manager = PresetManager()
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    # Set up non-default state via controls
    panel._cohesion_slider.set_cohesion_mode("strict")
    panel._mode_sliders.set_genre_mode("dynamic")
    panel._spacing_slider.setValue(3)  # very_strong
    panel._diversity_slider.setValue(5)  # One Each

    # Save preset from panel state
    original_state = panel.build_ui_state()
    manager.save_preset("Test Preset", original_state)

    # Reset panel to defaults
    panel.apply_ui_state(UIStateModel())
    assert panel.build_ui_state().cohesion_mode == "dynamic"

    # Load preset and apply
    loaded_state = manager.load_preset("Test Preset")
    assert loaded_state is not None
    panel.apply_ui_state(loaded_state)

    # Verify round-trip
    final_state = panel.build_ui_state()
    assert final_state.cohesion_mode == "strict"
    assert final_state.genre_mode == "dynamic"
    assert final_state.artist_spacing == "very_strong"
    assert final_state.artist_diversity_mode == "one_per_artist"


def test_session_save_load_round_trip_through_panel(qtbot, tmp_path, monkeypatch):
    """Full workflow: build state → save session → load session → apply → verify."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    manager = PresetManager()
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    # Set up state
    panel._mode_sliders.set_pace_mode("strict")
    panel._recency_check.setChecked(False)

    # Save session
    state = panel.build_ui_state()
    manager.save_session(state)

    # Reset
    panel.apply_ui_state(UIStateModel())
    assert panel.build_ui_state().pace_mode == "dynamic"
    assert panel.build_ui_state().recency_enabled is True

    # Restore session
    session = manager.load_session()
    assert session is not None
    panel.apply_ui_state(session)

    final = panel.build_ui_state()
    assert final.pace_mode == "strict"
    assert final.recency_enabled is False
