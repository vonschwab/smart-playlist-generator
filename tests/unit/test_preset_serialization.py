"""Tests for UIState serialization/deserialization."""

from dataclasses import asdict

from src.playlist_gui.config.presets import deserialize_ui_state, serialize_ui_state
from src.playlist_gui.ui_state import UIStateModel


def test_round_trip_default_state():
    state = UIStateModel()
    data = serialize_ui_state(state)
    restored = deserialize_ui_state(data)
    assert restored == state


def test_round_trip_custom_state():
    state = UIStateModel(
        mode="genre",
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
        artist_spacing="very_strong",
        artist_queries=["Slowdive", "Cocteau Twins"],
        artist_presence="high",
        artist_variety="sprawling",
        include_collaborations=True,
        genre_query="ambient",
        seed_track_ids=["track_001", "track_002"],
        seed_auto_order=False,
    )
    data = serialize_ui_state(state)
    restored = deserialize_ui_state(data)
    assert restored == state


def test_deserialize_ignores_unknown_fields():
    data = asdict(UIStateModel())
    data["future_field"] = "some_value"
    data["another_unknown"] = 42
    restored = deserialize_ui_state(data)
    assert restored == UIStateModel()


def test_deserialize_fills_missing_fields_with_defaults():
    data = {"mode": "genre", "genre_query": "shoegaze"}
    restored = deserialize_ui_state(data)
    assert restored.mode == "genre"
    assert restored.genre_query == "shoegaze"
    assert restored.cohesion_mode == "dynamic"
    assert restored.track_count == 30
    assert restored.artist_spacing == "normal"


def test_serialize_produces_plain_dict():
    state = UIStateModel(artist_queries=["Miles Davis"])
    data = serialize_ui_state(state)
    assert isinstance(data, dict)
    assert data["artist_queries"] == ["Miles Davis"]
    assert "mode" in data
