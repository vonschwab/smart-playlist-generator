import numpy as np
import pytest

from src.playlist.config import resolve_pier_bridge_tuning


def test_tuning_genre_steering_defaults_off():
    """No overrides -> steering off, inert genre knobs, legacy weights intact."""
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=None)
    assert t.genre_steering_enabled is False
    assert t.weight_genre == 0.0
    assert t.genre_edge_floor == 0.0
    # Legacy narrow weights unchanged when steering off
    assert abs(t.weight_bridge - 0.7) < 1e-6
    assert abs(t.weight_transition - 0.3) < 1e-6


def test_tuning_genre_steering_renormalizes_weights():
    """When enabled with weight_genre, the three edge weights renormalize to sum 1."""
    overrides = {
        "pier_bridge": {
            "genre_steering_enabled": True,
            "weight_genre_narrow": 0.20,
            "genre_edge_floor_narrow": 0.40,
            # leave bridge/transition at narrow defaults 0.7/0.3
        }
    }
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert abs(t.genre_edge_floor - 0.40) < 1e-6
    total = t.weight_bridge + t.weight_transition + t.weight_genre
    assert abs(total - 1.0) < 1e-6
    # genre share is 0.20 / 1.20
    assert abs(t.weight_genre - (0.20 / 1.20)) < 1e-6


def test_tuning_steering_on_zero_weight_genre_is_noop():
    overrides = {"pier_bridge": {"genre_steering_enabled": True}}  # no weight_genre
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert t.weight_genre == 0.0
    assert abs(t.weight_bridge - 0.7) < 1e-6
    assert abs(t.weight_transition - 0.3) < 1e-6


def test_infeasible_handling_genre_floor_fields_default():
    from src.playlist.run_audit import InfeasibleHandlingConfig, parse_infeasible_handling_config
    cfg = InfeasibleHandlingConfig()
    assert cfg.genre_floor_relaxation_enabled is True
    assert cfg.min_genre_edge_floor == 0.0
    parsed = parse_infeasible_handling_config({
        "enabled": True, "min_genre_edge_floor": 0.15, "genre_floor_relaxation_enabled": False,
    })
    assert parsed.genre_floor_relaxation_enabled is False
    assert abs(parsed.min_genre_edge_floor - 0.15) < 1e-6
