import numpy as np
import pytest

from src.playlist.config import resolve_pier_bridge_tuning
from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _diag4():
    # 4 tracks: 0=pierA, 1=genre-OK candidate, 2=genre-misfit candidate, 3=pierB.
    # Sonic: all identical so sonic never decides (forces genre to break the tie).
    X = np.ones((4, 3), dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Dense genre (L2-normalized rows): cand 1 aligns with piers; cand 2 is orthogonal.
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [1.0, 0.0, 0.0],   # cand 1: same genre as piers (sim 1.0)
        [0.0, 1.0, 0.0],   # cand 2: orthogonal genre (sim 0.0)
        [1.0, 0.0, 0.0],   # pierB
    ], dtype=float)
    return Xn, dense


def test_beam_floor_rejects_off_genre_candidate():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.2, genre_edge_floor=0.5,
        weight_bridge=0.5, weight_transition=0.3,
    )
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert path == [1], f"expected genre-OK cand 1, got {path}"


def test_beam_steering_prefers_higher_genre_when_sonic_tied():
    Xn, dense = _diag4()
    # No floor (0.0) so cand 2 is allowed; steering weight should still rank cand 1 first.
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.3, genre_edge_floor=0.0,
        weight_bridge=0.4, weight_transition=0.3,
    )
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert path == [1]


def test_beam_legacy_unchanged_when_steering_off():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=False,
    )
    # Steering off: floor must NOT reject; both candidates valid, no crash.
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert len(path) == 1


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
