import numpy as np

from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    _beam_search_segment,
    _progress_arc_loss_value,
    _step_fraction,
    _progress_target_curve,
)


def test_progress_target_curve_monotonic():
    steps = 4
    linear = [_progress_target_curve(i, steps, "linear") for i in range(steps)]
    arc = [_progress_target_curve(i, steps, "arc") for i in range(steps)]
    assert linear == sorted(linear)
    assert arc == sorted(arc)
    assert linear[0] > 0.0
    assert linear[-1] < 1.0
    assert arc[0] > 0.0
    assert arc[-1] < 1.0


def test_step_fraction_convention():
    assert _step_fraction(0, 4) == 1.0 / 5.0
    assert _step_fraction(3, 4) == 4.0 / 5.0
    assert _step_fraction(0, 0) == 0.0


def test_progress_arc_loss_values():
    err = 0.2
    assert _progress_arc_loss_value(err, "abs", 0.1) == err
    assert _progress_arc_loss_value(err, "squared", 0.1) == err * err
    huber_small = _progress_arc_loss_value(0.05, "huber", 0.1)
    assert huber_small == 0.5 * 0.05 * 0.05
    huber_large = _progress_arc_loss_value(0.2, "huber", 0.1)
    assert huber_large == 0.1 * (0.2 - 0.5 * 0.1)


def test_progress_arc_tolerance_band():
    err0 = abs(0.6 - 0.5)
    tolerance = 0.2
    err = max(0.0, err0 - tolerance)
    assert err == 0.0
    assert _progress_arc_loss_value(err, "abs", 0.1) == 0.0


def test_progress_arc_autoscale_disables_below_min_distance():
    X = np.array([
        [1.0, 0.0],  # pier A
        [0.9, 0.1],  # candidate
        [0.0, 1.0],  # pier B
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    arc_stats = {}
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_penalty_weight=0.0,
        progress_arc_enabled=True,
        progress_arc_weight=0.5,
        progress_arc_autoscale_enabled=True,
        progress_arc_autoscale_min_distance=2.0,
        progress_arc_autoscale_distance_scale=1.0,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        2,
        1,
        [1],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        cfg,
        5,
        arc_stats=arc_stats,
    )
    assert err is None
    assert path == [1]
    assert arc_stats["effective_weight"] == 0.0


def test_progress_arc_per_step_scale():
    X = np.array([
        [1.0, 0.0],  # pier A
        [0.0, 1.0],  # pier B
        [0.8, 0.3],
        [0.3, 0.8],
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    arc_stats = {}
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_penalty_weight=0.0,
        progress_arc_enabled=True,
        progress_arc_weight=0.6,
        progress_arc_autoscale_per_step_scale=True,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        2,
        [2, 3],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        cfg,
        5,
        arc_stats=arc_stats,
    )
    assert err is None
    assert arc_stats["effective_weight"] == 0.3


def test_progress_arc_max_step_gate_blocks_large_jump():
    X = np.array([
        [1.0, 0.0],  # pier A
        [0.0, 1.0],  # pier B
        [0.1, 0.95],  # big jump
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_penalty_weight=0.0,
        progress_arc_enabled=True,
        progress_arc_max_step=0.2,
        progress_arc_max_step_mode="gate",
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        cfg,
        5,
    )
    assert err is not None
    assert path is None


def test_progress_arc_max_step_penalty_prefers_small_jump():
    X = np.array([
        [1.0, 0.0],  # pier A
        [1.0, 1.0],  # pier B (normalized) => equal final transition to cand2/cand3
        [1.0, 0.0],  # small jump
        [0.0, 1.0],  # big jump
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_penalty_weight=0.0,
        progress_arc_enabled=True,
        progress_arc_max_step=0.2,
        progress_arc_max_step_mode="penalty",
        progress_arc_max_step_penalty=1.0,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        cfg,
        5,
    )
    assert err is None
    assert path == [2]


def test_progress_arc_max_step_penalty_yields_to_better_final_transition():
    # When final transition quality is much better, it can outweigh max_step
    # penalty. This ensures we prioritize smooth transitions over strict step
    # size when the user hasn't set an extreme penalty.
    X = np.array([
        [1.0, 0.0],  # pier A
        [0.0, 1.0],  # pier B
        [0.9, 0.2],  # smaller jump, worse final transition
        [0.1, 0.95],  # bigger jump, much better final transition
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_penalty_weight=0.0,
        progress_arc_enabled=True,
        progress_arc_max_step=0.2,
        progress_arc_max_step_mode="penalty",
        progress_arc_max_step_penalty=0.1,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_norm,
        X_norm,
        None,
        None,
        None,
        None,
        cfg,
        5,
    )
    assert err is None
    assert path == [3]
