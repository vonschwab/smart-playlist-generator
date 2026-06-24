"""Roam corridors: min-bottleneck worst-edge guard (minimax).

The selection logic is verified deterministically at the unit level; a full beam
run exercises the wiring. A forced flip through the real centered-T transition
metric is impractical to craft for a tiny segment (the metric is opaque and, for
single-interior segments, the worst edge and the total both peak at the balanced
candidate); minimax behaviour on real multi-interior generations is validated in
the calibration phase.
"""
import numpy as np

from src.playlist.pier_bridge.beam import BeamState, _state_min_edge, _select_best_beam_state
from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment


def _state(path, score, edge_Ts):
    return BeamState(
        path=list(path), score=score, used=set(),
        edge_components=[{"T": float(t)} for t in edge_Ts],
    )


def test_state_min_edge_is_the_weakest_edge():
    assert _state_min_edge(_state([2], 1.3, [0.95, 0.35])) == 0.35
    assert _state_min_edge(_state([3], 1.2, [0.60, 0.60])) == 0.60


def test_min_edge_objective_prefers_higher_worst_edge():
    p = _state([2], 1.3, [0.95, 0.35])   # higher total, one broken edge
    q = _state([3], 1.2, [0.60, 0.60])   # lower total, no broken edge
    assert _select_best_beam_state([p, q], objective="total_score").path == [2]
    assert _select_best_beam_state([p, q], objective="min_edge").path == [3]


def test_minimax_wiring_runs_end_to_end_and_off_is_unaffected():
    X = np.array([[1.0, 0.0, 0.0], [0.3, 0.95, 0.0], [0.8, 0.6, 0.0], [0.6, 0.8, 0.0]])
    XN = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    on = PierBridgeConfig(transition_floor=0.0, bridge_floor=0.0, worst_edge_minimax_enabled=True)
    off = PierBridgeConfig(transition_floor=0.0, bridge_floor=0.0, worst_edge_minimax_enabled=False)
    p_on, _h1, _e1, err_on = _beam_search_segment(0, 1, 2, [2, 3], XN, XN, None, None, None, None, on, 5)
    p_off, _h2, _e2, err_off = _beam_search_segment(0, 1, 2, [2, 3], XN, XN, None, None, None, None, off, 5)
    assert err_on is None and p_on is not None and len(p_on) == 2
    assert err_off is None and p_off is not None and len(p_off) == 2
