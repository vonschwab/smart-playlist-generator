"""Roam corridors: the beam's per-dimension corridor penalty (flag-gated)."""
import numpy as np

from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

# A=0, B=1, candidates 2 and 3. All scoring weights are zeroed so the corridor
# penalty is the ONLY differentiator between the two candidates.
_X = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.6, 0.8]])
_XN = _X / (np.linalg.norm(_X, axis=1, keepdims=True) + 1e-12)


def _run(detour, enabled):
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        roam_corridors_enabled=enabled,
        roam_width_sonic=0.0,
        roam_penalty_slope=10.0,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0, 1, 1, [2, 3], _XN, _XN, None, None, None, None, cfg, 5,
        roam_detour_sonic=np.asarray(detour, dtype=np.float64),
    )
    assert err is None
    return path


def test_corridor_penalizes_the_high_detour_candidate():
    # Penalize candidate 3 (detour 5) -> candidate 2 chosen; and vice-versa.
    assert _run([0.0, 0.0, 0.0, 5.0], enabled=True) == [2]
    assert _run([0.0, 0.0, 5.0, 0.0], enabled=True) == [3]


def test_corridor_ignored_when_disabled():
    # Flag off: the detour array has no effect, so opposite detours give the same
    # winner (whatever the weight-zeroed tie-break is) — proving it's gated.
    assert _run([0.0, 0.0, 5.0, 0.0], enabled=False) == _run([0.0, 0.0, 0.0, 5.0], enabled=False)
