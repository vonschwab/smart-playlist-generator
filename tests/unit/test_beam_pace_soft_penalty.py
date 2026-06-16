"""Pace bridge bands as soft penalties, not hard gates.

Root cause (2026-06-16): the onset-rate bridge band is a HARD gate. When a pier
is an onset-rate outlier (e.g. Yo La Tengo's "Georgia (Tuesday)", onset 0.024 —
the library floor), the per-step onset target sweeps into a near-empty region and
the beam rejects every candidate ("no valid continuations at step N"), detonating
the relaxation cascade. Converting the band to a soft penalty (demote, never
reject) lets the beam still build the segment when nothing is in-band, while
keeping the band's pull when in-band candidates exist.
"""
import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _flat_space(n: int) -> np.ndarray:
    """n identical unit vectors so sonic gates pass with floors at -1."""
    X = np.ones((n, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_onset_soft_penalty_admits_out_of_band_candidate():
    # pier_a onset 2.0, only candidate onset 0.02 (far out of band vs the step-0
    # target = onset_a). With the soft penalty the candidate is demoted but kept.
    X = _flat_space(3)  # 0=pier_a, 1=candidate, 2=pier_b
    onset = np.array([2.0, 0.02, 2.0], dtype=float)

    path, _hits, _edges, err = _beam_search_segment(
        0, 2, 1, [1],
        X, X, None, None, None, None,
        PierBridgeConfig(
            bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
            onset_bridge_max_log_distance=0.85,
            onset_bridge_soft_penalty_strength=1.0,
        ),
        5,
        onset_rate=onset,
    )

    assert err is None
    assert path == [1]


def test_onset_hard_gate_is_default_backward_compatible():
    # strength defaults to 0.0 -> legacy hard gate -> out-of-band candidate rejected.
    X = _flat_space(3)
    onset = np.array([2.0, 0.02, 2.0], dtype=float)

    path, _hits, _edges, err = _beam_search_segment(
        0, 2, 1, [1],
        X, X, None, None, None, None,
        PierBridgeConfig(
            bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
            onset_bridge_max_log_distance=0.85,
        ),
        5,
        onset_rate=onset,
    )

    assert path is None
    assert err is not None and "continuations" in err


def test_onset_soft_penalty_inert_for_in_band_candidate():
    # In-band candidate: penalty is zero, so behavior is unchanged from the gate.
    X = _flat_space(3)
    onset = np.array([2.0, 2.0, 2.0], dtype=float)

    path, _hits, _edges, err = _beam_search_segment(
        0, 2, 1, [1],
        X, X, None, None, None, None,
        PierBridgeConfig(
            bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
            onset_bridge_max_log_distance=0.85,
            onset_bridge_soft_penalty_strength=1.0,
        ),
        5,
        onset_rate=onset,
    )

    assert err is None
    assert path == [1]


def test_bpm_soft_penalty_admits_out_of_band_candidate():
    # Same conversion for the BPM band (symmetry; same detonation risk).
    X = _flat_space(3)
    bpm = np.array([120.0, 200.0, 120.0], dtype=float)

    path, _hits, _edges, err = _beam_search_segment(
        0, 2, 1, [1],
        X, X, None, None, None, None,
        PierBridgeConfig(
            bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
            bpm_bridge_max_log_distance=0.10,
            bpm_bridge_soft_penalty_strength=1.0,
        ),
        5,
        perceptual_bpm=bpm,
    )

    assert err is None
    assert path == [1]
