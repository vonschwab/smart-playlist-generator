"""Instrumental-lean voice_prob soft penalty in beam search.

Mirrors tests/unit/test_beam_energy_pace.py's fixture shape (flat sonic
space, pier_a/pier_b at the ends, two otherwise-identical bridge
candidates) but drives the new voice_prob param instead of energy_matrix.

Contract:
  (1) With two candidates — one low-voice_prob (instrumental), one
      high-voice_prob (vocal) — and instrumental_penalty_weight > 0, the
      instrumental candidate is picked.
  (2) When ONLY the vocal candidate exists, the segment still builds
      (beam_failure_reason is None) — the penalty is additive, never a
      hard gate.
  (3) voice_prob=None -> no penalty; the segment still succeeds.
"""
import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _flat_space(n: int) -> np.ndarray:
    """n identical unit vectors so sonic gates pass with floors at -1."""
    X = np.ones((n, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fixture layout
#   idx 0  pier_a
#   idx 1  c_instrumental  voice_prob 0.02  (low -> negligible penalty)
#   idx 2  c_vocal         voice_prob 0.95  (high -> large penalty)
#   idx 3  pier_b
# ---------------------------------------------------------------------------
_PIER_A = 0
_PIER_B = 3
_C_INSTRUMENTAL = 1
_C_VOCAL = 2

_VOICE_PROB = np.array([np.nan, 0.02, 0.95, np.nan], dtype=np.float64)


def _cfg_instrumental(
    bridge_floor: float = -1.0,
    transition_floor: float = -1.0,
    progress_enabled: bool = False,
    instrumental_enabled: bool = True,
    instrumental_penalty_weight: float = 5.0,
) -> PierBridgeConfig:
    return PierBridgeConfig(
        bridge_floor=bridge_floor,
        transition_floor=transition_floor,
        progress_enabled=progress_enabled,
        instrumental_enabled=instrumental_enabled,
        instrumental_penalty_weight=instrumental_penalty_weight,
    )


def test_instrumental_penalty_demotes_vocal_candidate():
    """With both candidates available, the instrumental one must be picked."""
    X = _flat_space(4)
    cfg = _cfg_instrumental()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_INSTRUMENTAL, _C_VOCAL],
        X, X, None, None, None, None,
        cfg, 5,
        voice_prob=_VOICE_PROB,
    )

    assert err is None, f"Segment failed unexpectedly: {err}"
    assert path is not None, "path should not be None when c_instrumental is available"
    assert path == [_C_INSTRUMENTAL], (
        f"Expected c_instrumental ({_C_INSTRUMENTAL}) to be picked over "
        f"c_vocal ({_C_VOCAL}), got {path}"
    )


def test_instrumental_penalty_never_hard_fails():
    """With ONLY the vocal candidate available, the segment must still build."""
    X = _flat_space(4)
    cfg = _cfg_instrumental()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_VOCAL],
        X, X, None, None, None, None,
        cfg, 5,
        voice_prob=_VOICE_PROB,
    )

    # The penalty is purely additive — it can never exclude the only candidate.
    assert err is None, (
        f"beam_failure_reason must be None (never-hard-fail): got {err!r}"
    )
    assert path is not None, "path must not be None when the only candidate exists"
    assert path == [_C_VOCAL]


def test_instrumental_penalty_disabled_when_voice_prob_is_none():
    """voice_prob=None -> no instrumental penalty; segment still succeeds."""
    X = _flat_space(4)
    cfg = _cfg_instrumental()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_INSTRUMENTAL, _C_VOCAL],
        X, X, None, None, None, None,
        cfg, 5,
        voice_prob=None,
    )

    assert err is None, f"Segment failed unexpectedly: {err}"
    assert path is not None
