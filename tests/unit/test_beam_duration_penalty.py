"""Duration soft penalty in beam search (C1 rehome, corridor pooling).

Mirrors tests/unit/test_beam_instrumental_penalty.py's fixture shape (flat
sonic space, pier_a/pier_b at the ends, two otherwise-identical bridge
candidates) but drives the new duration_penalty_values param instead of
voice_prob. Unlike voice_prob (a raw probability the beam converts into a
penalty via compute_instrumental_penalty), duration_penalty_values arrives
PRE-COMPUTED -- the corridor call site in pier_bridge_builder.py bakes the
weight in via src.playlist.candidate_pool.compute_duration_penalty before
threading the array down, so this test only needs to prove the beam's
consumption side: a positive value in the array demotes that candidate's
score; None makes the term a pure no-op.

Contract:
  (1) With two candidates — one at/under the reference duration (penalty
      0.0), one far over it (large positive penalty) — the short candidate
      is picked.
  (2) When ONLY the over-length candidate exists, the segment still builds
      (beam_failure_reason is None) — the penalty is additive, never a hard
      gate.
  (3) duration_penalty_values=None -> no penalty; both candidates are true
      score-ties (identical sonic vectors, identical everything else), so
      whichever the beam is offered first wins the tie -- proving the two
      candidates score IDENTICALLY absent the array (the legacy-inertness
      proof: legacy pooling never sets this array, so it always behaves
      like this branch).
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
#   idx 1  c_short   duration_penalty_values 0.0  (at/under reference)
#   idx 2  c_long    duration_penalty_values 5.0  (far over reference)
#   idx 3  pier_b
# ---------------------------------------------------------------------------
_PIER_A = 0
_PIER_B = 3
_C_SHORT = 1
_C_LONG = 2

_DURATION_PENALTY_VALUES = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)


def _cfg(bridge_floor: float = -1.0, transition_floor: float = -1.0, progress_enabled: bool = False) -> PierBridgeConfig:
    return PierBridgeConfig(
        bridge_floor=bridge_floor,
        transition_floor=transition_floor,
        progress_enabled=progress_enabled,
    )


def test_duration_penalty_demotes_long_candidate():
    """With both candidates available, the short one must be picked."""
    X = _flat_space(4)
    cfg = _cfg()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_SHORT, _C_LONG],
        X, X, None, None, None, None,
        cfg, 5,
        duration_penalty_values=_DURATION_PENALTY_VALUES,
    )

    assert err is None, f"Segment failed unexpectedly: {err}"
    assert path is not None, "path should not be None when c_short is available"
    assert path == [_C_SHORT], (
        f"Expected c_short ({_C_SHORT}) to be picked over c_long ({_C_LONG}), got {path}"
    )

    # And the winner flips with candidate-list order too (rules out an
    # accidental order-dependent tie masking as a real demotion).
    path2, _hits2, _edges2, err2 = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_LONG, _C_SHORT],
        X, X, None, None, None, None,
        cfg, 5,
        duration_penalty_values=_DURATION_PENALTY_VALUES,
    )
    assert err2 is None
    assert path2 == [_C_SHORT], f"Order-flipped run still expected c_short, got {path2}"


def test_duration_penalty_never_hard_fails():
    """With ONLY c_long available (over the reference), the segment must still build."""
    X = _flat_space(4)
    cfg = _cfg()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_LONG],
        X, X, None, None, None, None,
        cfg, 5,
        duration_penalty_values=_DURATION_PENALTY_VALUES,
    )

    # The penalty is purely additive — it can never exclude the only candidate.
    assert err is None, (
        f"beam_failure_reason must be None (never-hard-fail): got {err!r}"
    )
    assert path is not None, "path must not be None when the only candidate exists"
    assert path == [_C_LONG]


def test_duration_penalty_disabled_when_values_is_none_proves_identical_scores():
    """duration_penalty_values=None -> no penalty; candidates are true score-ties.

    Legacy-inertness proof: with the array withheld (exactly what the legacy
    pooling call site does, always), c_short and c_long score IDENTICALLY --
    proven here by showing the winner is purely a function of candidate-list
    order (first-offered wins the tie), which can only happen when neither
    candidate's score dominates the other's.
    """
    X = _flat_space(4)
    cfg = _cfg()

    path_a, _h1, _e1, err_a = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_SHORT, _C_LONG],
        X, X, None, None, None, None,
        cfg, 5,
        duration_penalty_values=None,
    )
    path_b, _h2, _e2, err_b = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_LONG, _C_SHORT],
        X, X, None, None, None, None,
        cfg, 5,
        duration_penalty_values=None,
    )

    assert err_a is None and err_b is None
    assert path_a is not None and path_b is not None
    # Each run picks whichever candidate was offered first -- proof the two
    # score identically when the array is withheld (no inherent bias toward
    # c_short absent the penalty).
    assert path_a == [_C_SHORT], f"Expected first-offered c_short to win the tie, got {path_a}"
    assert path_b == [_C_LONG], f"Expected first-offered c_long to win the tie, got {path_b}"
