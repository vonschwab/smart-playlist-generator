"""Energy step-cap + arc-band soft penalty in beam search.

Contract:
  (1) With two candidates — one near the energy arc, one far — and
      energy_step_strength > 0, the low-jump candidate is picked.
  (2) When ONLY the high-jump candidate exists, the segment still builds
      (beam_failure_reason is None) — the energy penalty is never a hard gate.
"""
import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _flat_space(n: int) -> np.ndarray:
    """n identical unit vectors so sonic gates pass with floors at -1."""
    X = np.ones((n, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _make_energy_matrix(values) -> np.ndarray:
    """Convert a list of scalar energies into a (n, 1) float32 matrix."""
    return np.array([[v] for v in values], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixture layout
#   idx 0  pier_a  energy 0.5
#   idx 1  c_lo    energy 0.6  (small jump from pier_a: 0.1)
#   idx 2  c_hi    energy 1.9  (huge jump from pier_a: 1.4)
#   idx 3  pier_b  energy 0.7
# ---------------------------------------------------------------------------
_PIER_A = 0
_PIER_B = 3
_C_LO = 1
_C_HI = 2

_ENERGIES = [0.5, 0.6, 1.9, 0.7]


def _cfg_energy(
    bridge_floor: float = -1.0,
    transition_floor: float = -1.0,
    progress_enabled: bool = False,
    energy_step_cap: float = 0.2,
    energy_step_strength: float = 5.0,
    energy_arc_band: float = 0.5,
    energy_arc_strength: float = 0.0,
) -> PierBridgeConfig:
    return PierBridgeConfig(
        bridge_floor=bridge_floor,
        transition_floor=transition_floor,
        progress_enabled=progress_enabled,
        energy_step_cap=energy_step_cap,
        energy_step_strength=energy_step_strength,
        energy_arc_band=energy_arc_band,
        energy_arc_strength=energy_arc_strength,
    )


def test_energy_step_demotes_big_jump_candidate():
    """With c_lo and c_hi both available, c_lo must be chosen."""
    X = _flat_space(4)
    em = _make_energy_matrix(_ENERGIES)
    cfg = _cfg_energy()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_LO, _C_HI],
        X, X, None, None, None, None,
        cfg, 5,
        energy_matrix=em,
    )

    assert err is None, f"Segment failed unexpectedly: {err}"
    assert path is not None, "path should not be None when c_lo is available"
    assert path == [_C_LO], (
        f"Expected c_lo ({_C_LO}) to be picked over c_hi ({_C_HI}), got {path}"
    )


def test_energy_step_never_hard_fails():
    """With ONLY c_hi available (large energy jump), the segment must still build."""
    X = _flat_space(4)
    em = _make_energy_matrix(_ENERGIES)
    cfg = _cfg_energy()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_HI],
        X, X, None, None, None, None,
        cfg, 5,
        energy_matrix=em,
    )

    # The penalty is purely additive — it can never exclude the only candidate.
    assert err is None, (
        f"beam_failure_reason must be None (never-hard-fail): got {err!r}"
    )
    assert path is not None, "path must not be None when the only candidate exists"
    assert path == [_C_HI]


def test_energy_penalty_disabled_when_matrix_is_none():
    """energy_matrix=None → no energy penalty; both candidates score equally on sonic."""
    X = _flat_space(4)
    cfg = _cfg_energy()

    path, _hits, _edges, err = _beam_search_segment(
        _PIER_A, _PIER_B, 1, [_C_LO, _C_HI],
        X, X, None, None, None, None,
        cfg, 5,
        energy_matrix=None,
    )

    # Without energy matrix the beam should succeed (identical sonic vectors → ties
    # broken by list order or beam; either candidate is valid).
    assert err is None, f"Segment failed unexpectedly: {err}"
    assert path is not None
