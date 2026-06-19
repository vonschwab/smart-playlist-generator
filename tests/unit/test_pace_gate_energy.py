import numpy as np
from src.playlist.pier_bridge.pace_gate import (
    compute_step_energy_target,
    compute_energy_pace_penalty,
)


def test_step_energy_target_linear_midpoint():
    t = compute_step_energy_target(np.array([0.0, 0.0]), np.array([4.0, 2.0]), step=1, segment_length=2)
    assert np.allclose(t, [2.0, 1.0])


def _em(rows):
    return np.array(rows, dtype=float)


def test_step_cap_fires_above_cap_only():
    # current row0, cand row1 distance = 2.0 ; cap 1.0, strength 0.5 -> 0.5*(2-1)=0.5
    em = _em([[0.0], [2.0], [0.0], [0.0]])
    pen = compute_energy_pace_penalty(em, current=0, cand=1, pier_a=2, pier_b=3,
                                      step=0, segment_length=1, step_cap=1.0,
                                      step_strength=0.5, arc_band=99.0, arc_strength=0.0)
    assert abs(pen - 0.5) < 1e-9
    # within cap -> no penalty
    pen2 = compute_energy_pace_penalty(em, current=0, cand=2, pier_a=2, pier_b=3,
                                       step=0, segment_length=1, step_cap=1.0,
                                       step_strength=0.5, arc_band=99.0, arc_strength=0.0)
    assert pen2 == 0.0


def test_arc_band_penalizes_distance_from_target():
    # piers 0 and 4 ; step0/len1 target=0 ; cand=4 -> arc dist 4 ; band 1, strength 0.5 -> 1.5
    em = _em([[0.0], [4.0], [0.0], [4.0]])  # pier_a=row0(0), pier_b=row3(4)? use explicit rows
    em = _em([[0.0], [4.0]])
    pen = compute_energy_pace_penalty(em, current=0, cand=1, pier_a=0, pier_b=1,
                                      step=0, segment_length=2, step_cap=99.0,
                                      step_strength=0.0, arc_band=1.0, arc_strength=0.5)
    # target at step0 = pier_a (0); cand=row1=4 -> arc dist 4 -> 0.5*(4-1)=1.5
    assert abs(pen - 1.5) < 1e-9


def test_nan_and_none_are_zero_never_raise():
    assert compute_energy_pace_penalty(None, current=0, cand=1, pier_a=0, pier_b=1,
                                       step=0, segment_length=1, step_cap=0.1,
                                       step_strength=1.0, arc_band=0.1, arc_strength=1.0) == 0.0
    em = _em([[np.nan], [2.0]])
    assert compute_energy_pace_penalty(em, current=0, cand=1, pier_a=0, pier_b=1,
                                       step=0, segment_length=1, step_cap=0.1,
                                       step_strength=1.0, arc_band=0.1, arc_strength=1.0) == 0.0
