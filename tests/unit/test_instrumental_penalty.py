import numpy as np
from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty


def test_penalty_proportional_to_voice_prob():
    vp = np.array([0.95, 0.55, 0.0])
    assert abs(compute_instrumental_penalty(vp, cand=0, weight=0.6) - 0.57) < 1e-6
    assert abs(compute_instrumental_penalty(vp, cand=1, weight=0.6) - 0.33) < 1e-6
    assert compute_instrumental_penalty(vp, cand=2, weight=0.6) == 0.0


def test_penalty_nan_and_disabled_are_zero():
    vp = np.array([np.nan, 0.9])
    assert compute_instrumental_penalty(vp, cand=0, weight=0.6) == 0.0   # NaN -> unpunished
    assert compute_instrumental_penalty(vp, cand=1, weight=0.0) == 0.0   # weight 0 -> off
    assert compute_instrumental_penalty(None, cand=1, weight=0.6) == 0.0  # no data -> off
