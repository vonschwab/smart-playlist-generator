import numpy as np
from src.playlist.pier_bridge.pace_gate import compute_step_log_bpm_target


def test_compute_step_log_bpm_target_at_zero_returns_pier_a():
    result = compute_step_log_bpm_target(60.0, 240.0, step=0, segment_length=4)
    np.testing.assert_allclose(result, 60.0, atol=1e-9)


def test_compute_step_log_bpm_target_at_end_returns_pier_b():
    result = compute_step_log_bpm_target(60.0, 240.0, step=4, segment_length=4)
    np.testing.assert_allclose(result, 240.0, atol=1e-9)


def test_compute_step_log_bpm_target_midpoint_geometric_mean():
    # Geometric mean of 60 and 240 is sqrt(60*240) = 120
    result = compute_step_log_bpm_target(60.0, 240.0, step=2, segment_length=4)
    np.testing.assert_allclose(result, 120.0, atol=1e-9)


def test_compute_step_log_bpm_target_zero_length():
    result = compute_step_log_bpm_target(80.0, 120.0, step=0, segment_length=0)
    assert result == 80.0


# filter_candidates_by_bpm_target tests removed (dead code, Phase 0 Task 2,
# 2026-07-16): beam.py never called it — BPM banding is enforced via
# compute_energy_pace_penalty inline in the beam.
