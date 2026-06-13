import numpy as np
import pytest
from src.playlist.pier_bridge.pace_gate import (
    compute_step_log_onset_target,
    filter_candidates_by_onset_target,
)


def test_step_onset_target_geometric_midpoint():
    # geometric mean of 1 and 4 is 2
    assert compute_step_log_onset_target(1.0, 4.0, step=1, segment_length=2) == pytest.approx(2.0)


def test_filter_rejects_beyond_cap_keeps_nan():
    onset = np.array([2.0, 8.0, np.nan])  # target=2.0, cap=0.6 log2 (~1.5x)
    kept = filter_candidates_by_onset_target(
        candidate_indices=[0, 1, 2], onset_rate=onset, target_onset=2.0, max_log_distance=0.6,
    )
    assert kept == [0, 2]  # idx1 (8.0, 2 octaves away) rejected; idx2 NaN bypassed
