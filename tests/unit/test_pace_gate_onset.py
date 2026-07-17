import pytest
from src.playlist.pier_bridge.pace_gate import compute_step_log_onset_target


def test_step_onset_target_geometric_midpoint():
    # geometric mean of 1 and 4 is 2
    assert compute_step_log_onset_target(1.0, 4.0, step=1, segment_length=2) == pytest.approx(2.0)
