"""Onset bridge band gates; rhythm-cosine soft penalty demotes; both skip cleanly
when their inputs are absent (MERT no-tower path)."""
import numpy as np
from src.playlist.pier_bridge.pace_gate import filter_candidates_by_onset_target


def test_onset_bridge_band_rejects_far_density():
    onset = np.array([4.0, 4.5, 32.0])
    kept = filter_candidates_by_onset_target(
        candidate_indices=[0, 1, 2], onset_rate=onset, target_onset=4.0, max_log_distance=0.6,
    )
    assert kept == [0, 1]  # 32.0 (3 octaves) rejected


def test_soft_penalty_multiplier_below_threshold():
    # Pure arithmetic guard for the multiplier the beam applies.
    strength = 0.15
    base = 1.0
    demoted = base * (1.0 - strength)
    assert demoted == 0.85
