import numpy as np
from src.playlist.pier_bridge.pace_gate import (
    compute_step_log_bpm_target,
    filter_candidates_by_bpm_target,
)


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


def test_filter_candidates_inf_floor_keeps_all():
    perceptual_bpm = np.array([60.0, 120.0, 240.0])
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1, 2],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(3),
        target_bpm=120.0,
        max_log_distance=float("inf"),
        stability_min=0.5,
    )
    assert kept == [0, 1, 2]


def test_filter_candidates_drops_octave_away():
    perceptual_bpm = np.array([60.0, 120.0, 240.0, 100.0])
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1, 2, 3],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(4),
        target_bpm=120.0,
        max_log_distance=0.4,
        stability_min=0.5,
    )
    # 60 and 240 are 1 octave from 120 → dropped (distance 1.0 > 0.4)
    # 100 is log2(120/100) ≈ 0.263 → kept
    assert 1 in kept
    assert 3 in kept
    assert 0 not in kept
    assert 2 not in kept


def test_filter_candidates_low_stability_bypasses():
    perceptual_bpm = np.array([60.0, 240.0])
    tempo_stability = np.array([0.3, 0.9])
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
        target_bpm=120.0,
        max_log_distance=0.4,
        stability_min=0.5,
    )
    assert 0 in kept   # low stability → bypass
    assert 1 not in kept   # high stability + far → drop


def test_filter_candidates_nan_bpm_bypasses():
    perceptual_bpm = np.array([np.nan, 240.0])
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(2),
        target_bpm=120.0,
        max_log_distance=0.4,
        stability_min=0.5,
    )
    assert 0 in kept   # NaN → bypass
    assert 1 not in kept
