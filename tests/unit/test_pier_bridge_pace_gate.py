import numpy as np

from src.playlist.pier_bridge.pace_gate import (
    compute_step_rhythm_target,
    filter_candidates_by_rhythm_target,
)


def test_target_at_step_zero_equals_pier_a():
    R_a = np.array([1.0, 0.0])
    R_b = np.array([0.0, 1.0])

    target = compute_step_rhythm_target(R_a, R_b, step=0, segment_length=4)

    np.testing.assert_allclose(target, R_a)


def test_target_at_final_step_equals_pier_b():
    R_a = np.array([1.0, 0.0])
    R_b = np.array([0.0, 1.0])

    target = compute_step_rhythm_target(R_a, R_b, step=4, segment_length=4)

    np.testing.assert_allclose(target, R_b)


def test_target_midpoint_interpolates():
    R_a = np.array([1.0, 0.0])
    R_b = np.array([0.0, 1.0])

    target = compute_step_rhythm_target(R_a, R_b, step=2, segment_length=4)

    np.testing.assert_allclose(target, [0.5, 0.5])


def test_anchored_when_piers_share_rhythm():
    R_a = np.array([1.0, 0.0])
    R_b = np.array([1.0, 0.0])

    for step in range(5):
        target = compute_step_rhythm_target(R_a, R_b, step=step, segment_length=4)
        np.testing.assert_allclose(target, R_a)


def test_filter_drops_candidates_below_floor():
    R = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    kept = filter_candidates_by_rhythm_target(
        candidate_indices=[0, 1, 2],
        rhythm_matrix=R,
        target=np.array([1.0, 0.0]),
        floor=0.50,
    )

    assert kept == [0]


def test_filter_passes_all_when_floor_zero():
    R = np.random.default_rng(0).standard_normal((5, 4))

    kept = filter_candidates_by_rhythm_target(
        candidate_indices=[0, 1, 2, 3, 4],
        rhythm_matrix=R,
        target=R[0],
        floor=0.0,
    )

    assert kept == [0, 1, 2, 3, 4]
