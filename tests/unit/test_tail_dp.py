"""Tail-DP endgame (spec 2026-07-02): exact max-min re-optimization of the
last 2 interior slots per segment. Pure module tests."""
from pathlib import Path

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.tail_dp import batch_T, optimize_segment_tail
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge

C25 = [0.25, 0.9682458365518543]
C90 = [0.90, 0.4358898943540674]


def _ctx(X):
    X_arr = np.array(X, dtype=float)
    return build_transition_metric_context(
        X_sonic=X_arr, X_start=X_arr, X_mid=X_arr, X_end=X_arr,
        X_genre=np.eye(X_arr.shape[0]), center_transitions=False,
    )


def test_batch_T_matches_score_transition_edge():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    ctx = _ctx(X)
    M = batch_T(ctx, [0, 1, 2], [3, 4, 5])
    for i, a in enumerate([0, 1, 2]):
        for j, b in enumerate([3, 4, 5]):
            assert M[i, j] == pytest.approx(score_transition_edge(ctx, a, b)["T"], abs=1e-9)


def test_two_slot_swap_improves_min():
    # piers 0/1 = [1,0]; existing tail [2,3] orthogonal (min ~ 0);
    # candidates 4,5 = C90 both -> window min jumps to ~0.9-ish.
    X = [[1, 0], [1, 0], [0, 1], [0, 1], C90, C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None
    assert set(res.new_tail) == {4, 5}
    assert res.new_min > res.old_min + 0.02


def test_never_worse_returns_none():
    # candidates are WORSE than the existing decent tail -> None.
    X = [[1, 0], [1, 0], C90, C90, [0, 1], [0, 1]]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is None


def test_disallowed_pairs_are_skipped_for_next_best():
    # best pair (4,5) blocked by the callback -> falls back to (5,4) or mixed.
    X = [[1, 0], [1, 0], [0, 1], [0, 1], C90, C90]
    ctx = _ctx(X)
    blocked = {(4, 5)}
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02,
        is_allowed_pair=lambda x, y: (x, y) not in blocked,
    )
    assert res is not None
    assert tuple(res.new_tail) != (4, 5)


def test_one_slot_window():
    # single-interior segment: replace the lone slot.
    X = [[1, 0], [1, 0], [0, 1], C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2], pier_a=0, pier_b=1,
        candidates=[3], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None and res.new_tail == (3,)


def test_empty_path_and_no_candidates_noop():
    X = [[1, 0], [1, 0], [0, 1]]
    ctx = _ctx(X)
    assert optimize_segment_tail(ctx, segment_path=[], pier_a=0, pier_b=1,
                                 candidates=[2], epsilon=0.02,
                                 is_allowed_pair=lambda x, y: True) is None
    assert optimize_segment_tail(ctx, segment_path=[2], pier_a=0, pier_b=1,
                                 candidates=[], epsilon=0.02,
                                 is_allowed_pair=lambda x, y: True) is None
