"""Tail-DP endgame (spec 2026-07-02): exact max-min re-optimization of the
last 2 interior slots per segment. Pure module tests."""
import numpy as np
import pytest

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


@pytest.mark.parametrize("centered", [False, True])
def test_batch_T_matches_score_transition_edge_both_branches(centered):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 4))
    ctx = build_transition_metric_context(
        X_sonic=X, X_start=X, X_mid=X, X_end=X,
        X_genre=np.eye(6), center_transitions=centered,
    )
    M = batch_T(ctx, [0, 1, 2], [3, 4, 5])
    for i, a in enumerate([0, 1, 2]):
        for j, b in enumerate([3, 4, 5]):
            assert M[i, j] == pytest.approx(score_transition_edge(ctx, a, b)["T"], abs=1e-9)


@pytest.mark.parametrize("centered", [False, True])
def test_batch_T_mixed_none_start_matches_reference(centered):
    # X_end present, X_start ABSENT: reference falls back to full-full for the
    # end->start component; batch_T must do the same (review catch, 2026-07-02).
    # X_end is deliberately DISTINCT from X_sonic here -- using the same array
    # for both (as in the original review snippet) makes ctx.X_end == ctx.X_full
    # after identical normalization, which coincidentally makes the buggy
    # independent-substitution formula (X_end @ X_full.T) equal the correct
    # full-full fallback (X_full @ X_full.T), so the test would never go red.
    rng = np.random.default_rng(2)
    X_sonic = rng.standard_normal((5, 4))
    X_end = rng.standard_normal((5, 4))
    ctx = build_transition_metric_context(
        X_sonic=X_sonic, X_start=None, X_mid=X_sonic, X_end=X_end,
        X_genre=np.eye(5), center_transitions=centered,
    )
    M = batch_T(ctx, [0, 1], [2, 3])
    for i, a in enumerate([0, 1]):
        for j, b in enumerate([2, 3]):
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


def test_floor_gate_skips_already_good_landing():
    # existing tail [4,5]=C90 has a strong window min (~0.9, clearly >= the
    # 0.30 floor); candidates [2,3] are near-perfectly aligned (C99) and WOULD
    # win the re-opt on merit (0.99 > 0.9+epsilon) if the gate were absent --
    # the floor must suppress that otherwise-available swap.
    C99 = [0.99, (1 - 0.99 ** 2) ** 0.5]
    X = [[1, 0], [1, 0], C99, C99, C90, C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[4, 5], pier_a=0, pier_b=1,
        candidates=[2, 3], epsilon=0.02, floor=0.30, is_allowed_pair=lambda x, y: True,
    )
    assert res is None


def test_floor_gate_fires_on_weak_landing():
    # existing tail [2,3] orthogonal (window min ~0) < floor 0.30 => re-opt to C90 pair
    X = [[1, 0], [1, 0], [0, 1], [0, 1], C90, C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02, floor=0.30, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None and set(res.new_tail) == {4, 5}


def test_floor_gate_skips_already_good_landing_window_one():
    # single-interior-slot segment (window==1): existing slot=C90 already
    # clears the 0.30 floor (both prefix->slot and slot->pier_b edges land at
    # ~0.9 since pier_a and pier_b share the same [1,0] vector) -- even though
    # a strictly better candidate (C99) is available and would win on merit
    # (0.99 > 0.9+epsilon) if the gate were absent, the floor must suppress
    # the re-opt and return None.
    C99 = [0.99, (1 - 0.99 ** 2) ** 0.5]
    X = [[1, 0], [1, 0], C90, C99]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2], pier_a=0, pier_b=1,
        candidates=[3], epsilon=0.02, floor=0.30, is_allowed_pair=lambda x, y: True,
    )
    assert res is None


def test_floor_gate_fires_on_weak_landing_window_one():
    # single-interior-slot segment (window==1): existing slot [0,1] is
    # orthogonal to both pier_a and pier_b ([1,0]) -> window min ~0, below the
    # 0.30 floor -> re-opt swaps in the strong C90 candidate.
    X = [[1, 0], [1, 0], [0, 1], C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2], pier_a=0, pier_b=1,
        candidates=[3], epsilon=0.02, floor=0.30, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None and res.new_tail == (3,)


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


def test_tail_dp_knobs_default_and_override():
    from src.playlist.config import default_ds_config
    from src.playlist.pier_bridge.config import PierBridgeConfig
    from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

    cfg = PierBridgeConfig()
    assert cfg.tail_dp_enabled is True
    assert cfg.tail_dp_epsilon == 0.02
    assert cfg.tail_dp_floor == 0.30
    assert cfg.tail_dp_relative_epsilon == 0.25

    # mirror the invocation shape used by test_edge_repair_break_glass.py's
    # knob test (the real apply_pier_bridge_overrides signature).
    pb_cfg, _tuning, _sources = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides={},
        pb_overrides={
            "tail_dp": {
                "enabled": False, "epsilon": 0.05, "floor": 0.1, "relative_epsilon": 0.4,
            },
        },
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
    )

    assert pb_cfg.tail_dp_enabled is False
    assert pb_cfg.tail_dp_epsilon == 0.05
    assert pb_cfg.tail_dp_floor == 0.1
    assert pb_cfg.tail_dp_relative_epsilon == 0.4
