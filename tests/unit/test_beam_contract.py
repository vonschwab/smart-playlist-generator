"""Live enforcement of docs/BEAM_CONTRACT.md invariants I1/I2/I4.

Each test below is named for, and cited by, its invariant in
docs/BEAM_CONTRACT.md. Do not weaken an assertion here to force green — these
are lock-in tests: the properties already hold, so a failure means either the
beam regressed or the contract doc is describing behavior that no longer
exists (fix the doc in the same commit, per the contract's own rule).

Fixture reuse:
  - I1 reuses the `_beam_search_segment` direct-call pattern, plus
    `build_transition_metric_context` / `score_transition_edge`, from
    tests/unit/test_transition_metric_alignment.py.
  - I2 and I4 reuse `_make_bundle` and `SMOKE_SCENARIOS` from
    tests/unit/test_pier_bridge_smoke_golden.py.
"""
from __future__ import annotations

import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist
from src.playlist.transition_metrics import (
    build_transition_metric_context,
    score_transition_edge,
)
from tests.unit.test_pier_bridge_smoke_golden import SMOKE_SCENARIOS, _make_bundle


def test_ranking_bonus_does_not_change_edge_T():
    """I1 — the tag-steering ranking bonus never contaminates the reported edge T.

    Runs the same segment twice: once with the tag-steering term fully off,
    once with a large weight + an affinity vector that strongly favors a
    candidate the unweighted run does NOT use (so the bonus demonstrably
    changes candidate selection). In both runs, every realized edge's
    reported "T" must equal both `trans_score_in_beam` and an independently
    recomputed `score_transition_edge` for that edge's endpoints — proving
    the bonus is added to `combined_score` only, never to T.
    """
    rng = np.random.default_rng(11)
    n = 10
    X_full = rng.standard_normal((n, 512))

    ctx = build_transition_metric_context(
        X_sonic=X_full,
        X_start=X_full,
        X_mid=X_full,
        X_end=X_full,
        center_transitions=False,
    )
    cfg = PierBridgeConfig(
        center_transitions=False,
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    pier_a, pier_b = 0, n - 1
    candidates = list(range(1, n - 1))  # 1..8

    def _run(sonic_tag_affinity, sonic_tag_beam_weight):
        out: dict = {}
        path, _hits, _scored, err = _beam_search_segment(
            pier_a,
            pier_b,
            2,
            candidates,
            ctx.X_full,
            ctx.X_sonic_norm,
            ctx.X_start,
            ctx.X_mid,
            ctx.X_end,
            ctx.X_genre_norm,
            cfg,
            10,
            transition_metric_context=ctx,
            edge_components_out=out,
            sonic_tag_affinity=sonic_tag_affinity,
            sonic_tag_beam_weight=sonic_tag_beam_weight,
        )
        assert err is None
        assert path is not None
        return path, out

    path_off, out_off = _run(None, 0.0)

    # Bias heavily toward a candidate the unweighted run left unused, so the
    # weighted run's chosen PATH is expected to differ (that's fine/expected
    # per the contract — only T must stay untouched).
    off_path_candidates = [c for c in candidates if c not in path_off]
    assert off_path_candidates, "fixture must leave >=1 candidate unused by the baseline path"
    favored = off_path_candidates[0]
    affinity = np.zeros(n, dtype=float)
    affinity[favored] = 10.0
    path_on, out_on = _run(affinity, 5.0)

    # Sanity check: the bonus actually influenced selection (not a no-op fixture).
    assert favored in path_on

    for path, out in ((path_off, out_off), (path_on, out_on)):
        components = out["components"]
        assert len(components) == len(path) + 1  # pier_a->...->pier_b edges
        for edge in components:
            oracle = score_transition_edge(ctx, edge["from_idx"], edge["to_idx"])
            assert edge["T"] == edge["trans_score_in_beam"]
            assert edge["T"] == oracle["T"]


def test_tag_steering_beam_off_is_byte_identical():
    """I2 — the tag-steering beam params are true no-ops when off.

    Builds the same segment three ways through `build_pier_bridge_playlist`
    with an explicit (non-config.yaml) `PierBridgeConfig`: (a) no tag kwargs
    passed at all, (b) affinity=None with a nonzero weight, (c) a nonzero
    affinity vector with weight=0.0. All three must be byte-identical in both
    the emitted track order and every edge's reported T.
    """
    bundle = _make_bundle(n=50, sonic_dim=16, genre_dim=8, num_artists=10)
    scenario = SMOKE_SCENARIOS["two_seeds_default"]
    # Same cascade pinning as the smoke-golden baseline (core beam, not the
    # opt-in var-bridge/edge-delete passes) -- irrelevant to I2 itself since
    # cfg is identical across all three calls, but keeps the fixture aligned
    # with its source.
    cfg = PierBridgeConfig(
        variable_bridge_length=False,
        edge_delete_enabled=False,
        **scenario["cfg_kwargs"],
    )

    seed_ids = scenario["seed_track_ids"]
    seed_idx_set = {bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(bundle.track_ids)) if i not in seed_idx_set]

    affinity = np.zeros(len(bundle.track_ids), dtype=float)
    affinity[candidate_pool[:5]] = 1.0  # arbitrary nonzero vector; must be inert at weight 0.0

    def _build(extra_kwargs):
        return build_pier_bridge_playlist(
            seed_track_ids=seed_ids,
            total_tracks=scenario["total_tracks"],
            bundle=bundle,
            candidate_pool_indices=candidate_pool,
            cfg=cfg,
            min_genre_similarity=None,
            X_genre_smoothed=None,
            **extra_kwargs,
        )

    result_a = _build({})  # (a) no tag params passed at all
    result_b = _build({"sonic_tag_affinity": None, "sonic_tag_beam_weight": 0.15})  # (b)
    result_c = _build({"sonic_tag_affinity": affinity, "sonic_tag_beam_weight": 0.0})  # (c)

    assert result_a.success and result_b.success and result_c.success
    assert result_a.track_ids == result_b.track_ids
    assert result_a.track_ids == result_c.track_ids

    edges_a = result_a.stats["edge_scores"]
    edges_b = result_b.stats["edge_scores"]
    edges_c = result_c.stats["edge_scores"]
    assert len(edges_a) == len(edges_b) == len(edges_c) > 0
    ts_a = [e["T"] for e in edges_a]
    ts_b = [e["T"] for e in edges_b]
    ts_c = [e["T"] for e in edges_c]
    assert ts_a == ts_b
    assert ts_a == ts_c


def test_piers_preserved_through_cascade():
    """I4 — piers are never removed or reordered by the post-beam cascade.

    Runs with edge-repair and edge-delete both ON (the two cascade stages
    the contract calls out as pier-touching risks) and asserts the seed
    track IDs appear, in their original relative order, at
    `PierBridgeResult.seed_positions`. Holds whether or not the cascade
    actually fires any repair/delete on this fixture's edges.
    """
    bundle = _make_bundle(n=50, sonic_dim=16, genre_dim=8, num_artists=10)
    scenario = SMOKE_SCENARIOS["three_seeds_centered"]
    cfg = PierBridgeConfig(
        edge_repair_enabled=True,
        edge_delete_enabled=True,
        edge_delete_max_deletions=4,
        **scenario["cfg_kwargs"],
    )

    seed_ids = scenario["seed_track_ids"]
    seed_idx_set = {bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(bundle.track_ids)) if i not in seed_idx_set]

    result = build_pier_bridge_playlist(
        seed_track_ids=seed_ids,
        total_tracks=scenario["total_tracks"],
        bundle=bundle,
        candidate_pool_indices=candidate_pool,
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=None,
    )

    assert result.success
    seed_positions = result.seed_positions
    assert len(seed_positions) == len(seed_ids)
    assert seed_positions == sorted(seed_positions)  # strictly ascending -> never reordered
    found_ids_in_order = [result.track_ids[p] for p in seed_positions]
    assert found_ids_in_order == seed_ids
