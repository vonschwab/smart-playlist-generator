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
    # scenario["cfg_kwargs"] (bridge_floor=0.0, transition_floor=0.0,
    # center_transitions=False) is the smoke-golden scenario's own non-default
    # PierBridgeConfig -- borrowed as-is, not hand-picked, and held IDENTICAL
    # across all three variants below so only the tag params vary. This keeps
    # us off an explicit (non-config.yaml) PierBridgeConfig without
    # reintroducing the config.yaml-load confound the doc's I2 caveat warns
    # against.
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

    Isolates the cascade from `_order_seeds_by_bridgeability` (the pre-beam
    subsystem that decides pier order and is out of scope for I4 — it has
    changed twice this week). Runs the SAME seeds through the SAME
    scenario/bundle TWICE with identical config except the two cascade
    stages the contract calls out as pier-touching risks
    (docs/BEAM_CONTRACT.md I4): cascade OFF (edge_repair_enabled=False,
    edge_delete_enabled=False) vs cascade ON (edge_repair_enabled=True,
    edge_delete_enabled=True, edge_delete_max_deletions=4). Seed ordering
    runs identically upstream of the beam in both calls, so any divergence
    in the SUBSEQUENCE of seed IDs read off `seed_positions` can only be
    attributed to the cascade itself — never to whatever
    `_order_seeds_by_bridgeability` decided.

    Note: the raw `seed_positions` INDEX LISTS are allowed to differ between
    the two runs (I3 sanctions edge-delete as a remove-only fix on non-pier
    interior tracks -- deleting an interior track legitimately shortens the
    emitted list and shifts every later pier's absolute array index left).
    That is not a pier reordering; what I4 actually guards is that the same
    seed IDENTITIES occupy the same relative pier slots (1st, 2nd, 3rd...)
    in the same order, which is exactly what comparing the ID subsequence
    (not the index subsequence) below verifies.
    """
    bundle = _make_bundle(n=50, sonic_dim=16, genre_dim=8, num_artists=10)
    scenario = SMOKE_SCENARIOS["three_seeds_centered"]

    seed_ids = scenario["seed_track_ids"]
    seed_idx_set = {bundle.track_id_to_index[s] for s in seed_ids}
    candidate_pool = [i for i in range(len(bundle.track_ids)) if i not in seed_idx_set]

    def _run(edge_repair_enabled, edge_delete_enabled, edge_delete_max_deletions):
        cfg = PierBridgeConfig(
            edge_repair_enabled=edge_repair_enabled,
            edge_delete_enabled=edge_delete_enabled,
            edge_delete_max_deletions=edge_delete_max_deletions,
            **scenario["cfg_kwargs"],
        )
        return build_pier_bridge_playlist(
            seed_track_ids=seed_ids,
            total_tracks=scenario["total_tracks"],
            bundle=bundle,
            candidate_pool_indices=candidate_pool,
            cfg=cfg,
            min_genre_similarity=None,
            X_genre_smoothed=None,
        )

    result_cascade_off = _run(False, False, 0)
    result_cascade_on = _run(True, True, 4)

    assert result_cascade_off.success
    assert result_cascade_on.success

    ids_off = [result_cascade_off.track_ids[p] for p in result_cascade_off.seed_positions]
    ids_on = [result_cascade_on.track_ids[p] for p in result_cascade_on.seed_positions]

    # Cascade neither removed nor added a seed.
    assert len(ids_off) == len(seed_ids)
    assert len(ids_on) == len(seed_ids)
    assert set(ids_off) == set(seed_ids)
    assert set(ids_on) == set(seed_ids)

    # Cascade neither reordered nor relocated any pier: the subsequence of
    # seed IDs read off each run's own `seed_positions` is identical, i.e.
    # the same seed sits in the same relative pier slot in both runs (seed
    # ordering itself is held fixed across both calls, so this isolates the
    # cascade's effect specifically -- see the docstring note on why we do
    # NOT compare the raw index lists).
    assert ids_off == ids_on
