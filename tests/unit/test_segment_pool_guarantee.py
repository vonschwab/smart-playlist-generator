"""Unit tests for bridge-side Phase A (2026-07-09): relaxed bridge admission and
on-tag guarantee at segment-pool stage D.

See docs/superpowers/plans/2026-07-09-bridge-side-phase-a.md (Tasks 1-2).
"""
import numpy as np

from src.playlist.segment_pool_builder import SegmentCandidatePoolBuilder, SegmentPoolConfig


def _cfg(**kw):
    # 4 tracks: 0=pier_a, 1=pier_b, 2=near-a-far-from-b, 3=near both
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.98, 0.05], [0.7, 0.7]], dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    class _B:  # minimal bundle stand-in
        track_ids = np.array(["t0", "t1", "t2", "t3"])
    base = dict(
        pier_a=0, pier_b=1, X_full_norm=X, universe_indices=[2, 3],
        used_track_ids=set(), bundle=_B(), bridge_floor=0.30, segment_pool_max=10,
    )
    base.update(kw)
    return SegmentPoolConfig(**base)


def test_relaxed_admits_one_pier_candidate():
    b = SegmentCandidatePoolBuilder()
    # track 2 is near pier_a (~0.98) but far from pier_b (~0.05): min<0.30 fails, max>=0.30 passes
    strict = b._compute_bridge_scores(_cfg(bridge_admission_relaxed=False), [2, 3], {})
    assert 2 not in strict.passing_candidates and 3 in strict.passing_candidates
    relaxed = b._compute_bridge_scores(_cfg(bridge_admission_relaxed=True), [2, 3], {})
    assert 2 in relaxed.passing_candidates and 3 in relaxed.passing_candidates


def test_relaxed_off_path_is_default_and_matches_legacy_min_gate():
    """Steering-gated requirement: with no guarantee/relax knobs set, behavior must
    match the pre-Phase-A legacy min() gate exactly (byte-identical off-path)."""
    b = SegmentCandidatePoolBuilder()
    default_cfg = _cfg()
    assert default_cfg.bridge_admission_relaxed is False
    assert default_cfg.on_tag_guarantee_indices is None
    res = b._compute_bridge_scores(default_cfg, [2, 3], {})
    assert 2 not in res.passing_candidates
    assert 3 in res.passing_candidates


def test_guarantee_forces_track_past_floor_and_into_final():
    b = SegmentCandidatePoolBuilder()
    # pier_a=[1,0], pier_b=[0,1]. track2 is near BOTH piers (best sonic bridge score,
    # passes the strict floor naturally). track3 is near pier_a ONLY: min(sim_a, sim_b)
    # ~0.05 << 0.30, so it fails the STRICT gate outright (bridge_admission_relaxed
    # defaults False here) -- only the on-tag guarantee can admit it.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.99, 0.05]], dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    class _B:
        track_ids = np.array(["t0", "t1", "t2", "t3"])

    cfg = SegmentPoolConfig(
        pier_a=0, pier_b=1, X_full_norm=X, universe_indices=[2, 3], used_track_ids=set(),
        bundle=_B(), bridge_floor=0.30, segment_pool_max=1,   # tiny cap
        on_tag_guarantee_indices={3}, on_tag_guarantee_max=4, on_tag_guarantee_per_artist=4,
    )
    res = b._compute_bridge_scores(cfg, [2, 3], {})
    assert 3 in res.passing_candidates          # forced past the floor despite failing min()
    assert 2 in res.passing_candidates          # passes naturally (near both piers)
    assert res.diagnostics["on_tag_guarantee_forced"] == 1

    # segment_pool_max=1 and track 2 outranks track 3 on bridge score alone -- without
    # the priority-insert fix, the external-fill cap would drop track 3. It must survive.
    result = b.build(cfg)
    assert 3 in result.candidates


def test_guarantee_respects_max_and_per_artist_caps():
    b = SegmentCandidatePoolBuilder()
    # Two guarantee candidates (3, 4) both near pier_a only (fail the strict floor);
    # cap on_tag_guarantee_max=1 should force-include only one of them.
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.99, 0.05], [0.98, 0.04]],
        dtype=np.float64,
    )
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    class _B:
        track_ids = np.array(["t0", "t1", "t2", "t3", "t4"])

    cfg = SegmentPoolConfig(
        pier_a=0, pier_b=1, X_full_norm=X, universe_indices=[2, 3, 4], used_track_ids=set(),
        bundle=_B(), bridge_floor=0.30, segment_pool_max=10,
        on_tag_guarantee_indices={3, 4}, on_tag_guarantee_max=1, on_tag_guarantee_per_artist=4,
    )
    res = b._compute_bridge_scores(cfg, [2, 3, 4], {})
    assert res.diagnostics["on_tag_guarantee_forced"] == 1
    forced = {i for i in (3, 4) if i in res.passing_candidates}
    assert len(forced) == 1


def test_no_guarantee_no_relax_is_inert_by_default():
    """SegmentPoolConfig defaults: bridge_admission_relaxed=False,
    on_tag_guarantee_indices=None, on_tag_guarantee_max=0 -- non-steered runs must
    be unaffected by this feature."""
    cfg = _cfg()
    assert cfg.bridge_admission_relaxed is False
    assert cfg.on_tag_guarantee_indices is None
    assert cfg.on_tag_guarantee_max == 0
    assert cfg.on_tag_guarantee_per_artist == 0
