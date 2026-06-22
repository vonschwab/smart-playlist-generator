"""Unit tests for adaptive (percentile) sonic admission floor.

Task 1: sonic_admission_percentile replaces fixed min_sonic_similarity
when set and > 0.  When None / 0 -> legacy absolute-floor behavior unchanged.
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace as _replace
from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig


def _toy(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    track_ids = np.array([f"t{i}" for i in range(n)])
    artist_keys = np.array([f"a{i}" for i in range(n)])
    return X_sonic, track_ids, artist_keys


def _base_cfg(**overrides):
    """Minimal CandidateConfig with sensible defaults for unit tests."""
    defaults = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=0.99,  # absolute floor would reject almost all
        max_pool_size=10_000,
        target_artists=10_000,
        candidates_per_artist=10_000,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
    )
    defaults.update(overrides)
    return CandidateConfig(**defaults)


def test_sonic_percentile_admits_top_fraction():
    """With sonic_admission_percentile=0.80, ~top 20% by sonic sim to the seed
    are admitted, regardless of any absolute floor.  Adapts to the distribution.
    """
    X_sonic, track_ids, artist_keys = _toy()
    cfg = _base_cfg(sonic_admission_percentile=0.80)
    res = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0],
        embedding=X_sonic,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
    )
    n_admitted = len(res.pool_indices)
    # ~20% of 59 non-seed ≈ 12; assert it's in a sane band and NOT gutted by the 0.99 floor
    assert 6 <= n_admitted <= 20, (
        f"Expected 6-20 admitted (top ~20% of 59), got {n_admitted}. "
        "Likely the absolute floor (0.99) was applied instead of the percentile."
    )


def test_sonic_percentile_none_uses_absolute_floor():
    """Default (no percentile) respects the absolute floor: a tighter floor admits strictly fewer."""
    X_sonic, track_ids, artist_keys = _toy()
    # loose: floor=-1.0 -> admits essentially everything
    loose = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0],
        embedding=X_sonic,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=_base_cfg(min_sonic_similarity=-1.0, sonic_admission_percentile=None),
        random_seed=0,
        X_sonic=X_sonic,
    )
    # tight: floor=0.3 -> admits only tracks with cosine sim >= 0.3 to the seed
    tight = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0],
        embedding=X_sonic,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=_base_cfg(min_sonic_similarity=0.3, sonic_admission_percentile=None),
        random_seed=0,
        X_sonic=X_sonic,
    )
    assert len(tight.pool_indices) < len(loose.pool_indices), (
        f"Expected tighter absolute floor (0.3) to admit fewer than loose (-1.0), "
        f"got tight={len(tight.pool_indices)} loose={len(loose.pool_indices)}. "
        "Likely the absolute floor is not operative when sonic_admission_percentile=None."
    )


def test_genre_percentile_runs_without_dense():
    """genre_admission_percentile must compute an effective genre floor from the SPARSE
    genre vectors (X_genre_dense=None), not fall back to the absolute min_genre_similarity.

    When active, res.stats['effective_genre_floor'] must be set (data-derived, not None),
    and the pool must be non-empty.
    """
    n = 60
    rng = np.random.default_rng(1)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    # sparse genre: 5-dim one-hot
    X_genre = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        X_genre[i, rng.integers(0, 5)] = 1.0
    tids = [f"t{i}" for i in range(n)]
    aks = [f"a{i}" for i in range(n)]
    cfg = _base_cfg(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        duration_penalty_enabled=False,
    )
    res = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0],
        embedding=X_sonic,
        artist_keys=np.array(aks),
        track_ids=np.array(tids),
        cfg=_replace(cfg, sonic_admission_percentile=None),
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        X_genre_dense=None,
        min_genre_similarity=0.4,
        genre_admission_percentile=0.50,
    )
    # With percentile active on sparse vectors, the effective floor is data-derived, NOT the abs 0.4.
    # Assert the pool reflects percentile admission (not the degenerate all-or-nothing of abs 0.4).
    assert len(res.pool_indices) > 0, "Pool must be non-empty with genre_admission_percentile=0.50"
    assert res.stats.get("effective_genre_floor") is not None, (
        "res.stats['effective_genre_floor'] must be set when genre_admission_percentile is active. "
        "Got None — likely the percentile was not applied on the sparse path."
    )
    # Strengthen: the percentile path must have derived the floor from the data, not passed through
    # the fixed 0.4.  With 5-dim one-hot, genre sims are 0.0 or 1.0; p50 of a mostly-zero
    # distribution is 0.0 — never 0.4 — so this is always robust.
    assert res.stats["effective_genre_floor"] != 0.4, (
        f"effective_genre_floor is exactly 0.4 — the percentile path was bypassed and the "
        f"legacy absolute floor was used instead. Got {res.stats['effective_genre_floor']!r}"
    )


def test_genre_percentile_runs_without_absolute_floor():
    """genre_admission_percentile fires on the SPARSE path when min_genre_similarity=None
    (percentile-primary mode — no absolute floor companion).

    Before the fix, the sparse elif guard checked ``min_genre_similarity is not None``,
    so this call silently skipped genre admission entirely.  After the fix, the guard
    uses OR semantics and the branch runs.

    Assertions:
    - res.stats['effective_genre_floor'] is not None (data-derived floor was computed)
    - fewer than all non-seed tracks were admitted (the floor actually gated something)
    """
    n = 60
    rng = np.random.default_rng(2)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    # Continuous genre vectors: random uniform in [0, 1].
    # Cosine sims to the seed will be spread across (0, 1); p50 > 0 so the floor
    # actually cuts the lower half of candidates.
    X_genre = rng.random(size=(n, 10)).astype(np.float64)
    tids = np.array([f"t{i}" for i in range(n)])
    aks = np.array([f"a{i}" for i in range(n)])
    cfg = _base_cfg(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        duration_penalty_enabled=False,
    )
    res = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0],
        embedding=X_sonic,
        artist_keys=aks,
        track_ids=tids,
        cfg=_replace(cfg, sonic_admission_percentile=None),
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        X_genre_dense=None,
        min_genre_similarity=None,  # no absolute floor — percentile is the sole gate
        genre_admission_percentile=0.50,
    )
    # The percentile block must have produced a real floor.
    assert res.stats.get("effective_genre_floor") is not None, (
        "res.stats['effective_genre_floor'] must not be None when genre_admission_percentile=0.50 "
        "and min_genre_similarity=None. Likely the sparse elif guard still blocks entry."
    )
    # The floor must have actually gated candidates: fewer than all 59 non-seed tracks admitted.
    n_non_seed = n - 1  # seed_idx=0 excluded
    assert len(res.pool_indices) < n_non_seed, (
        f"Expected percentile-floor (p=0.50) to admit fewer than all {n_non_seed} non-seed tracks, "
        f"but got {len(res.pool_indices)}. The floor may not have been applied."
    )


def test_min_pool_guarantee_never_starves():
    """With a very tight percentile (0.98) that admits only ~2% of candidates,
    min_pool_size=15 must backfill using top sonic-sim, respecting per-artist cap.
    """
    import numpy as np
    from dataclasses import replace as _replace
    from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig
    n = 40
    rng = np.random.default_rng(2)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    tids = [f"t{i}" for i in range(n)]
    aks = [f"a{i%6}" for i in range(n)]  # 6 distinct artists
    # Tight percentile would admit very few; min_pool_size must backfill.
    cfg = CandidateConfig(similarity_floor=-1.0, min_sonic_similarity=None,
                          max_pool_size=10_000, target_artists=10_000,
                          candidates_per_artist=10_000, seed_artist_bonus=0,
                          max_artist_fraction_final=1.0, duration_penalty_enabled=False)
    res = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=X_sonic, artist_keys=np.array(aks),
        track_ids=np.array(tids),
        cfg=_replace(cfg, sonic_admission_percentile=0.98, min_pool_size=15),
        random_seed=0,
        X_sonic=X_sonic,
    )
    assert len(res.pool_indices) >= 15, (
        f"Expected min_pool_size backstop to ensure >= 15 pool entries, got {len(res.pool_indices)}"
    )
    # seeds must never be admitted by the backstop
    assert 0 not in res.pool_indices, (
        f"Seed index 0 must not appear in pool_indices (never-admit-seeds invariant), "
        f"but pool contains: {res.pool_indices[:20]!r}"
    )
    # diversity respected: not a single-artist pool
    pooled_artists = {aks[i] for i in res.pool_indices}
    assert len(pooled_artists) >= 2, (
        f"Expected at least 2 distinct artists in pool, got {pooled_artists!r}"
    )
