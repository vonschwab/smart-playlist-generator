"""Unit tests for the pure corridor builder (Phase 1 Task 1).

Spec: docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md §1.
Membership = min(sim_a, sim_b) >= quantile(min_sims, width_percentile), self-
calibrating per anchor pair. Ranking = harmonic mean of (sim_a, sim_b), copied
verbatim from src/playlist/segment_pool_builder.py's bridge-scoring math.
"""

import numpy as np
import pytest

from src.playlist.pier_bridge.corridor import build_corridor


def _norm(v):
    return v / np.linalg.norm(v)


def _mk_universe(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_membership_is_min_sim_percentile():
    X = _mk_universe()
    a, b = X[0], X[1]
    r = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.90, segment_pool_max=1000)
    sims = np.minimum(X @ a, X @ b)
    assert r.threshold == pytest.approx(np.quantile(sims, 0.90))
    assert set(r.indices) == set(np.arange(200)[sims >= r.threshold])


def test_wider_is_superset_of_narrower():
    X = _mk_universe()
    kw = dict(vec_a=X[0], vec_b=X[1], X_norm=X, universe_indices=np.arange(200),
              segment_pool_max=1000)
    narrow = build_corridor(width_percentile=0.95, **kw)
    wide = build_corridor(width_percentile=0.80, **kw)
    assert set(narrow.indices) <= set(wide.indices)


def test_identical_anchors_degenerates_to_knn_ball():
    X = _mk_universe()
    r = build_corridor(vec_a=X[3], vec_b=X[3], X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.90, segment_pool_max=1000)
    sims = X @ X[3]
    assert set(r.indices) == set(np.arange(200)[sims >= np.quantile(sims, 0.90)])


def test_degenerate_zero_vector_rows_never_admitted_by_rank():
    X = _mk_universe()
    X[5] = 0.0  # MuQ quiet-audio collapse class (pre-existing, parked)
    r = build_corridor(vec_a=_norm(np.ones(8)), vec_b=_norm(-np.ones(8)), X_norm=X,
                       universe_indices=np.arange(200), width_percentile=0.50,
                       segment_pool_max=1000)
    if 5 in r.indices:
        assert r.rank_scores[list(r.indices).index(5)] == 0.0  # hmean guard, no NaN


def test_empty_universe_returns_empty():
    r = build_corridor(vec_a=_norm(np.ones(8)), vec_b=_norm(np.ones(8)),
                       X_norm=np.zeros((0, 8)), universe_indices=np.arange(0),
                       width_percentile=0.90, segment_pool_max=100)
    assert len(r.indices) == 0 and not r.capped


def test_cap_keeps_top_ranked_and_sets_flag():
    X = _mk_universe()
    r = build_corridor(vec_a=X[0], vec_b=X[1], X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.0, segment_pool_max=10)
    assert len(r.indices) == 10 and r.capped
    assert list(r.rank_scores) == sorted(r.rank_scores, reverse=True)


def test_force_include_admitted_below_threshold():
    X = _mk_universe()
    a, b = X[0], X[1]
    sims = np.minimum(X @ a, X @ b)
    outsider = int(np.argmin(sims))
    r = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.95, segment_pool_max=1000,
                       force_include=np.array([outsider]))
    assert outsider in r.indices


def test_anchor_support_matches_manual():
    X = _mk_universe()
    r = build_corridor(vec_a=X[0], vec_b=X[1], X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.90, segment_pool_max=1000)
    top100_a = np.argsort(X @ X[0])[::-1][:100]
    manual = float(np.mean(np.minimum(X @ X[0], X @ X[1])[top100_a] >= r.threshold))
    assert r.stats["anchor_support_a"] == pytest.approx(manual)


# --- Review-fix regression tests (2026-07-16) --------------------------------


def _safe_hmean(a, b):
    """Independent (non-module) reimplementation for reference-fidelity checks."""
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float64)
    mask = denom > 1e-9
    out[mask] = (2.0 * a[mask] * b[mask]) / denom[mask]
    return out


def test_force_include_below_cap_cutoff_survives_truncation():
    """CRITICAL regression: a force-included candidate that passes the
    membership threshold but ranks well below the segment_pool_max cutoff
    must still appear in the output -- it must never land in neither the
    truncated ranked list nor the forced list (corridor.py review, 2026-07-16).
    """
    X = _mk_universe()
    a, b = X[0], X[1]
    sim_a = X @ a
    sim_b = X @ b
    hmean = _safe_hmean(sim_a, sim_b)
    order = np.argsort(-hmean, kind="stable")
    # width_percentile=0.0 admits the whole universe, so this candidate is a
    # genuine threshold-passer -- just ranked far below the cap.
    below_cutoff = int(order[50])

    r = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.0, segment_pool_max=10,
                       force_include=np.array([below_cutoff]))

    assert below_cutoff in r.indices
    assert len(r.indices) == 11  # 10 ranked (capped) + 1 forced, deduped
    assert r.capped is True


def test_capped_flag_not_set_by_force_include_overflow():
    """Important 3: force_include pushing total size past segment_pool_max
    must not set capped=True -- that flag reflects truncation of the ranked
    (non-forced) portion only, matching the reference's guaranteed-first
    selection (segment_pool_builder.py:1020-1116).
    """
    X = _mk_universe()
    a, b = X[0], X[1]
    sims = np.minimum(X @ a, X @ b)
    # width_percentile=0.95 admits ~10 candidates; cap exactly matches that,
    # so the ranked (non-forced) portion is never truncated on its own.
    r0 = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(200),
                        width_percentile=0.95, segment_pool_max=1000)
    n_passers = len(r0.indices)

    outsiders = np.argsort(sims)[:3]  # 3 candidates far below threshold
    r = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(200),
                       width_percentile=0.95, segment_pool_max=n_passers,
                       force_include=outsiders)

    assert r.capped is False
    assert len(r.indices) == n_passers + 3
    for o in outsiders:
        assert int(o) in r.indices


def test_genre_blend_matches_reference_math():
    """Important 1: fidelity check against segment_pool_builder.py:592-598's
    literal arithmetic (replicated independently here), including the
    genre-similarity clip-to-zero asymmetry (sonic hmean stays unclipped;
    only the genre component is max(0, .) clipped before its own hmean).
    """
    X = _mk_universe(n=50, d=8, seed=1)
    G = _mk_universe(n=50, d=4, seed=2)
    a, b = X[0], X[1]
    genre_vec_a, genre_vec_b = G[0], G[1]
    genre_w = 0.35

    r = build_corridor(vec_a=a, vec_b=b, X_norm=X, universe_indices=np.arange(50),
                       width_percentile=0.0, segment_pool_max=1000,
                       genre_blend_weight=genre_w, X_genre_dense=G,
                       genre_vec_a=genre_vec_a, genre_vec_b=genre_vec_b)

    sim_a = X @ a
    sim_b = X @ b
    expected = np.zeros(50)
    for i in range(50):
        denom = sim_a[i] + sim_b[i]
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a[i] * sim_b[i]) / denom
        g_a_i = max(0.0, float(G[i] @ genre_vec_a))
        g_b_i = max(0.0, float(G[i] @ genre_vec_b))
        gdenom = g_a_i + g_b_i
        genre_hmean = 0.0 if gdenom <= 1e-9 else (2.0 * g_a_i * g_b_i) / gdenom
        expected[i] = (1.0 - genre_w) * hmean + genre_w * genre_hmean

    pos_by_global = {int(idx): pos for pos, idx in enumerate(r.indices)}
    for i in range(50):
        assert r.rank_scores[pos_by_global[i]] == pytest.approx(expected[i])


def test_genre_blend_requires_full_inputs():
    """genre_blend_weight > 0 with incomplete genre inputs must raise, not
    silently fall back -- a configured knob that can't act is a bug."""
    X = _mk_universe(n=20, d=8, seed=3)
    with pytest.raises(ValueError):
        build_corridor(vec_a=X[0], vec_b=X[1], X_norm=X, universe_indices=np.arange(20),
                       width_percentile=0.5, segment_pool_max=100,
                       genre_blend_weight=0.5)
