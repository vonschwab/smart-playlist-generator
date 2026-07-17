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
