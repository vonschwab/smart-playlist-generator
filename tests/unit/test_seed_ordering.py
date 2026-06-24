"""Roam corridors: min-bottleneck (smoothest weakest-link) pier ordering option."""
import itertools

import numpy as np

from src.playlist.pier_bridge.seeds import _order_seeds_by_bridgeability


def _unit_rows(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _pair(X, a, b):
    return float(np.dot(X[a], X[b]))


def _consecutive(X, order):
    return [_pair(X, order[i], order[i + 1]) for i in range(len(order) - 1)]


def test_sum_objective_default_finds_max_sum_order():
    X = _unit_rows(5, 8, seed=0)
    seeds = [0, 1, 2, 3, 4]
    best_sum = max(sum(_consecutive(X, p)) for p in itertools.permutations(seeds))
    order = _order_seeds_by_bridgeability(
        seeds, X, None, None, weight_sonic=1.0, weight_genre=0.0, weight_bridge=0.0
    )
    assert abs(sum(_consecutive(X, order)) - best_sum) < 1e-9


def test_min_bottleneck_finds_max_min_order():
    X = _unit_rows(5, 8, seed=0)
    seeds = [0, 1, 2, 3, 4]
    best_min = max(min(_consecutive(X, p)) for p in itertools.permutations(seeds))
    order = _order_seeds_by_bridgeability(
        seeds, X, None, None,
        weight_sonic=1.0, weight_genre=0.0, weight_bridge=0.0, min_bottleneck=True,
    )
    assert abs(min(_consecutive(X, order)) - best_min) < 1e-9


def test_min_bottleneck_weakest_link_at_least_as_good_as_sum_order():
    # The invariant that justifies the feature: the min-bottleneck order's weakest
    # consecutive seam is never worse than the sum-order's weakest seam.
    for seed in range(6):
        X = _unit_rows(6, 8, seed=seed)
        seeds = [0, 1, 2, 3, 4, 5]
        sum_order = _order_seeds_by_bridgeability(
            seeds, X, None, None, weight_sonic=1.0, weight_bridge=0.0
        )
        min_order = _order_seeds_by_bridgeability(
            seeds, X, None, None, weight_sonic=1.0, weight_bridge=0.0, min_bottleneck=True
        )
        assert min(_consecutive(X, min_order)) >= min(_consecutive(X, sum_order)) - 1e-9


def test_default_is_sum_objective():
    X = _unit_rows(4, 6, seed=1)
    seeds = [0, 1, 2, 3]
    default = _order_seeds_by_bridgeability(seeds, X, None, None, weight_sonic=1.0, weight_bridge=0.0)
    explicit = _order_seeds_by_bridgeability(
        seeds, X, None, None, weight_sonic=1.0, weight_bridge=0.0, min_bottleneck=False
    )
    assert default == explicit
