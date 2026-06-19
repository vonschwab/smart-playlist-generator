import numpy as np
from scripts.research import pace_eval_metrics as m


def test_zscore_params_ignores_nan_and_zero_std():
    assert m.zscore_params(np.array([1.0, 1.0, 1.0])) == (1.0, 1.0)  # std 0 -> 1
    mean, std = m.zscore_params(np.array([0.0, 2.0, np.nan, 4.0]))
    assert mean == 2.0 and std > 0


def test_weighted_euclidean_nan_propagates():
    assert np.isnan(m.weighted_euclidean(np.array([np.nan]), np.array([0.0])))
    assert m.weighted_euclidean(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_auc_perfect_and_random():
    # all pos distances below all neg -> AUC 1.0
    assert m.auc_pos_below_neg(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    # interleaved -> 0.5
    assert m.auc_pos_below_neg(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5


def test_distribution_percentiles():
    d = m.distribution(np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan]))
    assert d["n"] == 5 and d["min"] == 0.0 and d["p50"] == 2.0
