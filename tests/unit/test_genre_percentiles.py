import numpy as np
from src.playlist.pier_bridge.percentiles import floor_at_percentile, relax_percentile


def test_floor_at_percentile_is_distribution_relative():
    # Sparse seed: most sims low. Dense seed: many sims high.
    sparse = np.concatenate([np.full(900, 0.05), np.full(100, 0.6)])
    dense = np.concatenate([np.full(500, 0.4), np.full(500, 0.8)])
    # Same percentile P -> different absolute floors.
    f_sparse = floor_at_percentile(sparse, p=0.90)
    f_dense = floor_at_percentile(dense, p=0.90)
    assert f_sparse < f_dense
    # p=0.90 keeps ~top 10%
    assert abs((sparse >= f_sparse).mean() - 0.10) < 0.03


def test_relax_percentile_lowers_toward_min():
    seq = relax_percentile(p=0.90, p_min=0.50, step=0.15)
    assert seq[0] == 0.90
    assert seq[-1] <= 0.50 + 1e-9
    assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
