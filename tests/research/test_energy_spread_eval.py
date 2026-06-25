import numpy as np

from scripts.research.energy_spread_eval import (
    pier_arousal_span,
    representativeness_tv,
)


def test_pier_arousal_span_basic():
    track_ids = ["a", "b", "c", "d"]
    arousal = np.array([0.0, 1.0, -1.0, np.nan])
    # medoids a,c span [−1, 0] => 1.0
    assert pier_arousal_span(["a", "c"], track_ids, arousal) == 1.0
    # single finite => 0.0
    assert pier_arousal_span(["a"], track_ids, arousal) == 0.0
    # NaN ignored
    assert pier_arousal_span(["a", "d"], track_ids, arousal) == 0.0


def test_representativeness_tv_rewards_matching_distribution():
    lo, hi = -0.5, 0.5
    # catalog: 5 soft, 3 mid, 2 aggressive (50/30/20)
    catalog = np.array([-1.0] * 5 + [0.0] * 3 + [1.0] * 2)
    matching = np.array([-1.0, -1.0, 0.0, 1.0])  # ~soft-leaning, mirrors catalog
    skewed = np.array([1.0, 1.0, 1.0, 1.0])       # all aggressive, opposite of catalog
    d_match = representativeness_tv(matching, catalog, lo, hi)
    d_skew = representativeness_tv(skewed, catalog, lo, hi)
    assert 0.0 <= d_match <= 1.0
    assert 0.0 <= d_skew <= 1.0
    assert d_match < d_skew                  # mirroring the band is more representative
    assert d_skew > 0.5                       # all-aggressive vs soft-leaning is far off


def test_representativeness_tv_zero_when_identical():
    lo, hi = -0.5, 0.5
    catalog = np.array([-1.0, -1.0, 0.0, 1.0])
    # same band mix as catalog => distance 0
    assert representativeness_tv(catalog.copy(), catalog, lo, hi) == 0.0
