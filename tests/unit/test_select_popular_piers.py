import numpy as np
from src.playlist.artist_style import select_popular_piers


def test_picks_top_n_by_popularity_descending():
    # indices 0..4; popularity score = 1 - rank/n (higher = more popular)
    pv = np.array([0.10, 0.95, np.nan, 0.80, 0.50])
    piers = select_popular_piers([0, 1, 2, 3, 4], pv, target_pier_count=3)
    assert piers == [1, 3, 4]          # 0.95, 0.80, 0.50; index 2 (NaN) excluded


def test_returns_fewer_when_hits_scarce():
    pv = np.array([np.nan, 0.90, np.nan, np.nan])
    piers = select_popular_piers([0, 1, 2, 3], pv, target_pier_count=3)
    assert piers == [1]                # only one hit; never pads with non-hits


def test_returns_empty_when_no_finite_scores():
    pv = np.array([np.nan, np.nan])
    assert select_popular_piers([0, 1], pv, target_pier_count=3) == []


def test_tie_broken_by_index():
    pv = np.array([0.5, 0.5, 0.5])
    assert select_popular_piers([2, 0, 1], pv, target_pier_count=2) == [0, 1]
