import numpy as np
from src.playlist.candidate_pool import _apply_popularity_gate


def test_gate_keeps_only_ranks_below_cutoff():
    ranks = np.array([0, 4, 9, 10, 49, -1], dtype=int)
    eligible = [0, 1, 2, 3, 4, 5]
    kept, excluded = _apply_popularity_gate(eligible, ranks, rank_cutoff=10)
    assert kept == [0, 1, 2]          # ranks 0,4,9 < 10
    assert excluded == 3              # rank 10, rank 49, and -1 (uncached) dropped


def test_gate_excludes_uncached_minus_one():
    ranks = np.array([-1, -1], dtype=int)
    kept, excluded = _apply_popularity_gate([0, 1], ranks, rank_cutoff=50)
    assert kept == []
    assert excluded == 2
