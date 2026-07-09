import numpy as np
from src.playlist.candidate_pool import select_pool_guarantee


def test_guarantee_ranks_by_sim_and_caps_total():
    track_ids = np.array([f"t{i}" for i in range(8)])
    artist_keys = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    sim = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    got = select_pool_guarantee(
        candidate_indices=range(8),
        guarantee_ids={"t0", "t2", "t4", "t6"},   # one per artist a,b,c,d
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted=set(), max_total=3, per_artist=2,
    )
    assert got == [2, 4, 6]        # highest-sim guarantee ids first (t2=.8,t4=.7,t6=.6), capped at 3


def test_guarantee_per_artist_cap():
    track_ids = np.array([f"t{i}" for i in range(4)])
    artist_keys = np.array(["a", "a", "a", "b"])
    sim = np.array([0.9, 0.8, 0.7, 0.6])
    got = select_pool_guarantee(
        candidate_indices=range(4), guarantee_ids={"t0", "t1", "t2", "t3"},
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted=set(), max_total=10, per_artist=2,
    )
    assert got == [0, 1, 3]        # artist 'a' capped at 2 (t0,t1); t2 dropped; t3 (artist b) kept


def test_guarantee_skips_already_admitted_and_non_guarantee():
    track_ids = np.array([f"t{i}" for i in range(4)])
    artist_keys = np.array(["a", "b", "c", "d"])
    sim = np.array([0.9, 0.8, 0.7, 0.6])
    got = select_pool_guarantee(
        candidate_indices=range(4), guarantee_ids={"t0", "t2"},
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted={0}, max_total=10, per_artist=5,
    )
    assert got == [2]              # t0 already admitted; t1/t3 not in guarantee_ids


def test_guarantee_empty_inputs():
    tids = np.array(["t0"])
    aks = np.array(["a"])
    sim = np.array([0.5])
    assert select_pool_guarantee(range(1), set(), tids, aks, sim, set(), 10, 5) == []
    assert select_pool_guarantee(range(1), {"t0"}, tids, aks, sim, set(), 0, 5) == []


def test_guarantee_none_sim_falls_back_to_index_order():
    tids = np.array([f"t{i}" for i in range(3)])
    aks = np.array(["a", "b", "c"])
    got = select_pool_guarantee(range(3), {"t0", "t1", "t2"}, tids, aks, None, set(), 10, 5)
    assert got == [0, 1, 2]
