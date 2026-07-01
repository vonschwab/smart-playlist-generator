# tests/unit/test_mini_pier_select.py
import numpy as np
from src.playlist.pier_bridge.mini_pier_select import select_waypoint

def _unit(M):
    M = np.asarray(M, float)
    return M / np.linalg.norm(M, axis=1, keepdims=True)

def test_picks_between_and_off_center():
    # A and B piers; c1 is central-to-both (blur), c2 is between but more distinctive,
    # c3 is a distant outlier (must be excluded by the smoothness floor).
    X = _unit([
        [1, 0, 0, 0],    # 0 pier A
        [0, 1, 0, 0],    # 1 pier B
        [1, 1, 0, 0],    # 2 c1: max between (the blur center)
        [1, 1, 0.6, 0],  # 3 c2: between + a distinctive component
        [0, 0, 1, 1],    # 4 c3: distant outlier
    ])
    got = select_waypoint(0, 1, [2, 3, 4], X, margin=0.12, k_broad=3)
    assert got in (2, 3)          # never the distant outlier c3
    assert got == 3               # among the smooth pair, the less-central one wins

def test_excludes_pier_and_excluded_indices():
    X = _unit(np.eye(6))
    assert select_waypoint(0, 1, [0, 1], X) is None          # only piers -> nothing
    assert select_waypoint(0, 1, [2, 3], X, exclude=frozenset({2, 3})) is None

def test_returns_none_on_empty_pool():
    X = _unit(np.eye(4))
    assert select_waypoint(0, 1, [], X) is None


from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence

def test_no_split_when_segments_short():
    X = _unit(np.eye(10))
    # 3 seeds, 30 tracks -> interior 27 / 2 segs ~13; but max_interior huge -> no split
    seq = plan_pier_sequence([0, 1, 2], 30, list(range(3, 10)), X,
                             max_interior=99, margin=0.12, k_broad=5,
                             exclude_base=frozenset(), max_waypoints=3)
    assert seq == [0, 1, 2]

def test_splits_longest_until_under_K():
    # 2 seeds, long interior -> must insert waypoints so each segment interior <= K.
    X = _unit(np.random.default_rng(0).normal(size=(60, 8)))
    seq = plan_pier_sequence([0, 1], 20, list(range(2, 60)), X,
                             max_interior=5, margin=0.20, k_broad=30,
                             exclude_base=frozenset(), max_waypoints=5)
    assert seq[0] == 0 and seq[-1] == 1     # original piers stay at the ends
    assert len(seq) > 2                      # at least one waypoint inserted
    # every segment's even-split interior is <= K
    n_seg = len(seq) - 1
    interior = 20 - len(seq)
    base = interior // n_seg
    assert base <= 5

def test_respects_max_waypoints_cap():
    X = _unit(np.random.default_rng(1).normal(size=(60, 8)))
    seq = plan_pier_sequence([0, 1], 40, list(range(2, 60)), X,
                             max_interior=3, margin=0.30, k_broad=30,
                             exclude_base=frozenset(), max_waypoints=2)
    assert len(seq) - 2 <= 2                  # never more than max_waypoints inserted
