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
