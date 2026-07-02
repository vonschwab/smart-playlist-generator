# tests/unit/test_mini_pier_spacing.py
import numpy as np
from src.playlist.pier_bridge import mini_pier_select
from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence


def _waypoints_per_gap(seeds, piers):
    """Count inserted (non-seed) piers between each consecutive pair of seeds."""
    seed_pos = [piers.index(s) for s in seeds]
    return [seed_pos[i + 1] - seed_pos[i] - 1 for i in range(len(seed_pos) - 1)]


def test_waypoints_distribute_across_gaps(monkeypatch):
    seeds = list(range(10))               # 10 seeds -> 9 gaps
    total_tracks = 100                    # interior 90, max_interior 5 -> 8 waypoints
    pool = list(range(100, 400))
    X = np.zeros((400, 4), dtype=np.float32)
    counter = {"n": 0}

    def fake_select_waypoint(a, b, cand, Xn, *, margin, k_broad, exclude):
        # deterministic fresh index each call, never a seed/existing pier
        for c in pool:
            if c not in exclude:
                counter["n"] += 1
                return int(c)
        return None

    monkeypatch.setattr(mini_pier_select, "select_waypoint", fake_select_waypoint)
    piers = plan_pier_sequence(
        seeds, total_tracks, pool, X,
        max_interior=5, margin=0.12, k_broad=150, max_waypoints=25,
    )
    per_gap = _waypoints_per_gap(seeds, piers)
    assert sum(per_gap) == 8                     # 8 waypoints inserted
    assert max(per_gap) <= 2                     # never all in one gap (bug was 8 in gap 0)
    assert sum(1 for g in per_gap if g >= 1) >= 7  # spread across >=7 of the 9 gaps
