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


def _always_fresh_select(pool):
    """A select_waypoint stub that always returns a fresh, unused pool index
    (feasibility never blocks), so tests exercise the count/distribution logic."""
    def _stub(a, b, cand, Xn, *, margin, k_broad, exclude):
        for c in pool:
            if c not in exclude:
                return int(c)
        return None
    return _stub


def test_balance_gaps_off_leaves_trailing_gap_unsplit(monkeypatch):
    """Baseline (the Helado case): 4 seeds / 30 tracks needs 2 waypoints to reach
    interior<=5, and the round-robin lands both in the first two gaps -> the last
    gap stays unsplit and the trailing two anchors bunch."""
    seeds = [0, 1, 2, 3]                   # 4 seeds -> 3 gaps
    pool = list(range(100, 400))
    X = np.zeros((400, 4), dtype=np.float32)
    monkeypatch.setattr(mini_pier_select, "select_waypoint", _always_fresh_select(pool))
    piers = plan_pier_sequence(
        seeds, 30, pool, X,
        max_interior=5, margin=0.12, k_broad=150, max_waypoints=7,
        balance_gaps=False,
    )
    assert _waypoints_per_gap(seeds, piers) == [1, 1, 0]   # last gap starved


def test_balance_gaps_on_equalizes_all_seed_gaps(monkeypatch):
    """Option 2: balancing tops the trailing gap up to one waypoint each, so every
    seed-gap holds an equal count and the seed anchors land evenly spaced."""
    seeds = [0, 1, 2, 3]
    pool = list(range(100, 400))
    X = np.zeros((400, 4), dtype=np.float32)
    monkeypatch.setattr(mini_pier_select, "select_waypoint", _always_fresh_select(pool))
    piers = plan_pier_sequence(
        seeds, 30, pool, X,
        max_interior=5, margin=0.12, k_broad=150, max_waypoints=7,
        balance_gaps=True,
    )
    per_gap = _waypoints_per_gap(seeds, piers)
    assert per_gap == [1, 1, 1]                 # every gap equal -> even anchors
    assert max(per_gap) - min(per_gap) == 0
    # 7 piers over 30 tracks -> interiors [4,4,4,4,4,3]; still within max_interior.
    assert len(piers) == 7
