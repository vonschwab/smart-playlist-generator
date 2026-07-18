"""Regression guard for the forward half of cross-segment min_gap enforcement
(Phase 1 min_gap fix, 2026-07-18).

``recent_boundary_artists`` (pier_bridge_builder.py, fixed 2026-06-01) only ever
blocked artists BACKWARD (already-placed tracks) from the next segment. Mini-pier
waypoints (SP3, 2026-06-30) made the full pier sequence (``ordered_seeds``) fixed
and known BEFORE the segment loop starts, but nothing used that knowledge: an
early segment could freely place an interior track by the same artist as a
not-yet-built pier/waypoint a few positions later, since piers are placed
unconditionally (never gated on artist novelty). See
tests/integration/test_gui_fidelity_regressions.py::
test_strong_artist_gap_enforced_across_segments for the end-to-end repro and
.superpowers/sdd/p1-mingap-fix-report.md for the full root-cause writeup.

These two pure helpers are the fix: ``_pier_nominal_positions`` computes each
pier's nominal position once, up front; ``_forward_pier_gap_block_indices``
answers, for a given segment, which UPCOMING piers land within min_gap.
"""
from src.playlist.pier_bridge_builder import (
    _forward_pier_gap_block_indices,
    _pier_nominal_positions,
)


def test_pier_nominal_positions_cumulative():
    # 3 segments of interior length 3, 2, 4 -> 4 piers at positions 0, 4, 7, 12.
    assert _pier_nominal_positions([3, 2, 4]) == [0, 4, 7, 12]


def test_pier_nominal_positions_single_segment():
    assert _pier_nominal_positions([5]) == [0, 6]


def test_forward_block_catches_the_reported_violation_shape():
    # Mirrors the repro fixture's actual geometry: 8 segments of interior 3 (x5)
    # then 2 (x3) -> piers at 0,4,8,12,16,20,23,26,29 (9 piers total, matching
    # the reported log line "Mini-piers: 4 waypoint(s) inserted (piers now 9)").
    positions = _pier_nominal_positions([3, 3, 3, 3, 3, 2, 2, 2])
    assert positions == [0, 4, 8, 12, 16, 20, 23, 26, 29]

    # Segment 1 (piers[1]=4 -> piers[2]=8): pier[3] (Golden Brown waypoint) sits
    # at nominal position 12, only 8 away from this segment's own start (4) --
    # inside min_gap=9, so it must be blocked. pier[4] (16) is 12 away -- clear.
    assert _forward_pier_gap_block_indices(positions, seg_idx=1, min_gap=9) == [3]

    # Segment 2 (piers[2]=8 -> piers[3]=12): pier[4] (Hayden Pedigo pier) sits at
    # 16, only 8 away from this segment's start (8) -- inside min_gap=9.
    assert _forward_pier_gap_block_indices(positions, seg_idx=2, min_gap=9) == [4]

    # Segment 3 (piers[3]=12 -> piers[4]=16): pier[5] (Dylan Golden Aycock pier)
    # sits at 20, 8 away from this segment's start (12) -- inside min_gap=9.
    assert _forward_pier_gap_block_indices(positions, seg_idx=3, min_gap=9) == [5]


def test_forward_block_never_includes_own_boundary_piers():
    # seg_idx's own two piers (index seg_idx and seg_idx+1) are already banned
    # locally via disallow_pier_artists_in_interiors -- the forward block must
    # start at seg_idx+2 or it would double-report the segment's own end pier.
    positions = [0, 4, 8, 12]
    blocked = _forward_pier_gap_block_indices(positions, seg_idx=0, min_gap=9)
    assert 0 not in blocked
    assert 1 not in blocked


def test_forward_block_stops_once_far_enough():
    # Positions increase strictly, so once one pier clears min_gap every later
    # one does too -- confirms the early-break doesn't miss a closer one first.
    # seg_idx=0 starts at position 0; pier[2]=8 is inside min_gap=9, pier[3]=12
    # already clears it (and everything after 12 is further still).
    positions = [0, 4, 8, 12, 30, 31, 32]
    assert _forward_pier_gap_block_indices(positions, seg_idx=0, min_gap=9) == [2]


def test_forward_block_disabled_when_min_gap_zero():
    positions = [0, 4, 8, 12]
    assert _forward_pier_gap_block_indices(positions, seg_idx=0, min_gap=0) == []


def test_forward_block_empty_when_no_upcoming_piers_within_gap():
    positions = [0, 20, 40]
    assert _forward_pier_gap_block_indices(positions, seg_idx=0, min_gap=9) == []
