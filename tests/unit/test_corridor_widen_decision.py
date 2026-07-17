"""Unit tests for the pure widening-ladder scarcity gate (Task 6 remediation).

Root cause traced (not re-derived here, see .superpowers/sdd/p1-task6-
remediation-report.md and the SADE/home A/B log): the widening ladder
triggers correctly on weak edges (min_edge_T < transition_floor) but widens
even when the corridor is NOT the binding constraint -- a beam-path-internal
weakness pays 3x beam cost (initial + 2 widen attempts), EXHAUSTS with no
improvement, and the repair stack fixes the edge anyway. The fix gates
WIDENING (not the trigger) on corridor scarcity via the Phase-0a-validated
anchor-support coverage metric: widen only when min(support_a, support_b) <
corridor_widen_support_threshold.
"""
import pytest

from src.playlist.pier_bridge.corridor import (
    CorridorWidenDecision,
    corridor_widen_decision,
)


def test_no_path_always_widens_regardless_of_support():
    """Hard infeasibility (path is None) always widens -- no path is
    inherently a pool problem, never gated on scarcity."""
    decision = corridor_widen_decision(
        path_found=False,
        min_edge_t=None,
        floor=0.20,
        support_a=0.99,  # healthy support -- must NOT suppress widening
        support_b=0.99,
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.WIDEN


def test_quality_ok_accepts_without_consulting_support():
    """min_edge_T >= floor: no widening needed at all, regardless of support."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.25,
        floor=0.20,
        support_a=0.01,  # starved support -- irrelevant, quality is already fine
        support_b=0.01,
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.ACCEPT


def test_weak_edge_with_starved_support_widens():
    """Weak edge (below floor) AND the corridor is plausibly starved
    (min support < threshold): widen -- this is the Swirlies-class case."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        support_a=0.15,
        support_b=0.30,
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.WIDEN


def test_weak_edge_with_healthy_support_skips():
    """Weak edge (below floor) but healthy corridor pools (min support >=
    threshold): the weakness is beam-path-internal, not pool-limited -- skip
    widening and hand the segment to the repair stack. This is the traced
    SADE/home root cause (min support well above 0.5, min_edge in 0.075-0.185
    vs floor 0.200)."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.15,
        floor=0.20,
        support_a=0.55,
        support_b=0.80,
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.SKIP


def test_min_of_both_supports_governs_not_max():
    """One anchor starved, one healthy -- min() must govern (a candidate
    corridor built from the WORSE anchor's coverage is the plausible
    bottleneck even if the other anchor is fine)."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        support_a=0.90,  # healthy
        support_b=0.10,  # starved -- must dominate via min()
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.WIDEN


@pytest.mark.parametrize("support_a,support_b,threshold,expected", [
    (0.5, 0.5, 0.5, CorridorWidenDecision.SKIP),   # exactly at threshold: not "< threshold" -> SKIP
    (0.4999, 0.9, 0.5, CorridorWidenDecision.WIDEN),  # just under -> WIDEN
])
def test_threshold_boundary_is_strict_less_than(support_a, support_b, threshold, expected):
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        support_a=support_a,
        support_b=support_b,
        threshold=threshold,
    )
    assert decision == expected


def test_min_edge_t_none_with_path_found_treated_as_not_ok():
    """Defensive: a path found but with no scoreable min_edge_t (None, not
    +inf -- a 0-interior segment path returns +inf per _segment_min_edge_t's
    own docstring, so a bare None here is an edge case the gate should not
    silently treat as passing quality)."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=None,
        floor=0.20,
        support_a=0.9,
        support_b=0.9,
        threshold=0.5,
    )
    assert decision == CorridorWidenDecision.SKIP
