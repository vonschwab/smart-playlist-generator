"""Unit tests for the pure widening-ladder continue-gate (Task 6 remediation,
iteration 2).

Iteration 1 gated widening on a PREDICTIVE signal (anchor-support coverage,
computed before the beam ever ran). Falsified by real evidence (see
.superpowers/sdd/p1-task6-remediation-report.md): Alex G/home's segment 1 had
healthy support (~0.8, well above the 0.5 threshold) yet still gained +0.42 T
from one widen attempt (0.189 -> 0.611) -- a wider pool can unlock better
beam-path (interior-to-interior) combinations no anchor-only metric predicts.

Iteration 2 replaces prediction with an EMPIRICAL continue-gate: always try
widen attempt 1 unconditionally when the quality trigger fires; widen further
only if the attempt just run improved the best-seen min_edge_T by more than
``corridor_widen_improvement_epsilon``.
"""
import pytest

from src.playlist.pier_bridge.corridor import (
    CorridorWidenDecision,
    corridor_widen_decision,
)


def test_no_path_always_widens_regardless_of_attempt_index_or_improvement():
    """Hard infeasibility (path is None) always widens to the full attempt
    budget -- no path is inherently a pool problem, never gated on
    improvement, at ANY attempt index."""
    for attempt_index, improvement in [(0, None), (1, -0.5), (2, 0.9)]:
        decision = corridor_widen_decision(
            path_found=False,
            min_edge_t=None,
            floor=0.20,
            attempt_index=attempt_index,
            improvement=improvement,
            epsilon=0.02,
        )
        assert decision == CorridorWidenDecision.WIDEN, (attempt_index, improvement)


def test_quality_ok_accepts_regardless_of_attempt_index():
    """min_edge_T >= floor: no widening needed at all, regardless of how many
    widen attempts already ran or whether they improved."""
    for attempt_index in (0, 1, 2):
        decision = corridor_widen_decision(
            path_found=True,
            min_edge_t=0.25,
            floor=0.20,
            attempt_index=attempt_index,
            improvement=-1.0,  # irrelevant -- quality is already fine
            epsilon=0.02,
        )
        assert decision == CorridorWidenDecision.ACCEPT


def test_first_evaluation_always_widens_unconditionally():
    """attempt_index == 0: the quality trigger just fired for the first
    time -- always try widening once, no prediction, no improvement signal
    available yet (this attempt itself hasn't tried widening)."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.05,  # far below floor
        floor=0.20,
        attempt_index=0,
        improvement=None,  # nothing to compare yet
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.WIDEN


def test_improving_attempt_continues_widening():
    """attempt_index >= 1 and this attempt improved the best-seen min_edge_T
    by more than epsilon: keep widening."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.15,
        floor=0.20,
        attempt_index=1,
        improvement=0.05,  # > epsilon=0.02
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.WIDEN


def test_non_improving_attempt_stops():
    """attempt_index >= 1 and this attempt did NOT improve beyond epsilon
    (flat or negligible gain): stop widening, hand off to the repair stack."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.15,
        floor=0.20,
        attempt_index=1,
        improvement=0.01,  # < epsilon=0.02
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.STOP


def test_worsening_attempt_stops():
    """A widen attempt that made min_edge_T WORSE (negative improvement)
    must stop, not widen further -- this is the real shape seen in the
    Swirlies/home corpus log (attempt 1 regressed 0.163 -> 0.145 before
    attempt 2 eventually recovered); iteration 2 accepts that this
    empirical rule can stop before a later attempt would have helped, in
    exchange for never paying for attempts that don't pay off on average."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        attempt_index=1,
        improvement=-0.018,
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.STOP


@pytest.mark.parametrize("improvement,expected", [
    (0.02, CorridorWidenDecision.STOP),   # exactly at epsilon: not "> epsilon" -> STOP
    (0.0201, CorridorWidenDecision.WIDEN),  # just over -> WIDEN
])
def test_epsilon_boundary_is_strict_greater_than(improvement, expected):
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        attempt_index=1,
        improvement=improvement,
        epsilon=0.02,
    )
    assert decision == expected


def test_improvement_none_at_nonzero_attempt_index_stops_not_widens():
    """Defensive: improvement=None at attempt_index >= 1 (with path_found
    True) is treated as no-improvement -- STOP, never silently WIDEN. This
    combination shouldn't arise from the real ladder (a 0-interior segment
    with +inf min_edge_t would already be quality_ok and short-circuit
    above), but the gate must not guess "keep going" from missing data."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.10,
        floor=0.20,
        attempt_index=1,
        improvement=None,
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.STOP


def test_continued_widening_can_run_past_attempt_two_if_still_improving():
    """The gate itself has no hardcoded cap at attempt 2 -- the ladder's own
    corridor_widen_attempts budget is what bounds the loop. Pin that a
    hypothetical attempt_index=2 (deciding whether to try a 3rd widen) still
    follows the same improvement rule as attempt_index=1."""
    decision = corridor_widen_decision(
        path_found=True,
        min_edge_t=0.18,
        floor=0.20,
        attempt_index=2,
        improvement=0.10,
        epsilon=0.02,
    )
    assert decision == CorridorWidenDecision.WIDEN
