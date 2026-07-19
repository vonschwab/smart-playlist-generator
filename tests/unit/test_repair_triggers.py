"""Relative repair triggers (Phase 2 Task 2): pure trigger-decision function
shared by tail-DP and edge-repair. See docs/corridor_baseline/phase2_mechanism_probes.md
for the evidence motivating this (both floor gates fired on "clears the absolute
floor" while a materially better in-pool connector sat unused)."""
import pytest

from src.playlist.pier_bridge.repair_triggers import compute_relative_trigger_floor


def test_epsilon_zero_is_absolute_only_legacy():
    # relative_epsilon <= 0 => byte-identical to the pre-Task-2 behavior:
    # effective floor is exactly base_floor, regardless of how high the mean is.
    result = compute_relative_trigger_floor(
        base_floor=0.30, reference_mean=0.90, relative_epsilon=0.0,
    )
    assert result.effective_floor == 0.30
    assert result.source == "absolute"


def test_negative_epsilon_also_treated_as_legacy():
    result = compute_relative_trigger_floor(
        base_floor=0.30, reference_mean=0.90, relative_epsilon=-0.1,
    )
    assert result.effective_floor == 0.30
    assert result.source == "absolute"


def test_relative_binds_when_segment_mean_is_high():
    # PC-shaped case: segment mean ~0.78 (healthy segment overall), epsilon 0.25
    # -> relative threshold 0.53, well above the 0.30 absolute floor -- an edge
    # at 0.394 (clears 0.30) would now be caught by the relative arm.
    result = compute_relative_trigger_floor(
        base_floor=0.30, reference_mean=0.78, relative_epsilon=0.25,
    )
    assert result.source == "relative"
    assert result.effective_floor == pytest.approx(0.53)
    assert result.relative_threshold == pytest.approx(0.53)


def test_absolute_wins_when_relative_threshold_is_lower():
    # Low segment mean (0.20): mean - epsilon (-0.05) is below the absolute
    # floor (0.30) -> absolute wins, exactly today's behavior.
    result = compute_relative_trigger_floor(
        base_floor=0.30, reference_mean=0.20, relative_epsilon=0.25,
    )
    assert result.source == "absolute"
    assert result.effective_floor == 0.30


def test_never_triggers_on_edges_above_both_floors():
    # An edge at 0.60 clears both the absolute floor (0.30) and the relative
    # threshold (mean 0.65 - epsilon 0.25 = 0.40) -> old_min >= effective_floor,
    # so the caller's own floor-gate ("old_min >= floor -> skip") stands down.
    result = compute_relative_trigger_floor(
        base_floor=0.30, reference_mean=0.65, relative_epsilon=0.25,
    )
    old_min = 0.60
    assert old_min >= result.effective_floor  # gate would skip -- no false trigger


def test_boundary_relative_threshold_equal_to_base_is_absolute():
    # relative_threshold exactly equal to base_floor (0.75 - 0.25 == 0.5, exact
    # in binary float): not strictly greater, so source stays "absolute" --
    # deterministic tie-break, avoids flip-flopping on floating point equality.
    result = compute_relative_trigger_floor(
        base_floor=0.5, reference_mean=0.75, relative_epsilon=0.25,
    )
    assert result.effective_floor == 0.5
    assert result.source == "absolute"
