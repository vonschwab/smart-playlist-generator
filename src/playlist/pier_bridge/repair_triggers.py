"""Relative repair triggers (Phase 2 Task 2, corridor-phase2 branch).

Both post-beam repair mechanisms -- tail-DP's landing-window re-optimization
and break-glass edge repair -- gate on a fixed absolute floor
(``tail_dp_floor`` / ``edge_repair_t_floor``, both 0.30). Phase 2 Task 1's
mechanism probes (docs/corridor_baseline/phase2_mechanism_probes.md) showed
this floor is calibrated to "not broken," not "not much worse than
achievable": Parquet Courts segment 4's worst edge (0.394) cleared
``tail_dp_floor=0.3`` so tail-DP never even searched, while a materially
better connector (Sonic Youth "Theresa's Sound-World", T ~= 0.7-0.8 once
swapped in by hand) sat unused in the very same admitted pool. SADE/home's
weakest edge (0.454) is its segment's FIRST edge, which tail-DP structurally
can't reach at all -- edge repair's fixed floor has the identical
"clears 0.3, so nobody looks" problem there.

This module holds the one pure decision both call sites share: the effective
trigger floor is whichever is higher, an absolute constant or a threshold
relative to the surrounding "achievable" level (a segment mean for tail-DP,
a whole-playlist mean for edge repair) minus a margin. The caller decides
what "reference_mean" means for its own currency; this function only does
the max()-and-tag-the-source arithmetic so both call sites log identically.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerFloor:
    """Resolved repair-trigger floor for one tail-DP or edge-repair invocation."""

    effective_floor: float
    source: str  # "absolute" | "relative"
    # reference_mean - relative_epsilon, computed and logged even when the
    # absolute floor wins (so a near-miss is visible in diagnostics).
    relative_threshold: float


def compute_relative_trigger_floor(
    *, base_floor: float, reference_mean: float, relative_epsilon: float,
) -> TriggerFloor:
    """``max(base_floor, reference_mean - relative_epsilon)``, with source tagging.

    ``relative_epsilon <= 0`` is the legacy escape hatch: absolute-only,
    byte-identical to pre-Phase-2-Task-2 behavior (effective_floor ==
    base_floor exactly, regardless of reference_mean).

    Otherwise the relative threshold binds only when it is STRICTLY greater
    than base_floor (a tie stays "absolute" -- deterministic, no floating-point
    flip-flopping at the boundary).
    """
    base = float(base_floor)
    eps = float(relative_epsilon)
    if eps <= 0.0:
        return TriggerFloor(effective_floor=base, source="absolute", relative_threshold=base)
    relative_threshold = float(reference_mean) - eps
    if relative_threshold > base:
        return TriggerFloor(
            effective_floor=relative_threshold, source="relative", relative_threshold=relative_threshold,
        )
    return TriggerFloor(effective_floor=base, source="absolute", relative_threshold=relative_threshold)
