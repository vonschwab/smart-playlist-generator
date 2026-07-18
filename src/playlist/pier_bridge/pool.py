"""
Tier-3.1 PR-7: bridge-score kernel.

Extracted verbatim from pier_bridge_builder.py:
  _compute_bridge_score           — scoring kernel (harmonic mean or experiment blend)

Phase 1 Task 8: `_build_segment_candidate_pool_legacy` (the KNN-union debug/
compat pool) and `_build_segment_candidate_pool_scored` (the production
segment-scored pool, a thin wrapper around
src.playlist.segment_pool_builder.SegmentCandidatePoolBuilder) were DELETED
along with the legacy/corridor pooling selector in pier_bridge_builder.py --
corridor pooling (src.playlist.pier_bridge.corridor.build_corridor +
src.playlist.pier_bridge.eligible_universe.build_eligible_universe) is now
the sole segment-pool strategy. `_compute_bridge_score` had zero production
callers even before this deletion (a standalone scoring-kernel utility,
exercised only by its direct unit tests in
tests/unit/test_pier_bridge_smoke_golden.py) and is kept as-is.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _compute_bridge_score(
    sim_a: float,
    sim_b: float,
    *,
    experiment_enabled: bool,
    experiment_min_weight: float,
    experiment_balance_weight: float,
) -> float:
    denom = sim_a + sim_b
    hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
    if not experiment_enabled:
        return float(hmean)

    min_weight = max(0.0, min(1.0, float(experiment_min_weight)))
    balance_weight = max(0.0, min(1.0 - min_weight, float(experiment_balance_weight)))
    hmean_weight = max(0.0, 1.0 - min_weight - balance_weight)

    min_sim = min(sim_a, sim_b)
    balance = 1.0 - abs(sim_a - sim_b)
    if balance < 0.0:
        balance = 0.0
    elif balance > 1.0:
        balance = 1.0

    return float(
        (hmean_weight * hmean)
        + (min_weight * min_sim)
        + (balance_weight * balance)
    )
