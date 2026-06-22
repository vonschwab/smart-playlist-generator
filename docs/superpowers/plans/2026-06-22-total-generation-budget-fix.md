# Total-generation wall-clock budget fix (pier-bridge) — finding + fix design

**Date:** 2026-06-22
**Branch:** worktree-v6-canonical-wiring
**Status:** Fix designed, not yet implemented. Plan-1 calibration (Task 5) PAUSED behind this.
**Severity:** CRITICAL — violates the hard "a playlist must NEVER take >90s" rule (`feedback_generation_time_budget`) by ~15×.

## How this surfaced

During Plan-1 (`2026-06-21-adaptive-pool-admission.md`) Task-5 calibration, the in-process grid
(`scripts/research/run_admission_grid.py`) measured real generations across diverse seed niches.
Partial grid (9/22 cells, results in `docs/run_audits/adaptive_admission/grid_results.jsonl`):

| mode/niche | label | admitted | distinct | worst_edge | wall |
|---|---|---|---|---|---|
| strict/hyperpop | baseline | 369 | 11 | 0.18 | **1401.9s** |
| strict/hyperpop | medium (sp0.7 gp0.8) | 389 | 11 | 0.18 | **1345.8s** |
| strict/hyperpop | loose (sp0.55 gp0.65) | 434 | 11 | 0.18 | **1577.8s** |
| strict/jazz | baseline | 806 | 27 | 0.029 | 76.0s |
| strict/jazz | medium | 846 | 27 | 0.029 | 78.9s |
| strict/jazz | loose | 928 | 27 | 0.029 | 83.1s |
| strict/metal | baseline | 161 | 12 | 0.123 | 52.3s |
| strict/metal | medium | 168 | 12 | 0.123 | 52.5s |
| strict/metal | loose | 189 | 12 | 0.123 | 54.9s |

**strict + hyperpop = ~23 minutes per generation.** jazz/metal are fine (<90s). The grid was stopped
(TaskStop) at 9/22 — the remaining hyperpop cells would each cost ~23 min.

## Why it's NOT the adaptive-admission work
The `baseline` cells use NO percentile / NO min_pool (`sonic_pct=None, genre_pct=None, min_pool=None`)
→ the Tasks 1-3 code is a no-op on them → they reproduce **current master** behavior. baseline
strict/hyperpop is 1401s, so this is a **pre-existing master bug**, merely *exposed* by exercising
strict mode on diverse low-cosine seeds (nobody had calibrated strict before).

## Root cause — the cascade-budget fix (`bc942d5`) is incomplete
`bc942d5` added `_SEGMENT_RELAXATION_BUDGET_S = 40s` anchored at `_pb_build_start`
(`pier_bridge_builder.py:1563`), but it only guards the **floor-relaxation tiers** (tier-2 transition
floor ~:1820, tier-3 genre-arc floor ~:1870). It does NOT bound:
1. **Tier-1** — `_run_segment_backoff_attempts`: `for attempt in range(max_expansion_attempts)` (=4) ×
   bridge-floor backoff (≈6 steps) ≈ up to ~24 beam runs PER segment, unguarded.
2. **The micro-pier fallback** (~:1862-1970) — unguarded.
3. **The One-Each retry in `core.py`** (`_relaxed_one_each_candidate_attempts`, ~:96-151) — on
   infeasibility re-invokes the whole pier-bridge build ~3× (sonic 0.08→0.05→0.00, genre
   0.30→0.20→0.00), and **each call resets `_pb_build_start`** → a fresh 40s budget per retry.

For a starved low-cosine pool (strict + hyperpop), tier-1 grinds (many slow beam runs over a thin pool)
× 2 segments × ~3 One-Each retries ≈ 100+ slow beam runs → 23 min. jazz/metal pools aren't starved, so
tier-1 resolves immediately and they finish fast. The 40s anchor never helps because the time is spent
in tier-1, not tiers 2-3.

## Fix design — one shared total-generation deadline
Replace the per-build relaxation anchor with a **single shared deadline** so NO path and NO retry can
collectively exceed the budget:

1. **Generation-level deadline.** Compute `_deadline = time.monotonic() + budget_s` ONCE at the start of
   the generation (in `core.py`, before the first pier-bridge build / One-Each loop). Thread it into
   `build_pier_bridge_playlist(..., deadline=_deadline)`. Inside, use the passed deadline instead of a
   fresh per-call `_pb_build_start`. The One-Each retries reuse the SAME deadline (no reset).
2. **Guard ALL the loops, not just tiers 2-3.** Check `time.monotonic() > deadline` at: the top of the
   segment loop (already), the top of tier-1's `for attempt` loop and the bridge-floor backoff loop,
   before micro-pier, and at the top of the One-Each retry loop in `core.py`. On exceed → stop relaxing
   / skip further attempts / bail that segment to the guaranteed-fill term-pool fallback (already exists,
   `pier_bridge_builder.py:270-321`).
3. **Budget value + knob.** Default total budget ~70s (headroom under the 90s ceiling for pool-build +
   Last.fm overhead, which happen before the build). Expose as a config knob
   `playlists.pier_bridge.generation_budget_s` (default 70.0) per "tunability over hardcoded behavior".
   Keep the existing `_SEGMENT_RELAXATION_BUDGET_S` behavior subsumed by the shared deadline.

This is a ROOT fix (bound every path + share one deadline across retries), not a cap on one path. The
guaranteed-fill fallback already produces a valid, diverse playlist when a segment is bailed.

## Verification
- Re-run strict/hyperpop via the harness → completes < 90s (the killer cell).
- jazz/metal unchanged (<90s, same pools — deadline never fires for them).
- Full unit suite green; goldens green.
- THEN resume Plan-1 Task-5 calibration (now tractable — every cell < 90s).

## Scope / merge note
Pre-existing master bug; fixing on `worktree-v6-canonical-wiring` (where we're working). It merges to
master with Plan 1. If production needs it sooner, cherry-pick after review. The earlier partial fix
`bc942d5` is already on master.

## State of Plan 1 when paused (for resume)
- Tasks 1-3 DONE + reviewed (sonic percentile, sparse genre percentile, min_pool backstop). Commits
  on the branch through `563f0ff`; calibration harness `3702a37`.
- Task 5 (calibration) PAUSED behind this budget fix.
- Task 4 (FINALE) still pending: per Dylan's decision **A**, DELETE `min_sonic_similarity` /
  `min_genre_similarity` entirely + migrate the 10 floor tests, AFTER calibration.
- Calibration signal so far (jazz/metal): percentile loosens the pool modestly; worst_edge unchanged —
  inconclusive until strict/hyperpop is measurable (blocked by this bug).
