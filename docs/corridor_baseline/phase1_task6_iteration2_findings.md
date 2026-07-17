# Phase 1 Task 6 remediation, iteration 2 — empirical improvement gate (2026-07-17)

Follow-up to `phase1_task6_remediation_findings.md` (iteration 1). Full
evidence/method in `.superpowers/sdd/p1-task6-remediation-report.md`'s
"Iteration 2 appendix" (gitignored — Dylan's copy). This note is the
committed, permanent record.

## Why iteration 1 was replaced

Iteration 1's gate predicted "healthy anchor support (>=0.5) means a weak
edge is beam-path-internal, skip widening." Falsified by real evidence:
Alex G/home's segment 1 had support ~0.8 (well above the threshold) yet
still gained +0.42 T from one widen attempt (0.189 -> 0.611). A wider pool
can unlock better beam-path combinations no anchor-only metric predicts.
`corridor_widen_support_threshold` never shipped past dev; retired
entirely.

## The fix

Empirical continue-gate, no prediction. The ladder always tries widen
attempt 1 unconditionally once the quality trigger fires. After that, it
widens further only if the attempt just run improved the best-seen
min_edge_T by more than a new knob, `corridor_widen_improvement_epsilon`
(default 0.02). A non-improving attempt stops the ladder, accepts the
best-seen path, and tags `widen_stopped_early: true`. Hard infeasibility
(no path) always widens to the full attempt budget. Pure decision helper
rewritten in `src/playlist/pier_bridge/corridor.py`; 10 unit tests cover
the new branch table. One real bug caught before the A/B ever ran: an
overly defensive `-inf` guard incorrectly treated "recovered from
infeasibility" as no-improvement — fixed (Python's `finite - (-inf) = +inf`
already gives the correct signal).

## 4-way A/B (legacy / AB1 no-gate / AB2 support-threshold / AB3 empirical)

Full table: `docs/corridor_baseline/phase1_corridor_ab3.json` (12/12 cells
clean, `err=None`, F7 health-line contract 72/72 segments verified).

- **THE bar — Swirlies/home `below_floor`: HOLDS** across all three A/Bs.
- **min_T aggregate bar: still FAILS** (8/12 flat-or-better vs legacy) but
  **Alex G/home's regression is gone** — min_T restored to 0.586, exactly
  matching AB1 (the pre-iteration-1 value). The rescue segment recovers via
  a single widen attempt, byte-identical to AB1's original behavior.
- **distinct ±2 bar: still FAILS** (5/12), pre-existing, not targeted.
- **wall ≤2x bar:** 4/12 cells over 2x (down from AB1's 6/12, up from AB2's
  2/12) — total corpus wall 386.3s, between AB1 (474.0s) and AB2 (332.6s),
  exactly matching the predicted "2 beam runs not 3" shape for segments
  whose weakness turns out to be beam-path-internal.

## Verified against real logs

- **Alex G/home rescue restored, confirmed.** Attempt 1 alone jumps
  0.189 -> 0.611 (well past epsilon), recovering the segment exactly as
  AB1 did.
- **SADE/home's exhausted segments correctly stop after 1 attempt**
  (improvement ~0.000 or negative in all 4), landing at the same min_T=0.606
  via the repair stack either way — wall cost sits between AB1 and AB2 as
  predicted.
- **Swirlies/home unchanged (0.459 in all three A/Bs), as predicted.**
  Swirlies/open, however, reverts to AB1's value (0.357) — NOT AB2's
  unexplained improvement (0.659). That improvement was flagged as an
  unconfirmed hypothesis in the iteration-1 findings and does not survive
  into iteration 2 — evidence it was a side effect specific to iteration
  1's particular skip pattern, not a durable quality gain.

## Verdict

The regression iteration 1 introduced (Alex G/home) is fixed, and the
wall-clock cost lands as predicted between the no-gate and support-gated
baselines. The two pre-existing corridor-vs-legacy gaps (min_T aggregate,
distinct ±2) remain open — neither iteration targeted them. **Hold point —
reported to Dylan, not advanced to Task 7.**
