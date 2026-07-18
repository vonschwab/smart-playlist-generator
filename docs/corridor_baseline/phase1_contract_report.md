# Phase 1 contract report — corridor-first pooling

**Date:** 2026-07-18 · **Author:** Task 9 (final validation gate) · Branch
`corridor-phase1-pooling` @ 639409a (this task's commits land on top).
Primary gate: `docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md` (categories
A-F below). Aggregate-quality bars: spec §6
(`docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md`).

This is the merge-gate summary Task 9's brief asks for: every contract
category with a verdict and an evidence pointer, the amended-bars history,
and known issues carried into Phase 2 / the listen test. It does not
re-derive evidence — every claim below cites the report or JSON that
actually measured it.

## Category verdicts

| Cat | Scope | Verdict | Evidence |
|---|---|---|---|
| A | Transforms & identity (artist normalization, diacritics, collab inheritance, alias/sibling linking, version-dedup, genre authority, computed-identity counting) | **PASS** | `.superpowers/sdd/p1-task-7-prep-report.md` §1: zero code-surface hits on any Category A module (`git diff 0d01285..HEAD --name-only`); 5-table empirical paired capture, every diff fully attributed to live-DB growth (+206 tracks across the ~1-day gap), zero unexplained deltas. `dedup_collapse` verified content-diffed, not just row-counted. |
| B | Hard gates (recency, blacklist, duration, title hygiene, sonic/genre/BPM/onset admission, popularity gate, BPM-trust-on-beatless) | **RESEATED, GREEN** | Recency/blacklist/duration/title hygiene stayed invariant exclusion sets (`eligible_universe.py`, same math as `candidate_pool.py`, Task 2 report). Sonic/genre/pace admission reseated onto corridor width + relevance mask + beam bands — dial-differentiation audits confirm no detent inert on any of the 3 axes: range/flow/pace (Task 8 report §7) and the per-mode sonic-width axis specifically (`.superpowers/sdd/p1-permode-width-report.md` dial audit, "no detent pair has identical resolved widths"). Popularity/bangers gate reseated onto the corridor universe (Task 5 reseat 1, `.superpowers/sdd/p1-task-5-report.md`). |
| C | Soft scoring terms (C1-C11: duration penalty, anti-center, popularity penalty, local-sonic-edge, genre soft penalty/tiebreak/pair-penalty, progress monotonicity, BPM/onset soft penalty, instrumental lean, cohesion weights) | **PASS-AS-AMENDED** | Full 413-checkpoint sweep initially found 13 fields regressing `changed→inert` (`docs/corridor_baseline/phase1_contract_knob_verdict.md`) — all 13 root-caused via engineered probes to category (a) CONDITIONALLY LIVE (the 2 sampled cells' corridor sizes/settings just never trigger the firing condition; zero were silently-dropped wires, zero were downstream clobbers). C1 (duration) specifically was GENUINELY BROKEN under corridor at first measurement (hardcoded no-op, `.superpowers/sdd/p1-task-7-prep-report.md` §5) — fixed via a beam-side rehome (`.superpowers/sdd/p1-task-7-c1beam-report.md`, GREEN) plus a title-hygiene wiring fix found in the same pass (`.superpowers/sdd/p1-task-7-c1fix-report.md`). C10 (instrumental lean) was GREEN on first measurement. |
| D | Topology & structure invariants (pier-bridge topology, mini-pier spacing, variable bridge length, fire-mode piers, pier-bridgeability veto, tail-DP, edge repair/delete/break-glass, weak-edge cascade, roam corridors, beam minimax, mini-pier promotion) | **GREEN, D14 IMPROVED** | Existing regression suite exercises every structural invariant unchanged (`test_mini_pier_integration.py`, `test_var_bridge_integration.py`, `test_edge_delete.py`, `test_edge_repair_break_glass.py`, all green in this task's full-suite run). D14's explicit "must IMPROVE" bar: Swirlies/home's mini-pier-armed crater went from `below_floor=4` (legacy, T=0.018) to `below_floor=0` at every phase gate since Task 6 (`.superpowers/sdd/p1-task-6-report.md` through this task's crater regression tests, below) — the clearest structural improvement in the whole contract. |
| E | Steering (taxonomy genre-arc, tag steering stage 1/2b, genre-arc floor) | **GREEN** | `test_genre_steering_integration.py` (arc-monotonic, Smiths edge-coherence-improves-with-steering, Charli narrow-still-feasible) and the `test_tag_steering_*`/`test_gui_fidelity_regressions.py` tag-first-pier/phase-B-anchor suites all pass in this task's full-suite run; genre_mode's production wiring (a Task 4→5 gap) closed and covered by `test_corridor_genre_mode_threads_through_production_wiring`. |
| F | Reporting & diagnostics (transition stats, weakest-edge report, distinct-artist count, per-playlist log, run audits, gate tallies, corridor health line) | **GREEN** | F7 (the new corridor health line) has dedicated coverage (`test_corridor_pooling_generation_completes_with_corridor_membership_and_health_line`'s health-line assertions, independent of this task's xfail on its membership recheck — see Known Issues). F1-F3 (transition stats/weakest-edge/distinct-artists) are present in every `playlist_stats["playlist"]` dict, exercised by every real-artifact test in the suite and by this task's new crater regression tests. F4-F6 unchanged (not corridor-touched). |

**Merge-gate checklist** (`CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md`'s 5 conditions):
1. Category A byte-identical — **met**.
2. Category B/C GREEN — **met (C via the amended-bars process below)**.
3. (Sweep harness health: 0 `override_failed`/`error` in 413 checkpoints — **met**.)
4. Category D structural fingerprints unchanged except D14 improve — **met**.
5. Category F every diagnostic still emitted — **met**.

## The amended-bars history

The spec's §6 aggregate-quality bars (`below_floor=0`, `min_T` flat-or-better
≥10/12, `distinct_artists` ±2, `wall` ≤2×) were never cleared as originally
written — every phase's own report says so plainly, and each amendment was a
**Dylan-adjudicated decision on real evidence**, not a unilateral relaxation:

1. **Task 6 gate (2026-07-17): HOLD, then Dylan-accepted amended bars.**
   First AB gate: THE bar (`below_floor=0/12`, Swirlies/home crater fixed
   4→0) passed clean; the three secondary bars (`min_T` aggregate,
   `distinct_artists`, `wall`) failed. Two remediation iterations followed
   (`.superpowers/sdd/p1-task6-remediation-report.md`): iteration 1's
   predictive support-threshold gate was **tried and falsified** by a real
   cell (Alex G/home lost a genuine rescue) and deleted before shipping;
   iteration 2's empirical improvement-gate (`corridor_widen_improvement_epsilon`)
   restored the rescue. Final AB3: THE bar + below_floor 0/12 + err/backstop/
   health PASS; `min_T` 8/12 (3 deltas within demonstrated ±0.09 noise);
   `distinct` 5/12 out (mostly increases on rescued cells); `wall` 4/12 over
   2× (worst 69s absolute — a stressed-cell cost, not a runaway). **Dylan
   accepted this amended bar set 2026-07-17** (`.superpowers/sdd/progress.md`:
   "DYLAN ACCEPTED Task 6 gate (amended bars) 2026-07-17. Proceeding Task 7.").
2. **The restrict_bundle discovery (Task 8, 2026-07-17/18).** The flip's
   own validation corpus (not a pre-planned check) surfaced a real bug: the
   `build_balanced_candidate_pool` deletion left Artist mode's
   `allowed_track_ids` still hard-clamping the bundle upstream of corridor's
   universe build — corridor had been scanning a small pre-restricted slice,
   not the library, for every Artist-mode generation since the flip started.
   Root-caused (DB/artifact drift and a `candidate_pool.py` code diff both
   ruled out byte-identical) and fixed: `allowed_track_ids` kept for the
   empty-pool guard/diagnostics, but `None` now passed to the actual DS call,
   so corridor genuinely scans the ~43k-track library. `below_floor` restored
   0/12; `admitted` counts jumped 2-15x (`.superpowers/sdd/p1-task-8-report.md`
   §3). This was flagged, not silently absorbed: it invalidated AB3 as a
   clean before/after baseline for Artist-mode `min_T` comparison (§6 of that
   report), since the fix is a second, legitimate widening beyond "flip the
   flag" — the ±0.02 tolerance bar doesn't cleanly apply to 10/12 cells
   post-fix, but `below_floor=0/12` and zero errors (the bars that validly
   compare like-for-like) both hold.
3. **Two width recalibrations, both evidence-driven.** (a) The width re-pin
   (`.superpowers/sdd/p1-width-repin-report.md`): post-restrict_bundle-fix,
   the flat 0.85 pin (calibrated against the old amputated universe) cratered
   SADE/home to 0.374 against the new full-library universe; re-probed and
   re-pinned to a flat 0.95. (b) The per-mode width mapping
   (`.superpowers/sdd/p1-permode-width-report.md`, spec §4, pulled forward
   from Phase 2 by **Dylan's 2026-07-18 decision**): a single flat percentile
   couldn't serve both strict and dynamic sonic modes over the same
   library-wide universe (SADE/home cratered again at flat 0.95 under
   `strict`, while `dynamic` cells were fine at the same value) — replaced
   with `corridor_width_percentile_{strict,narrow,dynamic,discover}`, each
   independently probed (strict/dynamic) or interpolated (narrow/discover,
   provisional).

## Aggregate-quality bars — current state (12-cell corpus)

Per `.superpowers/sdd/p1-permode-width-report.md`'s final corpus
(`docs/corridor_baseline/phase1_final_corpus2.json`), the most current
capture on the branch:

| Bar | Result |
|---|---|
| `below_floor` 0/12 | **PASS** — holds at 0/12 (Swirlies/home included) across every gate since Task 6 |
| `min_T` ≥10/12 flat-or-better | **Amended/accepted** — 5/12 flat-or-better against the ancient Phase-0a baseline (a bar no phase has cleared since Phase 1 began per the report's own accounting); vs the immediately-prior flat-0.95 corpus specifically (the correct like-for-like comparison for the per-mode-width task), every open cell is byte-identical and 4/6 home cells improve |
| `distinct_artists` ±2 | **Amended/accepted** — 7/12 exceed vs Phase-0a; mostly increases on previously-starved (rescued) cells, not diversity collapse |
| `wall` ≤2× | **Amended/accepted** — 2/12 exceed (both Swirlies, the hardest cell in the corpus, ladder-attributed); corpus total 1.05x in aggregate |

**Binding win:** `below_floor=0/12` — every phase's own report treats this as
THE bar (spec §1's stated purpose: fix the SADE+home / Swirlies+home
craters). It has held at every single gate since Task 6, including through
the restrict_bundle fix and both width recalibrations. The two dedicated,
golden-independent regression tests added by this task
(`tests/integration/test_corridor_crater_regressions.py`) pin exactly this —
`min_T >= transition_floor` and `below_floor == 0` for hand-picked multi-pier
SADE and Swirlies "home" fixtures, both currently green.

## Known issues (carried into Phase 2 / the listen test)

- **SADE/home plateaus at 0.454 under every tested strict width** (0.985,
  0.99, 0.995 all tried; 0.985 is best and is what's live). Tightening the
  corridor does not monotonically help this cell — its problem is not purely
  "corridor too wide." Mechanism hypotheses for Phase 2 (unresolved, flagged
  not diagnosed): genre relevance-mask interaction, specific candidate
  scarcity in this artist's sonic neighborhood, or a beam-path issue the
  width knob alone can't reach. See `.superpowers/sdd/p1-permode-width-report.md`
  Concerns.
- **`narrow`/`discover` sonic-mode widths are provisional interpolations**
  (midpoint and dynamic−0.02 respectively), not directly corpus-probed —
  only `strict`/`dynamic` had dedicated probe cells per the brief's scope.
  The dial-range audit confirms they resolve to distinct, sane, differently-
  sized corridors, but their quality profile (min_T, distinct_artists) is
  unverified against a real corpus.
- **`progress_arc_weight` tuning note** (`.superpowers/sdd/progress.md`,
  Task 7 entry): the config default (0.25) is too weak against corridor's
  tighter beam; live behavior currently runs it effectively at 3.0. Flagged
  for a Phase 2 default-value revisit, not changed here (out of this task's
  scope — a config *default* change is a tuning decision, not a test fix).
- **`below_floor==0/12` is the binding win** — repeated here deliberately:
  every other aggregate bar is amended/accepted against an increasingly
  stale baseline (Phase-0a, captured before the restrict_bundle fix widened
  the whole universe); `below_floor` is the one number every report, every
  phase, treats as non-negotiable, and it has never regressed.
- **Two STOP-and-report findings from this task's validation pass** (real,
  deterministic, NOT engine-fixed per the task's scope — see
  `.superpowers/sdd/p1-task-9-report.md` for the full writeups and the two
  `xfail`-marked tests):
  1. A diagnostics-staleness bug in the corridor-widening-ladder ×
     variable-bridge-length interaction: the once-per-segment health-line/
     diagnostics gate can latch onto a different (narrower) corridor attempt
     than the one that actually supplied the segment's emitted tracks.
     Affects log/diagnostic fidelity only — every emitted track is still a
     legitimate corridor member of *some* attempt, not a candidate-legality
     bug.
  2. A real, deterministic cross-segment `min_gap` violation
     (`artist_spacing=strong`) on a mini-pier-subdivided multi-pier fixture —
     the exact failure class commit 8101ac1 previously fixed, now
     reappearing. Not root-caused to a definitive fix; higher severity than
     finding 1 (diversity enforcement is a Layer 2 "enforce, don't
     recommend" architectural commitment, not just a diagnostics issue).

## Full-suite totals (this task, all markers)

`python -m pytest -q`: **3 failed, 2540 passed, 19 skipped, 2 xfailed**,
634.04s. All 3 failures are environment-only (2 pre-existing satellite-vs-
canonical identity checks + 1 satellite-local config gap — this satellite's
gitignored `config.yaml` omits `instrumental_penalty_weight`, confirmed via
a direct override that the mechanism works correctly when configured); zero
are corridor-touched. `ruff check .` and `mypy` on every touched file: clean.
Full breakdown, the 3 broken-test dispositions, and the slow-suite triage:
`.superpowers/sdd/p1-task-9-report.md`.
