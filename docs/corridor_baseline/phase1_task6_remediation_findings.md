# Phase 1 Task 6 remediation — scarcity-gated widening ladder (2026-07-17)

Follow-up to the Task 6 HOLD (`phase1_task6_findings.md`). Full evidence/method
in `.superpowers/sdd/p1-task6-remediation-report.md` (gitignored — Dylan's
copy). This note is the committed, permanent record.

## The fix

Root cause (traced at Task 6 gate time): the widening ladder widened even
when a weak edge's cause was beam-path-internal, not pool-limited — paying
3x beam cost per segment for zero improvement before the repair stack fixed
the edge anyway (e.g. SADE/home: 4/6 segments exhausted at healthy support
0.66-0.95, final min_T unaffected). Fix: gate widening on the existing
Phase-0a anchor-support coverage metric — `min(support_a, support_b) <
corridor_widen_support_threshold` (new knob, default 0.5) — evaluated once
when the quality trigger first fires. Below threshold: widen as before.
At/above threshold: skip widening, accept the initial-width path, tag
`widen_skipped: true`, hand off to the repair stack. Hard infeasibility (no
path) always widens. Pure decision helper + `CorridorWidenDecision` enum in
`src/playlist/pier_bridge/corridor.py`; 8 unit tests cover all branches.

## 12-cell 3-way A/B (legacy / AB1 pre-fix / AB2 post-fix)

Full table: `docs/corridor_baseline/phase1_corridor_ab2.json` (12/12 cells
clean, `err=None`, `always_on_patterns_never_seen: []`, F7 health-line
contract 72/72 segments).

- **THE bar — Swirlies/home `below_floor`: HOLDS** (4 legacy -> 0/0 in both
  AB1 and AB2).
- **min_T aggregate bar (>=10/12 flat-or-better vs legacy, no cell >0.03
  worse): still FAILS**, essentially unchanged from Task 6 (7/12
  flat-or-better; Alex G/home, Alex G/open, SADE/home, SADE/open exceed
  tolerance) — this gap predates the remediation; 3 of the 4 failing cells
  have identical AB1 and AB2 values.
- **distinct +/-2 bar: still FAILS**, on 3/12 cells (down from 4/12 in AB1)
  — pre-existing corridor characteristic, not targeted by this fix.
- **wall <=2x bar: mostly HOLDS now.** 6/12 cells over 2x in AB1 (worst
  4.35x) -> 2/12 in AB2 (Swirlies/home 4.04x, Swirlies/open 2.27x — both
  genuinely pool-starved, correctly still widening). Total corpus wall:
  474.0s (AB1) -> 332.6s (AB2), a 30% reduction (legacy floor: 242.9s).
- **err/backstop/health-lines: HOLDS.**

## Wall-tax shift, confirmed at the log level

SADE/home: AB1 widened 12x/exhausted 5 segments to reach min_T=0.606 via
repair; AB2 skips the same 4 weak segments outright (0 widen attempts) and
lands at the **identical** min_T=0.606 — per-segment logs show the AB1
widening added nothing there. Wall dropped 37.2s -> 20.3s. Several other
cells (The Strokes/home/open, Aaliyah/home/open) show the same shape.

## Regression found: Alex G/home

**One cell regressed materially: min_T 0.586 (AB1) -> 0.442 (AB2), a real
-0.144 loss**, alongside a wall-clock win (67.3s -> 34.3s). Cause: AB1's
segment 1 was a genuine widen-rescue (min_edge_T 0.189 -> 0.611 after one
widen attempt) at support 0.79-0.83 — comfortably above the 0.5 threshold.
AB2 skips that exact segment (support >= 0.5 by construction) and the
repair stack does not fully compensate. Bill Evans Trio/open shows a
smaller version of the same pattern (min_T -0.012, within tolerance, but
wall got *worse*: 12.7s -> 28.7s). This falsifies part of the design's
causal assumption for these specific segments: support well above 0.5 does
not reliably mean the weakness was purely beam-path-internal.

## Verdict

Net effect is strongly positive on wall clock (cells-over-2x 6 -> 2, total
corpus time down 30%) with THE bar and err/health bars holding, but **not
lossless** — Alex G/home is a genuine, measurable quality regression this
loop introduced. **Hold point — reported to Dylan, not advanced to Task 7.**
Candidate follow-ups (not implemented): raise the support threshold toward
~0.85-0.9, or make SKIP conditional on the repair stack's own candidate
pool actually containing a viable replacement rather than a single static
pre-beam threshold.
