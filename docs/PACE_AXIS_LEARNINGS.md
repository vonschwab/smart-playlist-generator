# Pace / Energy Axis — Learnings Log

> Living decision-journal for making pace/energy a first-class matching axis.
> Purpose: record what we tried, what failed, and *why*, so we don't repeat traps.
> Append-only by date; newest decisions at the bottom of each section. Eval artifacts
> (manifests, results.tsv) live under `docs/run_audits/pace_axis_eval/` (gitignored);
> this file is the tracked narrative.

## Vision (2026-06-18)

Pace/energy should be a **third co-equal matching axis: sonic (MERT) ⊗ genre ⊗ pace**,
not a bolt-on penalty. "Pace" = whatever *combination* of rhythm/BPM/onset/arousal/
danceability data best predicts that two tracks are pace-compatible. The combination is
an **empirical question** — settle it by testing (like the sonic MERT-vs-towers and genre
graph-vs-co-occurrence auditions), then wire the winner.

**Decomposition:**
1. **Pace-representation eval (current):** compare candidate representations/combinations;
   determine which best captures pace compatibility. Blind, independent ground-truth arm,
   eval-gated. Output = the validated representation (signals + weights + distance).
2. **Wire the validated pace axis (next):** integrate as a first-class axis fused with
   sonic & genre in matching + the beam. (The beam already has an arc-shaped per-step pace
   gate — BPM/onset bands as soft penalties — so wiring is "add the validated signal as a
   band / similarity term," structurally like the existing BPM/onset bands.)

## What the pace signals are, and what we've learned about each

| Signal | Source | Verdict | Evidence |
|--------|--------|---------|----------|
| Librosa **rhythm tower** (9-dim) | beat3tower towers | Weak alone; now rollback (MERT is default sonic) | rhythm-cosine pace gate underperformed; tower path inactive under `X_sonic_variant: mert` |
| **perceptual_bpm** | librosa.beat.tempo | Busted on beatless/drone | Georgia drone bpm=161.5 **byte-identical** to MBV wall-of-noise; trust-gated by onset_rate (shipped ad9403e) but not a cohesion signal on its own |
| **tempo_stability** | synthetic uniform beat grid | INERT / dead knob | ~0.96 for everything (1−CV of a uniform grid); 0.012% below the 0.5 bypass → never fires |
| **onset_rate** | onset envelope | Reliable beat-PRESENCE, but ≠ perceived energy | Georgia 0.024 vs Two Trains 3.50; separates drone from beats, not intensity |
| **LUFS loudness** | pyloudnorm | NOT energy (mastering-confounded) | calm drone Georgia reads 2nd-loudest (−7.3); intense MBV reads quietest (−11.9) → do NOT use loudness as the energy axis |
| **arousal** (emoMusic, p10/p50/p90) | Essentia msd-musicnn | VALIDATED — "intensity/activity" | 34-track pre-registered probe: 0 inversions; partly tempo-driven (slow-doom Black Sabbath reads MID 4.71 not HI) |
| **danceability** | Essentia msd-musicnn | VALIDATED — "groove/beat-presence" | cleanest single separator: dancefloor 0.98+ vs beatless ≤0.56, true ambient SOTL 0.21; passed traps (Max Richter strings 0.09, Marvin slow-groove 0.95) |

**Key representation finding:** arousal and danceability are **orthogonal** (intensity vs
groove) — likely both needed. **Mean masks dynamics** (post-rock builds: wide arousal
p10–p90) → the sidecar stores the distribution, not just the mean.

## Eval-methodology guardrails (from the evaluation-methodology skill)

- Full pool, not samples. Verify provenance of both arms (no A-vs-A).
- **Blind the perceptual arm + include a decoy** to measure discrimination.
- **Non-circular:** don't validate a representation with a metric computed from that same
  representation. The independent arm must be human ears or held-out labels.
- Distributions (min/p10/p50/p90), not means. State N. Don't overgeneralize from a dev subset.
- Experiments write to `docs/run_audits/<exp>/`, never production artifact paths.

## Process traps hit this session (don't repeat)

- **Subagent committed to the wrong checkout** (main `master` instead of the worktree) —
  always give subagents a branch-check guard; verify the commit landed on the worktree branch
  before reviewing.
- **`web/dist` is gitignored** — rebuild it in the target checkout after a front-end change;
  it won't merge.
- **Worktree cleanup**: remove junctions (`web/node_modules`, and never a `data/artifacts`
  junction) with `rmdir` BEFORE deleting a worktree, or a recursive delete nukes the target.
- **pytest never piped** through tail/head (hook-enforced).

## Open questions

- Role: **similarity** (pairwise pace-compatibility, fused like sonic/genre cosine) vs
  **arc** (whole-sequence pace trajectory) vs both. Working assumption: similarity primary,
  arc secondary. (confirming)
- Independent ground-truth arm for the eval (the crux — TBD this brainstorm).

## Decisions

- 2026-06-18: Energy descriptor sidecar shipped as an `analyze_library` stage (commit 4f4031f).
  PRODUCE-only; consumption is this effort.
- 2026-06-18: Adopt the two-sub-project decomposition above (eval first, wire second).
