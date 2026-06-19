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
- Gold corpus contents — need flow-sequenced albums spanning energy REGISTERS, not just dance
  (see decision below). Collaborative w/ Dylan.

## Decisions

- 2026-06-18: Energy descriptor sidecar shipped as an `analyze_library` stage (commit 4f4031f).
  PRODUCE-only; consumption is this effort.
- 2026-06-18: Adopt the two-sub-project decomposition above (eval first, wire second).
- 2026-06-18: **Ground-truth arm = adjacency in continuously-mixed / DJ-sequenced albums**
  (Dylan: Beyoncé–Renaissance, Jessy Lanza–DJ-Kicks, etc.), NOT arbitrary albums. In a
  continuous mix, adjacency is intentional pace-compatibility → strong truth, authored
  independently of our features.
  - **Control (de-confound + fine-grained test):** compare within-album **adjacent** pairs
    vs within-album **non-adjacent** pairs (same artist/genre/sonic, larger intended pace
    step) vs **random cross-album** pairs. A good representation ranks
    adjacent > non-adjacent-same-album > random. The within-album non-adjacent control is
    what isolates *pace* from "same record sounds alike" (sonic/genre confound).
  - **Register coverage trap:** if the corpus is ALL dance mixes, the eval just re-proves
    "danceability separates dance from non-dance" (already known). Corpus MUST span registers:
    high-energy continuous (house/dance mixes), LOW-energy continuous (ambient/drone works
    sequenced to flow — Stars of the Lid, Eno, GAS/Wolfgang Voigt), and mid (a flowing
    post-rock or hip-hop record). Then the eval tests calm↔calm and energetic↔energetic
    matching, i.e. a real pace *gradient*, not a dance binary.
  - Plan to also run a small **blind human pairwise/odd-one-out** check (Dylan's ears) on
    edge cases as the perceptual confirmation that the album-proxy isn't lying.
- 2026-06-18: **TRAP — "well-sequenced album" ≠ "adjacency is pace-compatible".** Dylan's
  candidate corpus (`~/Downloads/playlist_gold_corpus_candidates.md`, ~70 albums) is excellent
  but conflates two ground truths: (1) **continuous/tight-flow mixes** where adjacency =
  pace-COMPATIBLE by construction (Renaissance, Daft Punk–Discovery, The Avalanches,
  This Is Happening, J Dilla–Donuts, dub, continuous ambient like SAW II / Eno–On Land /
  Stars of the Lid / Hiroshi Yoshimura) — these are valid **similarity positives**; vs
  (2) **arc/contrast albums** sequenced for intentional dynamic *change* (Kid A "pacing arc
  across beat/no-beat", Fleet Foxes "large dynamic arcs", Black Sabbath "big dynamic shifts",
  Sophie "huge energy jumps that still belong", Unwound, Godspeed, Flaming Lips) — here adjacent
  pairs deliberately DIFFER in pace, so they are **false positives for a similarity metric** and
  belong instead to the **arc/trajectory** gold. FIX: tag every corpus album `flow_type ∈
  {tight_continuous, gradient_flow, arc_contrast, flat_uniform_mix}`; use tight/gradient as
  similarity positives, reserve arc_contrast for the (later) arc eval. This also concretely
  resolves the similarity-vs-arc split: the corpus supplies BOTH gold sets.
  - Sub-trap: **flat-uniform mixes** (a Renaissance-style record that stays in one pace pocket
    throughout) make the within-album non-adjacent control weak (non-adjacent pairs are ALSO
    compatible) → for those, only the binary test (adjacent-in-mix close vs random far) applies;
    the 3-tier control needs **gradient_flow** albums that traverse pace.
  - Minor data caveats: SAW II has duplicate/expanded rows (dedup needed, est_usable=nan);
    long-form suites (Godspeed, Fela, In A Silent Way) have too few/too-long tracks for pairwise
    adjacency (exclude or long-track-bench only); 4–6-track albums are thin for the non-adjacent
    control.
- 2026-06-18: **Product target = a pace SLIDER (pace_mode: strict/narrow/dynamic/off), like
  sonic & genre.** Dylan's framing of the behavior:
  - **Always-on FLOOR (all modes): avoid big swings between disparate energies/BPMs** between
    adjacent tracks. This is the adjacency-smoothness / similarity floor — the core thing the
    representation must get right.
  - **Gradient/arc adherence SCALES with the slider** — strict enforces a tighter pace gradient
    across the pier-bridge; dynamic/off relax the *gradient*, but the anti-big-swing floor still
    applies. (Not every bridge needs an exact gradient in every setting.)
  - **Eval implication:** the representation must yield a **graded, continuous pairwise distance**
    (not a binary class), because the slider sets per-mode thresholds/penalty strengths on that
    distance. So the eval scores each candidate on: (i) rates flow-adjacency CLOSE, (ii) rates
    cross-register/big-swing pairs FAR, (iii) monotonic in between — i.e. distance tracks
    *degree* of pace difference, not just same/different.
  - Maps to sub-project 2: validated distance → pace_mode per-mode knobs (floor penalty always
    on; gradient/arc weight strict→off), fused alongside sonic & genre.
- 2026-06-18: **Eval is multi-pass, coarse→fine (Dylan: "testing should narrow over the passes").**
  - Pass 1 (broad, cheap, automated): ALL candidates vs album-adjacency ranking on the full
    corpus; drop clear losers. No human.
  - Pass 2 (narrow): top ~2–4 candidates get the 3-tier within-album control (gradient_flow
    albums), per-register breakdown, monotonicity, + the small **blind human pairwise/odd-one-out**.
  - Pass 3 (confirm/tune): only if warranted, fit `pace_tuned` weights (cross-validated) on the
    finalist feature set; final blind confirmation. `pace_tuned` deferred out of Pass 1 (overfit risk).
  - Album selection + `flow_type` tagging DELEGATED to controller (register-balanced subset of the
    candidate list; tight_continuous + gradient_flow as similarity positives). Candidate menu ratified.
- 2026-06-18: **Spec + plan written, harness built (subagent-driven TDD), PASS 1 RUN.** Harness:
  `scripts/research/pace_eval_{metrics,corpus,features,run}.py` (+ unit tests, 10 pass; ruff/mypy
  clean). Spec/plan under `docs/superpowers/{specs,plans}/2026-06-18-pace-representation-eval*`.
  Corpus = 12 albums (Avalanches dropped — deluxe disc-track filename "1-NN" defeats the leading-int
  track parser; revisit w/ disc-aware parse if Pass 2 wants more high-tight data). N = 158 corpus
  tracks → 146 adjacent / 372 non-adjacent (gradient) / 2000 cross-register pairs. Artifacts:
  `docs/run_audits/pace_axis_eval/` (gitignored; copied to main checkout to survive worktree teardown).
- 2026-06-18: **PASS 1 RESULT (automated screen; AUC = P(more-compatible pair ranked closer)).**
  NOTE: the final code review caught a confound — the original hard discriminator mixed ALL-album
  adjacent positives against gradient-only non-adjacent negatives, which inflated/mis-ranked it
  (it had falsely shown "danceability best on hard 0.638"). FIXED to gradient-only adjacent vs
  gradient non-adjacent (`adjacent_gradient`), and re-run. Corrected:
  ```
  candidate       auc_adj_vs_random(COARSE floor)  auc_adj_vs_nonadj(FINE, gradient-only)
  energy_dist          0.797                            0.553   <- leader (coarse)
  energy_pair          0.775                            0.549
  arousal_p50          0.771                            0.547
  energy_onset         0.735                            0.528
  pace_full            0.727                            0.525   <- kitchen-sink, DILUTED
  rhythm_tower         0.724                            0.511
  beat_strength        0.718                            0.533
  danceability         0.694                            0.524
  onset_rate           0.631                            0.523
  perceptual_bpm       0.583                            0.510   <- near-random (confirms BPM weak)
  ```
  KEY FINDINGS: (1) **Energy decisively wins COARSE pace compatibility** (the always-on "avoid big
  swings" floor): energy_dist/energy_pair/arousal_p50 = 0.77–0.80; BPM near-random (0.583),
  onset/rhythm-tower weak. (2) **FINE within-album gradient ordering is near-chance for EVERY
  candidate (~0.51–0.55)** once apples-to-apples — BUT this may be a near-degenerate test (within one
  flowing album the real pace step is tiny), not feature failure. **Pass 2's blind human arm must
  first establish whether fine within-album pace ordering is even perceptible** before judging
  representations on it; if not, the COARSE floor is the real product signal. (3) kitchen-sink
  `pace_full` dilutes (< focused energy). (4) The confound lesson: a mixed positive set silently
  re-ranks candidates — exactly what the final-review gate is for.
  ADVANCING TO PASS 2: energy_dist, energy_pair, arousal_p50 (coarse-floor leaders). CAVEAT: Pass 1
  is the automated album-adjacency proxy only — **no verdict until Pass 2's blind human arm**.
  Full detail: `docs/run_audits/pace_axis_eval/findings_pass1.md`.
