# Pace-representation eval — design

**Date:** 2026-06-18
**Status:** Approved (design)
**Author:** session w/ Dylan
**Related:** `docs/PACE_AXIS_LEARNINGS.md` (living journal), `project_energy_feature_exploration` memory

## Problem & goal

We want pace/energy to become a **third co-equal matching axis** (sonic ⊗ genre ⊗ pace).
"Pace" is some *combination* of our rhythm/BPM/onset/arousal/danceability data — but which
combination, with what weights and distance, actually captures **pace compatibility** between
two tracks is an empirical question. Librosa rhythm alone was weak; BPM is busted on beatless
audio; onset is beat-presence not energy; LUFS is mastering-confounded; Essentia arousal +
danceability validated well (34-track pre-registered probe, 0 inversions). This sub-project
**finds the best pace representation by testing**, before any wiring.

**Output:** a ranked verdict naming the winning representation (signals + weights + distance)
and its measured separation/monotonicity, to be wired as the pace slider in sub-project 2.

## Scope

**In:** a research harness that compares candidate pace representations against a
ground-truth corpus and reports which best captures pace compatibility. Corpus selection +
tagging, the candidate set, the multi-pass metric, and the findings write-up.

**Out (sub-project 2):** wiring the validated representation into `pace_mode` (the pace
slider) / the beam / candidate matching. No production code paths change here. Also out: the
**arc** eval (whole-sequence trajectory) — its gold set (`arc_contrast` albums) is reserved.

## Product target this serves (context, not built here)

A **pace slider** (`pace_mode`: strict / narrow / dynamic / off), like the sonic & genre
sliders. Required behavior:
- **Always-on floor (all modes): avoid big swings** between disparate energies/BPMs on
  adjacent tracks. This is the similarity/adjacency-smoothness core.
- **Gradient/arc adherence scales with the slider** (strict tightest; dynamic/off relax the
  gradient but keep the anti-big-swing floor).
- Therefore the representation must yield a **graded, continuous pairwise distance** (not a
  binary), so the slider can set per-mode thresholds/penalty strengths on it.

## Key facts (verified this session)

- The beam's pace gate is already **arc-shaped**: per-step moving target interpolated
  pier→pier for BPM (`compute_step_log_bpm_target`) and onset, applied as soft penalties into
  `_pace_penalty` → subtracted from `combined_score` (beam.py ~1052–1200). With MERT as the
  default sonic variant the librosa **rhythm tower is rollback**, so pace falls back to the
  perceptual-BPM + onset bands. This is why `pace_mode` is weak today.
- Pace arrays load in `pipeline/core.py` via `bpm_loader.load_bpm_arrays` (perceptual_bpm,
  tempo_stability, onset_rate) and thread into the beam — the plumbing sub-project 2 mirrors.
- **Energy sidecar** (shipped 4f4031f) at `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz`:
  `track_ids` + `arousal_p10/p50/p90` + `danceability` + `valence` + `frames`, aligned to the
  artifact index, 40392/40393 filled. `tempo_stability` is inert (do not use).

## Design

### Ground-truth corpus (selected + tagged here)

Truth = **adjacency within continuously-mixed / flow-sequenced albums is pace-compatible by
construction**. `arc_contrast` albums (sequenced for intentional dynamic *change*) are
**excluded** — their adjacencies are big swings and would be false positives for a similarity
metric. Corpus is **register-balanced** so we test calm↔calm and energetic↔energetic, not a
dance binary.

`flow_type`: `tight_continuous` (one pace pocket, gapless/beatmatched), `gradient_flow`
(flows but traverses pace — best for the within-album control), `flat_uniform_mix` (binary
test only). Selected subset (~13, from Dylan's candidate list; `usable_range` honored):

| album | register | flow_type | role |
|-------|----------|-----------|------|
| Beyoncé – RENAISSANCE (explicit) | high | flat_uniform_mix | clean high-energy continuous positives (binary) |
| Daft Punk – Discovery | high→mid | gradient_flow | house/disco with energy shifts (3-tier control) |
| The Avalanches – Since I Left You (disc 1, tr 1–18) | mid-high | tight_continuous | sample-collage continuous |
| LCD Soundsystem – This Is Happening (tr 1–9) | high | gradient_flow | long-form dance-rock builds |
| Caribou – Swim (tr 1–9) | mid-high | gradient_flow | electronic, traverses |
| J Dilla – Donuts | mid | tight_continuous | short adjacent beat transitions |
| D'Angelo – Voodoo | mid | gradient_flow | neo-soul groove pocket |
| King Tubby – Dub From the Roots | mid-low | tight_continuous | steady dub pulse |
| Boards of Canada – Music Has The Right To Children | mid-low | gradient_flow | downtempo IDM + ambient haze |
| Beach House – Bloom | mid-low | gradient_flow | stable dream-pop |
| Brian Eno – Ambient 4: On Land | low | tight_continuous | beatless texture continuity |
| Hiroshi Yoshimura – Green | low | tight_continuous | environmental ambient |
| Stars of the Lid – Per Aspera Ad Astra | low | tight_continuous | drone/ambient |

Excluded (reserved for arc eval or unusable): Kid A, Fleet Foxes, Black Sabbath, Sophie,
Unwound, Godspeed, Flaming Lips (arc_contrast); long-form suites (Fela, *In A Silent Way*);
SAW II (duplicate-row dedup hassle — optional add later). Selection may be trimmed/extended
during implementation if a track count is too thin for the control.

### Pair sets (the labels)

For each corpus album (tracks in `usable_range`, sorted by track number):
- **adjacent** — consecutive pairs (i, i+1): **positives** (pace-compatible).
- **non_adjacent_same_album** — non-consecutive pairs within the same album: the
  **discriminating control** (same artist/genre/sonic, larger intended pace step). Only strong
  on `gradient_flow` albums (they *traverse* pace); `tight_continuous`/`flat_uniform_mix` sit
  in one pocket so their non-adjacent pairs are also compatible → those albums contribute mainly
  the binary test, not the 3-tier control.
- **random_cross** — random pairs across albums of **different registers**: the **easy floor
  negative** (big-swing exemplars; e.g. drone vs dancefloor). Sanity floor — the *hard* test is
  beating `non_adjacent_same_album`, not this. (Same-register cross-album pairs are NOT used as
  negatives — two different dance records can be genuinely pace-compatible, so they're not clean
  negatives.)

### Candidate representations (ratified)

Each maps a track → a feature vector; pairwise distance below.
- **Baselines:** `rhythm_tower` (librosa 9-dim), `perceptual_bpm` (onset-trust-gated, log),
  `onset_rate`, `beat_strength_median`.
- **Validated energy:** `arousal_p50`, `danceability`, `energy_pair=[arousal_p50,danceability]`,
  `energy_dist=[arousal_p10,p50,p90,danceability]`.
- **Combinations:** `energy_onset=[arousal_p50,danceability,onset_rate]`,
  `pace_full=[arousal_p50,danceability,onset_rate,log_bpm_trusted,beat_strength]`.
- **`pace_tuned`** (Pass 3 only): `pace_full` features with weights fit to the adjacency truth,
  cross-validated; deferred out of Pass 1 (overfit risk).

### Normalization & distance

Each scalar feature **z-scored library-wide** (mean/std over all artifact tracks, not just the
corpus, so the scale matches production) so features are comparable; multi-dim spaces
(rhythm_tower, energy_dist) z-scored per-dim. Pairwise distance = **weighted Euclidean** on the
z-scored vector (unit weights for fixed candidates; learned weights only for `pace_tuned`).
NaN handling: a track missing a feature (e.g. the 1 NaN energy track, or NaN BPM) is dropped
from pairs for candidates that need it; report coverage.

### Passes (coarse → fine)

1. **Broad / automated (no human):** every candidate scored on **ranking AUC** — does its
   distance rank `adjacent` below `random_cross`? Plus the full **distribution** report
   (min/p10/p50/p90 of each pair-set's distances per candidate). Drop clear losers.
2. **Narrow:** top 2–4 candidates get the **3-tier separation** (adjacent < non_adjacent_same
   < random, on gradient albums), a **per-register breakdown** (does it work at the low end,
   not just dance?), a **monotonicity** check (distance vs |pace step|), and a **blind human
   pairwise / odd-one-out** session (Dylan picks the odd track in a triad; agreement with each
   candidate's distance is the independent, non-circular confirmation).
3. **Confirm / tune (only if a fixed combo is close):** fit `pace_tuned` weights
   cross-validated on the finalist features; final blind check.

### Success criterion

A representation wins if it: (a) ranks adjacent > non_adjacent > random with clean
distribution separation (report AUC + the four percentiles per tier); (b) **holds across
registers** (works calm↔calm and energetic↔energetic, not just dance-vs-not); (c) is
monotonic in degree of pace difference; (d) agrees with Dylan's blind pairwise calls. We
report N and distributions, never just means; the winner is re-run on the full corpus before
the verdict.

### Harness architecture (modeled on sonic/genre auditions)

Focused modules under `scripts/research/`:
- `pace_eval_corpus.py` — the `flow_type`-tagged album list → resolves to track_ids + track
  numbers via metadata.db (read-only); writes `corpus.tsv`.
- `pace_eval_features.py` — loads each candidate's per-track features for corpus tracks from
  the energy sidecar + the artifact/metadata.db; library-wide z-score params; one job: build
  the candidate feature matrices.
- `pace_eval_pairs.py` — builds the adjacent / non_adjacent / random_cross pair sets from
  `corpus.tsv` (seeded RNG for reproducibility).
- `pace_eval_run.py` — Pass-1 scoring: per candidate, distances over pair sets → AUC +
  distribution table → `results_pass1.tsv` + a findings summary.
- (Pass 2/3 helpers added when Pass 1 narrows.) Blind human session served like the
  sonic-audition pattern.

Artifacts under `docs/run_audits/pace_axis_eval/` (gitignored). **Data access:** the harness
reads the main checkout's `data/` (metadata.db + the energy sidecar + artifact) via a
configurable path (a read-only `data/artifacts` junction into the worktree, removed before
worktree teardown — the documented junction-cleanup discipline).

## Risks / assumptions

- Album adjacency is a *proxy* for pace-compatibility; the blind human pass (2) is the guard
  that the proxy isn't lying. Stated as N-limited, not overgeneralized.
- `flat_uniform_mix` albums weaken the within-album control (non-adjacent also compatible) →
  they contribute only the binary test; the 3-tier control runs on `gradient_flow`.
- Corpus skews electronic at the high end (continuous mixes are mostly electronic) — register
  balance + the low/ambient tight_continuous albums mitigate; per-register reporting exposes
  any dance-only win.
- `pace_tuned` overfit risk on a ~13-album corpus → cross-validated, Pass-3-only, treated as a
  ceiling estimate not a shipping weight.
- Non-circularity: candidate distances are scored against album/human truth, never against
  another candidate's space.

## Testing

This is a research harness, not production code; "tests" = methodology guards:
- Unit-test the pure pieces (z-score params, pair-set construction with a seeded RNG, AUC
  computation) on small synthetic inputs.
- Provenance/determinism: harness records feature-matrix shapes + a fingerprint; reruns
  reproduce `results_pass1.tsv` byte-for-byte (deterministic given the RNG seed).
- No writes outside `docs/run_audits/pace_axis_eval/`; metadata.db opened read-only.
