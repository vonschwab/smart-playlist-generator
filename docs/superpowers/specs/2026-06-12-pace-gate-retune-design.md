# Pace Gate Retune — Design

## Purpose

Make `pace_mode` work the way the slider implies — "match the rhythmic feel" — using signals that survive the MERT sonic-embedding migration. Replace the rhythm-cosine **hard gate** (which is barely above noise and unsatisfiable for beatless/ambient artists) with two interpretable, embedding-independent **hard bands** (BPM log-distance + onset-rate log-distance) plus a **soft** rhythm-cosine penalty. Outcome: `pace_mode: narrow` becomes usable for a Green-House (ambient) artist playlist without neutering it for rhythmic music, and the whole mechanism is MERT-durable.

## Evidence (2026-06-12 diagnosis)

Reproduced from the CLI; isolated to a single variable (`--pace-mode narrow` throws `Segment infeasible under bridge_floor backoff`; `--pace-mode dynamic` generates 50 tracks, all else equal). Probes were read-only against the full `beat3tower_32k` artifact (N=40,393) and `metadata.db`.

| finding | data |
|---|---|
| Rhythm-cosine bridge floor (narrow 0.45) is near-noise for everyone | Random-pair rhythm cosine: p50 ≈ −0.01, only **11%** of pairs ≥ 0.45. Same-artist medians: J Dilla +0.33 (35% ≥ 0.45), De La Soul +0.31, Beastie Boys +0.24 |
| Beatless artists can't clear their own gate | Green-House intra-artist rhythm cosine p50 = **0.213**; only **24%** of its own pairs ≥ 0.45. Failing run's adjacent piers had rhythm cosines of +0.05 / +0.14 / +0.18 / −0.05 |
| The gate never backs off | Segment backoff relaxes only `bridge_floor` (0.02→0.01→0.0); the rhythm-cosine pace floor is fixed → infeasible segment → cascade → 151.9 s run (> 90 s budget) |
| `tempo_stability` is a dead signal | Entire library sits at **0.91–0.99** (librosa "finds" beats in beatless music). The BPM gate's stability bypass never fires; a stability-based bypass is dead on arrival |
| Rhythmic density *does* separate beatless from busy | Green-House `onset_rate` spans 0.49 (Ferndell Shade) → 5.15 (Perennial Bloom)/sec. Split at the median: sparse half intra-cosine p50 +0.37 (40% ≥ 0.45), busy half +0.14 (16%). The signal the cosine throws away (magnitude) is the one that carries "beatless" |
| MERT removes the rhythm tower entirely | MERT is a single learned embedding — no tower decomposition. The code already falls back to the BPM gate for "no-tower variant (e.g. mert)" (`pier_bridge_builder.py` ~L388, `pace_gate.bpm_fallback_max_log_distance`). The rhythm-cosine signal ceases to exist post-MERT |

Consequence: the durable substrate for pace gating is the beat3tower DB features (`bpm_info`, `rhythm.onset_rate`) in `metadata.db`, **not** the sonic embedding. They are untouched by the MERT fold (only a full beat3tower re-analysis would regenerate them).

## Architecture (current wiring, confirmed)

Pace gating has two symmetric sides, each already carrying a BPM band next to the rhythm-cosine floor:

| side | file | rhythm-cosine (to remove as gate) | BPM band (keep) |
|---|---|---|---|
| Admission | `candidate_pool.py` | `pace_admission_floor` (cosine to nearest seed) | `bpm_admission_max_log_distance` (+ stability bypass) |
| Bridge | `pier_bridge/beam.py` | `pace_bridge_floor` (cosine to step target) | `bpm_bridge_max_log_distance` (+ stability bypass) |

Both fed from `PACE_MODE_PRESETS` (`mode_presets.py`) → `config.py::resolve_thresholds` (admission, into `CandidatePoolConfig`) and `pipeline/core.py` (bridge, copied into `PierBridgeConfig`). The relevant dataclass fields already exist for the BPM analogs, so the onset-rate analog slots in symmetrically.

## Design

### 1. New signal — onset-rate array (embedding-independent)

Extend `bpm_loader.load_bpm_arrays` to also extract `json_extract(sonic_features, '$.full.rhythm.onset_rate')` into an `onset_rate` numpy array aligned to the bundle's `track_ids` (NaN for missing), returned alongside the existing keys. Same JSON object the loader already reads — one extra column, one extra array. Like BPM, it is a DB feature that survives the MERT fold.

### 2. Onset-rate band (hard) — both sides

Mirror the BPM band exactly, using **log-ratio distance** (onset rate is a rate; 2→4 events/sec is the same step as 4→8). New helpers in `pace_gate.py` paralleling the BPM ones:

- `compute_step_log_onset_target(onset_a, onset_b, step, segment_length)` — geometric interpolation between piers (reuse the `interpolate_log_bpm` pattern).
- `filter_candidates_by_onset_target(...)` — reject candidates whose onset-rate log-distance to the target exceeds the cap.

Applied:
- **Admission** (`candidate_pool.py`): reject candidates whose onset log-distance to the **nearest seed** exceeds `onset_admission_max_log_distance`.
- **Bridge** (`beam.py`): reject candidates beyond `onset_bridge_max_log_distance` from the **step-interpolated** target between piers. A beatless→busy segment sweeps density smoothly across steps (handles within-artist variation: not all Green-House is beatless).

**Bypass:** only on missing/NaN `onset_rate`. **No `tempo_stability` bypass** — onset density is meaningful even when tempo tracking is unstable (and stability is dead at 0.91–0.99). This is the one deliberate asymmetry from the BPM band.

### 3. Rhythm-cosine → soft penalty (default ON for tower variants)

- The hard rhythm-cosine floors are **disabled**: `admission_floor` / `bridge_floor` set to 0.0 in all presets (existing `> 0.0` guards make the gate dead), and the now-dead hard-gate code paths in `candidate_pool.py` (admission) and `beam.py` (bridge) are removed.
- At the **bridge**, rhythm-cosine to the step target becomes a **multiplicative score penalty** below a per-mode threshold — same mechanism as the existing genre soft penalty in `beam.py` (`combined_score *= (1 - strength)` when `rhythm_sim < threshold`). New knobs: `rhythm_soft_penalty_threshold`, `rhythm_soft_penalty_strength`.
- There is **no soft penalty at admission** — admission is a hard include/exclude stage with no score to demote; admission gating is done entirely by the BPM + onset hard bands.
- **MERT path:** under a no-tower variant there is no rhythm axis, so the rhythm-cosine penalty is skipped (0 contribution) while the hard BPM + onset bands still gate. The MERT-durability property is structural, not a special case.

### 4. Budget protection — bands widen on backoff

The segment-infeasibility backoff ladder currently relaxes only `bridge_floor`. The onset **and** BPM bridge caps now **widen by 1.5× per backoff step** in the same ladder, so an over-tight band relaxes rather than detonating the infeasibility→relaxation cascade past the 90 s hard ceiling. (Memory `feedback_generation_time_budget`: prefer soft penalties; never let a hard gate blow the budget.)

### 5. Per-mode presets (`PACE_MODE_PRESETS`) — INITIAL values

Log₂ caps. BPM caps kept at current values as the starting point; onset caps initialized to mirror BPM; rhythm-soft mirrors the genre-soft-penalty shape.

| mode | bpm adm/brd | onset adm/brd | rhythm-soft thresh/strength |
|------|-------------|----------------|------------------------------|
| strict  | 0.30 / 0.40 | 0.30 / 0.40 | 0.35 / 0.20 |
| narrow  | 0.50 / 0.60 | 0.50 / 0.60 | 0.25 / 0.15 |
| dynamic | 0.75 / 0.85 | 0.75 / 0.85 | 0.15 / 0.10 |
| off     | ∞ / ∞       | ∞ / ∞         | 0 / 0 |

**These are starting values, not final.** A full-pool calibration sweep (see below) finalizes them before they are locked — in particular whether narrow's caps need tightening and whether onset should be tighter or looser than BPM. The spec defines the *mechanism* and *initial* values; the sweep defines the *numbers*.

### 6. Calibration sweep (immediately after implementation)

Per the evaluation-methodology skill, against the full library (N≈40k), outputs under `docs/run_audits/pace_retune/` — never touching production artifact paths:

- Onset-rate log-distance pass-rate distributions (min/p10/p50/p90) across candidate caps, library-wide and for representative ambient *and* rhythmic seeds.
- Confirm `narrow` admits a workable pool for a beatless seed (Green-House) **and** still meaningfully tightens for a rhythmic seed (e.g. J Dilla).
- Generation-based check through `generate_like_gui` (multi-pier, per playlist-testing skill): the Green-House narrow case yields 50 tracks within budget; a rhythmic narrow case stays coherent.
- Finalize preset caps from the distributions; record the chosen numbers and the evidence.

### 7. Testing

- **Unit:** onset loader column (present/NaN); onset log-distance math; onset band admit/reject at a known cap; soft rhythm penalty applied below / skipped above threshold; soft penalty **skipped on a no-tower bundle** (MERT durability); band widening across backoff steps.
- **Integration** (`generate_like_gui`, marked `integration`/`slow`): the Green-House `narrow` reproduction that currently raises `Segment infeasible` returns 50 tracks; a rhythmic seed under `narrow` still tightens vs `dynamic`. Reference the fixing commit (per playlist-testing maintenance protocol).
- **Trap-catalog upkeep:** if this surfaces a new fidelity trap, add a row to the playlist-testing skill.

## Scope

`bpm_loader.py` (onset column), `mode_presets.py` (preset keys), `config.py` + `pier_bridge/config.py` (dataclass fields + resolution), `candidate_pool.py` (onset admission band; remove rhythm-cosine admission gate), `pier_bridge/pace_gate.py` (onset helpers), `pier_bridge/beam.py` (onset bridge band; rhythm-cosine soft penalty; remove rhythm-cosine bridge gate), `pier_bridge_builder.py` + `pipeline/core.py` (plumb onset arrays + widen-on-backoff), tests, calibration probe.

**Out of scope / non-goals:** no artifact rebuild; no `metadata.db` writes; no change to BPM-gate internals beyond adding backoff widening; the genre-corruption issue (Green-House mislabeled hip-hop) is a separate session.

## Risks & assumptions

- *Assumption:* `onset_rate` has near-full coverage like BPM (7221/7221 in the failing run). Same source JSON; verify in the loader test. If sparse, the NaN bypass degrades the band gracefully toward BPM-only.
- *Assumption:* MERT fold does not regenerate `bpm_info`/`onset_rate` (separate sidecar embedding). Holds unless a full beat3tower re-analysis is run.
- *Risk:* removing the rhythm-cosine hard gate changes admitted pools for existing tower runs. Mitigated by the soft penalty (default on) preserving demotion of off-rhythm edges, and by the calibration sweep before locking numbers.
- *Risk:* initial onset caps mirroring BPM may be mis-scaled (onset rate has a different dynamic range than BPM). This is exactly what the calibration sweep resolves; initial values are explicitly provisional.
