# Pace as a co-equal axis (energy admission-rescue) — design

**Date:** 2026-06-20
**Status:** Draft (design) — awaiting Dylan review
**Author:** session w/ Dylan
**Supersedes:** `2026-06-19-pace-cedes-sonic-authority-design.md` (sonic-cede — DISPROVEN: see below)
**Related:** `docs/PACE_AXIS_LEARNINGS.md` (north star: "third co-equal matching axis: sonic ⊗ genre ⊗ pace"); the energy soft-penalty wiring (`2026-06-18-energy-pace-mode-wiring-design.md`, shipped to master 898ae15); memories `project_pace_cedes_sonic`, `feedback_never_fail_three_axes`, `feedback_generation_time_budget`, `timbre_embedding_ceiling`.

## Problem & goal

The energy soft-penalty (shipped) steers the arousal arc in **dynamic/off** but is structurally inert in **strict/narrow**. Log-verified root cause: the candidate pool is built by a sequence of **hard admission gates** (onset band, BPM band, genre floor, sonic floor); at strict the **onset admission band alone rejects ~21,819 tracks** (vs the sonic floor's 101), leaving ~28 admitted. The segment-level reservoirs and the beam all draw from that post-admission pool (`pier_bridge_builder.py:410` `universe = deduped_pool`), so a soft penalty — even at penalty 77.9 (instrument-verified firing) — has only a handful of near-forced survivors to re-rank. **A late soft penalty cannot overcome early hard admission starvation.**

The three axes are not symmetric today:

| Axis | Admission | Pool reservoir | Beam scoring |
|---|---|---|---|
| Genre | `min_genre_similarity` floor | `k_genre` (genre-arc target) | arc vote + tiebreak |
| Sonic | `min_sonic_similarity` floor | `k_local`, `k_toward` | `weight_bridge` |
| Pace | onset + BPM **rhythm** bands | — none — | energy soft penalty (loses) |

Pace's admission judges *rhythm similarity*, not the *energy arc*, and the energy arc has no reservoir. **Goal: make pace co-equal — give the energy/arousal arc a seat in admission so on-arc candidates exist in the pool even in strict/narrow** — without hard-failing on pace, without weakening genre, within the 90s budget.

## Why sonic-cede was rejected (don't relitigate)

The prior spec ceded `min_sonic_similarity`. Disproven by reading real logs: the binding admission gate at strict is the **onset band (21,819 rejected)**, not the sonic floor (101). Ceding sonic admission was inert; energy stayed starved. The conclusion: the fix must add energy to admission, not loosen sonic. (Full evidence trail in `project_pace_cedes_sonic` memory and `docs/run_audits/pace_cedes_sonic/ABLATION.md`.)

## NON-NEGOTIABLE constraints

- **Never hard-fail on pace** (`feedback_never_fail_three_axes`): the rescue is **purely additive** — it only unions candidates into the pool, never removes or hard-gates. It cannot cause infeasibility or trigger the relaxation cascade. Missing energy data → no rescue → behavior unchanged.
- **Genre authority preserved**: rescued tracks must clear the genre floor. The rescue never touches any genre gate, weight, or arc floor.
- **Sonic safety floor preserved**: rescued tracks must clear the `min_sonic_similarity` admission floor AND the beam's per-edge `bridge_floor`. Pace can admit a rhythmically-dissimilar track; it can NEVER admit a sonically-disconnected one. (North star #5: the worst edge defines the experience.)
- **90s budget** (`feedback_generation_time_budget`): bounded `k_energy` + union cap → bounded pool growth → no new hard gate, no cascade.
- **Opt-in / no-op default**: all per-mode `k_energy` and energy strengths default to the no-op values (rescue off, energy strengths 0.0) → byte-identical to current behavior until calibrated values ship past the eval-gate.

## Design

### 1. Energy admission-rescue (the new mechanism) — `build_candidate_pool` (`src/playlist/candidate_pool.py`)

After the existing gates run (BPM/onset set the `seed_sim_all == -2.0` rhythm sentinel; genre + sonic floors computed), insert an energy-rescue step:

1. **Source set** = indices that clear the **genre floor AND the sonic floor**, *ignoring the rhythm sentinel*. (Reuse the genre-pass and sonic-pass masks already computed in this function; do not re-apply BPM/onset.)
2. **Selection — span the seed arousal range.** Compute the seeds' z-scored `arousal_p50` min..max. Build a small set of arousal targets covering that range (e.g. linspace of `n_targets`). For each target, pick the top-`k_energy_per_target` source tracks nearest the target in 1-D arousal. Union the picks. This guarantees the pool contains low/mid/high-arousal candidates *relative to the seeds*, so the per-segment beam has on-arc options at every step. (1-D arousal → negligible cost.)
3. **Force-admit the rescued set**: clear the rhythm `-2.0` sentinel for the chosen indices (restore their genuine sonic-seed similarity for ranking) so they survive into the admitted pool alongside the normally-admitted tracks.
4. **Bounded**: total rescued ≤ a per-mode cap. Rescue runs only when the mode's `k_energy` > 0.

Energy data (`X_energy` = z-scored arousal, from the sidecar) and the seed arousal range must be threaded into `build_candidate_pool` (today energy only reaches the beam). Mirror the existing `perceptual_bpm`/`onset_rate` threading.

### 2. Per-mode sizing (keyed by `pace_mode`)

| pace_mode | rescue `k_energy` | energy arc (beam) | step-cap (whiplash) |
|---|---|---|---|
| strict | most (pool needs arc room) | tight band, strong | tight, strong |
| narrow | some | medium | medium |
| dynamic | ~0 (pool already has room — measured) | loose | loose |
| off | 0 (rescue off) | disabled (`arc_strength=0`) | kept (small) |

Rescue is **one mechanism sized per mode** — no special-casing. dynamic/off get ~0 rescue because their pools already carry arousal spread (ceiling probe: ~80 on-arc/step in dynamic); the rescue earns its keep in strict/narrow.

Starting values are placeholders; the eval-gate (§5) sets the shipped numbers, ramping **up from minimal** (smallest rescue that yields an audible arc under the worst-edge ceiling).

### 3. Downstream placement (already exists, now fed)

The rescued tracks flow through the existing dj_union segment reservoirs and the merged beam energy penalty (`energy_arc_band`/`energy_step_cap`). No change needed there beyond calibration — admission guarantees the candidates exist; the beam places them on the per-segment arc.

### 4. Components (one responsibility each)

- **`src/playlist/candidate_pool.py`** — the energy-rescue step (source mask → arousal-spanning selection → force-admit), gated on `k_energy > 0`. New params: `X_energy`, `seed_arousal_range` (or `X_energy` + seed list), `k_energy`, `n_arousal_targets`.
- **`src/playlist/mode_presets.py`** — per-mode `pace_rescue_k_energy` (+ the existing energy strengths) in `PACE_MODE_PRESETS`, default no-op.
- **`src/playlist/pipeline/core.py`** — thread `X_energy` + seed arousal into `build_candidate_pool` (energy already loaded here for the beam; reuse it). Pass the per-mode `k_energy` from `pace_settings`.
- **`web/src/components/GenerateControls.tsx`** — add `off` to the pace dropdown (+ TS union), so the four-point pace scale exists in the GUI.
- **`scripts/research/pace_cede_eval.py`** (rename/extend, exists) — measurement: per-mode arousal curve (arc-dev + max-step), worst-edge sonic, pool composition, budget.

### 5. Eval-gate before default-on (the kill switch)

1. **Worst-edge sonic cost is the gate.** Weakest adjacent MERT cosine per mode, rescue-on vs -off. A mode ships rescue-on only if its weakest sonic edge does not drop below a pre-set threshold. Net quality, not arc-only — the "don't do more harm than good" kill criterion.
2. Energy benefit: arousal curve tightens strict→off (arc-deviation down), max-step bounded even at off.
3. Blind A/B (rescue/energy on vs off) with a decoy arm (evaluation-methodology skill).
4. **Diverse seeds**: mellow + high-arousal + wide-swing, multi-pier via `generate_like_gui`, BPM active (verify `BPM loaded: N/N` — the confound that wasted a day).
5. A mode that fails ships rescue-off (`k_energy=0`, energy strengths 0.0). Modes are independent.

## Error handling / graceful degradation

- Energy sidecar missing / `X_energy` None → rescue skipped, pool unchanged, log once. (configured-knob-must-act: if `k_energy>0` but data absent, warn loudly, degrade to no-op, never crash.)
- A seed with NaN arousal → excluded from the range computation; if all seeds NaN → no rescue.
- Rescue source empty (no genre+sonic-passing track in an arousal band) → that band contributes nothing; never an error.

## Testing

- **Unit:** the rescue selects arousal-spanning tracks from the genre+sonic-passing set; rescued tracks clear genre+sonic floors but may fail the rhythm band (the whole point); `k_energy=0` → no-op (golden-safe); NaN-arousal seed handled; rescue is additive (pool size monotonic non-decreasing vs no-rescue). Mirror `test_pace_mode_energy_presets` / `test_beam_energy_pace` patterns.
- **Generation (gui_fidelity / `generate_like_gui`, multi-pier, BPM active):** all four pace_modes complete < 90s (never-hard-fail); strict pool now carries arousal spread (rescue admits on-arc tracks); the realized arousal curve tightens strict→off and never whiplashes even at off; **genre distinct/cohesion metric unchanged across pace_modes** (proves genre untouched).
- **Eval-gate (§5)** before any default-on flip.

## Risks / assumptions

- **The intended trade:** rescue admits rhythmically-dissimilar (off-BPM/onset) tracks that are on the energy arc + genre/sonic-OK. That is "energy outranks rhythm-similarity for admission." The worst-edge sonic eval-gate is the guard against it going too far.
- Arousal is ~0.77 MERT-redundant, so in many cases the rescue adds little (the on-arc tracks were already admitted) — the rescue's value concentrates on the edges where energy and rhythm/sonic disagree. The eval-gate quantifies whether that's worth it per mode; a mode where it isn't ships rescue-off.
- `build_candidate_pool` is a hotspot; the change is additive and gated (`k_energy>0`), keeping the default path untouched.
- Small-N seed caveat from the ablation: the eval-gate must use ≥3 diverse seed sets before default-on.
