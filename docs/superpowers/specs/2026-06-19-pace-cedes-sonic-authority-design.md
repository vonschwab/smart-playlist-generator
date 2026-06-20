# Pace cedes sonic authority — design

**Date:** 2026-06-19
**Status:** Draft (design) — awaiting Dylan review
**Author:** session w/ Dylan
**Related:** extends `docs/superpowers/specs/2026-06-18-energy-pace-mode-wiring-design.md` (energy soft-penalty wiring, shipped); `docs/PACE_AXIS_LEARNINGS.md`; memories `project_pace_cedes_sonic`, `feedback_never_fail_three_axes`, `project_pace_gate_retune`, `timbre_embedding_ceiling`.

## Problem & goal

The energy soft-penalty wiring (2026-06-18 spec) shipped and is correct — the penalty is computed and subtracted from the beam's `combined_score` (verified: 2.1M calls, max penalty 1.5+ at `arc_strength=0.5`). **But in the normal gated modes it does not move the playlist at all** — even `energy_arc_strength=50` changed 0/20 tracks. The `pace_mode` slider therefore does not actually steer pace.

**Goal:** make the `pace_mode` slider (strict/narrow/dynamic/off) produce Dylan's energy-arc behavior in real, gated playlists — `strict` ≈ `1→2→3→4→4→5`, `off` loose but never `1→9→2→9` — by letting pace **outrank MERT sonic cohesion**, while **genre authority is preserved**, and **never hard-failing** on pace.

## Key findings (all measured this session via `generate_like_gui`; seeds Songs:Ohia / Bill Callahan / William Tyler)

1. **Not a wiring/discard bug.** The penalty fires and reaches `combined_score`; with genre+sonic gates OFF, energy bites hard (8/12 tracks changed; a jumpy arousal curve `[-1.2,-1.5,-1.5,-1.1,-2.1,…]` flattened to `[-1.2,-1.2,-1.2,…]`). Energy *can* steer when it has authority.
2. **Not a pool-room problem.** Ceiling probe (`scripts/research/energy_ceiling_probe.py`): the gated segment pool already holds ~80 on-arc candidates per step (100% HAD_ROOM, 0% NO_SUPPORT). **A `pool_k_energy` reservoir is therefore unnecessary and is explicitly rejected.**
3. **Blocker = cumulative genre+sonic authority in the beam.** Pinpoint probe (`scripts/research/energy_blocker_pinpoint.py`): genre-on/sonic-off → energy bites (7/12); genre-off/sonic-on → energy bites (8/12). Neither gate alone blocks; the two together pin selection. **ARM 1 (genre on, sonic off, energy works) proves "outrank sonic, keep genre" is achievable.**
4. **The mechanism is NOT the local sonic edge floor** — it is off by default (`local_sonic_edge_penalty_enabled=False`, `local_sonic_edge_floor=None`). What `sonic_mode=off` actually changes is the **pool admission floor** `candidate_pool.min_sonic_similarity` (dynamic=0.08 → None) and the hybrid **`sonic_weight`**; the beam **`weight_bridge`** (0.6) is a separate `cohesion_mode` knob. The exact lever to cede must be pinned by ablation (Task 1) — **do not guess.**

## NON-NEGOTIABLE constraints

- **Never hard-fail on pace** (`feedback_never_fail_three_axes`): the design only ever *loosens* sonic authority (wider admission / lower weight) and keeps energy a soft penalty. Loosening can never strand a segment or trigger the relaxation/expansion cascade. Pace `off` keeps a whiplash floor but must never block generation.
- **Genre authority is preserved** at every level (admission floor, weighting, genre-arc floor). Pace spends *sonic* cohesion only.
- **90s budget** (`feedback_generation_time_budget`): no new hard gate; soft penalties only.
- **Opt-in / backward-compatible:** default behavior unchanged until the per-mode cede values are non-zero (and gated behind the eval-gate before default-on).

## Scope

**In:** (a) an ablation to pin the exact sonic-authority lever; (b) a per-pace-mode sonic-authority *cede multiplier* applied to that lever, composing with `sonic_mode`; (c) per-mode calibration of the cede + the already-wired energy `arc_band`/`step_cap`/strengths; (d) add `off` to the GUI pace dropdown; (e) tests + eval-gate.
**Out:** any `pool_k_energy` reservoir (finding 2 — rejected); any change to genre authority; changing the energy representation (`arousal_p50` default stands, knob-swappable); the Pass-2 perceptual representation tie-break.

## Design

### 1. Pin the lever (Task 1, ablation — no guessing)
Run an A/B matrix that toggles, one at a time, the sonic levers `sonic_mode=off` changes — `min_sonic_similarity` (pool admission) and the hybrid `sonic_weight` — plus the beam `weight_bridge`, with energy strong and genre held on. Identify the minimal lever (or pair) whose relaxation lets energy act. This becomes "sonic authority" for the cede. Recorded under `docs/run_audits/pace_cedes_sonic/`.

### 2. Per-pace-mode sonic-authority cede
Introduce a cede multiplier keyed by `pace_mode`, applied to the pinned lever, **composing multiplicatively with `sonic_mode`** (so the sonic slider still sets the base; pace scales it down). Genre untouched.

| pace_mode | sonic authority ceded | energy arc | step-cap (whiplash) |
|---|---|---|---|
| strict | most (energy dominant) | tight band, strong | tight, strong |
| narrow | some | medium band/strength | medium |
| dynamic | little | loose band, low strength | loose |
| off | none | disabled (`arc_strength=0`) | **kept** (small) — the floor that forbids `1→9→2→9` |

Resolution rule for conflicting sliders (e.g. pace=strict + sonic=strict): pace wins per Dylan's directive — the cede multiplier reduces the effective sonic authority regardless of the sonic slider (it composes, so sonic=strict + pace=strict = reduced-but-nonzero sonic). This is a deliberate, documented interaction.

### 3. Energy terms (already wired) shape within the freed room
The existing `energy_arc_band`/`energy_arc_strength` (arc) and `energy_step_cap`/`energy_step_strength` (whiplash) from the 2026-06-18 wiring now have candidates to act on. Per-mode values calibrated in the eval-gate.

### 4. GUI
Add `off` to the pace dropdown in `web/src/components/GenerateControls.tsx` (currently `["dynamic","narrow","strict"]`) and widen the TS union type. No new control — pace stays one slider.

## Error handling / never-fail
- Ceding sonic authority only widens admission / lowers weight → strictly more candidates or gentler ranking → cannot cause infeasibility.
- Energy stays soft (additive penalty, NaN-safe, never `continue`).
- Missing energy sidecar → energy terms no-op (existing behavior); the cede still applies harmlessly (just looser sonic, no arc).

## Testing
- **Unit:** the cede multiplier composes with `sonic_mode` correctly per pace_mode; genre knobs are untouched by any pace_mode; `off` keeps a non-zero step-cap; default (cede=1.0, energy=0) reproduces current behavior (golden-safe).
- **Generation (gui_fidelity / `generate_like_gui`, multi-pier):** all four pace_modes complete within 90s (never-hard-fail); the realized arousal curve tightens strict→off and never whiplashes even at `off`; genre cohesion metric is unchanged across pace_modes (proves genre preserved).
- **Eval-gate before default-on:** per-mode arousal curve vs pier targets (arc-deviation + max-step), and a blind A/B (energy/cede on vs off) for weakest-edge + perceptual quality. Ship behind knobs (default off) until it passes.

## Risks / assumptions
- The cede trades sonic cohesion for pace — **intended** (Dylan: "let it outrank MERT"). Strict pace = more sonically varied; quantify the sonic-cohesion cost in the eval-gate so the tradeoff is visible per mode.
- Ablation may show the lever is `weight_bridge` (cohesion-keyed) rather than `min_sonic_similarity` (sonic-keyed); if so, the cede must avoid disturbing `cohesion_mode` semantics — handled by introducing a dedicated pace-cede factor rather than overwriting the cohesion knob.
- Small N in the probes (2 segments, mellow seeds). The eval-gate re-confirms on more seeds (incl. high-arousal and wide-swing piers) before default-on.
