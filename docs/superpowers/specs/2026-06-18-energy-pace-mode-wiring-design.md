# Energy pace_mode wiring — design

**Date:** 2026-06-18
**Status:** Approved (design)
**Author:** session w/ Dylan
**Related:** `docs/PACE_AXIS_LEARNINGS.md`; pace-representation eval (`docs/superpowers/specs/2026-06-18-pace-representation-eval-design.md`, Pass 1); `project_energy_feature_exploration`, `project_onset_band_soft_penalty`, `feedback_generation_time_budget` memories.

## Problem & goal

`pace_mode` (strict/narrow/dynamic/off) exists but steers on weak proxies (perceptual-BPM — near-random for pace; onset; the rollback rhythm tower). The pace-representation eval (Pass 1) validated **energy** (`energy_pair` = [arousal_p50, danceability]) as the pace signal — coarse AUC 0.77–0.80 vs BPM 0.58. This sub-project wires energy into the beam so `pace_mode` actually controls pace, as a **third co-equal axis alongside sonic (MERT) and genre**.

Behavior target (Dylan's spec, generalized from a 0–10 illustration to multidimensional energy):
- **Adjacent-step cap (anti-whiplash):** the energy jump between consecutive tracks is bounded in **every** mode, tightening strict→off. Even `off` forbids whiplash (1→9→2→9); it just imposes no arc.
- **Arc:** how the interior follows the pier→pier energy path — `strict` tight linear, `narrow` bounded near the piers (zig-zag ok), `dynamic` loose + overshoot, `off` none.

## NON-NEGOTIABLE: never hard-fail on pace

Both energy terms are **soft penalties only** — ranking nudges added to the beam's `_pace_penalty` (already subtracted from `combined_score`). They **never** `continue`/reject a candidate and are **never** an admission-pool filter. A candidate is only down-ranked, never excluded → a segment is always buildable in EVERY mode (incl. strict); pace can **never** cause infeasibility or trigger the relaxation/expansion cascade. This is the lesson from the onset/BPM **hard**-band failure earlier this session (stranded segments → cascade detonated → "took FOREVER"; the 90s ceiling) and CLAUDE.md "infeasible bridge → progressively relax, don't crash / edge cases get graceful fallbacks." **Strict = a stronger soft penalty, not a gate.**

## Scope

**In:** the two soft energy terms + a new `energy_loader` + config fields + `PACE_MODE_PRESETS` values + `pipeline/core.py` threading + `pace_gate` target helper + the beam application + tests + an eval-gate. **Out:** changing the existing BPM/onset bands (leave them — additive, low-risk; a later cleanup may demote them now that energy is the validated signal); any candidate-pool energy gate (forbidden — soft only); the fine within-album micro-gradient (Pass 1 near-chance — the arc here is the COARSE pier-to-pier path); the Pass-2 representation tie-break (`energy_pair`↔`energy_dist`↔`arousal_p50` is a knob swap).

## Key facts (verified this session)

- Beam per-step loop (`beam.py` ~1052–1259): `_pace_penalty = 0.0` accumulates the BPM + onset soft bands; `combined_score -= _pace_penalty`. The per-edge sonic penalty `_apply_local_sonic_edge_policy(score, int(current), int(cand))` (~1252) gives the **prev (`current`) and candidate (`cand`) indices** — the energy step-cap attaches here.
- `pace_gate.compute_step_rhythm_target` already interpolates an axis VECTOR via `interpolate_axis_vector(R_a, R_b, t)` — the energy arc target reuses this exact pattern.
- Pace arrays load in `pipeline/core.py` (`bpm_loader.load_bpm_arrays`) and thread to the beam — the energy loader/threading mirrors it.
- Energy sidecar: `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz` — `track_ids` + `arousal_p10/p50/p90` + `danceability`, aligned to the artifact index, 40392/40393 filled. Runtime reads it (no essentia).
- Pass-1 distance calibration (`energy_pair`, z-scored, weighted-Euclidean): adjacent-in-mix p50 ≈ 0.51, within-album non-adjacent ≈ 0.80, cross-register (big swing) ≈ 1.59. These set the per-mode caps/bands.

## Design

### Energy space
`X_energy` = a per-track vector built from a **configurable feature list** (config `analyze.pace.energy_features`), each feature **z-scored library-wide** (distances in std units, matching the eval); distance = Euclidean. A track missing energy → NaN row → both terms skipped for any edge touching it.

**Default feature list = `[arousal_p50]` (arousal-led), per the head-probe findings (learnings log 2026-06-18):** arousal is the strongest pace signal and the steerable scalar MERT can't express; **danceability is OFF by default** (0.89 MERT-redundant, +0.004 marginal — available in the list but not default); mood/texture heads (electronic/acoustic/aggressive/relaxed) are excluded (off-axis and/or MERT-redundant). The list is a knob so the **eval-gate + Pass-2 blind session finalize it** (e.g. add `danceability` or use the `arousal_p10/p50/p90` distribution if they win). `instrumental` is NOT in this list — if a vocal-continuity signal is wanted it's a *separate* small term (deferred, pending the blind session), not folded into the pace distance.

### Term 1 — adjacent-step cap (always on)
At the `_apply_local_sonic_edge_policy` site, with `a=current`, `b=cand`:
```
d_step = ||X_energy[a] - X_energy[b]||
if finite(d_step) and energy_step_strength > 0 and d_step > energy_step_cap:
    _pace_penalty += energy_step_strength * (d_step - energy_step_cap)
```

### Term 2 — arc-band (per-mode; disabled when strength 0)
New `pace_gate.compute_step_energy_target(e_a, e_b, *, step, segment_length)` = `interpolate_axis_vector(e_a, e_b, t)`, `t = step/segment_length` (linear — arousal & danceability are linear scales). In the beam:
```
target = compute_step_energy_target(X_energy[pier_a], X_energy[pier_b], step=step, segment_length=interior_length)
d_arc = ||X_energy[cand] - target||
if finite(d_arc) and energy_arc_strength > 0 and d_arc > energy_arc_band:
    _pace_penalty += energy_arc_strength * (d_arc - energy_arc_band)
```
`narrow`/`dynamic`/`strict` differ by `energy_arc_band` (tube width) + `energy_arc_strength`; `dynamic`'s wider band + lower strength yields the overshoot tolerance; `off` sets `energy_arc_strength = 0` (term disabled), leaving only the step cap.

### Config + presets
- `PierBridgeConfig` new fields (all default `0.0` = off = current behavior preserved): `energy_step_cap`, `energy_step_strength`, `energy_arc_band`, `energy_arc_strength`.
- `PACE_MODE_PRESETS` (starting values, tuned in the eval-gate; caps/bands in z-std units from the Pass-1 distributions):

  | mode | step_cap | step_strength | arc_band | arc_strength |
  |---|---|---|---|---|
  | strict | 0.6 | high | 0.4 | high |
  | narrow | 0.9 | med | 0.8 | med |
  | dynamic | 1.2 | low | 1.3 | low |
  | off | 1.6 | low | — | 0.0 (disabled) |

### Components (each one responsibility)
- **`src/playlist/energy_loader.py`** — `load_energy_matrix(track_ids, *, sidecar_path, features) -> np.ndarray (n, len(features))` z-scored, built from the configurable `features` list (default `["arousal_p50"]`), NaN rows for missing (mirrors `bpm_loader`; reads sidecar npz; no essentia).
- **`src/playlist/pier_bridge/pace_gate.py`** — `compute_step_energy_target`.
- **`src/playlist/pier_bridge/config.py`** — the 4 fields.
- **`src/playlist/mode_presets.py`** — per-mode values in `PACE_MODE_PRESETS`.
- **`src/playlist/pipeline/core.py`** — load `X_energy` when any energy strength > 0 in `pace_settings`; thread to the beam (and `pier_bridge_builder`) as a new kwarg `energy_matrix`.
- **`src/playlist/pier_bridge/beam.py`** — accept `energy_matrix`; apply the two terms in the per-step loop.

## Error handling / graceful degradation
- Missing sidecar / energy not loaded → `energy_matrix=None` → both terms no-op (beam unchanged). Logged once.
- NaN energy for a track (the 1 missing, or a pier) → that edge's terms skipped (finite-guards above).
- Soft-only → no candidate exclusion, no infeasibility, no cascade. A configured energy knob that can't load its data warns loudly (per CLAUDE.md "a configured knob that can't act is a startup error") but degrades to no-op rather than crashing generation.

## Testing
- **Unit (synthetic energy matrices, no real data):** `compute_step_energy_target` interpolation; step-cap penalty fires only above cap and scales linearly; arc-band penalty vs interpolated target; `off` disables the arc but keeps the cap; NaN energy → terms skipped; **a candidate is never excluded** (penalty path only adjusts score, never `continue`s on energy). Per-mode preset wiring (config threads through). Mirror the `test_beam_pace_soft_penalty` / `test_beam_bpm_trust` patterns.
- **Generation (playlist-testing skill — mandatory):** multi-pier artist seeds through the `gui_fidelity`/`generate_like_gui` harness (never single-seed). Confirm: (a) generation **completes in all four modes** incl. strict (never-hard-fail) within the 90s budget; (b) the realized energy curve tightens strict→off and never whiplashes even at `off`.
- **Eval-gate before default-on:** generate per mode, plot the energy curve vs the pier targets, and a blind A/B (energy-on vs -off) on weakest-edge quality. Ship behind the knobs (default off) until it passes.

## Risks / assumptions
- Caps/bands are seeded from Pass-1 z-std distributions; the eval-gate tunes them. If energy coverage drops on a future re-analysis, the loader's NaN-skip keeps it safe.
- `energy_pair` assumed (Pass-1 leader); swappable by knob if Pass-2 prefers `energy_dist`/`arousal_p50`.
- Coexists with BPM/onset bands (additive); no interaction beyond summing into `_pace_penalty`.
