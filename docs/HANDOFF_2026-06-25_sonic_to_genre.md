# HANDOFF: sonic centered-transition → genre soft-metric calibration

**To:** the genre soft-cosine session (calibrating `genre_pair_floor` / penalty strength).
**From:** the sonic centered-transition session.
**Date:** 2026-06-25.
**Branch:** `worktree-sonic-centered-transition` (not merged; based on master `449fc46`).

---

## BLUF — readiness

**✅ UNBLOCKED — T8 GREEN (roam generation verified 2026-06-25). Calibrate away.**

The full sonic fix has landed on this branch and is verified in real roam generation:
- ✅ Objective discriminates: the `(x+1)/2` rescale (good-vs-bad gap 8%) is replaced by a calibrated sigmoid (gap **88%** offline).
- ✅ Generation **path is final**: the `transition_floor` hard gate is **removed** (roam-only — roam shapes via corridor + worst-edge minimax, no elimination).
- ✅ **Verified in a real roam generation** (`scripts/research/verify_roam_transition.py`): 3-pier narrow run, `BPM 40812/40812`, beam-built, roam corridors active, **23.0s** (< 90s ceiling). Selected-edge `T` **de-compressed to 0.13 → 0.78** (p10 0.31 / p50 0.51 / p90 0.67), vs the old metric parking every edge at ~0.52–0.77. The worst edge (T=0.126, a fixed-pier transition) is now correctly surfaced where the old metric hid it ~0.56.

**Calibrate with roam enabled** (it already is in `config.yaml`: `pier_bridge.roam.enabled=true`, `worst_edge_minimax=true`). Playlist quality is no longer confounded by a washed-out sonic objective — your `genre_pair_floor`/penalty changes now reflect genre, not sonic noise.

Only remaining manual item on the sonic side: a **perceptual audition** (human ears on a few seeds' worst edge) — does not block your calibration.

## What is delivered (committed, won't change materially)

| commit | what |
|---|---|
| `66f86e2` | Calibrated sigmoid `_calibrate_transition_cos` replaces `(x+1)/2` in `transition_metrics.py::score_transition_edge`. Live via context defaults. 28 transition unit tests green. |
| `58b4cee` | `scripts/research/calibrate_transition_sigmoid.py` — derives params + (now-obsolete) floor sweep. |
| `467a954` | Spec re-grounded for roam: remove the gate, verify under roam. |

The objective itself: `T = Σ wᵢ · σ(gain·(cosᵢ − center)/scale)` over end→start / mid→mid / full→full (weights 0.70/0.15/0.15). Params **center≈0.32, scale≈0.062, gain≈1.0** (provisional defaults ≈ the script's recommendation; any final tweak is sub-0.01 and does not change discrimination). Library-wide this puts realistic-edge `T` at p10≈0.13 / p50≈0.30 / p90≈0.67 (was a compressed ~0.59–0.74).

## What is still pending (these move your target — wait for them)

- **Remove `transition_floor` hard gate** — drop `T<transition_floor` from `is_broken_transition` + the `beam.py:679` direct-floor reject + the dead config knob. Keep the `−0.5` anti-alignment safety. (Changes edge acceptance.)
- **Verify under roam** — generate with `roam_corridors_enabled` + `worst_edge_minimax_enabled`, read the per-edge audit, confirm the worst edge improved and `T` is de-compressed, generation < 90s. **This is the gating signal for you.**
- (Cleanup — rescale consolidation, dead-module/`T_used`/gamma removal — does NOT affect the objective or generation behavior; safe to ignore.)

## How to consume it (when the signal lands)

1. Base your calibration on `worktree-sonic-centered-transition` at the commit I tag "T8 green" (or after merge — see order below).
2. **Calibrate with roam enabled** (`overrides[pier_bridge][roam]` + worst-edge minimax). Roam is the target path; judge `genre_pair_floor`/strength against roam playlist quality, not the legacy beam.
3. The sonic objective is sound under roam at that point — playlist quality is no longer confounded by the sonic side, so your floor/strength changes reflect genre, not sonic noise.

## Order (locked)

**sonic fix lands → genre calibration → merge.** Roam-promotion (flipping roam to default / deleting the legacy beam) is a *separate* downstream effort this unblocks — not part of either of our current changes.

## Coordination notes

- Config overlap is minor (both touch `pier_bridge/config.py` + `config.yaml`, different fields) — a one-line merge; stage explicit paths.
- I'm removing the `transition_floor` knob from config; if your work references it, it's going away (it's a hard gate, counter to the roam/soft-penalty direction).
