# HANDOFF: sonic centered-transition → genre soft-metric calibration

**To:** the genre soft-cosine session (calibrating `genre_pair_floor` / penalty strength).
**From:** the sonic centered-transition session.
**Date:** 2026-06-25.
**Branch:** `worktree-sonic-centered-transition` (not merged; based on master `449fc46`).

---

## BLUF — readiness

**Not fully unblocked yet — hold for the "verified under roam" signal below (short runway).**

- ✅ The **blocker in substance is fixed**: the sonic transition objective now discriminates. The `(x+1)/2` rescale that compressed `T` (good-vs-bad gap 8%) is replaced by a calibrated sigmoid (gap **88%**, offline). This is the discriminating objective your calibration needs.
- ⏳ **But the generation *path* you calibrate against is still moving** on two counts, so calibrating *now* would re-introduce the moving-target trap:
  1. The **`transition_floor` hard gate is being removed** (roam-only direction) — this changes which edges generation accepts.
  2. The objective is **not yet verified in real roam generation** (corridors + worst-edge minimax) — only in unit tests + the offline probe.

**Recommendation:** wait for the signal "**T8 green: roam generation verified**" (I'll post it / ping you). At that point the sonic objective AND the generation path are final, and you calibrate once, cleanly.

If you want to start *prep* now (reading the objective, wiring your harness), everything below is stable. Just don't fit floor/strength until the signal.

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
