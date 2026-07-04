# Handoff — Roam Corridors: Phase-3 calibration against MuQ (never done)

**Date:** 2026-07-04. **Surfaced during:** a Tier-3 dead-code keep/kill review of the dense
`mutual_proximity()` in `pier_bridge/manifold.py`. The keep/kill turned into a real finding.

## TL;DR

Roam Corridors is **LIVE and actively shaping every playlist**, but it is running on **MERT-era
parameters it was never re-tuned for** after the MERT→MuQ swap. A calibration **was** done at build
time — **2026-06-24, commit `15e7b20`** (gate-sweep + the "minimax is essential" finding via
`slider_differentiation_eval.py --roam/--gate-sweep`) — so "calibration happened" is *true*, but it
was tuned against **MERT**. There has been **no re-calibration since the 2026-07-01 MuQ swap**;
`manifold.py`/`roam.py` are frozen at 2026-06-24. The exact Flexer/Schnitzer hubness correction
(`mutual_proximity()`) was built but **never wired** (only a cheap approximation is live), and MuQ's
known hubness/collapse problem makes the exact form potentially *more* valuable now than when it was
built. **Good news for redoing it:** the calibration harness (`slider_differentiation_eval.py`) was
MERT-hardcoded but a concurrent session just **modernized it for MuQ (`a9602b9`)** — so re-running the
roam sweep on MuQ is now feasible. This is a bridging-quality **opportunity**, not cleanup.
**DO NOT delete `mutual_proximity()`.**

## What Roam Corridors is

On-manifold kNN geodesic corridors: build a distance-weighted kNN graph over real tracks, take the
geodesic (shortest-path) distance from each pier, and **soft-penalize candidate bridge tracks that
detour off the on-manifold path** between pier A and pier B (`corridor_penalty`). Intended to keep
bridges on the "natural" manifold route rather than cutting straight across empty embedding space.
Original design/plan: `docs/superpowers/plans/2026-06-24-roam-corridors-engine.md` (Phase-3
calibration was the explicitly-deferred next step; the plan says the dense + approx MP "both ship").

## Current state (verified 2026-07-04)

- **LIVE, not inert.** `config.yaml:138` has a `roam:` block that enables it. `beam.py:748-758`
  computes `corridor_penalty(roam_detour_sonic, cfg.roam_width_sonic, slope=cfg.roam_penalty_slope)`
  when `cfg.roam_corridors_enabled`, and `beam.py:1303-1309` **adds `_roam_pen` to each candidate's
  beam score**. Effective config in a real 2026-07-03 Torrey log: `roam_corridors_enabled=true`,
  `roam_knn_k=25`, `roam_mutual_proximity=true`, `roam_width_sonic=1.0`, `roam_penalty_slope=1.0`;
  `Roam[seg N]: sonic detour …` lines fire per segment. (The `roam_corridors_enabled` dataclass
  DEFAULT is `False` — it is the live `config.yaml` block that turns it on.)
- **Frozen at MERT-era.** `manifold.py` + `roam.py` = three commits, **all 2026-06-24** (`83524b1`
  kNN graph + MP + geodesics, `4b0ce05` per-segment deviation assembly, `1e01260` review fixes).
  Untouched since. The **MERT→MuQ swap was 2026-07-01 (SP-B)** — *after* roam was built and tuned.
  Zero re-calibration since.
- **Embedding-agnostic.** Roam operates on `X_full_norm` (the current L2-normalized sonic matrix,
  now MuQ) with no MERT-specific dims/keys. So it is NOT MERT-broken — it runs on MuQ. It is just
  tuned for MERT's geometry.
- **Two mutual-proximity paths:**
  - **CHEAP approx (LIVE):** `build_knn_graph(X, k, mutual_proximity_approx=True)` — a cheap
    degree-based hubness rescale on the realised sparse edges. `roam_mutual_proximity=true` drives
    this. This is what runs today.
  - **EXACT dense (BUILT, NEVER WIRED):** `manifold.py:21 mutual_proximity(dist)` — the exact
    Flexer/Schnitzer transform, O(N²) dense. Referenced ONLY by `tests/unit/test_manifold.py`. Never
    called in production.
- **Undocumented in the template.** `config.example.yaml` has **no `roam:` block** — the feature is
  live-only in `config.yaml`; a fresh clone wouldn't know it exists.

## Why this is an opportunity, not cleanup

1. **Un-calibrated for the live space.** `roam_knn_k`, `roam_penalty_slope`, `roam_width_*`, and the
   MP-approx choice were set for MERT's geometry. MuQ's neighbourhood structure differs; the corridor
   width/steepness that made sense for MERT may be wrong (too tight, too loose, or net-harmful) on MuQ.
2. **MuQ hubness makes the exact MP more relevant.** Mutual proximity is a **hubness correction**.
   MuQ **collapses quiet/near-silent audio onto ~one vector** → extreme hubs (see
   `memory/project_muq_collapse_quiet_audio.md`, and the First Jam incident in
   `project_pier_bridgeability`). Under MERT hubness was milder; under MuQ it's worse. The exact dense
   MP is a candidate **mitigation for hubs in the corridor graph** — plausibly worth more now than at
   build time.
3. **The exact MP is cheap *here*.** Corridors run on a **small per-segment node set** (kNN over the
   segment's candidate nodes, not the whole 40k library), so the O(N²) dense MP is feasible in this
   context — an exact-vs-approx A/B is not expensive.

## Proposed Phase-3 work (future session)

1. **Measure first — is roam helping or hurting on MuQ?** A/B roam ON vs OFF on real multi-pier
   generations through the `gui_fidelity` harness (worst-edge T, min-T, mean-T, distinct-artist).
   Running a live lever on wrong-era params could be *net-negative* today — if so, that's urgent, not
   a nice-to-have.
2. **Sweep the corridor params against MuQ:** `roam_knn_k`, `roam_penalty_slope`, `roam_width_sonic`
   (and `roam_width_genre`/`roam_width_energy` if reviving those axes). Re-fit for MuQ geometry.
3. **A/B exact dense `mutual_proximity()` vs the cheap approx** in `build_knn_graph`: wire the exact
   form behind the `roam_mutual_proximity` flag (or a new tri-state), measure hubness reduction +
   worst-edge impact, **especially on collapse-prone (quiet/ambient) segments** where MuQ hubs bite.
4. **Decide + document:** keep approx / adopt exact / make it a mode. Add a calibrated `roam:` block
   to `config.example.yaml` (currently missing) so the feature is discoverable and the tuned defaults
   ship.

## Files

- `src/playlist/pier_bridge/manifold.py` — `mutual_proximity` (dense, unwired), `build_knn_graph`
  (approx, live), `geodesic_from_source`.
- `src/playlist/pier_bridge/roam.py` — `build_knn_graph(..., mutual_proximity_approx=…)`, per-segment
  deviation assembly.
- `src/playlist/pier_bridge/beam.py:748-758,1303-1309` — `corridor_penalty` consumption in scoring.
- `config.yaml:138` `roam:` block (LIVE); `config.example.yaml` (MISSING — add during step 4).
- `docs/superpowers/plans/2026-06-24-roam-corridors-engine.md` — original; Phase-3 was the deferred step.
- Related memory: `project_roam_corridors_engine`, `project_muq_collapse_quiet_audio`,
  `project_pier_bridgeability`.

## DO NOT

- **Delete `mutual_proximity()`** — it is the intended exact hubness correction, un-wired but relevant.
- **Calibrate blind** — measure roam ON-vs-OFF on MuQ (step 1) *before* touching any param.
