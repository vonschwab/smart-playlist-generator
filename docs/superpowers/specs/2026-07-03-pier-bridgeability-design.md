# Pier bridgeability check — design

**Date:** 2026-07-03
**Status:** approved design, pre-implementation
**Motivating incident:** `logs/playlists/2026-07-03_214535_Torrey_03350f.log` — Torrey "First Jam"
(a jam/filler recording, `Something Happy` track 10) was seated as pier 6 of 10 in an artist-mode
generation. Its best sonic cosine to anything in the library is ≈0.10 (calibrated T ≈ 0.03), so
both edges touching it landed at T=0.002/0.003. Every downstream fixer fired (bridge_floor relaxed
to 0.00, search expanded to beam 200, var-bridge flexed, tail-DP applied, edge repair evaluated
thousands of candidates) and none could win, because piers are fixed by design and no neighbor
scores better than ≈0.06 into that track. Weakest-edge #3 (Garage Intermission → Candy Claws,
T=0.074) is the same failure in miniature.

## Problem

Medoid scoring in `src/playlist/artist_style.py::_medoids_for_cluster` is entirely
*within-cluster*: similarity to the cluster's own centroid, duration typicality, and optional
energy/popularity/tag terms. A track can be perfectly central to its own (outlier) cluster and
duration-typical while having **no sonic neighbors anywhere in the library**. Nothing checks
cluster-vs-library. In artist mode the pier is *discovered*, not user-requested — the user asked
for "Torrey", not for "First Jam" — yet the pipeline grants it the same untouchability as an
explicit seed (repair and edge-delete both refuse piers by design).

## Decision

Add a **hard bridgeability veto at medoid-selection time** (approach A of three considered; the
soft-penalty variant was rejected because within an all-outlier cluster relative scoring changes
nothing, and the post-hoc-audit variant was rejected for duplicating cluster-representation logic
outside `artist_style`).

### Signal

For each candidate member track *i* of the seed artist:

```
bridgeability(i) = calibrate_T( kth_largest( cos(X_norm[i], X_norm[j])
                                             for all j NOT belonging to the seed artist ) )
```

- `X_norm` is the full-library L2-normalized `X_sonic` already computed in
  `cluster_artist_tracks` (MuQ, 512-dim).
- Same-artist exclusion uses the identity-normalized `artist_key` (interiors can never be
  seed-artist tracks, so same-artist neighbors are irrelevant to bridging).
- `calibrate_T` is `src/playlist/pier_bridge/vec.py::_calibrate_transition_cos` with the **same
  calibration params the live transition scorer uses** (read from config, not hardcoded) — the
  threshold is therefore in the same units as `tail_dp_floor` and repair's `t_floor`.
- **Pass iff `bridgeability(i) >= floor`.** Defaults: `k = 10`, `floor = 0.30`. k=10 means "at
  least ten library tracks this could sit next to at an acceptable T" — one accidental sonic twin
  cannot rescue a jam track.

Cost: one members×library matmul per generation (~100×40k×512 worst case), computed once in
`cluster_artist_tracks`; milliseconds.

### Placement and gating

- Computed once in `cluster_artist_tracks`; applied as a **member filter before**
  `_medoids_for_cluster` scoring. Surviving members are scored exactly as today.
- Because it runs on the medoid path, it covers artist mode with popularity **off** and **on**
  (popularity only biases scoring among surviving members).
- **Fire mode (`popular_seeds_mode == "fire"`) is exempt**: `select_popular_piers` output is
  never checked — top-N Last.fm hits are explicitly requested by the user. However, fire's
  cache-miss **fallback** to medoid piers IS checked (those piers are ordinary medoids, not
  bangers).
- Seeds mode (user-picked seed tracks as piers) is a different code path and is untouched.

### Reallocation on cluster failure (user decision 2026-07-03: reallocate)

- A cluster whose members ALL fail contributes **no piers**; its slots are refilled from passing
  clusters' next-best medoids (capped by cluster size) so `target_pier_count` holds — e.g. still
  10 Torrey piers, the jam cluster simply loses representation.
- If the total still falls short after reallocation, proceed with fewer piers and log a warning.
- If the check would leave **zero piers** (pathological all-outlier catalog), log a loud WARNING
  and fall back to unchecked medoids — a playlist never fails on a soft axis.

### Config

Under `artist_style` in `config.yaml` (+ `config.example.yaml`), live defaults per the
activate-fixes rule:

| Key | Default | Meaning |
|-----|---------|---------|
| `pier_bridgeability_enabled` | `true` | Rollback knob only; the check is the live default. |
| `pier_bridgeability_floor_t` | `0.30` | Minimum k-th-neighbor calibrated T to seat as a pier. |
| `pier_bridgeability_k` | `10` | Which neighbor rank must clear the floor. |

Discipline: enabled-but-can't-act (missing `X_sonic`, missing calibration) **raises at startup**
— never a silent no-op.

### Logging (part of the feature)

- Per-cluster INFO line: members vetoed / survivors, with each vetoed track's best-T (e.g.
  `Pier bridgeability: vetoed 'First Jam' (kth-T=0.03 < 0.30) — 0/4 eligible in cluster 3;
  slots reallocated`).
- Reallocation and zero-pier-fallback log at WARNING.

## Testing

1. **Unit** (`tests/`): bridgeability function on synthetic vectors (planted outlier fails,
   planted twin-cluster passes; k-th-rank semantics; same-artist exclusion); empty-cluster
   reallocation preserves pier count; zero-pier fallback path.
2. **Fast harness guard**: the new config keys resolve through `generate_like_gui` /
   `resolve_gui_overrides` (playlist-testing skill rules — no hand-built overrides).
3. **Live verify** (before declaring success): regenerate Torrey artist-mode through the real GUI
   path; confirm First Jam / Garage Intermission no longer seat as piers, min-T rises off the
   floor, and the per-playlist log shows the veto lines. Worker restart required first.
4. **Golden note**: this intentionally changes generation output; golden captures will shift.
   That is the feature, not a regression.

## Out of scope

- Making discovered piers swappable/deletable downstream (repair/edge-delete still refuse piers).
- Any change to fire-mode pier selection or seeds mode.
- Bridgeability against the *run-specific candidate pool* (circular: the pool is built from the
  piers). Library-wide is the non-circular signal; if nothing in the library is close, no pool
  can contain a good neighbor.
