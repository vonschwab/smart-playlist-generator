# Sonic Tower-Weighted Fix (Phase 1) — design

**Date:** 2026-06-01
**Branch:** sonic-neighborhood-validation (off `genre-edge-safeguards`)
**Status:** approved, ready for implementation plan
**Memory:** [[project-sonic-tower-weights-inert]]

## Problem

The intended timbre-dominant tower weighting (rhythm 0.20 / timbre 0.50 / harmony 0.30),
treated as load-bearing by CLAUDE.md principles #17/#18 and "Knob 0", is **not applied
anywhere in the production sonic path.** Verified by code execution 2026-06-01:

- The shipped artifact is the `robust_whiten` variant (global RobustScaler + global
  PCA-whiten over 86 dims), built on top of the `Beat3TowerNormalizer`'s per-tower PCA
  whitening — i.e. the space is **whitened twice** and carries **no tower weighting**.
- Three mechanisms that would apply the weighting all silently no-op against the 86-dim
  artifact (each expects 137-dim raw towers or a 32-dim tower_pca): the build variant,
  the runtime variant re-application (`embedding_setup.py`), and the scoring-time
  `apply_transition_weights`.
- Consequence (measured): candidate-pool sonic cosine compresses to a ~0.5 ceiling
  (whitening pushes vectors toward orthogonality), and timbre's 57 dims do not dominate.

## Goal

Make the production sonic space actually be the intended **tower-weighted** space, by
rebuilding the artifact in a `tower_weighted` variant and cleaning the runtime path so the
artifact's own preprocessing is used directly. Validate before/after on the reference seeds.
**This is a correctness fix, not a tuning change** — Phase 2 (audition) judges whether the
corrected space sounds better and whether to go further (e.g. raw-137, no whitening).

## Chosen variant: `tower_weighted` (86-dim)

Per-tower L2-normalize the existing normalized towers, scale each by √weight, concatenate:

```
tw = concat( sqrt(w_r)·l2(rhythm_9), sqrt(w_t)·l2(timbre_57), sqrt(w_h)·l2(harmony_20) )
```

with `(w_r, w_t, w_h) = (0.20, 0.50, 0.30)` read from `config.yaml`
`playlists.ds_pipeline.tower_weights`. This **applies** the timbre dominance and **drops
the redundant second (global) whitening**, while keeping the `Beat3TowerNormalizer`'s
per-tower denoising whitening (already baked into the per-tower matrices). It is the
conservative middle between the current double-whitened space and a raw-137 no-whitening
space (the latter is a Phase-2 lever if needed).

**Measured effect (cosine to all tracks, current vs tower_weighted):** top-end lifts
(Real Estate max 0.47→0.57, James Brown 0.46→0.58, Sonic Youth 0.68→0.74); random-pair
p90 also rises 0.137→0.159 (de-whitening spreads signal *and* noise — modest, with a
trade-off). The decisive win is correctness (weighting applied at all); the compression
relief is modest, which is why Phase 2 still matters.

## Surgical rebuild (no DB read, genre untouched)

All per-tower matrices for full + start/mid/end are already in the artifact npz
(`X_sonic_rhythm[_start/_mid/_end]`, `X_sonic_timbre*`, `X_sonic_harmony*`). So the rebuild
reads the existing npz and writes a new one with:

- `X_sonic_tower_weighted` = tower_weighted **full** (new variant key the loader selects)
- `X_sonic` = tower_weighted full (so the raw-key fallback is also consistent)
- `X_sonic_start` / `X_sonic_mid` / `X_sonic_end` = tower_weighted **segments**
  (the loader reads these keys directly for the transition metric — they are NOT
  variant-aware, so they must be overwritten)
- `X_sonic_variant` = `"tower_weighted"`, `X_sonic_pre_scaled` = `True`
- **everything else copied byte-identical**: all `X_genre_*`, `genre_vocab`, `track_ids`,
  `track_artists`, `track_titles`, `artist_keys`, `durations_ms`, `bpm_array`,
  per-tower matrices, `normalizer_params`, `tower_calibration`, `build_config`,
  `sonic_feature_names`, `tower_dims`.

Because `track_ids` and genre matrices are unchanged, the existing dense-genre sidecar
(`*_genre_emb_dim64.npz`) stays valid — no sidecar rebuild needed. The DB and audio files
are never touched (read-only honored; only a new npz is written).

Deployment: back up the current artifact to `*.npz.bak_<ts>`, then swap the rebuilt file
into the production path in place (same stem → sidecar still matches). Reversible by
restoring the backup. After swap in a running worker, `load_artifact_bundle.cache_clear()`.

## Runtime fix (`src/playlist/pipeline/embedding_setup.py`)

The current code matches `bundle.sonic_variant` against a `resolved_variant`
(which is `tower_pca`, since config's `sim_variant` is never passed). The names never match,
so it falls into a re-transform that no-ops and also mislabels the space as not pre-scaled —
causing an extra StandardScaler before the hybrid PCA. Fix:

- **When `bundle.sonic_pre_scaled` is True, use `bundle.X_sonic` directly** (the artifact is
  the source of truth for its own preprocessing); set `variant_stats = {variant: bundle.sonic_variant, pre_scaled: True}`.
  Drop the fragile name-equality check.
- Build the hybrid with `pre_scaled_sonic=True` and **`use_pca_sonic=True`** (always PCA the
  sonic block to 32 components, but skip the redundant StandardScaler when pre-scaled). This
  keeps the hybrid embedding at the design's balanced 32-dim-sonic + 32-dim-genre.

### Decision: keep hybrid sonic at 32-dim PCA (not full 86)

Feeding the full 86-dim tower_weighted into the hybrid would unbalance the
`[w_sonic·E_sonic, w_genre·E_genre]` concat (86 vs 32). Keeping the sonic block PCA'd to 32
confines Phase 1's behavioral change to "weighting applied" + "double-whiten removed" in the
**pool-sim and transition** paths (where they matter), without re-balancing the hybrid.
Reversible; revisit in Phase 2 if warranted.

### Note: this also changes the current artifact's runtime path

With the fix, the current `robust_whiten` artifact would also take the pre-scaled-direct
branch (the loader already sets `pre_scaled=True` for it), removing the extra StandardScaler.
That is why Phase 1 captures the **baseline on unmodified code** and validates the combined
(runtime fix + new artifact) result against it, rather than asserting equivalence.

## Scoring / invariant

With weighting baked into both full and start/end vectors, the "transition_weights ==
tower_weights" invariant (Knob 0) holds **structurally at build time**, and
`apply_transition_weights` is a legitimate no-op rather than a silent bug. `center_transitions`
continues to center the (now tower-weighted) start/end space.

## Config + docs

- `config.yaml`: set `playlists.sonic.sim_variant: tower_weighted` (with a comment pointing
  to this drift fix); confirm `tower_weights`/`transition_weights` stay 0.20/0.50/0.30.
- `docs/CONFIG.md`: correct the "tower_pca (default)" claim; document `tower_weighted` as the
  shipped variant and why.
- `CLAUDE.md` / `docs/PLAYLIST_ORDERING_TUNING.md` "Knob 0": note the weighting is now baked
  at build time in the `tower_weighted` artifact.

## Validation (definition of done)

A repeatable metrics script over the reference seeds (Charli XCX / Real Estate / Bill Evans /
Beach House / Minor Threat + the broader 17), run on baseline (unmodified) then on the fix:

1. **Smoke:** all reference seeds generate via the GUI-fidelity harness without crash or new
   infeasibility.
2. **Cosine spread:** top-end (max/p99/p90) of full-track sim widens vs baseline (matching the
   pre-measured deltas); record random-pair p90 trade-off.
3. **Weighting active:** per-tower contribution to the space reflects 0.20/0.50/0.30 (timbre
   dominant) — assert the tower L2 magnitudes/contributions, which baseline lacks.
4. **No ordering regressions:** distinct-artist counts, min-gap, and S/T transition
   distributions on reference playlists are reported before/after; no seed becomes infeasible.
5. Full fast test suite green (`pytest -m "not slow"`).

Findings + before/after metrics saved under `docs/run_audits/sonic_phase1/`.

## Components & boundaries

- `src/features/sonic_rebuild.py` — pure helpers: `tower_weighted_from_towers(...)` and
  `build_tower_weighted_arrays(npz_data, weights)`. Unit-tested, no I/O.
- `scripts/rebuild_sonic_tower_weighted.py` — thin CLI: read npz → build arrays → backup →
  write. No DB.
- `scripts/sonic_phase1_metrics.py` — baseline/after metrics over reference seeds (reuses the
  GUI-fidelity harness; writes JSON under `docs/run_audits/sonic_phase1/`).
- `src/playlist/pipeline/embedding_setup.py` — runtime pre-scaled-direct fix.
- `config.yaml`, `docs/CONFIG.md`, `CLAUDE.md`, `docs/PLAYLIST_ORDERING_TUNING.md` — config/docs.

## Out of scope (Phase 2 or later)

Raw-137 / no-whitening variant; re-balancing the hybrid to full-86; tuning the weight values
themselves; any audio re-extraction. Phase 1 only makes the *documented intent* real and
clean, then measures it.
