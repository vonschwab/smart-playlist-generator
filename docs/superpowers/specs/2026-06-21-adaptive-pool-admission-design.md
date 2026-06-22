# Adaptive candidate-pool admission + genre steering reactivation — design

**Date:** 2026-06-21
**Status:** Draft (design) — awaiting Dylan review
**Related:** `docs/WIRING_STATUS.md` (the living wiring tracker this work updates); MERT re-fold (this session); memories `feedback_never_fail_three_axes`, `feedback_generation_time_budget`, `project_enriched_genre_authority`, `project_graph_similarity_integration`, `timbre_embedding_ceiling`; the `evaluation-methodology` and `playlist-testing` skills.

## Problem & goal

The candidate pool is gated by **fixed global floors** that do not fit how a given seed actually sits in the (now MERT) feature space, so for whole niches the pool starves. Log-verified (Charli XCX, narrow/narrow/narrow, MERT live):

```
Sonic floor applied: before=40387 after=686        ← absolute sonic floor 0.18 cuts ~98%
Sonic sim distribution: median=0.156  p95=0.429  floor=0.18   ← floor sits ABOVE the median
Genre hard gate applied: 517 excluded
Candidate pool: admitted=50 ... universe after dedupe=156
```

Hyperpop is sonically tight but *low-cosine* in MERT (everything is electronic/compressed, so even near-twins score ~0.15). The narrow sonic floor (0.18) is above this seed's median (0.156) → it rejects the majority before genre even acts; the hard genre gate (`min_genre_sim=0.4`) cuts 517 more. Result: a 50–156 candidate pool for a 30-track playlist → 14/30 one-artist output + the relaxation grind (bounded separately in `bc942d5`). Genre **arc steering** is independently dead: `genre_steering_source=dense` but the dim64 dense sidecar has a vocab mismatch (`X_genre_dense=None`) → `no usable g_targets` every segment.

**Goal:** make sonic and genre admission **per-seed adaptive** (a percentile of the seed's own similarity distribution, not an absolute global number), guarantee the pool never starves, and reactivate genre arc steering — landing on the MERT-live + graph-genre artifact, within the 90s budget, never hard-failing on sonic/genre. **And** remove the deprecated machinery we replace so nothing can be plugged into the wrong (stale) thing again.

## Non-negotiable constraints

- **Never-fail on sonic/genre** (`feedback_never_fail_three_axes`): admission is a soft/relaxable preference. The minimum-pool guarantee makes starvation impossible by construction. Diversity (min_gap, per-artist cap) stays the only hard constraint.
- **90s budget** (`feedback_generation_time_budget`): admission is O(pool) over already-computed 1-D distributions; the min-pool backstop is bounded.
- **MERT-live + graph genre**: design against the current artifact (`X_sonic_variant=mert`, graph-sourced `X_genre_raw/smoothed`). Genre *data authority* (graph vectors via `authority.py`) is untouched — only the admission *floor* changes from hard to adaptive.
- **Diverse calibration** (`evaluation-methodology`): percentile values + `min_pool_size` are set against a **diverse seed corpus** (hyperpop, metal, jazz, ambient, hip-hop, folk, Top-40), reported as distributions (min/p10/p50/p90), never tuned to one niche. Charli XCX is an outlier, not the calibration target.
- **Wiring discipline & cleanup (first-class, enforced — see §5):** one mechanism not two; no silent no-ops; retire the deprecated path on sight; every flip to ✅ proven on a real playlist+log.
- **Phased landing (no legacy shadow path)**: the steering-source flip and the deprecated-path deletions land immediately — they fix dead/broken wiring. The percentile mechanism *replaces* the absolute floors directly; there is deliberately **no preserved legacy default**, because keeping one is the stale-shadow-path we are eliminating. Safety comes from (a) a migration-guard test (percentiles chosen to approximate the old floor on a typical mid-library seed, so the mechanism swap is provably near-neutral there), (b) conservative initial per-mode values, and (c) the diverse-seed worst-edge eval-gate, which must pass before the per-mode values are final.

## Design

### 1. Single effective-floor resolution (percentile-primary), symmetric across axes

There is **one** function that resolves a seed's admission floor for an axis: `effective_floor(seed_sim_distribution, percentile) -> float`, built on the existing `src/playlist/pier_bridge/percentiles.py::floor_at_percentile`. It is the *only* operative floor path.

- **Sonic:** new `sonic_admission_percentile` (per `sonic_mode`). The per-seed sonic distribution is already computed in `candidate_pool.py` (the `Sonic sim distribution` log line). `effective_sonic_floor = floor_at_percentile(sonic_seed_sim, p)`. "narrow" = "nearest X% to *this* seed", so Charli's narrow floor lands near her 0.156 median, not above it.
- **Genre:** `genre_admission_percentile` is already implemented (`candidate_pool.py:741-800`) and computes a per-seed `effective_genre_floor`. We **make it the operative gate** and **remove the absolute `min_genre_similarity` hard gate** from the live path. The existing soft genre-compatibility penalty (ranking demotion, not rejection) is retained unchanged — genre still shapes the pool, it just stops being a hard cliff.
- The absolute `min_sonic_similarity` / `min_genre_similarity` are **removed from the operative gate** (not kept as a shadow path). Multi-seed aggregation matches the existing genre behavior (`np.min` of per-seed floors = most permissive across the pier seeds).
- Both effective floors are **logged with their resolved value and the percentile that produced them** (single source of truth, no ambiguity about which floor is live).

### 2. Minimum-pool guarantee (the never-fail backstop)

After both percentile floors, if the admitted, diversity-respecting pool is below `min_pool_size` (per mode), relax in this order until it reaches the target:
1. widen (lower) the percentiles by a bounded step;
2. last resort — admit the top-K most-similar candidates regardless of floor, **still respecting the per-artist cap and min_gap eligibility** (never a one-artist pool).

This guarantees the beam always has a workable, diverse pool — starvation becomes impossible at the source, which also removes the *cause* of the pier-bridge relaxation grind (the `bc942d5` budget remains as a safety net). The backstop is logged (`pool below min N; relaxed via …`).

### 3. Genre steering source → taxonomy (reactivate the dead arc)

Default `genre_steering_source = taxonomy`. The taxonomy arc (`build_taxonomy_genre_targets`, `pier_bridge_builder.py:1607`) uses in-artifact `X_genre_raw` + `genre_vocab` + the SP3a taxonomy graph — all present and rebuilt with every artifact, so it cannot go stale the way the dense sidecar did. Success = the `no usable g_targets` warnings vanish, replaced by `Genre steering [taxonomy]: … via [labels]`.

### 4. Components (one responsibility each)
- `src/playlist/candidate_pool.py` — the single `effective_floor` resolution (sonic + genre), the percentile-primary gate, the min-pool backstop. Remove the absolute-floor operative branches.
- `src/playlist/pier_bridge/percentiles.py` — reuse `floor_at_percentile` (no change expected).
- `src/playlist/mode_presets.py` — per-`sonic_mode` `sonic_admission_percentile`, per-`genre_mode` `genre_admission_percentile`, per-mode `min_pool_size`. Remove the absolute-floor preset entries.
- `src/playlist/config.py` — config fields for the above; delete the retired absolute-floor fields and `sim_variant`.
- `config.yaml` / `config.example.yaml` — set `genre_steering_source: taxonomy`; remove `playlists.sonic.sim_variant`.
- Steering-source guard (startup) — see §5.
- `docs/WIRING_STATUS.md` — updated per landed piece.

## 5. Wiring discipline & cleanup (REQUIRED — definition of done)

This is not optional polish. Each item below is part of "done"; a change is not complete until its old path is gone.

**No silent no-ops:**
- Every new admission knob logs its effective resolved value.
- `genre_steering_source` (and any source-selecting knob) **raises or warns loudly at startup if the selected source's data is absent/stale** — the dense-sidecar vocab mismatch that silently disabled the arc must become a loud failure, per CLAUDE.md ("a configured knob that can't act is a startup error").

**Retire what we replace (grep-clean):**
- **Dense genre steering:** remove the dense steering code path (`build_dense_genre_targets` usage in `pier_bridge_builder.py:1646-1666`), the `X_genre_dense` loader plumbing in `src/features/artifacts.py`, and the stale `data/artifacts/beat3tower_32k/data_matrices_step1_genre_emb_dim64.npz`. If a `dense` source value is retained at all, selecting it is a startup error (data is gone).
- **Absolute admission floors:** delete `min_sonic_similarity` / `min_genre_similarity` from the operative gate and their preset entries; migrate (don't silence) any test that asserted them.
- **Stale `playlists.sonic.sim_variant: tower_weighted`:** remove (the loader uses the artifact-baked variant).
- Any other deprecated admission/steering remnant found while implementing gets removed in the same pass, not catalogued for later.

**Proof:** each row in `docs/WIRING_STATUS.md` flips to ✅ only after a real-playlist+log run shows the new path firing AND a grep confirms the old path is gone from code/config.

## Error handling / graceful degradation
- NaN / empty seed distribution → that seed contributes no floor; if all seeds degenerate, fall to the min-pool top-K backstop. Never raise from the admission path.
- `min_pool_size` unreachable even after top-K (library smaller than min) → admit all eligible, log once. Diversity caps still hold.
- Taxonomy graph missing a path for a segment → that segment's arc is inactive (logged once per segment, as already wired) — not an error; the beam still builds via sonic.

## Testing
- **Unit:** `effective_floor` returns the correct percentile value on a known distribution; percentile-primary admits the right set; min-pool backstop guarantees ≥ `min_pool_size` and never returns a one-artist pool; NaN/empty handled; removing the absolute floor does not change behavior when percentile reproduces it (migration guard).
- **Generation (gui_fidelity / real worker for artist-mode), DIVERSE seeds, MERT live, `BPM loaded` verified:** for each of ≥6 disparate seed niches — pool never starves (admitted ≥ min, distinct artists ≥ floor); the gate-tally log shows percentile floors (not absolute); taxonomy steering fires (`Genre steering [taxonomy]`); all modes < 90s; worst-edge sonic does not regress vs the current absolute-floor baseline.
- **Cleanup guard:** a grep test asserting the retired symbols/keys/artifact are gone.

## Calibration & eval-gate (last, on the diverse corpus)
Set `sonic_admission_percentile` / `genre_admission_percentile` / `min_pool_size` per mode by ramping from minimal against the diverse seed corpus; the gate is **worst-edge sonic must not drop below threshold** while pools de-starve and distinct-artist counts rise. Report distributions (min/p10/p50/p90), name N and the seed niches. A mode that fails ships its safest passing value. Output: `docs/run_audits/adaptive_admission/CALIBRATION.md`.

## Risks / assumptions
- **The intended trade:** percentile admission for an outlier niche admits some lower-cosine tracks (fuller pool). Guard = the worst-edge eval-gate + min-pool diversity caps; the beam + edge-repair place the final order.
- Removing the absolute floors is a behavior change for non-outlier artists too; the migration guard + diverse-seed eval quantify it before any default-on.
- `candidate_pool.py` is a hotspot; changes are scoped to the admission gate and are mostly a symmetric extension of the existing genre-percentile path.
- Deleting the dim64 sidecar is irreversible-on-disk but the file is derived (re-buildable) and currently broken; back it up per artifact-dir discipline before deletion.
