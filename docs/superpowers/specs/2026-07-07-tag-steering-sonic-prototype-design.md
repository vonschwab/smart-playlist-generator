# Tag steering — genre-conditioned sonic-prototype signal (design)

**Date:** 2026-07-07
**Status:** Design approved; implementation plan pending.
**Prior art:** `docs/superpowers/specs/2026-07-02-tag-steering-design.md` (Stage 1 pool+pier levers),
`docs/superpowers/specs/2026-07-03-tag-aware-pier-allocation-design.md` (Stage 1.5),
memory `project_tag_steering`. This is a **signal upgrade** to the existing levers, not a new subsystem.

## Problem

User report: tag steering is too weak at *choosing seeds and bridge tracks*. A live diagnostic
on **Brian Eno** (bimodal catalog: early art-rock/glam vs. later ambient) and **Real Estate**
(genre-blended jangle/dream-pop) confirmed the report and localized the cause.

### Diagnostic evidence (2026-07-07, real `create_playlist_for_artist` runs, fire off)

The steer is **not inert** — ambient vs. art-rock produce genuinely divergent playlists. But it
leaks in three places, all traceable to **one root cause**:

**Root cause — the lever scores on `X_genre_dense`, which is built from genre *tags*, and this
library's genre data lives at the artist/album level.** So a track's genre vector barely varies
within an artist:
- Real Estate: genre-dense affinity to `jangle pop` across its 55 tracks has std **0.090** — the
  lever literally cannot tell its jangliest track from its dreamiest.
- Eno only works because his *albums* are tagged differently, giving his genre vectors real
  variance.

Observed leaks:
1. **Beam (bridge interior) is tag-blind.** `pier_bridge_builder.py` has no tag term (its
   `steering` is the unrelated taxonomy genre-*arc* system). Bridges inherit only whatever lean
   the candidate pool carries.
2. **Pool lever is gentle and pole-asymmetric.** At `blend=0.5` the admitted bridge pool leans
   weakly: ambient `p50=0.079`, art-rock `p50=0.309`.
3. **Pier selection is capped by sonic-cluster / genre-pole misalignment.** Ambient piers reached
   only `[0.32, 0.205, 0.205, -0.009]` (one forced off-pole by the ≥1-pier-per-cluster arc floor)
   because only 1 of 5 sonic clusters was even mildly ambient. Art-rock aligned well: `[0.865,
   0.865, 0.819, 0.74]`.

## Validated insight

MuQ sonic (`X_sonic_muq`, 512-dim) is **per-track and high-resolution**. Let the selected tag(s)
learn a **sonic prototype** from the library — the centroid of MuQ vectors for tracks carrying the
tag — and score tracks by proximity to it. This gives the track-level resolution genre tags lack.

### Probe results (full pool, read-only, seed-artist excluded, album-era = independent ground truth)

Library support per tag is ample: `jangle pop` 775, `ambient` 4177, `art rock` 3152,
`dream pop` 2628, `art pop` 1940 tracks.

**Eno (ground truth = album era, independent of the MuQ space under test):**

| Prototype | ambient-era | art-rock-era | separation |
|---|---|---|---|
| sonic `ambient` (4138 lib tracks) | **+0.671** | +0.059 | **+0.613** |
| sonic `art rock` (3113) | −0.291 | +0.163 | **−0.454** |
| *null* `hip hop` (control) | −0.083 | −0.224 | +0.142 (noise) |
| ref: genre-dense `ambient` | +0.139 | +0.014 | +0.125 (**weak**) |
| ref: genre-dense `art rock` | −0.136 | +0.817 | −0.953 |

The one pole where genre-dense is weak (ambient, +0.125 — matching the flabby diagnosed piers/pool)
is exactly where the sonic prototype is strongest (+0.613, ~4× the null).

**Real Estate (genre-flat case):** genre-dense spread std **0.090** → sonic-prototype spread std
**0.234** (2.6× more resolution). Face-valid: the jangle prototype ranks "Talking Backwards" /
"Crime" (signature jangle singles) top and puts **"Sting" (the beatless ambient interlude) dead
last** — the exact track that wrecked the worst edge in an earlier RE steer. Genre-dense couldn't
see it.

**Caveats honored:** the null wasn't perfectly zero (+0.142 generic low-energy component → fixed by
centering the prototype against the global sonic mean); neither signal dominates universally
(genre-dense won for art rock where albums are cleanly tagged) → **combine them**.

## Design

### Combined signal

```
tag_score(track) = cos(X_genre_dense[t], genre_target)              # coarse, album-level
                 + w_sonic · cos(X_sonic_muq_centered[t], sonic_prototype)  # fine, track-level
```

Additive by design: for a genre-flat artist the genre term is ~constant and drops out of the
ranking, so the sonic term decides; for an album-varying artist both contribute. No mode switch,
no per-artist branching.

### Prototype resolver (`src/playlist/tag_steering.py`, sibling to `resolve_tag_steering_target`)

`resolve_tag_sonic_prototype(tags, *, X_sonic_muq, track_ids, tag_track_ids, seed_artist_track_ids,
global_sonic_mean) -> (prototype | None, support_n, cohesion)`:

1. Pool library track rows carrying **any** selected tag (label source = genre authority; see
   `genre-data-authority` skill — use the authority-consistent effective-genre view, not raw
   `track_genres`).
2. **Exclude the seed artist's own tracks** (else the prototype partly describes the artist).
3. Mean the L2-normalized MuQ rows; **subtract `global_sonic_mean`** (kills the generic component);
   renormalize.
4. **Support guard:** if `support_n < tag_steering_prototype_min_support` (~25) **or** intra-set
   cohesion (mean cosine of members to the prototype) is below a floor → return `None` and
   **WARN loudly**; callers fall back to the genre-dense-only signal. Never a silent no-op
   (project gotcha: "a configured knob that can't act is a startup error, not a silent no-op").

`global_sonic_mean` is the mean of all normalized `X_sonic_muq` rows (computed once per run, or
cached on the bundle).

### Consumer 1 — piers (fixes "choosing seeds")

- `artist_style.py` (~L911): the `tag_slice` fed to `_medoids_for_cluster` via `medoid_tag_weight`
  currently = `X_genre_dense[members] @ genre_target`. Enrich to the combined signal:
  add `w_sonic · (X_sonic_muq_centered[members] @ sonic_prototype)`.
- `playlist_generator.py` (~L1930): `cluster_affinities` for `allocate_piers_by_tag_affinity`
  currently = `mean(X_genre_dense[members] @ genre_target)`. Add the sonic term the same way.
- The ≥1-pier-per-cluster arc floor stays (arc preservation; user's option-2 lean). The improved
  signal simply picks better medoids and skews slots more accurately.

### Consumer 2 — candidate pool (makes bridges on-tag without touching the beam)

- The **existing genre pool lever stays** (`candidate_pool.py` ~L825, blends `genre_target` into
  the genre-admission centroid). We **add a parallel sonic pool lever**: blend the sonic prototype
  into the **sonic**-admission centroid, keyed by `tag_steering_sonic_blend`. The pool is thus
  steered on both axes — genre admission and sonic admission — rather than via a single additive
  per-track score (that additive form is the pier consumer). Result: the bridge candidate pool is
  genuinely on-tag, so the (still tag-blind) beam selects on-tag bridges with no beam change.
- Emit the same style of audit line the genre lever does (blend value + admitted-set affinity
  distribution p10/p50/p90/n), per "diagnostic logging is part of the feature."

### New config knobs (`playlists.ds_pipeline.pier_bridge`)

| Knob | Default | Meaning |
|---|---|---|
| `tag_steering_sonic_weight` | 0.5 | weight of the sonic-prototype term in pier scoring |
| `tag_steering_sonic_blend` | 0.35 | blend of prototype into the sonic-admission centroid (pool) |
| `tag_steering_prototype_min_support` | 25 | min library tracks/tag to trust the prototype |

Soft, tunable, documented with a recipe. **Byte-identical to today when no tags are selected**
(all new paths gated on a resolved prototype, which requires selected tags).

## Data flow

`UIStateModel.steering_tags` → policy → `playlists.ds_pipeline.pier_bridge.tag_steering_tags`
(unchanged) → generator resolves **both** targets once: `resolve_tag_steering_target` (genre) and
`resolve_tag_sonic_prototype` (sonic) → combined signal threaded to `cluster_artist_tracks` /
`allocate_piers_by_tag_affinity` (piers) and `build_candidate_pool` (pool).

## Testing / validation

Mirror production via the real artist path (playlist-testing skill — never hand-built overrides):
- **Unit** (`tests/unit/test_tag_steering.py`): resolver support guard + warn, seed-artist
  exclusion, global-mean centering, multi-tag union, unmapped/low-support fallback.
- **Integration** (mark `integration`+`slow`): re-run the Eno + Real Estate diagnostic; assert
  (a) Eno ambient pier affinity rises materially vs. the genre-dense-only baseline, (b) Real
  Estate pier-ranking spread increases and interlude-type tracks ("Sting") drop out of pier
  selection, (c) **no-tag runs are byte-identical** to pre-change.
- **Golden:** confirm no-tag path unchanged.
- **Centering check:** the probe used *uncentered* MuQ cosine (null bias +0.142). Confirm that
  subtracting `global_sonic_mean` actually shrinks the null-tag separation without eroding the
  signal separation before committing to centering as the default.

## Risks & tradeoffs

- **Sonically-multimodal tags** (e.g., "experimental") → muddy mean prototype. Guarded by the
  cohesion check + warn; medoid / kNN-density prototype is a v2 refinement.
- **Over-pushing `sonic_blend`** trades transition quality for tag purity → kept soft by default
  (option-2 lean); the break-glass worst-edge repair remains the floor.
- **Label provenance:** prototype membership inherits the genre authority; pull labels
  authority-consistently.

## Validation outcome (2026-07-08) — supersedes parts of the design above

Implemented and validated end-to-end on Brian Eno (bimodal) + Real Estate (genre-blended). Two design
points changed under evidence:

1. **The pool prototype must be CENTERED, not uncentered.** The design proposed blending an uncentered
   prototype into the seed sonic vectors. That pulled genre-blended artists toward the *generic-genre
   centroid*: Real Estate/jangle got worse (lean −0.048 vs off; worst-edge 0.463→0.315). Switching the
   pool lever to blend the **centered** (tag-specific) affinity into the sonic *admission similarity*
   fixed it (lean −0.048→**+0.020**; worst-edge 0.463→**+0.716**) while keeping Eno's win (lean +0.099,
   worst-edge 0.528→0.661). The uncentered path was removed.
2. **The beam term ships OFF by default.** A scope-up (Task 6) added a beam ranking term, but a weight
   sweep (0.0/0.15/0.5/1.0) proved it cannot raise the on-tag lean at any weight — the beam only
   reorders within the pool it is handed, so the **pool is the binding lever**. `tag_steering_sonic_beam_weight`
   defaults to 0.0; the term stays wired/tested/documented (`docs/BEAM_CONTRACT.md`) for opt-in use.

Net shipped: pier sonic term + **centered** sonic pool lever (the lever), beam term opt-in-off. Distinct
genres (ambient) lean strongly and improve worst-edge; sonically-adjacent tags (jangle≈dream-pop) lean
modestly but honestly with the worst edge improved.

## Out of scope (future stages)

- A sonic-prototype term **in the beam's edge scoring** (the "small beam term" option) — deferred;
  revisit only if pool+pier still reads too weak by ear.
- Precomputed per-genre sonic-prototype sidecar (perf) — on-demand + in-run cache is fine at this
  library size (mean of a few thousand rows).
- Robust prototype estimators (medoid, trimmed mean, kNN density).
