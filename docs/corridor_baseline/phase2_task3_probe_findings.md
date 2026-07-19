# Phase 2 Task 3 — Support-estimator probe findings

**Branch:** `corridor-phase2` @ e760104. Probe scripts are scratchpad (not committed); numbers
below are their actual output against the live artifact
(`data/artifacts/beat3tower_32k/data_matrices_step1.npz`, `sonic_variant_override=muq`).

## Question

Task 3's brief specifies a "library-support estimate" for pier candidates: fraction of a
candidate's top-K (K=100) eligible-universe sonic neighbors clearing a mode-typical threshold,
reusing the pier-bridgeability veto's machinery. Before wiring anything in, does this estimator
actually separate the two known outlier-EP pier candidates — Parquet Courts "Into the garden"
(`f86a9b71d2e35855ffc91f0488297915`, a noise/ambient track off Sunbathing Animal, tagged in prior
memory as from a "Monastic Living noise EP") and Swirlies "His Life of Academic Freedom"
(`fa09a81cec0805f33649ad86ea0389f1`, a 2:07 noise miniature) — from normal pier candidates?

## Finding 1 (REFUTED as literally specified): whole-library top-K density does not separate PC's outlier

Computed, for every candidate track of 6 corpus artists (SADE, Bill Evans Trio, The Strokes,
Swirlies, Aaliyah, Alex G) + Parquet Courts (added for this known-outlier case), several
formulations of "library-wide neighbor density":

- Fraction of top-100 genre-gated library neighbors clearing a calibrated-T threshold (0.20/0.30/0.40).
  **Saturates near 1.0 for nearly everything** — calibrated T compresses raw cosine so hard that a
  raw-cosine corridor threshold (~0.40, the actual value logged for PC segment 4) calibrates to
  T≈0.11, far below any of the tested T floors. Useless discriminator.
- Mean / k-th-neighbor RAW cosine of top-K (K=15/30/50/100) genre-gated OR ungated library
  neighbors, normalized by the artist's own median: Swirlies' outlier separates cleanly (3-10th
  percentile across every K/metric combination tested), but **Parquet Courts' "Into the garden"
  does not** — it lands at the 25th-36th percentile of its artist's 78 candidates, never in a
  clear outlier tail, regardless of K or whether the genre gate is applied.
- Independent confirmation: the EXISTING `compute_pier_bridgeability` signal (k=10, genre-gated,
  calibrated T) already scores "Into the garden" at kth10_T=0.865 — comfortably above the 0.30
  bridgeability floor. It is **not** a library-wide sonic isolate. It has plenty of close sonic
  neighbors in the ~43k-track library — mostly by OTHER artists (e.g. Sonic Youth "Theresa's
  Sound-World", sim=0.73 to "Into the garden", confirmed in the Task-1 mechanism probe).

**Conclusion:** whole-library density measures "is this track bridgeable to the wider library,"
which "Into the garden" already passes. It does not measure what actually makes it a bad PIER
CHOICE for Parquet Courts specifically.

## Finding 2: within-artist-catalog KNN density separates BOTH cases cleanly

Reformulated as: mean cosine similarity to a candidate's K nearest tracks BY THE SAME ARTIST
(same-artist catalog only, not the wider library), normalized by the artist's own median. Tested
K=5/10/15/20:

| Artist | Track | K=5 pctile | K=10 pctile | K=15 pctile | K=20 pctile |
|---|---|---|---|---|---|
| Parquet Courts | Into the garden | 15.4 | 16.7 | 17.9 | 19.2 |
| Swirlies | His Life of Academic Freedom | 19.4 | 12.9 | 9.7 | 12.9 |

Both known outliers land in the bottom ~10-20th percentile of their artist's candidate pool,
**robust across every K tested** — a real, stable separation, unlike the whole-library metric.

**Independent corroboration from production logs:** the live Parquet Courts generation log
(`logs/playlists/2026-07-18_174422_Parquet_Courts_b06d13.log`) shows "Into the garden"'s raw
`_medoids_for_cluster` combined score (sim-to-centroid × 0.7 + duration-typicality × 0.3) was
**0.774 — the lowest of all 10 medoid candidates across all 5 of Parquet Courts' clusters** (the
other 9 candidates scored 0.90–1.91). Its own cluster's OTHER candidate (0.902) was also
depressed relative to the other 4 clusters, i.e. the whole cluster is comparatively diffuse — the
within-artist-catalog signal picks up exactly this.

**Why within-artist, not library-wide, is the right scope:** "Into the garden" is an
off-character track FOR PARQUET COURTS (an ambient/noise piece against an otherwise
punk/garage-rock catalog) but is NOT globally rare — similar-sounding tracks by other artists are
common in a large, stylistically diverse library. A pier-candidacy support signal needs to ask
"does this resemble the rest of THIS artist's catalog," not "does anything in the whole library
resemble this."

## Decision

Estimator shipped: `compute_within_artist_support` (`src/playlist/artist_style.py`) — mean cosine
to the candidate's `k` (default 10, `pier_support_k`) nearest SAME-ARTIST tracks, normalized by the
artist's own median. Applied as a continuous RANK DEMOTION (`pier_support_demotion_strength`,
default 1.0) in `_medoids_for_cluster`'s combined score — never a hard veto, never reduces pier
count (matches `project_mini_pier_v1_failed_validation`'s lesson: a promotion veto that depletes
downstream pools failed validation before; this is a same-slot re-rank, not a pool-depleting gate).

## Live-data caveat found while writing the integration tests (not a defect)

The exact historical Parquet Courts/Swirlies incidents (specific pier list + terminal seat) are
**no longer byte-reproducible via a fresh live generation** — the library has grown
(38,844→43,241+ rows per the Task-1 probe) and re-clustering with a grown library shifts k-means
cluster membership for both artists. A fresh generation today, even with the Task 3 fix fully
disabled, no longer selects "Into the garden" as a Parquet Courts pier under the shipped
`max_artist_fraction=0.125` (4 piers) — and at `max_artist_fraction=0.20` (6 piers, matching the
log), Swirlies' outlier's cluster now has exactly as many real members as medoid slots requested,
so it is selected regardless of demotion (nothing to rank against). Neither is a code regression —
it's expected library churn. The integration test suite instead pins:
1. The per-track support signal directly against live data (stable, no clustering-RNG dependency).
2. The demotion MECHANISM against Parquet Courts' *actual current* k-means cluster-0 membership
   (3 real tracks, real embeddings) — demonstrating the outlier still wins its slot without the
   fix and loses it with the fix, on 100%-real, presently-true data.

See `tests/integration/test_gui_fidelity_regressions.py` (`test_compute_within_artist_support_flags_known_outliers_live`,
`test_medoids_for_cluster_demotes_real_pc_outlier_when_competitive`) for the pinned assertions.
