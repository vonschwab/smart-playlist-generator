# Tag-aware pier allocation (artist mode) — design spec

**Date:** 2026-07-03
**Status:** approved design, pre-plan
**Feature group:** genre-tag steering ([[project_tag_steering]]) — Stage 1.5 (pier lever, cluster level)

## Problem

Tag steering (Stage 1) leans the candidate *pool* toward the selected tags, but the
artist-mode *anchors* (piers) don't follow. Evidence: the 2026-07-03 Real Estate run
(`logs/playlists/2026-07-03_000601_Real_Estate_82f228.log`) with tags `jangle pop` +
`indie rock` — the pool genre-cohesion rose (G mean 0.790), but **4 of 10 piers scored
~0.385 tag affinity** (`Tag steering piers: affinity per selected pier = [0.767, 0.607,
0.767, 0.767, 0.387, 0.665, 0.665, 0.385, 0.385, 0.385]`). The worst edge (T=0.264) landed
into Real Estate's "Sting" — a 1:49 off-jangle interlude planted as a pier.

**Root cause (verified in code):** artist-mode allocates `target_pier_count` slots
*evenly* across sonic clusters — `medoid_top_k = ceil(target_pier_count / predicted_k)`
per cluster (`playlist_generator.py:1803`), then flat-truncates the sonic-ordered list to
`target_pier_count` (`:1900-1909`). Every cluster — including the artist's off-tag
"interludes" cluster — contributes ~`medoid_top_k` piers regardless of tag affinity. The
within-cluster tag lever (Task 5, `_medoids_for_cluster`'s `tag_weight`/`tag_affinity`
term) only re-ranks *inside* a cluster; when a cluster has ≤ `medoid_top_k` members, all of
them become piers and the lever filters nothing. So an off-tag *cluster* yields off-tag
anchors that no existing lever can suppress.

## Decision (locked with Dylan, 2026-07-03)

**Soft skew, not concentrate or guard-only.** Anchors still span the artist's sonic range
(the arc must hold), but pier slots skew *across clusters* toward on-tag clusters. Consistent
with the feature's "lean, never a lane, never-fail" philosophy. Approach chosen: **tag-weighted
cluster allocation** (over global tag-ranked selection or affinity-floor-drop+backfill) — it
reuses the existing clustering + `medoids_by_cluster` + the Task-5 within-cluster lever, and a
per-cluster floor keeps the arc intact.

## 1. Architecture

New pure helper in `src/playlist/artist_style.py`:

```python
def allocate_piers_by_tag_affinity(
    medoids_by_cluster: list[list[int]],   # per-cluster, tag-ranked (Task 5), bundle indices
    cluster_affinities: list[float],       # mean member cos-to-target per cluster (same order)
    target_pier_count: int,
    skew: float,                           # 0 = uniform (today), 1 = pure affinity weighting
) -> list[int]:                            # selected pier indices (unordered; caller orders)
```

Consumed in `create_playlist_for_artist` (`playlist_generator.py`), replacing the current
flatten→`order_clusters`→truncate block at `:1900-1909`. When `steering_target is None`
(no tags), the caller keeps the existing path unchanged (byte-identical). When steering is
active, the caller:
1. computes `cluster_affinities[c] = mean(X_genre_dense[members_c] @ steering_target)` from
   the already-returned `clusters` + `bundle.X_genre_dense`,
2. calls `allocate_piers_by_tag_affinity(...)`,
3. runs `order_clusters(selected, X_norm)` as today (the post-cap at `:1904-1909` becomes a
   no-op because the allocator already returns exactly `target_pier_count`).

`cluster_artist_tracks`'s contract is unchanged **except** it must over-produce medoids per
cluster when steering, so each cluster offers enough tag-ranked candidates for any
`n_c ≤ cluster_size`. Concretely: at the `medoid_top_k` computation
(`playlist_generator.py:1803`), when `steering_target is not None` set
`medoid_top_k = target_pier_count` (each cluster then returns up to `min(cluster_size,
target_pier_count)` medoids — `_medoids_for_cluster` already caps at cluster size). Non-steering
path keeps `medoid_top_k = ceil(target_pier_count / predicted_k)` exactly as today. Medoid
extraction is cheap; this only lengthens each `medoids_by_cluster[c]` on the steering path.

## 2. Allocation algorithm

Given clusters with sizes `s_c`, affinities `a_c`, and target `P`:

1. **Weight:** normalize affinities to `[0,1]` across non-empty clusters
   (`a_c' = (a_c − min_a) / (max_a − min_a)`, or all-equal → `0.5` when `max_a == min_a`).
   `W_c = (1 − skew)·(1/K) + skew·(a_c' / Σ a_c')` where `K` = non-empty cluster count. The
   blend is what makes it *soft*: `skew=0` ⇒ uniform; `skew=1` ⇒ pure affinity.
2. **Apportion** `P` slots proportional to `W_c` via **largest-remainder (Hamilton)**, with:
   - **floor 1** per non-empty cluster (preserves the arc),
   - **cap `s_c`** (can't take more medoids than the cluster has),
   - exact sum: `Σ n_c = min(P, Σ s_c)`.
   If `Σ s_c ≤ P` (few tracks), take all — every medoid becomes a pier (matches today).
   Distribute the floors first, then apportion the remaining `P − K` by `W_c`, re-flowing any
   slot that would exceed a cluster's cap to the next-highest-remainder cluster under cap.
3. **Select:** take the top `n_c` from each `medoids_by_cluster[c]` (already tag-ranked by the
   Task-5 within-cluster lever). Return the flattened selection.

Result for Real Estate (K=4, P=10): interlude cluster `a_c≈0.39` → floored to 1 (was 3);
jangle clusters `a_c≈0.77` absorb the freed 2 slots. Anchors skew on-tag; all four clusters
still represented → arc holds.

## 3. Inertness, arc, never-fail

- **No tags** → `steering_target is None` → caller keeps the existing even-allocation +
  truncate path. Byte-identical to today. The allocator is never called.
- **`skew = 0`** → `W_c` uniform → allocation equals today's even split (guard the equivalence
  in a unit test).
- **Arc preserved** by the floor-of-1 per non-empty cluster; **feasibility** by cap-at-size.
- **All-off-tag catalog** → affinities ~equal → `a_c'` ~equal → ≈uniform (no starvation).
- No hard exclusion, no new failure mode; a playlist can never fail on this.

## 4. Config & composition

- One knob: `playlists.ds_pipeline.pier_bridge.pier_tag_skew` (default **0.6**, live). Added
  to `config.yaml` (gitignored — edit, don't commit) and `config.example.yaml`. Read from
  `ds_cfg["pier_bridge"]` alongside `tag_steering_pier_weight`.
- **`popular_seeds: on`** — composes: popularity still weights the *within-cluster* medoid
  pick (`medoid_popularity_weight`); tag-skew weights *across* clusters. Both soft, they stack.
- **`popular_seeds: fire`** — **skip** tag allocation: fire overrides `medoids` wholesale with
  `select_popular_piers` (`playlist_generator.py:1884-1892`) before ordering. The allocator
  runs only on the cluster-medoid path.

## 5. Diagnostic logging (Layer 4)

When the allocator runs, log at INFO: `Tag steering pier allocation: skew=0.60 per-cluster
[(affinity, slots), ...] (was uniform=k×top_k)` so the reallocation is visible in the
per-playlist log. The existing `Tag steering piers: affinity per selected pier = [...]` line
already reports the outcome; expect its low values to rise after this lands.

## 6. Testing

- **Unit** (`tests/unit/test_pier_tag_allocation.py`): `skew=0` ≡ uniform split; `skew=1`
  starves the off-tag cluster to its floor of 1; exact-sum apportionment (`Σ n_c = P`); cap
  respected (never exceeds cluster size); floor respected (every non-empty cluster ≥1);
  fewer-tracks case (`Σ s_c ≤ P` → take all); all-equal-affinity → uniform.
- **Behavioral** (extend `tests/integration/test_tag_steering_behavioral.py` or a sibling):
  a well-represented artist ± tags — assert the mean/min pier tag-affinity rises with skew on
  vs off, and the pool/pier log lines fire. Read the allocation log line, not just a metric
  (playlist-testing skill).
- **Live acceptance** (Dylan): re-run Real Estate + `jangle pop`/`indie rock`, confirm the
  ~0.385 piers drop out and the worst-edge/`Tag steering piers` line improves; worker restart
  first.

## Out of scope

- Min-duration interlude pier guard (fiddly threshold vs. legit short songs; the skew already
  pulls the interlude cluster to 1 pier — revisit only if Sting-type fragments persist).
- Global tag-ranked pier selection (Approach 2) and affinity-floor-drop+backfill (Approach 3)
  — rejected in favor of cluster allocation.
- Any change to the pool lever, the within-cluster lever (Task 5), or non-artist modes.

## Key references (verified 2026-07-03)

- Even allocation: `playlist_generator.py:1793` (`target_pier_count`), `:1803`
  (`medoid_top_k = ceil(target/predicted_k)`), `:1870-1881` (`cluster_artist_tracks` call,
  already passes `steering_target`), `:1900-1909` (flatten/order/truncate — replace here),
  `:1957-1962` (`Tag steering piers` affinity log).
- Within-cluster lever + `medoids_by_cluster`: `artist_style.py:669-704` (medoid loop),
  `:684-702` (`_medoids_for_cluster` call with `tag_weight`/`tag_slice`).
- Fire override (skip): `playlist_generator.py:1884-1892`.
