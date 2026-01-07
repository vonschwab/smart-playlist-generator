# DJ Genre Bridge Plan (Implementation-Ready Reference)

========================================================
1) Executive summary (what we are building)
========================================================

DJ bridging is an opt-in, pier-bridge enhancement that adds a genre-aware route plan and waypoint-following execution to improve long-range coherence without sacrificing smooth transitions. A "genre ladder / waypoints" plan means we precompute a per-segment sequence of target genre vectors (derived from `X_genre_smoothed`) and then score candidates against those targets during beam search, using the existing progress-arc "clock" to align position in the route. Goals: default behavior unchanged unless explicitly enabled; smooth/cohesive transitions by default; allow user-controlled drift toward more adventurous, looser chains; use artifact genre vectors as primary signals while acknowledging noise; provide progressive relaxation and clear warnings when genre data is missing or weak. Non-goals: replacing existing pier-bridge scoring, changing default candidate pools, or adding new per-artist repetition logic (1-per-artist-per-segment stays as-is).

========================================================
2) Terminology + mental model
========================================================

- Anchors / piers / seeds: the fixed seed tracks that define bridge endpoints. In the DS pipeline, these are passed as `anchor_seed_ids` and become piers in the pier-bridge system (`src/playlist/pier_bridge_builder.py`).
- Route planning: precompute the order of anchors and per-segment genre targets before segment construction.
- Route execution: use the computed targets inside beam search to bias candidate selection.
- Route shape vs chain tension:
  - Route shape = how target genre evolves across steps (linear drift or arc).
  - Chain tension = how strongly the search is pulled toward that target (waypoint weight, thresholds).
- Progress-arc as shared clock: the existing `progress_t` projection (AB direction in sonic space) in `_beam_search_segment(...)` provides step-aligned timing for genre targets, so we can use the same `(step_idx, steps)` to compute both sonic progress and genre drift.

========================================================
3) Current-state recap (grounded in discovery doc)
========================================================

- Genre normalization and similarity:
  - Normalization: `src/genre/normalize_unified.py` (canonical).
  - Similarity YAML: `data/genre_similarity.yaml`, loaded by `src/genre_similarity_v2.py` and `src/genre/similarity.py`.
  - Genre vectors: `X_genre_raw` / `X_genre_smoothed` from artifacts built in `src/analyze/artifact_builder.py`.
- Pier-bridge flow:
  - Entrypoint: `src/playlist/pipeline.py` -> `build_pier_bridge_playlist(...)` in `src/playlist/pier_bridge_builder.py`.
  - Auto-ordering: `_order_seeds_by_bridgeability(...)` inside `build_pier_bridge_playlist(...)`.
  - Execution scoring: `_beam_search_segment(...)` in `src/playlist/pier_bridge_builder.py`.
  - Candidate pooling: `SegmentCandidatePoolBuilder.build(...)` in `src/playlist/segment_pool_builder.py` (with a legacy union helper in `pier_bridge_builder.py`).

========================================================
4) Proposed architecture (high level)
========================================================

Add a Route Planner layer (opt-in, before `build_pier_bridge_playlist(...)`):
  A) Anchor ordering: auto-order by default, allow lock to preserve input order.
  B) Per-segment genre route plan: a list of target genre vectors per step or a compact representation (gA/gB + shape).

Add Waypoint-Aware Execution (opt-in inside `_beam_search_segment(...)`):
  - Add per-step genre target scoring term: `g_target(step)`.
  - Optional connector bias when endpoints are far or adventurous mode enabled.

Add Progressive Relaxation + warnings:
  - Extend existing `InfeasibleHandlingConfig` (already wired) to add waypoint relax steps.
  - Emit structured warnings for missing/weak genre data and relaxed constraints.

Architecture diagram (text):

GUI/CLI config
  -> Route Planner (new, opt-in)
     - anchor order (auto|fixed)
     - per-segment genre targets
  -> build_pier_bridge_playlist(...) `src/playlist/pier_bridge_builder.py`
     -> segment pool builder `src/playlist/segment_pool_builder.py`
     -> beam search `_beam_search_segment(...)`
        - transition scoring
        - progress-arc t
        - genre waypoint scoring (DJ bridging only)
     -> run audit + warnings

========================================================
5) Algorithms / data science design (detailed)
========================================================

A) Anchor auto-ordering

Goal: "auto-order unless locked" while keeping default unchanged.

- Implementation: add a config flag `seed_ordering: auto|fixed` and bypass `_order_seeds_by_bridgeability(...)` when `fixed`.
  - Hook: `src/playlist/pier_bridge_builder.py` inside `build_pier_bridge_playlist(...)` where `ordered_seeds = _order_seeds_by_bridgeability(...)` is called.
  - Keep default `auto` so behavior is unchanged.
- Planning/execution alignment rule:
  - Planning must reuse the same normalization/centering/weighting as pier-bridge transition scoring (or explicitly document any divergence and why).
  - Canonical planning space: the transition scoring space used by `_compute_transition_score_extracted(...)` (same `X_full_norm` / `X_start_norm` / `X_end_norm` and `center_transitions` behavior).

Ordering score (deterministic, fast):
- Use a weighted sum of:
  - Sonic similarity: dot product between `X_full_norm[a]` and `X_full_norm[b]`.
  - Genre similarity: dot product between `X_genre_norm[a]` and `X_genre_norm[b]` (if available).
  - Feasibility proxy: bridgeability score (already implemented) or `min(sim_a, sim_b)` against pool stats.

Suggested scoring:
```
score(a, b) =
  w_sonic * sim_sonic(a, b)
  + w_genre * sim_genre(a, b)
  + w_bridge * compute_bridgeability_score(a, b)
```
Defaults: `w_sonic=0.6`, `w_genre=0.2`, `w_bridge=0.2` (tunable).

Ordering policy:
- If `n <= 8`: brute-force permutations (existing `SeedOrderingConfig.max_exhaustive_search` defaults to 6; we can lift when DJ enabled).
- Else: greedy nearest-neighbor followed by 2-opt refinement.

Pseudocode (greedy + 2-opt):
```
order = [seed0]
while remaining:
  pick next = argmax score(order[-1], candidate)
  append next

for _ in range(k_passes):
  for i < j:
    if swap improves total score: swap segment
```
Use the same `TransitionScorer` or direct dot products for determinism.

B) Genre route planning (ladder)

Data sources (existing):
- `X_genre_smoothed` and `genre_vocab` from artifacts (preferred).
- `data/genre_similarity.yaml` (manual similarity).
- Optional `genre_similarity.npz` (S matrix) built by `src/analyze/genre_similarity.py`.

Route representation:
- Preferred: list of target vectors `g_targets[step]` in the same basis as `X_genre_smoothed`.
- Compact: `(gA, gB, shape, steps)` and compute `g_target(step)` on demand.

Endpoint genre vectors:
- Primary: `gA = X_genre_smoothed[pier_a]`, `gB = X_genre_smoothed[pier_b]`.
- Fallback (missing genres):
  - If `gA` or `gB` is all-zeros, estimate from local neighborhood in genre space:
    - For pier index `i`, compute top-K neighbors in `X_sonic` or `X_full_norm`.
    - Average their `X_genre_smoothed` rows (normalized).
  - If still missing: warn + reduce waypoint weight to near-zero for that segment.

Genre ladder label-to-vector bridge (for waypoint ladder):
- Select top-N endpoint labels from `gA` and `gB` (highest weights in `genre_vocab`).
- Find a shortest path between labels using:
  - YAML similarity edges (`data/genre_similarity.yaml`) or
  - NPZ similarity matrix (`genre_similarity.npz` S matrix).
- Convert intermediate labels to waypoint vectors in `genre_vocab` space:
  - One-hot in `genre_vocab`, or
  - Similarity-smoothed vector using the S matrix row for that label.
- Fallback when labels do not map cleanly:
  - Use the nearest known label in `genre_vocab`, or
  - Skip ladder and fall back to linear drift with a warning.

Route shapes:
- Linear drift (default): `g_target(step) = normalize(lerp(gA, gB, frac))`.
- Adventurous arc:
  - Option 1: same endpoints but non-linear `frac` (cosine arc):
    - `frac = 0.5 - 0.5*cos(pi * (step+1)/(steps+1))`.
  - Option 2 (waypoint ladder):
    - Build a path in genre graph (YAML or S matrix) between endpoint genres.
    - Interpolate between consecutive waypoints.

Step fraction convention (authoritative):
- `frac = (step_index + 1) / (num_steps + 1)`
- This same `frac` drives both progress-arc target and `g_target(step)` to avoid off-by-one drift.

Far endpoints criteria (for allowing detours/connectors):
- Compute:
  - `sonic_dist = 1 - sim_sonic(a, b)`
  - `genre_dist = 1 - sim_genre(a, b)` (if available)
  - `connector_scarcity` = fraction of pool candidates with `min(sim_a, sim_b) >= bridge_floor`.
- Treat as "far" if:
  - `sonic_dist > sonic_far_threshold` OR
  - `genre_dist > genre_far_threshold` OR
  - `connector_scarcity < scarcity_threshold`.

C) Execution: waypoint-aware scoring + pooling

Integration points:
- Scoring: `src/playlist/pier_bridge_builder.py` -> `_beam_search_segment(...)`.
- Pooling: `src/playlist/segment_pool_builder.py` -> `SegmentCandidatePoolBuilder.build(...)`.

Scoring term (bounded / tie-break gated):
- Tie-break band option (default for smooth-first):
  - Apply waypoint influence only when candidates are within `epsilon` of the best primary score.
  - This keeps waypoint guidance secondary unless the user maxes Genre Drift.
- Capped contribution option:
  - `delta = clamp(w_waypoint * genre_target_sim, -cap, +cap)`
  - `combined_score += delta`
- In both cases, waypoint influence is bounded so transition quality stays dominant.
- "Smooth-first" guarantee: waypoint influence must never dominate transition scoring unless the user explicitly maxes Genre Drift.
- Only active when DJ bridging is enabled and `X_genre_norm` is available.

Candidate pool strategy (opt-in):
- Default: unchanged `segment_scored` pool (safest).
- DJ strategy (union of sources, still 1-per-artist enforced by pool builder):
  1) Baseline segment pool scored against both piers (existing).
  2) "Toward B" pool: candidates near the arc target in sonic space.
  3) "Genre waypoint" pool: top-K by dot with `g_target(step)` (or waypoint vectors).
- Add a config key to switch pool strategy, e.g., `segment_pool_strategy: dj_union`.
- Implementation: extend `SegmentCandidatePoolBuilder.build(...)` to accept optional extra candidate indices and merge before filtering.
- Union pooling caps and caching:
  - Per-source K caps (local sonic, toward-B sonic, waypoint genre).
  - Max union size after dedupe.
  - Cache per-step or per-segment `topK_by_genre_target` to avoid recompute inside beam expansion loops.
  - Effort slider governs these caps to control runtime.

D) Connector track logic (minimal, reuse existing hooks)

- Existing connectors:
  - `SegmentCandidatePoolBuilder` already computes `bridge_sim` and can accept `internal_connectors`.
  - `internal_connectors` are already supported end-to-end (pipeline -> segment pool).
- DJ bridging rule:
  - In linear mode, only inject connectors when endpoints are far (as defined above).
  - In adventurous mode, allow more connectors by increasing `internal_connector_cap`.

E) Progressive relaxation & warnings

Use existing infeasible handling (config already wired):
- `InfeasibleHandlingConfig` in `src/playlist/run_audit.py` and usage in `build_pier_bridge_playlist(...)`.
- Add waypoint-specific relaxation steps without changing defaults:
  - Step 1: relax genre waypoint strictness (reduce `w_waypoint`, widen tolerance, disable waypoint penalty).
  - Step 2: increase effort (beam width, pool sizes).
  - Step 3: relax transition floor slightly (last resort, tiny deltas, gated).
  - Relaxing transition floor must emit a warning.

Warnings to emit:
- Missing/weak genre vectors for anchors or a high % of candidates.
- Fallback used (neighbor avg, K, or sonic-only) and any auto-reduction in Genre Drift.
- Waypoint relaxations applied (include which step).
- Fallback to sonic-only for segment.

Warning plumbing:
- Add structured warnings to result payload in `src/playlist/pipeline.py` or `src/playlist_generator.py`.
- Surface in GUI via `src/playlist_gui/worker.py` (emit as `type="log"` warning or attach to playlist result).

========================================================
6) Config + feature flags (explicit keys)
========================================================

Additions under `playlists.ds_pipeline.pier_bridge` (defaults preserve current behavior):

```
dj_bridging:
  enabled: false
  seed_ordering: auto            # auto | fixed
  anchors:
    must_include_all: true       # lock membership / must include all anchors
  route_shape: linear            # linear | arc | ladder
  waypoint_weight: 0.15
  waypoint_floor: 0.20
  waypoint_penalty: 0.10
  allow_detours_when_far: true
  far_thresholds:
    sonic: 0.45
    genre: 0.60
    connector_scarcity: 0.10
  pool_strategy: segment_scored  # segment_scored | dj_union
  connector_bias:
    enabled: true
    max_per_segment_linear: 1
    max_per_segment_adventurous: 3
```

Slider mapping (UI):
- Mix Smoothness:
  - `progress_arc.weight`, `progress_arc.max_step`, `transition_floor`.
- Genre Drift:
  - `dj_bridging.waypoint_weight`, `dj_bridging.waypoint_floor`, `dj_bridging.route_shape`.
- Effort/Quality:
  - `initial_beam_width`, `max_beam_width`, `segment_pool_max`, `initial_neighbors_m`.

Defaults:
- `dj_bridging.enabled: false` (no behavior change).
- `seed_ordering: auto` matches current behavior.
- If `anchors.must_include_all` is true and infeasible, return a warning and recommend increasing Effort, reducing Genre Drift, or adding genre metadata.

========================================================
7) GUI plan (minimal but complete)
========================================================

Where to add controls:
- `src/playlist_gui/config/settings_schema.py`:
  - Add new SettingSpec entries for DJ bridging toggles/sliders.
- Config wiring:
  - GUI overrides flow to `src/playlist_gui/worker.py` -> `load_config_with_overrides(...)`.
  - The DS pipeline reads `playlists.ds_pipeline.pier_bridge` in `src/playlist/pipeline.py`.
- Anchor IDs must be stable `track_id` values (not index-like "seed numbers"). Index resolution is testing-only and must not leak into UX.

Controls:
- Toggle: "DJ Bridging Enabled" (maps to `dj_bridging.enabled`).
- Toggle: "Lock Anchor Order" (maps to `dj_bridging.seed_ordering: fixed`).
- Toggle: "Must Include All Anchors" (maps to `dj_bridging.anchors.must_include_all`).
- Sliders: Mix Smoothness, Genre Drift, Effort/Quality (map to keys above).

Warnings:
- Use playlist result payload `warnings` list.
- GUI: show warnings in the log panel or a warning banner (use existing warning display in `main_window.py`).
- If `must_include_all` is true and infeasible, return a warning with recommendations (increase Effort, reduce Genre Drift, add genre metadata).
- Recommended UI message when genre data is weak: "Genre guidance reduced because metadata is missing; consider adding genres."

========================================================
8) Testing & evaluation plan (measurable)
========================================================

Extend sweep harness (`scripts/sweep_pier_bridge_dials.py`):
- Scenarios:
  - Dance neighborhood: Justice / Charli / Dua / Frankie.
  - Indie ladder: Destroyer / Wilco / Early Day Miners / North Americans / MJ Lenderman / Sufjan / Elliott.
- Ensure anchors are passed as stable `track_id` values; any index-based resolution is test-only and should be explicitly labeled as such.

Post-hoc metrics (additions):
- Genre path adherence:
  - mean/p50/p90 of `cos(X_genre_norm[track], g_target(step))`.
- Genre travel smoothness:
  - mean absolute delta of consecutive genre similarities to target.
- Connector usage rate:
  - fraction of selected tracks coming from connector-biased sets.

Success criteria:
- DJ mode improves `bridge_raw_sonic_sim_min` while reducing `p90_arc_dev` and `max_jump` vs baseline.
- "Genre Drift" presets produce observable separation in genre path adherence and tracklist hashes.
- Performance bounded by Effort slider (runtime stays within configured limits).
- Balanced success should include improved/maintained `bridge_raw_sonic_sim_min`, reduced `p90_arc_dev`, bounded `max_jump`, and improved genre adherence over time (distance to `g_target(step)`).

========================================================
9) Phased implementation roadmap (MVP -> v2 -> v3)
========================================================

Phase 1 (MVP)
- Add `dj_bridging.enabled` + `seed_ordering` (auto|fixed).
- Linear drift waypoint scoring term in `_beam_search_segment(...)`.
- Add warnings for missing genre data and waypoint relaxations.
- Extend sweep harness with genre adherence metrics.

Phase 2
- Genre ladder route planning (YAML/NPZ graph).
- DJ union candidate pool strategy.
- Connector bias with far-endpoint criteria.
- Micro-pier fallback: when endpoints are far or connector scarcity is high, insert 1-2 connector tracks as intermediate mini-anchors and run pier-bridge per hop (gated by Genre Drift/adventure).

Phase 3
- Refined UI presets and slider mapping.
- Progressive relaxation tuning ladder.
- Expanded multi-anchor scenarios and regression tests.

========================================================
10) File touch list (exact files)
========================================================

Phase 1 (MVP)
- `src/playlist/pier_bridge_builder.py` (seed_ordering flag, waypoint scoring, warnings)
- `src/playlist/pipeline.py` (plumb config into PierBridgeConfig)
- `src/playlist/scoring/constraints.py` (if adding new config dataclass fields)
- `scripts/sweep_pier_bridge_dials.py` (new genre metrics)
- `tests/unit/test_progress_arc.py` (extend for waypoint curve math)
- `src/playlist_gui/config/settings_schema.py` (add DJ controls)
- `src/playlist_gui/worker.py` (surface warnings)

Phase 2
- New module: `src/playlist/genre_route_planner.py` (or `src/genre/route_planner.py`)
- `src/analyze/genre_similarity.py` (optional adjacency precompute)
- `src/playlist/segment_pool_builder.py` (dj_union pool strategy)

Phase 3
- `docs/diagnostics/pier_bridge_dial_sweep_*.md` (new scenario docs)
- `tests/unit/test_segment_pool_builder.py`
- `src/playlist_gui/main_window.py` (warning banner polish)

Missing primitives and smallest additions

- Missing: runtime genre adjacency graph.
  - Smallest addition: a new cached helper that loads `data/genre_similarity.yaml` or `genre_similarity.npz` and returns adjacency lists on demand.
- Missing: per-run structured warnings in GUI payload.
  - Smallest addition: add `warnings` list in `run_ds_pipeline(...)` return payload and pass through worker response.
