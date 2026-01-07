# DJ Bridging MVP Summary

This summary captures the current DJ bridging MVP status, what is implemented, how to run and tune it, and what remains. It is grounded in the current stress runs and architecture docs, with defaults unchanged unless DJ bridging is explicitly enabled.

## 1) What DJ Bridging is

DJ bridging is an opt-in pier-bridge enhancement that keeps the generator smooth-first while adding genre-guided, waypoint-aware steering between multiple anchors (“piers”). It aligns a genre route with the existing progress-arc “clock,” so playlists arc from pier A to pier B with minimal teleportation.

All DJ bridging behavior is gated behind `dj_bridging.enabled`. When off, production behavior is unchanged; candidate pooling, beam search, and scoring remain as they were.

## 2) Current implementation (with file pointers)

Phase 1 (waypoints, ordering, warnings)
- Waypoint guidance in beam search (bounded/tie-break influence) and progress-arc alignment in `src/playlist/pier_bridge_builder.py` (`_beam_search_segment(...)`).
- Seed ordering auto vs fixed via `_order_seeds_by_bridgeability(...)` in `src/playlist/pier_bridge_builder.py`, with config plumbed through `src/playlist/pipeline.py`.
- Warnings for missing/weak genre data and fallback behavior are wired through the pier-bridge result and run-audit pipeline (`src/playlist/pier_bridge_builder.py`, `src/playlist/run_audit.py`).

Phase 2A (dj_union pooling)
- DJ-only union pooling implemented in `src/playlist/segment_pool_builder.py` (`SegmentCandidatePoolBuilder.build(...)`), gated by `dj_bridging.pooling.strategy: dj_union`.
- Pool sources: local sonic neighbors, toward-B arc targets, genre target candidates.
- Per-source caps + union cap + caching in `src/playlist/segment_pool_builder.py`.
- Overlap + chosen-source diagnostics written into run audits via `src/playlist/pier_bridge_builder.py` and rendered in `src/playlist/run_audit.py`.
- Stress runners: `scripts/run_dj_union_pooling_stress_ab.py` (charli + indie scenarios).

Phase 2B (connector bias)
- Connector bias injection is implemented in `src/playlist/pier_bridge_builder.py` and gated via DJ config in `src/playlist/pipeline.py`.
- Far detection uses `_segment_far_stats(...)` in `src/playlist/pier_bridge_builder.py` (sonic/genre distance + connector scarcity).
- Diagnostics for connector injection/selection are added in `src/playlist/pier_bridge_builder.py` and show up in run audits.
- Connector A/B runner: `scripts/run_dj_connector_bias_ab.py`.

## 3) Config reference

Key DJ config knobs (from `config.example.yaml` and `src/playlist/pier_bridge_builder.py`):
- `playlists.ds_pipeline.pier_bridge.dj_bridging.enabled`: master gate.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering`: `auto` or `fixed`.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.anchors.must_include_all`: require all anchors.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.route_shape`: `linear` / `arc` / `ladder`.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.waypoint_weight`: strength of waypoint guidance.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.waypoint_floor`: minimum similarity required to count as a waypoint fit.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.waypoint_penalty`: penalty when below waypoint floor.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.waypoint_tie_break_band`: only apply waypoint influence within epsilon of best transition.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.waypoint_cap`: cap on waypoint contribution.

DJ union pooling
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy`: `baseline` or `dj_union`.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_local`: per-step local sonic cap.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_toward`: per-step toward-B cap.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre`: per-step genre target cap.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_union_max`: max union size.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.step_stride`: stride for step-targets.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.cache_enabled`: enable per-segment cache.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.debug_compare_baseline`: computes overlap stats (diagnostic only).

Connector bias (alias config)
- `playlists.ds_pipeline.pier_bridge.dj_bridging.connectors.enabled`: enable connector bias.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.connectors.max_connectors`: cap (applies to linear and adventurous).
- `playlists.ds_pipeline.pier_bridge.dj_bridging.connectors.use_only_when_far`: only inject when endpoints are far.
- `playlists.ds_pipeline.pier_bridge.dj_bridging.connectors.far_threshold`: set sonic/genre far threshold together.
- Optional override: `playlists.ds_pipeline.pier_bridge.dj_bridging.connectors.far_thresholds` with `sonic`, `genre`, `connector_scarcity`.

Starter presets (values to tune from)
- Smooth / cohesive:
  - `dj_bridging.enabled: true`
  - `waypoint_weight: 0.05`, `waypoint_tie_break_band: 0.02`, `waypoint_cap: 0.03`
  - `pooling.strategy: baseline` or `dj_union` with low `k_genre` (e.g., 20)
  - `connectors.enabled: false`
- Balanced:
  - `waypoint_weight: 0.15`, `waypoint_tie_break_band: 0.02`, `waypoint_cap: 0.05`
  - `pooling.strategy: dj_union`, `k_local: 200`, `k_toward: 80`, `k_genre: 80`, `k_union_max: 900`
  - `connectors.enabled: true`, `max_connectors: 2`, `use_only_when_far: true`
- Adventurous:
  - `waypoint_weight: 0.30`, `waypoint_tie_break_band: 0.01`, `waypoint_cap: 0.07`
  - `pooling.strategy: dj_union`, `k_genre: 120`, `k_union_max: 1200`
  - `connectors.enabled: true`, `max_connectors: 3`, `use_only_when_far: true`

UI mapping proposal (3 sliders)
- Smoothness:
  - `progress_arc.weight`, `progress_arc.max_step`, `transition_floor` (primary)
- Genre Drift:
  - `dj_bridging.waypoint_weight`, `waypoint_tie_break_band`, `waypoint_cap`, `pooling.k_genre`
- Effort / Quality:
  - `initial_beam_width`, `segment_pool_max`, `pooling.k_union_max`, `pooling.step_stride`

## 4) Empirical results (stress runs)

From `docs/diagnostics/dj_union_pooling_stress_ab.md` (charli scenario, stress `segment_pool_max=80`):
- DJ union pooling yields different tracklist hashes vs baseline under stress.
- Non-zero chosen-source counts (e.g., local/toward/genre) confirm union sources influence selection.
- Smoothness and pacing are comparable; union tends to slightly improve bridge minima and arc deviations while runtime stays similar.

From `docs/diagnostics/dj_union_pooling_stress_ab_indie.md` (indie ladder scenario, stress `segment_pool_max=80`):
- DJ union pooling again yields divergent hashes and non-zero chosen_from_toward/local/genre counts.
- Genre targets are hit modestly; toward-B dominates, consistent with smooth-first behavior.

From `docs/diagnostics/dj_connector_bias_ab_indie.md`:
- Connectors are injected (count ~3 per segment) and sometimes chosen (median 1 per segment), but hashes and metrics are unchanged in this run.
- Interpretation: connectors are already competitive in the candidate pool, so the bias acts as a safety / infeasible-handling assist rather than an ordering switch (at least in this scenario).

Concise result table (charli stress)
| label | pooling | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_min | p90_arc_dev | runtime_s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| A_baseline_low | baseline | 0.05 | 3d861181db1f | 0.5653 | 0.3505 | 1.49 |
| B_union_low | dj_union | 0.05 | 08a2358e55d3 | 0.6053 | 0.3267 | 1.68 |
| B_union_high | dj_union | 0.30 | 5b37fba2d29f | 0.5947 | 0.3267 | 1.70 |

Concise result table (indie stress)
| label | pooling | waypoint_weight | tracklist_hash | bridge_raw_sonic_sim_min | p90_arc_dev | runtime_s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| A_baseline_low | baseline | 0.05 | a4b3efb0113f | 0.5387 | 0.3359 | 2.74 |
| B_union_low | dj_union | 0.05 | 080bd88c69ea | 0.5994 | 0.3254 | 2.74 |
| B_union_high | dj_union | 0.30 | 952fe8da2bca | 0.5387 | 0.3359 | 2.82 |

## 5) Repro instructions

Run stress A/B for union pooling (both scenarios):
- `python scripts\run_dj_union_pooling_stress_ab.py`
  - Outputs:
    - `docs/diagnostics/dj_union_pooling_stress_ab.md`
    - `docs/diagnostics/dj_union_pooling_stress_ab_indie.md`

Run connector bias A/B (indie scenario):
- `python scripts\run_dj_connector_bias_ab.py`
  - Output:
    - `docs/diagnostics/dj_connector_bias_ab_indie.md`

To replicate stress conditions in a config run:
- Set `playlists.ds_pipeline.pier_bridge.segment_pool_max: 80`
- Set `playlists.ds_pipeline.pier_bridge.max_segment_pool_max: 80`
- Enable DJ bridging and set pooling strategy to `dj_union` (or baseline for A/B)

## 6) Known limitations / current gaps

- Genre candidates are selected less frequently than toward-B; this matches smooth-first + tie-break gating and is expected unless Genre Drift is pushed higher.
- Connector bias has not yet produced hash-level changes in the indie scenario; it seems to serve as a safety net rather than a reordering driver in that test.
- Full “genre ladder route planning” (graph shortest path / labeled waypoints) is not implemented yet. Linear drift via g-target interpolation is the current behavior.
- Micro-pier fallback exists in `src/playlist/pier_bridge_builder.py` (`_attempt_micro_pier_split(...)`) but is gated and currently off by default.

## 7) Next steps (prioritized)

1) Decide whether connector bias should become an infeasible-handling ladder step (recommended as a safe fallback).
2) Add the genre ladder route planner only if linear drift + union + connectors still fails to produce the desired “genre travel.”
3) UI work: add the three sliders (Smoothness / Genre Drift / Effort), plus warnings surfacing in the GUI.

## 8) Minor hygiene (optional)

Deprecation warnings observed during tests:
- `scripts/scan_library.py`: uses deprecated `src/genre_normalization` (replace with `src.genre.normalize_unified`).
- `src/genre/__init__.py`: deprecated `src.genre.normalize` import.
- `src/playlist_generator.py`: deprecated `artist_utils` import (use `src.string_utils.normalize_artist_name`).

References
- `docs/diagnostics/dj_union_pooling_stress_ab.md`
- `docs/diagnostics/dj_union_pooling_stress_ab_indie.md`
- `docs/diagnostics/dj_connector_bias_ab_indie.md`
- `docs/diagnostics/dj_genre_bridge_plan.md`
- `docs/diagnostics/dj_genre_bridge_architecture_discovery.md`
