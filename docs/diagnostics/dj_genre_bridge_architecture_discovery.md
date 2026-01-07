# DJ Genre Bridge Architecture Discovery

This document answers the requested architecture questions with file-specific references and concrete hook points.

========================================================
0) GENRE DATA: WHAT WE ALREADY HAVE
========================================================

1) Where are these implemented and how do we access them at runtime?

- Normalized genres
  - Implementation: unified normalization in `src/genre/normalize_unified.py`.
    - Entry points: `normalize_and_split_genre()`, `normalize_and_filter_genres()`, `normalize_genre_list()`.
  - Legacy wrappers: `src/genre_normalization.py` and `src/genre/normalize.py` (see `src/feature_flags.py` flag `use_unified_genre_normalization`).
  - Runtime access:
    - DS artifact builder uses it when `normalize_genres=True`:
      - `src/analyze/artifact_builder.py` -> `build_ds_artifacts(...)` (uses `normalize_and_filter_genres()` if taxonomy available).
    - File-tag updates normalize before DB insert:
      - `scripts/update_file_genres.py` -> `normalize_genre_list()`.

- Genre similarity YAML
  - File: `data/genre_similarity.yaml`.
  - Load locations:
    - `src/genre_similarity_v2.py` -> `GenreSimilarityV2.__init__()` loads YAML into `self.similarity_matrix`.
    - `src/genre/similarity.py` -> `load_yaml_overrides(filepath)` loads YAML into module-level `_yaml_matrix` for structural similarity overrides.
  - Runtime access:
    - `src/similarity_calculator.py` uses `GenreSimilarityV2` (config key `playlists.genre_similarity.similarity_file` in `config.example.yaml`).
  - Caching:
    - YAML is held in memory in `GenreSimilarityV2.similarity_matrix`.
    - `src/genre/similarity.py` uses LRU cache for `pairwise_genre_similarity` and a module cache for YAML overrides.

- Track "effective genres" (with artist/album inheritance)
  - Storage: `metadata.db` table `track_effective_genres`.
    - Write paths:
      - `scripts/update_file_genres.py` inserts rows with columns `(track_id, genre, source, priority, weight, last_updated)` when table supports them.
      - `scripts/fix_compound_genres.py` reads/writes `track_effective_genres`.
    - Query paths:
      - `src/local_library_client.py` -> `get_tracks_for_genre()` uses UNION priority:
        - `track_effective_genres` (priority 1) -> `album_genres` (priority 2) -> `artist_genres` (priority 3).
      - `src/playlist_gui/autocomplete.py` loads distinct genres from `track_effective_genres`.
  - NOTE: DS pipeline does not directly read `track_effective_genres`; it uses `X_genre_raw` / `X_genre_smoothed` from artifacts.

2) What is the "effective genre" type per track?

- Type: list of normalized genre tokens (one per row in `track_effective_genres`).
- Weights/confidence:
  - `scripts/update_file_genres.py` inserts `weight=1.0` but read paths do not use weight (see `LocalLibraryClient.get_tracks_for_genre()` and GUI autocomplete).
  - No explicit confidence used at runtime in GUI or DS pipeline for effective genres.
- Unknown/no-genre representation:
  - For DB: no rows in `track_effective_genres`.
  - For runtime logic:
    - In `src/similarity_calculator.py`, `_get_combined_genres()` returns `[]` if no data; similarity falls back to sonic-only.
    - In DS artifacts, `X_genre_raw` row is all-zeros for tracks with no genres.

3) How do we currently use genre in scoring?

- Candidate pool gating (global pipeline):
  - `src/playlist/candidate_pool.py` -> `_compute_genre_similarity()` and `build_candidate_pool(...)`.
  - Inputs: `X_genre_raw` and/or `X_genre_smoothed` from artifacts.
  - Config keys:
    - `playlists.genre_similarity.min_genre_similarity`
    - `playlists.genre_similarity.method` (e.g., "ensemble")
    - `playlists.ds_pipeline.candidate_pool.broad_filters`
  - Symmetry: cosine and jaccard are symmetric; no directionality.

- Pier-bridge soft genre penalty + tiebreak:
  - `src/playlist/pier_bridge_builder.py` -> `_beam_search_segment(...)`.
  - Inputs: `X_genre_smoothed` (normalized to `X_genre_norm`).
  - Behavior:
    - Tiebreak bonus: `combined_score += cfg.genre_tiebreak_weight * genre_sim`.
    - Soft penalty: if `genre_sim < cfg.genre_penalty_threshold`, multiply score by `(1 - cfg.genre_penalty_strength)`.
    - Optional tie-break band: `cfg.genre_tie_break_band` applies penalty only for close-call candidates.
  - Config keys (from `config.example.yaml` and `PierBridgeConfig` fields):
    - `playlists.ds_pipeline.pier_bridge.genre_tiebreak_weight`
    - `playlists.ds_pipeline.pier_bridge.soft_genre_penalty_threshold`
    - `playlists.ds_pipeline.pier_bridge.soft_genre_penalty_strength`
    - `playlists.ds_pipeline.pier_bridge.genre.tie_break_band`
  - Symmetry: cosine similarity of genre vectors is symmetric; no directionality.

========================================================
1) PLANNING LAYER: MULTI-ANCHOR AUTO-ORDERING + GENRE LADDER
========================================================

4) Multi-anchor input:

- GUI -> worker: `src/playlist_gui/main_window.py` collects `seed_tracks` via `SeedTracksInput`; passed to worker in `args.seed_tracks`.
- Worker -> generator: `src/playlist_gui/worker.py` -> `handle_generate_playlist()` calls:
  - `generator.create_playlist_for_artist(... track_titles=seed_tracks ...)`
  - OR `generator.create_playlist_from_seed_tracks(seed_tracks, ...)`.
- Generator -> DS pipeline:
  - `src/playlist_generator.py` resolves `anchor_seed_ids` and passes to `run_ds_pipeline(...)` via `_maybe_generate_ds_playlist(...)`.
  - The DS pipeline entry point accepts `anchor_seed_ids`:
    - `src/playlist/pipeline.py` -> `generate_playlist_ds(... anchor_seed_ids=...)`.
- "Lock seed tracks" representation:
  - There is no explicit "lock order" flag in pier-bridge. Seeds are auto-ordered every run.
  - `anchor_seed_ids` currently represent fixed membership, not fixed order.

5) Auto-ordering hook point:

- Current auto-ordering is always on:
  - `src/playlist/pier_bridge_builder.py` -> `_order_seeds_by_bridgeability(...)` is called from `build_pier_bridge_playlist(...)` and reorders seeds.
  - This uses `_compute_bridgeability_score(...)` based on end->start and full similarities.
- Hook point for "auto-order unless locked":
  - Add a config flag or optional argument to bypass `_order_seeds_by_bridgeability(...)` and preserve input order.
  - Best carrier: extend `PierBridgeConfig` with something like `seed_ordering: "auto"|"fixed"` and pass from `src/playlist/pipeline.py`.

6) Pairwise "difficulty" features we can compute cheaply using existing infra:

- Sonic similarity between two track IDs:
  - Use L2-normalized sonic matrices from artifacts:
    - `src/features/artifacts.py` -> `get_sonic_matrix(bundle, segment)` returns `X_sonic_*`.
    - Dot products in `src/playlist/scoring/transition_scoring.py`.
- Genre similarity between two tracks:
  - `X_genre_smoothed` from artifacts (normalized in `build_pier_bridge_playlist(...)`) and dot product.
  - See `src/playlist/pier_bridge_builder.py` -> `_get_genre_sim(...)` inside `_beam_search_segment(...)`.
- Connector availability heuristic:
  - Existing "bridgeability" scoring (cheap) uses direct and full similarity:
    - `src/playlist/scoring/bridge_scoring.py` -> `compute_bridgeability_score(...)`.
  - Segment pool builder already computes `bridge_sim = min(sim_a, sim_b)`:
    - `src/playlist/segment_pool_builder.py` -> `SegmentCandidatePoolBuilder.build(...)`.
  - For a fast connector proxy, reuse `min(sim_a, sim_b)` or harmonic mean used in segment pool scoring.

7) Genre ladder / route planning:

- Genre adjacency graph availability:
  - Existing sources:
    - `data/genre_similarity.yaml` (manual similarity matrix).
    - `src/analyze/genre_similarity.py` builds `genre_similarity.npz` (S matrix) from co-occurrence.
  - There is no explicit "graph" object or cached adjacency list in runtime code.
- Where to add cached build step:
  - Best place: artifact build phase (same place as `X_genre_smoothed`):
    - `src/analyze/artifact_builder.py` already loads `genre_similarity.npz` if provided.
  - Alternative: runtime cache in a new module alongside `src/genre/similarity.py` or `src/genre_similarity_v2.py` that loads YAML and builds adjacency once.
- Best representation for a "route":
  - A list of target distributions (vectors in `genre_vocab`) is most compatible with `X_genre_smoothed` scoring.
  - A list of genre tokens (waypoints) is simpler, but requires mapping to target vectors for scoring.

========================================================
2) EXECUTION LAYER: WAYPOINT-AWARE BRIDGING (SMOOTH BY DEFAULT)
========================================================

8) In the pier-bridge segment builder, where are these implemented?

- Segment candidate pool construction:
  - `src/playlist/segment_pool_builder.py`:
    - `SegmentCandidatePoolBuilder.build(...)`
    - Config type `SegmentPoolConfig`
  - Legacy pool builder (union of neighbors + helpers):
    - `src/playlist/pier_bridge_builder.py` -> `_build_segment_candidate_pool_legacy(...)`.
- Per-step / beam expansion scoring:
  - `src/playlist/pier_bridge_builder.py` -> `_beam_search_segment(...)`.
  - Combined score uses:
    - transition score (`_compute_transition_score_extracted`)
    - bridge score term
    - destination pull
    - progress penalties / arc penalties
    - genre tiebreak / soft penalty
- Progress-arc projection (t) and arc tracking:
  - Target curve: `_progress_target_curve(step_idx, steps, shape)`
  - Loss: `_progress_arc_loss_value(err, loss, huber_delta)`
  - Metrics: `_compute_progress_tracking_metrics(...)`
  - Projection is computed via AB vector in `_beam_search_segment(...)`.
- Hook for genre waypoint targets:
  - `_beam_search_segment(...)` is the correct place to add a per-step `genre_target` scoring term.
  - Candidate pool mixing would be added before the call, in `build_pier_bridge_playlist(...)` or in `SegmentCandidatePoolBuilder`.

9) Candidate pooling:

- Current default pooling is per-segment, not per-step:
  - `segment_pool_strategy: segment_scored` in config.
  - `SegmentCandidatePoolBuilder.build(...)` scores against both piers.
- Union pool support exists in legacy path:
  - `_build_segment_candidate_pool_legacy(...)` unions neighbors of A, neighbors of B, and bridge helpers.
- Minimal-refactor hook for multi-source pools:
  - Extend `SegmentCandidatePoolBuilder.build(...)` to accept extra candidate lists (e.g., `extra_candidates`) and merge before filtering.
  - Alternatively add a new "strategy" in `build_pier_bridge_playlist(...)` to union:
    - local neighbors of current track
    - "toward B" candidates by progress target
    - genre waypoint candidates (by genre similarity to waypoint vector)

10) Linear drift default:

- A per-step target progress fraction already exists:
  - `_progress_target_curve(...)` in `src/playlist/pier_bridge_builder.py`.
  - Used by `_beam_search_segment(...)` for progress penalties/arc tracking.
- Genre target `g_target(step)`:
  - Not implemented.
  - Simplest insertion point: in `_beam_search_segment(...)` where `combined_score` is computed per candidate.
  - Use step index to compute a target genre vector (from waypoint plan) and score `cos(X_genre_norm[cand], g_target)`.

11) Connector tracks:

- Existing "connector" logic:
  - Internal connectors list can be injected into segment pools:
    - Pipeline passes `internal_connector_ids` -> `SegmentPoolConfig.internal_connectors`.
    - `SegmentCandidatePoolBuilder._process_internal_connectors(...)` prioritizes them.
  - Artist-style mode can compute internal connectors:
    - `src/playlist/artist_style.py` -> `get_internal_connectors(...)`.
- No general "best connector to both A and B" cache outside the pool scoring.
  - The closest hook is `SegmentCandidatePoolBuilder.build(...)` where `bridge_sim` is computed.
  - For a new connector heuristic, add a preselection step there and expose it as `internal_connectors`.

========================================================
3) PROGRESSIVE RELAXATION + USER-FACING WARNINGS
========================================================

12) What constraints currently cause failure/dead-ends?

- One track per artist per segment:
  - Enforced in `SegmentCandidatePoolBuilder._select_final_candidates(...)`.
- Transition floors (hard gate):
  - `PierBridgeConfig.transition_floor` and `ScoringConstraints.transition_floor`.
  - Enforced in `_beam_search_segment(...)` via `transition_floor` checks.
- Bridge floor gate:
  - `PierBridgeConfig.bridge_floor` in segment pool builder.
- Recency, blacklist, duration:
  - Applied in candidate pool / pipeline:
    - `src/playlist/candidate_pool.py` handles duration penalty and candidate exclusions.
    - `src/playlist/pipeline.py` applies `allowed_track_ids`, `excluded_track_ids`, recency exclusions.
- Failure reporting:
  - `_beam_search_segment(...)` returns `beam_failure_reason`.
  - `build_pier_bridge_playlist(...)` emits `RunAuditEvent` with `failure_reason` (see `src/playlist/run_audit.py`).

13) Progressive relaxation ladder:

- Existing hook (disabled by default):
  - `InfeasibleHandlingConfig` in `src/playlist/run_audit.py`.
  - `build_pier_bridge_playlist(...)` uses it to:
    - back off `bridge_floor`
    - widen neighbors/helpers/beam width (see config.example.yaml).
- Where to add the ladder:
  - Extend `build_pier_bridge_playlist(...)` where the backoff attempts happen.
  - Gate it behind a new "DJ bridging enabled" config flag (best place: `playlists.ds_pipeline.pier_bridge.infeasible_handling.enabled`).
- Suggested config keys (existing + minimal additions):
  - Existing:
    - `playlists.ds_pipeline.pier_bridge.infeasible_handling.*`
  - New (if needed):
    - `playlists.ds_pipeline.pier_bridge.dj_bridging.enabled` (soft gate).
    - `playlists.ds_pipeline.pier_bridge.dj_bridging.genre_waypoint_relax` (max genre distance).

14) Warnings/notes plumbing:

- GUI surface:
  - Worker emits `{"type":"log","level":"WARNING","msg":...}` via `src/playlist_gui/worker.py`.
  - GUI logs warnings and can display diagnostics banners (`src/playlist_gui/main_window.py`).
- Structured warnings from pipeline:
  - Run audit artifacts: `src/playlist/run_audit.py` writes markdown files to `docs/run_audits`.
  - `RunAuditEvent` can carry structured payloads; the GUI does not currently parse these.
- Hook point for structured warnings:
  - Add a warnings list into the playlist result payload in `src/playlist/pipeline.py` or `src/playlist_generator.py`, then pass to GUI in `handle_generate_playlist(...)`.

========================================================
4) UI MAPPING TO THREE SLIDERS (GOAL STATE)
========================================================

15) Where are existing mode presets defined?

- Mode presets (strict/narrow/dynamic/discover/off):
  - `src/playlist/mode_presets.py` -> `GENRE_MODE_PRESETS` and `SONIC_MODE_PRESETS`.
  - Applied at config load time in `src/config_loader.py` -> `_apply_mode_presets()`.
  - Default config example in `config.example.yaml` (comments and keys `playlists.genre_mode`, `playlists.sonic_mode`).

Proposed mapping for the DJ bridging slider controls:

- Mix Smoothness (sonic continuity + max_step guardrails)
  - Map to `playlists.ds_pipeline.pier_bridge.progress_arc.*` and `transition_floor`.
  - Hook: `PierBridgeConfig` fields in `src/playlist/pier_bridge_builder.py`.
- Genre Drift (genre ladder strength / roam distance)
  - Map to future "genre waypoint" weights and acceptable distance.
  - Hook: add to `PierBridgeConfig` and apply in `_beam_search_segment(...)`.
- Effort / Quality (beam width + pool sizes)
  - Map to `initial_neighbors_m`, `initial_bridge_helpers`, `initial_beam_width`, `max_beam_width` in `PierBridgeConfig`.
  - Hook: `build_pier_bridge_playlist(...)` initialization and backoff expansion.
- GUI pass-through:
  - `src/playlist_gui/config/settings_schema.py` would need new settings keys.
  - Values flow into config overrides, then into `src/playlist/pipeline.py`.

========================================================
5) TESTING & METRICS
========================================================

16) Best place to add evaluation support for DJ bridging?

- Reuse sweep harness:
  - `scripts/sweep_pier_bridge_dials.py` already computes post-hoc sonic and pacing metrics.
- Add multi-anchor scenarios:
  - Add new scenario definitions in `scripts/sweep_pier_bridge_dials.py`.
- Post-hoc metrics:
  - Use the same post-hoc structure already in the sweep harness:
    - arc deviation, max_jump, monotonic_violations, raw/bridge sonic similarity.
  - Add genre-path adherence by computing per-step cosine similarity against a genre target vector.
- Best placement:
  - `scripts/` for diagnostics and sweeps.
  - `tests/unit/` for math/logic (e.g., waypoint scoring, monotonic rules).

17) Deliverable summary

Architecture diagram (text-based):

GUI
  -> worker (NDJSON) `src/playlist_gui/worker.py`
    -> PlaylistGenerator `src/playlist_generator.py`
      -> DS pipeline `src/playlist/pipeline.py`
        -> build_pier_bridge_playlist `src/playlist/pier_bridge_builder.py`
          -> Segment pool builder `src/playlist/segment_pool_builder.py`
          -> Beam search `_beam_search_segment(...)`
            -> transition scoring `src/playlist/scoring/transition_scoring.py`
            -> genre penalty/tiebreak (X_genre_smoothed)
            -> progress arc (t projection)
          -> run audit `src/playlist/run_audit.py`
Artifacts:
  DB (`metadata.db`) -> build artifacts `src/analyze/artifact_builder.py`
  -> `ArtifactBundle` `src/features/artifacts.py`

Recommended implementation path (2-3 phases):

Phase 1 (Planning layer):
1) Add optional "seed_ordering" flag to preserve anchor order when locked.
2) Build a lightweight genre adjacency/waypoint planner using YAML or NPZ S matrix.

Phase 2 (Execution layer, opt-in):
1) Add optional genre waypoint scoring in `_beam_search_segment(...)`.
2) Add multi-source candidate pool union in `SegmentCandidatePoolBuilder.build(...)`.

Phase 3 (Reliability + UX):
1) Progressive relaxation ladder behind a DJ bridging flag.
2) Structured warnings in pipeline results, surfaced in GUI.
3) Add sweep scenarios + post-hoc genre path metrics.

File touch list (for future implementation):

- Auto-ordering default with lock override:
  - `src/playlist/pier_bridge_builder.py`
  - `src/playlist/pipeline.py`
  - `src/playlist/scoring/constraints.py` (if adding new config)

- Genre ladder planning:
  - New module under `src/genre/` or `src/playlist/` for waypoint planning
  - `src/analyze/genre_similarity.py` (optional precomputed graph)
  - `src/analyze/artifact_builder.py` (optional caching)

- Waypoint-aware execution (opt-in):
  - `src/playlist/pier_bridge_builder.py` (_beam_search_segment scoring)
  - `src/playlist/segment_pool_builder.py` (candidate union hook)

- Progressive relaxation + warnings:
  - `src/playlist/pier_bridge_builder.py`
  - `src/playlist/run_audit.py`
  - `src/playlist_gui/worker.py` (optional structured warning passthrough)

- Test plan + metrics:
  - `scripts/sweep_pier_bridge_dials.py`
  - `tests/unit/test_progress_arc.py` (extend for genre waypoint logic)
  - `tests/unit/test_segment_pool_builder.py`
