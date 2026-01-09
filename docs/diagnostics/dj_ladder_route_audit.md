DJ Ladder Route Audit (Step 0)
==============================

Scope: locate existing genre similarity assets, DS genre vector basis, current
DJ waypoint targeting, and config wiring. No code changes.

1) Genre similarity sources and runtime loaders
- data/genre_similarity.yaml
  - Loaded by src/genre_similarity_v2.py (GenreSimilarityV2.__init__) for
    similarity scoring.
  - Loaded by src/genre/similarity.py via YAML overrides inside
    pairwise_genre_similarity.
  - Loaded by src/playlist/pier_bridge_builder.py::_load_genre_similarity_graph
    when dj_bridging.route_shape == ladder (uses YAML edges only).
- Optional similarity matrix:
  - src/analyze/genre_similarity.py builds genre_similarity.npz (S matrix).
  - No runtime graph loader currently uses this NPZ for ladder planning.

2) DS genre vectors and basis
- Artifacts: src/features/artifacts.py loads X_genre_raw, X_genre_smoothed,
  and genre_vocab from the artifact bundle (npz).
- DS pipeline: src/playlist/pipeline.py carries X_genre_smoothed and genre_vocab
  through bundle restriction and into pier_bridge_builder.
- Pier-bridge: src/playlist/pier_bridge_builder.py normalizes X_genre_smoothed
  into X_genre_norm (dot-product basis for scoring and waypoints).

3) Current DJ waypoint guidance (gA/gB and g_target)
- gA/gB: src/playlist/pier_bridge_builder.py::_build_genre_targets uses
  g_a = X_genre_norm[pier_a], g_b = X_genre_norm[pier_b].
- Linear/arc targets: _build_genre_targets builds per-step g_targets by lerping
  g_a -> g_b (arc uses _progress_target_curve).
- Ladder targets: _build_genre_targets uses YAML graph and _shortest_genre_path
  to find label waypoints, then converts each label to a one-hot vector
  (_label_to_genre_vector), then linearly interpolates between waypoint vectors.
- Execution: _beam_search_segment reads g_targets[step] and uses
  waypoint_sim = dot(X_genre_norm[cand], g_target) to modify scores.

4) DJ config schema wiring
- Config example: config.example.yaml under playlists.pier_bridge.dj_bridging.
  Includes route_shape, ladder.{top_labels,min_label_weight,min_similarity,
  max_steps}, and waypoint_* settings.
- Config parsing: src/playlist/pipeline.py reads dj_bridging.* overrides and
  maps them to PierBridgeConfig fields.
- Runtime config: src/playlist/pier_bridge_builder.py PierBridgeConfig defines
  dj_* fields (route_shape, ladder, waypoint, pooling, etc.).

5) What appears reusable for ladder route planning
- Existing ladder planner logic exists in
  src/playlist/pier_bridge_builder.py:
  _load_genre_similarity_graph, _shortest_genre_path,
  _select_top_genre_labels, _label_to_genre_vector, _build_genre_targets.
- Current ladder waypoints use one-hot vectors (not similarity-smoothed rows)
  and only YAML edges (no NPZ S matrix).
- Waypoint vectors and scoring are already in the DS artifact basis via
  X_genre_smoothed -> X_genre_norm and genre_vocab.

Planned reuse
- Reuse existing DJ config keys (route_shape and ladder.*).
- Reuse X_genre_smoothed / genre_vocab basis from artifacts.
- Reuse waypoint scoring in _beam_search_segment; replace or wrap ladder
  planning to conform to existing g_targets vector basis and diagnostics.
