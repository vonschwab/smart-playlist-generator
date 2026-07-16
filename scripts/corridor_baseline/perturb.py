"""Pure perturbation logic for the 'no knob goes inert' sweep (contract §
'Automated completeness net'). No engine imports — unit-testable in isolation.

Perturbation rules (recorded in feature_baseline.json meta):
  bool  -> flipped
  C-term fields (soft scoring strengths/weights, contract Category C) -> 0.0
  float -> x1.5; 0.0 -> 0.3; *percentile*/*_floor* fields halve instead when
           x1.5 would exceed 1.0
  int   -> +1
  str / list / dict / None -> SKIP (recorded with reason; mode strings are
           covered by the dial audit / Category E, not this sweep)
"""
from __future__ import annotations

from typing import Any, Optional

SKIP = object()

_PREFIX_MAP = {
    "candidate_pool.": "playlists.ds_pipeline.candidate_pool.",
    "playlist.": "playlists.ds_pipeline.pier_bridge.",
}
_UNMAPPED_PREFIXES = ("sonic_variant.", "embedding.")

# The effective blob nests most PierBridgeConfig fields one level deeper
# (playlist.pier_config.<x>) than config.yaml stores them
# (playlists.ds_pipeline.pier_bridge.<x> — flat, no "pier_config" key).
# Verified against a real "Bill Evans Trio"/open effective blob (Step 5):
# without this strip, all ~180 playlist.pier_config.* leaves round-tripped
# to a nonexistent playlists.ds_pipeline.pier_bridge.pier_config.* path.
_INFIX_STRIP = {
    "playlist.": "pier_config.",
}

_PB = "playlists.ds_pipeline.pier_bridge."

# Per-field exception table for playlist.pier_config.<leaf> -> real yaml path,
# root-caused against production consumption (Fix-wave, task-5-report.md
# appendix has full evidence per family):
#   - src/playlist/pipeline/pier_bridge_overrides.py::apply_pier_bridge_overrides
#     reads overrides["pier_bridge"] (== playlists.ds_pipeline.pier_bridge verbatim
#     from src/playlist_generator.py::build_ds_overrides). ~11 sub-blocks
#     (dj_bridging, edge_repair, edge_delete, progress, genre, tail_dp,
#     title_artifact_penalty, experiments, progress_arc, roam) are read via
#     pb_overrides.get("<subblock>") -- a DIFFERENT dataclass-field-name shape
#     than the flat blob leaf, so the naive flat mapping silently misses them
#     (confirmed via a real generation: dj_genre_coverage_weight came back
#     did_not_resolve under the flat mapping).
#   - src/playlist/config.py::resolve_pier_bridge_tuning reads a DIFFERENT key
#     name for two fields (soft_genre_penalty_threshold/strength, not
#     genre_penalty_threshold/strength -- the blob leaf names).
#   - src/playlist/pipeline/core.py:865-888 unconditionally clobbers
#     pb_cfg.pace_bridge_floor from cfg.candidate.pace_bridge_floor (a
#     CandidatePoolConfig field, itself sourced from candidate_pool.*, not
#     pier_bridge.*) and pb_cfg.bpm_stability_min from cfg.candidate.bpm_stability_min
#     -- any playlists.ds_pipeline.pier_bridge.pace_bridge_floor /
#     .bpm_stability_min override is silently overwritten after
#     apply_pier_bridge_overrides returns.
#   - src/playlist/config.py::default_ds_config sources center_transitions from
#     constraints.center_transitions (with a ds_pipeline-top-level back-compat
#     alias) and transition_floor from constraints.transition_floor(_<mode>) --
#     neither is ever read from pier_bridge.* despite living in the pier_config
#     blob namespace.
#   - Several fields are populated ONLY from src/playlist/mode_presets.py's
#     PACE_MODE_PRESETS (resolve_pace_mode(pace_mode) is called WITHOUT an
#     overrides param in pipeline/core.py:362) or are dataclass-default-only
#     (never assigned from any override dict anywhere in src/) -- these are
#     genuine dead outlets, mapped to None rather than a path that will never
#     resolve.
_PIER_CONFIG_FIELD_MAP: dict[str, Optional[str]] = {
    # ---- cross-family redirects (blob leaf lives under pier_config, but
    #      production reads a DIFFERENT top-level ds_pipeline family) ----
    "bpm_stability_min": "playlists.ds_pipeline.candidate_pool.bpm_stability_min",
    "pace_bridge_floor": "playlists.ds_pipeline.candidate_pool.pace_bridge_floor",
    "center_transitions": "playlists.ds_pipeline.constraints.center_transitions",
    "transition_floor": "playlists.ds_pipeline.constraints.transition_floor",

    # ---- base-name mismatch (resolve_pier_bridge_tuning reads "soft_" keys) ----
    "genre_penalty_threshold": _PB + "soft_genre_penalty_threshold",
    "genre_penalty_strength": _PB + "soft_genre_penalty_strength",

    # ---- dj_bridging family (nested; some Phase-2/3 sub-keys keep the "dj_"
    #      prefix even nested -- see pier_bridge_overrides.py:711-760) ----
    "dj_allow_detours_when_far": _PB + "dj_bridging.allow_detours_when_far",
    "dj_anchors_must_include_all": _PB + "dj_bridging.anchors.must_include_all",
    "dj_bridging_enabled": _PB + "dj_bridging.enabled",
    "dj_connector_bias_enabled": _PB + "dj_bridging.connector_bias.enabled",
    "dj_connector_max_per_segment_adventurous": _PB + "dj_bridging.connector_bias.max_per_segment_adventurous",
    "dj_connector_max_per_segment_linear": _PB + "dj_bridging.connector_bias.max_per_segment_linear",
    "dj_coverage_mode": _PB + "dj_bridging.dj_coverage_mode",
    "dj_coverage_presence_source": _PB + "dj_bridging.dj_coverage_presence_source",
    "dj_diagnostics_pool_verbose": _PB + "dj_bridging.diagnostics.pool_verbose",
    "dj_diagnostics_waypoint_rank_impact_enabled": _PB + "dj_bridging.diagnostics.waypoint_rank_impact_enabled",
    "dj_diagnostics_waypoint_rank_sample_steps": _PB + "dj_bridging.diagnostics.waypoint_rank_sample_steps",
    "dj_far_threshold_connector_scarcity": _PB + "dj_bridging.far_thresholds.connector_scarcity",
    "dj_far_threshold_genre": _PB + "dj_bridging.far_thresholds.genre",
    "dj_far_threshold_sonic": _PB + "dj_bridging.far_thresholds.sonic",
    "dj_genre_coverage_power": _PB + "dj_bridging.dj_genre_coverage_power",
    "dj_genre_coverage_top_k": _PB + "dj_bridging.dj_genre_coverage_top_k",
    "dj_genre_coverage_weight": _PB + "dj_bridging.dj_genre_coverage_weight",
    "dj_genre_idf_norm": _PB + "dj_bridging.dj_genre_idf_norm",
    "dj_genre_idf_power": _PB + "dj_bridging.dj_genre_idf_power",
    "dj_genre_pool_transition_blend": None,  # dead: never read from any override dict (src grep confirmed)
    "dj_genre_presence_threshold": _PB + "dj_bridging.dj_genre_presence_threshold",
    "dj_genre_use_coverage": _PB + "dj_bridging.dj_genre_use_coverage",
    "dj_genre_use_idf": _PB + "dj_bridging.dj_genre_use_idf",
    "dj_genre_vector_source": _PB + "dj_bridging.dj_genre_vector_source",
    "dj_ladder_max_steps": _PB + "dj_bridging.ladder.max_steps",
    "dj_ladder_min_label_weight": _PB + "dj_bridging.ladder.min_label_weight",
    "dj_ladder_min_similarity": _PB + "dj_bridging.ladder.min_similarity",
    "dj_ladder_smooth_min_sim": _PB + "dj_bridging.ladder.smooth_min_sim",
    "dj_ladder_smooth_top_k": _PB + "dj_bridging.ladder.smooth_top_k",
    "dj_ladder_target_mode": _PB + "dj_bridging.dj_ladder_target_mode",
    "dj_ladder_top_labels": _PB + "dj_bridging.ladder.top_labels",
    "dj_ladder_use_smoothed_waypoint_vectors": _PB + "dj_bridging.ladder.use_smoothed_waypoint_vectors",
    "dj_micro_piers_candidate_source": _PB + "dj_bridging.micro_piers.candidate_source",
    "dj_micro_piers_enabled": _PB + "dj_bridging.micro_piers.enabled",
    "dj_micro_piers_max": _PB + "dj_bridging.micro_piers.max",
    "dj_micro_piers_selection_metric": _PB + "dj_bridging.micro_piers.selection_metric",
    "dj_micro_piers_topk": _PB + "dj_bridging.micro_piers.topk",
    "dj_pooling_cache_enabled": _PB + "dj_bridging.pooling.cache_enabled",
    "dj_pooling_debug_compare_baseline": _PB + "dj_bridging.pooling.debug_compare_baseline",
    "dj_pooling_k_genre": _PB + "dj_bridging.pooling.k_genre",
    "dj_pooling_k_local": _PB + "dj_bridging.pooling.k_local",
    "dj_pooling_k_toward": _PB + "dj_bridging.pooling.k_toward",
    "dj_pooling_k_union_max": _PB + "dj_bridging.pooling.k_union_max",
    "dj_pooling_step_stride": _PB + "dj_bridging.pooling.step_stride",
    "dj_pooling_strategy": _PB + "dj_bridging.pooling.strategy",
    "dj_relaxation_allow_floor_relaxation": _PB + "dj_bridging.relaxation.allow_floor_relaxation",
    "dj_relaxation_emit_warnings": _PB + "dj_bridging.relaxation.emit_warnings",
    "dj_relaxation_enabled": _PB + "dj_bridging.relaxation.enabled",
    "dj_relaxation_max_attempts": _PB + "dj_bridging.relaxation.max_attempts",
    "dj_route_shape": _PB + "dj_bridging.route_shape",
    "dj_seed_ordering": _PB + "dj_bridging.seed_ordering",
    "dj_seed_ordering_weight_bridge": _PB + "dj_bridging.seed_ordering_weight_bridge",
    "dj_seed_ordering_weight_genre": _PB + "dj_bridging.seed_ordering_weight_genre",
    "dj_seed_ordering_weight_sonic": _PB + "dj_bridging.seed_ordering_weight_sonic",
    "dj_waypoint_cap": _PB + "dj_bridging.waypoint_cap",
    "dj_waypoint_centered_baseline": _PB + "dj_bridging.dj_waypoint_centered_baseline",
    "dj_waypoint_delta_mode": _PB + "dj_bridging.dj_waypoint_delta_mode",
    "dj_waypoint_fallback_k": _PB + "dj_bridging.waypoint_fallback_k",
    "dj_waypoint_floor": _PB + "dj_bridging.waypoint_floor",
    "dj_waypoint_penalty": _PB + "dj_bridging.waypoint_penalty",
    "dj_waypoint_squash": _PB + "dj_bridging.dj_waypoint_squash",
    "dj_waypoint_squash_alpha": _PB + "dj_bridging.dj_waypoint_squash_alpha",
    "dj_waypoint_tie_break_band": _PB + "dj_bridging.waypoint_tie_break_band",
    "dj_waypoint_weight": _PB + "dj_bridging.waypoint_weight",
    "taxonomy_waypoint_min_library_mass": None,  # dead: only ever read from its own prior value

    # ---- edge_repair / edge_delete (nested) ----
    "edge_delete_enabled": _PB + "edge_delete.enabled",
    "edge_delete_floor": _PB + "edge_delete.floor",
    "edge_delete_max_deletions": _PB + "edge_delete.max_deletions",
    "edge_repair_centered_cos_floor": _PB + "edge_repair.centered_cos_floor",
    "edge_repair_enabled": _PB + "edge_repair.enabled",
    "edge_repair_margin": _PB + "edge_repair.margin",
    "edge_repair_t_floor": _PB + "edge_repair.t_floor",
    "edge_repair_variety_guard_enabled": _PB + "edge_repair.variety_guard.enabled",
    "edge_repair_variety_guard_threshold": _PB + "edge_repair.variety_guard.threshold",

    # ---- progress / progress_arc (nested; progress_arc has TWO source paths in
    #      production -- pier_bridge.progress_arc.* (unconditional) and
    #      pier_bridge.experiments.progress_arc.* (dry_run/audit-gated) -- the
    #      unconditional one is mapped so perturbations are visible in real runs) ----
    "progress_enabled": _PB + "progress.enabled",
    "progress_monotonic_epsilon": _PB + "progress.monotonic_epsilon",
    "progress_penalty_weight": _PB + "progress.penalty_weight",
    "progress_arc_enabled": _PB + "progress_arc.enabled",
    "progress_arc_weight": _PB + "progress_arc.weight",
    "progress_arc_shape": _PB + "progress_arc.shape",
    "progress_arc_tolerance": _PB + "progress_arc.tolerance",
    "progress_arc_loss": _PB + "progress_arc.loss",
    "progress_arc_huber_delta": _PB + "progress_arc.huber_delta",
    "progress_arc_max_step": _PB + "progress_arc.max_step",
    "progress_arc_max_step_mode": _PB + "progress_arc.max_step_mode",
    "progress_arc_max_step_penalty": _PB + "progress_arc.max_step_penalty",
    "progress_arc_autoscale_enabled": _PB + "progress_arc.autoscale.enabled",
    "progress_arc_autoscale_min_distance": _PB + "progress_arc.autoscale.min_distance",
    "progress_arc_autoscale_distance_scale": _PB + "progress_arc.autoscale.distance_scale",
    "progress_arc_autoscale_per_step_scale": _PB + "progress_arc.autoscale.per_step_scale",

    # ---- tail_dp (nested) ----
    "tail_dp_enabled": _PB + "tail_dp.enabled",
    "tail_dp_epsilon": _PB + "tail_dp.epsilon",
    "tail_dp_floor": _PB + "tail_dp.floor",

    # ---- genre.* (nested; only tie_break_band is wired) ----
    "genre_tie_break_band": _PB + "genre.tie_break_band",

    # ---- title_artifact_penalty (nested) ----
    "title_artifact_penalty_enabled": _PB + "title_artifact_penalty.enabled",
    "title_artifact_penalty_weights": _PB + "title_artifact_penalty.weights",

    # ---- experiments (nested; dry_run/audit-gated -- production forces these
    #      False/no-op outside dry_run/audit, so a real-generation sweep will
    #      see did_not_resolve here even though the path IS what production
    #      reads; see task-5-report.md appendix Concern) ----
    "experiment_bridge_scoring_enabled": _PB + "experiments.bridge_scoring.enabled",
    "experiment_bridge_min_weight": _PB + "experiments.bridge_scoring.min_weight",
    "experiment_bridge_balance_weight": _PB + "experiments.bridge_scoring.balance_weight",

    # ---- roam (nested; roam_kwargs_from_dict is the sole reader) ----
    "roam_corridors_enabled": _PB + "roam.enabled",
    "roam_knn_k": _PB + "roam.knn_k",
    "roam_mutual_proximity": _PB + "roam.mutual_proximity",
    "roam_width_sonic": _PB + "roam.width_sonic",
    "roam_width_genre": _PB + "roam.width_genre",
    "roam_width_energy": _PB + "roam.width_energy",
    "roam_penalty_slope": _PB + "roam.penalty_slope",
    "worst_edge_minimax_enabled": _PB + "roam.worst_edge_minimax",  # ONLY path; no flat alias

    # ---- genuinely dead outlets: never assigned from ANY override dict in
    #      src/ (confirmed by grep across src/playlist*) -- either a
    #      pace_mode-preset-only value (resolve_pace_mode is called WITHOUT an
    #      overrides param in pipeline/core.py:362, so these can never be
    #      influenced by pier_bridge.* config even though a preset->override
    #      seam exists in the function signature) or a dataclass-default-only
    #      constant with no override call site at all ----
    "bpm_bridge_max_log_distance": None,
    "bpm_bridge_soft_penalty_strength": None,
    "bpm_trust_min_onset_rate": None,
    "onset_bridge_max_log_distance": None,
    "onset_bridge_soft_penalty_strength": None,
    "weight_end_start": None,
    "weight_mid_mid": None,
    "weight_full_full": None,
    "eta_destination_pull": None,
    "transition_calib_center": None,
    "transition_calib_scale": None,
    "transition_calib_gain": None,
    "max_expansion_attempts": None,
    "popularity_penalty_strength": None,  # artist-mode: derived from popularity_mode + playlists.bangers.*, not pier_bridge.*
    "layered_transition_scoring_enabled": None,  # derived from genre_graph.source == "layered", not a direct bool leaf
    "layered_transition_weight": None,  # only assigned when genre_graph.source == "layered"
    "layered_transition_mode": None,  # hardcoded to cfg.mode (cohesion_mode), never read from an override
}

# Contract Category C soft-term knobs -> perturb to 0.0 ("term fires" differential).
C_TERM_FIELDS = frozenset({
    "duration_penalty_weight",          # C1
    "seed_character_strength",          # C2
    "popularity_penalty_strength",      # C3
    "local_sonic_edge_penalty_strength",  # C4
    "soft_genre_penalty_strength",      # C5
    "genre_tiebreak_weight",            # C6
    "genre_pair_penalty",               # C7
    "progress_penalty_weight",          # C8
    "bpm_bridge_soft_penalty_strength",   # C9
    "onset_bridge_soft_penalty_strength",  # C9
    "instrumental_penalty_weight",      # C10
    "weight_bridge",                    # C11
    "weight_transition",                # C11
})


def flatten_leaves(blob: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in blob.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_leaves(v, f"{path}."))
        else:
            out[path] = v
    return out


def config_path_for(blob_path: str) -> str | None:
    if blob_path.startswith(_UNMAPPED_PREFIXES):
        return None
    for pfx, target in _PREFIX_MAP.items():
        if blob_path.startswith(pfx):
            suffix = blob_path[len(pfx):]
            infix = _INFIX_STRIP.get(pfx)
            if infix and suffix.startswith(infix):
                suffix = suffix[len(infix):]
                # playlist.pier_config.<leaf> leaves: ~11 config.yaml sub-blocks
                # (dj_bridging, edge_repair, progress, roam, ...) nest PierBridgeConfig
                # fields under a family key that does NOT match the flat dataclass
                # field name production reads them as (pier_bridge_overrides.py); a
                # handful of others are cross-family redirects or genuine dead
                # outlets. Check the per-field table BEFORE falling back to the flat
                # default -- see _PIER_CONFIG_FIELD_MAP's docstring-comment for the
                # full root-cause evidence.
                if pfx == "playlist." and suffix in _PIER_CONFIG_FIELD_MAP:
                    return _PIER_CONFIG_FIELD_MAP[suffix]
            return target + suffix
    return None


def perturb_value(field_name: str, value: Any) -> Any:
    leaf = field_name.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        return not value
    if leaf in C_TERM_FIELDS and isinstance(value, (int, float)) and value != 0:
        return 0.0
    if isinstance(value, float):
        if value == 0.0:
            return 0.3
        if ("percentile" in leaf or leaf.endswith("_floor")) and value * 1.5 > 1.0:
            return value * 0.5
        return value * 1.5
    if isinstance(value, int):
        return value + 1
    return SKIP
