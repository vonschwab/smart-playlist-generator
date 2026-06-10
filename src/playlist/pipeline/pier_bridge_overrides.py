"""Pier-bridge override translation — user-facing dict -> PierBridgeConfig.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

This is the largest single chunk of pipeline.py: ~600 LOC of
``if isinstance(...): pb_cfg = replace(pb_cfg, ...)`` translation that
takes a free-form ``overrides["pier_bridge"]`` dict (pulled from
config.yaml or constructed by the GUI / CLI) and applies it on top of
either a caller-supplied ``PierBridgeConfig`` or one built fresh from
the resolved tuning defaults.

The block is preserved **verbatim** from the original inline
implementation. The pier_bridge override-parsing golden tests
(tests/unit/test_pipeline_smoke_golden.py) compare the resulting
PierBridgeConfig dict to checked-in baselines for four representative
override scenarios; any drift fails CI.

Future cleanup (out of scope for Tier-1.5): replace this with a
declarative schema validator (e.g. pydantic) so the parser is
table-driven rather than handwritten. Doing that requires its own
test pass and is tracked separately.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from src.playlist.config import DSPipelineConfig, resolve_pier_bridge_tuning
from src.playlist.pier_bridge_builder import PierBridgeConfig

logger = logging.getLogger(__name__)


def apply_pier_bridge_overrides(
    *,
    pier_bridge_config: Optional[PierBridgeConfig],
    cfg: DSPipelineConfig,
    overrides: Optional[dict],
    pb_overrides: dict,
    artist_playlist: bool,
    dry_run: bool,
    audit_cfg: Any,  # src.playlist.run_audit.RunAuditConfig
    resolved_variant: str,
) -> Tuple[PierBridgeConfig, Any, Dict[str, Any], Optional[Tuple[float, float, float]]]:
    """Translate ``overrides`` into a fully-populated PierBridgeConfig.

    Returns ``(pb_cfg, tuning, tuning_sources, transition_weights)`` —
    the tuning + sources are returned because the orchestrator emits
    them in the preflight audit event downstream.
    """
    transition_weights = None
    try:
        tw_raw = (overrides or {}).get("transition_weights")
        if isinstance(tw_raw, dict):
            transition_weights = (
                float(tw_raw.get("rhythm", 0.4)),
                float(tw_raw.get("timbre", 0.35)),
                float(tw_raw.get("harmony", 0.25)),
            )
        elif isinstance(tw_raw, (list, tuple)) and len(tw_raw) == 3:
            transition_weights = (
                float(tw_raw[0]),
                float(tw_raw[1]),
                float(tw_raw[2]),
            )
    except Exception:
        transition_weights = None

    tuning, tuning_sources = resolve_pier_bridge_tuning(
        mode=cfg.mode,
        similarity_floor=float(cfg.candidate.similarity_floor),
        overrides=overrides,
    )
    logger.info(
        "Pier-bridge tuning resolved: mode=%s transition_floor=%.2f bridge_floor=%.2f weight_bridge=%.2f weight_transition=%.2f genre_tiebreak_weight=%.2f genre_penalty_threshold=%.2f genre_penalty_strength=%.2f",
        cfg.mode,
        float(tuning.transition_floor),
        float(tuning.bridge_floor),
        float(tuning.weight_bridge),
        float(tuning.weight_transition),
        float(tuning.genre_tiebreak_weight),
        float(tuning.genre_penalty_threshold),
        float(tuning.genre_penalty_strength),
    )
    if logger.isEnabledFor(logging.DEBUG):
        for field, source in sorted(tuning_sources.items()):
            if source != "default":
                logger.debug(
                    "Pier-bridge tuning override: %s=%s source=%s",
                    field,
                    getattr(tuning, field, None),
                    source,
                )
    pb_cfg = pier_bridge_config or PierBridgeConfig(
        transition_floor=float(tuning.transition_floor),
        bridge_floor=float(tuning.bridge_floor),
        pace_bridge_floor=float(getattr(cfg.candidate, "pace_bridge_floor", 0.0)),
        center_transitions=cfg.construct.center_transitions,
        transition_weights=transition_weights,
        sonic_variant=resolved_variant,
        weight_bridge=float(tuning.weight_bridge),
        weight_transition=float(tuning.weight_transition),
        genre_tiebreak_weight=float(tuning.genre_tiebreak_weight),
        genre_penalty_threshold=float(tuning.genre_penalty_threshold),
        genre_penalty_strength=float(tuning.genre_penalty_strength),
        genre_steering_enabled=bool(tuning.genre_steering_enabled),
        genre_steering_source=str(getattr(tuning, "genre_steering_source", "dense")),
        segment_pool_genre_weight=float(getattr(tuning, "segment_pool_genre_weight", 0.0)),
        weight_genre=float(tuning.weight_genre),
        genre_arc_floor=float(tuning.genre_arc_floor),
        genre_arc_floor_percentile=float(tuning.genre_arc_floor_percentile),
        genre_admission_percentile=float(tuning.genre_admission_percentile),
    )
    if isinstance(pb_overrides.get("pace_bridge_floor"), (int, float)):
        pb_cfg = replace(pb_cfg, pace_bridge_floor=float(pb_overrides.get("pace_bridge_floor")))

    genre_graph_source = "legacy"
    genre_graph_raw = (overrides or {}).get("genre_graph") if isinstance(overrides, dict) else None
    if isinstance(genre_graph_raw, dict):
        genre_graph_source = str(genre_graph_raw.get("source") or "legacy").strip().lower()
        if genre_graph_source == "layered":
            raw_weight = genre_graph_raw.get("transition_weight", 0.15)
            transition_weight = float(raw_weight) if isinstance(raw_weight, (int, float)) else 0.15
            pb_cfg = replace(
                pb_cfg,
                layered_transition_scoring_enabled=True,
                layered_transition_weight=max(0.0, float(transition_weight)),
                layered_transition_mode=str(cfg.mode),
            )

    # Segment-local pier-bridge policy defaults (with optional overrides).
    # CRITICAL: For artist playlists, seed artist must ONLY appear as piers (design principle)
    if artist_playlist:
        pb_cfg = replace(
            pb_cfg,
            disallow_seed_artist_in_interiors=True,
        )
    else:
        disallow_seed_raw = pb_overrides.get("disallow_seed_artist_in_interiors")
        if isinstance(disallow_seed_raw, bool):
            pb_cfg = replace(
                pb_cfg,
                disallow_seed_artist_in_interiors=bool(disallow_seed_raw),
            )

    disallow_pier_raw = pb_overrides.get("disallow_pier_artists_in_interiors")
    if isinstance(disallow_pier_raw, bool):
        pb_cfg = replace(
            pb_cfg,
            disallow_pier_artists_in_interiors=bool(disallow_pier_raw),
        )

    max_non_seed_tracks_per_artist = pb_overrides.get("max_non_seed_tracks_per_artist")
    if max_non_seed_tracks_per_artist is None:
        pb_cfg = replace(pb_cfg, max_non_seed_tracks_per_artist=None)
    elif isinstance(max_non_seed_tracks_per_artist, int) and int(max_non_seed_tracks_per_artist) > 0:
        pb_cfg = replace(
            pb_cfg,
            max_non_seed_tracks_per_artist=int(max_non_seed_tracks_per_artist),
        )

    segment_pool_strategy = pb_overrides.get("segment_pool_strategy")
    if isinstance(segment_pool_strategy, str) and segment_pool_strategy.strip():
        pb_cfg = replace(
            pb_cfg,
            segment_pool_strategy=str(segment_pool_strategy).strip(),
        )

    segment_pool_max = pb_overrides.get("segment_pool_max")
    if isinstance(segment_pool_max, int) and int(segment_pool_max) > 0:
        pb_cfg = replace(pb_cfg, segment_pool_max=int(segment_pool_max))

    max_segment_pool_max = pb_overrides.get("max_segment_pool_max")
    if isinstance(max_segment_pool_max, int) and int(max_segment_pool_max) > 0:
        pb_cfg = replace(pb_cfg, max_segment_pool_max=int(max_segment_pool_max))

    collapse_segment_pool_by_artist = pb_overrides.get("collapse_segment_pool_by_artist")
    if isinstance(collapse_segment_pool_by_artist, bool):
        pb_cfg = replace(
            pb_cfg,
            collapse_segment_pool_by_artist=bool(collapse_segment_pool_by_artist),
        )

    title_artifact = pb_overrides.get("title_artifact_penalty")
    if isinstance(title_artifact, dict):
        enabled = title_artifact.get("enabled")
        weights = title_artifact.get("weights")
        if isinstance(enabled, bool):
            pb_cfg = replace(pb_cfg, title_artifact_penalty_enabled=enabled)
        if isinstance(weights, dict):
            normalized = {
                str(k): float(v) for k, v in weights.items()
                if isinstance(v, (int, float))
            }
            pb_cfg = replace(pb_cfg, title_artifact_penalty_weights=normalized or None)

    emit_audit = pb_overrides.get("emit_selected_edge_audit")
    if isinstance(emit_audit, bool):
        pb_cfg = replace(pb_cfg, emit_selected_edge_audit=bool(emit_audit))

    min_edge_obj = pb_overrides.get("min_edge_objective")
    if isinstance(min_edge_obj, str) and min_edge_obj.strip():
        pb_cfg = replace(pb_cfg, min_edge_objective=min_edge_obj.strip())

    edge_repair = pb_overrides.get("edge_repair")
    if isinstance(edge_repair, dict):
        if isinstance(edge_repair.get("enabled"), bool):
            pb_cfg = replace(pb_cfg, edge_repair_enabled=bool(edge_repair.get("enabled")))
        if isinstance(edge_repair.get("centered_cos_floor"), (int, float)):
            pb_cfg = replace(
                pb_cfg,
                edge_repair_centered_cos_floor=float(edge_repair.get("centered_cos_floor")),
            )
        if isinstance(edge_repair.get("margin"), (int, float)):
            pb_cfg = replace(pb_cfg, edge_repair_margin=float(edge_repair.get("margin")))
        variety_guard = edge_repair.get("variety_guard")
        if isinstance(variety_guard, dict):
            if isinstance(variety_guard.get("enabled"), bool):
                pb_cfg = replace(
                    pb_cfg,
                    edge_repair_variety_guard_enabled=bool(variety_guard.get("enabled")),
                )
            if isinstance(variety_guard.get("threshold"), (int, float)):
                pb_cfg = replace(
                    pb_cfg,
                    edge_repair_variety_guard_threshold=float(variety_guard.get("threshold")),
                )

    progress_raw = pb_overrides.get("progress")
    if isinstance(progress_raw, dict):
        if isinstance(progress_raw.get("enabled"), bool):
            pb_cfg = replace(pb_cfg, progress_enabled=bool(progress_raw.get("enabled")))
        if isinstance(progress_raw.get("monotonic_epsilon"), (int, float)):
            pb_cfg = replace(
                pb_cfg,
                progress_monotonic_epsilon=float(progress_raw.get("monotonic_epsilon")),
            )
        if isinstance(progress_raw.get("penalty_weight"), (int, float)):
            pb_cfg = replace(
                pb_cfg,
                progress_penalty_weight=float(progress_raw.get("penalty_weight")),
            )

    genre_raw = pb_overrides.get("genre")
    if isinstance(genre_raw, dict):
        tie_break_band = genre_raw.get("tie_break_band")
        if isinstance(tie_break_band, (int, float)):
            pb_cfg = replace(
                pb_cfg,
                genre_tie_break_band=float(tie_break_band),
            )

    local_sonic_enabled = pb_cfg.local_sonic_edge_penalty_enabled
    local_sonic_threshold = pb_cfg.local_sonic_edge_penalty_threshold
    local_sonic_strength = pb_cfg.local_sonic_edge_penalty_strength
    local_sonic_mode = pb_cfg.local_sonic_edge_penalty_mode
    local_sonic_scale = pb_cfg.local_sonic_edge_penalty_scale
    local_sonic_floor = pb_cfg.local_sonic_edge_floor
    if isinstance(pb_overrides.get("local_sonic_edge_penalty_enabled"), bool):
        local_sonic_enabled = bool(pb_overrides.get("local_sonic_edge_penalty_enabled"))
    if isinstance(pb_overrides.get("local_sonic_edge_penalty_threshold"), (int, float)):
        local_sonic_threshold = float(pb_overrides.get("local_sonic_edge_penalty_threshold"))
    if isinstance(pb_overrides.get("local_sonic_edge_penalty_strength"), (int, float)):
        local_sonic_strength = float(pb_overrides.get("local_sonic_edge_penalty_strength"))
    if isinstance(pb_overrides.get("local_sonic_edge_penalty_mode"), str):
        local_sonic_mode = str(pb_overrides.get("local_sonic_edge_penalty_mode")).strip()
    if isinstance(pb_overrides.get("local_sonic_edge_penalty_scale"), (int, float)):
        local_sonic_scale = float(pb_overrides.get("local_sonic_edge_penalty_scale"))
    if "local_sonic_edge_floor" in pb_overrides:
        raw_floor = pb_overrides.get("local_sonic_edge_floor")
        local_sonic_floor = float(raw_floor) if isinstance(raw_floor, (int, float)) else None
    local_sonic_raw = pb_overrides.get("local_sonic_edge_penalty")
    if isinstance(local_sonic_raw, dict):
        if isinstance(local_sonic_raw.get("enabled"), bool):
            local_sonic_enabled = bool(local_sonic_raw.get("enabled"))
        if isinstance(local_sonic_raw.get("threshold"), (int, float)):
            local_sonic_threshold = float(local_sonic_raw.get("threshold"))
        if isinstance(local_sonic_raw.get("strength"), (int, float)):
            local_sonic_strength = float(local_sonic_raw.get("strength"))
        if isinstance(local_sonic_raw.get("mode"), str):
            local_sonic_mode = str(local_sonic_raw.get("mode")).strip()
        if isinstance(local_sonic_raw.get("scale"), (int, float)):
            local_sonic_scale = float(local_sonic_raw.get("scale"))
        if "floor" in local_sonic_raw:
            raw_floor = local_sonic_raw.get("floor")
            local_sonic_floor = float(raw_floor) if isinstance(raw_floor, (int, float)) else None
    if (
        bool(local_sonic_enabled) != bool(pb_cfg.local_sonic_edge_penalty_enabled)
        or float(local_sonic_threshold) != float(pb_cfg.local_sonic_edge_penalty_threshold)
        or float(local_sonic_strength) != float(pb_cfg.local_sonic_edge_penalty_strength)
        or str(local_sonic_mode) != str(pb_cfg.local_sonic_edge_penalty_mode)
        or float(local_sonic_scale) != float(pb_cfg.local_sonic_edge_penalty_scale)
        or local_sonic_floor != pb_cfg.local_sonic_edge_floor
    ):
        pb_cfg = replace(
            pb_cfg,
            local_sonic_edge_penalty_enabled=bool(local_sonic_enabled),
            local_sonic_edge_penalty_threshold=float(local_sonic_threshold),
            local_sonic_edge_penalty_strength=max(0.0, float(local_sonic_strength)),
            local_sonic_edge_penalty_mode=str(local_sonic_mode),
            local_sonic_edge_penalty_scale=max(0.0, float(local_sonic_scale)),
            local_sonic_edge_floor=local_sonic_floor,
        )

    experiment_enabled = False
    experiment_min_weight = float(pb_cfg.experiment_bridge_min_weight)
    experiment_balance_weight = float(pb_cfg.experiment_bridge_balance_weight)
    experiments_raw = pb_overrides.get("experiments")
    progress_arc_enabled = bool(pb_cfg.progress_arc_enabled)
    progress_arc_weight = float(pb_cfg.progress_arc_weight)
    progress_arc_shape = str(pb_cfg.progress_arc_shape or "linear")
    progress_arc_tolerance = float(pb_cfg.progress_arc_tolerance)
    progress_arc_loss = str(pb_cfg.progress_arc_loss or "abs")
    progress_arc_huber_delta = float(pb_cfg.progress_arc_huber_delta)
    progress_arc_max_step = pb_cfg.progress_arc_max_step
    progress_arc_max_step_mode = str(pb_cfg.progress_arc_max_step_mode or "penalty")
    progress_arc_max_step_penalty = float(pb_cfg.progress_arc_max_step_penalty)
    progress_arc_autoscale_enabled = bool(pb_cfg.progress_arc_autoscale_enabled)
    progress_arc_autoscale_min_distance = float(pb_cfg.progress_arc_autoscale_min_distance)
    progress_arc_autoscale_distance_scale = float(pb_cfg.progress_arc_autoscale_distance_scale)
    progress_arc_autoscale_per_step_scale = bool(pb_cfg.progress_arc_autoscale_per_step_scale)
    progress_arc_source = None

    progress_arc_raw = pb_overrides.get("progress_arc")
    if isinstance(progress_arc_raw, dict):
        progress_arc_source = "pier_bridge.progress_arc"
        if isinstance(progress_arc_raw.get("enabled"), bool):
            progress_arc_enabled = bool(progress_arc_raw.get("enabled"))
        if isinstance(progress_arc_raw.get("weight"), (int, float)):
            progress_arc_weight = float(progress_arc_raw.get("weight"))
        if isinstance(progress_arc_raw.get("shape"), str) and progress_arc_raw.get("shape").strip():
            progress_arc_shape = str(progress_arc_raw.get("shape")).strip().lower()
        if isinstance(progress_arc_raw.get("tolerance"), (int, float)):
            progress_arc_tolerance = float(progress_arc_raw.get("tolerance"))
        if isinstance(progress_arc_raw.get("loss"), str) and progress_arc_raw.get("loss").strip():
            progress_arc_loss = str(progress_arc_raw.get("loss")).strip().lower()
        if isinstance(progress_arc_raw.get("huber_delta"), (int, float)):
            progress_arc_huber_delta = float(progress_arc_raw.get("huber_delta"))
        if isinstance(progress_arc_raw.get("max_step"), (int, float)):
            progress_arc_max_step = float(progress_arc_raw.get("max_step"))
        elif progress_arc_raw.get("max_step") is None:
            progress_arc_max_step = None
        if isinstance(progress_arc_raw.get("max_step_mode"), str) and progress_arc_raw.get("max_step_mode").strip():
            progress_arc_max_step_mode = str(progress_arc_raw.get("max_step_mode")).strip().lower()
        if isinstance(progress_arc_raw.get("max_step_penalty"), (int, float)):
            progress_arc_max_step_penalty = float(progress_arc_raw.get("max_step_penalty"))
        autoscale_raw = progress_arc_raw.get("autoscale")
        if isinstance(autoscale_raw, dict):
            if isinstance(autoscale_raw.get("enabled"), bool):
                progress_arc_autoscale_enabled = bool(autoscale_raw.get("enabled"))
            if isinstance(autoscale_raw.get("min_distance"), (int, float)):
                progress_arc_autoscale_min_distance = float(autoscale_raw.get("min_distance"))
            if isinstance(autoscale_raw.get("distance_scale"), (int, float)):
                progress_arc_autoscale_distance_scale = float(autoscale_raw.get("distance_scale"))
            if isinstance(autoscale_raw.get("per_step_scale"), bool):
                progress_arc_autoscale_per_step_scale = bool(autoscale_raw.get("per_step_scale"))

    if isinstance(experiments_raw, dict):
        bridge_scoring_raw = experiments_raw.get("bridge_scoring", {})
        if isinstance(bridge_scoring_raw, dict):
            if isinstance(bridge_scoring_raw.get("enabled"), bool):
                experiment_enabled = bool(bridge_scoring_raw.get("enabled"))
            if isinstance(bridge_scoring_raw.get("min_weight"), (int, float)):
                experiment_min_weight = float(bridge_scoring_raw.get("min_weight"))
            if isinstance(bridge_scoring_raw.get("balance_weight"), (int, float)):
                experiment_balance_weight = float(bridge_scoring_raw.get("balance_weight"))
        progress_raw = experiments_raw.get("progress_arc", {})
        if isinstance(progress_raw, dict) and progress_arc_source is None:
            progress_arc_source = "pier_bridge.experiments.progress_arc"
            if isinstance(progress_raw.get("enabled"), bool):
                progress_arc_enabled = bool(progress_raw.get("enabled"))
            if isinstance(progress_raw.get("weight"), (int, float)):
                progress_arc_weight = float(progress_raw.get("weight"))
            if isinstance(progress_raw.get("shape"), str) and progress_raw.get("shape").strip():
                progress_arc_shape = str(progress_raw.get("shape")).strip().lower()
            if isinstance(progress_raw.get("tolerance"), (int, float)):
                progress_arc_tolerance = float(progress_raw.get("tolerance"))
            if isinstance(progress_raw.get("loss"), str) and progress_raw.get("loss").strip():
                progress_arc_loss = str(progress_raw.get("loss")).strip().lower()
            if isinstance(progress_raw.get("huber_delta"), (int, float)):
                progress_arc_huber_delta = float(progress_raw.get("huber_delta"))
            if isinstance(progress_raw.get("max_step"), (int, float)):
                progress_arc_max_step = float(progress_raw.get("max_step"))
            elif progress_raw.get("max_step") is None:
                progress_arc_max_step = None
            if isinstance(progress_raw.get("max_step_mode"), str) and progress_raw.get("max_step_mode").strip():
                progress_arc_max_step_mode = str(progress_raw.get("max_step_mode")).strip().lower()
            if isinstance(progress_raw.get("max_step_penalty"), (int, float)):
                progress_arc_max_step_penalty = float(progress_raw.get("max_step_penalty"))
            autoscale_raw = progress_raw.get("autoscale")
            if isinstance(autoscale_raw, dict):
                if isinstance(autoscale_raw.get("enabled"), bool):
                    progress_arc_autoscale_enabled = bool(autoscale_raw.get("enabled"))
                if isinstance(autoscale_raw.get("min_distance"), (int, float)):
                    progress_arc_autoscale_min_distance = float(autoscale_raw.get("min_distance"))
                if isinstance(autoscale_raw.get("distance_scale"), (int, float)):
                    progress_arc_autoscale_distance_scale = float(autoscale_raw.get("distance_scale"))
                if isinstance(autoscale_raw.get("per_step_scale"), bool):
                    progress_arc_autoscale_per_step_scale = bool(autoscale_raw.get("per_step_scale"))

    experiments_allowed = bool(dry_run or (audit_cfg and audit_cfg.enabled))
    if experiment_enabled and not experiments_allowed:
        logger.info(
            "Pier-bridge experiment bridge scoring ignored outside dry-run/audit."
        )
        experiment_enabled = False
    if progress_arc_enabled and progress_arc_source == "pier_bridge.experiments.progress_arc" and not experiments_allowed:
        logger.info(
            "Pier-bridge experiment progress arc ignored outside dry-run/audit."
        )
        progress_arc_enabled = False

    if experiment_enabled:
        pb_cfg = replace(
            pb_cfg,
            experiment_bridge_scoring_enabled=True,
            experiment_bridge_min_weight=float(experiment_min_weight),
            experiment_bridge_balance_weight=float(experiment_balance_weight),
        )
        logger.info(
            "Pier-bridge experiment: bridge scoring enabled (min_weight=%.2f balance_weight=%.2f)",
            float(experiment_min_weight),
            float(experiment_balance_weight),
        )
    # Apply progress_arc override (can enable or disable)
    if progress_arc_source is not None:
        pb_cfg = replace(
            pb_cfg,
            progress_arc_enabled=progress_arc_enabled,
            progress_arc_weight=float(progress_arc_weight),
            progress_arc_shape=str(progress_arc_shape),
            progress_arc_tolerance=float(progress_arc_tolerance),
            progress_arc_loss=str(progress_arc_loss),
            progress_arc_huber_delta=float(progress_arc_huber_delta),
            progress_arc_max_step=progress_arc_max_step,
            progress_arc_max_step_mode=str(progress_arc_max_step_mode),
            progress_arc_max_step_penalty=float(progress_arc_max_step_penalty),
            progress_arc_autoscale_enabled=bool(progress_arc_autoscale_enabled),
            progress_arc_autoscale_min_distance=float(progress_arc_autoscale_min_distance),
            progress_arc_autoscale_distance_scale=float(progress_arc_autoscale_distance_scale),
            progress_arc_autoscale_per_step_scale=bool(progress_arc_autoscale_per_step_scale),
        )
        if progress_arc_enabled:
            logger.info(
                "Pier-bridge progress arc enabled (weight=%.2f shape=%s source=%s)",
                float(progress_arc_weight),
                str(progress_arc_shape),
                str(progress_arc_source),
            )
        else:
            logger.info("Pier-bridge progress arc disabled by override (source=%s)", str(progress_arc_source))

    dj_raw = pb_overrides.get("dj_bridging")
    if isinstance(dj_raw, dict):
        dj_enabled = bool(dj_raw.get("enabled", pb_cfg.dj_bridging_enabled))
        seed_ordering = dj_raw.get("seed_ordering", pb_cfg.dj_seed_ordering)
        route_shape = dj_raw.get("route_shape", pb_cfg.dj_route_shape)
        pooling_raw = dj_raw.get("pooling")
        pool_strategy = pb_cfg.dj_pooling_strategy
        anchors_raw = dj_raw.get("anchors")
        anchors_must_include_all = pb_cfg.dj_anchors_must_include_all
        if isinstance(anchors_raw, dict) and isinstance(anchors_raw.get("must_include_all"), bool):
            anchors_must_include_all = bool(anchors_raw.get("must_include_all"))
        waypoint_weight = pb_cfg.dj_waypoint_weight
        waypoint_floor = pb_cfg.dj_waypoint_floor
        waypoint_penalty = pb_cfg.dj_waypoint_penalty
        waypoint_tie_break_band = pb_cfg.dj_waypoint_tie_break_band
        waypoint_cap = pb_cfg.dj_waypoint_cap
        seed_ordering_weight_sonic = pb_cfg.dj_seed_ordering_weight_sonic
        seed_ordering_weight_genre = pb_cfg.dj_seed_ordering_weight_genre
        seed_ordering_weight_bridge = pb_cfg.dj_seed_ordering_weight_bridge
        pool_k_local = pb_cfg.dj_pooling_k_local
        pool_k_toward = pb_cfg.dj_pooling_k_toward
        pool_k_genre = pb_cfg.dj_pooling_k_genre
        pool_union_max = pb_cfg.dj_pooling_k_union_max
        pool_step_stride = pb_cfg.dj_pooling_step_stride
        pool_cache_enabled = pb_cfg.dj_pooling_cache_enabled
        pool_debug_compare_baseline = pb_cfg.dj_pooling_debug_compare_baseline
        allow_detours_when_far = pb_cfg.dj_allow_detours_when_far
        far_threshold_sonic = pb_cfg.dj_far_threshold_sonic
        far_threshold_genre = pb_cfg.dj_far_threshold_genre
        far_threshold_connector = pb_cfg.dj_far_threshold_connector_scarcity
        connector_bias_enabled = pb_cfg.dj_connector_bias_enabled
        connector_max_linear = pb_cfg.dj_connector_max_per_segment_linear
        connector_max_adventurous = pb_cfg.dj_connector_max_per_segment_adventurous
        ladder_top_labels = pb_cfg.dj_ladder_top_labels
        ladder_min_label_weight = pb_cfg.dj_ladder_min_label_weight
        ladder_min_similarity = pb_cfg.dj_ladder_min_similarity
        ladder_max_steps = pb_cfg.dj_ladder_max_steps
        ladder_use_smoothed = pb_cfg.dj_ladder_use_smoothed_waypoint_vectors
        ladder_smooth_top_k = pb_cfg.dj_ladder_smooth_top_k
        ladder_smooth_min_sim = pb_cfg.dj_ladder_smooth_min_sim
        waypoint_fallback_k = pb_cfg.dj_waypoint_fallback_k
        micro_piers_enabled = pb_cfg.dj_micro_piers_enabled
        micro_piers_max = pb_cfg.dj_micro_piers_max
        micro_piers_topk = pb_cfg.dj_micro_piers_topk
        micro_piers_candidate_source = pb_cfg.dj_micro_piers_candidate_source
        micro_piers_selection_metric = pb_cfg.dj_micro_piers_selection_metric
        relax_enabled = pb_cfg.dj_relaxation_enabled
        relax_max_attempts = pb_cfg.dj_relaxation_max_attempts
        relax_emit_warnings = pb_cfg.dj_relaxation_emit_warnings
        relax_allow_floor = pb_cfg.dj_relaxation_allow_floor_relaxation
        route_raw = dj_raw.get("route")
        if isinstance(route_raw, dict):
            if isinstance(route_raw.get("shape"), str):
                route_shape = str(route_raw.get("shape"))
            if isinstance(route_raw.get("max_hops"), int):
                ladder_max_steps = int(route_raw.get("max_hops"))
            if isinstance(route_raw.get("top_n_genres"), int):
                ladder_top_labels = int(route_raw.get("top_n_genres"))
            if isinstance(route_raw.get("min_genre_weight"), (int, float)):
                ladder_min_label_weight = float(route_raw.get("min_genre_weight"))
            if isinstance(route_raw.get("use_similarity_smoothed_waypoints"), bool):
                ladder_use_smoothed = bool(route_raw.get("use_similarity_smoothed_waypoints"))
        if isinstance(dj_raw.get("waypoint_weight"), (int, float)):
            waypoint_weight = float(dj_raw.get("waypoint_weight"))
        if isinstance(dj_raw.get("waypoint_floor"), (int, float)):
            waypoint_floor = float(dj_raw.get("waypoint_floor"))
        if isinstance(dj_raw.get("waypoint_penalty"), (int, float)):
            waypoint_penalty = float(dj_raw.get("waypoint_penalty"))
        if isinstance(dj_raw.get("waypoint_tie_break_band"), (int, float)):
            waypoint_tie_break_band = float(dj_raw.get("waypoint_tie_break_band"))
        elif "waypoint_tie_break_band" in dj_raw and dj_raw.get("waypoint_tie_break_band") is None:
            waypoint_tie_break_band = None
        if isinstance(dj_raw.get("waypoint_cap"), (int, float)):
            waypoint_cap = float(dj_raw.get("waypoint_cap"))
        if isinstance(dj_raw.get("seed_ordering_weight_sonic"), (int, float)):
            seed_ordering_weight_sonic = float(dj_raw.get("seed_ordering_weight_sonic"))
        if isinstance(dj_raw.get("seed_ordering_weight_genre"), (int, float)):
            seed_ordering_weight_genre = float(dj_raw.get("seed_ordering_weight_genre"))
        if isinstance(dj_raw.get("seed_ordering_weight_bridge"), (int, float)):
            seed_ordering_weight_bridge = float(dj_raw.get("seed_ordering_weight_bridge"))
        if isinstance(pooling_raw, dict):
            if isinstance(pooling_raw.get("strategy"), str):
                pool_strategy = str(pooling_raw.get("strategy"))
            if isinstance(pooling_raw.get("k_local"), int):
                pool_k_local = int(pooling_raw.get("k_local"))
            if isinstance(pooling_raw.get("k_toward"), int):
                pool_k_toward = int(pooling_raw.get("k_toward"))
            if isinstance(pooling_raw.get("k_genre"), int):
                pool_k_genre = int(pooling_raw.get("k_genre"))
            if isinstance(pooling_raw.get("k_union_max"), int):
                pool_union_max = int(pooling_raw.get("k_union_max"))
            if isinstance(pooling_raw.get("step_stride"), int):
                pool_step_stride = int(pooling_raw.get("step_stride"))
            if isinstance(pooling_raw.get("cache_enabled"), bool):
                pool_cache_enabled = bool(pooling_raw.get("cache_enabled"))
            if isinstance(pooling_raw.get("debug_compare_baseline"), bool):
                pool_debug_compare_baseline = bool(
                    pooling_raw.get("debug_compare_baseline")
                )
        # Fallback: flat-key alias for backward compatibility
        # (deprecated; prefer nested dj_bridging.pooling.strategy)
        if isinstance(dj_raw.get("dj_pooling_strategy"), str):
            flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
            # Only use flat key if nested pooling.strategy was not set
            if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
                pool_strategy = flat_key_strategy
                logger.warning(
                    "DJ bridging: flat key 'dj_pooling_strategy=%s' is deprecated; "
                    "use nested 'dj_bridging.pooling.strategy' instead",
                    flat_key_strategy
                )
        if isinstance(dj_raw.get("allow_detours_when_far"), bool):
            allow_detours_when_far = bool(dj_raw.get("allow_detours_when_far"))
        far_raw = dj_raw.get("far_thresholds")
        if isinstance(far_raw, dict):
            if isinstance(far_raw.get("sonic"), (int, float)):
                far_threshold_sonic = float(far_raw.get("sonic"))
            if isinstance(far_raw.get("genre"), (int, float)):
                far_threshold_genre = float(far_raw.get("genre"))
            if isinstance(far_raw.get("connector_scarcity"), (int, float)):
                far_threshold_connector = float(far_raw.get("connector_scarcity"))
        connector_raw = dj_raw.get("connector_bias")
        if isinstance(connector_raw, dict):
            if isinstance(connector_raw.get("enabled"), bool):
                connector_bias_enabled = bool(connector_raw.get("enabled"))
            if isinstance(connector_raw.get("max_per_segment_linear"), int):
                connector_max_linear = int(connector_raw.get("max_per_segment_linear"))
            if isinstance(connector_raw.get("max_per_segment_adventurous"), int):
                connector_max_adventurous = int(connector_raw.get("max_per_segment_adventurous"))
        connectors_raw = dj_raw.get("connectors")
        if isinstance(connectors_raw, dict):
            if isinstance(connectors_raw.get("enabled"), bool):
                connector_bias_enabled = bool(connectors_raw.get("enabled"))
            if isinstance(connectors_raw.get("max_connectors"), int):
                connector_max_linear = int(connectors_raw.get("max_connectors"))
                connector_max_adventurous = int(connectors_raw.get("max_connectors"))
            if isinstance(connectors_raw.get("use_only_when_far"), bool):
                allow_detours_when_far = bool(
                    connectors_raw.get("use_only_when_far")
                )
            if isinstance(connectors_raw.get("far_threshold"), (int, float)):
                far_threshold_sonic = float(connectors_raw.get("far_threshold"))
                far_threshold_genre = float(connectors_raw.get("far_threshold"))
            far_overrides = connectors_raw.get("far_thresholds")
            if isinstance(far_overrides, dict):
                if isinstance(far_overrides.get("sonic"), (int, float)):
                    far_threshold_sonic = float(far_overrides.get("sonic"))
                if isinstance(far_overrides.get("genre"), (int, float)):
                    far_threshold_genre = float(far_overrides.get("genre"))
                if isinstance(far_overrides.get("connector_scarcity"), (int, float)):
                    far_threshold_connector = float(
                        far_overrides.get("connector_scarcity")
                    )
        ladder_raw = dj_raw.get("ladder")
        if isinstance(ladder_raw, dict):
            if isinstance(ladder_raw.get("top_labels"), int):
                ladder_top_labels = int(ladder_raw.get("top_labels"))
            if isinstance(ladder_raw.get("min_label_weight"), (int, float)):
                ladder_min_label_weight = float(ladder_raw.get("min_label_weight"))
            if isinstance(ladder_raw.get("min_similarity"), (int, float)):
                ladder_min_similarity = float(ladder_raw.get("min_similarity"))
            if isinstance(ladder_raw.get("max_steps"), int):
                ladder_max_steps = int(ladder_raw.get("max_steps"))
            if isinstance(ladder_raw.get("use_smoothed_waypoint_vectors"), bool):
                ladder_use_smoothed = bool(ladder_raw.get("use_smoothed_waypoint_vectors"))
            if isinstance(ladder_raw.get("smooth_top_k"), int):
                ladder_smooth_top_k = int(ladder_raw.get("smooth_top_k"))
            if isinstance(ladder_raw.get("smooth_min_sim"), (int, float)):
                ladder_smooth_min_sim = float(ladder_raw.get("smooth_min_sim"))
        relaxation_raw = dj_raw.get("relaxation")
        if isinstance(relaxation_raw, dict):
            if isinstance(relaxation_raw.get("enabled"), bool):
                relax_enabled = bool(relaxation_raw.get("enabled"))
            if isinstance(relaxation_raw.get("max_attempts"), int):
                relax_max_attempts = int(relaxation_raw.get("max_attempts"))
            if isinstance(relaxation_raw.get("emit_warnings"), bool):
                relax_emit_warnings = bool(relaxation_raw.get("emit_warnings"))
            if isinstance(relaxation_raw.get("allow_floor_relaxation"), bool):
                relax_allow_floor = bool(relaxation_raw.get("allow_floor_relaxation"))
        if isinstance(dj_raw.get("waypoint_fallback_k"), int):
            waypoint_fallback_k = int(dj_raw.get("waypoint_fallback_k"))
        micro_raw = dj_raw.get("micro_piers")
        if isinstance(micro_raw, dict):
            if isinstance(micro_raw.get("enabled"), bool):
                micro_piers_enabled = bool(micro_raw.get("enabled"))
            if isinstance(micro_raw.get("max"), int):
                micro_piers_max = int(micro_raw.get("max"))
            if isinstance(micro_raw.get("max_micro_piers_per_segment"), int):
                micro_piers_max = int(micro_raw.get("max_micro_piers_per_segment"))
            if isinstance(micro_raw.get("topk"), int):
                micro_piers_topk = int(micro_raw.get("topk"))
            if isinstance(micro_raw.get("top_k"), int):
                micro_piers_topk = int(micro_raw.get("top_k"))
            if isinstance(micro_raw.get("candidate_source"), str):
                micro_piers_candidate_source = str(micro_raw.get("candidate_source"))
            if isinstance(micro_raw.get("selection_metric"), str):
                micro_piers_selection_metric = str(micro_raw.get("selection_metric"))
        # DJ diagnostics (opt-in, default false)
        diagnostics_rank_impact_enabled = pb_cfg.dj_diagnostics_waypoint_rank_impact_enabled
        diagnostics_rank_sample_steps = pb_cfg.dj_diagnostics_waypoint_rank_sample_steps
        diagnostics_pool_verbose = pb_cfg.dj_diagnostics_pool_verbose  # Phase 3 fix
        diagnostics_raw = dj_raw.get("diagnostics")
        if isinstance(diagnostics_raw, dict):
            if isinstance(diagnostics_raw.get("waypoint_rank_impact_enabled"), bool):
                diagnostics_rank_impact_enabled = bool(diagnostics_raw.get("waypoint_rank_impact_enabled"))
            if isinstance(diagnostics_raw.get("waypoint_rank_sample_steps"), int):
                diagnostics_rank_sample_steps = int(diagnostics_raw.get("waypoint_rank_sample_steps"))
            if isinstance(diagnostics_raw.get("pool_verbose"), bool):  # Phase 3 fix
                diagnostics_pool_verbose = bool(diagnostics_raw.get("pool_verbose"))
        # Phase 2: Vector mode + IDF + Coverage
        ladder_target_mode = pb_cfg.dj_ladder_target_mode
        genre_vector_source = pb_cfg.dj_genre_vector_source
        genre_use_idf = pb_cfg.dj_genre_use_idf
        genre_idf_power = pb_cfg.dj_genre_idf_power
        genre_idf_norm = pb_cfg.dj_genre_idf_norm
        genre_use_coverage = pb_cfg.dj_genre_use_coverage
        genre_coverage_top_k = pb_cfg.dj_genre_coverage_top_k
        genre_coverage_weight = pb_cfg.dj_genre_coverage_weight
        genre_coverage_power = pb_cfg.dj_genre_coverage_power
        genre_presence_threshold = pb_cfg.dj_genre_presence_threshold
        if isinstance(dj_raw.get("dj_ladder_target_mode"), str):
            ladder_target_mode = str(dj_raw.get("dj_ladder_target_mode"))
        if isinstance(dj_raw.get("dj_genre_vector_source"), str):
            genre_vector_source = str(dj_raw.get("dj_genre_vector_source"))
        if isinstance(dj_raw.get("dj_genre_use_idf"), bool):
            genre_use_idf = bool(dj_raw.get("dj_genre_use_idf"))
        if isinstance(dj_raw.get("dj_genre_idf_power"), (int, float)):
            genre_idf_power = float(dj_raw.get("dj_genre_idf_power"))
        if isinstance(dj_raw.get("dj_genre_idf_norm"), str):
            genre_idf_norm = str(dj_raw.get("dj_genre_idf_norm"))
        if isinstance(dj_raw.get("dj_genre_use_coverage"), bool):
            genre_use_coverage = bool(dj_raw.get("dj_genre_use_coverage"))
        if isinstance(dj_raw.get("dj_genre_coverage_top_k"), int):
            genre_coverage_top_k = int(dj_raw.get("dj_genre_coverage_top_k"))
        if isinstance(dj_raw.get("dj_genre_coverage_weight"), (int, float)):
            genre_coverage_weight = float(dj_raw.get("dj_genre_coverage_weight"))
        if isinstance(dj_raw.get("dj_genre_coverage_power"), (int, float)):
            genre_coverage_power = float(dj_raw.get("dj_genre_coverage_power"))
        if isinstance(dj_raw.get("dj_genre_presence_threshold"), (int, float)):
            genre_presence_threshold = float(dj_raw.get("dj_genre_presence_threshold"))

        # Phase 3: Waypoint delta mode + squashing + coverage enhancements
        waypoint_delta_mode = pb_cfg.dj_waypoint_delta_mode
        waypoint_centered_baseline = pb_cfg.dj_waypoint_centered_baseline
        waypoint_squash = pb_cfg.dj_waypoint_squash
        waypoint_squash_alpha = pb_cfg.dj_waypoint_squash_alpha
        coverage_presence_source = pb_cfg.dj_coverage_presence_source
        coverage_mode = pb_cfg.dj_coverage_mode
        if isinstance(dj_raw.get("dj_waypoint_delta_mode"), str):
            waypoint_delta_mode = str(dj_raw.get("dj_waypoint_delta_mode"))
        if isinstance(dj_raw.get("dj_waypoint_centered_baseline"), str):
            waypoint_centered_baseline = str(dj_raw.get("dj_waypoint_centered_baseline"))
        if isinstance(dj_raw.get("dj_waypoint_squash"), str):
            waypoint_squash = str(dj_raw.get("dj_waypoint_squash"))
        if isinstance(dj_raw.get("dj_waypoint_squash_alpha"), (int, float)):
            waypoint_squash_alpha = float(dj_raw.get("dj_waypoint_squash_alpha"))
        if isinstance(dj_raw.get("dj_coverage_presence_source"), str):
            coverage_presence_source = str(dj_raw.get("dj_coverage_presence_source"))
        if isinstance(dj_raw.get("dj_coverage_mode"), str):
            coverage_mode = str(dj_raw.get("dj_coverage_mode"))

        pb_cfg = replace(
            pb_cfg,
            dj_bridging_enabled=bool(dj_enabled),
            dj_seed_ordering=str(seed_ordering),
            dj_route_shape=str(route_shape),
            dj_anchors_must_include_all=bool(anchors_must_include_all),
            dj_waypoint_weight=float(waypoint_weight),
            dj_waypoint_floor=float(waypoint_floor),
            dj_waypoint_penalty=float(waypoint_penalty),
            dj_waypoint_tie_break_band=waypoint_tie_break_band,
            dj_waypoint_cap=float(waypoint_cap),
            dj_seed_ordering_weight_sonic=float(seed_ordering_weight_sonic),
            dj_seed_ordering_weight_genre=float(seed_ordering_weight_genre),
            dj_seed_ordering_weight_bridge=float(seed_ordering_weight_bridge),
            dj_pooling_strategy=str(pool_strategy),
            dj_pooling_k_local=int(pool_k_local),
            dj_pooling_k_toward=int(pool_k_toward),
            dj_pooling_k_genre=int(pool_k_genre),
            dj_pooling_k_union_max=int(pool_union_max),
            dj_pooling_step_stride=int(pool_step_stride),
            dj_pooling_cache_enabled=bool(pool_cache_enabled),
            dj_pooling_debug_compare_baseline=bool(pool_debug_compare_baseline),
            dj_allow_detours_when_far=bool(allow_detours_when_far),
            dj_far_threshold_sonic=float(far_threshold_sonic),
            dj_far_threshold_genre=float(far_threshold_genre),
            dj_far_threshold_connector_scarcity=float(far_threshold_connector),
            dj_connector_bias_enabled=bool(connector_bias_enabled),
            dj_connector_max_per_segment_linear=int(connector_max_linear),
            dj_connector_max_per_segment_adventurous=int(connector_max_adventurous),
            dj_ladder_top_labels=int(ladder_top_labels),
            dj_ladder_min_label_weight=float(ladder_min_label_weight),
            dj_ladder_min_similarity=float(ladder_min_similarity),
            dj_ladder_max_steps=int(ladder_max_steps),
            dj_ladder_use_smoothed_waypoint_vectors=bool(ladder_use_smoothed),
            dj_ladder_smooth_top_k=int(ladder_smooth_top_k),
            dj_ladder_smooth_min_sim=float(ladder_smooth_min_sim),
            dj_waypoint_fallback_k=int(waypoint_fallback_k),
            dj_micro_piers_enabled=bool(micro_piers_enabled),
            dj_micro_piers_max=int(micro_piers_max),
            dj_micro_piers_topk=int(micro_piers_topk),
            dj_micro_piers_candidate_source=str(micro_piers_candidate_source),
            dj_micro_piers_selection_metric=str(micro_piers_selection_metric),
            dj_relaxation_enabled=bool(relax_enabled),
            dj_relaxation_max_attempts=int(relax_max_attempts),
            dj_relaxation_emit_warnings=bool(relax_emit_warnings),
            dj_relaxation_allow_floor_relaxation=bool(relax_allow_floor),
            dj_diagnostics_waypoint_rank_impact_enabled=bool(diagnostics_rank_impact_enabled),
            dj_diagnostics_waypoint_rank_sample_steps=int(diagnostics_rank_sample_steps),
            dj_diagnostics_pool_verbose=bool(diagnostics_pool_verbose),  # Phase 3 fix
            dj_ladder_target_mode=str(ladder_target_mode),
            dj_genre_vector_source=str(genre_vector_source),
            dj_genre_use_idf=bool(genre_use_idf),
            dj_genre_idf_power=float(genre_idf_power),
            dj_genre_idf_norm=str(genre_idf_norm),
            dj_genre_use_coverage=bool(genre_use_coverage),
            dj_genre_coverage_top_k=int(genre_coverage_top_k),
            dj_genre_coverage_weight=float(genre_coverage_weight),
            dj_genre_coverage_power=float(genre_coverage_power),
            dj_genre_presence_threshold=float(genre_presence_threshold),
            # Phase 3 parameters
            dj_waypoint_delta_mode=str(waypoint_delta_mode),
            dj_waypoint_centered_baseline=str(waypoint_centered_baseline),
            dj_waypoint_squash=str(waypoint_squash),
            dj_waypoint_squash_alpha=float(waypoint_squash_alpha),
            dj_coverage_presence_source=str(coverage_presence_source),
            dj_coverage_mode=str(coverage_mode),
        )

    if genre_graph_source == "layered":
        raw_weight = genre_graph_raw.get("transition_weight", 0.15) if isinstance(genre_graph_raw, dict) else 0.15
        transition_weight = float(raw_weight) if isinstance(raw_weight, (int, float)) else 0.15
        pb_cfg = replace(
            pb_cfg,
            layered_transition_scoring_enabled=True,
            layered_transition_weight=max(0.0, float(transition_weight)),
            layered_transition_mode=str(cfg.mode),
            genre_steering_enabled=False,
            weight_genre=0.0,
            dj_bridging_enabled=False,
            dj_pooling_k_genre=0,
        )

    return pb_cfg, tuning, tuning_sources, transition_weights
