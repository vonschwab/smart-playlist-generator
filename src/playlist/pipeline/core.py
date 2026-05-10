from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import DSPipelineConfig, default_ds_config
from src.playlist.constructor import PlaylistResult  # Type only, no longer calling construct_playlist
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    PierBridgeResult,
    build_pier_bridge_playlist,
)
from src.playlist.artist_identity_resolver import ArtistIdentityConfig
from src.playlist.pipeline.audit_emitter import AuditEmitter
from src.playlist.pipeline.bundle_restrict import restrict_bundle
from src.playlist.pipeline.embedding_setup import setup_embedding
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides
from src.playlist.pipeline.pier_resolver import (
    dedupe_pool_by_track_key,
    resolve_pier_seeds,
)
from src.playlist.run_audit import (
    parse_infeasible_handling_config,
    parse_run_audit_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPipelineResult:
    track_ids: list[str]
    track_indices: list[int]
    stats: Dict[str, Any]
    params_requested: Dict[str, Any]
    params_effective: Dict[str, Any]


def _apply_overrides(cfg: DSPipelineConfig, overrides: Optional[dict]) -> DSPipelineConfig:
    if not overrides:
        return cfg
    cand_cfg = cfg.candidate
    cons_cfg = cfg.construct
    rep_cfg = cfg.repair

    if "candidate" in overrides:
        cand_updates = {**cand_cfg.__dict__, **overrides["candidate"]}
        cand_cfg = type(cand_cfg)(**cand_updates)
    if "construct" in overrides:
        cons_updates = {**cons_cfg.__dict__, **overrides["construct"]}
        cons_cfg = type(cons_cfg)(**cons_updates)
    if "repair" in overrides:
        rep_updates = {**rep_cfg.__dict__, **overrides["repair"]}
        rep_cfg = type(rep_cfg)(**rep_updates)

    return replace(cfg, candidate=cand_cfg, construct=cons_cfg, repair=rep_cfg)


def _params_from_config(cfg: DSPipelineConfig) -> Dict[str, Any]:
    return {
        "mode": cfg.mode,
        "candidate": cfg.candidate.__dict__,
        "construct": cfg.construct.__dict__,
        "repair": cfg.repair.__dict__,
    }


def enforce_allowed_invariant(track_ids: list[str], allowed: set[str], context: str = "") -> None:
    """Raise if any track_id is not within allowed set."""
    out = [tid for tid in track_ids if tid not in allowed]
    if out:
        raise ValueError(f"Out-of-pool tracks detected {context}: {out[:5]}")


def generate_playlist_ds(
    *,
    artifact_path: str | Path,
    seed_track_id: str,
    num_tracks: int,
    mode: str,
    random_seed: int,
    overrides: Optional[dict] = None,
    allowed_track_ids: Optional[list[str]] = None,
    excluded_track_ids: Optional[set[str]] = None,
    single_artist: bool = False,
    sonic_variant: Optional[str] = None,
    anchor_seed_ids: Optional[List[str]] = None,
    pier_bridge_config: Optional[PierBridgeConfig] = None,
    # Hybrid-level tuning dials (not part of DSPipelineConfig)
    sonic_weight: Optional[float] = None,
    genre_weight: Optional[float] = None,
    min_genre_similarity: Optional[float] = None,
    genre_method: Optional[str] = None,
    allowed_track_ids_set: Optional[set[str]] = None,
    internal_connector_ids: Optional[List[str]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    # Optional pier-bridge infeasible handling + audit context (CLI/GUI)
    dry_run: bool = False,
    pool_source: Optional[str] = None,
    artist_style_enabled: bool = False,
    artist_playlist: bool = False,
    audit_context_extra: Optional[Dict[str, Any]] = None,
) -> DSPipelineResult:
    """
    Orchestrate:
    - load_artifact_bundle
    - build embedding (use smoothed genres for embedding)
    - build candidate pool
    - construct playlist via pier-bridge strategy (no repair pass)
    Returns ordered track_ids and stats.

    Playlist construction uses pier-bridge strategy exclusively:
    - Multiple seeds: seeds become fixed "piers", bridge segments connect them
    - Single seed: seed acts as both start and end pier (arc structure)
    - Artist playlists can optionally enforce "seed artist = piers only"

    Optional hybrid-level tuning dials:
    - sonic_weight, genre_weight: control hybrid embedding balance
    - min_genre_similarity: gate threshold
    - genre_method: which genre similarity algorithm to use
    """
    # Normalize optional lists to avoid NoneType len issues downstream.
    anchor_seed_ids = list(dict.fromkeys(anchor_seed_ids or []))

    # Optional pier-bridge run audit + infeasible handling configs (default OFF).
    pb_overrides = (overrides or {}).get("pier_bridge", {}) if isinstance(overrides, dict) else {}
    audit_cfg = parse_run_audit_config(pb_overrides.get("audit_run"))
    infeasible_cfg = parse_infeasible_handling_config(pb_overrides.get("infeasible_handling"))
    audit = AuditEmitter(audit_cfg)

    # Parse artist identity config from constraints.artist_identity
    constraints_overrides = (overrides or {}).get("constraints", {}) if isinstance(overrides, dict) else {}
    artist_identity_overrides = constraints_overrides.get("artist_identity", {})
    if isinstance(artist_identity_overrides, dict):
        # Parse config fields with defaults
        artist_identity_cfg = ArtistIdentityConfig(
            enabled=bool(artist_identity_overrides.get("enabled", False)),
            split_delimiters=list(artist_identity_overrides.get("split_delimiters", [])) if artist_identity_overrides.get("split_delimiters") else None or ArtistIdentityConfig().split_delimiters,
            strip_trailing_ensemble_terms=bool(artist_identity_overrides.get("strip_trailing_ensemble_terms", True)),
            trailing_ensemble_terms=list(artist_identity_overrides.get("trailing_ensemble_terms", [])) if artist_identity_overrides.get("trailing_ensemble_terms") else None or ArtistIdentityConfig().trailing_ensemble_terms,
        )
    else:
        # No config provided or invalid: use defaults (disabled)
        artist_identity_cfg = ArtistIdentityConfig()

    if artist_identity_cfg.enabled:
        logger.info("Artist identity resolution enabled for min_gap enforcement")

    bundle = load_artifact_bundle(artifact_path)
    if seed_track_id not in bundle.track_id_to_index:
        raise ValueError(f"Seed track_id not found in artifact: {seed_track_id}")
    seed_idx = bundle.track_id_to_index[seed_track_id]

    # Log requested tuning dials for verification
    if any([sonic_weight, genre_weight, min_genre_similarity, genre_method]):
        logger.info(
            "DS pipeline tuning dials: sonic_weight=%s, genre_weight=%s, min_genre_sim=%s, genre_method=%s",
            sonic_weight, genre_weight, min_genre_similarity, genre_method
        )

    # Restrict bundle to allowed_track_ids / excluded_track_ids when provided.
    bundle, seed_idx, allowed_track_ids_set = restrict_bundle(
        bundle,
        seed_track_id,
        seed_idx,
        anchor_seed_ids=anchor_seed_ids,
        allowed_track_ids=allowed_track_ids,
        excluded_track_ids=excluded_track_ids,
        allowed_track_ids_set=allowed_track_ids_set,
    )

    internal_connector_indices: Optional[Set[int]] = None
    if internal_connector_ids:
        internal_connector_indices = set()
        for tid in internal_connector_ids:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                internal_connector_indices.add(idx)

    playlist_len = min(num_tracks, bundle.track_ids.shape[0])
    # Pass config.yaml overrides to default_ds_config for initial config creation
    cfg = default_ds_config(mode, playlist_len=playlist_len, overrides=overrides)
    if single_artist:
        # Disable artist cap for single-artist runs
        cfg = replace(cfg, construct=replace(cfg.construct, max_artist_fraction_final=1.0))
    # Also apply any runtime overrides (for backward compatibility with nested structure)
    cfg = _apply_overrides(cfg, overrides)

    # Extract transition weights from overrides (for constructor)
    transition_weights = None
    if overrides and overrides.get("transition_weights"):
        tw = overrides["transition_weights"]
        if isinstance(tw, (list, tuple)) and len(tw) == 3:
            transition_weights = tuple(tw)
        elif isinstance(tw, dict):
            transition_weights = (
                tw.get("rhythm", 0.4),
                tw.get("timbre", 0.35),
                tw.get("harmony", 0.25),
            )

    embedding = setup_embedding(
        bundle,
        seed_track_id,
        seed_idx,
        anchor_seed_ids=anchor_seed_ids,
        sonic_variant=sonic_variant,
        mode=mode,
        cfg=cfg,
        sonic_weight=sonic_weight,
        genre_weight=genre_weight,
        min_genre_similarity=min_genre_similarity,
        random_seed=random_seed,
    )
    # The orchestrator still needs these by name for the remaining phases.
    min_genre_similarity = embedding.min_genre_similarity
    resolved_variant = embedding.resolved_variant
    variant_stats = embedding.variant_stats
    broad_filters = embedding.broad_filters
    genre_vocab = embedding.genre_vocab
    X_genre_raw = embedding.X_genre_raw
    X_genre_smoothed = embedding.X_genre_smoothed

    pool = build_candidate_pool(
        seed_idx=seed_idx,
        seed_indices=embedding.seed_indices_for_floor,
        embedding=embedding.embedding_model.embedding,
        artist_keys=bundle.artist_keys,
        track_ids=bundle.track_ids,
        track_titles=bundle.track_titles,
        track_artists=bundle.track_artists,
        durations_ms=bundle.durations_ms,
        cfg=cfg.candidate,
        random_seed=random_seed,
        X_sonic=embedding.X_sonic_for_embed,
        # NEW: genre-based candidate filtering (skip when min_genre_similarity is None)
        X_genre_raw=X_genre_raw if min_genre_similarity is not None else None,
        X_genre_smoothed=X_genre_smoothed if min_genre_similarity is not None else None,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method or "ensemble",
        genre_vocab=genre_vocab,
        broad_filters=broad_filters,
        mode=mode,
    )
    pool.stats["target_length"] = num_tracks

    max_per_artist = max(1, math.ceil(playlist_len * cfg.construct.max_artist_fraction_final))
    logger.info(
        "PIPELINE STRATEGY: pier-bridge (anchor_seed_ids=%s)",
        len(anchor_seed_ids),
    )

    playlist: PlaylistResult

    # ═══════════════════════════════════════════════════════════════════════════
    # PIER + BRIDGE STRATEGY (always used)
    # - Multiple seeds: seeds as fixed piers, beam-search bridges between
    # - Single seed: seed acts as both start and end pier (arc structure)
    # - No repair pass
    # ═══════════════════════════════════════════════════════════════════════════
    if True:  # Always use pier-bridge
        # ─── PIER BRIDGE PATH: No construct_playlist, no repair ───
        logger.info("ENTERING PIER-BRIDGE PATH with %d anchor seeds", len(anchor_seed_ids))

        pier_seed_indices, seed_track_ids_for_pier = resolve_pier_seeds(
            bundle, seed_idx, anchor_seed_ids,
        )

        # Pier-bridge handles any number of seeds (including 1 seed as arc structure)
        if True:
            # Deduplicate candidate pool by (artist, title), keeping canonical version.
            pool_indices = list(getattr(pool, "eligible_indices", pool.pool_indices))
            pool_indices = dedupe_pool_by_track_key(bundle, pool_indices)

            audit.ensure_context(
                bundle=bundle,
                seed_idx=seed_idx,
                seed_track_id=seed_track_id,
                mode=cfg.mode,
                dry_run=dry_run,
                artifact_path=artifact_path,
                sonic_variant=resolved_variant,
                allowed_ids_count=int(len(allowed_track_ids_set or set())),
                pool_source=pool_source,
                artist_style_enabled=artist_style_enabled,
                artist_playlist=artist_playlist,
                audit_context_extra=audit_context_extra,
            )

            pb_cfg, tuning, tuning_sources, transition_weights = apply_pier_bridge_overrides(
                pier_bridge_config=pier_bridge_config,
                cfg=cfg,
                overrides=overrides,
                pb_overrides=pb_overrides,
                artist_playlist=artist_playlist,
                dry_run=dry_run,
                audit_cfg=audit_cfg,
                resolved_variant=resolved_variant,
            )

            logger.info(
                "Pier-bridge segment policy: artist_playlist=%s strategy=%s pool_max=%d progress=%s disallow_seed_artist_in_interiors=%s disallow_pier_artists_in_interiors=%s",
                bool(artist_playlist),
                str(pb_cfg.segment_pool_strategy),
                int(pb_cfg.segment_pool_max),
                bool(pb_cfg.progress_enabled),
                bool(pb_cfg.disallow_seed_artist_in_interiors),
                bool(pb_cfg.disallow_pier_artists_in_interiors),
            )
            if audit.active and not audit.has_kind("preflight"):
                pool_summary = {
                    "allowed_ids_count": int(len(allowed_track_ids_set or set())),
                    "piers_count": int(len(seed_track_ids_for_pier)),
                    "internal_connectors_count": int(len(internal_connector_indices or set())),
                    "candidate_pool_stats": dict(pool.stats or {}),
                    "candidate_pool_indices_after_dedupe": int(len(pool_indices)),
                }
                style_summary = None
                if isinstance(audit_context_extra, dict):
                    style_summary = audit_context_extra.get("style_summary")

                excluded_ids_set = {str(t) for t in (excluded_track_ids or set())}
                pier_ids_set = {str(t) for t in seed_track_ids_for_pier}
                exempt_piers_count = int(len(excluded_ids_set & pier_ids_set))
                recency_lookback_days = None
                if isinstance(audit_context_extra, dict):
                    recency_raw = audit_context_extra.get("recency")
                    if isinstance(recency_raw, dict) and isinstance(recency_raw.get("lookback_days"), (int, float)):
                        recency_lookback_days = int(recency_raw.get("lookback_days"))
                audit.append(
                    "preflight",
                    {
                            "tuning": {
                                "mode": str(cfg.mode),
                                "transition_floor": float(tuning.transition_floor),
                                "bridge_floor": float(tuning.bridge_floor),
                                "weight_bridge": float(tuning.weight_bridge),
                                "weight_transition": float(tuning.weight_transition),
                                "genre_tiebreak_weight": float(tuning.genre_tiebreak_weight),
                                "genre_penalty_threshold": float(tuning.genre_penalty_threshold),
                                "genre_penalty_strength": float(tuning.genre_penalty_strength),
                                "genre_tie_break_band": (
                                    float(pb_cfg.genre_tie_break_band)
                                    if pb_cfg.genre_tie_break_band is not None
                                    else None
                                ),
                                "segment_pool_strategy": str(pb_cfg.segment_pool_strategy),
                                "segment_pool_max": int(pb_cfg.segment_pool_max),
                                "max_segment_pool_max": int(pb_cfg.max_segment_pool_max),
                                "progress": {
                                    "enabled": bool(pb_cfg.progress_enabled),
                                    "monotonic_epsilon": float(pb_cfg.progress_monotonic_epsilon),
                                    "penalty_weight": float(pb_cfg.progress_penalty_weight),
                                },
                                "experiments": {
                                    "bridge_scoring": {
                                        "enabled": bool(pb_cfg.experiment_bridge_scoring_enabled),
                                        "min_weight": float(pb_cfg.experiment_bridge_min_weight),
                                        "balance_weight": float(pb_cfg.experiment_bridge_balance_weight),
                                    },
                                },
                                "progress_arc": {
                                    "enabled": bool(pb_cfg.progress_arc_enabled),
                                    "weight": float(pb_cfg.progress_arc_weight),
                                    "shape": str(pb_cfg.progress_arc_shape),
                                    "tolerance": float(pb_cfg.progress_arc_tolerance),
                                    "loss": str(pb_cfg.progress_arc_loss),
                                    "huber_delta": float(pb_cfg.progress_arc_huber_delta),
                                    "max_step": (float(pb_cfg.progress_arc_max_step) if pb_cfg.progress_arc_max_step is not None else None),
                                    "max_step_mode": str(pb_cfg.progress_arc_max_step_mode),
                                    "max_step_penalty": float(pb_cfg.progress_arc_max_step_penalty),
                                    "autoscale": {
                                        "enabled": bool(pb_cfg.progress_arc_autoscale_enabled),
                                        "min_distance": float(pb_cfg.progress_arc_autoscale_min_distance),
                                        "distance_scale": float(pb_cfg.progress_arc_autoscale_distance_scale),
                                        "per_step_scale": bool(pb_cfg.progress_arc_autoscale_per_step_scale),
                                    },
                                },
                                "disallow_seed_artist_in_interiors": bool(pb_cfg.disallow_seed_artist_in_interiors),
                                "disallow_pier_artists_in_interiors": bool(pb_cfg.disallow_pier_artists_in_interiors),
                                "infeasible_handling": {
                                    "enabled": bool(infeasible_cfg.enabled),
                                    "strategy": str(infeasible_cfg.strategy),
                                    "min_bridge_floor": float(infeasible_cfg.min_bridge_floor),
                                    "backoff_steps": list(infeasible_cfg.backoff_steps),
                                    "max_attempts_per_segment": int(infeasible_cfg.max_attempts_per_segment),
                                    "widen_search_on_backoff": bool(infeasible_cfg.widen_search_on_backoff),
                                    "extra_neighbors_m": int(infeasible_cfg.extra_neighbors_m),
                                    "extra_bridge_helpers": int(infeasible_cfg.extra_bridge_helpers),
                                    "extra_beam_width": int(infeasible_cfg.extra_beam_width),
                                    "extra_expansion_attempts": int(infeasible_cfg.extra_expansion_attempts),
                                },
                                "audit_run": {
                                    "enabled": bool(audit_cfg.enabled),
                                    "out_dir": str(audit_cfg.out_dir),
                                    "include_top_k": int(audit_cfg.include_top_k),
                                    "max_bytes": int(audit_cfg.max_bytes),
                                    "write_on_success": bool(audit_cfg.write_on_success),
                                    "write_on_failure": bool(audit_cfg.write_on_failure),
                                },
                            },
                            "tuning_sources": dict(tuning_sources),
                            "pool_summary": pool_summary,
                            "ds_inputs": {
                                "allowed_ids_count": int(len(allowed_track_ids_set or set())),
                                "excluded_ids_count": int(len(excluded_ids_set)),
                            },
                            "recency": {
                                "lookback_days": recency_lookback_days,
                                "excluded_count": int(len(excluded_ids_set)),
                                "exempt_piers_count": exempt_piers_count,
                            },
                            "style_summary": style_summary,
                    },
                )
            pb_result: PierBridgeResult = build_pier_bridge_playlist(
                seed_track_ids=seed_track_ids_for_pier,
                total_tracks=playlist_len,
                bundle=bundle,
                candidate_pool_indices=pool_indices,
                cfg=pb_cfg,
                allowed_track_ids_set=set(allowed_track_ids) if allowed_track_ids else None,
                internal_connector_indices=internal_connector_indices,
                internal_connector_max_per_segment=internal_connector_max_per_segment,
                internal_connector_priority=internal_connector_priority,
                infeasible_handling=infeasible_cfg,
                audit_config=audit_cfg,
                audit_events=audit.events,
                artist_identity_cfg=artist_identity_cfg,
            )
            if not pb_result.success:
                # Build diagnostic error message
                pool_stats = pool.stats or {}
                failure_reason = str(pb_result.failure_reason or "pier-bridge failed")

                # Detect bottleneck type
                admitted = pool_stats.get("pool_size", len(pool_indices))
                rejected_sonic = pool_stats.get("below_sonic_similarity", 0)
                rejected_genre = pool_stats.get("below_genre_similarity", 0)
                total_considered = pool_stats.get("total_candidates_considered", 0)

                # Determine if genre gating is the bottleneck
                diagnostic_msg = failure_reason
                if admitted == 0 and rejected_genre > 0:
                    # Genre gating eliminated all candidates
                    passed_sonic = total_considered - rejected_sonic if total_considered > rejected_sonic else 0
                    diagnostic_msg = (
                        f"{failure_reason}\n\n"
                        f"🔍 Root cause: GENRE ISOLATION\n"
                        f"   - {passed_sonic} tracks passed sonic similarity\n"
                        f"   - ALL {rejected_genre} were rejected by genre filter (min_similarity={cfg.genre_gate_min_similarity:.2f})\n"
                        f"   - Zero candidates remained for playlist generation\n\n"
                        f"💡 This artist's genres are too isolated from your library.\n"
                        f"   Suggestions:\n"
                        f"   - Reduce genre mode from {cfg.mode} to 'narrow' or 'dynamic'\n"
                        f"   - Add more music in similar genres to your library\n"
                        f"   - Use genre mode 'discover' to disable hard genre filtering"
                    )
                elif admitted == 0 and rejected_sonic > 0:
                    # Sonic similarity eliminated all candidates
                    diagnostic_msg = (
                        f"{failure_reason}\n\n"
                        f"🔍 Root cause: SONIC ISOLATION\n"
                        f"   - {rejected_sonic} of {total_considered} tracks rejected by sonic floor\n"
                        f"   - Zero candidates remained for playlist generation\n\n"
                        f"💡 This artist's sound is too sonically isolated from your library.\n"
                        f"   Suggestions:\n"
                        f"   - Reduce sonic mode from {cfg.mode} to 'narrow' or 'dynamic'\n"
                        f"   - Add more sonically similar music to your library"
                    )
                elif admitted > 0 and admitted < 10:
                    # Small pool passed filters but not enough for bridging
                    diagnostic_msg = (
                        f"{failure_reason}\n\n"
                        f"🔍 Root cause: INSUFFICIENT CANDIDATE POOL\n"
                        f"   - Only {admitted} candidates passed all filters\n"
                        f"   - Rejected: {rejected_sonic} sonic, {rejected_genre} genre\n"
                        f"   - Need more candidates to bridge between {len(seed_track_ids_for_pier)} seeds\n\n"
                        f"💡 The candidate pool is too small for multi-seed bridging.\n"
                        f"   Suggestions:\n"
                        f"   - Use fewer seeds (1-2 instead of {len(seed_track_ids_for_pier)})\n"
                        f"   - Reduce filtering: set genre/sonic modes to 'discover'\n"
                        f"   - Choose seeds that are more sonically/genre similar to each other"
                    )

                if audit.can_flush():
                    audit.append(
                        "final_failure",
                        {
                            "failure_reason": failure_reason,
                            "diagnostic_msg": diagnostic_msg,
                            "segment_bridge_floors_used": (pb_result.stats or {}).get("segment_bridge_floors_used"),
                            "segment_backoff_attempts_used": (pb_result.stats or {}).get("segment_backoff_attempts_used"),
                            "pool_diagnostics": {
                                "admitted": admitted,
                                "rejected_sonic": rejected_sonic,
                                "rejected_genre": rejected_genre,
                                "total_considered": total_considered,
                            },
                        },
                    )
                    audit.flush()
                    raise ValueError(
                        f"{diagnostic_msg}\n\n(Detailed audit: {audit.path})"
                    )
                raise ValueError(diagnostic_msg)

            # 6. Log diagnostics
            logger.info(
                "Pier bridge result: %d tracks, %d segments, success=%s",
                len(pb_result.track_indices),
                len(pb_result.segment_diagnostics),
                pb_result.success,
            )
            if not pb_result.success:
                logger.warning("Pier bridge did not fully succeed: %s", pb_result.failure_reason)

            # 7. Build PlaylistResult-compatible output (no repair, no construct_playlist)
            # We create a minimal PlaylistResult to match the expected interface
            pb_track_indices = np.array(pb_result.track_indices, dtype=np.int32)

            # Edge scores (T) produced by pier-bridge builder (matches its scoring)
            edge_scores_list = list((pb_result.stats or {}).get("edge_scores") or [])
            t_vals = [
                float(e.get("T"))
                for e in edge_scores_list
                if isinstance(e, dict) and isinstance(e.get("T"), (int, float))
            ]
            below_floor_count = (
                sum(1 for v in t_vals if v < float(pb_cfg.transition_floor))
                if t_vals
                else 0
            )
            min_transition = float(min(t_vals)) if t_vals else None
            mean_transition = float(sum(t_vals) / len(t_vals)) if t_vals else None

            # Create minimal PlaylistResult
            playlist = PlaylistResult(
                track_indices=pb_track_indices,
                stats={
                    "strategy": "pier_bridge",
                    "num_segments": len(pb_result.segment_diagnostics),
                    "success": pb_result.success,
                    "failure_reason": pb_result.failure_reason,
                    "segment_diagnostics": [s.__dict__ for s in pb_result.segment_diagnostics],
                    "edge_scores": edge_scores_list,
                    "transition_floor": float(pb_cfg.transition_floor),
                    "transition_gamma": float(cfg.construct.transition_gamma),
                    "transition_centered": bool(pb_cfg.center_transitions),
                    "below_floor_count": below_floor_count,
                    "min_transition": min_transition,
                    "mean_transition": mean_transition,
                    "repair_applied": False,  # No repair in pier bridge mode
                    "warnings": (pb_result.stats or {}).get("warnings") or [],
                },
                params_requested={"strategy": "pier_bridge"},
                params_effective={"strategy": "pier_bridge", "pier_config": pb_cfg.__dict__},
            )

    # Legacy paths (anchor_builder, standard construct_playlist) have been removed.
    # All playlist construction now goes through pier-bridge.

    ordered_track_ids = [str(bundle.track_ids[i]) for i in playlist.track_indices]

    # Post-order validation only: DS ordering must be final (no post-filtering).
    expected_len = int(playlist_len)
    excluded_ids_set = {str(t) for t in (excluded_track_ids or set())}
    pier_ids_set = {str(t) for t in seed_track_ids_for_pier}
    recency_overlap_ids = [
        tid for tid in ordered_track_ids if (tid in excluded_ids_set and tid not in pier_ids_set)
    ]
    post_order_validation = {
        "recency_overlap_count": int(len(recency_overlap_ids)),
        "final_size": int(len(ordered_track_ids)),
        "expected_size": int(expected_len),
    }

    validation_errors: list[str] = []
    if expected_len > 0 and len(ordered_track_ids) != expected_len:
        validation_errors.append(f"length_mismatch final={len(ordered_track_ids)} expected={expected_len}")

    if recency_overlap_ids:
        offenders: list[str] = []
        for tid in recency_overlap_ids[:10]:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is None:
                offenders.append(str(tid))
                continue
            artist = ""
            title = ""
            try:
                if bundle.track_artists is not None:
                    artist = str(bundle.track_artists[int(idx)])
                elif bundle.artist_keys is not None:
                    artist = str(bundle.artist_keys[int(idx)])
                if bundle.track_titles is not None:
                    title = str(bundle.track_titles[int(idx)])
            except Exception:
                artist = ""
                title = ""
            offenders.append(f"{tid} ({artist} - {title})")
        validation_errors.append(f"recency_overlap={len(recency_overlap_ids)} offenders={offenders}")

    if validation_errors:
        msg = "post_order_validation_failed: " + " | ".join(validation_errors)
        if audit.can_flush():
            audit.append(
                "final_failure",
                {
                    "failure_reason": msg,
                    "post_order_filters_applied": [],
                    "post_order_validation": dict(post_order_validation),
                },
            )
            audit.flush()
            raise ValueError(f"{msg} (audit: {audit.path})")
        raise ValueError(msg)
    try:
        if allowed_track_ids_set is not None:
            logger.info(
                "Allowed pool enforcement: allowed=%d playlist=%d",
                len(allowed_track_ids_set),
                len(ordered_track_ids),
            )
            enforce_allowed_invariant(ordered_track_ids, allowed_track_ids_set, context="final_playlist")
    except Exception as exc:
        if audit.can_flush():
            audit.append(
                "final_failure",
                {
                    "failure_reason": f"allowed_set_invariant_failed: {exc}",
                },
            )
            audit.flush()
            raise ValueError(f"{exc} (audit: {audit.path})") from exc
        raise

    if audit.can_flush() and bool(audit_cfg.write_on_success):
        playlist_tracks: list[dict[str, Any]] = []
        for idx in playlist.track_indices.tolist():
            artist = ""
            title = ""
            try:
                if bundle.artist_keys is not None:
                    artist = str(bundle.artist_keys[int(idx)])
                if bundle.track_titles is not None:
                    title = str(bundle.track_titles[int(idx)])
            except Exception:
                artist = ""
                title = ""
            playlist_tracks.append(
                {
                    "track_id": str(bundle.track_ids[int(idx)]),
                    "artist": artist,
                    "title": title,
                }
            )

        weakest_edges: list[dict[str, Any]] = []
        try:
            weakest_edges = sorted(
                [e for e in (edge_scores_list or []) if isinstance(e, dict) and isinstance(e.get("T"), (int, float))],
                key=lambda e: float(e.get("T")),
            )[: min(10, len(edge_scores_list or []))]
        except Exception:
            weakest_edges = []

        audit.append(
            "final_success",
            {
                "post_order_filters_applied": [],
                "post_order_validation": dict(post_order_validation),
                "playlist_tracks": playlist_tracks,
                "weakest_edges": weakest_edges,
                "summary_stats": {
                    "final_playlist_size": int(len(ordered_track_ids)),
                    "transition_floor": float(getattr(pb_cfg, "transition_floor", 0.0)),
                    "below_floor_count": int(below_floor_count),
                    "min_transition": min_transition,
                    "mean_transition": mean_transition,
                    "segment_bridge_floors_used": (pb_result.stats or {}).get("segment_bridge_floors_used"),
                    "segment_backoff_attempts_used": (pb_result.stats or {}).get("segment_backoff_attempts_used"),
                    "soft_genre_penalty_hits": (pb_result.stats or {}).get("soft_genre_penalty_hits"),
                    "soft_genre_penalty_edges_scored": (pb_result.stats or {}).get("soft_genre_penalty_edges_scored"),
                    "post_order_filters_applied": [],
                    "post_order_validation": dict(post_order_validation),
                },
            },
        )
        audit.flush()
    params_requested = _params_from_config(cfg)
    if overrides:
        params_requested["overrides"] = overrides
    params_effective: Dict[str, Any] = {
        "sonic_variant": variant_stats,
        "embedding": embedding.embedding_model.params_effective,
        "candidate_pool": pool.params_effective,
        "playlist": playlist.params_effective,
    }
    stats: Dict[str, Any] = {
        "candidate_pool": pool.stats,
        "playlist": playlist.stats,
    }
    if audit.path is not None:
        try:
            stats.setdefault("playlist", {})
            stats["playlist"]["audit_path"] = str(audit.path)
        except Exception:
            pass

    return DSPipelineResult(
        track_ids=ordered_track_ids,
        track_indices=playlist.track_indices.tolist(),
        stats=stats,
        params_requested=params_requested,
        params_effective=params_effective,
    )


