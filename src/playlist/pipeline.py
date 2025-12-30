from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.features.artifacts import ArtifactBundle, get_sonic_matrix, load_artifact_bundle
from src.playlist.candidate_pool import CandidatePoolResult, build_candidate_pool
from src.playlist.config import DSPipelineConfig, default_ds_config
from src.playlist.constructor import PlaylistResult  # Type only, no longer calling construct_playlist
from src.playlist.pier_bridge_builder import PierBridgeConfig, PierBridgeResult, build_pier_bridge_playlist
from src.similarity.hybrid import HybridEmbeddingModel, build_hybrid_embedding
from src.similarity.sonic_variant import compute_sonic_variant_matrix, resolve_sonic_variant

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
    - Seed artist tracks can appear in bridges with same constraints as others

    Optional hybrid-level tuning dials:
    - sonic_weight, genre_weight: control hybrid embedding balance
    - min_genre_similarity: gate threshold
    - genre_method: which genre similarity algorithm to use
    """
    # Normalize optional lists to avoid NoneType len issues downstream.
    anchor_seed_ids = anchor_seed_ids or []

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

    # Restrict bundle to allowed_track_ids when provided
    if allowed_track_ids:
        allowed_indices = []
        for tid in allowed_track_ids:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                allowed_indices.append(idx)
        if seed_idx not in allowed_indices:
            allowed_indices.append(seed_idx)
        allowed_indices = sorted(set(allowed_indices))
        N_allowed = len(allowed_indices)
        if N_allowed == 0:
            raise ValueError("No allowed track_ids were found in artifact bundle.")
        if N_allowed > 10000:
            raise ValueError(f"Allowed track_id set too large ({N_allowed}); refusing DS run.")

        def _slice(arr):
            if arr is None:
                return None
            return arr[allowed_indices]

        bundle = ArtifactBundle(
            artifact_path=bundle.artifact_path,
            track_ids=bundle.track_ids[allowed_indices],
            artist_keys=bundle.artist_keys[allowed_indices],
            track_artists=_slice(bundle.track_artists),
            track_titles=_slice(bundle.track_titles),
            X_sonic=bundle.X_sonic[allowed_indices],
            X_sonic_start=_slice(bundle.X_sonic_start),
            X_sonic_mid=_slice(bundle.X_sonic_mid),
            X_sonic_end=_slice(bundle.X_sonic_end),
            X_genre_raw=bundle.X_genre_raw[allowed_indices],
            X_genre_smoothed=bundle.X_genre_smoothed[allowed_indices],
            genre_vocab=bundle.genre_vocab,
            track_id_to_index={str(tid): i for i, tid in enumerate(bundle.track_ids[allowed_indices])},
        )
        seed_idx = bundle.track_id_to_index[str(seed_track_id)]
        logger.info(
            "DS pipeline bundle restricted: N=%d X_sonic=%s X_genre_smoothed=%s",
            N_allowed,
            getattr(bundle.X_sonic, "shape", None),
            getattr(bundle.X_genre_smoothed, "shape", None),
        )
    elif excluded_track_ids:
        total_tracks = int(bundle.track_ids.shape[0])
        mask_keep = []
        applied_excluded = 0
        # Build set of exempt IDs (primary seed + all anchor seeds)
        exempt_ids = {str(seed_track_id)}
        if anchor_seed_ids:
            exempt_ids.update(str(sid) for sid in anchor_seed_ids)
        for tid in bundle.track_ids:
            sid = str(tid)
            if sid in exempt_ids:
                mask_keep.append(True)
                continue
            if sid in excluded_track_ids:
                applied_excluded += 1
                mask_keep.append(False)
                continue
            mask_keep.append(True)
        mask_keep = np.array(mask_keep, dtype=bool)
        if not np.any(mask_keep):
            raise ValueError("Excluded set removed all tracks; cannot run DS.")
        allowed_indices = np.nonzero(mask_keep)[0].tolist()
        if seed_idx not in allowed_indices:
            allowed_indices = sorted(set(allowed_indices + [seed_idx]))
        def _slice(arr):
            if arr is None:
                return None
            return arr[allowed_indices]
        bundle = ArtifactBundle(
            artifact_path=bundle.artifact_path,
            track_ids=bundle.track_ids[allowed_indices],
            artist_keys=bundle.artist_keys[allowed_indices],
            track_artists=_slice(bundle.track_artists),
            track_titles=_slice(bundle.track_titles),
            X_sonic=bundle.X_sonic[allowed_indices],
            X_sonic_start=_slice(bundle.X_sonic_start),
            X_sonic_mid=_slice(bundle.X_sonic_mid),
            X_sonic_end=_slice(bundle.X_sonic_end),
            X_genre_raw=bundle.X_genre_raw[allowed_indices],
            X_genre_smoothed=bundle.X_genre_smoothed[allowed_indices],
            genre_vocab=bundle.genre_vocab,
            track_id_to_index={str(tid): i for i, tid in enumerate(bundle.track_ids[allowed_indices])},
        )
        seed_idx = bundle.track_id_to_index[str(seed_track_id)]
        if os.environ.get("PLAYLIST_DIAG_RECENCY") or os.environ.get("PLAYLIST_DIAG_POOL"):
            logger.info(
                "DS candidate pool after exclusions: total=%d requested_excluded=%d applied_excluded=%d final_pool=%d",
                total_tracks,
                len(excluded_track_ids),
                applied_excluded,
                len(allowed_indices),
            )

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

    resolved_variant = resolve_sonic_variant(explicit_variant=sonic_variant, config_variant=None)
    variant_stats: dict[str, Any] = {"variant": resolved_variant}
    X_sonic_for_embed = bundle.X_sonic
    pre_scaled_sonic = False
    if bundle.X_sonic is not None:
        if getattr(bundle, "sonic_variant", None) == resolved_variant and getattr(bundle, "sonic_pre_scaled", False):
            X_sonic_for_embed = bundle.X_sonic
            variant_stats = {"variant": resolved_variant, "pre_scaled": True}
        else:
            X_sonic_for_embed, variant_stats = compute_sonic_variant_matrix(bundle.X_sonic, resolved_variant, l2=False)
    else:
        raise ValueError("Artifact missing X_sonic matrix.")
    # Warn if the transformed space is too flat (indicates bad artifact/variant)
    try:
        sims = np.dot(
            X_sonic_for_embed / (np.linalg.norm(X_sonic_for_embed, axis=1, keepdims=True) + 1e-12),
            (X_sonic_for_embed[seed_idx] / (np.linalg.norm(X_sonic_for_embed[seed_idx]) + 1e-12)).T,
        )
        p10 = float(np.percentile(sims, 10))
        p90 = float(np.percentile(sims, 90))
        if (p90 - p10) < 0.1:
            logger.warning(
                "Sonic space appears flat for seed %s (p90-p10=%.3f). Check artifact/variant (resolved=%s).",
                seed_track_id,
                p90 - p10,
                resolved_variant,
            )
    except Exception:
        logger.debug("Skipped flatness heuristic; unable to compute percentile diagnostics.", exc_info=True)
    pre_scaled_sonic = bool(variant_stats.get("pre_scaled", False))

    # Compute effective hybrid embedding weights
    # Default to balanced approach: 0.6 sonic / 0.4 genre
    effective_w_sonic = 0.6
    effective_w_genre = 0.4
    if mode == "sonic_only":
        min_genre_similarity = None
        effective_w_sonic = 1.0
        effective_w_genre = 0.0
        logger.info("Sonic-only mode: disabling genre gate and setting genre weight to 0.")

    if sonic_weight is not None and genre_weight is not None:
        # Both provided: use them directly (assume they sum to 1.0 or normalize)
        total = sonic_weight + genre_weight
        if total > 0:
            effective_w_sonic = sonic_weight / total
            effective_w_genre = genre_weight / total
        logger.info(
            "Applying hybrid embedding weights: w_sonic=%.3f, w_genre=%.3f (normalized from %.3f, %.3f)",
            effective_w_sonic, effective_w_genre, sonic_weight, genre_weight
        )
    elif sonic_weight is not None:
        # Only sonic weight provided: genre = 1 - sonic
        effective_w_sonic = sonic_weight
        effective_w_genre = 1.0 - sonic_weight
        logger.info(
            "Applying hybrid embedding weight: w_sonic=%.3f (w_genre=%.3f inferred)",
            effective_w_sonic, effective_w_genre
        )
    elif genre_weight is not None:
        # Only genre weight provided: sonic = 1 - genre
        effective_w_genre = genre_weight
        effective_w_sonic = 1.0 - genre_weight
        logger.info(
            "Applying hybrid embedding weight: w_genre=%.3f (w_sonic=%.3f inferred)",
            effective_w_genre, effective_w_sonic
        )

    embedding_model = build_hybrid_embedding(
        X_sonic_for_embed,
        bundle.X_genre_smoothed,
        n_components_sonic=32,
        n_components_genre=32,
        w_sonic=effective_w_sonic,
        w_genre=effective_w_genre,
        random_seed=random_seed,
        pre_scaled_sonic=pre_scaled_sonic,
        use_pca_sonic=not pre_scaled_sonic,
    )

    pool = build_candidate_pool(
        seed_idx=seed_idx,
        embedding=embedding_model.embedding,
        artist_keys=bundle.artist_keys,
        cfg=cfg.candidate,
        random_seed=random_seed,
        # NEW: genre-based candidate filtering (skip when min_genre_similarity is None)
        X_genre_raw=bundle.X_genre_raw if min_genre_similarity is not None else None,
        X_genre_smoothed=bundle.X_genre_smoothed if min_genre_similarity is not None else None,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method or "ensemble",
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
        from src.title_dedupe import normalize_title_for_dedupe, normalize_artist_key, calculate_version_preference_score

        # 1. Resolve seed IDs to indices
        pier_seed_indices: list[int] = []
        not_found = []
        for sid in anchor_seed_ids:
            idx = bundle.track_id_to_index.get(str(sid))
            if idx is not None:
                pier_seed_indices.append(idx)
            else:
                not_found.append(sid)
        if not_found:
            logger.warning(
                "Pier seeds NOT FOUND in bundle (%d/%d): %s",
                len(not_found),
                len(anchor_seed_ids),
                not_found[:3],
            )
        else:
            logger.info("All %d pier seeds found in bundle", len(anchor_seed_ids))

        # 2. Deduplicate seeds by title (keep first occurrence)
        pier_seed_indices = list(dict.fromkeys(pier_seed_indices))
        if bundle.track_titles is not None:
            seen_titles: set[str] = set()
            dedupe_indices: list[int] = []
            for idx in pier_seed_indices:
                title = bundle.track_titles[idx] or ""
                norm_title = normalize_title_for_dedupe(str(title), mode="loose")
                if norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    dedupe_indices.append(idx)
                else:
                    logger.debug("Removing duplicate-titled pier seed: %s", title)
            pier_seed_indices = dedupe_indices

        # Ensure primary seed is included
        if seed_idx not in pier_seed_indices:
            should_insert = True
            if bundle.track_titles is not None:
                seed_title = bundle.track_titles[seed_idx] or ""
                norm_seed = normalize_title_for_dedupe(str(seed_title), mode="loose")
                for idx in pier_seed_indices:
                    anchor_title = bundle.track_titles[idx] or ""
                    if normalize_title_for_dedupe(str(anchor_title), mode="loose") == norm_seed:
                        should_insert = False
                        break
            if should_insert:
                pier_seed_indices.insert(0, seed_idx)

        seed_labels = []
        for idx in pier_seed_indices:
            tid = bundle.track_ids[idx]
            title = bundle.track_titles[idx] if bundle.track_titles is not None else ""
            seed_labels.append(f"{tid}:{title}")
        logger.info("Pier seeds (%d): %s", len(pier_seed_indices), seed_labels)

        # Pier-bridge handles any number of seeds (including 1 seed as arc structure)
        if True:
            # 3. Deduplicate candidate pool by (artist, title), keeping canonical version
            pool_indices = list(pool.pool_indices)
            if bundle.track_titles is not None and bundle.artist_keys is not None:
                # Group by (artist_key, normalized_title)
                key_to_indices: Dict[tuple, list[int]] = {}
                for idx in pool_indices:
                    artist = str(bundle.artist_keys[idx]) if bundle.artist_keys is not None else ""
                    title = bundle.track_titles[idx] or ""
                    key = (normalize_artist_key(artist), normalize_title_for_dedupe(str(title), mode="loose"))
                    if key not in key_to_indices:
                        key_to_indices[key] = []
                    key_to_indices[key].append(idx)

                # Select canonical version for each group
                deduped_pool_indices: list[int] = []
                for key, indices in key_to_indices.items():
                    if len(indices) == 1:
                        deduped_pool_indices.append(indices[0])
                    else:
                        # Score each candidate for version preference
                        best_idx = indices[0]
                        best_score = -1
                        for idx in indices:
                            title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""
                            score = calculate_version_preference_score(title)
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        deduped_pool_indices.append(best_idx)
                        logger.debug(
                            "Pool dedupe: kept idx=%d for key=%s (from %d versions)",
                            best_idx, key, len(indices)
                        )
                logger.info(
                    "Pier bridge candidate pool deduped: %d → %d tracks",
                    len(pool_indices), len(deduped_pool_indices)
                )
                pool_indices = deduped_pool_indices

            # 4. Convert seed track IDs for pier bridge builder
            seed_track_ids_for_pier = [str(bundle.track_ids[idx]) for idx in pier_seed_indices]

            # 5. Call pier bridge builder
            pb_cfg = pier_bridge_config or PierBridgeConfig(
                transition_floor=cfg.construct.transition_floor,
            )
            pb_result: PierBridgeResult = build_pier_bridge_playlist(
                seed_track_ids=seed_track_ids_for_pier,
                total_tracks=playlist_len,
                bundle=bundle,
                candidate_pool_indices=pool_indices,
                cfg=pb_cfg,
            )

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
            from dataclasses import dataclass as dc_dataclass
            pb_track_indices = np.array(pb_result.track_indices, dtype=np.int32)

            # Build edge scores for stats
            edge_scores_list = []
            for i in range(1, len(pb_result.track_indices)):
                prev_idx = pb_result.track_indices[i - 1]
                cur_idx = pb_result.track_indices[i]
                prev_id = str(bundle.track_ids[prev_idx])
                cur_id = str(bundle.track_ids[cur_idx])
                # Compute transition score
                X_end = get_sonic_matrix(bundle, "end")
                X_start = get_sonic_matrix(bundle, "start")
                if X_end is not None and X_start is not None:
                    end_vec = X_end[prev_idx]
                    start_vec = X_start[cur_idx]
                    trans_sim = float(np.dot(end_vec, start_vec) / (
                        np.linalg.norm(end_vec) * np.linalg.norm(start_vec) + 1e-12
                    ))
                else:
                    trans_sim = 0.0
                edge_scores_list.append({
                    "prev_id": prev_id,
                    "cur_id": cur_id,
                    "T": trans_sim,
                })

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
                    "repair_applied": False,  # No repair in pier bridge mode
                },
                params_requested={"strategy": "pier_bridge"},
                params_effective={"strategy": "pier_bridge", "pier_config": pb_cfg.__dict__},
            )

    # Legacy paths (anchor_builder, standard construct_playlist) have been removed.
    # All playlist construction now goes through pier-bridge.

    ordered_track_ids = [str(bundle.track_ids[i]) for i in playlist.track_indices]
    params_requested = _params_from_config(cfg)
    if overrides:
        params_requested["overrides"] = overrides
    params_effective: Dict[str, Any] = {
        "sonic_variant": variant_stats,
        "embedding": embedding_model.params_effective,
        "candidate_pool": pool.params_effective,
        "playlist": playlist.params_effective,
    }
    stats: Dict[str, Any] = {
        "candidate_pool": pool.stats,
        "playlist": playlist.stats,
    }

    return DSPipelineResult(
        track_ids=ordered_track_ids,
        track_indices=playlist.track_indices.tolist(),
        stats=stats,
        params_requested=params_requested,
        params_effective=params_effective,
    )


def build_run_artifact(
    result: DSPipelineResult,
    bundle: ArtifactBundle,
    cfg: DSPipelineConfig,
    seed_track_id: str,
    *,
    sonic_weight: float = 0.67,
    genre_weight: float = 0.33,
    genre_method: str = "ensemble",
    min_genre_similarity: float = 0.2,
    genre_gate_mode: str = "hard",
    sonic_variant: str = "robust_whiten",
):
    """
    Build a RunArtifact from a DSPipelineResult for instrumentation.

    Args:
        result: The pipeline result
        bundle: The artifact bundle used
        cfg: The DS pipeline config
        seed_track_id: Original seed track ID
        sonic_weight: Sonic weight used in scoring
        genre_weight: Genre weight used in scoring
        genre_method: Genre similarity method
        min_genre_similarity: Genre floor threshold
        genre_gate_mode: "hard" or "soft"
        sonic_variant: Sonic variant used

    Returns:
        RunArtifact ready for writing
    """
    from src.eval.run_artifact import (
        EdgeRecord,
        ExclusionCounters,
        RunArtifact,
        SettingsSnapshot,
        SummaryMetrics,
        TrackRecord,
        compute_summary_metrics,
        generate_run_id,
    )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run_id = generate_run_id(seed_track_id, cfg.mode, timestamp)

    # Get seed info
    seed_idx = bundle.track_id_to_index.get(seed_track_id, 0)
    seed_artist = str(bundle.artist_keys[seed_idx]) if bundle.artist_keys is not None else ""
    seed_title = str(bundle.track_titles[seed_idx]) if bundle.track_titles is not None else ""

    # Build settings snapshot
    settings = SettingsSnapshot(
        run_id=run_id,
        timestamp=timestamp,
        mode=cfg.mode,
        pipeline="ds",
        seed_track_id=seed_track_id,
        seed_artist=seed_artist,
        seed_title=seed_title,
        playlist_length=len(result.track_ids),
        random_seed=result.params_effective.get("candidate_pool", {}).get("rng_seed", 0),
        sonic_weight=sonic_weight,
        genre_weight=genre_weight,
        genre_method=genre_method,
        min_genre_similarity=min_genre_similarity,
        genre_gate_mode=genre_gate_mode,
        transition_floor=cfg.construct.transition_floor,
        transition_gamma=cfg.construct.transition_gamma,
        hard_floor=cfg.construct.hard_floor,
        center_transitions=cfg.construct.center_transitions,
        alpha=cfg.construct.alpha,
        beta=cfg.construct.beta,
        gamma=cfg.construct.gamma,
        alpha_schedule=cfg.construct.alpha_schedule,
        similarity_floor=cfg.candidate.similarity_floor,
        max_pool_size=cfg.candidate.max_pool_size,
        max_artist_fraction=cfg.candidate.max_artist_fraction_final,
        min_gap=cfg.construct.min_gap,
        sonic_variant=sonic_variant,
    )

    # Build track records
    tracks: List[TrackRecord] = []
    playlist_stats = result.stats.get("playlist", {})
    edge_scores = playlist_stats.get("edge_scores", [])

    # Build seed_sim lookup from edge_scores or fallback
    seed_sim_lookup: Dict[str, float] = {}
    for i, track_id in enumerate(result.track_ids):
        idx = bundle.track_id_to_index.get(track_id)
        if idx is not None:
            # Try to get seed_sim from playlist stats
            seed_sim = 0.0
            if i == 0:
                seed_sim = 1.0  # Seed itself
            elif i - 1 < len(edge_scores):
                # Approximate from edge data or use a placeholder
                edge = edge_scores[i - 1] if i - 1 < len(edge_scores) else {}
                seed_sim = edge.get("H", 0.0)  # Use hybrid as proxy

            artist_key = str(bundle.artist_keys[idx]) if bundle.artist_keys is not None else ""
            artist_name = str(bundle.track_artists[idx]) if bundle.track_artists is not None else artist_key
            title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""

            tracks.append(TrackRecord(
                position=i,
                track_id=track_id,
                artist_key=artist_key,
                artist_name=artist_name,
                title=title,
                duration_ms=None,  # Not in bundle
                seed_sim=seed_sim,
                genres=[],  # Could be populated from X_genre_raw if needed
            ))

    # Build edge records
    edges: List[EdgeRecord] = []
    for i, edge_data in enumerate(edge_scores):
        prev_id = str(edge_data.get("prev_id", ""))
        cur_id = str(edge_data.get("cur_id", ""))
        prev_idx = bundle.track_id_to_index.get(prev_id)
        cur_idx = bundle.track_id_to_index.get(cur_id)

        prev_artist = ""
        cur_artist = ""
        if prev_idx is not None and bundle.artist_keys is not None:
            prev_artist = str(bundle.artist_keys[prev_idx])
        if cur_idx is not None and bundle.artist_keys is not None:
            cur_artist = str(bundle.artist_keys[cur_idx])

        # Extract similarity scores
        sonic_sim = float(edge_data.get("S", 0.0))
        genre_sim = float(edge_data.get("G", 0.0))
        hybrid_sim = float(edge_data.get("H", 0.0))
        transition_sim = float(edge_data.get("T", edge_data.get("T_used", 0.0)))
        transition_raw = float(edge_data.get("T_raw_uncentered", transition_sim))
        transition_centered = float(edge_data.get("T_centered_cos", 0.0))

        below_floor = transition_sim < cfg.construct.transition_floor

        edges.append(EdgeRecord(
            position=i,
            prev_track_id=prev_id,
            next_track_id=cur_id,
            prev_artist=prev_artist,
            next_artist=cur_artist,
            sonic_sim=sonic_sim,
            genre_sim=genre_sim,
            hybrid_sim=hybrid_sim,
            transition_sim=transition_sim,
            transition_raw=transition_raw,
            transition_centered=transition_centered,
            below_floor=below_floor,
            same_artist=(prev_artist == cur_artist and prev_artist != ""),
        ))

    # Build exclusion counters from pool stats
    pool_stats = result.stats.get("candidate_pool", {})
    exclusions = ExclusionCounters(
        below_similarity_floor=pool_stats.get("below_similarity_floor", 0),
        genre_gate_rejected=0,  # Not tracked at pool level currently
        artist_cap_rejected=pool_stats.get("artist_cap_excluded", 0),
        adjacency_rejected=0,  # Tracked in constructor, not exposed yet
        min_gap_rejected=0,
        transition_floor_rejected=playlist_stats.get("below_floor_count", 0),
        total_candidates_considered=pool_stats.get("total_candidates_considered", 0),
    )

    # Compute summary metrics
    metrics = compute_summary_metrics(tracks, edges, min_genre_similarity)

    return RunArtifact(
        settings=settings,
        tracks=tracks,
        edges=edges,
        exclusions=exclusions,
        metrics=metrics,
    )
