from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.features.artifacts import ArtifactBundle, get_sonic_matrix, load_artifact_bundle
from src.playlist.candidate_pool import CandidatePoolResult, build_candidate_pool
from src.playlist.config import DSPipelineConfig, default_ds_config, resolve_pier_bridge_tuning
from src.playlist.constructor import PlaylistResult  # Type only, no longer calling construct_playlist
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    PierBridgeResult,
    build_pier_bridge_playlist,
)
from src.playlist.artist_identity_resolver import ArtistIdentityConfig
from src.playlist.run_audit import (
    RunAuditContext,
    RunAuditEvent,
    now_utc_iso,
    parse_infeasible_handling_config,
    parse_run_audit_config,
    write_markdown_report,
)
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
    audit_events: Optional[list[RunAuditEvent]] = [] if bool(audit_cfg.enabled) else None
    audit_context: Optional[RunAuditContext] = None
    audit_path: Optional[Path] = None

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

    # Restrict bundle to allowed_track_ids when provided.
    # IMPORTANT: if both allowed_track_ids and excluded_track_ids are provided,
    # apply BOTH (intersection); otherwise style-aware runs bypass exclusions (e.g., recency).
    if allowed_track_ids:
        allowed_indices: list[int] = []
        allowed_track_ids_set = {str(tid) for tid in allowed_track_ids}
        for tid in allowed_track_ids_set:
            idx = bundle.track_id_to_index.get(tid)
            if idx is not None:
                allowed_indices.append(idx)

        # Always include seed + anchor seeds (piers), even if not explicitly in allowed list.
        exempt_ids = {str(seed_track_id)}
        if anchor_seed_ids:
            exempt_ids.update(str(sid) for sid in anchor_seed_ids)
        for tid in exempt_ids:
            idx = bundle.track_id_to_index.get(tid)
            if idx is not None:
                allowed_indices.append(idx)

        # Add exempt_ids to allowed_track_ids_set for final enforcement check
        allowed_track_ids_set.update(exempt_ids)

        applied_excluded = 0
        if excluded_track_ids:
            excluded_set = {str(t) for t in excluded_track_ids}
            kept: list[int] = []
            for idx in allowed_indices:
                tid = str(bundle.track_ids[int(idx)])
                if tid in exempt_ids:
                    kept.append(idx)
                    continue
                if tid in excluded_set:
                    applied_excluded += 1
                    continue
                kept.append(idx)
            allowed_indices = kept

        allowed_indices = sorted(set(int(i) for i in allowed_indices))
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
            durations_ms=_slice(bundle.durations_ms),
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
        if excluded_track_ids and (os.environ.get("PLAYLIST_DIAG_RECENCY") or os.environ.get("PLAYLIST_DIAG_POOL")):
            logger.info(
                "DS bundle clamp+exclude: allowed_ids=%d excluded_ids=%d applied_excluded=%d final_N=%d",
                len(allowed_track_ids_set),
                len(excluded_track_ids),
                applied_excluded,
                N_allowed,
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

    # Resolve all seed indices for admission gating (max over seeds)
    seed_indices_for_floor = [seed_idx]
    for sid in anchor_seed_ids:
        idx = bundle.track_id_to_index.get(str(sid))
        if idx is not None and idx not in seed_indices_for_floor:
            seed_indices_for_floor.append(idx)
        elif idx is None:
            logger.debug("Anchor seed %s not found in bundle for sonic floor", sid)
    if len(seed_indices_for_floor) > 1:
        logger.info("Sonic admission uses %d seeds (max similarity across seeds)", len(seed_indices_for_floor))

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

    # Apply optional broad-genre masking for genre embeddings/gating
    raw_broad_filters = cfg.candidate.broad_filters or ()
    try:
        if isinstance(raw_broad_filters, np.ndarray):
            raw_broad_filters = raw_broad_filters.tolist()
    except Exception:
        raw_broad_filters = ()
    broad_filters = tuple(str(b).lower() for b in raw_broad_filters)

    X_genre_smoothed = bundle.X_genre_smoothed
    X_genre_raw = bundle.X_genre_raw
    raw_vocab = getattr(bundle, "genre_vocab", [])
    try:
        if hasattr(raw_vocab, "tolist"):
            genre_vocab: list[str] = list(raw_vocab.tolist())
        else:
            genre_vocab = list(raw_vocab or [])
    except Exception:
        genre_vocab = [str(g) for g in raw_vocab] if raw_vocab is not None else []
    genre_mask = None
    if broad_filters and genre_vocab:
        genre_mask = np.array([g.lower() not in broad_filters for g in genre_vocab], dtype=bool)
        if X_genre_smoothed is not None and genre_mask.shape[0] == X_genre_smoothed.shape[1]:
            X_genre_smoothed = X_genre_smoothed[:, genre_mask]
            genre_vocab = [g for g, keep in zip(genre_vocab, genre_mask) if keep]
        if X_genre_raw is not None and genre_mask.shape[0] == X_genre_raw.shape[1]:
            X_genre_raw = X_genre_raw[:, genre_mask]

    embedding_model = build_hybrid_embedding(
        X_sonic_for_embed,
        X_genre_smoothed,
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
        seed_indices=seed_indices_for_floor,
        embedding=embedding_model.embedding,
        artist_keys=bundle.artist_keys,
        track_ids=bundle.track_ids,
        track_titles=bundle.track_titles,
        track_artists=bundle.track_artists,
        durations_ms=bundle.durations_ms,
        cfg=cfg.candidate,
        random_seed=random_seed,
        X_sonic=X_sonic_for_embed,
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
        from src.title_dedupe import normalize_title_for_dedupe, calculate_version_preference_score
        from src.playlist.identity_keys import identity_keys_for_index

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
            pool_indices = list(getattr(pool, "eligible_indices", pool.pool_indices))
            if bundle.track_titles is not None:
                # Group by (artist_key, normalized_title)
                key_to_indices: Dict[tuple, list[int]] = {}
                for idx in pool_indices:
                    key = identity_keys_for_index(bundle, int(idx)).track_key
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

            if audit_events is not None and audit_context is None:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_id = f"ds_{cfg.mode}_{ts}_{uuid.uuid4().hex[:8]}"
                seed_artist_val: Optional[str] = None
                try:
                    if bundle.track_artists is not None:
                        seed_artist_val = str(bundle.track_artists[seed_idx])
                    elif bundle.artist_keys is not None:
                        seed_artist_val = str(bundle.artist_keys[seed_idx])
                except Exception:
                    seed_artist_val = None
                audit_context = RunAuditContext(
                    timestamp_utc=now_utc_iso(),
                    run_id=run_id,
                    ds_mode=str(cfg.mode),
                    seed_track_id=str(seed_track_id),
                    seed_artist=seed_artist_val,
                    dry_run=bool(dry_run),
                    artifact_path=str(artifact_path),
                    sonic_variant=str(resolved_variant) if resolved_variant else None,
                    allowed_ids_count=int(len(allowed_track_ids_set or set())),
                    pool_source=str(pool_source) if pool_source is not None else None,
                    artist_style_enabled=bool(artist_style_enabled),
                    artist_playlist=bool(artist_playlist),
                    extra=dict(audit_context_extra or {}),
                )
                audit_path = Path(audit_cfg.out_dir) / f"{audit_context.run_id}.md"

            # 5. Call pier bridge builder
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
                center_transitions=cfg.construct.center_transitions,
                transition_weights=transition_weights,
                sonic_variant=resolved_variant,
                weight_bridge=float(tuning.weight_bridge),
                weight_transition=float(tuning.weight_transition),
                genre_tiebreak_weight=float(tuning.genre_tiebreak_weight),      
                genre_penalty_threshold=float(tuning.genre_penalty_threshold),  
                genre_penalty_strength=float(tuning.genre_penalty_strength),    
            )

            # Segment-local pier-bridge policy defaults (with optional overrides).
            disallow_seed_raw = pb_overrides.get("disallow_seed_artist_in_interiors")
            if isinstance(disallow_seed_raw, bool):
                pb_cfg = replace(
                    pb_cfg,
                    disallow_seed_artist_in_interiors=bool(disallow_seed_raw),
                )
            else:
                pb_cfg = replace(
                    pb_cfg,
                    disallow_seed_artist_in_interiors=bool(artist_playlist),
                )

            disallow_pier_raw = pb_overrides.get("disallow_pier_artists_in_interiors")
            if isinstance(disallow_pier_raw, bool):
                pb_cfg = replace(
                    pb_cfg,
                    disallow_pier_artists_in_interiors=bool(disallow_pier_raw),
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

            experiment_enabled = False
            experiment_min_weight = float(pb_cfg.experiment_bridge_min_weight)
            experiment_balance_weight = float(pb_cfg.experiment_bridge_balance_weight)
            experiments_raw = pb_overrides.get("experiments")
            progress_arc_enabled = False
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
            if progress_arc_enabled:
                pb_cfg = replace(
                    pb_cfg,
                    progress_arc_enabled=True,
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
                logger.info(
                    "Pier-bridge progress arc enabled (weight=%.2f shape=%s source=%s)",
                    float(progress_arc_weight),
                    str(progress_arc_shape),
                    str(progress_arc_source or "default"),
                )

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

            logger.info(
                "Pier-bridge segment policy: artist_playlist=%s strategy=%s pool_max=%d progress=%s disallow_seed_artist_in_interiors=%s disallow_pier_artists_in_interiors=%s",
                bool(artist_playlist),
                str(pb_cfg.segment_pool_strategy),
                int(pb_cfg.segment_pool_max),
                bool(pb_cfg.progress_enabled),
                bool(pb_cfg.disallow_seed_artist_in_interiors),
                bool(pb_cfg.disallow_pier_artists_in_interiors),
            )
            if audit_events is not None and not any(e.kind == "preflight" for e in audit_events):
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
                audit_events.append(
                    RunAuditEvent(
                        kind="preflight",
                        ts_utc=now_utc_iso(),
                        payload={
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
                audit_events=audit_events,
                artist_identity_cfg=artist_identity_cfg,
            )
            if not pb_result.success:
                if audit_events is not None and audit_context is not None and audit_path is not None:
                    audit_events.append(
                        RunAuditEvent(
                            kind="final_failure",
                            ts_utc=now_utc_iso(),
                            payload={
                                "failure_reason": str(pb_result.failure_reason or "pier-bridge failed"),
                                "segment_bridge_floors_used": (pb_result.stats or {}).get("segment_bridge_floors_used"),
                                "segment_backoff_attempts_used": (pb_result.stats or {}).get("segment_backoff_attempts_used"),
                            },
                        )
                    )
                    try:
                        write_markdown_report(
                            context=audit_context,
                            events=audit_events,
                            path=audit_path,
                            max_bytes=int(audit_cfg.max_bytes),
                        )
                    except Exception as exc:
                        logger.exception("Failed to write run audit report: %s", exc)
                    raise ValueError(
                        f"Pier-bridge failed: {pb_result.failure_reason} (audit: {audit_path})"
                    )
                raise ValueError(f"Pier-bridge failed: {pb_result.failure_reason}")

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
        if audit_events is not None and audit_context is not None and audit_path is not None:
            audit_events.append(
                RunAuditEvent(
                    kind="final_failure",
                    ts_utc=now_utc_iso(),
                    payload={
                        "failure_reason": msg,
                        "post_order_filters_applied": [],
                        "post_order_validation": dict(post_order_validation),
                    },
                )
            )
            try:
                write_markdown_report(
                    context=audit_context,
                    events=audit_events,
                    path=audit_path,
                    max_bytes=int(audit_cfg.max_bytes),
                )
            except Exception as write_exc:
                logger.exception("Failed to write run audit report: %s", write_exc)
            raise ValueError(f"{msg} (audit: {audit_path})")
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
        if audit_events is not None and audit_context is not None and audit_path is not None:
            audit_events.append(
                RunAuditEvent(
                    kind="final_failure",
                    ts_utc=now_utc_iso(),
                    payload={
                        "failure_reason": f"allowed_set_invariant_failed: {exc}",
                    },
                )
            )
            try:
                write_markdown_report(
                    context=audit_context,
                    events=audit_events,
                    path=audit_path,
                    max_bytes=int(audit_cfg.max_bytes),
                )
            except Exception as write_exc:
                logger.exception("Failed to write run audit report: %s", write_exc)
            raise ValueError(f"{exc} (audit: {audit_path})") from exc
        raise

    if audit_events is not None and audit_context is not None and audit_path is not None and bool(audit_cfg.write_on_success):
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

        audit_events.append(
            RunAuditEvent(
                kind="final_success",
                ts_utc=now_utc_iso(),
                payload={
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
        )
        try:
            write_markdown_report(
                context=audit_context,
                events=audit_events,
                path=audit_path,
                max_bytes=int(audit_cfg.max_bytes),
            )
        except Exception as write_exc:
            logger.exception("Failed to write run audit report: %s", write_exc)
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
    if audit_path is not None:
        try:
            stats.setdefault("playlist", {})
            stats["playlist"]["audit_path"] = str(audit_path)
        except Exception:
            pass

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
    pool_stats = result.stats.get("candidate_pool", {}) if isinstance(result.stats, dict) else {}
    edge_scores = playlist_stats.get("edge_scores", [])
    seed_sonic_lookup = pool_stats.get("seed_sonic_sim_track_ids") or {}

    # Build seed_sim lookup from edge_scores or fallback
    seed_sim_lookup: Dict[str, float] = {}
    for i, track_id in enumerate(result.track_ids):
        idx = bundle.track_id_to_index.get(track_id)
        if idx is not None:
            # Try to get seed_sim from playlist stats
            seed_sim = seed_sonic_lookup.get(str(track_id), 0.0)
            if not seed_sim:
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
