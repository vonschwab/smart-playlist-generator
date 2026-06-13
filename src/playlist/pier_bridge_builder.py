"""
Pier + Bridge Playlist Builder
==============================

A new playlist ordering strategy where:
- Each seed track is a fixed "pier"
- Bridge segments connect consecutive piers
- No repair pass after ordering

Key features:
- Candidate pool deduped BEFORE ordering (no duplicate songs by normalized artist+title)
- Genre gating stays enabled with hard floors (no relaxation)
- Global used_track_ids prevents duplicates across segments
- One track per artist per segment enforced during beam search
- Cross-segment min_gap enforced during generation via boundary-aware constraints
- No post-order filtering or dropping (guarantees exact length)
- Single seed mode: seed acts as both start AND end pier, creating an arc structure
- Seed artist is allowed in bridges with same constraints as other artists
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from src.features.artifacts import ArtifactBundle
from src.title_dedupe import normalize_artist_key
from src.string_utils import sanitize_for_logging
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    resolve_artist_identity_keys,
)
from src.playlist.run_audit import InfeasibleHandlingConfig, RunAuditConfig, RunAuditEvent, now_utc_iso

# Tier-3.1 PR-1: 14 pure genre helpers extracted to pier_bridge/genre.py.
# Re-exported here for backward compatibility (callers + tests still import
# these names from src.playlist.pier_bridge_builder).
from src.playlist.pier_bridge.genre import (  # noqa: F401  (re-exported for tests/back-compat)
    _apply_idf_weighting,
    _compute_coverage,
    _compute_coverage_bonus,
    _compute_genre_idf as _compute_genre_idf_impl,
    _ensure_genre_similarity_overrides_loaded,
    _extract_top_genres,
    _genre_similarity_score,
    _genre_vocab_map,
    _label_to_genre_vector,
    _label_to_smoothed_vector,
    _load_genre_similarity_graph,
    _normalize_vec,
    _select_top_genre_labels,
    _shortest_genre_path,
)


# Tier-3.1 PR-2: vector math + transition scoring.
from src.playlist.pier_bridge.vec import (  # noqa: F401
    _cosine_sim,
    _l2_normalize_rows,
    _compute_transition_score as _compute_transition_score_impl,
    _compute_transition_score_raw_and_transformed as _compute_transition_score_raw_and_transformed_impl,
)

# Tier-3.1 PR-2: progress + diagnostic metrics.
from src.playlist.pier_bridge.metrics import (  # noqa: F401
    _compute_chosen_source_counts,
    _compute_pool_overlap_metrics,
    _compute_progress_tracking_metrics,
    _dist,
    _progress_arc_loss_value,
    _progress_target_curve,
    _step_fraction,
)

# Tier-3.1 PR-3: config dataclasses + resolver + back-compat wrappers.
from src.playlist.pier_bridge.config import (  # noqa: F401  (re-exported for callers + tests)
    PierBridgeConfig,
    PierBridgeResult,
    SegmentDiagnostics,
    resolve_pier_bridge_tuning,
    _compute_genre_idf,
    _compute_transition_score,
    _compute_transition_score_raw_and_transformed,
)

# Tier-3.1 PR-4: audit summary + bridgeability scoring.
from src.playlist.pier_bridge.audit_summary import (  # noqa: F401
    _compute_bridgeability_score,
    _summarize_candidates_for_audit,
)

# Tier-3.1 PR-5: micro-pier + relaxation subsystem.
from src.playlist.pier_bridge.micro_pier import (  # noqa: F401
    _attempt_micro_pier_split,
    _build_dj_relaxation_attempts,
    _micro_pier_candidate_pool,
    _score_micro_pier_candidates,
    _select_micro_pier_candidates,
    _should_attempt_micro_pier,
)

# Tier-3.1 PR-6: seed ordering + pool prep helpers.
from src.playlist.pier_bridge.seeds import (  # noqa: F401
    _dedupe_candidate_pool,
    _order_seeds_by_bridgeability,
    _segment_far_stats,
    _select_connector_candidates,
)

# Tier-3.1 PR-7: pool builders + bridge-score kernel.
from src.playlist.pier_bridge.pool import (  # noqa: F401  (re-exported for tests/back-compat)
    _build_segment_candidate_pool_legacy,
    _build_segment_candidate_pool_scored,
    _compute_bridge_score,
)

# Tier-3.1 PR-8: genre waypoint target builder.
from src.playlist.pier_bridge.genre_targets import (  # noqa: F401  (re-exported for tests/back-compat)
    _build_genre_targets,
    _fallback_genre_vector,
    build_dense_genre_targets,
)

# Genre-arc steering: distribution-relative floor relaxation.
from src.playlist.pier_bridge.percentiles import relax_percentile

# Tier-3.1 PR-8: beam search engine + duration penalty.
from src.playlist.pier_bridge.beam import (  # noqa: F401  (re-exported for tests/back-compat)
    BeamState,
    _beam_search_segment,
    _compute_duration_penalty,
)
from src.playlist.transition_metrics import TransitionMetricContext, score_transition_edge


logger = logging.getLogger(__name__)


def _compute_edge_scores(
    path: List[int],
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    metric_context: Optional[TransitionMetricContext] = None,
) -> Tuple[float, float]:
    """Compute worst and mean edge scores for a path."""
    if len(path) < 2:
        return (1.0, 1.0)

    scores = []
    for i in range(len(path) - 1):
        if metric_context is not None:
            score = float(score_transition_edge(metric_context, path[i], path[i + 1]).get("T"))
        else:
            score = _compute_transition_score(
                path[i], path[i + 1], X_full, X_start, X_mid, X_end, cfg
            )
        scores.append(score)

    return (min(scores), sum(scores) / len(scores))


def _enforce_min_gap_global(
    indices: List[int],
    artist_keys: Optional[np.ndarray] = None,
    min_gap: int = 1,
    *,
    bundle: Optional[ArtifactBundle] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
) -> Tuple[List[int], int]:
    """
    Drop tracks that would violate a global min_gap across concatenated segments.

    Pier-bridge already enforces one-per-artist per segment, but adjacent
    duplicates can appear at segment boundaries. This pass removes any track
    that would repeat a normalized artist within the last `min_gap` slots.

    If artist_identity_cfg is provided and enabled, uses identity-based matching
    (collapsing ensemble variants and splitting collaborations). Each collaboration
    track contributes ALL participant identity keys to the recent window.
    """
    if not indices or min_gap <= 0:
        return indices, 0

    recent: List[str] = []
    output: List[int] = []
    dropped = 0

    use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

    for idx in indices:
        if use_identity:
            # Identity mode: use raw artist string to preserve collaborations
            # (identity_keys_for_index has collaborations stripped)
            artist_str = ""
            if bundle is not None and bundle.track_artists is not None:
                try:
                    artist_str = str(bundle.track_artists[int(idx)] or "")
                except Exception:
                    artist_str = ""
            if not artist_str and artist_keys is not None:
                try:
                    artist_str = str(artist_keys[int(idx)])
                except Exception:
                    artist_str = ""

            # Resolve to identity keys
            identity_keys_set = resolve_artist_identity_keys(artist_str, artist_identity_cfg)

            # Check if ANY identity key violates min_gap
            violated_key = None
            for identity_key in identity_keys_set:
                if identity_key in recent:
                    violated_key = identity_key
                    break

            if violated_key is not None:
                dropped += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Rejected candidate idx=%d due to identity_min_gap: key=%r in recent window (distance<=%d)",
                        idx, violated_key, min_gap
                    )
                continue

            # Accept track: add to output and update recent window with ALL identity keys
            output.append(idx)
            for identity_key in identity_keys_set:
                recent.append(identity_key)
            # Trim recent window to size min_gap
            while len(recent) > min_gap:
                recent.pop(0)
        else:
            # Legacy mode: single artist key
            key = ""
            if bundle is not None:
                try:
                    key = identity_keys_for_index(bundle, int(idx)).artist_key
                except Exception:
                    key = ""
            if not key and artist_keys is not None:
                try:
                    key = normalize_artist_key(str(artist_keys[int(idx)]))
                except Exception:
                    key = ""
            if not key:
                key = f"unknown_artist:{idx}"

            if key in recent:
                dropped += 1
                continue

            output.append(idx)
            recent.append(key)
            if len(recent) > min_gap:
                recent.pop(0)

    return output, dropped



def build_pier_bridge_playlist(
    *,
    seed_track_ids: List[str],
    total_tracks: int,
    bundle: ArtifactBundle,
    candidate_pool_indices: List[int],
    cfg: Optional[PierBridgeConfig] = None,
    min_genre_similarity: Optional[float] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_method: str = "ensemble",
    internal_connector_indices: Optional[Set[int]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    allowed_track_ids_set: Optional[set[str]] = None,
    infeasible_handling: Optional[InfeasibleHandlingConfig] = None,
    audit_config: Optional[RunAuditConfig] = None,
    audit_events: Optional[list[RunAuditEvent]] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
    perceptual_bpm: Optional[np.ndarray] = None,
    tempo_stability_arr: Optional[np.ndarray] = None,
    onset_rate: Optional[np.ndarray] = None,
    min_gap: int = 1,
) -> PierBridgeResult:
    """
    Build playlist using pier + bridge strategy.

    Args:
        seed_track_ids: List of seed track IDs (will become piers)
        total_tracks: Target total playlist length
        bundle: Artifact bundle with sonic features
        candidate_pool_indices: Pre-filtered candidate pool indices
        cfg: Configuration (uses defaults if None)
        min_genre_similarity: Optional genre gate threshold
        X_genre_smoothed: Genre vectors for gating
        genre_method: Genre similarity method

    Returns:
        PierBridgeResult with ordered track IDs and diagnostics
    """
    if cfg is None:
        cfg = PierBridgeConfig()
    # Genre steering supersedes the older dj_bridging waypoint system. They are two
    # overlapping genre-arc implementations (dense 64-dim arc vote vs. 893-dim waypoint
    # pooling) and were never meant to run together. When steering is on, force
    # dj_bridging off so only the steering path drives genre routing.
    if bool(cfg.genre_steering_enabled) and bool(cfg.dj_bridging_enabled):
        logger.info(
            "genre_steering_enabled=True supersedes dj_bridging_enabled; disabling dj_bridging for this run."
        )
        cfg = replace(cfg, dj_bridging_enabled=False)
    if infeasible_handling is None:
        infeasible_handling = InfeasibleHandlingConfig()
    if audit_config is None:
        audit_config = RunAuditConfig()
    audit_enabled = bool(audit_config.enabled) and audit_events is not None
    top_k = int(audit_config.include_top_k) if audit_enabled else 0

    num_seeds = len(seed_track_ids)
    if num_seeds == 0:
        raise ValueError("At least one seed is required")
    if num_seeds > total_tracks:
        raise ValueError(f"Number of seeds ({num_seeds}) exceeds total_tracks ({total_tracks})")

    # Resolve seed indices
    seed_indices: List[int] = []
    for tid in seed_track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            raise ValueError(f"Seed track not found in bundle: {tid}")
        seed_indices.append(idx)

    # Remove duplicates while preserving order
    seed_indices = list(dict.fromkeys(seed_indices))
    num_seeds = len(seed_indices)
    seed_id_set = {str(bundle.track_ids[i]) for i in seed_indices}

    logger.info("Pier+Bridge: %d seeds, target %d tracks", num_seeds, total_tracks)

    # Deduplicate candidate pool by artist+title
    deduped_pool, _ = _dedupe_candidate_pool(candidate_pool_indices, bundle)

    # Exclude seed indices from candidate pool
    seed_set = set(seed_indices)
    universe = [idx for idx in deduped_pool if idx not in seed_set]

    logger.info("Pier+Bridge: universe size after dedupe and seed exclusion: %d", len(universe))

    # Get sonic matrices (raw beat3tower space)
    X_full_raw = bundle.X_sonic
    X_start_raw = bundle.X_sonic_start
    X_mid_raw = bundle.X_sonic_mid
    X_end_raw = bundle.X_sonic_end

    # Similarity space for bridge gating (full vectors) must match DS admission
    from src.similarity.sonic_variant import compute_sonic_variant_matrix, resolve_sonic_variant

    sonic_variant = resolve_sonic_variant(explicit_variant=cfg.sonic_variant, config_variant=None)
    X_full_variant, _ = compute_sonic_variant_matrix(X_full_raw, sonic_variant, l2=False)
    X_full_norm = _l2_normalize_rows(X_full_variant)
    logger.debug("Pier+Bridge sonic sim space: variant=%s dim=%d", sonic_variant, int(X_full_norm.shape[1]))

    # Rhythm axis for the beam's pace gate (cfg.pace_bridge_floor). The gate in
    # _beam_search_segment silently no-ops when rhythm_matrix is None — a
    # configured gate must either receive its data or warn loudly.
    rhythm_matrix: Optional[np.ndarray] = None
    if float(getattr(cfg, "pace_bridge_floor", 0.0)) > 0.0:
        _td = getattr(bundle, "tower_dims", None)
        _td_tuple = tuple(int(v) for v in _td) if _td is not None else None
        if (
            _td_tuple is not None
            and len(_td_tuple) == 3
            and sum(_td_tuple) == int(X_full_raw.shape[1])
        ):
            from src.playlist.sonic_axes import extract_axis_vectors

            rhythm_matrix = extract_axis_vectors(X_full_raw, tower_pca_dims=_td_tuple)["rhythm"]
            logger.info(
                "Pace bridge gate active: floor=%.2f rhythm_dims=%d",
                float(cfg.pace_bridge_floor),
                int(rhythm_matrix.shape[1]),
            )
        elif perceptual_bpm is not None:
            # No rhythm axis (no-tower variant, e.g. mert): fall back to the
            # perceptual-BPM bridge gate so the configured floor still acts.
            from src.playlist.pier_bridge.pace_gate import bpm_fallback_max_log_distance

            _bpm_cap = float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf")))
            if not np.isfinite(_bpm_cap):
                _bpm_cap = bpm_fallback_max_log_distance(float(cfg.pace_bridge_floor))
                cfg = replace(cfg, bpm_bridge_max_log_distance=_bpm_cap)
            logger.warning(
                "Pace bridge gate FALLBACK: pace_bridge_floor=%.2f is set but the "
                "artifact bundle has no usable tower_dims (got %r for blend dim %d); "
                "rhythm-axis gating is unavailable — pace gating falls back to the "
                "perceptual-BPM gate (bpm_bridge_max_log_distance=%.2f)",
                float(cfg.pace_bridge_floor),
                _td,
                int(X_full_raw.shape[1]),
                _bpm_cap,
            )
        else:
            logger.warning(
                "Pace bridge gate INACTIVE: pace_bridge_floor=%.2f is set but the "
                "artifact bundle has no usable tower_dims (got %r for blend dim %d) "
                "and no perceptual-BPM data is available; "
                "rhythm gating will not run",
                float(cfg.pace_bridge_floor),
                _td,
                int(X_full_raw.shape[1]),
            )

    # Transition space (optional tower weights + optional mean-centering)
    from src.similarity.sonic_variant import apply_transition_weights

    X_full_tr, _ = apply_transition_weights(X_full_raw, config_weights=cfg.transition_weights)
    X_start_tr = None
    X_mid_tr = None
    X_end_tr = None
    if X_start_raw is not None:
        X_start_tr, _ = apply_transition_weights(X_start_raw, config_weights=cfg.transition_weights)
    if X_mid_raw is not None:
        X_mid_tr, _ = apply_transition_weights(X_mid_raw, config_weights=cfg.transition_weights)
    if X_end_raw is not None:
        X_end_tr, _ = apply_transition_weights(X_end_raw, config_weights=cfg.transition_weights)

    if cfg.center_transitions:
        X_full_tr = X_full_tr - X_full_tr.mean(axis=0, keepdims=True)
        if X_start_tr is not None:
            X_start_tr = X_start_tr - X_start_tr.mean(axis=0, keepdims=True)
        if X_mid_tr is not None:
            X_mid_tr = X_mid_tr - X_mid_tr.mean(axis=0, keepdims=True)
        if X_end_tr is not None:
            X_end_tr = X_end_tr - X_end_tr.mean(axis=0, keepdims=True)

    X_full_tr_norm = _l2_normalize_rows(X_full_tr)
    X_start_tr_norm = _l2_normalize_rows(X_start_tr) if X_start_tr is not None else None
    X_mid_tr_norm = _l2_normalize_rows(X_mid_tr) if X_mid_tr is not None else None
    X_end_tr_norm = _l2_normalize_rows(X_end_tr) if X_end_tr is not None else None

    # Instrument transition saturation (sampled); compare raw vs transformed end→start
    if logger.isEnabledFor(logging.DEBUG) and X_end_raw is not None and X_start_raw is not None:
        rng = np.random.default_rng(0)
        n = int(X_full_raw.shape[0])
        sample_n = int(min(5000, n))
        prev = rng.integers(0, n, size=sample_n)
        cand = rng.integers(0, n, size=sample_n)
        end_raw = X_end_raw[prev]
        start_raw = X_start_raw[cand]
        raw_sims = np.sum(end_raw * start_raw, axis=1) / (
            (np.linalg.norm(end_raw, axis=1) * np.linalg.norm(start_raw, axis=1)) + 1e-12
        )
        end_tr = X_end_tr_norm[prev] if X_end_tr_norm is not None else None
        start_tr = X_start_tr_norm[cand] if X_start_tr_norm is not None else None
        if end_tr is not None and start_tr is not None:
            tr_sims = np.sum(end_tr * start_tr, axis=1)
            if cfg.center_transitions:
                tr_sims = (tr_sims + 1.0) / 2.0
            logger.debug(
                "Transition end→start sample: raw[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] "
                "transformed[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] center_transitions=%s",
                float(np.min(raw_sims)),
                float(np.percentile(raw_sims, 5)),
                float(np.percentile(raw_sims, 50)),
                float(np.percentile(raw_sims, 95)),
                float(np.max(raw_sims)),
                float(np.min(tr_sims)),
                float(np.percentile(tr_sims, 5)),
                float(np.percentile(tr_sims, 50)),
                float(np.percentile(tr_sims, 95)),
                float(np.max(tr_sims)),
                bool(cfg.center_transitions),
            )

    # For seed ordering bridgeability heuristic, prefer transition-normalized mats when present
    X_start_norm = X_start_tr_norm
    X_end_norm = X_end_tr_norm

    # Extract genre matrices from bundle (Phase 2: needed for IDF and vector mode)
    # Prefer parameter if provided, otherwise extract from bundle
    X_genre_raw = getattr(bundle, "X_genre_raw", None)
    X_genre_dense = getattr(bundle, "X_genre_dense", None)
    if X_genre_smoothed is None:
        X_genre_smoothed = getattr(bundle, "X_genre_smoothed", None)

    # Per-genre track counts for taxonomy waypoint mass filter (routing fix)
    _genre_vocab_list = list(np.asarray(bundle.genre_vocab, dtype=object)) if getattr(bundle, "genre_vocab", None) is not None else []
    genre_track_counts: Optional[dict[str, int]] = None
    if X_genre_raw is not None and _genre_vocab_list:
        genre_track_counts = {
            str(_genre_vocab_list[j]): int((X_genre_raw[:, j] > 0).sum())
            for j in range(len(_genre_vocab_list))
        }

    # Pairwise genre-edge floor: under taxonomy steering the gate MUST score at
    # tag level (calibration 2026-06-10: the smoothed-vector cosine cannot
    # separate bad edges from good — Sharp Pins->Springsteen 0.693 vs the
    # praised YYY->StVincent 0.677). Without the provider the beam falls back
    # to that cosine, so its absence here is a loud misconfiguration.
    pair_sim_provider = None
    if (
        bool(cfg.genre_steering_enabled)
        and str(getattr(cfg, "genre_steering_source", "dense")) == "taxonomy"
        and float(getattr(cfg, "genre_pair_floor", 0.0)) > 0.0
    ):
        if X_genre_raw is not None and _genre_vocab_list:
            from src.playlist.pier_bridge.taxonomy_steering import (
                build_taxonomy_pair_provider,
                get_taxonomy_steering,
            )
            pair_sim_provider = build_taxonomy_pair_provider(
                get_taxonomy_steering(), X_genre_raw, bundle.genre_vocab
            )
            logger.info(
                "Pairwise genre-edge floor active: floor=%.2f (tag-level taxonomy provider)",
                float(cfg.genre_pair_floor),
            )
        else:
            logger.warning(
                "Pairwise genre-edge floor DEGRADED: genre_pair_floor=%.2f is set but "
                "X_genre_raw/genre_vocab unavailable — the beam will gate on the "
                "smoothed-vector cosine, which does NOT separate bad edges from good",
                float(cfg.genre_pair_floor),
            )

    # Genre similarity for soft edge penalty / tiebreak (cosine on smoothed genre vectors)
    X_genre_use = X_genre_smoothed if X_genre_smoothed is not None else None
    X_genre_norm = None
    if X_genre_use is not None:
        denom_g = np.linalg.norm(X_genre_use, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre_use / denom_g

    transition_metric_context = TransitionMetricContext(
        X_full=X_full_tr_norm,
        X_start=X_start_tr_norm,
        X_mid=X_mid_tr_norm,
        X_end=X_end_tr_norm,
        X_sonic_norm=X_full_norm,
        X_genre_norm=X_genre_norm,
        center_transitions=bool(cfg.center_transitions),
        weight_end_start=float(cfg.weight_end_start),
        weight_mid_mid=float(cfg.weight_mid_mid),
        weight_full_full=float(cfg.weight_full_full),
    )

    # Compute IDF for genre vector mode (Phase 2)
    genre_idf: Optional[np.ndarray] = None
    X_genre_norm_idf: Optional[np.ndarray] = None
    if bool(cfg.dj_genre_use_idf) and bool(cfg.dj_bridging_enabled):
        if X_genre_raw is not None:
            logger.info("Computing genre IDF (power=%.2f norm=%s)...",
                        cfg.dj_genre_idf_power, cfg.dj_genre_idf_norm)
            genre_idf = _compute_genre_idf(X_genre_raw, cfg)
            logger.info("  IDF computed: min=%.3f median=%.3f max=%.3f",
                        float(np.min(genre_idf)), float(np.median(genre_idf)), float(np.max(genre_idf)))

            # Create IDF-weighted matrix for S3 pooling and beam search
            if X_genre_norm is not None:
                X_genre_norm_idf = _apply_idf_weighting(X_genre_norm, genre_idf)
        else:
            logger.warning("IDF enabled but X_genre_raw unavailable; using base genre weights")

    warnings: list[dict[str, Any]] = []
    if bool(cfg.dj_bridging_enabled):
        if X_genre_norm is None:
            warnings.append({
                "type": "genre_missing",
                "scope": "global",
                "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                "anchors_missing": int(num_seeds),
            })
        else:
            seed_vecs = X_genre_norm[seed_indices]
            norms = np.linalg.norm(seed_vecs, axis=1)
            missing_ids = [
                str(bundle.track_ids[idx])
                for idx, nval in zip(seed_indices, norms)
                if float(nval) <= 1e-8
            ]
            if missing_ids:
                warnings.append({
                    "type": "genre_missing",
                    "scope": "anchors",
                    "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                    "missing_anchor_ids": missing_ids,
                })
        if bool(cfg.dj_anchors_must_include_all) and len(seed_indices) != len(seed_track_ids):
            warnings.append({
                "type": "anchor_deduped",
                "scope": "anchors",
                "message": "Duplicate anchors were removed while must_include_all is set.",
                "requested_count": int(len(seed_track_ids)),
                "resolved_count": int(len(seed_indices)),
            })

    genre_graph: Optional[dict[str, list[tuple[str, float]]]] = None
    genre_vocab: Optional[np.ndarray] = None
    if bool(cfg.dj_bridging_enabled):
        route_shape = str(cfg.dj_route_shape or "linear").strip().lower()
        if route_shape == "ladder":
            genre_vocab = getattr(bundle, "genre_vocab", None)
            if genre_vocab is None:
                warnings.append({
                    "type": "genre_ladder_unavailable",
                    "scope": "global",
                    "message": "Genre ladder disabled; missing genre vocab.",
                })
            else:
                repo_root = Path(__file__).resolve().parents[2]
                genre_yaml = repo_root / "data" / "genre_similarity.yaml"
                if bool(cfg.dj_ladder_use_smoothed_waypoint_vectors):
                    _ensure_genre_similarity_overrides_loaded(genre_yaml)
                genre_graph = _load_genre_similarity_graph(
                    genre_yaml,
                    min_similarity=float(cfg.dj_ladder_min_similarity),
                )
                if not genre_graph:
                    warnings.append({
                        "type": "genre_ladder_unavailable",
                        "scope": "global",
                        "message": "Genre ladder disabled; similarity graph unavailable.",
                    })

    # Genre-arc steering: build the niche genre graph ONCE per run from the dense
    # genre embedding (distinct from the hand-curated YAML graph above). Used to
    # route dense per-segment g_targets when steering is enabled.
    genre_graph_arc: Optional[dict[str, list[tuple[str, float]]]] = None
    if bool(cfg.genre_steering_enabled) and getattr(bundle, "genre_emb", None) is not None and getattr(bundle, "genre_vocab", None) is not None:
        from src.playlist.pier_bridge.genre_graph import build_genre_graph
        genre_graph_arc = build_genre_graph(
            bundle.genre_emb, bundle.genre_vocab,
            k=int(getattr(cfg, "dj_ladder_top_labels", 8) or 8),
            min_cos=float(getattr(cfg, "dj_ladder_min_similarity", 0.35) or 0.35),
            hub_labels={"rock", "indie", "alternative", "pop", "indie rock", "electronic"},
        )

    # Precompute allowed indices set if caller passed allowed_track_ids_set.
    # (In style-aware runs, the bundle is often already restricted, but this still
    # acts as a hard gate for candidate admission inside pier-bridge.)
    allowed_set_indices: Optional[Set[int]] = None
    if allowed_track_ids_set is not None:
        allowed_set_indices = set()
        for tid in allowed_track_ids_set:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                allowed_set_indices.add(idx)
        # Ensure piers are always allowed
        allowed_set_indices.update(seed_indices)

    # Order seeds by bridgeability (or preserve order if fixed)
    seed_ordering = str(cfg.dj_seed_ordering or "auto").strip().lower()
    if seed_ordering not in {"auto", "fixed"}:
        warnings.append({
            "type": "seed_ordering_invalid",
            "scope": "anchors",
            "message": f"Unknown seed_ordering '{seed_ordering}', defaulting to auto.",
        })
        seed_ordering = "auto"
    if bool(cfg.dj_bridging_enabled) and seed_ordering == "fixed":
        ordered_seeds = list(seed_indices)
    else:
        if bool(cfg.dj_bridging_enabled):
            ordered_seeds = _order_seeds_by_bridgeability(
                seed_indices,
                X_full_norm,
                X_start_norm,
                X_end_norm,
                X_genre_norm,
                weight_sonic=float(cfg.dj_seed_ordering_weight_sonic),
                weight_genre=float(cfg.dj_seed_ordering_weight_genre),
                weight_bridge=float(cfg.dj_seed_ordering_weight_bridge),
            )
        else:
            ordered_seeds = _order_seeds_by_bridgeability(
                seed_indices, X_full_norm, X_start_norm, X_end_norm
            )

    logger.info("Pier+Bridge: seed order = %s",
               [str(bundle.track_ids[i]) for i in ordered_seeds])

    # Handle single seed as both start AND end pier (arc structure)
    # This creates a playlist that starts from seed, explores, and returns to seed-similar sounds
    is_single_seed_arc = (num_seeds == 1)
    if is_single_seed_arc:
        # Duplicate the seed as both start and end pier
        ordered_seeds = [ordered_seeds[0], ordered_seeds[0]]
        num_segments = 1
        total_interior = total_tracks - 1  # Only one seed in final output
        logger.info("Pier+Bridge: single-seed arc mode (seed is both start and end pier)")
    else:
        num_segments = num_seeds - 1
        total_interior = total_tracks - num_seeds

    # Even split with remainder distributed to earlier segments
    base_length = total_interior // num_segments
    remainder = total_interior % num_segments
    segment_lengths = [
        base_length + (1 if i < remainder else 0)
        for i in range(num_segments)
    ]

    logger.info("Pier+Bridge: segment lengths = %s (total_interior=%d)",
               segment_lengths, total_interior)

    # Build segments
    global_used: Set[int] = set(ordered_seeds)  # Seeds are already "used"
    # Track-key dedupe across the full run: prevent "same song twice" even if track_id differs.
    seed_artist_key: Optional[str] = None
    try:
        if seed_indices:
            seed_artist_key = identity_keys_for_index(bundle, int(seed_indices[0])).artist_key
    except Exception:
        seed_artist_key = None

    seed_track_keys: Set[tuple[str, str]] = set()
    for sidx in set(int(i) for i in seed_indices):
        try:
            seed_track_keys.add(identity_keys_for_index(bundle, int(sidx)).track_key)
        except Exception:
            continue
    used_track_keys: Set[tuple[str, str]] = set(seed_track_keys)

    all_segments: List[List[int]] = []
    all_beam_components: List[dict] = []  # flat per-edge component dicts for audit
    diagnostics: List[SegmentDiagnostics] = []
    soft_genre_penalty_hits_total = 0
    soft_genre_penalty_edges_scored_total = 0
    local_sonic_penalty_hits_total = 0
    local_sonic_edges_scored_total = 0
    local_sonic_gate_rejected_total = 0
    local_sonic_penalty_total = 0.0
    segment_bridge_floors_used: list[float] = []
    segment_backoff_attempts_used: list[int] = []

    # Boundary context tracking for cross-segment min_gap enforcement.
    # Each segment's beam enforces one-track-per-artist WITHIN the segment, but
    # repeats across a segment boundary were previously only blocked 1 position
    # back (hardcoded), so e.g. an artist at the end of segment N and the start of
    # segment N+1 sat far closer than the configured min_gap. Use the resolved
    # min_gap so the last `min_gap` placed artists are carried into the next
    # segment's blocked set. Segments are typically <= min_gap long, so a flat
    # block of the boundary window matches min_gap semantics without over-blocking.
    MIN_GAP_GLOBAL = max(1, int(min_gap))  # Cross-segment min_gap constraint
    recent_boundary_artists: List[str] = []
    global_non_seed_artist_counts: Dict[str, int] = {}

    def _artist_keys_for_cap(idx: int) -> Set[str]:
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
        if use_identity:
            artist_str = ""
            if bundle is not None and bundle.track_artists is not None:
                try:
                    artist_str = str(bundle.track_artists[int(idx)] or "")
                except Exception:
                    artist_str = ""
            if artist_str:
                return set(resolve_artist_identity_keys(artist_str, artist_identity_cfg))
        try:
            artist_key = identity_keys_for_index(bundle, int(idx)).artist_key
        except Exception:
            artist_key = ""
        return {str(artist_key)} if artist_key else set()

    def _recent_artists_for_segment(seg_idx: int) -> Optional[List[str]]:
        artists: List[str] = []
        if seg_idx > 0:
            artists.extend(recent_boundary_artists)
        cap = cfg.max_non_seed_tracks_per_artist
        if isinstance(cap, int) and cap > 0:
            artists.extend(
                artist_key
                for artist_key, count in global_non_seed_artist_counts.items()
                if int(count) >= int(cap)
            )
        return list(dict.fromkeys(artists)) or None

    def _bridge_floor_attempts(initial_floor: float) -> list[float]:
        if not infeasible_handling or not infeasible_handling.enabled:
            return [float(initial_floor)]
        steps = list(infeasible_handling.backoff_steps or ())
        if not steps:
            cur = float(initial_floor)
            while cur >= float(infeasible_handling.min_bridge_floor) - 1e-9:
                steps.append(round(cur, 2))
                cur -= 0.01
        attempts: list[float] = [float(initial_floor)]
        for v in steps:
            if not isinstance(v, (int, float)):
                continue
            f = float(v)
            if f < float(initial_floor) and f >= float(infeasible_handling.min_bridge_floor) - 1e-9:
                attempts.append(float(f))
        attempts = list(dict.fromkeys(attempts))
        max_attempts = max(1, int(infeasible_handling.max_attempts_per_segment))
        return attempts[:max_attempts]

    def _transition_floor_attempts(initial_floor: float) -> list[float]:
        """Return transition_floor values to try in order (initial first, then lower).

        Only relaxes when infeasible_handling is enabled and
        transition_floor_relaxation_enabled is True.  Steps down by 0.07
        from initial_floor to min_transition_floor (up to 3 extra attempts),
        e.g. 0.35→0.28→0.21→0.20 for initial=0.35, min=0.20.
        """
        if (
            not infeasible_handling
            or not infeasible_handling.enabled
            or not infeasible_handling.transition_floor_relaxation_enabled
        ):
            return [float(initial_floor)]
        min_t = float(infeasible_handling.min_transition_floor)
        if min_t >= float(initial_floor) - 1e-9:
            return [float(initial_floor)]
        attempts: list[float] = [float(initial_floor)]
        cur = round(float(initial_floor) - 0.07, 2)
        while cur > min_t + 1e-9 and len(attempts) < 4:
            attempts.append(cur)
            cur = round(cur - 0.07, 2)
        if not any(abs(a - min_t) < 1e-9 for a in attempts):
            attempts.append(min_t)
        return attempts

    def _run_segment_backoff_attempts(
        *,
        cfg_attempt_base: PierBridgeConfig,
        segment_allow_detours: bool,
        segment_g_targets: Optional[list[np.ndarray]],
        segment_g_targets_dense: Optional[list[np.ndarray]] = None,
        pier_a: int,
        pier_b: int,
        interior_len: int,
        pier_a_id: str,
        pier_b_id: str,
        seg_idx: int,
        recent_boundary_artists: Optional[List[str]],
        transition_floor_override: Optional[float] = None,
        genre_arc_floor_percentile_override: Optional[float] = None,
    ) -> dict[str, Any]:
        cfg = cfg_attempt_base
        if transition_floor_override is not None:
            cfg = replace(cfg, transition_floor=float(transition_floor_override))
        if genre_arc_floor_percentile_override is not None:
            cfg = replace(cfg, genre_arc_floor_percentile=float(genre_arc_floor_percentile_override))
        segment_path: Optional[List[int]] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts = _bridge_floor_attempts(float(cfg.bridge_floor))
        backoff_used_count = 0
        widened_search_used = False
        last_failure_reason: Optional[str] = None

        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0
        local_sonic_stats_segment: Dict[str, Any] = {}
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_waypoint_stats: Dict[str, Any] = {}
        last_edge_components: List[dict] = []

        for floor_attempt_idx, bridge_floor in enumerate(backoff_attempts):
            backoff_used_count = floor_attempt_idx + 1
            widened = bool(
                infeasible_handling
                and infeasible_handling.enabled
                and infeasible_handling.widen_search_on_backoff
                and floor_attempt_idx > 0
            )
            widened_search_used = widened_search_used or widened
            cfg_attempt = replace(cfg, bridge_floor=float(bridge_floor))

            segment_pool_max = int(cfg.segment_pool_max)
            beam_width = cfg.initial_beam_width
            max_expansion_attempts = cfg.max_expansion_attempts
            if widened:
                extra_pool = int(infeasible_handling.extra_neighbors_m) + int(
                    infeasible_handling.extra_bridge_helpers
                )
                segment_pool_max = min(
                    segment_pool_max + extra_pool, int(cfg.max_segment_pool_max)
                )
                beam_width = min(
                    beam_width + int(infeasible_handling.extra_beam_width),
                    cfg.max_beam_width,
                )
                max_expansion_attempts = max_expansion_attempts + int(
                    infeasible_handling.extra_expansion_attempts
                )

            expansions = 0
            pool_size_initial = 0
            pool_size_final = 0
            soft_genre_penalty_hits_segment = 0
            soft_genre_penalty_edges_scored_segment = 0
            local_sonic_stats_segment = {}
            last_failure_reason = None
            expansion_attempts_used = 0
            last_pool_diag: Dict[str, Any] = {}
            last_segment_candidates = []
            last_candidate_artist_keys = {}
            last_arc_stats: Dict[str, Any] = {}
            last_genre_cache_stats: Dict[str, int] = {}
            last_waypoint_stats = {}  # Reset for this backoff attempt
            segment_pool_cache: Optional[Dict[str, Any]] = (
                {} if bool(cfg.dj_pooling_cache_enabled) else None
            )
            last_segment_pool_max = int(segment_pool_max)
            last_beam_width = int(beam_width)

            for attempt in range(max_expansion_attempts):
                pool_diag: Dict[str, Any] = {}
                cand_artist_keys: Dict[int, str] = {}
                arc_stats_segment: Dict[str, Any] = {}
                genre_cache_stats_segment: Dict[str, int] = {}
                waypoint_stats_segment: Dict[str, Any] = {}
                pool_strategy = str(cfg.segment_pool_strategy).strip().lower()
                dj_pooling_strategy = str(cfg.dj_pooling_strategy or "baseline").strip().lower()
                if bool(cfg.dj_bridging_enabled) and dj_pooling_strategy == "dj_union":
                    pool_strategy = "dj_union"

                segment_internal_connectors = internal_connector_indices
                segment_connector_cap = int(internal_connector_max_per_segment)
                if bool(cfg.dj_bridging_enabled) and bool(cfg.dj_connector_bias_enabled) and segment_allow_detours:
                    adventurous = str(cfg.dj_route_shape or "linear").strip().lower() in {"arc", "ladder"}
                    dj_connector_cap = (
                        int(cfg.dj_connector_max_per_segment_adventurous)
                        if adventurous
                        else int(cfg.dj_connector_max_per_segment_linear)
                    )
                    dj_connector_cap = max(0, dj_connector_cap)
                    if dj_connector_cap > 0:
                        available = [int(i) for i in universe if int(i) not in global_used]
                        if allowed_set_indices is not None:
                            allowed = set(int(i) for i in allowed_set_indices)
                            available = [int(i) for i in available if int(i) in allowed]
                        if available:
                            dj_connectors = _select_connector_candidates(
                                available,
                                X_full_norm,
                                pier_a,
                                pier_b,
                                dj_connector_cap,
                            )
                        else:
                            dj_connectors = []
                        pool_diag["dj_connectors_selected"] = int(len(dj_connectors))
                        pool_diag["dj_connectors_injected_count"] = int(len(dj_connectors))
                        if dj_connectors:
                            try:
                                pool_diag["dj_connectors_preview"] = [
                                    str(bundle.track_ids[int(i)])
                                    for i in dj_connectors[:5]
                                ]
                            except Exception:
                                pool_diag["dj_connectors_preview"] = [
                                    str(int(i)) for i in dj_connectors[:5]
                                ]
                            if segment_pool_cache is not None:
                                segment_pool_cache["dj_connectors"] = set(
                                    int(i) for i in dj_connectors
                                )
                        if dj_connectors:
                            if segment_internal_connectors:
                                segment_internal_connectors = set(segment_internal_connectors) | set(dj_connectors)
                            else:
                                segment_internal_connectors = set(dj_connectors)
                            segment_connector_cap = max(segment_connector_cap, dj_connector_cap)

                if pool_strategy == "legacy":
                    neighbors_m = min(
                        int(cfg.initial_neighbors_m) * (2 ** int(attempt)),
                        int(cfg.max_neighbors_m),
                    )
                    bridge_helpers = min(
                        int(cfg.initial_bridge_helpers) * (2 ** int(attempt)),
                        int(cfg.max_bridge_helpers),
                    )
                    pool_diag["pool_strategy"] = "legacy"
                    pool_diag["neighbors_m"] = int(neighbors_m)
                    pool_diag["bridge_helpers"] = int(bridge_helpers)
                    segment_candidates = _build_segment_candidate_pool_legacy(
                        pier_a,
                        pier_b,
                        X_full_norm,
                        universe,
                        global_used,
                        int(neighbors_m),
                        int(bridge_helpers),
                        artist_keys=bundle.artist_keys,
                        bridge_floor=float(bridge_floor),
                        allowed_set=(allowed_set_indices if allowed_set_indices is not None else None),
                        internal_connectors=segment_internal_connectors,
                        internal_connector_cap=segment_connector_cap,
                        internal_connector_priority=internal_connector_priority,
                        diagnostics=pool_diag,
                    )
                    try:
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(
                            bundle, int(pier_a)
                        ).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(
                            bundle, int(pier_b)
                        ).artist_key
                    except Exception:
                        cand_artist_keys = {}
                else:
                    pool_diag["pool_strategy"] = pool_strategy
                    segment_candidates, cand_artist_keys, _cand_title_keys = _build_segment_candidate_pool_scored(
                        pier_a=pier_a,
                        pier_b=pier_b,
                        X_full_norm=X_full_norm,
                        universe_indices=universe,
                        used_track_ids=global_used,
                        bundle=bundle,
                        bridge_floor=float(bridge_floor),
                        segment_pool_max=int(segment_pool_max),
                        allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                        internal_connectors=segment_internal_connectors,
                        internal_connector_cap=segment_connector_cap,
                        internal_connector_priority=internal_connector_priority,
                        seed_artist_key=seed_artist_key,
                        disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                        disallow_seed_artist_in_interiors=bool(cfg.disallow_seed_artist_in_interiors),
                        used_track_keys=used_track_keys,
                        seed_track_keys=seed_track_keys,
                        diagnostics=pool_diag,
                        experiment_bridge_scoring_enabled=bool(
                            cfg.experiment_bridge_scoring_enabled
                        ),
                        experiment_bridge_min_weight=float(
                            cfg.experiment_bridge_min_weight
                        ),
                        experiment_bridge_balance_weight=float(
                            cfg.experiment_bridge_balance_weight
                        ),
                        pool_strategy=str(pool_strategy),
                        interior_length=int(interior_len),
                        progress_arc_enabled=bool(cfg.progress_arc_enabled),
                        progress_arc_shape=str(cfg.progress_arc_shape),
                        X_genre_norm=X_genre_norm,
                        X_genre_norm_idf=X_genre_norm_idf,
                        genre_targets=segment_g_targets,
                        pool_k_local=int(cfg.dj_pooling_k_local),
                        pool_k_toward=int(cfg.dj_pooling_k_toward),
                        pool_k_genre=int(cfg.dj_pooling_k_genre),
                        pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                        pool_step_stride=int(cfg.dj_pooling_step_stride),
                        pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                        pooling_cache=segment_pool_cache,
                        pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                        genre_pool_transition_blend=float(cfg.dj_genre_pool_transition_blend),  # Task D
                        collapse_by_artist=bool(cfg.collapse_segment_pool_by_artist),
                        X_genre_dense=getattr(bundle, "X_genre_dense", None),
                        genre_bridge_weight=float(getattr(cfg, "segment_pool_genre_weight", 0.0)),
                    )
                    try:
                        cand_artist_keys = dict(cand_artist_keys)
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(
                            bundle, int(pier_a)
                        ).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(
                            bundle, int(pier_b)
                        ).artist_key
                    except Exception:
                        cand_artist_keys = {}
                    if (
                        len(segment_candidates) < interior_len
                        and bool(cfg.disallow_seed_artist_in_interiors)
                        and seed_artist_key
                    ):
                        relaxed_candidates, relaxed_artist_keys, _relaxed_title_keys = _build_segment_candidate_pool_scored(
                            pier_a=pier_a,
                            pier_b=pier_b,
                            X_full_norm=X_full_norm,
                            universe_indices=universe,
                            used_track_ids=global_used,
                            bundle=bundle,
                            bridge_floor=float(bridge_floor),
                            segment_pool_max=int(segment_pool_max),
                            allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                            internal_connectors=segment_internal_connectors,
                            internal_connector_cap=segment_connector_cap,
                            internal_connector_priority=internal_connector_priority,
                            seed_artist_key=seed_artist_key,
                            disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                            disallow_seed_artist_in_interiors=False,
                            used_track_keys=used_track_keys,
                            seed_track_keys=seed_track_keys,
                            diagnostics=None,
                            experiment_bridge_scoring_enabled=bool(
                                cfg.experiment_bridge_scoring_enabled
                            ),
                            experiment_bridge_min_weight=float(
                                cfg.experiment_bridge_min_weight
                            ),
                            experiment_bridge_balance_weight=float(
                                cfg.experiment_bridge_balance_weight
                            ),
                            pool_strategy=str(pool_strategy),
                            interior_length=int(interior_len),
                            progress_arc_enabled=bool(cfg.progress_arc_enabled),
                            progress_arc_shape=str(cfg.progress_arc_shape),
                            X_genre_norm=X_genre_norm,
                            X_genre_norm_idf=X_genre_norm_idf,
                            genre_targets=segment_g_targets,
                            pool_k_local=int(cfg.dj_pooling_k_local),
                            pool_k_toward=int(cfg.dj_pooling_k_toward),
                            pool_k_genre=int(cfg.dj_pooling_k_genre),
                            pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                            pool_step_stride=int(cfg.dj_pooling_step_stride),
                            pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                            pooling_cache=segment_pool_cache,
                            pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                            genre_pool_transition_blend=float(cfg.dj_genre_pool_transition_blend),  # Task D
                            collapse_by_artist=bool(cfg.collapse_segment_pool_by_artist),
                            X_genre_dense=getattr(bundle, "X_genre_dense", None),
                            genre_bridge_weight=float(getattr(cfg, "segment_pool_genre_weight", 0.0)),
                        )
                        if len(relaxed_candidates) > len(segment_candidates):
                            segment_candidates = relaxed_candidates
                            cand_artist_keys = dict(relaxed_artist_keys)
                            pool_diag["relaxed_seed_artist_in_interiors"] = True
                            pool_diag["relaxed_seed_artist_pool_size"] = int(
                                len(relaxed_candidates)
                            )
                            warnings.append(
                                {
                                    "type": "relax_seed_artist_in_interiors",
                                    "scope": "segment",
                                    "segment_index": int(seg_idx),
                                    "message": (
                                        "Relaxed seed-artist exclusion in bridge interiors "
                                        "due to insufficient candidates."
                                    ),
                                }
                            )
                    if (
                        bool(cfg.dj_bridging_enabled)
                        and str(cfg.dj_pooling_strategy or "baseline")
                        .strip()
                        .lower()
                        == "dj_union"
                        and bool(cfg.dj_pooling_debug_compare_baseline)
                    ):
                        baseline_candidates, _, _ = _build_segment_candidate_pool_scored(
                            pier_a=pier_a,
                            pier_b=pier_b,
                            X_full_norm=X_full_norm,
                            universe_indices=universe,
                            used_track_ids=global_used,
                            bundle=bundle,
                            bridge_floor=float(bridge_floor),
                            segment_pool_max=int(segment_pool_max),
                            allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                            internal_connectors=segment_internal_connectors,
                            internal_connector_cap=segment_connector_cap,
                            internal_connector_priority=internal_connector_priority,
                            seed_artist_key=seed_artist_key,
                            disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                            disallow_seed_artist_in_interiors=bool(cfg.disallow_seed_artist_in_interiors),
                            used_track_keys=used_track_keys,
                            seed_track_keys=seed_track_keys,
                            diagnostics=None,
                            experiment_bridge_scoring_enabled=bool(
                                cfg.experiment_bridge_scoring_enabled
                            ),
                            experiment_bridge_min_weight=float(
                                cfg.experiment_bridge_min_weight
                            ),
                            experiment_bridge_balance_weight=float(
                                cfg.experiment_bridge_balance_weight
                            ),
                            pool_strategy=str(pool_strategy),
                            interior_length=int(interior_len),
                            progress_arc_enabled=bool(cfg.progress_arc_enabled),
                            progress_arc_shape=str(cfg.progress_arc_shape),
                            X_genre_norm=X_genre_norm,
                            X_genre_norm_idf=X_genre_norm_idf,
                            genre_targets=segment_g_targets,
                            pool_k_local=int(cfg.dj_pooling_k_local),
                            pool_k_toward=int(cfg.dj_pooling_k_toward),
                            pool_k_genre=int(cfg.dj_pooling_k_genre),
                            pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                            pool_step_stride=int(cfg.dj_pooling_step_stride),
                            pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                            pooling_cache=segment_pool_cache,
                            pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                            genre_pool_transition_blend=float(cfg.dj_genre_pool_transition_blend),  # Task D
                            collapse_by_artist=bool(cfg.collapse_segment_pool_by_artist),
                            X_genre_dense=getattr(bundle, "X_genre_dense", None),
                            genre_bridge_weight=float(getattr(cfg, "segment_pool_genre_weight", 0.0)),
                        )
                        if segment_pool_cache is not None:
                            segment_pool_cache["dj_baseline_pool"] = set(
                                int(i) for i in baseline_candidates
                            )

                if segment_candidates and recent_boundary_artists:
                    blocked_artist_keys = set(str(k) for k in recent_boundary_artists if k)
                    if blocked_artist_keys:
                        before_artist_gate = len(segment_candidates)
                        segment_candidates = [
                            int(i)
                            for i in segment_candidates
                            if not (_artist_keys_for_cap(int(i)) & blocked_artist_keys)
                        ]
                        removed_by_artist_gate = before_artist_gate - len(segment_candidates)
                        if removed_by_artist_gate > 0:
                            pool_diag["removed_by_global_artist_gate"] = int(
                                removed_by_artist_gate
                            )

                if segment_candidates:
                    last_segment_candidates = list(segment_candidates)
                    last_candidate_artist_keys = dict(cand_artist_keys or {})
                last_pool_diag.update(pool_diag)

                # TASK A: Track pool_before_gating (after merge, before gates)
                pool_size_initial = len(segment_candidates) if segment_candidates else 0

                if not segment_candidates or len(segment_candidates) < int(interior_len):
                    last_failure_reason = f"pool_after_gate {len(segment_candidates)} < interior_len {interior_len}"
                    pool_size_final = 0
                else:
                    _edge_components_buf: Dict[str, Any] = {}
                    segment_path, soft_genre_penalty_hits_segment, soft_genre_penalty_edges_scored_segment, beam_failure_reason = _beam_search_segment(
                        pier_a,
                        pier_b,
                        interior_len,
                        segment_candidates,
                        X_full_tr_norm,
                        X_full_norm,
                        X_start_tr_norm,
                        X_mid_tr_norm,
                        X_end_tr_norm,
                        X_genre_norm,
                        cfg_attempt,
                        beam_width,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        X_genre_dense=X_genre_dense,
                        genre_idf=genre_idf,
                        genre_vocab=genre_vocab,
                        artist_key_by_idx=(cand_artist_keys if cand_artist_keys else None),
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        arc_stats=arc_stats_segment,
                        genre_cache_stats=genre_cache_stats_segment,
                        # Beam arc vote uses the 64-dim dense targets when steering is on;
                        # falls back to the 893-dim dj-bridging targets otherwise. Keeping
                        # these separate from the pooling's genre_targets prevents a
                        # 893-vs-64 dimension crash when both systems are enabled.
                        g_targets_override=(
                            segment_g_targets_dense
                            if segment_g_targets_dense is not None
                            else segment_g_targets
                        ),
                        waypoint_stats=waypoint_stats_segment,
                        local_sonic_stats=local_sonic_stats_segment,
                        edge_components_out=_edge_components_buf,
                        transition_metric_context=transition_metric_context,
                        perceptual_bpm=perceptual_bpm,
                        tempo_stability=tempo_stability_arr,
                        onset_rate=onset_rate,
                        rhythm_matrix=rhythm_matrix,
                        pair_sim_provider=pair_sim_provider,
                    )
                    last_failure_reason = beam_failure_reason
                    if segment_path is not None:
                        last_edge_components = list(_edge_components_buf.get("components") or [])
                        # Capture waypoint stats for successful path
                        last_waypoint_stats = dict(waypoint_stats_segment)

                        # TASK A: Extract pool_after_gating count from waypoint_stats
                        pool_size_final = int(waypoint_stats_segment.get("pool_after_gating_count", 0))

                        baseline_pool = None
                        if (
                            segment_pool_cache is not None
                            and "dj_baseline_pool" in segment_pool_cache
                        ):
                            baseline_pool = segment_pool_cache.get(
                                "dj_baseline_pool"
                            )
                        elif pool_strategy != "dj_union":
                            baseline_pool = set(int(i) for i in segment_candidates)
                        sources = None
                        if segment_pool_cache is not None:
                            sources = segment_pool_cache.get("dj_pool_sources")
                        last_pool_diag.update(
                            _compute_chosen_source_counts(
                                segment_path,
                                sources=sources,
                                baseline_pool=baseline_pool,
                                log_per_track=bool(cfg.dj_diagnostics_pool_verbose),  # Task C
                            )
                        )
                        if segment_pool_cache is not None and "dj_connectors" in segment_pool_cache:
                            connector_set = segment_pool_cache.get("dj_connectors", set())
                            chosen_connectors = [
                                int(i) for i in segment_path if int(i) in connector_set
                            ]
                            last_pool_diag["dj_connectors_chosen_count"] = int(
                                len(chosen_connectors)
                            )
                            if chosen_connectors:
                                try:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(bundle.track_ids[int(i)])
                                        for i in chosen_connectors[:5]
                                    ]
                                except Exception:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(int(i)) for i in chosen_connectors[:5]
                                    ]

                if segment_path is not None:
                    break

                expansions += 1
                if str(cfg.segment_pool_strategy).strip().lower() != "legacy":
                    segment_pool_max = min(int(segment_pool_max) * 2, int(cfg.max_segment_pool_max))
                beam_width = min(int(beam_width) * 2, int(cfg.max_beam_width))

                if infeasible_handling and infeasible_handling.enabled:
                    if str(cfg.segment_pool_strategy).strip().lower() == "legacy":
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d strategy=legacy beam=%d)",
                            seg_idx,
                            attempt + 1,
                            beam_width,
                        )
                    else:
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d segment_pool_max=%d beam=%d)",
                            seg_idx,
                            attempt + 1,
                            int(segment_pool_max),
                            beam_width,
                        )

            if infeasible_handling and infeasible_handling.enabled:
                logger.info(
                    "Segment %d attempt %d: bridge_floor=%.2f widened=%s pool_after_gate=%d",
                    seg_idx,
                    floor_attempt_idx + 1,
                    float(bridge_floor),
                    widened,
                    int(len(last_segment_candidates)),
                )

            if audit_enabled:
                top_rows, dists = _summarize_candidates_for_audit(
                    candidates=last_segment_candidates,
                    pier_a=pier_a,
                    pier_b=pier_b,
                    X_full_norm=X_full_norm,
                    X_full_tr_norm=X_full_tr_norm,
                    X_start_tr_norm=X_start_tr_norm,
                    X_mid_tr_norm=X_mid_tr_norm,
                    X_end_tr_norm=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_attempt,
                    bundle=bundle,
                    internal_connector_indices=internal_connector_indices,
                    top_k=top_k,
                )
                audit_events.append(
                    RunAuditEvent(
                        kind="segment_attempt",
                        ts_utc=now_utc_iso(),
                        payload={
                            "segment_index": int(seg_idx),
                            "segment_header": f"{pier_a_id} -> {pier_b_id} (interior={interior_len})",
                            "attempt_number": int(floor_attempt_idx + 1),
                            "expansion_attempts": int(expansion_attempts_used),
                            "bridge_floor": float(bridge_floor),
                            "widened": bool(widened),
                            "segment_pool_strategy": str(
                                last_pool_diag.get("pool_strategy", cfg.segment_pool_strategy)
                            ),
                            "segment_pool_max": int(last_segment_pool_max),
                            "beam_width": int(last_beam_width),
                            "pool_counts": dict(last_pool_diag),
                            "pool_size_initial": int(pool_size_initial),
                            "pool_size_final": int(pool_size_final),
                            "distributions": dists,
                            "soft_genre_penalty": {
                                "edges_scored": int(soft_genre_penalty_edges_scored_segment),
                                "hits": int(soft_genre_penalty_hits_segment),
                                "threshold": float(cfg_attempt.genre_penalty_threshold),
                                "strength": float(cfg_attempt.genre_penalty_strength),
                            },
                            "local_sonic_edge_penalty": dict(local_sonic_stats_segment),
                            "genre_cache": dict(last_genre_cache_stats),
                            "progress_arc": dict(last_arc_stats),
                        },
                    )
                )

            if segment_path is not None:
                chosen_bridge_floor = float(bridge_floor)
                beam_width_used = int(last_beam_width)
                break

            last_segment_pool_cache = segment_pool_cache

        return {
            "segment_path": segment_path,
            "chosen_bridge_floor": float(chosen_bridge_floor),
            "backoff_attempts": [float(x) for x in backoff_attempts],
            "backoff_used_count": int(backoff_used_count),
            "widened_search_used": bool(widened_search_used),
            "expansions": int(expansions),
            "pool_size_initial": int(pool_size_initial),
            "pool_size_final": int(pool_size_final),
            "beam_width_used": int(beam_width_used),
            "soft_genre_penalty_hits_segment": int(soft_genre_penalty_hits_segment),
            "soft_genre_penalty_edges_scored_segment": int(soft_genre_penalty_edges_scored_segment),
            "local_sonic_stats_segment": dict(local_sonic_stats_segment),
            "last_failure_reason": (str(last_failure_reason) if last_failure_reason else None),
            "last_segment_candidates": list(last_segment_candidates),
            "last_candidate_artist_keys": dict(last_candidate_artist_keys),
            "segment_pool_cache": dict(last_segment_pool_cache or {}),
            "last_waypoint_stats": dict(last_waypoint_stats),
            "last_pool_diag": dict(last_pool_diag),
            "edge_components": list(last_edge_components),
        }

    for seg_idx in range(num_segments):
        pier_a = ordered_seeds[seg_idx]
        pier_b = ordered_seeds[seg_idx + 1]
        interior_len = segment_lengths[seg_idx]

        pier_a_id = str(bundle.track_ids[pier_a])
        pier_b_id = str(bundle.track_ids[pier_b])

        logger.info("Building segment %d: %s -> %s (interior=%d)",
                   seg_idx, pier_a_id, pier_b_id, interior_len)

        segment_g_targets: Optional[list[np.ndarray]] = None
        segment_g_targets_dense: Optional[list[np.ndarray]] = None
        segment_ladder_diag: dict[str, Any] = {}
        segment_far_stats: Optional[dict[str, Optional[float]]] = None
        segment_is_far = False
        if bool(cfg.dj_bridging_enabled) and X_genre_norm is not None:
            # Phase 3 fix: genre_vocab is optional for vector mode, always try to build targets
            genre_vocab = getattr(bundle, "genre_vocab", None)
            segment_g_targets = _build_genre_targets(
                pier_a=pier_a,
                pier_b=pier_b,
                interior_length=interior_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=genre_vocab,  # Can be None for vector mode
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                ladder_diag=segment_ladder_diag,
                X_genre_raw=X_genre_raw,
                X_genre_smoothed=X_genre_smoothed,
                genre_idf=genre_idf,
            )
            segment_far_stats = _segment_far_stats(
                pier_a=pier_a,
                pier_b=pier_b,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                universe=universe,
                used_track_ids=global_used,
                bridge_floor=float(cfg.bridge_floor),
            )
            if segment_far_stats:
                sonic_sim = segment_far_stats.get("sonic_sim")
                genre_sim = segment_far_stats.get("genre_sim")
                scarcity = segment_far_stats.get("connector_scarcity")
                if sonic_sim is not None and (1.0 - float(sonic_sim)) > float(cfg.dj_far_threshold_sonic):
                    segment_is_far = True
                if genre_sim is not None and (1.0 - float(genre_sim)) > float(cfg.dj_far_threshold_genre):
                    segment_is_far = True
                if scarcity is not None and float(scarcity) < float(cfg.dj_far_threshold_connector_scarcity):
                    segment_is_far = True

        # Genre-arc steering: build per-step g_targets that feed the beam's first-class
        # arc vote (via g_targets_override). Two sources, selected by genre_steering_source:
        #   - "taxonomy": route the arc through the SP3a taxonomy graph (hub-damped);
        #     targets live in the genre-vocab space (beam scores against X_genre_norm).
        #   - "dense" (legacy): interpolate the 64-dim dense PMI-SVD vectors (beam scores
        #     against X_genre_dense). Kept SEPARATE from segment_g_targets (dj-bridging
        #     pooling) to avoid dimension clashes.
        _steering_source = str(getattr(cfg, "genre_steering_source", "dense"))
        if (
            bool(cfg.genre_steering_enabled)
            and _steering_source == "taxonomy"
            and getattr(bundle, "X_genre_raw", None) is not None
            and getattr(bundle, "genre_vocab", None) is not None
        ):
            from src.playlist.pier_bridge.taxonomy_steering import (
                build_taxonomy_genre_targets,
                get_taxonomy_steering,
            )
            _tax_diag: dict[str, Any] = {}
            segment_g_targets_dense = build_taxonomy_genre_targets(
                pier_a=pier_a,
                pier_b=pier_b,
                interior_length=interior_len,
                X_genre_raw=bundle.X_genre_raw,
                genre_vocab=bundle.genre_vocab,
                steering=get_taxonomy_steering(),
                top_labels=int(cfg.dj_ladder_top_labels),
                min_label_weight=float(cfg.dj_ladder_min_label_weight),
                smooth_top_k=int(cfg.dj_ladder_smooth_top_k),
                smooth_min_sim=float(cfg.dj_ladder_smooth_min_sim),
                max_steps=int(cfg.dj_ladder_max_steps),
                genre_track_counts=genre_track_counts,
                min_waypoint_mass=int(getattr(cfg, "taxonomy_waypoint_min_library_mass", 0)),
                ladder_diag=_tax_diag,
            )
            if segment_g_targets_dense is not None and _tax_diag.get("taxonomy_waypoint_labels"):
                segment_ladder_diag.update(_tax_diag)
                logger.info(
                    "Genre steering [taxonomy]: %s -> %s via %s",
                    bundle.track_ids[pier_a],
                    bundle.track_ids[pier_b],
                    _tax_diag.get("taxonomy_waypoint_labels"),
                )
            elif segment_g_targets_dense is None:
                logger.info(
                    "Genre steering [taxonomy]: no taxonomy path for segment %s -> %s "
                    "(uncovered genres); genre arc inactive this segment",
                    bundle.track_ids[pier_a],
                    bundle.track_ids[pier_b],
                )

        # Dense steering (legacy). Skipped entirely in taxonomy mode so the beam's
        # genre-vocab arc vote never receives 64-dim dense targets.
        if (
            _steering_source != "taxonomy"
            and bool(cfg.genre_steering_enabled)
            and getattr(bundle, "X_genre_dense", None) is not None
        ):
            labels_a = _select_top_genre_labels(
                bundle.X_genre_raw[pier_a], bundle.genre_vocab,
                top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight),
            ) if getattr(bundle, "genre_vocab", None) is not None else None
            labels_b = _select_top_genre_labels(
                bundle.X_genre_raw[pier_b], bundle.genre_vocab,
                top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight),
            ) if getattr(bundle, "genre_vocab", None) is not None else None
            segment_g_targets_dense = build_dense_genre_targets(
                bundle.X_genre_dense[pier_a], bundle.X_genre_dense[pier_b],
                interior_length=interior_len, route=str(cfg.dj_route_shape or "linear"),
                genre_emb=getattr(bundle, "genre_emb", None),
                genre_vocab=list(bundle.genre_vocab) if getattr(bundle, "genre_vocab", None) is not None else None,
                genre_graph=genre_graph_arc, labels_a=labels_a, labels_b=labels_b,
                max_steps=int(cfg.dj_ladder_max_steps),
            )

        segment_allow_detours = bool(cfg.dj_allow_detours_when_far) and segment_is_far

        relaxation_enabled = (
            bool(cfg.dj_bridging_enabled)
            and bool(cfg.dj_relaxation_enabled)
            and str(cfg.dj_route_shape or "linear").strip().lower() == "ladder"
        )
        relaxation_attempts = (
            _build_dj_relaxation_attempts(cfg)
            if relaxation_enabled
            else [{"label": "baseline", "cfg": cfg, "changes": [], "force_allow_detours": False}]
        )
        segment_relaxation_attempts: list[dict[str, Any]] = []
        relaxation_success_attempt: Optional[int] = None
        cfg_base = cfg
        cfg_used_for_segment = cfg
        segment_allow_detours_base = segment_allow_detours
        segment_path: Optional[List[int]] = None
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_failure_reason: Optional[str] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts: list[float] = [float(cfg.bridge_floor)]
        backoff_used_count = 0
        widened_search_used = False
        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0
        local_sonic_stats_segment: Dict[str, Any] = {}

        for relax_idx, relax in enumerate(relaxation_attempts):
            cfg = relax["cfg"]
            cfg_used_for_segment = cfg
            attempt_allow_detours = segment_allow_detours_base or bool(relax.get("force_allow_detours"))
            attempt_result = _run_segment_backoff_attempts(
                cfg_attempt_base=cfg,
                segment_allow_detours=attempt_allow_detours,
                segment_g_targets=segment_g_targets,
                segment_g_targets_dense=segment_g_targets_dense,
                pier_a=pier_a,
                pier_b=pier_b,
                interior_len=interior_len,
                pier_a_id=pier_a_id,
                pier_b_id=pier_b_id,
                seg_idx=seg_idx,
                recent_boundary_artists=_recent_artists_for_segment(seg_idx),
            )
            segment_path = attempt_result["segment_path"]
            chosen_bridge_floor = float(attempt_result["chosen_bridge_floor"])
            backoff_used_count = int(attempt_result["backoff_used_count"])
            backoff_attempts = list(attempt_result.get("backoff_attempts") or [])
            widened_search_used = bool(attempt_result["widened_search_used"])
            expansions = int(attempt_result["expansions"])
            pool_size_initial = int(attempt_result["pool_size_initial"])
            pool_size_final = int(attempt_result["pool_size_final"])
            beam_width_used = int(attempt_result["beam_width_used"])
            soft_genre_penalty_hits_segment = int(attempt_result["soft_genre_penalty_hits_segment"])
            soft_genre_penalty_edges_scored_segment = int(attempt_result["soft_genre_penalty_edges_scored_segment"])
            local_sonic_stats_segment = dict(attempt_result.get("local_sonic_stats_segment", {}))
            last_failure_reason = attempt_result["last_failure_reason"]
            last_segment_candidates = list(attempt_result["last_segment_candidates"])
            last_candidate_artist_keys = dict(attempt_result["last_candidate_artist_keys"])
            pool_cache = attempt_result.get("segment_pool_cache")
            last_segment_pool_cache = dict(pool_cache) if pool_cache is not None else None
            last_waypoint_stats = dict(attempt_result.get("last_waypoint_stats", {}))
            last_pool_diag = dict(attempt_result.get("last_pool_diag", {}))
            segment_edge_components = list(attempt_result.get("edge_components") or [])

            segment_relaxation_attempts.append({
                "attempt_index": int(relax_idx),
                "label": str(relax.get("label", "")),
                "changes": list(relax.get("changes") or []),
                "failure_reason": (str(last_failure_reason) if segment_path is None else None),
            })
            if segment_path is not None:
                relaxation_success_attempt = int(relax_idx)
                break

        cfg = cfg_base
        segment_allow_detours = segment_allow_detours_base

        # Transition-floor relaxation tier: if all bridge_floor backoffs exhausted,
        # progressively lower transition_floor before declaring infeasibility.
        if segment_path is None:
            _t_attempts = _transition_floor_attempts(float(cfg_base.transition_floor))
            for _t_floor in _t_attempts[1:]:  # first value already tried in relax loop above
                for _relax in relaxation_attempts:
                    _t_result = _run_segment_backoff_attempts(
                        cfg_attempt_base=_relax["cfg"],
                        segment_allow_detours=segment_allow_detours_base or bool(_relax.get("force_allow_detours")),
                        segment_g_targets=segment_g_targets,
                        segment_g_targets_dense=segment_g_targets_dense,
                        pier_a=pier_a,
                        pier_b=pier_b,
                        interior_len=interior_len,
                        pier_a_id=pier_a_id,
                        pier_b_id=pier_b_id,
                        seg_idx=seg_idx,
                        recent_boundary_artists=_recent_artists_for_segment(seg_idx),
                        transition_floor_override=float(_t_floor),
                    )
                    if _t_result["segment_path"] is not None:
                        segment_path = _t_result["segment_path"]
                        chosen_bridge_floor = float(_t_result["chosen_bridge_floor"])
                        backoff_used_count = int(_t_result["backoff_used_count"])
                        backoff_attempts = list(_t_result.get("backoff_attempts") or [])
                        widened_search_used = bool(_t_result["widened_search_used"])
                        expansions = int(_t_result["expansions"])
                        pool_size_initial = int(_t_result["pool_size_initial"])
                        pool_size_final = int(_t_result["pool_size_final"])
                        beam_width_used = int(_t_result["beam_width_used"])
                        soft_genre_penalty_hits_segment = int(_t_result["soft_genre_penalty_hits_segment"])
                        soft_genre_penalty_edges_scored_segment = int(_t_result["soft_genre_penalty_edges_scored_segment"])
                        local_sonic_stats_segment = dict(_t_result.get("local_sonic_stats_segment", {}))
                        last_failure_reason = _t_result["last_failure_reason"]
                        last_segment_candidates = list(_t_result["last_segment_candidates"])
                        last_candidate_artist_keys = dict(_t_result["last_candidate_artist_keys"])
                        pool_cache = _t_result.get("segment_pool_cache")
                        last_segment_pool_cache = dict(pool_cache) if pool_cache is not None else None
                        last_waypoint_stats = dict(_t_result.get("last_waypoint_stats", {}))
                        last_pool_diag = dict(_t_result.get("last_pool_diag", {}))
                        segment_edge_components = list(_t_result.get("edge_components") or [])
                        break
                if segment_path is not None:
                    break

        # Genre-arc-floor relaxation tier: if the transition-floor tier still
        # could not build the segment, progressively lower the genre-arc floor
        # percentile toward infeasible_handling.min_genre_arc_percentile so
        # genre-sparse seeds don't go infeasible. Gated on steering +
        # infeasible_handling + genre_arc_relaxation.
        if segment_path is None and bool(getattr(cfg_base, "genre_steering_enabled", False)) \
           and infeasible_handling and infeasible_handling.enabled \
           and infeasible_handling.genre_arc_relaxation_enabled:
            _gfloors = relax_percentile(
                float(cfg_base.genre_arc_floor_percentile),
                float(infeasible_handling.min_genre_arc_percentile),
                step=0.15,
            )
            # Niche-genre seeds (jazz, hyperpop) often need the genre-arc floor AND the
            # transition floor relaxed *together* — the genre-coherent candidates and the
            # sonically-reachable ones only overlap once both gates ease. The earlier tiers
            # relax these dimensions independently and never explore the joint relaxation,
            # so sweep transition_floor inside the arc-floor loop here.
            _t_attempts_arc = _transition_floor_attempts(float(cfg_base.transition_floor))
            for _gf in _gfloors[1:]:  # first value already tried in the relax loop above
                for _t_floor in _t_attempts_arc:
                    for _relax in relaxation_attempts:
                        _g_result = _run_segment_backoff_attempts(
                            cfg_attempt_base=_relax["cfg"],
                            segment_allow_detours=segment_allow_detours_base or bool(_relax.get("force_allow_detours")),
                            segment_g_targets=segment_g_targets,
                            segment_g_targets_dense=segment_g_targets_dense,
                            pier_a=pier_a,
                            pier_b=pier_b,
                            interior_len=interior_len,
                            pier_a_id=pier_a_id,
                            pier_b_id=pier_b_id,
                            seg_idx=seg_idx,
                            recent_boundary_artists=_recent_artists_for_segment(seg_idx),
                            genre_arc_floor_percentile_override=float(_gf),
                            transition_floor_override=float(_t_floor),
                        )
                        if _g_result["segment_path"] is not None:
                            break
                    if _g_result["segment_path"] is not None:
                        segment_path = _g_result["segment_path"]
                        chosen_bridge_floor = float(_g_result["chosen_bridge_floor"])
                        backoff_used_count = int(_g_result["backoff_used_count"])
                        backoff_attempts = list(_g_result.get("backoff_attempts") or [])
                        widened_search_used = bool(_g_result["widened_search_used"])
                        expansions = int(_g_result["expansions"])
                        pool_size_initial = int(_g_result["pool_size_initial"])
                        pool_size_final = int(_g_result["pool_size_final"])
                        beam_width_used = int(_g_result["beam_width_used"])
                        soft_genre_penalty_hits_segment = int(_g_result["soft_genre_penalty_hits_segment"])
                        soft_genre_penalty_edges_scored_segment = int(_g_result["soft_genre_penalty_edges_scored_segment"])
                        local_sonic_stats_segment = dict(_g_result.get("local_sonic_stats_segment", {}))
                        last_failure_reason = _g_result["last_failure_reason"]
                        last_segment_candidates = list(_g_result["last_segment_candidates"])
                        last_candidate_artist_keys = dict(_g_result["last_candidate_artist_keys"])
                        pool_cache = _g_result.get("segment_pool_cache")
                        last_segment_pool_cache = dict(pool_cache) if pool_cache is not None else None
                        last_waypoint_stats = dict(_g_result.get("last_waypoint_stats", {}))
                        last_pool_diag = dict(_g_result.get("last_pool_diag", {}))
                        segment_edge_components = list(_g_result.get("edge_components") or [])
                        break
                if segment_path is not None:
                    break

        micro_pier_diag: dict[str, Any] = {}
        if relaxation_enabled and bool(cfg.dj_relaxation_emit_warnings):
            warnings.append({
                "type": "dj_relaxation_attempts",
                "scope": "segment",
                "segment_index": int(seg_idx),
                "message": "DJ relaxation attempts executed.",
                "attempts": list(segment_relaxation_attempts),
                "success_attempt": relaxation_success_attempt,
            })

        if segment_path is None:
            if _should_attempt_micro_pier(relaxation_enabled=relaxation_enabled, segment_path=segment_path):
                if bool(cfg.dj_micro_piers_enabled):
                    candidates = _micro_pier_candidate_pool(
                        cfg.dj_micro_piers_candidate_source,
                        last_segment_candidates,
                        last_segment_pool_cache,
                    )
                    micro_path = _attempt_micro_pier_split(
                        pier_a=pier_a,
                        pier_b=pier_b,
                        interior_length=interior_len,
                        candidates=candidates,
                        X_full=X_full_tr_norm,
                        X_full_norm=X_full_norm,
                        X_start=X_start_tr_norm,
                        X_mid=X_mid_tr_norm,
                        X_end=X_end_tr_norm,
                        X_genre_norm=X_genre_norm,
                        cfg=cfg_used_for_segment,
                        beam_width=int(beam_width_used),
                        artist_key_by_idx=last_candidate_artist_keys,
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=_recent_artists_for_segment(seg_idx),
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        warnings=warnings,
                        X_genre_vocab=getattr(bundle, "genre_vocab", None),
                        genre_graph=genre_graph,
                        micro_diag=micro_pier_diag,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        genre_idf=genre_idf,
                        transition_metric_context=transition_metric_context,
                    )
                    if micro_path is not None and len(micro_path) == interior_len:
                        segment_path = micro_path
                        warnings.append({
                            "type": "micro_pier_fallback",
                            "scope": "segment",
                            "segment_index": int(seg_idx),
                            "message": "Inserted micro-pier due to infeasible bridge; consider lowering genre drift, increasing effort, or adding genre metadata.",
                        })
            elif bool(cfg.dj_bridging_enabled) and bool(cfg.dj_micro_piers_enabled) and segment_allow_detours:
                candidates = _micro_pier_candidate_pool(
                    cfg.dj_micro_piers_candidate_source,
                    last_segment_candidates,
                    last_segment_pool_cache,
                )
                micro_path = _attempt_micro_pier_split(
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_length=interior_len,
                    candidates=candidates,
                    X_full=X_full_tr_norm,
                    X_full_norm=X_full_norm,
                    X_start=X_start_tr_norm,
                    X_mid=X_mid_tr_norm,
                    X_end=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_used_for_segment,
                    beam_width=int(beam_width_used),
                    artist_key_by_idx=last_candidate_artist_keys,
                    seed_artist_key=seed_artist_key,
                    recent_global_artists=_recent_artists_for_segment(seg_idx),
                    durations_ms=bundle.durations_ms,
                    artist_identity_cfg=artist_identity_cfg,
                    bundle=bundle,
                    warnings=warnings,
                    X_genre_vocab=getattr(bundle, "genre_vocab", None),
                    genre_graph=genre_graph,
                    micro_diag=micro_pier_diag,
                    X_genre_norm_idf=X_genre_norm_idf,
                    X_genre_raw=X_genre_raw,
                    X_genre_smoothed=X_genre_smoothed,
                    genre_idf=genre_idf,
                    transition_metric_context=transition_metric_context,
                )
                if micro_path is not None and len(micro_path) == interior_len:
                    segment_path = micro_path

            if audit_enabled:
                audit_events.append(
                    RunAuditEvent(
                        kind="segment_failure",
                        ts_utc=now_utc_iso(),
                        payload={
                            "segment_index": int(seg_idx),
                            "failure_reason": str(last_failure_reason or "segment infeasible"),
                            "attempted_bridge_floors": [float(x) for x in backoff_attempts],
                        },
                    )
                )
            if infeasible_handling and infeasible_handling.enabled:
                failure = f"Segment {seg_idx} infeasible under bridge_floor backoff (attempted={backoff_attempts}; last_reason={last_failure_reason})"
            else:
                failure = f"Segment {seg_idx} infeasible under bridge_floor={cfg.bridge_floor}"
            logger.error(failure)
            return PierBridgeResult(
                track_ids=[],
                track_indices=[],
                seed_positions=[],
                segment_diagnostics=[],
                stats={},
                success=False,
                failure_reason=failure,
            )

        soft_genre_penalty_hits_total += int(soft_genre_penalty_hits_segment)
        soft_genre_penalty_edges_scored_total += int(soft_genre_penalty_edges_scored_segment)
        local_sonic_penalty_hits_total += int(local_sonic_stats_segment.get("local_sonic_penalty_hits", 0))
        local_sonic_edges_scored_total += int(local_sonic_stats_segment.get("local_sonic_edges_scored", 0))
        local_sonic_gate_rejected_total += int(local_sonic_stats_segment.get("local_sonic_gate_rejected", 0))
        local_sonic_penalty_total += float(local_sonic_stats_segment.get("local_sonic_penalty_total", 0.0))
        if cfg.genre_penalty_strength > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Segment %d: soft_genre_penalty_hits=%d edges_scored=%d threshold=%.2f strength=%.2f",
                seg_idx,
                int(soft_genre_penalty_hits_segment),
                int(soft_genre_penalty_edges_scored_segment),
                float(cfg.genre_penalty_threshold),
                float(cfg.genre_penalty_strength),
            )
        if (
            (cfg.local_sonic_edge_penalty_enabled and cfg.local_sonic_edge_penalty_strength > 0)
            or cfg.local_sonic_edge_floor is not None
        ) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Segment %d: local_sonic_penalty_hits=%d edges_scored=%d gate_rejected=%d threshold=%.2f strength=%.2f floor=%s",
                seg_idx,
                int(local_sonic_stats_segment.get("local_sonic_penalty_hits", 0)),
                int(local_sonic_stats_segment.get("local_sonic_edges_scored", 0)),
                int(local_sonic_stats_segment.get("local_sonic_gate_rejected", 0)),
                float(cfg.local_sonic_edge_penalty_threshold),
                float(cfg.local_sonic_edge_penalty_strength),
                (
                    f"{float(cfg.local_sonic_edge_floor):.2f}"
                    if cfg.local_sonic_edge_floor is not None
                    else "None"
                ),
            )

        # Compute edge scores for diagnostics
        full_segment = [pier_a] + segment_path + [pier_b]
        worst_edge, mean_edge = _compute_edge_scores(
            full_segment,
            X_full_tr_norm,
            X_start_tr_norm,
            X_mid_tr_norm,
            X_end_tr_norm,
            cfg,
            metric_context=transition_metric_context,
        )
        micro_pier_used = "micro_pier_index" in micro_pier_diag
        micro_pier_track_id = None
        if micro_pier_used:
            try:
                micro_pier_track_id = str(
                    bundle.track_ids[int(micro_pier_diag["micro_pier_index"])]
                )
            except Exception:
                micro_pier_track_id = str(micro_pier_diag.get("micro_pier_index"))

        # Record diagnostics
        diagnostics.append(SegmentDiagnostics(
            pier_a_id=pier_a_id,
            pier_b_id=pier_b_id,
            target_length=interior_len,
            actual_length=len(segment_path),
            pool_size_initial=pool_size_initial,
            pool_size_final=pool_size_final,
            expansions=expansions,
            beam_width_used=beam_width_used,
            worst_edge_score=worst_edge,
            mean_edge_score=mean_edge,
            success=segment_path is not None and len(segment_path) == interior_len,
            bridge_floor_used=float(chosen_bridge_floor),
            backoff_attempts_used=int(backoff_used_count),
            widened_search=bool(widened_search_used),
            route_shape=str(segment_ladder_diag.get("route_shape", str(cfg.dj_route_shape or "linear"))),
            ladder_waypoint_labels=list(segment_ladder_diag.get("ladder_waypoint_labels", [])),
            ladder_waypoint_count=int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
            ladder_waypoint_vector_mode=str(segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")),
            ladder_waypoint_vector_stats=list(segment_ladder_diag.get("ladder_waypoint_vector_stats", [])),
            relaxation_attempts=list(segment_relaxation_attempts),
            relaxation_success_attempt=relaxation_success_attempt,
            micro_pier_used=bool(micro_pier_used),
            micro_pier_track_id=micro_pier_track_id,
            micro_pier_metric_value=(
                float(micro_pier_diag.get("micro_pier_metric_value"))
                if micro_pier_diag.get("micro_pier_metric_value") is not None
                else None
            ),
            micro_pier_left_success=micro_pier_diag.get("left_success"),
            micro_pier_right_success=micro_pier_diag.get("right_success"),
            waypoint_enabled=bool(last_waypoint_stats.get("waypoint_enabled", False)),
            mean_waypoint_sim=last_waypoint_stats.get("mean_waypoint_sim"),
            p50_waypoint_sim=last_waypoint_stats.get("p50_waypoint_sim"),
            p90_waypoint_sim=last_waypoint_stats.get("p90_waypoint_sim"),
            min_waypoint_sim=last_waypoint_stats.get("min_waypoint_sim"),
            max_waypoint_sim=last_waypoint_stats.get("max_waypoint_sim"),
            waypoint_delta_applied_count=int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
            mean_waypoint_delta=last_waypoint_stats.get("mean_waypoint_delta"),
        ))
        segment_bridge_floors_used.append(float(chosen_bridge_floor))
        segment_backoff_attempts_used.append(int(backoff_used_count))
        logger.info(
            "Segment %d: %s -> %s bridge_floor=%.2f pool_before=%d pool_after=%d",
            seg_idx, pier_a_id, pier_b_id, float(chosen_bridge_floor), pool_size_initial, pool_size_final,
        )

        # Log waypoint influence stats (Phase 2 diagnostics)
        if last_waypoint_stats.get("waypoint_enabled"):
            logger.info(
                "  Waypoint stats: enabled=True mean_sim=%.3f p50=%.3f p90=%.3f delta_applied=%d/%d mean_delta=%.4f",
                float(last_waypoint_stats.get("mean_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p50_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p90_waypoint_sim", 0.0)),
                int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
                interior_len,
                float(last_waypoint_stats.get("mean_waypoint_delta", 0.0)),
            )

            # Log rank impact metrics (TASK B: opt-in diagnostic)
            rank_impact_results = last_waypoint_stats.get("rank_impact_results", [])
            if rank_impact_results:
                sampled_steps_count = len(rank_impact_results)
                winner_changed_count = sum(1 for r in rank_impact_results if r.get("winner_changed"))
                mean_reordered = float(np.mean([r.get("topK_reordered_count", 0) for r in rank_impact_results]))
                mean_topK = float(np.mean([r.get("topK", 10) for r in rank_impact_results]))
                mean_rank_delta = float(np.mean([r.get("mean_abs_rank_delta", 0.0) for r in rank_impact_results]))

                logger.info(
                    "  Waypoint rank impact: sampled_steps=%d winner_changed=%d/%d topK_reordered=%.1f/%.0f mean_rank_delta=%.1f",
                    sampled_steps_count,
                    winner_changed_count,
                    sampled_steps_count,
                    mean_reordered,
                    mean_topK,
                    mean_rank_delta,
                )

                # Task E: Saturation diagnostics
                mean_sim0 = float(np.mean([r.get("waypoint_sim0", 0.0) for r in rank_impact_results]))
                mean_delta_mean = float(np.mean([r.get("waypoint_delta_mean", 0.0) for r in rank_impact_results]))
                mean_delta_p50 = float(np.mean([r.get("waypoint_delta_p50", 0.0) for r in rank_impact_results]))
                mean_delta_p90 = float(np.mean([r.get("waypoint_delta_p90", 0.0) for r in rank_impact_results]))
                mean_frac_near_cap = float(np.mean([r.get("waypoint_frac_near_cap", 0.0) for r in rank_impact_results]))
                mean_frac_at_cap = float(np.mean([r.get("waypoint_frac_at_cap", 0.0) for r in rank_impact_results]))

                logger.info(
                    "  Waypoint saturation: sim0=%.3f delta(mean=%.4f p50=%.4f p90=%.4f) near_cap=%.1f%% at_cap=%.1f%%",
                    mean_sim0,
                    mean_delta_mean,
                    mean_delta_p50,
                    mean_delta_p90,
                    100.0 * mean_frac_near_cap,
                    100.0 * mean_frac_at_cap,
                )

                # Phase 2: Coverage bonus impact (compare base+waypoint vs full)
                if bool(cfg.dj_genre_use_coverage):
                    coverage_winner_changed_count = 0
                    coverage_mean_bonus = []
                    for r in rank_impact_results:
                        top10_table = r.get("top10_table", [])
                        if top10_table:
                            # Find winner by base+waypoint score (before coverage)
                            base_waypoint_scores = [(entry["cand_idx"], entry["base_score"] + entry["waypoint_delta"])
                                                     for entry in top10_table]
                            base_waypoint_winner = max(base_waypoint_scores, key=lambda x: x[1])[0]
                            # Find winner by full score (after coverage)
                            full_winner = top10_table[0]["cand_idx"]  # Already sorted by base_rank
                            # Actually need to re-sort by full_score to get true full_winner
                            full_scores = [(entry["cand_idx"], entry["full_score"]) for entry in top10_table]
                            full_winner = max(full_scores, key=lambda x: x[1])[0]

                            if base_waypoint_winner != full_winner:
                                coverage_winner_changed_count += 1

                            # Collect mean coverage bonus for this step
                            coverage_bonuses = [entry["coverage_bonus"] for entry in top10_table]
                            coverage_mean_bonus.append(float(np.mean(coverage_bonuses)))

                    if coverage_mean_bonus:
                        logger.info(
                            "  Coverage bonus impact: winner_changed=%d/%d mean_bonus=%.4f",
                            coverage_winner_changed_count,
                            sampled_steps_count,
                            float(np.mean(coverage_mean_bonus)),
                        )

        # Log chosen edge provenance (dj_union) - TASK A: renamed from "Pool sources"
        if last_pool_diag:
            if "chosen_from_local_count" in last_pool_diag or "dj_pool_strategy" in last_pool_diag:
                # Legacy exclusive counts (priority-based)
                logger.info(
                    "  Chosen edge provenance (exclusive): strategy=%s local=%d toward=%d genre=%d baseline_only=%d",
                    str(last_pool_diag.get("dj_pool_strategy", last_pool_diag.get("pool_strategy", "unknown"))),
                    int(last_pool_diag.get("chosen_from_local_count", 0)),
                    int(last_pool_diag.get("chosen_from_toward_count", 0)),
                    int(last_pool_diag.get("chosen_from_genre_count", 0)),
                    int(last_pool_diag.get("chosen_from_baseline_only_count", 0)),
                )
                # Phase 3: Membership-based counts (all overlaps)
                if "local_only" in last_pool_diag:
                    logger.info(
                        "  Provenance memberships (Phase3): local_only=%d toward_only=%d genre_only=%d " +
                        "local+toward=%d local+genre=%d toward+genre=%d local+toward+genre=%d baseline_only=%d",
                        int(last_pool_diag.get("local_only", 0)),
                        int(last_pool_diag.get("toward_only", 0)),
                        int(last_pool_diag.get("genre_only", 0)),
                        int(last_pool_diag.get("local+toward", 0)),
                        int(last_pool_diag.get("local+genre", 0)),
                        int(last_pool_diag.get("toward+genre", 0)),
                        int(last_pool_diag.get("local+toward+genre", 0)),
                        int(last_pool_diag.get("baseline_only", 0)),
                    )

        # TASK A: Invariant checks (log WARNINGs for inconsistencies)
        if pool_size_final > 0 and pool_size_initial == 0:
            logger.warning(
                "  WARNING: pool_before_gating=0 but pool_after_gating=%d (possible missing instrumentation)",
                pool_size_final
            )

        if last_pool_diag and "chosen_from_local_count" in last_pool_diag:
            chosen_sum = (
                int(last_pool_diag.get("chosen_from_local_count", 0)) +
                int(last_pool_diag.get("chosen_from_toward_count", 0)) +
                int(last_pool_diag.get("chosen_from_genre_count", 0)) +
                int(last_pool_diag.get("chosen_from_baseline_only_count", 0))
            )
            if chosen_sum != interior_len:
                logger.warning(
                    "  WARNING: chosen_from_* sum (%d) != interior_length (%d) (possible provenance tracking gap)",
                    chosen_sum,
                    interior_len
                )

        # Log ladder waypoint labels (route planning)
        if segment_ladder_diag and segment_ladder_diag.get("ladder_waypoint_count", 0) > 0:
            labels = segment_ladder_diag.get("ladder_waypoint_labels", [])
            mode = segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")
            logger.info(
                "  Ladder route: shape=%s mode=%s waypoints=%d labels=%s",
                str(segment_ladder_diag.get("route_shape", "linear")),
                mode,
                int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
                ", ".join(labels[:6]) if labels else "none",
            )
        # DEBUG top candidates for this segment
        if logger.isEnabledFor(logging.DEBUG):
            scores_dbg = []
            sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
            sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])
            for cand in last_segment_candidates[: min(200, len(last_segment_candidates))]:
                sim_a = float(sim_to_a[cand])
                sim_b = float(sim_to_b[cand])
                denom = sim_a + sim_b
                hmean = 0.0 if denom <= 1e-9 else (2 * sim_a * sim_b) / denom
                trans = _compute_transition_score(
                    cand,
                    pier_b,
                    X_full_tr_norm,
                    X_start_tr_norm,
                    X_mid_tr_norm,
                    X_end_tr_norm,
                    cfg,
                )
                final_score = cfg.weight_bridge * hmean + cfg.weight_transition * trans
                scores_dbg.append((final_score, sim_a, sim_b, hmean, trans, cand))
            scores_dbg = sorted(scores_dbg, key=lambda t: t[0], reverse=True)[:10]
            dbg_rows = []
            for final_score, sim_a, sim_b, hmean, trans, cand in scores_dbg:
                keys = identity_keys_for_index(bundle, int(cand))
                artist = (
                    str(bundle.track_artists[cand])
                    if bundle.track_artists is not None
                    else (str(bundle.artist_keys[cand]) if bundle.artist_keys is not None else "")
                )
                title = str(bundle.track_titles[cand]) if bundle.track_titles is not None else ""
                dbg_rows.append({
                    "track_id": str(bundle.track_ids[cand]),
                    "artist": sanitize_for_logging(artist),
                    "title": sanitize_for_logging(title),
                    "artist_key": keys.artist_key,
                    "title_key": keys.title_key,
                    "simA": round(sim_a, 3),
                    "simB": round(sim_b, 3),
                    "hmean": round(hmean, 3),
                    "transition": round(trans, 3),
                    "final": round(final_score, 3),
                    "internal": bool(internal_connector_indices and cand in internal_connector_indices),
                })
            logger.debug("Segment %d top candidates: %s", seg_idx, dbg_rows)

        # Commit segment path to used set
        for idx in segment_path:
            global_used.add(idx)
            try:
                used_track_keys.add(identity_keys_for_index(bundle, int(idx)).track_key)
            except Exception:
                pass
            for artist_key in _artist_keys_for_cap(int(idx)):
                global_non_seed_artist_counts[artist_key] = (
                    global_non_seed_artist_counts.get(artist_key, 0) + 1
                )

        all_segments.append(full_segment)
        all_beam_components.extend(segment_edge_components)

        # Update boundary context for next segment (cross-segment min_gap enforcement)
        # Build the concatenated result so far to extract recent artists
        current_concat: List[int] = []
        for concat_seg_idx, concat_seg in enumerate(all_segments):
            if concat_seg_idx == 0:
                current_concat.extend(concat_seg)
            else:
                current_concat.extend(concat_seg[1:])  # Drop duplicate pier

        # Extract artist keys from the last MIN_GAP_GLOBAL positions
        # If artist_identity_cfg is enabled, resolve to identity keys (collapsing ensemble variants)
        recent_boundary_artists = []
        start_pos = max(0, len(current_concat) - MIN_GAP_GLOBAL)
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

        for pos in range(start_pos, len(current_concat)):
            try:
                if use_identity:
                    # Identity mode: use raw artist string to preserve collaborations
                    artist_str = ""
                    if bundle is not None and bundle.track_artists is not None:
                        try:
                            artist_str = str(bundle.track_artists[int(current_concat[pos])] or "")
                        except Exception:
                            pass
                    if artist_str:
                        identity_keys_set = resolve_artist_identity_keys(artist_str, artist_identity_cfg)
                        # Add ALL identity keys to boundary tracking
                        for identity_key in identity_keys_set:
                            recent_boundary_artists.append(identity_key)
                else:
                    # Legacy mode: single artist_key
                    artist_key = identity_keys_for_index(bundle, int(current_concat[pos])).artist_key
                    if artist_key:
                        recent_boundary_artists.append(str(artist_key))
            except Exception:
                continue

    # Concatenate segments
    # First segment: keep full [A, ..., B]
    # Subsequent segments: drop first element (the pier) to avoid duplication
    # Single-seed arc: drop last element (the duplicated seed) to avoid repetition
    final_indices: List[int] = []
    seed_positions: List[int] = []

    if is_single_seed_arc:
        # Single-seed arc: segment is [seed, interior..., seed]
        # Output only [seed, interior...] to avoid duplicate seed at end
        segment = all_segments[0] if all_segments else [ordered_seeds[0]]
        final_indices = segment[:-1]  # Drop the trailing duplicate seed
        seed_positions = [0]  # Seed is at position 0
        logger.info("Pier+Bridge: single-seed arc output: %d tracks (seed at start, arc returns to seed-similar)", len(final_indices))
    else:
        for seg_idx, segment in enumerate(all_segments):
            if seg_idx == 0:
                final_indices.extend(segment)
                seed_positions.append(0)  # First pier
                seed_positions.append(len(final_indices) - 1)  # Second pier
            else:
                # Drop first element (the pier, already included)
                final_indices.extend(segment[1:])
                seed_positions.append(len(final_indices) - 1)  # New pier

    edge_repair_swap_log: list[dict[str, Any]] = []
    if bool(getattr(cfg, "edge_repair_enabled", False)):
        from src.playlist.repair.edge_repair import repair_playlist_edges

        disallowed_repair_artist_keys: set[str] = set()
        if bool(cfg.disallow_seed_artist_in_interiors) or bool(cfg.disallow_pier_artists_in_interiors):
            for sidx in seed_indices:
                try:
                    disallowed_repair_artist_keys.add(
                        identity_keys_for_index(bundle, int(sidx)).artist_key
                    )
                except Exception:
                    continue
        repair_result = repair_playlist_edges(
            final_indices=final_indices,
            candidate_indices=universe,
            metric_context=transition_metric_context,
            bundle=bundle,
            seed_indices=seed_indices,
            pier_positions=seed_positions,
            transition_floor=float(cfg.transition_floor),
            centered_cos_floor=float(getattr(cfg, "edge_repair_centered_cos_floor", -0.5)),
            margin=float(getattr(cfg, "edge_repair_margin", 0.05)),
            allowed_indices=allowed_set_indices,
            disallowed_artist_keys=disallowed_repair_artist_keys,
            variety_guard_enabled=bool(getattr(cfg, "edge_repair_variety_guard_enabled", False)),
            variety_guard_threshold=float(getattr(cfg, "edge_repair_variety_guard_threshold", 0.85)),
            max_non_seed_tracks_per_artist=getattr(cfg, "max_non_seed_tracks_per_artist", None),
            artist_identity_cfg=artist_identity_cfg,
        )
        edge_repair_swap_log = list(repair_result.swap_log)
        if list(repair_result.indices) != list(final_indices):
            final_indices = list(repair_result.indices)
            all_beam_components = []

    # Convert to track IDs
    # Cross-segment min_gap is enforced DURING generation (boundary-aware beam search),
    # not as a post-order filter. This ensures exact length guarantees.
    final_track_ids = [str(bundle.track_ids[i]) for i in final_indices]

    # Strict length validation: pier-bridge must return EXACTLY the requested number of tracks
    if len(final_track_ids) != total_tracks:
        failure_msg = (
            f"Pier-bridge length mismatch: generated {len(final_track_ids)} tracks "
            f"but expected exactly {total_tracks}. This indicates a bug in segment generation."
        )
        logger.error(failure_msg)
        return PierBridgeResult(
            track_ids=[],
            track_indices=[],
            seed_positions=[],
            segment_diagnostics=diagnostics,
            stats={"error": "length_mismatch", "expected": total_tracks, "actual": len(final_track_ids)},
            success=False,
            failure_reason=failure_msg,
        )

    # Compute per-edge transition scores for reporting (matches builder scoring)
    edge_scores: list[dict[str, Any]] = []
    transition_vals: list[float] = []
    for i in range(1, len(final_indices)):
        prev_idx = final_indices[i - 1]
        cur_idx = final_indices[i]
        edge = score_transition_edge(transition_metric_context, prev_idx, cur_idx)
        edge["floor"] = float(cfg.transition_floor)
        t_val = float(edge.get("T"))
        transition_vals.append(float(t_val))
        # BPM columns (diagnostic only)
        bpm_a: Optional[float] = None
        bpm_b: Optional[float] = None
        bpm_log_dist: Optional[float] = None
        if perceptual_bpm is not None:
            _ba = perceptual_bpm[prev_idx]
            _bb = perceptual_bpm[cur_idx]
            if not np.isnan(_ba):
                bpm_a = float(_ba)
            if not np.isnan(_bb):
                bpm_b = float(_bb)
            if bpm_a is not None and bpm_b is not None:
                from src.playlist.bpm_axis import bpm_log_distance
                bpm_log_dist = float(bpm_log_distance(bpm_a, bpm_b))
        edge_scores.append(
            {
                "prev_id": str(bundle.track_ids[prev_idx]),
                "cur_id": str(bundle.track_ids[cur_idx]),
                "prev_idx": int(prev_idx),
                "cur_idx": int(cur_idx),
                **edge,
                "bpm_a": bpm_a,
                "bpm_b": bpm_b,
                "bpm_log_dist": bpm_log_dist,
            }
        )

    # Recompute seed positions from final track IDs for diagnostic accuracy
    seed_positions = [idx for idx, tid in enumerate(final_track_ids) if tid in seed_id_set]
    if len(seed_positions) != (1 if is_single_seed_arc else len(seed_id_set)):
        logger.warning(
            "Pier+Bridge: seed count mismatch in final result (expected %d, found %d)",
            (1 if is_single_seed_arc else len(seed_id_set)),
            len(seed_positions),
        )

    # Compute overall stats
    actual_num_seeds = 1 if is_single_seed_arc else len(seed_indices)
    seed_index_set = set(int(i) for i in seed_indices)
    artist_counts: Dict[str, int] = {}
    non_seed_artist_counts: Dict[str, int] = {}
    for idx in final_indices:
        artist_key = ""
        try:
            artist_key = identity_keys_for_index(bundle, int(idx)).artist_key
        except Exception:
            artist_key = ""
        if not artist_key:
            continue
        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        if int(idx) not in seed_index_set:
            non_seed_artist_counts[artist_key] = non_seed_artist_counts.get(artist_key, 0) + 1

    # BPM per-playlist summary (diagnostic)
    _bpm_summary: Optional[dict] = None
    if perceptual_bpm is not None:
        _bpms = [float(perceptual_bpm[i]) for i in final_indices if not np.isnan(perceptual_bpm[i])]
        if _bpms:
            _bpm_summary = {
                "min": float(min(_bpms)),
                "mean": float(sum(_bpms) / len(_bpms)),
                "max": float(max(_bpms)),
                "std": float(np.std(_bpms)),
                "n": len(_bpms),
                "total": len(final_indices),
            }
            logger.info(
                "BPM (perceptual): min=%.0f mean=%.0f max=%.0f std=%.0f (%d/%d tracks have data)",
                _bpm_summary["min"],
                _bpm_summary["mean"],
                _bpm_summary["max"],
                _bpm_summary["std"],
                _bpm_summary["n"],
                _bpm_summary["total"],
            )

    stats = {
        "num_seeds": actual_num_seeds,
        "single_seed_arc": is_single_seed_arc,
        "target_tracks": total_tracks,
        "actual_tracks": len(final_indices),
        "artist_counts": artist_counts,
        "non_seed_artist_counts": non_seed_artist_counts,
        "max_non_seed_tracks_per_artist": cfg.max_non_seed_tracks_per_artist,
        "universe_size": len(universe),
        "segments_built": len(all_segments),
        "segments_successful": sum(1 for d in diagnostics if d.success),
        "total_expansions": sum(d.expansions for d in diagnostics),
        "edge_scores": edge_scores,
        "bpm_summary": _bpm_summary,
        "min_transition": float(np.min(transition_vals)) if transition_vals else None,
        "mean_transition": float(np.mean(transition_vals)) if transition_vals else None,
        "transition_centered": bool(cfg.center_transitions),
        "soft_genre_penalty_hits": int(soft_genre_penalty_hits_total),
        "soft_genre_penalty_edges_scored": int(soft_genre_penalty_edges_scored_total),
        "local_sonic_penalty_hits": int(local_sonic_penalty_hits_total),
        "local_sonic_edges_scored": int(local_sonic_edges_scored_total),
        "local_sonic_gate_rejected": int(local_sonic_gate_rejected_total),
        "local_sonic_penalty_total": float(local_sonic_penalty_total),
        "segment_bridge_floors_used": [float(x) for x in segment_bridge_floors_used],
        "segment_backoff_attempts_used": [int(x) for x in segment_backoff_attempts_used],
        "beam_edge_components": list(all_beam_components),
        "edge_repair_enabled": bool(getattr(cfg, "edge_repair_enabled", False)),
        "edge_repair_applied": bool(
            any(isinstance(entry, dict) and "new_idx" in entry for entry in edge_repair_swap_log)
        ),
        "edge_repair_swap_log": edge_repair_swap_log,
        "warnings": warnings,
        "config": {
            "transition_floor": cfg.transition_floor,
            "transition_weights": cfg.transition_weights,
            "edge_repair_enabled": bool(cfg.edge_repair_enabled),
            "edge_repair_centered_cos_floor": float(cfg.edge_repair_centered_cos_floor),
            "edge_repair_margin": float(cfg.edge_repair_margin),
            "edge_repair_variety_guard_enabled": bool(cfg.edge_repair_variety_guard_enabled),
            "edge_repair_variety_guard_threshold": float(cfg.edge_repair_variety_guard_threshold),
            "initial_neighbors_m": cfg.initial_neighbors_m,
            "initial_beam_width": cfg.initial_beam_width,
            "eta_destination_pull": cfg.eta_destination_pull,
            "genre_tiebreak_weight": float(cfg.genre_tiebreak_weight),
            "genre_penalty_threshold": float(cfg.genre_penalty_threshold),
            "genre_penalty_strength": float(cfg.genre_penalty_strength),
            "genre_tie_break_band": (
                float(cfg.genre_tie_break_band) if cfg.genre_tie_break_band is not None else None
            ),
            "local_sonic_edge_penalty_enabled": bool(cfg.local_sonic_edge_penalty_enabled),
            "local_sonic_edge_penalty_threshold": float(cfg.local_sonic_edge_penalty_threshold),
            "local_sonic_edge_penalty_strength": float(cfg.local_sonic_edge_penalty_strength),
            "local_sonic_edge_floor": (
                float(cfg.local_sonic_edge_floor)
                if cfg.local_sonic_edge_floor is not None
                else None
            ),
            "bridge_floor": float(cfg.bridge_floor),
            "max_non_seed_tracks_per_artist": cfg.max_non_seed_tracks_per_artist,
            "infeasible_handling_enabled": bool(infeasible_handling and infeasible_handling.enabled),
            "experiment_bridge_scoring": {
                "enabled": bool(cfg.experiment_bridge_scoring_enabled),
                "min_weight": float(cfg.experiment_bridge_min_weight),
                "balance_weight": float(cfg.experiment_bridge_balance_weight),
            },
            "dj_bridging": {
                "enabled": bool(cfg.dj_bridging_enabled),
                "seed_ordering": str(cfg.dj_seed_ordering),
                "anchors_must_include_all": bool(cfg.dj_anchors_must_include_all),
                "route_shape": str(cfg.dj_route_shape),
                "waypoint_weight": float(cfg.dj_waypoint_weight),
                "waypoint_floor": float(cfg.dj_waypoint_floor),
                "waypoint_penalty": float(cfg.dj_waypoint_penalty),
                "waypoint_tie_break_band": (
                    float(cfg.dj_waypoint_tie_break_band) if cfg.dj_waypoint_tie_break_band is not None else None
                ),
                "waypoint_cap": float(cfg.dj_waypoint_cap),
                "seed_ordering_weights": {
                    "sonic": float(cfg.dj_seed_ordering_weight_sonic),
                    "genre": float(cfg.dj_seed_ordering_weight_genre),
                    "bridge": float(cfg.dj_seed_ordering_weight_bridge),
                },
                "pooling_strategy": str(cfg.dj_pooling_strategy),
                "pooling_k_local": int(cfg.dj_pooling_k_local),
                "pooling_k_toward": int(cfg.dj_pooling_k_toward),
                "pooling_k_genre": int(cfg.dj_pooling_k_genre),
                "pooling_k_union_max": int(cfg.dj_pooling_k_union_max),
                "pooling_step_stride": int(cfg.dj_pooling_step_stride),
                "pooling_cache_enabled": bool(cfg.dj_pooling_cache_enabled),
                "pooling_debug_compare_baseline": bool(
                    cfg.dj_pooling_debug_compare_baseline
                ),
                "allow_detours_when_far": bool(cfg.dj_allow_detours_when_far),
                "far_thresholds": {
                    "sonic": float(cfg.dj_far_threshold_sonic),
                    "genre": float(cfg.dj_far_threshold_genre),
                    "connector_scarcity": float(cfg.dj_far_threshold_connector_scarcity),
                },
                "connector_bias": {
                    "enabled": bool(cfg.dj_connector_bias_enabled),
                    "max_per_segment_linear": int(cfg.dj_connector_max_per_segment_linear),
                    "max_per_segment_adventurous": int(cfg.dj_connector_max_per_segment_adventurous),
                },
                "ladder": {
                    "top_labels": int(cfg.dj_ladder_top_labels),
                    "min_label_weight": float(cfg.dj_ladder_min_label_weight),
                    "min_similarity": float(cfg.dj_ladder_min_similarity),
                    "max_steps": int(cfg.dj_ladder_max_steps),
                    "use_smoothed_waypoint_vectors": bool(
                        cfg.dj_ladder_use_smoothed_waypoint_vectors
                    ),
                    "smooth_top_k": int(cfg.dj_ladder_smooth_top_k),
                    "smooth_min_sim": float(cfg.dj_ladder_smooth_min_sim),
                },
                "waypoint_fallback_k": int(cfg.dj_waypoint_fallback_k),
                "micro_piers": {
                    "enabled": bool(cfg.dj_micro_piers_enabled),
                    "max": int(cfg.dj_micro_piers_max),
                    "topk": int(cfg.dj_micro_piers_topk),
                    "candidate_source": str(cfg.dj_micro_piers_candidate_source),
                    "selection_metric": str(cfg.dj_micro_piers_selection_metric),
                },
                "relaxation": {
                    "enabled": bool(cfg.dj_relaxation_enabled),
                    "max_attempts": int(cfg.dj_relaxation_max_attempts),
                    "emit_warnings": bool(cfg.dj_relaxation_emit_warnings),
                    "allow_floor_relaxation": bool(cfg.dj_relaxation_allow_floor_relaxation),
                },
            },
            "progress_arc": {
                "enabled": bool(cfg.progress_arc_enabled),
                "weight": float(cfg.progress_arc_weight),
                "shape": str(cfg.progress_arc_shape),
                "tolerance": float(cfg.progress_arc_tolerance),
                "loss": str(cfg.progress_arc_loss),
                "huber_delta": float(cfg.progress_arc_huber_delta),
                "max_step": (float(cfg.progress_arc_max_step) if cfg.progress_arc_max_step is not None else None),
                "max_step_mode": str(cfg.progress_arc_max_step_mode),
                "max_step_penalty": float(cfg.progress_arc_max_step_penalty),
                "autoscale": {
                    "enabled": bool(cfg.progress_arc_autoscale_enabled),
                    "min_distance": float(cfg.progress_arc_autoscale_min_distance),
                    "distance_scale": float(cfg.progress_arc_autoscale_distance_scale),
                    "per_step_scale": bool(cfg.progress_arc_autoscale_per_step_scale),
                },
            },
        },
    }

    logger.info("Pier+Bridge complete: %d tracks, %d segments, %d successful",
               len(final_indices), len(all_segments),
               sum(1 for d in diagnostics if d.success))

    return PierBridgeResult(
        track_ids=final_track_ids,
        track_indices=final_indices,
        seed_positions=seed_positions,
        segment_diagnostics=diagnostics,
        stats=stats,
    )

