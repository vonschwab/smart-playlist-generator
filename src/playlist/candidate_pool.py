from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import CandidatePoolConfig
from .genre_compatibility import compute_raw_genre_compatibility
from src.playlist.title_quality import detect_title_artifacts

logger = logging.getLogger(__name__)


def _compute_genre_similarity(
    seed_genres: np.ndarray,
    candidate_genres: np.ndarray,
    method: str = "cosine",
    idf_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute genre similarity between seed and all candidates.

    When idf_weights is provided, multiply seed and candidate vectors by the weights
    before computing similarity. Rare-tag matches contribute more.

    Args:
        seed_genres: (D,) 1D array of seed's genres (binary or float)
        candidate_genres: (N, D) matrix of candidate genres
        method: "weighted_jaccard", "cosine", or "ensemble"
        idf_weights: (D,) optional per-genre IDF weights

    Returns:
        (N,) similarity scores [0, 1]
    """
    if idf_weights is not None:
        weights = np.asarray(idf_weights, dtype=float)
        seed_genres = seed_genres * weights
        candidate_genres = candidate_genres * weights.reshape(1, -1)

    N = candidate_genres.shape[0]

    if method == "weighted_jaccard":
        # Binary Jaccard: intersection / union
        # Treat as binary presence/absence
        seed_binary = (seed_genres > 0).astype(float)
        cand_binary = (candidate_genres > 0).astype(float)

        intersection = np.sum(seed_binary * cand_binary, axis=1)
        union = np.sum(np.maximum(seed_binary, cand_binary), axis=1)
        union = np.maximum(union, 1e-12)  # avoid div by zero

        return intersection / union

    elif method == "cosine":
        # Cosine similarity on float vectors (X_genre_smoothed)
        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        cand_norm = candidate_genres / (np.linalg.norm(candidate_genres, axis=1, keepdims=True) + 1e-12)

        sim = np.dot(cand_norm, seed_norm)
        return np.clip(sim, 0.0, 1.0)  # clamp to [0,1]

    elif method == "ensemble":
        # Combine both methods: 0.6*cosine + 0.4*jaccard
        seed_binary = (seed_genres > 0).astype(float)
        cand_binary = (candidate_genres > 0).astype(float)
        intersection = np.sum(seed_binary * cand_binary, axis=1)
        union = np.sum(np.maximum(seed_binary, cand_binary), axis=1)
        union = np.maximum(union, 1e-12)
        jaccard_sim = intersection / union

        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        cand_norm = candidate_genres / (np.linalg.norm(candidate_genres, axis=1, keepdims=True) + 1e-12)
        cosine_sim = np.clip(np.dot(cand_norm, seed_norm), 0.0, 1.0)

        return 0.6 * cosine_sim + 0.4 * jaccard_sim

    else:
        raise ValueError(f"Unknown genre_method: {method}")





@dataclass(frozen=True)
class CandidatePoolResult:
    pool_indices: np.ndarray  # indices into global track arrays (size P)
    eligible_indices: np.ndarray  # indices eligible after hard gates (size E)
    seed_sim: np.ndarray  # seed similarity for each pool element (P,)
    sonic_sim: Optional[np.ndarray]  # sonic similarity for each pool element (P,) if available
    stats: Dict[str, Any]  # pool size, distinct artists, etc.
    params_effective: Dict[str, Any]


def _normalize_artist_key(raw: Any) -> str:
    from src.string_utils import normalize_artist_key

    return normalize_artist_key("" if raw is None else str(raw))


def _compute_duration_penalty(
    candidate_duration_ms: float,
    reference_duration_ms: float,
    weight: float,
) -> float:
    """
    Four-phase geometric penalty based on percentage excess over reference.

    0-20%:   Gentle (barely noticeable)
    20-50%:  Moderate (increasing)
    50-100%: Steep (strong discouragement)
    >100%:   Severe (track is 2x+ longer)
    """
    if candidate_duration_ms <= 0 or reference_duration_ms <= 0:
        return 0.0
    if candidate_duration_ms <= reference_duration_ms:
        return 0.0

    excess_ratio = (candidate_duration_ms - reference_duration_ms) / reference_duration_ms
    if excess_ratio <= 0.20:
        # Phase 1: Gentle (power 1.5)
        penalty = weight * 0.05 * (excess_ratio / 0.20) ** 1.5
    elif excess_ratio <= 0.50:
        # Phase 2: Moderate (power 2.0)
        phase_ratio = (excess_ratio - 0.20) / 0.30
        penalty = weight * 0.05 + weight * 0.25 * (phase_ratio ** 2.0)
    elif excess_ratio <= 1.00:
        # Phase 3: Steep (power 2.5)
        phase_ratio = (excess_ratio - 0.50) / 0.50
        penalty = weight * 0.30 + weight * 0.45 * (phase_ratio ** 2.5)
    else:
        # Phase 4: Severe (power 3.0)
        phase_ratio = excess_ratio - 1.00
        penalty = weight * 0.75 + weight * 2.25 * (phase_ratio ** 3.0)

    return penalty


def _first_rejection_reason(
    *,
    idx: int,
    seed_mask: np.ndarray,
    seed_sim_all: np.ndarray,
    similarity_floor: float,
    sonic_seed_sim: Optional[np.ndarray],
    sonic_floor: Optional[float],
    genre_sim_all: Optional[np.ndarray],
    min_genre_similarity: Optional[float],
    genre_conflict_result: Any,
    genre_conflict_min_confidence: Optional[float],
    pool_set: set[int],
    eligible_set: set[int],
) -> str:
    if bool(seed_mask[idx]):
        return "seed"
    if float(seed_sim_all[idx]) < float(similarity_floor):
        return "below_similarity_floor"
    if sonic_seed_sim is not None and sonic_floor is not None and float(sonic_seed_sim[idx]) < float(sonic_floor):
        return "below_sonic_similarity"
    if (
        genre_sim_all is not None
        and min_genre_similarity is not None
        and float(genre_sim_all[idx]) < float(min_genre_similarity)
    ):
        return "below_genre_similarity"
    if genre_conflict_result is not None and genre_conflict_min_confidence is not None:
        missing = bool(genre_conflict_result.missing_or_sparse[idx])
        confidence = float(genre_conflict_result.confidence[idx])
        if not missing and confidence < float(genre_conflict_min_confidence):
            return "genre_conflict"
    if idx in eligible_set and idx not in pool_set:
        return "artist_cap"
    return "admitted" if idx in pool_set else "not_selected"


def build_candidate_pool(
    *,
    seed_idx: int,
    seed_indices: Optional[list[int]] = None,
    embedding: np.ndarray,  # (N, D)
    artist_keys: np.ndarray,  # (N,)
    track_ids: Optional[np.ndarray] = None,
    track_titles: Optional[np.ndarray] = None,
    track_artists: Optional[np.ndarray] = None,
    durations_ms: Optional[np.ndarray] = None,
    cfg: CandidatePoolConfig,
    random_seed: int,
    # Optional raw sonic space for hard floor
    X_sonic: Optional[np.ndarray] = None,
    # Genre gating (optional)
    X_genre_raw: Optional[np.ndarray] = None,  # (N, D_genre) binary genres
    X_genre_smoothed: Optional[np.ndarray] = None,  # (N, D_genre) float genres
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
    genre_vocab: Optional[list[str]] = None,
    broad_filters: tuple[str, ...] = (),
    mode: str = "dynamic",  # "dynamic", "narrow", "discover"
) -> CandidatePoolResult:
    """
    Implement current experiments behavior with optional genre gating:
    - seed_sim for all tracks = cosine(hybrid[i], hybrid[seed])
        - filter by cfg.similarity_floor and i != seed
        - [NEW] optionally filter by genre_similarity (hard gate in dynamic/narrow)
        - group by artist_keys
        - rank artists by max seed_sim within artist
        - walk artists in rank order, taking up to candidates_per_artist,
      plus seed_artist_bonus for seed artist (if present),
      until max_pool_size reached AND target_artists satisfied (when possible).
    """
    rng = np.random.default_rng(random_seed)
    emb_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
    seed_list = list(dict.fromkeys([seed_idx] + list(seed_indices or [])))
    seed_vecs = emb_norm[seed_list]
    # Use the best seed match for admission (multi-seed support)
    seed_sim_matrix = np.dot(emb_norm, seed_vecs.T)
    seed_sim_all = np.max(seed_sim_matrix, axis=1)
    seed_mask = np.zeros_like(seed_sim_all, dtype=bool)
    seed_mask[seed_list] = True
    seed_sim_all[seed_mask] = -1.0

    duration_penalty_active = bool(cfg.duration_penalty_enabled) and durations_ms is not None
    reference_duration_ms = 0.0
    if duration_penalty_active:
        seed_durations = []
        for idx in seed_list:
            if 0 <= idx < len(durations_ms):
                dur = float(durations_ms[idx])
                if dur > 0:
                    seed_durations.append(dur)
        if seed_durations:
            seed_durations.sort()
            mid = len(seed_durations) // 2
            if len(seed_durations) % 2 == 1:
                reference_duration_ms = seed_durations[mid]
            else:
                reference_duration_ms = (seed_durations[mid - 1] + seed_durations[mid]) / 2.0
        else:
            reference_duration_ms = 0.0
        if reference_duration_ms <= 0:
            duration_penalty_active = False

    duration_penalty_count = 0
    duration_cutoff_count = 0
    if duration_penalty_active:
        seed_sim_all = seed_sim_all.copy()
        cutoff_ms = reference_duration_ms * float(cfg.duration_cutoff_multiplier)
        for i in range(len(seed_sim_all)):
            if seed_mask[i]:
                continue
            if i >= len(durations_ms):
                continue
            cand_duration = float(durations_ms[i])
            if cand_duration <= 0:
                continue
            if cand_duration > cutoff_ms:
                seed_sim_all[i] = -1.0
                duration_cutoff_count += 1
                continue
            penalty = _compute_duration_penalty(
                cand_duration,
                reference_duration_ms,
                cfg.duration_penalty_weight,
            )
            if penalty > 0:
                seed_sim_all[i] -= penalty
                duration_penalty_count += 1
        if duration_penalty_count and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Duration penalty applied: reference_ms=%.0f penalized=%d weight=%.3f",
                reference_duration_ms,
                duration_penalty_count,
                cfg.duration_penalty_weight,
            )

    # Optional sonic-only similarity (for hard floor)
    sonic_seed_sim = None
    below_sonic_floor = 0
    sonic_floor = cfg.min_sonic_similarity
    epsilon = 1e-9
    if X_sonic is not None:
        sonic_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
        seed_vecs_sonic = sonic_norm[seed_list]
        sonic_seed_sim_matrix = np.dot(sonic_norm, seed_vecs_sonic.T)
        sonic_seed_sim = np.max(sonic_seed_sim_matrix, axis=1)
        sonic_seed_sim[seed_mask] = -1.0

    # Track exclusion reasons for instrumentation
    total_candidates = len(seed_sim_all) - len(seed_list)  # exclude all seeds
    below_floor_count = 0
    below_genre_count = 0
    genre_overlap_guard_rejected = 0
    genre_conflict_rejected = 0
    genre_conflict_penalty_applied = 0
    title_exclusion_rejected = 0
    rejected_sonic: list[tuple[int, float]] = []

    # Compute genre similarity if provided
    genre_sim_all = None
    genre_raw_matrix = X_genre_raw
    genre_mask = None
    genre_vocab_effective = [] if genre_vocab is None else [str(g) for g in list(genre_vocab)]
    if broad_filters and genre_vocab_effective:
        genre_mask = np.array([g.lower() not in broad_filters for g in genre_vocab_effective], dtype=bool)
        if genre_raw_matrix is not None and genre_mask.shape[0] == genre_raw_matrix.shape[1]:
            genre_raw_matrix = genre_raw_matrix[:, genre_mask]
            genre_vocab_effective = [
                str(g) for g, keep in zip(genre_vocab_effective, genre_mask) if bool(keep)
            ]

    if min_genre_similarity is not None and (X_genre_raw is not None or X_genre_smoothed is not None):
        # Choose matrix based on method
        if genre_method == "weighted_jaccard" and genre_raw_matrix is not None:
            genre_matrix = genre_raw_matrix
        elif X_genre_smoothed is not None:
            genre_matrix = X_genre_smoothed[:, genre_mask] if genre_mask is not None else X_genre_smoothed
        else:
            genre_matrix = genre_raw_matrix if genre_raw_matrix is not None else X_genre_smoothed

        # Compute IDF weights from the raw matrix when enabled.
        idf_weights = None
        if (
            bool(getattr(cfg, "genre_idf_enabled", True))
            and genre_raw_matrix is not None
            and genre_raw_matrix.shape[1] == genre_matrix.shape[1]
        ):
            from src.playlist.genre_idf import compute_genre_idf
            idf_weights = compute_genre_idf(
                X_genre_raw=genre_raw_matrix,
                power=1.0,
                norm="max1",
            )

        seed_genres = np.max(genre_matrix[seed_list], axis=0)
        genre_sim_all = _compute_genre_similarity(
            seed_genres,
            genre_matrix,
            method=genre_method,
            idf_weights=idf_weights,
        )
        genre_sim_all[seed_idx] = 1.0  # seed matches itself perfectly
        logger.info(
            "Candidate pool genre gating: method=%s, min_threshold=%.3f, mode=%s, idf=%s",
            genre_method, min_genre_similarity, mode, "on" if idf_weights is not None else "off",
        )

    genre_conflict_result = None
    if (
        cfg.genre_conflict_enabled
        and genre_raw_matrix is not None
        and genre_raw_matrix.ndim == 2
        and genre_raw_matrix.shape[0] == len(seed_sim_all)
        and genre_raw_matrix.shape[1] == len(genre_vocab_effective)
    ):
        seed_raw = np.max(genre_raw_matrix[seed_list], axis=0)
        genre_conflict_result = compute_raw_genre_compatibility(
            seed_raw=seed_raw,
            candidate_raw=genre_raw_matrix,
            genre_vocab=genre_vocab_effective,
            compatible_threshold=cfg.genre_conflict_compatible_threshold,
            conflict_threshold=cfg.genre_conflict_conflict_threshold,
            penalty_strength=cfg.genre_conflict_penalty_strength,
        )
        if cfg.genre_conflict_penalty_strength > 0:
            penalty = np.asarray(genre_conflict_result.penalty, dtype=float)
            penalized = (penalty > 0) & (~seed_mask)
            genre_conflict_penalty_applied = int(np.count_nonzero(penalized))
            seed_sim_all = seed_sim_all - penalty
            logger.info(
                "Genre conflict penalty applied: penalized=%d strength=%.3f",
                genre_conflict_penalty_applied,
                float(cfg.genre_conflict_penalty_strength),
            )

    # Build initial eligible list (by hybrid similarity floor and sonic floor if provided)
    eligible: list[int] = []
    for i, sim in enumerate(seed_sim_all):
        if seed_mask[i]:
            continue
        if (
            track_titles is not None
            and cfg.title_hard_exclude_flags
            and detect_title_artifacts(str(track_titles[i])) & cfg.title_hard_exclude_flags
        ):
            title_exclusion_rejected += 1
            continue
        if sim < cfg.similarity_floor:
            below_floor_count += 1
            continue
        if sonic_seed_sim is not None and sonic_floor is not None and (sonic_seed_sim[i] + epsilon) < sonic_floor:
            below_sonic_floor += 1
            if logger.isEnabledFor(logging.DEBUG):
                rejected_sonic.append((i, float(sonic_seed_sim[i])))
            continue
        eligible.append(i)

    if sonic_seed_sim is not None and sonic_floor is not None:
        try:
            sims = np.array([s for idx, s in enumerate(sonic_seed_sim) if not seed_mask[idx]], dtype=float)
            if sims.size:
                logger.info(
                    "Sonic floor applied: mode=%s floor=%.2f before=%d after=%d rejected=%d",
                    mode,
                    sonic_floor,
                    len(sims),
                    len(eligible),
                    below_sonic_floor,
                )
                logger.info(
                    "Sonic sim distribution: min=%.3f p05=%.3f median=%.3f p95=%.3f max=%.3f",
                    float(np.min(sims)),
                    float(np.percentile(sims, 5)),
                    float(np.percentile(sims, 50)),
                    float(np.percentile(sims, 95)),
                    float(np.max(sims)),
                )
                if logger.isEnabledFor(logging.DEBUG) and rejected_sonic:
                    rejected_sonic = sorted(rejected_sonic, key=lambda t: t[1])[:10]
                    def _label(idx: int) -> str:
                        title = str((track_titles[idx] if track_titles is not None else "") or "")
                        artist = str((track_artists[idx] if track_artists is not None else artist_keys[idx]) or "")
                        tid = track_ids[idx] if track_ids is not None else idx
                        return f"{artist} - {title} [{tid}]"
                    logger.debug(
                        "Top sonic rejects (floor=%.2f): %s",
                        sonic_floor,
                        [(round(val, 3), _label(idx)) for idx, val in rejected_sonic],
                    )
        except Exception:
            logger.debug("Candidate filter (sonic floor): summary unavailable (stats error)")

    # Apply genre gate (hard gate for dynamic/narrow, soft for discover)
    if genre_sim_all is not None:
        # Optional overlap guard: require at least one raw shared tag after broad
        # filters. This prevents smoothed vectors from admitting tracks that only
        # match through generic tags such as "rock" or "pop".
        overlap_guard_enabled = genre_raw_matrix is not None and (
            mode == "narrow" or (mode == "dynamic" and bool(broad_filters))
        )
        if overlap_guard_enabled:
            seed_binary = (np.max(genre_raw_matrix[seed_list], axis=0) > 0).astype(float)
            overlaps = (genre_raw_matrix > 0) & (seed_binary > 0)
            zero_overlap_mask = overlaps.sum(axis=1) == 0
            if mode in ("dynamic", "narrow"):
                genre_overlap_guard_rejected = int(
                    sum(1 for i in eligible if bool(zero_overlap_mask[i]))
                )
            genre_sim_all = np.where(zero_overlap_mask, 0.0, genre_sim_all)

        if mode in ("dynamic", "narrow"):
            # Hard gate: exclude candidates below threshold
            eligible_before_genre = len(eligible)
            eligible = [i for i in eligible if genre_sim_all[i] >= min_genre_similarity]
            below_genre_count = eligible_before_genre - len(eligible)
            logger.info(
                "Genre hard gate applied: %d candidates excluded (mode=%s)",
                below_genre_count, mode
            )
        # For "discover" mode, we compute genre_sim but don't exclude (soft penalty later if needed)

    if genre_conflict_result is not None and cfg.genre_conflict_min_confidence is not None:
        min_confidence = float(cfg.genre_conflict_min_confidence)
        eligible_before_conflict = len(eligible)
        eligible = [
            i for i in eligible
            if bool(genre_conflict_result.missing_or_sparse[i])
            or float(genre_conflict_result.confidence[i]) >= min_confidence
        ]
        genre_conflict_rejected = eligible_before_conflict - len(eligible)
        if genre_conflict_rejected:
            logger.info(
                "Genre conflict confidence gate applied: rejected=%d min_confidence=%.3f",
                genre_conflict_rejected,
                min_confidence,
            )
    grouped: Dict[str, list[int]] = {}
    for idx in eligible:
        key = _normalize_artist_key(artist_keys[idx])
        grouped.setdefault(key, []).append(idx)

    artist_rank: list[tuple[str, float, list[int]]] = []
    for artist, idxs in grouped.items():
        max_sim = max(seed_sim_all[i] for i in idxs)
        max_genre = max(float(genre_sim_all[i]) for i in idxs) if genre_sim_all is not None else 0.0
        artist_rank.append((artist, max_sim, idxs, max_genre))
    artist_rank.sort(key=lambda t: (-t[1], -t[3], t[0]))

    pool_indices: list[int] = []
    pool_artists: set[str] = set()
    seed_artist_key = _normalize_artist_key(artist_keys[seed_idx])

    for artist, _, idxs, _max_genre in artist_rank:
        per_artist_cap = cfg.candidates_per_artist
        if artist == seed_artist_key:
            per_artist_cap += cfg.seed_artist_bonus
        _genre_key = genre_sim_all if genre_sim_all is not None else None
        sorted_idxs = sorted(
            idxs,
            key=lambda i: (-seed_sim_all[i], -(float(_genre_key[i]) if _genre_key is not None else 0.0), i),
        )
        take = sorted_idxs[:per_artist_cap]
        for idx in take:
            if len(pool_indices) >= cfg.max_pool_size and len(pool_artists) >= cfg.target_artists:
                break
            pool_indices.append(idx)
        pool_artists.add(artist)
        if len(pool_indices) >= cfg.max_pool_size and len(pool_artists) >= cfg.target_artists:
            break

    pool_indices = list(dict.fromkeys(pool_indices))  # dedupe, preserve order
    seed_sim_pool = np.array([seed_sim_all[i] for i in pool_indices], dtype=float)
    sonic_sim_pool = (
        np.array([sonic_seed_sim[i] for i in pool_indices], dtype=float)
        if sonic_seed_sim is not None
        else None
    )

    # Count how many were excluded due to artist cap (those not taken from eligible artists)
    artist_cap_excluded = len(eligible) - len(pool_indices)

    params_effective = {
        "similarity_floor": cfg.similarity_floor,
        "max_pool_size": cfg.max_pool_size,
        "target_artists": cfg.target_artists,
        "candidates_per_artist": cfg.candidates_per_artist,
        "seed_artist_bonus": cfg.seed_artist_bonus,
        "min_sonic_similarity": cfg.min_sonic_similarity,
    }
    if duration_penalty_active:
        params_effective["duration_penalty_weight"] = cfg.duration_penalty_weight
        params_effective["duration_reference_ms"] = reference_duration_ms
        params_effective["duration_cutoff_multiplier"] = float(
            cfg.duration_cutoff_multiplier
        )
    if min_genre_similarity is not None:
        params_effective["genre_method"] = genre_method
        params_effective["min_genre_similarity"] = min_genre_similarity
        if broad_filters:
            params_effective["broad_filters"] = list(broad_filters)
    if cfg.genre_conflict_enabled:
        params_effective["genre_conflict_enabled"] = True
        params_effective["genre_conflict_min_confidence"] = cfg.genre_conflict_min_confidence
        params_effective["genre_conflict_penalty_strength"] = cfg.genre_conflict_penalty_strength
        params_effective["genre_conflict_compatible_threshold"] = cfg.genre_conflict_compatible_threshold
        params_effective["genre_conflict_conflict_threshold"] = cfg.genre_conflict_conflict_threshold
    if cfg.title_hard_exclude_flags:
        params_effective["title_hard_exclude_flags"] = sorted(cfg.title_hard_exclude_flags)

    stats = {
        "pool_size": len(pool_indices),
        "distinct_artists": len(pool_artists),
        "eligible_artists": len(grouped),
        "seed_artist_key": seed_artist_key,
        "rng_seed": random_seed,
        # Exclusion counters for instrumentation
        "total_candidates_considered": total_candidates,
        "below_similarity_floor": below_floor_count,
        "below_genre_similarity": below_genre_count,  # NEW: genre gating exclusions
        "genre_overlap_guard_rejected": genre_overlap_guard_rejected,
        "genre_conflict_rejected": genre_conflict_rejected,
        "genre_conflict_penalty_applied": genre_conflict_penalty_applied,
        "title_exclusion_rejected": title_exclusion_rejected,
        "below_sonic_similarity": below_sonic_floor,
        "artist_cap_excluded": max(0, artist_cap_excluded),
        "eligible_count": len(eligible),
    }
    if duration_penalty_active:
        stats["duration_penalty_applied"] = duration_penalty_count
        stats["duration_cutoff_excluded"] = duration_cutoff_count
    if sonic_sim_pool is not None:
        if track_ids is not None:
            stats["seed_sonic_sim_track_ids"] = {
                str(track_ids[idx]): float(sonic_seed_sim[idx])
                for idx in pool_indices
            }
        stats["seed_sonic_sim"] = {int(idx): float(sonic_seed_sim[idx]) for idx in pool_indices}

    watched_raw = os.environ.get("PLAYLIST_WATCHED_ARTISTS", "")
    watched_keys = {
        _normalize_artist_key(part)
        for part in watched_raw.split(",")
        if part.strip()
    }
    if watched_keys:
        pool_set = set(int(i) for i in pool_indices)
        eligible_set = set(int(i) for i in eligible)
        logger.info(
            "Watched artists diagnostics: artist | total_tracks | admitted_count | in_allowed_pool | "
            "best_seed_sonic_sim | best_hybrid_sim | best_genre_sim | rejected_reason_counts | "
            "segment_pool_count | selected_count"
        )
        for watched in sorted(watched_keys):
            idxs = [
                int(i) for i, raw_artist in enumerate(artist_keys)
                if _normalize_artist_key(raw_artist) == watched
            ]
            if not idxs:
                logger.info(
                    "Watched artist: %s | total_tracks=0 | admitted_count=0 | in_allowed_pool=0 | "
                    "best_seed_sonic_sim=n/a | best_hybrid_sim=n/a | best_genre_sim=n/a | "
                    "rejected_reason_counts={} | segment_pool_count=n/a | selected_count=n/a",
                    watched,
                )
                continue
            reason_counts: Dict[str, int] = {}
            for idx in idxs:
                reason = _first_rejection_reason(
                    idx=idx,
                    seed_mask=seed_mask,
                    seed_sim_all=seed_sim_all,
                    similarity_floor=cfg.similarity_floor,
                    sonic_seed_sim=sonic_seed_sim,
                    sonic_floor=sonic_floor,
                    genre_sim_all=genre_sim_all,
                    min_genre_similarity=min_genre_similarity,
                    genre_conflict_result=genre_conflict_result,
                    genre_conflict_min_confidence=cfg.genre_conflict_min_confidence,
                    pool_set=pool_set,
                    eligible_set=eligible_set,
                )
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            best_sonic = (
                max(float(sonic_seed_sim[i]) for i in idxs)
                if sonic_seed_sim is not None
                else None
            )
            best_genre = (
                max(float(genre_sim_all[i]) for i in idxs)
                if genre_sim_all is not None
                else None
            )
            logger.info(
                "Watched artist: %s | total_tracks=%d | admitted_count=%d | in_allowed_pool=%d | "
                "best_seed_sonic_sim=%s | best_hybrid_sim=%.3f | best_genre_sim=%s | "
                "rejected_reason_counts=%s | segment_pool_count=n/a | selected_count=n/a",
                watched,
                len(idxs),
                sum(1 for i in idxs if i in eligible_set),
                sum(1 for i in idxs if i in pool_set),
                "n/a" if best_sonic is None else f"{best_sonic:.3f}",
                max(float(seed_sim_all[i]) for i in idxs),
                "n/a" if best_genre is None else f"{best_genre:.3f}",
                reason_counts,
            )

    # Stage-level logging
    try:
        sims = np.array(
            [s for idx, s in enumerate(sonic_seed_sim if sonic_seed_sim is not None else seed_sim_all) if not seed_mask[idx]],
            dtype=float,
        )
        if sims.size:
            # Include genre gating stats in log message
            genre_msg = f" rejected_genre={below_genre_count}" if below_genre_count > 0 else ""
            logger.info(
                "Candidate pool: mode=%s floor=%.2f admitted=%d rejected_sonic=%d%s total=%d min=%.3f p05=%.3f median=%.3f p95=%.3f max=%.3f",
                mode,
                sonic_floor if sonic_floor is not None else float("nan"),
                len(pool_indices),
                below_sonic_floor,
                genre_msg,
                total_candidates,
                float(np.min(sims)),
                float(np.percentile(sims, 5)),
                float(np.percentile(sims, 50)),
                float(np.percentile(sims, 95)),
                float(np.max(sims)),
            )
            if logger.isEnabledFor(logging.DEBUG) and sonic_seed_sim is not None:
                rejected_pairs = sorted(
                    [
                        (i, sonic_seed_sim[i])
                        for i in range(len(sonic_seed_sim))
                        if (not seed_mask[i]) and ((sonic_floor is not None and sonic_seed_sim[i] < sonic_floor) or seed_sim_all[i] < cfg.similarity_floor)
                    ],
                    key=lambda t: t[1],
                )
                if rejected_pairs:
                    logger.debug("Top 10 most negative sims (any floor): %s", rejected_pairs[:10])
                if sonic_floor is not None:
                    delta = 0.05
                    borderline = [
                        (i, sonic_seed_sim[i])
                        for i in range(len(sonic_seed_sim))
                        if (not seed_mask[i]) and sonic_seed_sim[i] >= sonic_floor - delta and sonic_seed_sim[i] <= sonic_floor + delta
                    ]
                    borderline = sorted(borderline, key=lambda t: abs(t[1] - sonic_floor))[:10]
                    if borderline:
                        logger.debug("Borderline around floor=%.2f: %s", sonic_floor, borderline)
    except Exception:
        logger.debug("Candidate pool sonic distribution logging failed", exc_info=True)

    return CandidatePoolResult(
        pool_indices=np.array(pool_indices, dtype=int),
        eligible_indices=np.array(eligible, dtype=int),
        seed_sim=seed_sim_pool,
        sonic_sim=sonic_sim_pool,
        stats=stats,
        params_effective=params_effective,
    )
