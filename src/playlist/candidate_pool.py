from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import CandidatePoolConfig
from src.similarity.hybrid import cosine_sim_matrix_to_vector

logger = logging.getLogger(__name__)


def _compute_genre_similarity(
    seed_genres: np.ndarray,
    candidate_genres: np.ndarray,
    method: str = "cosine",
) -> np.ndarray:
    """
    Compute genre similarity between seed and all candidates.

    Args:
        seed_genres: (D,) 1D array of seed's genres (binary or float)
        candidate_genres: (N, D) matrix of candidate genres
        method: "weighted_jaccard", "cosine", or "ensemble"

    Returns:
        (N,) similarity scores [0, 1]
    """
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


def build_candidate_pool(
    *,
    seed_idx: int,
    seed_indices: Optional[list[int]] = None,
    embedding: np.ndarray,  # (N, D)
    artist_keys: np.ndarray,  # (N,)
    track_ids: Optional[np.ndarray] = None,
    track_titles: Optional[np.ndarray] = None,
    track_artists: Optional[np.ndarray] = None,
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
    rejected_sonic: list[tuple[int, float]] = []

    # Compute genre similarity if provided
    genre_sim_all = None
    genre_raw_matrix = X_genre_raw
    genre_mask = None
    if broad_filters and genre_vocab:
        genre_mask = np.array([g.lower() not in broad_filters for g in genre_vocab], dtype=bool)
        if genre_raw_matrix is not None and genre_mask.shape[0] == genre_raw_matrix.shape[1]:
            genre_raw_matrix = genre_raw_matrix[:, genre_mask]

    if min_genre_similarity is not None and (X_genre_raw is not None or X_genre_smoothed is not None):
        # Choose matrix based on method
        if genre_method == "weighted_jaccard" and genre_raw_matrix is not None:
            genre_matrix = genre_raw_matrix
        elif X_genre_smoothed is not None:
            genre_matrix = X_genre_smoothed[:, genre_mask] if genre_mask is not None else X_genre_smoothed
        else:
            genre_matrix = genre_raw_matrix if genre_raw_matrix is not None else X_genre_smoothed

        seed_genres = genre_matrix[seed_idx]
        genre_sim_all = _compute_genre_similarity(seed_genres, genre_matrix, method=genre_method)
        genre_sim_all[seed_idx] = 1.0  # seed matches itself perfectly
        logger.info(
            "Candidate pool genre gating: method=%s, min_threshold=%.3f, mode=%s",
            genre_method, min_genre_similarity, mode
        )

    # Build initial eligible list (by hybrid similarity floor and sonic floor if provided)
    eligible: list[int] = []
    for i, sim in enumerate(seed_sim_all):
        if seed_mask[i]:
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
        # Optional overlap guard for narrow: require at least one non-broad shared tag
        if mode == "narrow" and genre_raw_matrix is not None:
            seed_binary = (genre_raw_matrix[seed_idx] > 0).astype(float)
            overlaps = (genre_raw_matrix > 0) & (seed_binary > 0)
            zero_overlap_mask = overlaps.sum(axis=1) == 0
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
    grouped: Dict[str, list[int]] = {}
    for idx in eligible:
        key = _normalize_artist_key(artist_keys[idx])
        grouped.setdefault(key, []).append(idx)

    artist_rank: list[tuple[str, float, list[int]]] = []
    for artist, idxs in grouped.items():
        max_sim = max(seed_sim_all[i] for i in idxs)
        artist_rank.append((artist, max_sim, idxs))
    artist_rank.sort(key=lambda t: (-t[1], t[0]))

    pool_indices: list[int] = []
    pool_artists: set[str] = set()
    seed_artist_key = _normalize_artist_key(artist_keys[seed_idx])

    for artist, _, idxs in artist_rank:
        per_artist_cap = cfg.candidates_per_artist
        if artist == seed_artist_key:
            per_artist_cap += cfg.seed_artist_bonus
        sorted_idxs = sorted(idxs, key=lambda i: (-seed_sim_all[i], i))
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
    if min_genre_similarity is not None:
        params_effective["genre_method"] = genre_method
        params_effective["min_genre_similarity"] = min_genre_similarity
        if broad_filters:
            params_effective["broad_filters"] = list(broad_filters)

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
        "below_sonic_similarity": below_sonic_floor,
        "artist_cap_excluded": max(0, artist_cap_excluded),
        "eligible_count": len(eligible),
    }
    if sonic_sim_pool is not None:
        if track_ids is not None:
            stats["seed_sonic_sim_track_ids"] = {
                str(track_ids[idx]): float(sonic_seed_sim[idx])
                for idx in pool_indices
            }
        stats["seed_sonic_sim"] = {int(idx): float(sonic_seed_sim[idx]) for idx in pool_indices}

    # Stage-level logging
    try:
        sims = np.array(
            [s for idx, s in enumerate(sonic_seed_sim if sonic_seed_sim is not None else seed_sim_all) if not seed_mask[idx]],
            dtype=float,
        )
        if sims.size:
            logger.info(
                "Candidate pool: mode=%s floor=%.2f admitted=%d rejected_sonic=%d total=%d min=%.3f p05=%.3f median=%.3f p95=%.3f max=%.3f",
                mode,
                sonic_floor if sonic_floor is not None else float("nan"),
                len(pool_indices),
                below_sonic_floor,
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
