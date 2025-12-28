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
    seed_sim: np.ndarray  # seed similarity for each pool element (P,)
    stats: Dict[str, Any]  # pool size, distinct artists, etc.
    params_effective: Dict[str, Any]


def _normalize_artist_key(raw: Any) -> str:
    txt = "" if raw is None else str(raw)
    txt = txt.strip().lower()
    return txt


def build_candidate_pool(
    *,
    seed_idx: int,
    embedding: np.ndarray,  # (N, D)
    artist_keys: np.ndarray,  # (N,)
    cfg: CandidatePoolConfig,
    random_seed: int,
    # Genre gating (optional)
    X_genre_raw: Optional[np.ndarray] = None,  # (N, D_genre) binary genres
    X_genre_smoothed: Optional[np.ndarray] = None,  # (N, D_genre) float genres
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
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
    seed_vec = emb_norm[seed_idx]
    seed_sim_all = cosine_sim_matrix_to_vector(emb_norm, seed_vec)
    seed_sim_all[seed_idx] = -1.0

    # Track exclusion reasons for instrumentation
    total_candidates = len(seed_sim_all) - 1  # exclude seed itself
    below_floor_count = 0
    below_genre_count = 0

    # Compute genre similarity if provided
    genre_sim_all = None
    if min_genre_similarity is not None and (X_genre_raw is not None or X_genre_smoothed is not None):
        # Choose matrix based on method
        if genre_method == "weighted_jaccard" and X_genre_raw is not None:
            genre_matrix = X_genre_raw
        elif X_genre_smoothed is not None:
            genre_matrix = X_genre_smoothed
        else:
            genre_matrix = X_genre_raw if X_genre_raw is not None else X_genre_smoothed

        seed_genres = genre_matrix[seed_idx]
        genre_sim_all = _compute_genre_similarity(seed_genres, genre_matrix, method=genre_method)
        genre_sim_all[seed_idx] = 1.0  # seed matches itself perfectly
        logger.info(
            "Candidate pool genre gating: method=%s, min_threshold=%.3f, mode=%s",
            genre_method, min_genre_similarity, mode
        )

    # Build initial eligible list (by similarity floor)
    for i, sim in enumerate(seed_sim_all):
        if i != seed_idx and sim < cfg.similarity_floor:
            below_floor_count += 1

    eligible = [
        i for i, sim in enumerate(seed_sim_all) if i != seed_idx and sim >= cfg.similarity_floor
    ]

    # Apply genre gate (hard gate for dynamic/narrow, soft for discover)
    if genre_sim_all is not None:
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

    # Count how many were excluded due to artist cap (those not taken from eligible artists)
    artist_cap_excluded = len(eligible) - len(pool_indices)

    params_effective = {
        "similarity_floor": cfg.similarity_floor,
        "max_pool_size": cfg.max_pool_size,
        "target_artists": cfg.target_artists,
        "candidates_per_artist": cfg.candidates_per_artist,
        "seed_artist_bonus": cfg.seed_artist_bonus,
    }
    if min_genre_similarity is not None:
        params_effective["genre_method"] = genre_method
        params_effective["min_genre_similarity"] = min_genre_similarity

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
        "artist_cap_excluded": max(0, artist_cap_excluded),
        "eligible_count": len(eligible),
    }
    return CandidatePoolResult(
        pool_indices=np.array(pool_indices, dtype=int),
        seed_sim=seed_sim_pool,
        stats=stats,
        params_effective=params_effective,
    )

