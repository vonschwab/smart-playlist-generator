from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .config import CandidatePoolConfig
from .energy_rescue import select_energy_rescue
from .genre_compatibility import compute_raw_genre_compatibility

# Alias so callers can import as CandidateConfig from this module
CandidateConfig = CandidatePoolConfig
from .layered_genre_scoring import (
    layered_decision_to_diagnostics,
    score_layered_candidate,
)
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
    genre_compatibility_result: Any,
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
    if idx in eligible_set and idx not in pool_set:
        return "artist_cap"
    return "admitted" if idx in pool_set else "not_selected"


def _coerce_vocab(vocab: Optional[Sequence[Any]], width: int) -> Optional[list[str]]:
    if vocab is None:
        return None
    try:
        values = [str(value) for value in np.asarray(vocab, dtype=object).reshape(-1).tolist()]
    except Exception:
        return None
    if len(values) != int(width):
        return None
    return values


def _active_terms(vector: np.ndarray, vocab: Optional[list[str]]) -> list[str]:
    if vocab is None:
        return []
    values = np.asarray(vector, dtype=float).reshape(-1)
    return [vocab[i] for i, value in enumerate(values) if i < len(vocab) and float(value) > 0.0]


def _shared_terms(left: np.ndarray, right: np.ndarray, vocab: Optional[list[str]]) -> list[str]:
    if vocab is None:
        return []
    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    width = min(len(vocab), left_values.shape[0], right_values.shape[0])
    return [
        vocab[i]
        for i in range(width)
        if float(left_values[i]) > 0.0 and float(right_values[i]) > 0.0
    ]


def _layered_term_diagnostics(
    *,
    seed_leaf: np.ndarray,
    candidate_leaf: np.ndarray,
    seed_family: np.ndarray,
    candidate_family: np.ndarray,
    seed_bridge: np.ndarray,
    candidate_bridge: np.ndarray,
    seed_facet: np.ndarray,
    candidate_facet: np.ndarray,
    genre_leaf_vocab: Optional[Sequence[Any]],
    genre_family_vocab: Optional[Sequence[Any]],
    genre_bridge_vocab: Optional[Sequence[Any]],
    facet_vocab: Optional[Sequence[Any]],
) -> dict[str, list[str]]:
    leaf_vocab = _coerce_vocab(genre_leaf_vocab, np.asarray(seed_leaf).reshape(-1).shape[0])
    family_vocab = _coerce_vocab(genre_family_vocab, np.asarray(seed_family).reshape(-1).shape[0])
    bridge_vocab = _coerce_vocab(genre_bridge_vocab, np.asarray(seed_bridge).reshape(-1).shape[0])
    facet_names = _coerce_vocab(facet_vocab, np.asarray(seed_facet).reshape(-1).shape[0])
    return {
        "seed_leaf_terms": _active_terms(seed_leaf, leaf_vocab),
        "candidate_leaf_terms": _active_terms(candidate_leaf, leaf_vocab),
        "shared_leaf_terms": _shared_terms(seed_leaf, candidate_leaf, leaf_vocab),
        "seed_family_terms": _active_terms(seed_family, family_vocab),
        "candidate_family_terms": _active_terms(candidate_family, family_vocab),
        "shared_family_terms": _shared_terms(seed_family, candidate_family, family_vocab),
        "seed_bridge_terms": _active_terms(seed_bridge, bridge_vocab),
        "candidate_bridge_terms": _active_terms(candidate_bridge, bridge_vocab),
        "shared_bridge_terms": _shared_terms(seed_bridge, candidate_bridge, bridge_vocab),
        "seed_facet_terms": _active_terms(seed_facet, facet_names),
        "candidate_facet_terms": _active_terms(candidate_facet, facet_names),
        "shared_facet_terms": _shared_terms(seed_facet, candidate_facet, facet_names),
    }


def _build_layered_genre_shadow_diagnostics(
    *,
    seed_list: list[int],
    seed_mask: np.ndarray,
    pool_indices: list[int],
    eligible: list[int],
    seed_sim_all: np.ndarray,
    similarity_floor: float,
    sonic_seed_sim: Optional[np.ndarray],
    sonic_floor: Optional[float],
    genre_sim_all: Optional[np.ndarray],
    effective_genre_floor: Optional[float],
    genre_compatibility_result: Any,
    track_ids: Optional[np.ndarray],
    X_genre_leaf_idf: Optional[np.ndarray],
    X_genre_family: Optional[np.ndarray],
    X_genre_bridge: Optional[np.ndarray],
    X_facet: Optional[np.ndarray],
    genre_leaf_vocab: Optional[Sequence[Any]],
    genre_family_vocab: Optional[Sequence[Any]],
    genre_bridge_vocab: Optional[Sequence[Any]],
    facet_vocab: Optional[Sequence[Any]],
    mode: str,
    sample_limit: int,
) -> dict[str, Any]:
    matrices = (X_genre_leaf_idf, X_genre_family, X_genre_bridge, X_facet)
    if any(matrix is None for matrix in matrices):
        return {
            "enabled": False,
            "reason": "missing_layered_matrices",
        }

    leaf = np.asarray(X_genre_leaf_idf, dtype=float)
    family = np.asarray(X_genre_family, dtype=float)
    bridge = np.asarray(X_genre_bridge, dtype=float)
    facet = np.asarray(X_facet, dtype=float)
    row_count = len(seed_mask)
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in (leaf, family, bridge, facet)):
        return {
            "enabled": False,
            "reason": "layered_matrix_shape_mismatch",
        }
    if bridge.shape[1] != leaf.shape[1]:
        return {
            "enabled": False,
            "reason": "bridge_leaf_vocab_mismatch",
        }

    seed_leaf = np.max(leaf[seed_list], axis=0)
    seed_family = np.max(family[seed_list], axis=0)
    seed_bridge = np.max(bridge[seed_list], axis=0)
    seed_facet = np.max(facet[seed_list], axis=0)
    pool_set = set(int(i) for i in pool_indices)
    eligible_set = set(int(i) for i in eligible)
    sample_limit = max(0, int(sample_limit))

    evaluated_count = 0
    would_admit_count = 0
    broad_only_reject_count = 0
    bridge_supported_count = 0
    unexplained_jump_reject_count = 0
    legacy_disagreement_count = 0
    rows: list[dict[str, Any]] = []

    for idx in range(row_count):
        if bool(seed_mask[idx]):
            continue
        evaluated_count += 1
        decision = score_layered_candidate(
            seed_leaf=seed_leaf,
            candidate_leaf=leaf[idx],
            seed_family=seed_family,
            candidate_family=family[idx],
            seed_bridge=seed_bridge,
            candidate_bridge=bridge[idx],
            seed_facet=seed_facet,
            candidate_facet=facet[idx],
            mode=mode,
        )
        diagnostic = layered_decision_to_diagnostics(decision)
        legacy_admitted = idx in pool_set
        if bool(decision.admitted):
            would_admit_count += 1
        if decision.reason == "broad_only_without_leaf_support":
            broad_only_reject_count += 1
        if decision.reason == "bridge_supported":
            bridge_supported_count += 1
        if decision.reason == "unexplained_family_jump":
            unexplained_jump_reject_count += 1
        if bool(decision.admitted) != legacy_admitted:
            legacy_disagreement_count += 1

        row = {
            "index": int(idx),
            "track_id": str(track_ids[idx]) if track_ids is not None else int(idx),
            "legacy_admitted": bool(legacy_admitted),
            "legacy_eligible": bool(idx in eligible_set),
            "legacy_reason": _first_rejection_reason(
                idx=idx,
                seed_mask=seed_mask,
                seed_sim_all=seed_sim_all,
                similarity_floor=similarity_floor,
                sonic_seed_sim=sonic_seed_sim,
                sonic_floor=sonic_floor,
                genre_sim_all=genre_sim_all,
                min_genre_similarity=effective_genre_floor,
                genre_compatibility_result=genre_compatibility_result,
                pool_set=pool_set,
                eligible_set=eligible_set,
            ),
            **diagnostic,
            **_layered_term_diagnostics(
                seed_leaf=seed_leaf,
                candidate_leaf=leaf[idx],
                seed_family=seed_family,
                candidate_family=family[idx],
                seed_bridge=seed_bridge,
                candidate_bridge=bridge[idx],
                seed_facet=seed_facet,
                candidate_facet=facet[idx],
                genre_leaf_vocab=genre_leaf_vocab,
                genre_family_vocab=genre_family_vocab,
                genre_bridge_vocab=genre_bridge_vocab,
                facet_vocab=facet_vocab,
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            bool(row["admitted"]) == bool(row["legacy_admitted"]),
            0 if row["reason"] in {"broad_only_without_leaf_support", "unexplained_family_jump", "bridge_supported"} else 1,
            -float(row["score"]),
            int(row["index"]),
        )
    )
    if sample_limit:
        rows = rows[:sample_limit]
    else:
        rows = []

    return {
        "enabled": True,
        "mode": mode,
        "evaluated_count": int(evaluated_count),
        "would_admit_count": int(would_admit_count),
        "would_reject_count": int(evaluated_count - would_admit_count),
        "legacy_disagreement_count": int(legacy_disagreement_count),
        "broad_only_reject_count": int(broad_only_reject_count),
        "bridge_supported_count": int(bridge_supported_count),
        "unexplained_jump_reject_count": int(unexplained_jump_reject_count),
        "sample_limit": int(sample_limit),
        "samples": rows,
    }


def _validate_layered_matrices(
    *,
    row_count: int,
    X_genre_leaf_idf: Optional[np.ndarray],
    X_genre_family: Optional[np.ndarray],
    X_genre_bridge: Optional[np.ndarray],
    X_facet: Optional[np.ndarray],
) -> tuple[bool, str]:
    matrices = (X_genre_leaf_idf, X_genre_family, X_genre_bridge, X_facet)
    if any(matrix is None for matrix in matrices):
        return False, "missing_layered_matrices"
    leaf = np.asarray(X_genre_leaf_idf)
    family = np.asarray(X_genre_family)
    bridge = np.asarray(X_genre_bridge)
    facet = np.asarray(X_facet)
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in (leaf, family, bridge, facet)):
        return False, "layered_matrix_shape_mismatch"
    if bridge.shape[1] != leaf.shape[1]:
        return False, "bridge_leaf_vocab_mismatch"
    return True, "ok"


def _apply_layered_genre_admission(
    *,
    seed_list: list[int],
    eligible: list[int],
    X_genre_leaf_idf: np.ndarray,
    X_genre_family: np.ndarray,
    X_genre_bridge: np.ndarray,
    X_facet: np.ndarray,
    mode: str,
    track_ids: Optional[np.ndarray],
    genre_leaf_vocab: Optional[Sequence[Any]],
    genre_family_vocab: Optional[Sequence[Any]],
    genre_bridge_vocab: Optional[Sequence[Any]],
    facet_vocab: Optional[Sequence[Any]],
) -> tuple[list[int], dict[str, Any]]:
    leaf = np.asarray(X_genre_leaf_idf, dtype=float)
    family = np.asarray(X_genre_family, dtype=float)
    bridge = np.asarray(X_genre_bridge, dtype=float)
    facet = np.asarray(X_facet, dtype=float)

    seed_leaf = np.max(leaf[seed_list], axis=0)
    seed_family = np.max(family[seed_list], axis=0)
    seed_bridge = np.max(bridge[seed_list], axis=0)
    seed_facet = np.max(facet[seed_list], axis=0)

    admitted: list[int] = []
    rejection_reason_counts: dict[str, int] = {}
    admitted_track_ids: list[str] = []
    rejected_samples: list[dict[str, Any]] = []

    for idx in eligible:
        decision = score_layered_candidate(
            seed_leaf=seed_leaf,
            candidate_leaf=leaf[idx],
            seed_family=seed_family,
            candidate_family=family[idx],
            seed_bridge=seed_bridge,
            candidate_bridge=bridge[idx],
            seed_facet=seed_facet,
            candidate_facet=facet[idx],
            mode=mode,
        )
        if decision.admitted:
            admitted.append(int(idx))
            admitted_track_ids.append(str(track_ids[idx]) if track_ids is not None else str(idx))
            continue
        rejection_reason_counts[decision.reason] = rejection_reason_counts.get(decision.reason, 0) + 1
        if len(rejected_samples) < 25:
            row = {
                "index": int(idx),
                "track_id": str(track_ids[idx]) if track_ids is not None else int(idx),
                **layered_decision_to_diagnostics(decision),
                **_layered_term_diagnostics(
                    seed_leaf=seed_leaf,
                    candidate_leaf=leaf[idx],
                    seed_family=seed_family,
                    candidate_family=family[idx],
                    seed_bridge=seed_bridge,
                    candidate_bridge=bridge[idx],
                    seed_facet=seed_facet,
                    candidate_facet=facet[idx],
                    genre_leaf_vocab=genre_leaf_vocab,
                    genre_family_vocab=genre_family_vocab,
                    genre_bridge_vocab=genre_bridge_vocab,
                    facet_vocab=facet_vocab,
                ),
            }
            rejected_samples.append(row)

    return admitted, {
        "source": "layered",
        "applied": True,
        "input_eligible_count": int(len(eligible)),
        "admitted_count": int(len(admitted)),
        "rejected_count": int(len(eligible) - len(admitted)),
        "rejection_reason_counts": rejection_reason_counts,
        "admitted_track_ids": admitted_track_ids,
        "rejected_samples": rejected_samples,
    }


def _apply_popularity_gate(
    eligible: list[int],
    popularity_ranks: np.ndarray,
    rank_cutoff: int,
) -> tuple[list[int], int]:
    """Oops, All Bangers admission gate: keep only candidates whose 0-based Last.fm
    rank is in [0, rank_cutoff). -1 (uncached / not in the artist's top-N) and any
    rank >= cutoff are non-bangers and excluded. Returns (kept, excluded_count)."""
    kept = [i for i in eligible if 0 <= int(popularity_ranks[i]) < rank_cutoff]
    return kept, len(eligible) - len(kept)


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
    X_genre_dense: Optional[np.ndarray] = None,  # (N, dim) L2-normalized dense embedding
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
    genre_vocab: Optional[list[str]] = None,
    broad_filters: tuple[str, ...] = (),
    mode: str = "dynamic",  # "dynamic", "narrow", "discover"
    uncap_pool: bool = False,  # seeded mode: skip max_pool_size; per-artist cap still applies
    perceptual_bpm: Optional[np.ndarray] = None,
    tempo_stability: Optional[np.ndarray] = None,
    onset_rate: Optional[np.ndarray] = None,
    X_energy: Optional[np.ndarray] = None,
    genre_admission_percentile: Optional[float] = None,
    genre_admission_aggregate: str = "centroid",
    layered_genre_diagnostics: bool = False,
    X_genre_leaf_idf: Optional[np.ndarray] = None,
    X_genre_family: Optional[np.ndarray] = None,
    X_genre_bridge: Optional[np.ndarray] = None,
    X_facet: Optional[np.ndarray] = None,
    genre_leaf_vocab: Optional[Sequence[Any]] = None,
    genre_family_vocab: Optional[Sequence[Any]] = None,
    genre_bridge_vocab: Optional[Sequence[Any]] = None,
    facet_vocab: Optional[Sequence[Any]] = None,
    layered_genre_diagnostics_limit: int = 25,
    genre_graph_source: str = "legacy",
    popularity_ranks: Optional[np.ndarray] = None,
    popularity_rank_cutoff: Optional[int] = None,
    # Tag steering (soft): blend a user-selected genre target into the dense
    # admission centroid. None = feature off (byte-identical legacy behavior).
    steering_target: Optional[np.ndarray] = None,
    steering_blend: float = 0.5,
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

        # Per-seed adaptive sonic floor: replaces min_sonic_similarity when
        # sonic_admission_percentile is set and > 0.  Admits ~top (1-p) fraction
        # of the seed's own sonic similarity distribution (distribution-relative,
        # so it survives embedding rebuilds and adapts to sparse vs dense seeds).
        _sap = getattr(cfg, "sonic_admission_percentile", None)
        if _sap is not None and float(_sap) > 0.0:
            from src.playlist.pier_bridge.percentiles import floor_at_percentile
            _sdist = np.asarray(sonic_seed_sim, dtype=np.float64).copy()
            for _si in seed_list:
                if 0 <= int(_si) < _sdist.shape[0]:
                    _sdist[int(_si)] = np.nan
            _sfin = _sdist[np.isfinite(_sdist)]
            sonic_floor = floor_at_percentile(_sfin, float(_sap))
            logger.info(
                "Sonic admission percentile active: p=%.2f -> effective sonic_floor=%.3f (was abs=%s)",
                float(_sap),
                float(sonic_floor),
                cfg.min_sonic_similarity,
            )

    # ── Rhythm-fail accumulator (for energy rescue) ──────────────────────────
    # Tracks which candidates were rejected by rhythm bands so rescue can re-admit
    # the genre+sonic-OK subset.  Initialized to all-False (no-op when rescue is off).
    rhythm_fail = np.zeros(len(seed_sim_all), dtype=bool)

    # ── BPM admission gate (Tier 1) ──────────────────────────────────────────
    bpm_seed_min_dist: Optional[np.ndarray] = None
    below_bpm_floor = 0
    if (
        not np.isinf(float(getattr(cfg, "bpm_admission_max_log_distance", float("inf"))))
        and perceptual_bpm is not None
    ):
        from src.playlist.bpm_axis import bpm_log_distance

        max_log = float(cfg.bpm_admission_max_log_distance)
        stability_min = float(getattr(cfg, "bpm_stability_min", 0.5))
        seed_bpm_vals = perceptual_bpm[seed_list]
        dist_cols = np.stack(
            [bpm_log_distance(perceptual_bpm, float(sb)) for sb in seed_bpm_vals], axis=1
        )
        bpm_seed_min_dist = np.min(dist_cols, axis=1)

        bypass = np.isnan(perceptual_bpm)
        if tempo_stability is not None:
            bypass = bypass | (np.asarray(tempo_stability) < stability_min)

        bpm_fail = ~bypass & (bpm_seed_min_dist > max_log)
        bpm_fail[seed_list] = False  # seeds never self-rejected
        rhythm_fail = rhythm_fail | bpm_fail

        count_bpm_rejected = int(np.sum(bpm_fail))
        below_bpm_floor = count_bpm_rejected
        seed_sim_all[bpm_fail] = -2.0  # below any valid cosine similarity [-1, 1]

        logger.info(
            "BPM admission gate: max_log_distance=%.2f rejected=%d",
            max_log, count_bpm_rejected,
        )

    # ── Onset-rate admission band ────────────────────────────────────────────
    if (
        not np.isinf(float(getattr(cfg, "onset_admission_max_log_distance", float("inf"))))
        and onset_rate is not None
    ):
        from src.playlist.bpm_axis import bpm_log_distance as _onset_log_distance

        max_log_onset = float(cfg.onset_admission_max_log_distance)
        seed_onset_vals = onset_rate[seed_list]
        onset_dist_cols = np.stack(
            [_onset_log_distance(onset_rate, float(so)) for so in seed_onset_vals], axis=1
        )
        onset_seed_min_dist = np.min(onset_dist_cols, axis=1)

        onset_bypass = np.isnan(onset_rate)  # NaN bypass only; no stability bypass
        onset_fail = ~onset_bypass & (onset_seed_min_dist > max_log_onset)
        onset_fail[seed_list] = False  # seeds never self-rejected
        rhythm_fail = rhythm_fail | onset_fail

        seed_sim_all[onset_fail] = -2.0
        logger.info(
            "Onset admission band: max_log_distance=%.2f rejected=%d",
            max_log_onset, int(np.sum(onset_fail)),
        )

    # Track exclusion reasons for instrumentation
    total_candidates = len(seed_sim_all) - len(seed_list)  # exclude all seeds
    below_floor_count = 0
    below_genre_count = 0
    genre_overlap_guard_rejected = 0
    genre_compatibility_penalty_applied = 0
    title_exclusion_rejected = 0
    rejected_sonic: list[tuple[int, float]] = []
    genre_graph_source = str(genre_graph_source or "legacy").strip().lower()
    if genre_graph_source not in {"legacy", "layered_shadow", "layered"}:
        genre_graph_source = "legacy"
    legacy_flat_genre_gate_applied = genre_graph_source != "layered"
    if not legacy_flat_genre_gate_applied:
        X_genre_raw = None
        X_genre_smoothed = None
        X_genre_dense = None
        min_genre_similarity = None
    layered_genre_admission_summary: Optional[dict[str, Any]] = None

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

    # Effective genre floor: may be overridden by per-seed adaptive percentile in the dense path.
    # Initialized to the fixed floor so the sparse path and the None-percentile dense path are
    # IDENTICAL to legacy behavior.
    effective_genre_floor: Optional[float] = min_genre_similarity

    # Dense PMI-SVD path: when X_genre_dense is available, use it in preference to sparse methods.
    # Note: also activates when genre_admission_percentile is set without a fixed min_genre_similarity.
    _use_dense = X_genre_dense is not None and (
        min_genre_similarity is not None or genre_admission_percentile is not None
    )
    if _use_dense:
        _agg = str(genre_admission_aggregate or "centroid").strip().lower()
        if _agg not in {"centroid", "per_seed"}:
            _agg = "centroid"
        if steering_target is not None and _agg == "per_seed":
            logger.info("Tag steering: forcing genre_admission_aggregate=centroid (was per_seed)")
            _agg = "centroid"

        if _agg == "per_seed":
            # Union-of-neighborhoods: each seed contributes its own genre floor.
            # A track is admitted if it passes ANY seed's floor (union semantics).
            # genre_sim_all is set to the max-over-seeds cosine for downstream use
            # (artist ranking, overlap guard).  The effective_genre_floor is the
            # minimum per-seed floor; combined with max-over-seeds, this approximates
            # true union semantics (can only over-admit by one edge case, never under).
            from src.playlist.pier_bridge.percentiles import floor_at_percentile
            seed_mat = X_genre_dense[seed_list]  # (S, dim)
            sims_per_seed = (X_genre_dense @ seed_mat.T).astype(np.float64)  # (N, S)
            # Seeds are always admitted — force their rows to 1.0 so they don't
            # distort the floor calculation when used as non-seed members.
            seed_set = set(seed_list)
            non_seed_mask = np.array(
                [i not in seed_set for i in range(sims_per_seed.shape[0])], dtype=bool
            )
            if genre_admission_percentile is not None:
                per_seed_floors = np.array([
                    floor_at_percentile(
                        sims_per_seed[non_seed_mask, si],
                        genre_admission_percentile,
                    )
                    for si in range(len(seed_list))
                ])
                effective_genre_floor = float(np.min(per_seed_floors))
            else:
                effective_genre_floor = min_genre_similarity
            genre_sim_all = np.max(sims_per_seed, axis=1)
            genre_sim_all[seed_idx] = 1.0
        else:
            # Centroid path (legacy default): average all seed vectors into one centroid.
            # X_genre_dense rows are already L2-normalized; seed vec = average of seed rows.
            seed_dense = X_genre_dense[seed_list].mean(axis=0)
            seed_dense_norm = np.linalg.norm(seed_dense)
            if seed_dense_norm > 1e-12:
                seed_dense = seed_dense / seed_dense_norm
            if steering_target is not None:
                _blend = float(np.clip(steering_blend, 0.0, 1.0))
                _steered = (1.0 - _blend) * seed_dense + _blend * np.asarray(
                    steering_target, dtype=seed_dense.dtype
                )
                _steered_norm = np.linalg.norm(_steered)
                if _steered_norm > 1e-12:
                    seed_dense = _steered / _steered_norm
                logger.info(
                    "Tag steering pool lever: blend=%.2f applied to admission centroid",
                    _blend,
                )
            genre_sim_all = (X_genre_dense @ seed_dense).astype(np.float64)
            # Do NOT clip to [0,1]. The mean-centered dense embedding produces negative cosine
            # similarities for genuinely dissimilar genres — clipping collapses them all to 0.000
            # and destroys the rank ordering, making the percentile floor non-functional for
            # niche artists (jazz, hyperpop) where most of the library is negative-sim.
            genre_sim_all[seed_idx] = 1.0
            # Per-seed adaptive admission floor: derive from THIS seed's dense-sim distribution.
            # When genre_admission_percentile is set, the percentile IS the floor — do not override
            # with the fixed min_genre_similarity, which was calibrated for the old anisotropic
            # embedding (p50≈0.24) and is far too tight at the new mean-centered scale (p50≈-0.14).
            if genre_admission_percentile is not None:
                from src.playlist.pier_bridge.percentiles import floor_at_percentile
                _dist = genre_sim_all.copy()
                _dist[seed_idx] = np.nan
                effective_genre_floor = floor_at_percentile(_dist[~np.isnan(_dist)], genre_admission_percentile)
            else:
                effective_genre_floor = min_genre_similarity

            if steering_target is not None and effective_genre_floor is not None:
                _aff = (X_genre_dense @ np.asarray(steering_target, dtype=np.float64)).astype(np.float64)
                _adm = _aff[genre_sim_all >= float(effective_genre_floor)]
                if _adm.size:
                    logger.info(
                        "Tag steering pool affinity (genre-admitted set): "
                        "p10=%.3f p50=%.3f p90=%.3f n=%d",
                        float(np.percentile(_adm, 10)), float(np.percentile(_adm, 50)),
                        float(np.percentile(_adm, 90)), int(_adm.size),
                    )

        logger.info(
            "Candidate pool genre gating: method=dense (PMI-SVD), dim=%d, "
            "aggregate=%s, admission_percentile=%s, effective_floor=%.3f, mode=%s",
            X_genre_dense.shape[1],
            _agg,
            genre_admission_percentile,
            float(effective_genre_floor) if effective_genre_floor is not None else float("nan"),
            mode,
        )

    elif (
        min_genre_similarity is not None
        or (genre_admission_percentile is not None and float(genre_admission_percentile) > 0.0)
    ) and (X_genre_raw is not None or X_genre_smoothed is not None):
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
            "Candidate pool genre gating: method=%s, min_threshold=%s, mode=%s, idf=%s",
            genre_method,
            f"{float(min_genre_similarity):.3f}" if min_genre_similarity is not None else "None",
            mode,
            "on" if idf_weights is not None else "off",
        )
        # Adaptive percentile floor on the sparse distribution.
        # Mirrors the dense centroid path: NaN-mask seed rows, drop non-finite, floor_at_percentile.
        # When genre_admission_percentile is None/0, effective_genre_floor stays = min_genre_similarity
        # (legacy, golden-safe).
        if genre_admission_percentile is not None and float(genre_admission_percentile) > 0.0:
            from src.playlist.pier_bridge.percentiles import floor_at_percentile
            _gdist = np.asarray(genre_sim_all, dtype=np.float64).copy()
            for _si in seed_list:
                if 0 <= int(_si) < _gdist.shape[0]:
                    _gdist[int(_si)] = np.nan
            _gfin = _gdist[np.isfinite(_gdist)]
            effective_genre_floor = floor_at_percentile(_gfin, float(genre_admission_percentile))
            logger.info(
                "Genre admission percentile (sparse) active: p=%.2f -> effective_genre_floor=%.3f",
                float(genre_admission_percentile),
                float(effective_genre_floor),
            )

    genre_compatibility_result = None
    if (
        cfg.genre_compatibility_enabled
        and genre_raw_matrix is not None
        and genre_raw_matrix.ndim == 2
        and genre_raw_matrix.shape[0] == len(seed_sim_all)
        and genre_raw_matrix.shape[1] == len(genre_vocab_effective)
    ):
        seed_raw = np.max(genre_raw_matrix[seed_list], axis=0)
        genre_compatibility_result = compute_raw_genre_compatibility(
            seed_raw=seed_raw,
            candidate_raw=genre_raw_matrix,
            genre_vocab=genre_vocab_effective,
            compatible_threshold=cfg.genre_compatibility_compatible_threshold,
            conflict_threshold=cfg.genre_compatibility_conflict_threshold,
            penalty_strength=cfg.genre_compatibility_penalty_strength,
        )
        if cfg.genre_compatibility_penalty_strength > 0:
            penalty = np.asarray(genre_compatibility_result.penalty, dtype=float)
            penalized = (penalty > 0) & (~seed_mask)
            genre_compatibility_penalty_applied = int(np.count_nonzero(penalized))
            seed_sim_all = seed_sim_all - penalty
            logger.info(
                "Genre compatibility penalty applied: penalized=%d strength=%.3f",
                genre_compatibility_penalty_applied,
                float(cfg.genre_compatibility_penalty_strength),
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
            # Hard gate: exclude candidates below threshold.
            # Uses effective_genre_floor (which equals min_genre_similarity when
            # genre_admission_percentile is None — exact legacy behavior).
            eligible_before_genre = len(eligible)
            eligible = [i for i in eligible if genre_sim_all[i] >= effective_genre_floor]
            below_genre_count = eligible_before_genre - len(eligible)
            logger.info(
                "Genre hard gate applied: %d candidates excluded (mode=%s)",
                below_genre_count, mode
            )
        # For "discover" mode, we compute genre_sim but don't exclude (soft penalty later if needed)

    # ── Energy admission-rescue (pace as a co-equal axis) ────────────────────
    # Re-admit tracks rejected ONLY by the rhythm bands (onset/BPM) that still
    # clear the genre AND sonic floors, choosing an arousal-spanning subset so
    # the pool carries on-arc-energy candidates even in tight modes. Additive;
    # never removes. Genre + sonic floors fully respected. No-op when k_energy=0.
    k_energy = int(getattr(cfg, "pace_rescue_k_energy", 0))
    if k_energy > 0 and X_energy is not None and np.any(rhythm_fail):
        eligible_set = set(eligible)
        sonic_ok = (
            sonic_seed_sim is not None and sonic_floor is not None
        )
        source = []
        for i in np.nonzero(rhythm_fail)[0]:
            i = int(i)
            if i in eligible_set or seed_mask[i]:
                continue
            if sonic_ok and (sonic_seed_sim[i] + epsilon) < sonic_floor:
                continue  # sonic safety floor — never rescue a disconnected track
            if (
                genre_sim_all is not None
                and effective_genre_floor is not None
                and genre_sim_all[i] < effective_genre_floor
            ):
                continue  # genre authority preserved
            source.append(i)
        rescued = select_energy_rescue(np.asarray(X_energy, dtype=float), source, k_energy)
        for i in rescued:
            # Restore a genuine rank score so the track survives similarity_floor
            # and ranks sensibly (it was set to the rhythm sentinel -2.0).
            seed_sim_all[i] = float(sonic_seed_sim[i]) if sonic_ok else float(cfg.similarity_floor)
            eligible.append(i)
        if rescued:
            logger.info(
                "Energy rescue: admitted=%d from rhythm-rejected (k_energy=%d, source=%d)",
                len(rescued), k_energy, len(source),
            )

    layered_ready, layered_reason = _validate_layered_matrices(
        row_count=len(seed_sim_all),
        X_genre_leaf_idf=X_genre_leaf_idf,
        X_genre_family=X_genre_family,
        X_genre_bridge=X_genre_bridge,
        X_facet=X_facet,
    )
    if genre_graph_source == "layered":
        if layered_ready:
            eligible_before_layered = len(eligible)
            eligible, layered_genre_admission_summary = _apply_layered_genre_admission(
                seed_list=seed_list,
                eligible=eligible,
                X_genre_leaf_idf=np.asarray(X_genre_leaf_idf, dtype=float),
                X_genre_family=np.asarray(X_genre_family, dtype=float),
                X_genre_bridge=np.asarray(X_genre_bridge, dtype=float),
                X_facet=np.asarray(X_facet, dtype=float),
                mode=mode,
                track_ids=track_ids,
                genre_leaf_vocab=genre_leaf_vocab,
                genre_family_vocab=genre_family_vocab,
                genre_bridge_vocab=genre_bridge_vocab,
                facet_vocab=facet_vocab,
            )
            layered_genre_admission_summary["legacy_flat_genre_gate_applied"] = False
            logger.info(
                "Layered genre admission applied: before=%d after=%d rejected=%d mode=%s",
                eligible_before_layered,
                len(eligible),
                eligible_before_layered - len(eligible),
                mode,
            )
        else:
            layered_genre_admission_summary = {
                "source": "layered",
                "applied": False,
                "reason": layered_reason,
                "legacy_flat_genre_gate_applied": False,
            }
            logger.warning(
                "Layered genre admission requested but unavailable: %s; using non-genre candidate admission.",
                layered_reason,
            )
    elif genre_graph_source == "layered_shadow":
        layered_genre_admission_summary = {
            "source": "layered_shadow",
            "applied": False,
            "reason": "shadow_only",
        }

    # ── Oops, All Bangers: popularity admission gate ─────────────────────────
    # Final eligibility filter so EVERY pooled track is a banger, including any
    # energy-rescued tracks. NaN/-1 (uncached / not in the artist's top-N) excluded.
    if popularity_ranks is not None and popularity_rank_cutoff is not None:
        _before_pop = len(eligible)
        eligible, _pop_excluded = _apply_popularity_gate(
            eligible, np.asarray(popularity_ranks), int(popularity_rank_cutoff)
        )
        logger.info(
            "Popularity gate applied: cutoff=top-%d before=%d after=%d excluded=%d",
            int(popularity_rank_cutoff), _before_pop, len(eligible), _pop_excluded,
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
            if not uncap_pool and len(pool_indices) >= cfg.max_pool_size and len(pool_artists) >= cfg.target_artists:
                break
            pool_indices.append(idx)
        pool_artists.add(artist)
        if not uncap_pool and len(pool_indices) >= cfg.max_pool_size and len(pool_artists) >= cfg.target_artists:
            break

    pool_indices = list(dict.fromkeys(pool_indices))  # dedupe, preserve order

    # Capture pool size before backstop so artist_cap_excluded reflects the normal
    # walk only (not inflated/deflated by backfill).
    _pool_size_before_backstop = len(pool_indices)

    # Never-starve backstop (Task 3): if the pool is below the minimum target,
    # backfill from the highest-sonic-sim candidates not yet admitted.
    # Per-artist cap is respected; seeds are never admitted.
    # Default min_pool_size=0 disables this → byte-identical legacy behavior.
    _min_pool_size = int(getattr(cfg, "min_pool_size", 0) or 0)
    if _min_pool_size > 0 and len(pool_indices) < _min_pool_size:
        from collections import Counter
        _already = set(int(i) for i in pool_indices)
        _per_artist: Counter = Counter(
            str(artist_keys[i]) for i in pool_indices
        )
        _cap = int(getattr(cfg, "candidates_per_artist", 6) or 6)
        _seed_set = set(int(i) for i in seed_list)
        _gate_on = popularity_ranks is not None and popularity_rank_cutoff is not None
        _ranked = sorted(
            (i for i in range(len(track_ids))
             if i not in _already and i not in _seed_set
             and (not _gate_on or 0 <= int(popularity_ranks[i]) < int(popularity_rank_cutoff))),
            # When no sonic embedding is present, admission order falls back to
            # index order (arbitrary but bounded + still artist-cap-respecting).
            key=lambda i: float(sonic_seed_sim[i]) if sonic_seed_sim is not None else 0.0,
            reverse=True,
        )
        _added = 0
        for i in _ranked:
            if len(pool_indices) >= _min_pool_size:
                break
            _ak = str(artist_keys[i])
            if _per_artist[_ak] >= _cap:
                continue
            pool_indices.append(int(i))
            _already.add(int(i))
            _per_artist[_ak] += 1
            pool_artists.add(_ak)  # keep distinct_artists accurate after backfill
            _added += 1
        if _added:
            logger.info(
                "Min-pool backstop: pool %d below min %d; admitted %d more (top sonic-sim, artist-cap respected)",
                len(pool_indices) - _added,
                _min_pool_size,
                _added,
            )

    seed_sim_pool = np.array([seed_sim_all[i] for i in pool_indices], dtype=float)
    sonic_sim_pool = (
        np.array([sonic_seed_sim[i] for i in pool_indices], dtype=float)
        if sonic_seed_sim is not None
        else None
    )

    # Count how many were excluded due to artist cap during the normal walk only.
    # Uses _pool_size_before_backstop so backfill doesn't make this go negative.
    artist_cap_excluded = len(eligible) - _pool_size_before_backstop

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
    if genre_graph_source != "legacy":
        params_effective["genre_graph_source"] = genre_graph_source
    if cfg.genre_compatibility_enabled:
        params_effective["genre_compatibility_enabled"] = True
        params_effective["genre_compatibility_penalty_strength"] = cfg.genre_compatibility_penalty_strength
        params_effective["genre_compatibility_compatible_threshold"] = cfg.genre_compatibility_compatible_threshold
        params_effective["genre_compatibility_conflict_threshold"] = cfg.genre_compatibility_conflict_threshold
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
        "genre_compatibility_penalty_applied": genre_compatibility_penalty_applied,
        "title_exclusion_rejected": title_exclusion_rejected,
        "below_sonic_similarity": below_sonic_floor,
        "below_bpm_floor": below_bpm_floor,
        "artist_cap_excluded": max(0, artist_cap_excluded),
        "eligible_count": len(eligible),
        # Effective genre floor after percentile compute (or fixed absolute floor when
        # genre_admission_percentile is None/0).  None when genre gating is off entirely.
        "effective_genre_floor": float(effective_genre_floor) if effective_genre_floor is not None else None,
    }
    if duration_penalty_active:
        stats["duration_penalty_applied"] = duration_penalty_count
        stats["duration_cutoff_excluded"] = duration_cutoff_count
    if layered_genre_admission_summary is not None:
        stats["layered_genre_admission"] = layered_genre_admission_summary
    if sonic_sim_pool is not None:
        if track_ids is not None:
            stats["seed_sonic_sim_track_ids"] = {
                str(track_ids[idx]): float(sonic_seed_sim[idx])
                for idx in pool_indices
            }
        stats["seed_sonic_sim"] = {int(idx): float(sonic_seed_sim[idx]) for idx in pool_indices}
    if layered_genre_diagnostics or genre_graph_source in {"layered_shadow", "layered"}:
        stats["layered_genre_shadow"] = _build_layered_genre_shadow_diagnostics(
            seed_list=seed_list,
            seed_mask=seed_mask,
            pool_indices=pool_indices,
            eligible=eligible,
            seed_sim_all=seed_sim_all,
            similarity_floor=cfg.similarity_floor,
            sonic_seed_sim=sonic_seed_sim,
            sonic_floor=sonic_floor,
            genre_sim_all=genre_sim_all,
            effective_genre_floor=effective_genre_floor,
            genre_compatibility_result=genre_compatibility_result,
            track_ids=track_ids,
            X_genre_leaf_idf=X_genre_leaf_idf,
            X_genre_family=X_genre_family,
            X_genre_bridge=X_genre_bridge,
            X_facet=X_facet,
            genre_leaf_vocab=genre_leaf_vocab,
            genre_family_vocab=genre_family_vocab,
            genre_bridge_vocab=genre_bridge_vocab,
            facet_vocab=facet_vocab,
            mode=mode,
            sample_limit=layered_genre_diagnostics_limit,
        )

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
                    min_genre_similarity=effective_genre_floor,
                    genre_compatibility_result=genre_compatibility_result,
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
