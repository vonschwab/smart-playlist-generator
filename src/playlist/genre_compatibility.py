from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GenreCompatibilityResult:
    compatible_mass: np.ndarray
    conflict_mass: np.ndarray
    neutral_mass: np.ndarray
    confidence: np.ndarray
    penalty: np.ndarray
    missing_or_sparse: np.ndarray


def _idf_weights(candidate_raw: np.ndarray) -> np.ndarray:
    presence = np.count_nonzero(candidate_raw > 0, axis=0).astype(float)
    n = max(1, int(candidate_raw.shape[0]))
    idf = np.log((1.0 + n) / (1.0 + presence)) + 1.0
    max_val = float(np.max(idf)) if idf.size else 1.0
    return idf / max(max_val, 1e-12)


def _affinity_matrix(
    genre_vocab: Sequence[str],
    genre_affinity: np.ndarray | None,
) -> np.ndarray:
    n = len(genre_vocab)
    if genre_affinity is not None and tuple(genre_affinity.shape) == (n, n):
        return np.asarray(genre_affinity, dtype=float)
    return np.eye(n, dtype=float)


def compute_raw_genre_compatibility(
    *,
    seed_raw: np.ndarray,
    candidate_raw: np.ndarray,
    genre_vocab: Sequence[str],
    genre_affinity: np.ndarray | None = None,
    compatible_threshold: float = 0.35,
    conflict_threshold: float = 0.15,
    penalty_strength: float = 1.0,
) -> GenreCompatibilityResult:
    seed = np.asarray(seed_raw, dtype=float).reshape(-1)
    candidates = np.asarray(candidate_raw, dtype=float)
    if candidates.ndim != 2:
        raise ValueError("candidate_raw must be a 2D matrix")
    if candidates.shape[1] != seed.shape[0]:
        raise ValueError("seed_raw and candidate_raw must have the same genre dimension")

    seed_active = seed > 0
    candidate_active = candidates > 0
    missing = np.count_nonzero(candidate_active, axis=1) == 0
    if not bool(np.any(seed_active)):
        zeros = np.zeros(candidates.shape[0], dtype=float)
        return GenreCompatibilityResult(
            compatible_mass=zeros,
            conflict_mass=zeros,
            neutral_mass=zeros,
            confidence=np.ones(candidates.shape[0], dtype=float),
            penalty=zeros,
            missing_or_sparse=np.ones(candidates.shape[0], dtype=bool),
        )

    affinity = _affinity_matrix(genre_vocab, genre_affinity)
    max_to_seed = np.max(affinity[:, seed_active], axis=1)
    weights = _idf_weights(candidates)

    compatible_tag = max_to_seed >= float(compatible_threshold)
    conflict_tag = max_to_seed <= float(conflict_threshold)
    neutral_tag = ~(compatible_tag | conflict_tag)

    weighted_active = candidate_active.astype(float) * weights.reshape(1, -1)
    compatible_mass = np.sum(weighted_active[:, compatible_tag], axis=1)
    conflict_mass = np.sum(weighted_active[:, conflict_tag], axis=1)
    neutral_mass = np.sum(weighted_active[:, neutral_tag], axis=1)

    denom = compatible_mass + conflict_mass
    valid = denom > 1e-12
    confidence = np.ones(candidates.shape[0], dtype=float)
    np.divide(
        compatible_mass,
        denom,
        out=confidence,
        where=valid,
    )
    conflict_ratio = np.zeros(candidates.shape[0], dtype=float)
    np.divide(
        conflict_mass,
        denom,
        out=conflict_ratio,
        where=valid,
    )
    penalty = float(max(0.0, penalty_strength)) * conflict_ratio

    return GenreCompatibilityResult(
        compatible_mass=compatible_mass,
        conflict_mass=conflict_mass,
        neutral_mass=neutral_mass,
        confidence=np.clip(confidence, 0.0, 1.0),
        penalty=np.clip(penalty, 0.0, 1.0),
        missing_or_sparse=missing,
    )
