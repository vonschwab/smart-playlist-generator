"""Rhythm-axis moving target gate for pier-bridge beam search."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.playlist.sonic_axes import axis_cosine_similarity, interpolate_axis_vector


def compute_step_rhythm_target(
    R_a: np.ndarray,
    R_b: np.ndarray,
    *,
    step: int,
    segment_length: int,
) -> np.ndarray:
    """Return the interpolated rhythm target for a beam step."""
    if int(segment_length) <= 0:
        return np.asarray(R_a, dtype=float)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_axis_vector(R_a, R_b, t)


def filter_candidates_by_rhythm_target(
    *,
    candidate_indices: Sequence[int],
    rhythm_matrix: np.ndarray,
    target: np.ndarray,
    floor: float,
) -> List[int]:
    """Return candidates whose rhythm cosine to ``target`` is at least ``floor``."""
    if float(floor) <= 0.0:
        return list(candidate_indices)
    indices = list(candidate_indices)
    if not indices:
        return []

    sims = axis_cosine_similarity(rhythm_matrix[indices], np.asarray(target, dtype=float)).reshape(-1)
    return [idx for idx, sim in zip(indices, sims) if float(sim) >= float(floor)]
