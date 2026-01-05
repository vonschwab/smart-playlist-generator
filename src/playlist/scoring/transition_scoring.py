"""
Transition Scoring for Pier-Bridge Playlists
=============================================

Extracted from pier_bridge_builder.py (Phase 3.1).

This module computes transition quality scores between tracks using
multi-segment similarity (end-to-start, mid-to-mid, full-to-full).

Functions extracted from pier_bridge_builder.py:
- _compute_transition_score() → compute_transition_score()
- _compute_transition_score_raw_and_transformed() → compute_transition_score_raw_and_transformed()
"""

from typing import Optional, Tuple
import numpy as np


def compute_transition_score(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    weight_end_start: float,
    weight_mid_mid: float,
    weight_full_full: float,
    center_transitions: bool,
) -> float:
    """
    Compute multi-segment transition score from track A to track B.

    Uses weighted combination of:
    - End-to-start similarity: cos(end(A), start(B))
    - Mid-to-mid similarity: cos(mid(A), mid(B))
    - Full-to-full similarity: cos(full(A), full(B))

    Args:
        idx_a: Index of track A (current track)
        idx_b: Index of track B (next track)
        X_full: Full-track similarity matrix (N, D) - L2 normalized
        X_start: Start-segment similarity matrix (N, D) - L2 normalized
        X_mid: Mid-segment similarity matrix (N, D) - L2 normalized
        X_end: End-segment similarity matrix (N, D) - L2 normalized
        weight_end_start: Weight for end-to-start similarity
        weight_mid_mid: Weight for mid-to-mid similarity
        weight_full_full: Weight for full-to-full similarity
        center_transitions: If True, rescale cosine similarity from [-1,1] to [0,1]

    Returns:
        Transition score in [0, 1] if center_transitions=True, else [-1, 1]

    Note:
        X_* matrices are expected to be row L2-normalized, so dot product equals cosine similarity.
    """
    # Full-track similarity (always available)
    sim_full = float(np.dot(X_full[idx_a], X_full[idx_b]))

    # End-start similarity (use full as fallback)
    if X_end is not None and X_start is not None:
        sim_end_start = float(np.dot(X_end[idx_a], X_start[idx_b]))
    else:
        sim_end_start = sim_full

    # Mid-mid similarity (use full as fallback)
    if X_mid is not None:
        sim_mid = float(np.dot(X_mid[idx_a], X_mid[idx_b]))
    else:
        sim_mid = sim_full

    # Optional centering: rescale from [-1, 1] to [0, 1]
    if center_transitions:
        sim_full = (sim_full + 1.0) / 2.0
        sim_end_start = (sim_end_start + 1.0) / 2.0
        sim_mid = (sim_mid + 1.0) / 2.0

    # Weighted combination
    return (
        weight_end_start * sim_end_start
        + weight_mid_mid * sim_mid
        + weight_full_full * sim_full
    )


def compute_transition_score_raw_and_transformed(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    weight_end_start: float,
    weight_mid_mid: float,
    weight_full_full: float,
    center_transitions: bool,
) -> Tuple[float, float]:
    """
    Return both raw and transformed transition scores.

    This is useful for diagnostics where you want to see the score
    before and after optional centering/rescaling.

    Args:
        (Same as compute_transition_score)

    Returns:
        Tuple of (raw_score, transformed_score) where:
        - raw_score: Weighted combination before centering
        - transformed_score: After centering (same as compute_transition_score)
    """
    # Compute raw similarities
    sim_full_raw = float(np.dot(X_full[idx_a], X_full[idx_b]))

    if X_end is not None and X_start is not None:
        sim_end_start_raw = float(np.dot(X_end[idx_a], X_start[idx_b]))
    else:
        sim_end_start_raw = sim_full_raw

    if X_mid is not None:
        sim_mid_raw = float(np.dot(X_mid[idx_a], X_mid[idx_b]))
    else:
        sim_mid_raw = sim_full_raw

    # Raw score (before centering)
    raw = (
        weight_end_start * sim_end_start_raw
        + weight_mid_mid * sim_mid_raw
        + weight_full_full * sim_full_raw
    )

    # If centering disabled, raw and transformed are identical
    if not center_transitions:
        return raw, raw

    # Apply centering transformation
    sim_full = (sim_full_raw + 1.0) / 2.0
    sim_end_start = (sim_end_start_raw + 1.0) / 2.0
    sim_mid = (sim_mid_raw + 1.0) / 2.0

    transformed = (
        weight_end_start * sim_end_start
        + weight_mid_mid * sim_mid
        + weight_full_full * sim_full
    )

    return raw, transformed
