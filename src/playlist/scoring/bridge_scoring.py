"""
Bridge Scoring for Pier-Bridge Playlists
=========================================

Extracted from pier_bridge_builder.py (Phase 3.1).

This module computes "bridgeability" scores - how well two seed tracks
can be connected via bridge tracks.

Functions extracted from pier_bridge_builder.py:
- _compute_bridgeability_score() → compute_bridgeability_score()
"""

from typing import Optional
import numpy as np


def compute_bridgeability_score(
    idx_a: int,
    idx_b: int,
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> float:
    """
    Compute how well two seed tracks can be bridged.

    Uses a heuristic combining:
    - Direct transition similarity: end(A) → start(B)
    - Overall coherence: full(A) · full(B)

    This score is used to order seeds in a way that makes bridge
    construction easier.

    Args:
        idx_a: Index of first seed (pier A)
        idx_b: Index of second seed (pier B)
        X_full_norm: Full-track similarity matrix (N, D) - L2 normalized
        X_start_norm: Start-segment similarity matrix (N, D) - L2 normalized
        X_end_norm: End-segment similarity matrix (N, D) - L2 normalized

    Returns:
        Bridgeability score in [-1, 1] (higher = easier to bridge)

    Formula:
        score = 0.6 * direct_transition_sim + 0.4 * full_similarity

    Note:
        Uses end→start similarity when available (better for transitions),
        falls back to full similarity otherwise.
    """
    # Direct transition similarity (end of A → start of B)
    if X_end_norm is not None and X_start_norm is not None:
        direct_sim = float(np.dot(X_end_norm[idx_a], X_start_norm[idx_b]))
    else:
        # Fallback to full similarity if segment matrices unavailable
        direct_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Full similarity (overall coherence between tracks)
    full_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Weighted combination: favor direct transition quality
    return 0.6 * direct_sim + 0.4 * full_sim
