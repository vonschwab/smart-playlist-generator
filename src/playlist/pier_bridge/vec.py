"""Vector math + multi-segment transition scoring (Tier-3.1 PR-2).

Extracted from pier_bridge_builder.py. All functions take primitives + numpy
arrays — no PierBridgeConfig dependency, so this module has no back-reference
to the parent. The two transition-score functions previously took a
PierBridgeConfig instance for four fields (center_transitions,
weight_end_start, weight_mid_mid, weight_full_full); they now take those as
primitives. Thin back-compat wrappers in pier_bridge_builder.py unpack the
config before forwarding.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2 normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_transition_score(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    *,
    center_transitions: bool,
    weight_end_start: float,
    weight_mid_mid: float,
    weight_full_full: float,
) -> float:
    """
    Compute multi-segment transition score from track A to track B.

    score = w_end_start * cos(end(A), start(B))
          + w_mid_mid * cos(mid(A), mid(B))
          + w_full_full * cos(full(A), full(B))
    """
    # NOTE: X_* matrices are expected to be row L2-normalized so dot() == cosine.
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

    if center_transitions:
        # When centering is enabled, rescale cosine sims from [-1,1] to [0,1]
        sim_full = (sim_full + 1.0) / 2.0
        sim_end_start = (sim_end_start + 1.0) / 2.0
        sim_mid = (sim_mid + 1.0) / 2.0

    return (
        weight_end_start * sim_end_start
        + weight_mid_mid * sim_mid
        + weight_full_full * sim_full
    )


def _compute_transition_score_raw_and_transformed(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    *,
    center_transitions: bool,
    weight_end_start: float,
    weight_mid_mid: float,
    weight_full_full: float,
) -> tuple[float, float]:
    """
    Return (raw, transformed) transition scores where "transformed" matches
    `_compute_transition_score()`, and "raw" is before optional centering/rescale.
    """
    sim_full_raw = float(np.dot(X_full[idx_a], X_full[idx_b]))
    if X_end is not None and X_start is not None:
        sim_end_start_raw = float(np.dot(X_end[idx_a], X_start[idx_b]))
    else:
        sim_end_start_raw = sim_full_raw
    if X_mid is not None:
        sim_mid_raw = float(np.dot(X_mid[idx_a], X_mid[idx_b]))
    else:
        sim_mid_raw = sim_full_raw

    raw = (
        weight_end_start * sim_end_start_raw
        + weight_mid_mid * sim_mid_raw
        + weight_full_full * sim_full_raw
    )

    if not center_transitions:
        return raw, raw

    sim_full = (sim_full_raw + 1.0) / 2.0
    sim_end_start = (sim_end_start_raw + 1.0) / 2.0
    sim_mid = (sim_mid_raw + 1.0) / 2.0
    transformed = (
        weight_end_start * sim_end_start
        + weight_mid_mid * sim_mid
        + weight_full_full * sim_full
    )
    return raw, transformed
