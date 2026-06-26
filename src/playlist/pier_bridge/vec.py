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

import math
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


def _calibrate_transition_cos(
    value: float, *, center: float, scale: float, gain: float
) -> float:
    """Calibrated logistic remap of a transition cosine into (0, 1).

    SINGLE SOURCE OF TRUTH for the centered-transition rescale (Platt-style:
    ``sigma(gain * (x - center) / scale)``). Replaces the legacy ``(x + 1) / 2``,
    which wasted its output range on the negative cosines that real edges never
    produce, compressing the realistic band [~0.14, 0.50] into [~0.57, 0.75].
    Monotonic; soft-saturating (no hard clip, no ties).

    Lives in this low-level vec module (no upward imports) so both the live
    scorer (transition_metrics.score_transition_edge) and the opt-in audit path
    (_compute_transition_score*) call ONE implementation — no divergence.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    z = gain * (v - center) / scale
    # Numerically stable logistic (avoid overflow for large |z|).
    if z >= 0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


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
    calib_center: float = 0.32,
    calib_scale: float = 0.0625,
    calib_gain: float = 1.0,
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
        # Calibrated logistic rescale (single source of truth: _calibrate_transition_cos).
        sim_full = _calibrate_transition_cos(sim_full, center=calib_center, scale=calib_scale, gain=calib_gain)
        sim_end_start = _calibrate_transition_cos(sim_end_start, center=calib_center, scale=calib_scale, gain=calib_gain)
        sim_mid = _calibrate_transition_cos(sim_mid, center=calib_center, scale=calib_scale, gain=calib_gain)

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
    calib_center: float = 0.32,
    calib_scale: float = 0.0625,
    calib_gain: float = 1.0,
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

    sim_full = _calibrate_transition_cos(sim_full_raw, center=calib_center, scale=calib_scale, gain=calib_gain)
    sim_end_start = _calibrate_transition_cos(sim_end_start_raw, center=calib_center, scale=calib_scale, gain=calib_gain)
    sim_mid = _calibrate_transition_cos(sim_mid_raw, center=calib_center, scale=calib_scale, gain=calib_gain)
    transformed = (
        weight_end_start * sim_end_start
        + weight_mid_mid * sim_mid
        + weight_full_full * sim_full
    )
    return raw, transformed
