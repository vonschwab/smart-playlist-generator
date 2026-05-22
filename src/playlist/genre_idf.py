"""Shared IDF weight computation for the genre vocabulary.

Single source of truth used by:
 - the dj_bridging waypoint scoring in src/playlist/pier_bridge_builder.py
 - the candidate-pool admission genre similarity in src/playlist/candidate_pool.py

Pure function; no I/O, no logging.
"""
from __future__ import annotations

import numpy as np


def compute_genre_idf(
    *,
    X_genre_raw: np.ndarray,
    power: float = 1.0,
    norm: str = "max1",
) -> np.ndarray:
    """Return one IDF weight per genre column.

    Args:
        X_genre_raw: (N, V) raw genre presence matrix (binary or float >0).
        power: exponent applied to raw IDF before normalization. 0 collapses to
            uniform weights. 1.0 is standard IDF.
        norm: "max1" (largest weight is 1.0), "sum1" (weights sum to 1.0), or
            "none" (raw IDF values).

    Returns:
        (V,) array of weights, higher for rare tags, lower for common tags.
    """
    presence = (np.asarray(X_genre_raw) > 0).astype(float)
    n = float(presence.shape[0])
    if presence.size == 0:
        return np.zeros(presence.shape[1], dtype=float)
    df = presence.sum(axis=0)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    if float(power) != 1.0:
        idf = idf ** float(power)

    method = str(norm).strip().lower()
    if method == "max1":
        max_val = float(np.max(idf))
        if max_val > 1e-12:
            idf = idf / max_val
    elif method == "sum1":
        total = float(np.sum(idf))
        if total > 1e-12:
            idf = idf / total
    elif method == "none":
        pass
    else:
        max_val = float(np.max(idf))
        if max_val > 1e-12:
            idf = idf / max_val
    return idf
