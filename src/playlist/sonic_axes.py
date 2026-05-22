"""Helpers for slicing combined sonic PCA vectors into perceptual axes."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def extract_axis_vectors(
    X_sonic: np.ndarray,
    *,
    tower_pca_dims: Tuple[int, int, int],
) -> Dict[str, np.ndarray]:
    """Return rhythm, timbre, harmony, and color views into ``X_sonic``."""
    X = np.asarray(X_sonic)
    if X.ndim != 2:
        raise ValueError(f"X_sonic must be 2D, got shape {X.shape}")

    r_dim, t_dim, h_dim = (int(v) for v in tower_pca_dims)
    expected = r_dim + t_dim + h_dim
    if X.shape[1] != expected:
        raise ValueError(
            f"X_sonic has {X.shape[1]} dims, but sum of tower_pca_dims is "
            f"{expected} (rhythm={r_dim}, timbre={t_dim}, harmony={h_dim})"
        )

    rhythm = X[:, :r_dim]
    timbre = X[:, r_dim : r_dim + t_dim]
    harmony = X[:, r_dim + t_dim : expected]
    return {
        "rhythm": rhythm,
        "timbre": timbre,
        "harmony": harmony,
        "color": X[:, r_dim:expected],
    }


def axis_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of ``a`` and rows of ``b``."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim == 1:
        a_arr = a_arr.reshape(1, -1)
    if b_arr.ndim == 1:
        b_arr = b_arr.reshape(1, -1)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or a_arr.shape[1] != b_arr.shape[1]:
        raise ValueError(f"axis vectors must be compatible 2D arrays, got {a_arr.shape} and {b_arr.shape}")

    a_norm = a_arr / (np.linalg.norm(a_arr, axis=1, keepdims=True) + 1e-12)
    b_norm = b_arr / (np.linalg.norm(b_arr, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def interpolate_axis_vector(R_a: np.ndarray, R_b: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate between two axis vectors."""
    return (1.0 - float(t)) * np.asarray(R_a, dtype=float) + float(t) * np.asarray(R_b, dtype=float)
