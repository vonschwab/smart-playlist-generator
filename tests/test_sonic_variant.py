import os

import numpy as np

from src.similarity.sonic_variant import compute_sonic_variant_norm


def _cosine_stats(norm_mat: np.ndarray) -> dict:
    N = norm_mat.shape[0]
    ia, ib = np.triu_indices(N, k=1)
    sims = np.sum(norm_mat[ia] * norm_mat[ib], axis=1)
    return {
        "p10": float(np.percentile(sims, 10)),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
        "spread": float(np.percentile(sims, 90) - np.percentile(sims, 10)),
    }


def test_variant_raw_matches_manual_norm():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    norm_manual = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    norm_variant, stats = compute_sonic_variant_norm(X, "raw")
    assert np.allclose(norm_variant, norm_manual)
    assert stats["variant"] == "raw"


def test_variant_z_increases_spread_on_dominant_dim():
    rng = np.random.default_rng(0)
    # dominant dim 0, tiny noise elsewhere
    base = rng.normal(loc=5.0, scale=0.01, size=(50, 1))
    noise = rng.normal(scale=0.001, size=(50, 4))
    X = np.hstack([base, noise])
    norm_raw, _ = compute_sonic_variant_norm(X, "raw")
    norm_z, _ = compute_sonic_variant_norm(X, "z")
    raw_stats = _cosine_stats(norm_raw)
    z_stats = _cosine_stats(norm_z)
    assert raw_stats["spread"] < 0.02  # very tight because of dominant dim
    assert z_stats["spread"] > raw_stats["spread"]
    assert z_stats["p50"] < raw_stats["p50"]


def test_variant_z_clip_and_whiten_do_not_error():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    for var in ["z_clip", "whiten_pca"]:
        normed, stats = compute_sonic_variant_norm(X, var)
        assert normed.shape == X.shape
        assert stats["variant"] == var
