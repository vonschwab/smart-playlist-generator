"""
Blend PMI-SVD corpus embedding with LLM prior — Phase 2.

For a genre with high corpus support the data-driven embedding dominates.
For rare genres (low support) the LLM prior dominates.

Blend weight formula (support-adaptive):

    α[i] = alpha_at_zero / (1 + support[i] / half_life)

    blended[i] = (1 - α[i]) * corpus[i] + α[i] * prior[i]

Then L2-normalize rows.

Public API:
    blend_with_prior(corpus_emb, prior_emb, support, *, alpha_at_zero, half_life)
    -> (V, dim) float32, L2-normalized
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Defaults calibrated so that:
#   support=0   → α ≈ 0.90  (almost entirely prior)
#   support=25  → α ≈ 0.45  (half-and-half)
#   support=100 → α ≈ 0.18  (mostly corpus)
#   support=500 → α ≈ 0.04  (essentially corpus)
DEFAULT_ALPHA_AT_ZERO = 0.90
DEFAULT_HALF_LIFE = 25


def _validated_alpha_schedule(
    support: np.ndarray,
    *,
    alpha_at_zero: float,
    half_life: float,
) -> np.ndarray:
    """Validate blend inputs and return the support-adaptive alpha vector."""
    if not np.isfinite(alpha_at_zero) or not 0.0 <= alpha_at_zero <= 1.0:
        raise ValueError("alpha_at_zero must be finite and in [0, 1]")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be finite and > 0")

    s = np.asarray(support, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"support must be a 1-D array, got shape {s.shape}")
    if s.size == 0:
        raise ValueError("support must not be empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("support must contain only finite values")
    if np.any(s < 0):
        raise ValueError("support must not contain negative values")

    return (alpha_at_zero / (1.0 + s / half_life)).astype(np.float32)


def blend_with_prior(
    corpus_emb: np.ndarray,
    prior_emb: np.ndarray,
    support: np.ndarray,
    *,
    alpha_at_zero: float = DEFAULT_ALPHA_AT_ZERO,
    half_life: float = DEFAULT_HALF_LIFE,
) -> np.ndarray:
    """
    Blend corpus PMI-SVD embedding with LLM prior using support-adaptive weights.

    Args:
        corpus_emb:   (V, dim) float32 — L2-normalized PMI-SVD embedding
        prior_emb:    (V, dim) float32 — L2-normalized LLM prior embedding
        support:      (V,) int — track count per genre (from build_genre_matrix)
        alpha_at_zero: prior weight when support = 0 (in [0, 1])
        half_life:    support value at which prior weight halves

    Returns:
        (V, dim) float32, L2-normalized
    """
    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)
    prior_emb = np.asarray(prior_emb, dtype=np.float32)

    if corpus_emb.shape != prior_emb.shape:
        raise ValueError(
            f"Shape mismatch: corpus {corpus_emb.shape} vs prior {prior_emb.shape}"
        )
    if corpus_emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D arrays, got shape {corpus_emb.shape}")
    if corpus_emb.shape[0] == 0:
        raise ValueError("embeddings must contain vocabulary rows")
    if not np.all(np.isfinite(corpus_emb)):
        raise ValueError("corpus_emb must contain only finite values")
    if not np.all(np.isfinite(prior_emb)):
        raise ValueError("prior_emb must contain only finite values")

    alpha = _validated_alpha_schedule(
        support,
        alpha_at_zero=alpha_at_zero,
        half_life=half_life,
    )
    if alpha.shape[0] != corpus_emb.shape[0]:
        raise ValueError(
            f"support length {alpha.shape[0]} != vocab size {corpus_emb.shape[0]}"
        )

    # α[i] = alpha_at_zero / (1 + support[i] / half_life)
    alpha = alpha[:, np.newaxis]  # (V, 1) for broadcasting

    blended = (1.0 - alpha) * corpus_emb + alpha * prior_emb  # (V, dim)

    # L2-normalize
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    blended = blended / np.maximum(norms, 1e-12)

    logger.info(
        "blend_with_prior: α mean=%.3f, min=%.3f, max=%.3f (alpha_at_zero=%.2f, half_life=%.0f)",
        float(alpha.mean()),
        float(alpha.min()),
        float(alpha.max()),
        alpha_at_zero,
        half_life,
    )
    return blended.astype(np.float32)


def alpha_schedule(
    support: np.ndarray,
    *,
    alpha_at_zero: float = DEFAULT_ALPHA_AT_ZERO,
    half_life: float = DEFAULT_HALF_LIFE,
) -> np.ndarray:
    """Return the (V,) alpha vector for diagnostic use."""
    return _validated_alpha_schedule(
        support,
        alpha_at_zero=alpha_at_zero,
        half_life=half_life,
    )
