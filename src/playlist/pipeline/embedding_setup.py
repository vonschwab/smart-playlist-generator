"""Embedding setup phase — hybrid embedding build.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

Uses the bundle's sonic matrix (pre-scaled tower artifact or raw), runs a
flatness diagnostic, applies optional broad-genre masking, and constructs
the hybrid (sonic+genre) embedding the candidate-pool builder expects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.config import DSPipelineConfig
from src.similarity.hybrid import build_hybrid_embedding

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSetup:
    """Bundle of artifacts produced by ``setup_embedding``.

    All consumers downstream of embedding setup take their inputs from
    here; previously these were ten implicit local variables threaded
    through ``generate_playlist_ds``.
    """
    embedding_model: Any  # src.similarity.hybrid.HybridEmbedding
    X_sonic_for_embed: np.ndarray
    X_genre_raw: Optional[np.ndarray]
    X_genre_smoothed: Optional[np.ndarray]
    genre_vocab: List[str]
    broad_filters: tuple
    seed_indices_for_floor: List[int]
    variant_stats: Dict[str, Any]
    # min_genre_similarity may be reset to None in sonic_only mode; the
    # orchestrator reads it back out for downstream gating decisions.
    min_genre_similarity: Optional[float]
    effective_w_sonic: float
    effective_w_genre: float


def setup_embedding(
    bundle: ArtifactBundle,
    seed_track_id: str,
    seed_idx: int,
    *,
    anchor_seed_ids: List[str],
    mode: str,
    cfg: DSPipelineConfig,
    sonic_weight: Optional[float],
    genre_weight: Optional[float],
    min_genre_similarity: Optional[float],
    random_seed: int,
) -> EmbeddingSetup:
    """Build hybrid embedding + apply genre mask.

    Mirrors the legacy in-line phase exactly, including the flatness-warning
    diagnostic, the sonic_only mode override that disables genre gating,
    and the three-way (both / sonic-only / genre-only) weight resolution.
    """
    variant_stats: Dict[str, Any] = {}
    X_sonic_for_embed = bundle.X_sonic
    pre_scaled_sonic = False
    if bundle.X_sonic is not None:
        if getattr(bundle, "sonic_pre_scaled", False):
            # The artifact is the source of truth for its own preprocessing
            # (e.g. tower_weighted baked at build time). Use it directly; do NOT
            # re-apply a variant transform — that path no-ops via a dim-mismatch
            # fallback AND mislabels the space as un-scaled, causing a spurious
            # StandardScaler before the hybrid PCA. See
            # docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md.
            X_sonic_for_embed = bundle.X_sonic
            variant_stats = {
                "variant": getattr(bundle, "sonic_variant", None),
                "pre_scaled": True,
                "dim": int(bundle.X_sonic.shape[1]),
            }
        else:
            # No baked-in variant transform: the sonic space is used as-is
            # (raw passthrough — matches the pre-SP-B tower_pca fallback
            # behavior under a dimension mismatch, now made explicit).
            X_sonic_for_embed = bundle.X_sonic
            variant_stats = {"variant": getattr(bundle, "sonic_variant", None), "fallback": False}
    else:
        raise ValueError("Artifact missing X_sonic matrix.")

    # Warn if the transformed space is too flat (indicates bad artifact/variant).
    try:
        sims = np.dot(
            X_sonic_for_embed / (np.linalg.norm(X_sonic_for_embed, axis=1, keepdims=True) + 1e-12),
            (
                X_sonic_for_embed[seed_idx]
                / (np.linalg.norm(X_sonic_for_embed[seed_idx]) + 1e-12)
            ).T,
        )
        p10 = float(np.percentile(sims, 10))
        p90 = float(np.percentile(sims, 90))
        if (p90 - p10) < 0.1:
            logger.warning(
                "Sonic space appears flat for seed %s (p90-p10=%.3f). Check artifact/variant (variant=%s).",
                seed_track_id,
                p90 - p10,
                variant_stats.get("variant"),
            )
    except Exception:
        logger.debug(
            "Skipped flatness heuristic; unable to compute percentile diagnostics.",
            exc_info=True,
        )
    pre_scaled_sonic = bool(variant_stats.get("pre_scaled", False))

    # Resolve all seed indices for admission gating (max over seeds).
    seed_indices_for_floor: List[int] = [seed_idx]
    for sid in anchor_seed_ids:
        idx = bundle.track_id_to_index.get(str(sid))
        if idx is not None and idx not in seed_indices_for_floor:
            seed_indices_for_floor.append(idx)
        elif idx is None:
            logger.debug("Anchor seed %s not found in bundle for sonic floor", sid)
    if len(seed_indices_for_floor) > 1:
        logger.info(
            "Sonic admission uses %d seeds (max similarity across seeds)",
            len(seed_indices_for_floor),
        )

    # Compute effective hybrid embedding weights.
    # Default to balanced approach: 0.6 sonic / 0.4 genre.
    effective_w_sonic = 0.6
    effective_w_genre = 0.4
    if mode == "sonic_only":
        min_genre_similarity = None
        effective_w_sonic = 1.0
        effective_w_genre = 0.0
        logger.info("Sonic-only mode: disabling genre gate and setting genre weight to 0.")

    if sonic_weight is not None and genre_weight is not None:
        # Both provided: use them directly (assume they sum to 1.0 or normalize).
        total = sonic_weight + genre_weight
        if total > 0:
            effective_w_sonic = sonic_weight / total
            effective_w_genre = genre_weight / total
        logger.info(
            "Applying hybrid embedding weights: w_sonic=%.3f, w_genre=%.3f (normalized from %.3f, %.3f)",
            effective_w_sonic, effective_w_genre, sonic_weight, genre_weight,
        )
    elif sonic_weight is not None:
        # Only sonic weight provided: genre = 1 - sonic.
        effective_w_sonic = sonic_weight
        effective_w_genre = 1.0 - sonic_weight
        logger.info(
            "Applying hybrid embedding weight: w_sonic=%.3f (w_genre=%.3f inferred)",
            effective_w_sonic, effective_w_genre,
        )
    elif genre_weight is not None:
        # Only genre weight provided: sonic = 1 - genre.
        effective_w_genre = genre_weight
        effective_w_sonic = 1.0 - genre_weight
        logger.info(
            "Applying hybrid embedding weight: w_genre=%.3f (w_sonic=%.3f inferred)",
            effective_w_genre, effective_w_sonic,
        )

    # Apply optional broad-genre masking for genre embeddings/gating.
    raw_broad_filters = cfg.candidate.broad_filters or ()
    try:
        if isinstance(raw_broad_filters, np.ndarray):
            raw_broad_filters = raw_broad_filters.tolist()
    except Exception:
        raw_broad_filters = ()
    broad_filters = tuple(str(b).lower() for b in raw_broad_filters)

    X_genre_smoothed = bundle.X_genre_smoothed
    X_genre_raw = bundle.X_genre_raw
    raw_vocab = getattr(bundle, "genre_vocab", [])
    try:
        if hasattr(raw_vocab, "tolist"):
            genre_vocab: List[str] = list(raw_vocab.tolist())
        else:
            genre_vocab = list(raw_vocab or [])
    except Exception:
        genre_vocab = [str(g) for g in raw_vocab] if raw_vocab is not None else []
    if broad_filters and genre_vocab:
        genre_mask = np.array(
            [g.lower() not in broad_filters for g in genre_vocab], dtype=bool
        )
        if X_genre_smoothed is not None and genre_mask.shape[0] == X_genre_smoothed.shape[1]:
            X_genre_smoothed = X_genre_smoothed[:, genre_mask]
            genre_vocab = [g for g, keep in zip(genre_vocab, genre_mask) if keep]
        if X_genre_raw is not None and genre_mask.shape[0] == X_genre_raw.shape[1]:
            X_genre_raw = X_genre_raw[:, genre_mask]

    embedding_model = build_hybrid_embedding(
        X_sonic_for_embed,
        X_genre_smoothed,
        n_components_sonic=32,
        n_components_genre=32,
        w_sonic=effective_w_sonic,
        w_genre=effective_w_genre,
        random_seed=random_seed,
        pre_scaled_sonic=pre_scaled_sonic,
        # Always PCA the sonic block to 32 dims to keep the hybrid balanced with
        # the 32-dim genre block; pre_scaled_sonic only skips the redundant
        # StandardScaler inside _fit_pca when the space is already scaled.
        use_pca_sonic=True,
    )

    return EmbeddingSetup(
        embedding_model=embedding_model,
        X_sonic_for_embed=X_sonic_for_embed,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        broad_filters=broad_filters,
        seed_indices_for_floor=seed_indices_for_floor,
        variant_stats=variant_stats,
        min_genre_similarity=min_genre_similarity,
        effective_w_sonic=effective_w_sonic,
        effective_w_genre=effective_w_genre,
    )
