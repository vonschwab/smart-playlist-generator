"""User-selected genre-tag steering: tag names -> dense target vector.

Single resolver shared by the candidate-pool lever (pipeline/core.py) and the
artist pier lever (playlist_generator -> artist_style). Soft-bias only:
callers blend or re-rank with the target; nothing here gates or excludes.
A selected tag that cannot act WARNS loudly — never a silent no-op.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def resolve_tag_steering_target(
    tags: Sequence[str],
    *,
    genre_vocab: Sequence[str],
    genre_emb: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], list[str], list[str]]:
    """Map tag names to a unit-norm mean of their vocabulary embeddings.

    Returns ``(target | None, mapped_tags, unmapped_tags)``. Matching is
    case-insensitive on the artifact ``genre_vocab``. Returns ``None`` when
    nothing maps or the dense vocabulary embedding is absent.
    """
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return None, [], []
    if genre_emb is None:
        logger.warning(
            "Tag steering requested (%s) but the artifact's dense genre sidecar "
            "has no vocabulary embedding (genre_emb) — steering disabled for this run.",
            wanted,
        )
        return None, [], list(wanted)
    vocab_lower = {str(v).strip().lower(): i for i, v in enumerate(genre_vocab)}
    mapped: list[str] = []
    unmapped: list[str] = []
    rows: list[int] = []
    for tag in wanted:
        idx = vocab_lower.get(tag.lower())
        if idx is None or idx >= int(genre_emb.shape[0]):
            unmapped.append(tag)
        else:
            mapped.append(tag)
            rows.append(int(idx))
    if unmapped:
        logger.warning(
            "Tag steering: %d/%d selected tags not in the artifact genre vocabulary: %s",
            len(unmapped), len(wanted), unmapped,
        )
    if not rows:
        logger.warning("Tag steering: no selected tags mapped — steering disabled for this run.")
        return None, mapped, unmapped
    target = np.asarray(genre_emb, dtype=np.float64)[rows].mean(axis=0)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        logger.warning("Tag steering: degenerate zero-norm target — steering disabled for this run.")
        return None, mapped, unmapped
    target = target / norm
    logger.info("Tag steering target: tags=%s (mapped %d/%d)", mapped, len(rows), len(wanted))
    return target, mapped, unmapped
