"""
Deprecated shim for the legacy GenreSimilarity implementation.

Prefer importing GenreSimilarityV2 directly. This shim preserves backwards
compatibility while emitting a deprecation warning.
"""
import logging
import warnings

from .genre_similarity_v2 import GenreSimilarityV2

logger = logging.getLogger(__name__)


class GenreSimilarity(GenreSimilarityV2):
    """Deprecated: use GenreSimilarityV2 instead."""

    def __init__(self, similarity_file: str = "data/genre_similarity.yaml"):
        warnings.warn(
            "genre_similarity.py is deprecated; use GenreSimilarityV2 instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning("GenreSimilarity (legacy) was instantiated; prefer GenreSimilarityV2")
        super().__init__(similarity_file)
