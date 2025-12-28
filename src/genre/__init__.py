"""
Genre Taxonomy v1 - Normalization and Similarity
=================================================
Comprehensive genre processing including:
- Deterministic normalization and splitting of raw genre strings
- Structured token-role similarity (base/modifier/scene)
- SQLite-backed canonical mapping tables
"""

from .normalize import (
    normalize_and_split_genre,
    normalize_genre_token,
    GenreAction,
)
from .similarity import (
    pairwise_genre_similarity,
    GenreSimilarityResult,
)
from .vocabulary import (
    BASE_GENRES,
    MODIFIERS,
    SCENE_TAGS,
    WEAK_ADJECTIVES,
    parse_genre_structure,
)

__all__ = [
    # Normalization
    'normalize_and_split_genre',
    'normalize_genre_token',
    'GenreAction',
    # Similarity
    'pairwise_genre_similarity',
    'GenreSimilarityResult',
    # Vocabulary
    'BASE_GENRES',
    'MODIFIERS',
    'SCENE_TAGS',
    'WEAK_ADJECTIVES',
    'parse_genre_structure',
]
