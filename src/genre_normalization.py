"""
Genre Normalization Module
===========================
DEPRECATED: This module is deprecated and will be removed in July 2026.
Use src.genre.normalize_unified instead.

This module now delegates to the unified implementation via deprecation wrappers.
All functions maintain backward compatibility but will issue deprecation warnings.

Migration:
    # Old (deprecated):
    from src.genre_normalization import normalize_genre_token

    # New (recommended):
    from src.genre.normalize_unified import normalize_genre_token
"""

import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Dict

logger = logging.getLogger(__name__)

# Issue deprecation warning on module import
warnings.warn(
    "genre_normalization module is deprecated and will be removed in July 2026. "
    "Use src.genre.normalize_unified instead.",
    DeprecationWarning,
    stacklevel=2
)


# Language translation mappings
# These translate non-English genre tags to their English equivalents
LANGUAGE_TRANSLATIONS = {
    # French
    'alternatif et indé': 'indie',
    'alternatif et inde': 'indie',
    'pop, alternatif et indé, rock': 'indie rock',
    'pop, rock, alternatif et indé': 'indie rock',
    'alternative en indie': 'indie',  # Also appears in some French tags
    'électronique': 'electronic',
    'electro': 'electronic',
    'hip hop': 'hip-hop',
    'pop rock': 'pop rock',
    'rock alternatif': 'alternative rock',

    # German
    'alternativ und indie': 'indie',
    'elektronisch': 'electronic',
    'indie rock': 'indie rock',  # Sometimes appears in German tags

    # Dutch
    'alternative en indie': 'indie',
    'elektronische': 'electronic',

    # Common variations and typos
    'rnb': 'r&b',
    'rhythm and blues': 'r&b',
    'drum n bass': 'drum and bass',
    'drum & bass': 'drum and bass',
    'dnb': 'drum and bass',
}

# Synonym mappings - map similar genre names to canonical form
SYNONYM_MAPPINGS = {
    # Indie variants
    'indie / alternative': 'indie',
    'alternative / indie': 'indie',
    'indie / rock / alternative': 'indie rock',
    'alt. rock, indie rock': 'indie rock',
    'alternative / indie rock': 'indie rock',
    'alternative, indie': 'indie',
    'alternative and indie': 'indie',
    'rock; alternative; indie': 'indie rock',
    'alternative,rock,indie rock/rock pop': 'indie rock',

    # Alternative variants
    'alt rock': 'alternative rock',
    'alt. rock': 'alternative rock',
    'alternative; indie; pop; rock': 'indie pop',

    # Electronic variants
    'electro': 'electronic',
    'electronica': 'electronic',

    # Dance variants
    'edm': 'electronic dance',
    'dance music': 'dance',

    # Hip-hop variants
    'hiphop': 'hip-hop',
    'hip hop': 'hip-hop',
    'rap': 'hip-hop',

    # Rock variants
    'rock and roll': 'rock',
    'rock & roll': 'rock',
    'rock n roll': 'rock',
}


def normalize_genre_token(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Optional[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_genre_token() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import normalize_genre_token as unified_normalize
    return unified_normalize(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms)


def split_and_normalize(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.split_and_normalize() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import split_and_normalize as unified_split
    return unified_split(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms)


def normalize_and_filter_genres(
    raw_genres: Iterable[str],
    *,
    broad_set: Optional[Set[str]] = None,
    garbage_set: Optional[Set[str]] = None,
    meta_set: Optional[Set[str]] = None,
    canonical_set: Optional[Set[str]] = None,
    apply_translations: bool = True,
    apply_synonyms: bool = True,
) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_and_filter_genres() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import normalize_and_filter_genres as unified_filter
    return unified_filter(
        raw_genres,
        broad_set=broad_set,
        garbage_set=garbage_set,
        meta_set=meta_set,
        canonical_set=canonical_set,
        apply_translations=apply_translations,
        apply_synonyms=apply_synonyms,
    )


def _load_genre_csv(path: Path) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified._load_genre_csv() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import _load_genre_csv as unified_load
    return unified_load(path)


def load_filter_sets(
    broad_filters: Optional[Iterable[str]] = None,
    garbage_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    DEPRECATED: Use src.genre.normalize_unified.load_filter_sets() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import load_filter_sets as unified_load_filters
    return unified_load_filters(broad_filters=broad_filters, garbage_path=garbage_path, meta_path=meta_path)


def normalize_genre_list(genres: Iterable[str], filter_broad: bool = True) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_genre_list() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .genre.normalize_unified import normalize_genre_list as unified_normalize_list
    return unified_normalize_list(genres, filter_broad=filter_broad)
