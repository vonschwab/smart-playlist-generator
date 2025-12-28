"""
Genre Normalization Module
===========================
Comprehensive genre normalization including:
- Language translation (French, German, Dutch -> English)
- Punctuation and separator normalization
- Synonym mapping
- Splitting composite tags

This module ensures consistent genre tagging across the library.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Dict

logger = logging.getLogger(__name__)


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
    Normalize a single raw genre string.

    Steps:
    - Strip leading/trailing whitespace
    - Lowercase
    - Apply language translations (if enabled)
    - Collapse internal whitespace to a single space
    - Remove obvious trailing punctuation like commas/semicolons
    - Replace punctuation variants:
        * Convert separators like "-", "/", "\\" to spaces (except in known compounds like "r&b")
        * Normalize '&' to 'and' (e.g., "r&b" -> "r and b")
    - Apply synonym mappings (if enabled)
    - Collapse multiple spaces again
    - If empty after cleaning, return None

    Args:
        raw: Raw genre string
        apply_translations: Whether to apply language translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Normalized genre string or None if empty
    """
    if raw is None:
        return None

    # Trim and lowercase
    s = raw.strip().lower()
    if not s:
        return None

    # Apply language translations first (before other normalization)
    if apply_translations:
        # Direct match
        if s in LANGUAGE_TRANSLATIONS:
            s = LANGUAGE_TRANSLATIONS[s]
        else:
            # Try partial matches for composite tags
            for foreign, english in LANGUAGE_TRANSLATIONS.items():
                if foreign in s:
                    s = s.replace(foreign, english)

    # Replace common separators (\ / -) with spaces
    # But preserve compound words like "post-rock", "drum-and-bass"
    s = re.sub(r"[\\/]", " ", s)
    # Only replace hyphens that have spaces around them or at word boundaries
    s = re.sub(r"\s+-\s+", " ", s)

    # Normalize ampersand to "and"
    s = s.replace("&", " and ")

    # Remove trailing punctuation like commas/semicolons
    s = s.rstrip(";,")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Apply synonym mappings
    if apply_synonyms and s in SYNONYM_MAPPINGS:
        s = SYNONYM_MAPPINGS[s]

    return s


def split_and_normalize(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Set[str]:
    """
    Take a raw genre field that may contain multiple genres and return a set of normalized tokens.

    Splitting rules:
    - Always split on semicolons ';'
    - Also split on commas ',' and forward slashes '/' as separators
    - Normalize each piece via normalize_genre_token
    - Drop Nones/empties

    Args:
        raw: Raw genre string (may contain multiple genres)
        apply_translations: Whether to apply language translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Set of normalized genre tokens
    """
    tokens: Set[str] = set()
    if raw is None:
        return tokens

    # First pass: split on semicolons
    pieces = []
    for part in raw.split(";"):
        pieces.extend(re.split(r"[,/]", part))

    for piece in pieces:
        norm = normalize_genre_token(piece, apply_translations=apply_translations, apply_synonyms=apply_synonyms)
        if norm:
            tokens.add(norm)
    return tokens


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
    Given raw genre strings (possibly composite), return a set of normalized, filtered tokens.

    Steps:
    - For each raw genre, split_and_normalize(raw) to get normalized tokens
    - Union all tokens
    - Drop tokens in broad_set, garbage_set, meta_set if provided
    - If canonical_set is provided, keep only tokens present in canonical_set

    Args:
        raw_genres: Iterable of raw genre strings
        broad_set: Overly-broad genres to filter out (e.g., "rock", "pop")
        garbage_set: Garbage/invalid genres to filter out
        meta_set: Meta tags to filter out (e.g., "seen live", "favorites")
        canonical_set: If provided, only keep genres in this set (whitelist)
        apply_translations: Whether to apply language translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Set of normalized, filtered genre tokens
    """
    tokens: Set[str] = set()
    for g in raw_genres:
        tokens.update(split_and_normalize(g, apply_translations=apply_translations, apply_synonyms=apply_synonyms))

    def _drop(source: Set[str], drops: Optional[Set[str]]) -> Set[str]:
        return source if not drops else {t for t in source if t not in drops}

    tokens = _drop(tokens, broad_set)
    tokens = _drop(tokens, garbage_set)
    tokens = _drop(tokens, meta_set)

    if canonical_set is not None:
        tokens = {t for t in tokens if t in canonical_set}

    return tokens


def _load_genre_csv(path: Path) -> Set[str]:
    """Load genre set from CSV file with 'genre' column."""
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = normalize_genre_token(row.get("genre", ""))
            if norm:
                out.add(norm)
    return out


def load_filter_sets(
    broad_filters: Optional[Iterable[str]] = None,
    garbage_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load/normalize filter sets for genres.

    Args:
        broad_filters: Iterable from config.yaml (already in memory)
        garbage_path: CSV with column 'genre' (optional)
        meta_path: CSV with column 'genre' (optional)

    Returns:
        (broad_set, garbage_set, meta_set) as normalized sets
    """
    broad_set: Set[str] = set()
    garbage_set: Set[str] = set()
    meta_set: Set[str] = set()

    if broad_filters:
        for g in broad_filters:
            norm = normalize_genre_token(g)
            if norm:
                broad_set.add(norm)

    if garbage_path:
        garbage_set = _load_genre_csv(Path(garbage_path))
    if meta_path:
        meta_set = _load_genre_csv(Path(meta_path))

    logger.info(
        "Loaded filter sets: broad=%d, garbage=%d, meta=%d",
        len(broad_set),
        len(garbage_set),
        len(meta_set),
    )

    return broad_set, garbage_set, meta_set


def normalize_genre_list(genres: Iterable[str], filter_broad: bool = True) -> Set[str]:
    """
    Convenience function to normalize a list of genres with default settings.

    Args:
        genres: List of raw genre strings
        filter_broad: Whether to filter overly-broad tags

    Returns:
        Set of normalized genre tokens
    """
    # Default broad filters - only filter meta tags and useless descriptors
    # DO NOT filter actual music genres like rock, pop, indie, electronic!
    broad_set = {
        'seen live', 'favorites', 'favourite', 'owned', 'my music',
        '2000s', '2010s', '2020s', '1990s', '1980s', '1970s',
        'american', 'british', 'canadian', 'english', 'uk', 'usa',
        'awesome', 'cool', 'good', 'great', 'catchy',
        'liked', 'to buy', 'unknown', 'various', 'other', 'misc',
        'favorites', 'favourite', 'underground', 'classic', 'modern', 'contemporary',
    } if filter_broad else None

    return normalize_and_filter_genres(
        genres,
        broad_set=broad_set,
        apply_translations=True,
        apply_synonyms=True
    )
