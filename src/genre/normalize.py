"""
Genre Normalization Module - Taxonomy v1
=========================================
DEPRECATED: This module is deprecated and will be removed in July 2026.
Use src.genre.normalize_unified instead.

This module now delegates to the unified implementation via deprecation wrappers.
All functions maintain backward compatibility but will issue deprecation warnings.

Migration:
    # Old (deprecated):
    from src.genre.normalize import normalize_and_split_genre

    # New (recommended):
    from src.genre.normalize_unified import normalize_and_split_genre
"""

import re
import unicodedata
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple

# Issue deprecation warning on module import
warnings.warn(
    "src.genre.normalize module is deprecated and will be removed in July 2026. "
    "Use src.genre.normalize_unified instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import GenreAction from unified module for backward compatibility
from .normalize_unified import GenreAction


# High-impact compound translations
# Maps complex multi-language/multi-genre phrases to canonical tokens
PHRASE_MAP = {
    # French
    "alternatif et indé": ["alternative", "indie"],
    "alternatif et inde": ["alternative", "indie"],
    "pop alternatif et indé rock": ["indie rock", "pop"],
    "pop rock alternatif et indé": ["indie rock", "pop"],
    "rock alternatif": ["alternative rock"],
    "pop indé": ["indie pop"],
    "rock indé": ["indie rock"],
    "électronique": ["electronic"],
    "musiques de noël": ["christmas"],
    "ambiance": [],  # Remove vague term
    # German
    "alternativ und indie": ["alternative", "indie"],
    "elektronisch": ["electronic"],
    # Dutch
    "alternative en indie": ["alternative", "indie"],
    "elektronische": ["electronic"],
    # R&B special cases (must be before ampersand splitting)
    "r & b": ["rnb"],
    "r&b": ["rnb"],
    "r'n'b": ["rnb"],
    "rhythm & blues": ["rnb"],
    "rhythm and blues": ["rnb"],
    # Discogs compound styles
    "folk, world, & country": ["folk", "world", "country"],
    "funk / soul": ["funk", "soul"],
    "hip hop": ["hip hop"],
    "rock & roll": ["rock"],
    "drum & bass": ["drum and bass"],
    "jazz / holiday / soundtrack": ["jazz"],  # Drop non-genre parts
    # Stage/screen (not real genres)
    "stage & screen": [],
    "stage and screen": [],
    # Common formatting variants
    "hip-hop/rap": ["hip hop"],
    "pop/rock": ["pop", "rock"],
    # Singer-songwriter (keep as one token)
    "singer/songwriter": ["singer-songwriter"],
    "singer / songwriter": ["singer-songwriter"],
}

# Synonym normalization for atomic tokens
# Applied AFTER splitting
SYNONYM_MAP = {
    # Hip-hop variants
    "hiphop": "hip hop",
    "hip-hop": "hip hop",
    "rap": "hip hop",
    # R&B variants
    "r&b": "rnb",
    "r & b": "rnb",
    "r'n'b": "rnb",
    "rhythm and blues": "rnb",
    "contemporary r&b": "rnb",
    # Electronic variants
    "electro": "electronic",
    "electronica": "electronic",
    "elektronisch": "electronic",
    "elektronische": "electronic",
    "électronique": "electronic",
    # Rock variants
    "rock and roll": "rock",
    "rock n roll": "rock",
    "rock & roll": "rock",
    "rock'n'roll": "rock",
    # Alternative variants
    "alt rock": "alternative rock",
    "alt. rock": "alternative rock",
    "alt-rock": "alternative rock",
    # Punk variants
    "punk-rock": "punk rock",
    "punk rock": "punk",
    # Drum and bass
    "dnb": "drum and bass",
    "d&b": "drum and bass",
    "drum n bass": "drum and bass",
    # Post-rock
    "post rock": "post-rock",
    # Lo-fi
    "lofi": "lo-fi",
    "lo fi": "lo-fi",
    # Synth-pop
    "synthpop": "synth-pop",
    "synth pop": "synth-pop",
    # Shoegaze
    "shoe gaze": "shoegaze",
    # Dream pop
    "dreampop": "dream pop",
    # Art rock
    "artrock": "art rock",
    # Math rock
    "mathrock": "math rock",
    # Noise rock
    "noiserock": "noise rock",
    # Chamber pop
    "chamberpop": "chamber pop",
    # Indie variants
    "indie-rock": "indie rock",
    "indie-pop": "indie pop",
    "indie-folk": "indie folk",
    "indie-electronic": "indie electronic",
    "indietronica": "indie electronic",
    # Post-punk
    "post punk": "post-punk",
    # Jazz variants
    "jazz-funk": "jazz funk",
    "jazz funk": "jazz funk",
    # Progressive
    "prog rock": "progressive rock",
    "prog-rock": "progressive rock",
    "prog": "progressive",
    # Psych variants
    "psych rock": "psychedelic rock",
    "psych-rock": "psychedelic rock",
    "psych pop": "psychedelic pop",
    "psych-pop": "psychedelic pop",
    # Neo variants
    "neo psychedelia": "neo-psychedelia",
    "neo soul": "neo-soul",
    "neo-soul": "neo-soul",
    # Dance variants
    "dance-pop": "dance pop",
    "dance music": "dance",
    "edm": "electronic dance",
    # Avant-garde
    "avantgarde": "avant-garde",
    "avant garde": "avant-garde",
    # Hard bop
    "hardbop": "hard bop",
    # Singer-songwriter
    "singer songwriter": "singer-songwriter",
    # Romanian (seen in data)
    "alternativă": "alternative",
}

# Tokens to drop (placeholders, meta-tags, overly vague)
DROP_TOKENS = {
    "__empty__",
    "empty",
    "unknown",
    "other",
    "misc",
    "various",
    "n/a",
    "none",
    "-",
    "",
}

# Meta-tags and non-genre descriptors to filter
META_TAGS = {
    "seen live",
    "favorites",
    "favourite",
    "my music",
    "owned",
    "liked",
    "to buy",
    "catchy",
    "awesome",
    "cool",
    "good",
    "great",
    # Decade tags
    "50s", "60s", "70s", "80s", "90s", "00s", "10s", "20s",
    "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
    "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020",
    # Geographic (not genres)
    "american", "british", "canadian", "english", "uk", "usa",
    "australian", "german", "french", "swedish", "japanese",
    # Format/release type
    "reissue", "compilation", "bootleg", "demo", "cover", "covers",
    "remix", "remixes", "live", "soundtrack", "ost",
    # Vague descriptors
    "underground", "mainstream", "classic", "modern", "contemporary",
}


def remove_diacritics(text: str) -> str:
    """
    Remove diacritics/accents from text.
    e.g., "électronique" -> "electronique"
    """
    # Normalize to NFD (decomposed form), then remove combining marks
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def _preprocess_raw(raw: str) -> str:
    """Initial preprocessing: lowercase, trim, remove diacritics."""
    if not raw:
        return ""
    text = raw.strip().lower()
    text = remove_diacritics(text)
    return text


def _apply_phrase_map(text: str) -> Tuple[str, bool]:
    """
    Apply phrase map for known multi-token patterns.
    Returns (processed_text, was_mapped).
    """
    for phrase, replacements in PHRASE_MAP.items():
        phrase_normalized = remove_diacritics(phrase.lower())
        if phrase_normalized in text:
            if replacements:
                replacement = "; ".join(replacements)
                text = text.replace(phrase_normalized, replacement)
            else:
                # Remove the phrase entirely
                text = text.replace(phrase_normalized, "")
            return text, True
    return text, False


def _split_on_delimiters(text: str) -> List[str]:
    """
    Split on common delimiters: semicolon, comma, slash, pipe, ampersand
    Returns list of raw token strings.
    """
    # Replace delimiters with semicolons for uniform splitting
    # Be careful with ampersand in valid tokens like "r&b"
    text = re.sub(r'\s*[|]\s*', ';', text)  # Pipe
    text = re.sub(r'\s*/\s*', ';', text)    # Forward slash
    text = re.sub(r'\s*,\s*', ';', text)    # Comma

    # Handle ampersand: only split if surrounded by spaces
    # This preserves "r&b" but splits "rock & roll"
    text = re.sub(r'\s+&\s+', ';', text)

    # Split on semicolons
    pieces = [p.strip() for p in text.split(';')]
    return [p for p in pieces if p]


def _normalize_single_token(token: str) -> Optional[str]:
    """
    Normalize a single atomic token.
    Returns None if token should be dropped.
    """
    if not token:
        return None

    token = token.strip().lower()

    # Check drop list
    if token in DROP_TOKENS:
        return None

    # Check meta-tags
    if token in META_TAGS:
        return None

    # Apply synonym mapping
    if token in SYNONYM_MAP:
        token = SYNONYM_MAP[token]

    # Clean up punctuation at edges
    token = token.strip('.,;:!?()[]{}')

    # Collapse internal whitespace
    token = re.sub(r'\s+', ' ', token).strip()

    if not token or token in DROP_TOKENS:
        return None

    return token


def normalize_genre_token(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Optional[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_genre_token() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import normalize_genre_token as unified_normalize
    return unified_normalize(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms)


def normalize_and_split_genre(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> List[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_and_split_genre() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import normalize_and_split_genre as unified_split
    return unified_split(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms)


def classify_normalization(raw: str, normalized: List[str]) -> GenreAction:
    """
    DEPRECATED: Use src.genre.normalize_unified.classify_normalization() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import classify_normalization as unified_classify
    return unified_classify(raw, normalized)


def detect_normalization_flags(raw: str) -> dict:
    """
    DEPRECATED: Use src.genre.normalize_unified.detect_normalization_flags() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import detect_normalization_flags as unified_detect
    return unified_detect(raw)


# Compatibility with existing code
def normalize_genre_list(genres, filter_broad: bool = True) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.normalize_genre_list() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import normalize_genre_list as unified_normalize_list
    return unified_normalize_list(genres, filter_broad=filter_broad)


def split_and_normalize(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Set[str]:
    """
    DEPRECATED: Use src.genre.normalize_unified.split_and_normalize() instead.

    This function delegates to the unified implementation.
    Will be removed in July 2026.
    """
    from .normalize_unified import split_and_normalize as unified_split
    return unified_split(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms)
