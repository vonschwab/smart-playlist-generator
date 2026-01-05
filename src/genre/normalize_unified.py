"""
Unified Genre Normalization Module
====================================
Consolidates genre_normalization.py and genre/normalize.py into a single,
comprehensive implementation.

Features from both modules:
- Language translation (French, German, Dutch, Romanian)
- Diacritic removal (électronique → electronic)
- Synonym mapping (150+ mappings)
- Phrase pattern matching
- Multi-token splitting
- Meta-tag filtering
- CSV filter set loading
- Broad/garbage/meta filtering

This is the canonical implementation going forward.
Old modules will delegate to this via deprecation wrappers.
"""

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations and Data Classes
# =============================================================================

class GenreAction(Enum):
    """Action taken during normalization."""
    KEEP = "KEEP"           # No change
    MAP = "MAP"             # Single token changed
    SPLIT_MAP = "SPLIT+MAP" # Multiple tokens produced
    DROP = "DROP"           # Empty/placeholder


@dataclass
class NormalizationResult:
    """Result of genre normalization."""
    original: str
    tokens: List[str]
    action: GenreAction


# =============================================================================
# Phrase and Synonym Mappings
# =============================================================================

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

    # Additional from genre_normalization.py
    "pop, alternatif et indé, rock": ["indie rock", "pop"],
    "pop, rock, alternatif et indé": ["pop", "rock", "indie"],
    "alternative en indie": ["indie"],
    # "electro" removed - single-token synonym, not a phrase
    "indie / alternative": ["indie"],
    "alternative / indie": ["indie"],
    "alternative / indie rock": ["indie rock"],
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
    # "électronique" removed - handled in PHRASE_MAP as language translation

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
    "drum & bass": "drum and bass",

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

    # Romanian
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
    "nyc", "london",
    # Format/release type
    "reissue", "compilation", "bootleg", "demo", "cover", "covers",
    "remix", "remixes", "live", "soundtrack", "ost",
    # Vague descriptors
    "underground", "mainstream", "classic", "modern", "contemporary",
    "male vocalists", "female vocalists", "male vocalist", "female vocalist",
    "beautiful", "melancholy", "sad", "happy", "love", "mellow",
}


# =============================================================================
# Core Normalization Functions
# =============================================================================

def remove_diacritics(text: str) -> str:
    """
    Remove diacritics/accents from text.
    e.g., "électronique" -> "electronique"

    Args:
        text: Input text with potential diacritics

    Returns:
        Text with diacritics removed
    """
    if not text:
        return ""
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


def _apply_phrase_map(text: str, apply_translations: bool = True) -> Tuple[str, bool]:
    """
    Apply phrase map for known multi-token patterns.
    Uses exact string matching to avoid false substring matches.
    Returns (processed_text, was_mapped).
    """
    if not apply_translations:
        return text, False

    for phrase, replacements in PHRASE_MAP.items():
        phrase_normalized = remove_diacritics(phrase.lower())
        # Use exact match to avoid matching "electro" in "electronic"
        if text == phrase_normalized:
            if replacements:
                replacement = "; ".join(replacements)
                return replacement, True
            else:
                # Remove the phrase entirely
                return "", True
    return text, False


def _split_on_delimiters(text: str) -> List[str]:
    """
    Split on common delimiters: semicolon, comma, slash, pipe, ampersand.
    Returns list of raw token strings.
    """
    # Replace delimiters with semicolons for uniform splitting
    text = re.sub(r'\s*[|]\s*', ';', text)  # Pipe
    text = re.sub(r'\s*/\s*', ';', text)    # Forward slash
    text = re.sub(r'\s*,\s*', ';', text)    # Comma

    # Handle ampersand: only split if surrounded by spaces
    # This preserves "r&b" but splits "rock & roll"
    text = re.sub(r'\s+&\s+', ';', text)

    # Also handle backslash (from genre_normalization.py)
    text = re.sub(r'[\\/]', ';', text)

    # Split on semicolons
    pieces = [p.strip() for p in text.split(';')]
    return [p for p in pieces if p]


def _normalize_single_token(token: str, apply_synonyms: bool = True) -> Optional[str]:
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

    # Apply synonym mapping (if enabled)
    if apply_synonyms and token in SYNONYM_MAP:
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
    Normalize a single raw genre string to a canonical token.
    Does NOT split - use normalize_and_split_genre for multi-token strings.

    Args:
        raw: Raw genre string
        apply_translations: Whether to apply language translations (phrase map)
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Normalized genre string or None if empty/dropped
    """
    if not raw:
        return None

    processed = _preprocess_raw(raw)
    processed, _ = _apply_phrase_map(processed, apply_translations=apply_translations)

    # If phrase map produced multiple tokens, just take the first one
    # (This function is for single-token normalization)
    tokens = _split_on_delimiters(processed)
    if not tokens:
        return None

    return _normalize_single_token(tokens[0], apply_synonyms=apply_synonyms)


def normalize_and_split_genre(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> List[str]:
    """
    Normalize a raw genre string and split into atomic canonical tokens.

    This is the main entry point for genre normalization.

    Rules:
    - Lowercase + trim
    - Remove diacritics
    - Apply phrase map for known patterns
    - Split on delimiters (semicolon, comma, slash, pipe, ampersand with spaces)
    - Normalize each token via synonym map
    - Filter out meta-tags and drop tokens
    - Return deduplicated list preserving order

    Args:
        raw: Raw genre string (may contain multiple genres)
        apply_translations: Whether to apply phrase map translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        List of normalized canonical tokens (may be empty)
    """
    if not raw:
        return []

    processed = _preprocess_raw(raw)

    # Check if entire string is a drop token
    if processed in DROP_TOKENS:
        return []

    # Apply phrase map
    processed, _ = _apply_phrase_map(processed, apply_translations=apply_translations)

    # Split on delimiters
    pieces = _split_on_delimiters(processed)

    # Normalize each piece
    tokens = []
    seen = set()
    for piece in pieces:
        # Recursively check for nested phrase patterns
        sub_processed, was_mapped = _apply_phrase_map(piece, apply_translations=apply_translations)
        if was_mapped:
            sub_pieces = _split_on_delimiters(sub_processed)
            for sub_piece in sub_pieces:
                norm = _normalize_single_token(sub_piece, apply_synonyms=apply_synonyms)
                if norm and norm not in seen:
                    tokens.append(norm)
                    seen.add(norm)
        else:
            # No mapping, normalize directly
            norm = _normalize_single_token(piece, apply_synonyms=apply_synonyms)
            if norm and norm not in seen:
                tokens.append(norm)
                seen.add(norm)

    return tokens


def split_and_normalize(raw: str, apply_translations: bool = True, apply_synonyms: bool = True) -> Set[str]:
    """
    Normalize and split a genre string, returning a set of tokens.

    Compatibility alias for normalize_and_split_genre that returns a set.

    Args:
        raw: Raw genre string
        apply_translations: Whether to apply phrase map translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Set of normalized genre tokens
    """
    return set(normalize_and_split_genre(raw, apply_translations=apply_translations, apply_synonyms=apply_synonyms))


# =============================================================================
# Filtering and Batch Processing
# =============================================================================

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
    - For each raw genre, normalize_and_split_genre(raw) to get normalized tokens
    - Union all tokens
    - Drop tokens in broad_set, garbage_set, meta_set if provided
    - If canonical_set is provided, keep only tokens present in canonical_set

    Args:
        raw_genres: Iterable of raw genre strings
        broad_set: Overly-broad genres to filter out (e.g., "rock", "pop")
        garbage_set: Garbage/invalid genres to filter out
        meta_set: Meta tags to filter out (e.g., "seen live", "favorites")
        canonical_set: If provided, only keep genres in this set (whitelist)
        apply_translations: Whether to apply phrase map translations
        apply_synonyms: Whether to apply synonym mappings

    Returns:
        Set of normalized, filtered genre tokens
    """
    tokens: Set[str] = set()
    for g in raw_genres:
        tokens.update(normalize_and_split_genre(g, apply_translations=apply_translations, apply_synonyms=apply_synonyms))

    def _drop(source: Set[str], drops: Optional[Set[str]]) -> Set[str]:
        return source if not drops else {t for t in source if t not in drops}

    tokens = _drop(tokens, broad_set)
    tokens = _drop(tokens, garbage_set)
    tokens = _drop(tokens, meta_set)

    if canonical_set is not None:
        tokens = {t for t in tokens if t in canonical_set}

    return tokens


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
    broad_set = META_TAGS if filter_broad else None

    return normalize_and_filter_genres(
        genres,
        broad_set=broad_set,
        apply_translations=True,
        apply_synonyms=True
    )


# =============================================================================
# CSV Loading and Filter Sets
# =============================================================================

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


# =============================================================================
# Classification and Diagnostics
# =============================================================================

def classify_normalization(raw: str, normalized: List[str]) -> GenreAction:
    """
    Classify what action was taken during normalization.

    Args:
        raw: Original raw genre string
        normalized: List of normalized tokens

    Returns:
        GenreAction indicating the transformation type
    """
    if not raw or raw.strip().lower() in DROP_TOKENS:
        return GenreAction.DROP

    if not normalized:
        return GenreAction.DROP

    # Check if it's a simple keep (single token, no change)
    raw_lower = raw.strip().lower()
    if len(normalized) == 1:
        if normalized[0] == raw_lower or normalized[0] == remove_diacritics(raw_lower):
            return GenreAction.KEEP
        else:
            return GenreAction.MAP
    else:
        return GenreAction.SPLIT_MAP


def detect_normalization_flags(raw: str) -> dict:
    """
    Detect characteristics of the raw genre string.

    Returns dict with boolean flags:
    - has_non_ascii: Contains non-ASCII characters
    - has_delimiters: Contains splitting delimiters
    - has_diacritics: Contains accented characters
    - is_multi_token: Would split into multiple tokens
    """
    raw_lower = raw.lower() if raw else ""

    has_non_ascii = any(ord(c) > 127 for c in raw_lower)
    has_delimiters = bool(re.search(r'[;,/|]|\s&\s', raw_lower))
    has_diacritics = raw_lower != remove_diacritics(raw_lower)

    tokens = normalize_and_split_genre(raw)
    is_multi_token = len(tokens) > 1

    return {
        "has_non_ascii": has_non_ascii,
        "has_delimiters": has_delimiters,
        "has_diacritics": has_diacritics,
        "is_multi_token": is_multi_token,
    }
