"""
Genre Normalization Module - Taxonomy v1
=========================================
Deterministic normalization and splitting of raw genre strings.

Rules:
- Lowercase + trim
- Remove diacritics (e.g., "électronique" -> "electronique" -> "electronic")
- Standardize separators: comma, slash, ampersand, semicolon, pipe
- Split multi-genre strings into atomic tokens
- Normalize common synonyms/abbreviations
- Keep tokens "human canonical" (no over-stemming)
- Drop empty/placeholder markers like "__EMPTY__"
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple


class GenreAction(Enum):
    """Action taken during normalization."""
    KEEP = "KEEP"           # No change
    MAP = "MAP"             # Single token changed
    SPLIT_MAP = "SPLIT+MAP" # Multiple tokens produced
    DROP = "DROP"           # Empty/placeholder


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


def normalize_genre_token(raw: str) -> Optional[str]:
    """
    Normalize a single raw genre string to a canonical token.
    Does NOT split - use normalize_and_split_genre for multi-token strings.

    Returns None if the token should be dropped.
    """
    if not raw:
        return None

    processed = _preprocess_raw(raw)
    processed, _ = _apply_phrase_map(processed)

    # If phrase map produced multiple tokens, just take the first one
    # (This function is for single-token normalization)
    tokens = _split_on_delimiters(processed)
    if not tokens:
        return None

    return _normalize_single_token(tokens[0])


def normalize_and_split_genre(raw: str) -> List[str]:
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
    processed, _ = _apply_phrase_map(processed)

    # Split on delimiters
    pieces = _split_on_delimiters(processed)

    # Normalize each piece
    tokens = []
    seen = set()
    for piece in pieces:
        # Recursively check for nested phrase patterns
        sub_processed, was_mapped = _apply_phrase_map(piece)
        if was_mapped:
            sub_pieces = _split_on_delimiters(sub_processed)
            for sub_piece in sub_pieces:
                norm = _normalize_single_token(sub_piece)
                if norm and norm not in seen:
                    tokens.append(norm)
                    seen.add(norm)
        else:
            norm = _normalize_single_token(piece)
            if norm and norm not in seen:
                tokens.append(norm)
                seen.add(norm)

    return tokens


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


# Compatibility with existing code
def normalize_genre_list(genres, filter_broad: bool = True) -> Set[str]:
    """
    Normalize a list of genre strings.

    This is a compatibility wrapper for the existing genre_normalization.py interface.
    """
    result = set()
    for g in genres:
        tokens = normalize_and_split_genre(g)
        result.update(tokens)
    return result


def split_and_normalize(raw: str) -> Set[str]:
    """
    Compatibility alias for normalize_and_split_genre.
    Returns a set instead of list.
    """
    return set(normalize_and_split_genre(raw))
