"""
Shared string normalization utilities used across playlist generation modules.

These helpers consolidate the previously duplicated normalization logic in
playlist_generator.py and track_matcher.py without changing behavior.
"""
import re
import unicodedata
from typing import List

# Pre-compiled patterns for song title normalization (remasters/live/etc.)
_SONG_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\s*\(.*remaster.*\)",
        r"\s*\[.*remaster.*\]",
        r"\s*\(.*live.*\)",
        r"\s*\[.*live.*\]",
        r"\s*\(.*demo.*\)",
        r"\s*\[.*demo.*\]",
        r"\s*\(.*remix.*\)",
        r"\s*\[.*remix.*\]",
        r"\s*\(.*version.*\)",
        r"\s*\[.*version.*\]",
        r"\s*\(.*edit.*\)",
        r"\s*\[.*edit.*\]",
        r"\s*\(.*mix.*\)",
        r"\s*\[.*mix.*\]",
        r"\s*\(mono\)",
        r"\s*\[mono\]",
        r"\s*\(stereo\)",
        r"\s*\[stereo\]",
        r"\s*\(\d{4}\)",  # Year in parentheses like (2010)
        r"\s*\[\d{4}\]",  # Year in brackets like [2010]
    ]
]

# Abbreviation expansions used by playlist_generator.normalize_genre
_GENRE_ABBREVIATIONS = {
    "rnb": "rhythm and blues",
    "r and b": "rhythm and blues",
    "dnb": "drum and bass",
    "edm": "electronic dance music",
    "idm": "intelligent dance music",
}

# Typography normalization for artist keys
_ARTIST_TYPOGRAPHY_TRANSLATION = {
    ord("\u2018"): "'",  # left single quotation mark
    ord("\u2019"): "'",  # right single quotation mark
    ord("\u201A"): "'",  # single low-9 quotation mark
    ord("\u201B"): "'",  # single high-reversed-9 quotation mark
    ord("\u2032"): "'",  # prime
    ord("\u2035"): "'",  # reversed prime
    ord("\u201C"): '"',  # left double quotation mark
    ord("\u201D"): '"',  # right double quotation mark
    ord("\u201E"): '"',  # double low-9 quotation mark
    ord("\u201F"): '"',  # double high-reversed-9 quotation mark
    ord("\u2033"): '"',  # double prime
    ord("\u2036"): '"',  # reversed double prime
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
}


def normalize_text(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """
    Normalize text for consistent comparisons.

    Handles Unicode normalization (NFC), optional case folding, and whitespace.
    This ensures international characters (Japanese, Korean, etc.) compare correctly.

    Args:
        text: Text to normalize
        lowercase: Apply case folding (uses casefold() for better Unicode support)
        strip: Remove leading/trailing whitespace

    Returns:
        Normalized text string
    """
    if text is None:
        return ""

    # Unicode normalization (NFC = canonical composition)
    # This ensures "きゃりー" in different forms compares as equal
    text = unicodedata.normalize('NFC', text)

    if lowercase:
        # casefold() is better than lower() for international characters
        text = text.casefold()

    if strip:
        text = text.strip()

    return text


def normalize_song_title(title: str) -> str:
    """
    Normalize song title by removing common annotations (remaster, live, demo, etc.).
    Mirrors the original playlist_generator.normalize_song_title behavior.
    """
    if not title:
        return ""

    normalized = normalize_text(title)

    for pattern in _SONG_TITLE_PATTERNS:
        normalized = pattern.sub("", normalized)

    # Collapse whitespace
    return " ".join(normalized.split())


def normalize_genre(genre: str) -> str:
    """
    Aggressively normalize genre for matching (punctuation and abbreviations).
    Mirrors the original playlist_generator.normalize_genre behavior.
    """
    if not genre:
        return ""

    genre = normalize_text(genre)
    genre = genre.replace("-", " ")
    genre = genre.replace("&", "and")
    genre = genre.replace("/", " ")
    genre = genre.replace("_", " ")

    genre = " ".join(genre.split())
    return _GENRE_ABBREVIATIONS.get(genre, genre)


def normalize_match_string(value: str, is_artist: bool = False) -> str:
    """
    Normalize strings for exact/fuzzy matching.
    Mirrors the original track_matcher._normalize_string behavior.
    """
    if not value:
        return ""

    text = value.lower()

    # Remove featuring/with/versus suffixes
    text = re.sub(r"\s+(feat|ft|featuring|with|vs)[\.\s]+.*$", "", text, flags=re.IGNORECASE)

    # Remove parenthetical/bracketed content (remixes, versions, etc.)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Remove leading "the "
    text = re.sub(r"^the\s+", "", text)

    # Normalize common variations
    text = text.replace("&", "and")
    text = text.replace("+", "and")

    if is_artist:
        # Don't split if it contains "and the" (likely a band name)
        if not re.search(r"\s+and\s+the\s+", text, flags=re.IGNORECASE):
            text = re.split(r"\s*(?:and|,|;)\s+", text)[0]

    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_artist_key(name: str) -> str:
    """
    Normalize artist names to a stable comparison key.

    Steps:
    - Strip and collapse whitespace
    - Normalize typography variants (quotes/dashes)
    - Unicode NFKD + remove combining marks (diacritics)
    - Casefold
    - Replace punctuation with spaces
    """
    if not name:
        return ""

    text = str(name).strip()
    if not text:
        return ""

    text = text.translate(_ARTIST_TYPOGRAPHY_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()

    normalized = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in text
    )
    normalized = " ".join(normalized.split())
    if normalized:
        return normalized

    # Preserve punctuation-only artist names (e.g., "@", "!!!") as stable keys.
    fallback = " ".join(text.split())
    return fallback


# Backward-compatible alias
normalize_string = normalize_text
