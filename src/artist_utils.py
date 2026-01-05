"""
Artist-related normalization helpers shared across modules.

DEPRECATED: This module is deprecated and will be removed in July 2026.
Use src.string_utils.normalize_artist_name() instead.

These functions now delegate to the canonical implementation in string_utils.py
via deprecation wrappers. All functions maintain backward compatibility but will
issue deprecation warnings.

Migration:
    # Old (deprecated):
    from src.artist_utils import extract_primary_artist

    # New (recommended):
    from src.string_utils import normalize_artist_name
    # Use: normalize_artist_name(artist, strip_ensemble=True, strip_collaborations=True)
"""
import re
import warnings
from typing import List

from .string_utils import normalize_text, normalize_artist_name

# Issue deprecation warning on module import
warnings.warn(
    "artist_utils module is deprecated and will be removed in July 2026. "
    "Use src.string_utils.normalize_artist_name() instead.",
    DeprecationWarning,
    stacklevel=2
)


def extract_primary_artist(artist: str, lowercase: bool = True) -> str:
    """
    DEPRECATED: Use src.string_utils.normalize_artist_name() instead.

    This function delegates to the canonical implementation.
    Will be removed in July 2026.

    Migration:
        # Old:
        extract_primary_artist("Bill Evans Trio")

        # New:
        normalize_artist_name("Bill Evans Trio", strip_ensemble=True, strip_collaborations=True)
    """
    return normalize_artist_name(
        artist,
        strip_ensemble=True,
        strip_collaborations=True,
        lowercase=lowercase,
        normalize_unicode=False,  # extract_primary_artist didn't do Unicode normalization
    )


def parse_collaboration(artist: str) -> List[str]:
    """
    Parse collaboration string into constituent artists.
    Handles jazz ensembles and band name patterns intelligently.

    Returns [artist] unchanged if it's a band name, ensemble, or solo artist.
    Returns list of constituent artists for real collaborations.

    Examples:
        "Echo & The Bunnymen" → ["Echo & The Bunnymen"]  (band name)
        "The Horace Silver Quintet & Trio" → ["The Horace Silver Quintet & Trio"]  (ensemble)
        "Pink Siifu & Fly Anakin" → ["Pink Siifu", "Fly Anakin"]  (real collab)
        "John Coltrane feat. Cannonball Adderly" → ["John Coltrane", "Cannonball Adderly"]

    Args:
        artist: Artist name string, possibly containing collaboration markers

    Returns:
        List of constituent artist names. Single-element list if solo artist.
    """

    if not artist:
        return [""]

    # Step 1: Detect jazz ensemble suffixes with "&" (NOT a collaboration)
    # "The Horace Silver Quintet & Trio" describes ensemble composition, not collaboration
    if re.search(
        r"&\s+(?:Trio|Quartet|Quintet|Sextet|Septet|Octet|Ensemble)\s*$",
        artist,
        flags=re.IGNORECASE
    ):
        return [artist]  # Preserve as-is

    # Step 2: Preserve band names with "& The", "& His", "& Her", "& Their"
    # "Echo & The Bunnymen", "Sly & The Family Stone", "Sun Ra & His Arkestra"
    if re.search(
        r"(?:^The\s+|\s+(?:&|and)\s+(?:The|His|Her|Their)\s+)",
        artist,
        flags=re.IGNORECASE
    ):
        return [artist]  # Don't split band names

    # Step 3: Split on collaboration delimiters
    delimiters = r"\s*(?:featuring|feat\.|ft\.|with|vs\.|versus|\bx\b|&|,|;|\sand\s)\s*"
    parts = re.split(delimiters, artist, flags=re.IGNORECASE)

    # Step 4: Clean and deduplicate
    constituents = [p.strip() for p in parts if p and p.strip()]
    constituents = list(dict.fromkeys(constituents))  # Preserve order

    # Step 5: Return as-is if solo, or list of constituents if collaboration
    return constituents if len(constituents) > 1 else [artist]


# Backward-compatible alias
def split_collaborators(artist: str) -> List[str]:
    """Alias for parse_collaboration()."""
    return parse_collaboration(artist)


def get_artist_variations(artist: str) -> List[str]:
    """Generate artist name variations for better matching (unchanged logic)."""
    variations: List[str] = []

    if not artist:
        return variations

    lower_artist = artist.lower()

    # Handle "The Band" vs "Band, The"
    if lower_artist.startswith("the "):
        variations.append(artist[4:] + ", The")
    elif lower_artist.endswith(", the"):
        variations.append("The " + artist[:-5])

    # Handle "&" vs "and"
    if " & " in artist:
        variations.append(artist.replace(" & ", " and "))
    if " and " in lower_artist:
        variations.append(artist.replace(" and ", " & ").replace(" And ", " & "))

    return variations
