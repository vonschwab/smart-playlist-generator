"""
Artist-related normalization helpers shared across modules.

These functions consolidate logic that previously lived in playlist_generator.py
and track_matcher.py without changing behavior.
"""
import re
from typing import List

from .string_utils import normalize_text


def extract_primary_artist(artist: str, lowercase: bool = True) -> str:
    """
    Extract the primary/first artist from collaborative credits.
    Mirrors the original playlist_generator.extract_primary_artist behavior.
    """
    if not artist:
        return ""

    # Remove anything after featuring/with/vs (clear collaboration markers)
    base_artist = re.split(
        r"\s+(?:feat\.?|ft\.?|featuring|with|vs\.?|x)\s+",
        artist,
        flags=re.IGNORECASE,
    )[0]

    # Check for jazz ensemble suffixes
    has_ensemble_suffix = re.search(
        r"\s+(?:Trio|Quartet|Quintet|Sextet|Septet|Octet)\s*$",
        base_artist,
        flags=re.IGNORECASE,
    )

    if has_ensemble_suffix:
        base_artist = re.sub(r"^The\s+", "", base_artist, flags=re.IGNORECASE)
        base_artist = re.sub(
            r"\s+(?:Trio|Quartet|Quintet|Sextet|Septet|Octet)\s*$",
            "",
            base_artist,
            flags=re.IGNORECASE,
        )
        base_artist = base_artist.strip()
        return base_artist.lower() if lowercase and base_artist else base_artist

    # Preserve likely band names (contains "& The" or "and The")
    if re.search(r"(?:^The\s+|\s+(?:&|and)\s+The\s+)", base_artist, flags=re.IGNORECASE):
        base_artist = base_artist.strip()
        return base_artist.lower() if lowercase and base_artist else base_artist

    # Otherwise, split on common separators for collaborations
    base_artist = re.split(r"\s*[&,;]\s*|\s+and\s+", base_artist, flags=re.IGNORECASE)[0]
    base_artist = normalize_text(base_artist, lowercase=lowercase)
    return base_artist


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
