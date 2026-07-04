from __future__ import annotations

import re
import unicodedata

from src.playlist.identity_keys import normalize_primary_artist_key
from src.string_utils import normalize_artist_key


def normalize_release_name(value: str | None) -> str:
    """Normalize album/release names for stable cache keys while preserving non-Latin text."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(char)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def normalize_release_artist(value: str | None) -> str:
    """Normalize artist names using the playlist engine's identity rules."""
    normalized = normalize_primary_artist_key(value or "")
    return normalize_artist_key(normalized) or normalize_artist_key(value or "")


def _release_name_or_symbol_fallback(album: str | None) -> str:
    """Release-name component for a key, resilient to symbol-only titles.

    ``normalize_release_name`` maps punctuation/symbol categories to spaces, so
    an all-symbol title (Beak> ">>>" vs ">>>>", Sigur Rós "( )") collapses to
    the empty string — distinct releases would then share one release_key and
    publish would merge their genres onto a single album. When the normalized
    name is empty but the raw title has content, fall back to an NFKC-casefolded
    raw title (symbols preserved) so the releases stay distinct and stable.
    """
    name = normalize_release_name(album)
    if name or not album:
        return name
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(album)).casefold().strip())


def make_release_key(artist: str | None, album: str | None) -> str:
    return f"{normalize_release_artist(artist)}::{_release_name_or_symbol_fallback(album)}"
