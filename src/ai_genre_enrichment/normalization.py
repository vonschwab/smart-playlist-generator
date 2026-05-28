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


def make_release_key(artist: str | None, album: str | None) -> str:
    return f"{normalize_release_artist(artist)}::{normalize_release_name(album)}"
