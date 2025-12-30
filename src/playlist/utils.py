"""
Shared utility functions for playlist generation.

This module contains pure utility functions used across playlist modules.

Migrated from src/playlist_generator.py module-level functions.
"""
from typing import Dict, Any, Optional, List
import re
from src.string_utils import normalize_artist_key


def safe_get_artist(track: Dict[str, Any], lowercase: bool = True) -> str:
    """
    Safely get artist name from a track dictionary with None-safe fallback.

    Args:
        track: Track dictionary
        lowercase: Whether to convert to lowercase

    Returns:
        Artist name (empty string if None or missing)
    """
    artist = track.get('artist') or ''
    return artist.lower() if lowercase and artist else artist


def safe_get_artist_key(track: Dict[str, Any]) -> str:
    """
    Safely get a normalized artist key from a track dictionary.
    """
    key = track.get("artist_key")
    if key:
        return str(key)
    artist = track.get("artist") or ""
    key = normalize_artist_key(artist)
    if key:
        track["artist_key"] = key
    return key


def convert_seconds_to_ms(seconds: Optional[int]) -> int:
    """
    Convert seconds to milliseconds with None safety.

    Args:
        seconds: Duration in seconds (or None)

    Returns:
        Duration in milliseconds (0 if None or conversion fails)
    """
    if seconds is None:
        return 0
    try:
        return int(seconds) * 1000
    except (TypeError, ValueError):
        return 0


def sanitize_for_logging(text: str) -> str:
    """
    Sanitize text for Windows console logging by replacing unencodable characters.

    Uses cp1252 encoding (Windows console encoding) to ensure text can be
    displayed without errors.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for console logging
    """
    if not text:
        return text
    try:
        # Try to encode with cp1252 (Windows console encoding)
        text.encode('cp1252')
        return text
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Replace unencodable characters with '?'
        return text.encode('cp1252', errors='replace').decode('cp1252')


def select_canonical_track(
    *,
    artist_tracks: List[Dict[str, Any]],
    target_title: str,
) -> Optional[Dict[str, Any]]:
    """
    Select a canonical track for a given artist/title by normalizing titles.

    Prefers non-live/demo/remix variants and shorter titles when multiple
    matches exist.

    Args:
        artist_tracks: List of tracks from the same artist
        target_title: The title to match

    Returns:
        The canonical track, or None if no match found
    """
    from ..string_utils import normalize_song_title

    def _strip_punct(txt: str) -> str:
        """Remove all punctuation and normalize to lowercase."""
        return re.sub(r"[^a-z0-9\s]", "", txt.lower()).strip()

    target_norm = normalize_song_title(target_title)
    target_norm_loose = _strip_punct(target_title)
    if not target_norm:
        return None

    best = None
    best_score = (10, 10_000)  # (penalty, length)
    penalties = ("live", "demo", "remix", "version", "alt", "alternate", "edit")

    for track in artist_tracks:
        title = track.get("title") or ""
        norm = normalize_song_title(title)
        norm_loose = _strip_punct(title)
        if norm != target_norm and norm_loose != target_norm_loose:
            continue
        lower_title = title.lower()
        penalty = 0
        if "(" in title:
            penalty += 1
        for p in penalties:
            if p in lower_title:
                penalty += 1
        score = (penalty, len(title))
        if score < best_score:
            best = track
            best_score = score

    return best
