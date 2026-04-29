"""
Seed Resolver - Resolves autocomplete selections to SeedChip objects.

This module handles the conversion from autocomplete display strings
to fully-populated SeedChip objects with track_id and artist_key.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from src.string_utils import normalize_artist_key

from .widgets.seed_chips import SeedChip

logger = logging.getLogger(__name__)


def parse_track_display(display: str) -> tuple[str, str, str]:
    """
    Parse autocomplete display string into components.

    Display format: "Title - Artist (Album)" or "Title - Artist"

    Returns:
        (title, artist, album) tuple. Album may be empty string.
    """
    title = ""
    artist = ""
    album = ""

    # Check for album in parentheses at the end
    if display.endswith(")"):
        paren_start = display.rfind("(")
        if paren_start > 0:
            album = display[paren_start + 1:-1].strip()
            display = display[:paren_start].strip()

    # Split by " - " to get title and artist
    if " - " in display:
        parts = display.split(" - ", 1)
        title = parts[0].strip()
        artist = parts[1].strip() if len(parts) > 1 else ""
    else:
        title = display.strip()

    return title, artist, album


def resolve_track_from_display(
    display: str,
    db_path: str,
) -> Optional[SeedChip]:
    """
    Resolve an autocomplete display string to a SeedChip.

    Queries the database to find the track and get its track_id.

    Args:
        display: Autocomplete display string "Title - Artist (Album)"
        db_path: Path to the metadata database

    Returns:
        SeedChip with track_id and artist_key, or None if not found
    """
    if not display or not display.strip():
        return None

    title, artist, album = parse_track_display(display)

    if not title:
        return None

    if not Path(db_path).exists():
        logger.warning("Database not found: %s", db_path)
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if artist:
            if album:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE title = ? AND artist = ? AND album = ?
                    LIMIT 1
                """, (title, artist, album))
            else:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE title = ? AND artist = ?
                    LIMIT 1
                """, (title, artist))
        else:
            if album:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE title = ? AND album = ?
                    LIMIT 1
                """, (title, album))
            else:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE title = ?
                    LIMIT 1
                """, (title,))

        row = cursor.fetchone()

        if not row:
            # Try case-insensitive match
            if artist:
                if album:
                    cursor.execute("""
                        SELECT track_id, title, artist, album, artist_key
                        FROM tracks
                        WHERE LOWER(title) = LOWER(?)
                          AND LOWER(artist) = LOWER(?)
                          AND LOWER(album) = LOWER(?)
                        LIMIT 1
                    """, (title, artist, album))
                else:
                    cursor.execute("""
                        SELECT track_id, title, artist, album, artist_key
                        FROM tracks
                        WHERE LOWER(title) = LOWER(?) AND LOWER(artist) = LOWER(?)
                        LIMIT 1
                    """, (title, artist))
            else:
                if album:
                    cursor.execute("""
                        SELECT track_id, title, artist, album, artist_key
                        FROM tracks
                        WHERE LOWER(title) = LOWER(?) AND LOWER(album) = LOWER(?)
                        LIMIT 1
                    """, (title, album))
                else:
                    cursor.execute("""
                        SELECT track_id, title, artist, album, artist_key
                        FROM tracks
                        WHERE LOWER(title) = LOWER(?)
                        LIMIT 1
                    """, (title,))

            row = cursor.fetchone()

        if not row and album:
            if artist:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE LOWER(title) = LOWER(?) AND LOWER(artist) = LOWER(?)
                    LIMIT 1
                """, (title, artist))
            else:
                cursor.execute("""
                    SELECT track_id, title, artist, album, artist_key
                    FROM tracks
                    WHERE LOWER(title) = LOWER(?)
                    LIMIT 1
                """, (title,))
            row = cursor.fetchone()

        conn.close()

        if row:
            track_id, db_title, db_artist, db_album, db_artist_key = row
            artist_key = db_artist_key or normalize_artist_key(db_artist or "")

            # Reconstruct display string from DB data for consistency
            if db_artist:
                final_display = f"{db_title} - {db_artist}"
                if db_album:
                    final_display += f" ({db_album})"
            else:
                final_display = db_title

            return SeedChip(
                track_id=str(track_id),
                display=final_display,
                artist_key=artist_key,
                title=db_title,
                artist=db_artist or "",
            )

        logger.debug("Track not found in database: %s", display)
        return None

    except Exception as e:
        logger.warning("Error resolving track: %s - %s", display, e)
        return None


def resolve_tracks_batch(
    displays: list[str],
    db_path: str,
) -> list[SeedChip]:
    """
    Resolve multiple display strings to SeedChips.

    Args:
        displays: List of autocomplete display strings
        db_path: Path to the metadata database

    Returns:
        List of resolved SeedChips (may be shorter if some not found)
    """
    chips = []
    for display in displays:
        chip = resolve_track_from_display(display, db_path)
        if chip:
            chips.append(chip)
    return chips


def create_seed_chip_from_parts(
    track_id: str,
    title: str,
    artist: str,
    album: str = "",
) -> SeedChip:
    """
    Create a SeedChip from individual components.

    Useful when track data is already available from elsewhere.
    """
    artist_key = normalize_artist_key(artist) if artist else ""

    if artist:
        display = f"{title} - {artist}"
        if album:
            display += f" ({album})"
    else:
        display = title

    return SeedChip(
        track_id=track_id,
        display=display,
        artist_key=artist_key,
        title=title,
        artist=artist,
    )
