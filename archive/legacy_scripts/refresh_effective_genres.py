#!/usr/bin/env python3
"""
Refresh Effective Genres

Populates track_effective_genres table with effective genres for all tracks.
Applies precedence rules: file > album > artist > inherited collaboration

Usage:
    python scripts/refresh_effective_genres.py [--limit N] [--verbose]

Options:
    --limit N    Only refresh first N tracks (for testing)
    --verbose    Print debug output for each track
"""

import sqlite3
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/metadata.db")


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_track_genres(conn: sqlite3.Connection, track_id: str, source: str = None) -> List[str]:
    """Get genres for a track from track_genres table."""
    cursor = conn.cursor()
    if source:
        cursor.execute("""
            SELECT DISTINCT genre FROM track_genres
            WHERE track_id = ? AND source = ? AND genre != '__EMPTY__'
            ORDER BY genre
        """, (track_id, source))
    else:
        cursor.execute("""
            SELECT DISTINCT genre FROM track_genres
            WHERE track_id = ? AND genre != '__EMPTY__'
            ORDER BY genre
        """, (track_id,))
    return [row['genre'] for row in cursor.fetchall()]


def get_album_genres(conn: sqlite3.Connection, album_id: str) -> List[str]:
    """Get genres for an album from album_genres table."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT genre FROM album_genres
        WHERE album_id = ? AND genre != '__EMPTY__'
        ORDER BY genre
    """, (album_id,))
    return [row['genre'] for row in cursor.fetchall()]


def get_artist_genres(
    conn: sqlite3.Connection,
    artist: str,
    include_inherited: bool = False
) -> List[str]:
    """Get genres for an artist from artist_genres table."""
    cursor = conn.cursor()
    if include_inherited:
        cursor.execute("""
            SELECT DISTINCT genre FROM artist_genres
            WHERE artist = ?
              AND genre != '__EMPTY__'
              AND source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
            ORDER BY genre
        """, (artist,))
    else:
        cursor.execute("""
            SELECT DISTINCT genre FROM artist_genres
            WHERE artist = ? AND genre != '__EMPTY__' AND source = 'musicbrainz_artist'
            ORDER BY genre
        """, (artist,))
    return [row['genre'] for row in cursor.fetchall()]


def parse_collaboration(artist: str) -> List[str]:
    """Parse collaboration string into constituent artists."""
    from src.artist_utils import parse_collaboration as parse_collab
    return parse_collab(artist)


def get_constituent_genres(
    conn: sqlite3.Connection,
    artist: str,
    max_per_artist: int = 5,
    total_cap: int = 10
) -> List[Tuple[str, str]]:
    """Get genres from constituent artists (collaboration inheritance)."""
    constituents = parse_collaboration(artist)

    if len(constituents) <= 1:
        return []  # Not a collaboration

    inherited = []
    for constituent in constituents:
        genres = get_artist_genres(conn, constituent, include_inherited=False)
        # Take top max_per_artist genres from each constituent
        for genre in genres[:max_per_artist]:
            inherited.append((genre, f"musicbrainz_artist_inherited"))

    # Cap at total_cap and deduplicate
    unique_inherited = []
    seen = set()
    for genre, source in inherited:
        if genre not in seen:
            unique_inherited.append((genre, source))
            seen.add(genre)
        if len(unique_inherited) >= total_cap:
            break

    return unique_inherited


def compute_effective_genres(
    conn: sqlite3.Connection,
    track_id: str,
    artist: str,
    album_id: str,
    verbose: bool = False
) -> List[Tuple[str, str, int]]:
    """
    Compute effective genres for a track.

    Returns:
        List of (genre, source, priority) tuples
    """
    effective = []

    # Priority 1: File-embedded genres
    file_genres = get_track_genres(conn, track_id, source='file')
    for genre in file_genres:
        effective.append((genre, 'file', 1))
    if verbose and file_genres:
        logger.debug(f"  File genres: {file_genres}")

    # Priority 2: Album genres
    album_genres = get_album_genres(conn, album_id) if album_id else []
    for genre in album_genres:
        effective.append((genre, 'musicbrainz_release', 2))
    if verbose and album_genres:
        logger.debug(f"  Album genres: {album_genres}")

    # Priority 3: Artist genres
    artist_genres = get_artist_genres(conn, artist, include_inherited=False)
    for genre in artist_genres:
        effective.append((genre, 'musicbrainz_artist', 3))
    if verbose and artist_genres:
        logger.debug(f"  Artist genres: {artist_genres}")

    # Priority 4: Inherited from collaboration constituents (if no genres yet)
    if len(effective) == 0:
        inherited_genres = get_constituent_genres(conn, artist)
        for genre, source in inherited_genres:
            effective.append((genre, source, 4))
        if verbose and inherited_genres:
            logger.debug(f"  Inherited genres: {[g for g, _ in inherited_genres]}")

    # Deduplicate, preserving lowest priority (highest number)
    dedup_dict = {}
    for genre, source, priority in effective:
        key = genre.lower()
        if key not in dedup_dict or priority < dedup_dict[key][2]:
            dedup_dict[key] = (genre, source, priority)

    return list(dedup_dict.values())


def refresh_effective_genres(limit: int = None, verbose: bool = False):
    """Populate track_effective_genres table."""
    try:
        conn = get_connection()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False

    logger.info("Starting effective genres refresh...")

    # Get all tracks with sonic features
    cursor = conn.cursor()
    cursor.execute("""
        SELECT track_id, artist, album_id
        FROM tracks
        WHERE sonic_features IS NOT NULL
        ORDER BY artist, title
    """)

    if limit:
        tracks = list(cursor.fetchall())[:limit]
        logger.info(f"Processing {len(tracks)} tracks (limit={limit})")
    else:
        tracks = cursor.fetchall()
        logger.info(f"Processing {len(tracks)} tracks")

    # Clear existing data
    cursor.execute("DELETE FROM track_effective_genres")
    conn.commit()

    # Process each track
    inserted = 0
    inherited_count = 0

    for i, track in enumerate(tracks, 1):
        track_id = track['track_id']
        artist = track['artist']
        album_id = track['album_id']

        if verbose:
            logger.debug(f"\n[{i}/{len(tracks)}] {artist} - {track_id}")

        try:
            effective_genres = compute_effective_genres(
                conn, track_id, artist, album_id, verbose=verbose
            )

            # Insert into database
            for genre, source, priority in effective_genres:
                cursor.execute("""
                    INSERT INTO track_effective_genres
                    (track_id, genre, source, priority, weight)
                    VALUES (?, ?, ?, ?, 1.0)
                """, (track_id, genre, source, priority))

            if effective_genres:
                inserted += 1
                if any(source.endswith('inherited') for _, source, _ in effective_genres):
                    inherited_count += 1

        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            continue

        # Progress indicator every 100 tracks
        if i % 100 == 0:
            logger.info(f"  [{i}/{len(tracks)}] {inserted} tracks with effective genres")

    conn.commit()

    # Report statistics
    cursor.execute("SELECT COUNT(DISTINCT track_id) FROM track_effective_genres")
    tracks_with_genres = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT track_id) FROM tracks WHERE sonic_features IS NOT NULL")
    total_tracks = cursor.fetchone()[0]

    logger.info("\n" + "="*70)
    logger.info("EFFECTIVE GENRES REFRESH COMPLETE")
    logger.info("="*70)
    logger.info(f"Total tracks (sonic features):  {total_tracks}")
    logger.info(f"Tracks with effective genres:   {tracks_with_genres} ({100*tracks_with_genres/total_tracks:.1f}%)")
    logger.info(f"Tracks with inherited genres:   {inherited_count}")
    logger.info(f"Tracks with ZERO genres:        {total_tracks - tracks_with_genres}")

    conn.close()
    return True


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Refresh effective genres from all sources")
    parser.add_argument("--limit", type=int, help="Only refresh first N tracks (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Print debug output")

    args = parser.parse_args()

    success = refresh_effective_genres(limit=args.limit, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
