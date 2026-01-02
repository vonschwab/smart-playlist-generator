#!/usr/bin/env python3
"""
Quick Genre Update from File Tags
===================================
Re-reads genre tags from audio files for specific artists/tracks and updates the database.
No full rescan needed - only processes specified tracks.

Usage:
    python update_file_genres.py --artist "leon todd johnson"
    python update_file_genres.py --artist "Artist Name" --dry-run
    python update_file_genres.py --track-id abc123def456
    python update_file_genres.py --all-missing  # Update all tracks with no genres

Workflow:
    1. Edit genre tags in your audio files (using MusicBrainz Picard, Mp3tag, etc.)
    2. Run this script to update the database from the file tags
    3. Genres are normalized and stored in track_genres table
"""
import sys
import sqlite3
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.genre_normalization import normalize_genre_list
from src.logging_utils import configure_logging

try:
    import mutagen
except ImportError:
    print("ERROR: mutagen library not found. Install with: pip install mutagen")
    sys.exit(1)


def get_tags_list(audio, tag_names: List[str]) -> List[str]:
    """Extract list of values from audio tags (e.g., genres)"""
    for tag_name in tag_names:
        values = audio.get(tag_name, [])
        if values:
            # Handle different tag formats
            if isinstance(values, list):
                return [str(v) for v in values if v]
            elif isinstance(values, str):
                return [values]
    return []


def read_genres_from_file(file_path: str) -> List[str]:
    """Read genre tags from an audio file"""
    try:
        audio = mutagen.File(file_path, easy=True)
        if audio is None:
            print(f"  WARNING: Could not read {file_path}")
            return []

        # Try different genre tag names
        genres = get_tags_list(audio, ['genre', 'TCON'])
        return genres
    except Exception as e:
        print(f"  ERROR reading {file_path}: {e}")
        return []


def update_track_genres(conn: sqlite3.Connection, track_id: str, genres: List[str], dry_run: bool = False):
    """Update genres for a track from file tags"""
    cursor = conn.cursor()

    # Normalize genres
    normalized_genres = normalize_genre_list(genres, filter_broad=True)

    if dry_run:
        print(f"    [DRY RUN] Would update genres: {normalized_genres}")
        return

    # Remove existing file-sourced genres
    cursor.execute("""
        DELETE FROM track_genres
        WHERE track_id = ? AND source = 'file'
    """, (track_id,))

    # Insert new genres
    for genre in normalized_genres:
        cursor.execute("""
            INSERT OR IGNORE INTO track_genres (track_id, genre, source, weight)
            VALUES (?, ?, 'file', 1.0)
        """, (track_id, genre))

    # Also update track_effective_genres if it exists
    cursor.execute("""
        DELETE FROM track_effective_genres
        WHERE track_id = ? AND source = 'file'
    """, (track_id,))

    for priority, genre in enumerate(normalized_genres):
        cursor.execute("""
            INSERT OR IGNORE INTO track_effective_genres
            (track_id, genre, source, priority, weight, last_updated)
            VALUES (?, ?, 'file', ?, 1.0, datetime('now'))
        """, (track_id, genre, priority))

    conn.commit()
    print(f"    ✓ Updated genres: {normalized_genres}")


def update_artist_genres(conn: sqlite3.Connection, artist: str, dry_run: bool = False):
    """Update genres for all tracks by an artist"""
    cursor = conn.cursor()

    # Get all tracks for this artist
    cursor.execute("""
        SELECT track_id, artist, title, file_path
        FROM tracks
        WHERE LOWER(artist) = LOWER(?)
        ORDER BY album, title
    """, (artist,))

    tracks = cursor.fetchall()

    if not tracks:
        print(f"No tracks found for artist: {artist}")
        return 0

    print(f"\nFound {len(tracks)} tracks for '{artist}':")
    updated = 0

    for row in tracks:
        track_id = row[0]
        title = row[2]
        file_path = row[3]

        if not file_path or not Path(file_path).exists():
            print(f"  ⚠ {title}: File not found, skipping")
            continue

        print(f"\n  {title}")
        print(f"    File: {file_path}")

        # Read genres from file
        genres = read_genres_from_file(file_path)

        if not genres:
            print(f"    → No genres in file tags")
            continue

        print(f"    → Raw genres from file: {genres}")

        # Update database
        update_track_genres(conn, track_id, genres, dry_run)
        updated += 1

    return updated


def update_all_missing_genres(conn: sqlite3.Connection, dry_run: bool = False, limit: Optional[int] = None):
    """Update genres for all tracks that have no genres"""
    cursor = conn.cursor()

    # Get tracks with no genres
    cursor.execute("""
        SELECT t.track_id, t.artist, t.title, t.file_path
        FROM tracks t
        WHERE t.file_path IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM track_genres tg
            WHERE tg.track_id = t.track_id AND tg.source = 'file'
        )
        ORDER BY t.artist, t.album, t.title
        LIMIT ?
    """, (limit or 999999,))

    tracks = cursor.fetchall()

    if not tracks:
        print("No tracks found with missing genres")
        return 0

    print(f"\nFound {len(tracks)} tracks with no file-sourced genres:")
    updated = 0

    for row in tracks:
        track_id = row[0]
        artist = row[1]
        title = row[2]
        file_path = row[3]

        if not file_path or not Path(file_path).exists():
            continue

        print(f"\n  {artist} - {title}")

        # Read genres from file
        genres = read_genres_from_file(file_path)

        if not genres:
            print(f"    → No genres in file tags")
            continue

        print(f"    → Raw genres: {genres}")

        # Update database
        update_track_genres(conn, track_id, genres, dry_run)
        updated += 1

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Update genre metadata from file tags without full rescan"
    )
    parser.add_argument('--artist', help='Update all tracks by this artist')
    parser.add_argument('--track-id', help='Update specific track by ID')
    parser.add_argument('--all-missing', action='store_true',
                       help='Update all tracks with no file-sourced genres')
    parser.add_argument('--limit', type=int, help='Limit number of tracks to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    parser.add_argument('--db', default='data/metadata.db', help='Database path')

    args = parser.parse_args()

    # Configure logging
    configure_logging('INFO', 'genre_update.log')

    # Connect to database
    db_path = ROOT_DIR / args.db
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=" * 60)
    print("Genre Update from File Tags")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made\n")

    updated = 0

    try:
        if args.artist:
            updated = update_artist_genres(conn, args.artist, args.dry_run)
        elif args.track_id:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM tracks WHERE track_id = ?", (args.track_id,))
            row = cursor.fetchone()
            if row and row[0]:
                genres = read_genres_from_file(row[0])
                if genres:
                    update_track_genres(conn, args.track_id, genres, args.dry_run)
                    updated = 1
        elif args.all_missing:
            updated = update_all_missing_genres(conn, args.dry_run, args.limit)
        else:
            parser.print_help()
            sys.exit(1)

        print("\n" + "=" * 60)
        if args.dry_run:
            print(f"DRY RUN: Would update {updated} tracks")
        else:
            print(f"✓ Updated {updated} tracks")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
