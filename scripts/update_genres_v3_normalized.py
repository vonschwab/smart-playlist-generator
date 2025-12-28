#!/usr/bin/env python3
"""
Multi-Source Genre Updater V3 (Normalized Schema)
===================================================
Efficiently fetches genres using normalized database schema:
- Artist genres: Fetched once per artist → stored in artist_genres
- Album genres: Fetched once per album → stored in album_genres
- Track genres: Fetched per track → stored in track_genres

This dramatically reduces API calls compared to V2.

Usage:
    python update_genres_v3_normalized.py                  # Update all missing genres
    python update_genres_v3_normalized.py --artists        # Update only artist genres
    python update_genres_v3_normalized.py --albums         # Update only album genres
    python update_genres_v3_normalized.py --tracks         # Update only track genres
    python update_genres_v3_normalized.py --limit 10       # Limit to 10 items
    python update_genres_v3_normalized.py --stats          # Show statistics
"""
import sys
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Dict, Set
import logging

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config_loader import Config
from src.multi_source_genre_fetcher import MusicBrainzGenreFetcher
from src.genre_normalization import normalize_genre_list

# Configure logging (centralized)
from src.logging_config import setup_logging
logger = setup_logging(name='update_genres', log_file='genre_update_v3.log')


class NormalizedGenreUpdater:
    """Updates genres using normalized schema"""

    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize updater"""
        if config_path is None:
            config_path = ROOT_DIR / 'config.yaml'

        self.config = Config(config_path)
        self.db_path = Path(db_path) if db_path else ROOT_DIR / 'data' / 'metadata.db'
        self.conn = None

        # Initialize API clients
        self.musicbrainz = MusicBrainzGenreFetcher()

        self._init_db()
        logger.info("Initialized Normalized Genre Updater V3")

    def _init_db(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row

    def get_artists_needing_genres(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get artists that need genre data from any source

        Returns artists missing either Last.FM or MusicBrainz data
        """
        cursor = self.conn.cursor()

        query = """
            SELECT DISTINCT t.artist,
                   CASE WHEN mb.artist IS NULL THEN 1 ELSE 0 END as needs_musicbrainz
            FROM tracks t
            LEFT JOIN (
                SELECT DISTINCT artist FROM artist_genres
                WHERE source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
            ) mb ON t.artist = mb.artist
            WHERE t.artist IS NOT NULL AND TRIM(t.artist) != ''
              AND t.file_path IS NOT NULL AND t.file_path != ''
              AND (mb.artist IS NULL)
            ORDER BY t.artist
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_albums_needing_genres(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get albums that need genre data from any source

        Returns albums missing either Last.FM or MusicBrainz data
        """
        cursor = self.conn.cursor()

        query = """
            SELECT a.album_id, a.artist, a.title,
                   CASE WHEN mb.album_id IS NULL THEN 1 ELSE 0 END as needs_musicbrainz
            FROM albums a
            LEFT JOIN (
                SELECT DISTINCT album_id FROM album_genres WHERE source = 'musicbrainz_release'
            ) mb ON a.album_id = mb.album_id
            WHERE mb.album_id IS NULL
              AND a.album_id IN (
                  SELECT DISTINCT album_id
                  FROM tracks
                  WHERE file_path IS NOT NULL AND file_path != ''
                    AND album_id IS NOT NULL AND album != ''
              )
            ORDER BY a.artist, a.title
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_tracks_needing_genres(self, limit: Optional[int] = None) -> List[Dict]:
        """Track-level genre fetching from Last.FM is disabled; return empty list."""
        return []

    def update_artist_genres(self, limit: Optional[int] = None):
        """Update artist-level genres with collaboration inheritance fallback"""
        logger.info("=" * 70)
        logger.info("Updating Artist Genres (with collaboration inheritance)")
        logger.info("=" * 70)

        artists = self.get_artists_needing_genres(limit)
        total = len(artists)

        if total == 0:
            logger.info("No artists need genre updates!")
            return

        logger.info(f"Found {total} artists needing genres from one or more sources")

        stats = {
            'total': total,
            'musicbrainz': 0,
            'inherited': 0,
            'empty': 0,
            'failed': 0
        }
        start_time = time.time()

        for i, artist_data in enumerate(artists, 1):
            artist = artist_data['artist']
            needs_musicbrainz = bool(artist_data['needs_musicbrainz'])

            logger.info(f"[{i}/{total}] {artist}")

            try:
                # MusicBrainz artist genres (if needed)
                if needs_musicbrainz:
                    mb_genres = self.musicbrainz.fetch_musicbrainz_artist_genres(artist)
                    if mb_genres:
                        self._store_artist_genres(artist, mb_genres, 'musicbrainz_artist')
                        stats['musicbrainz'] += 1
                        logger.info(f"  + MusicBrainz: {', '.join(mb_genres[:3])}" +
                                  (f" (+{len(mb_genres)-3} more)" if len(mb_genres) > 3 else ""))
                    else:
                        # Try collaboration fallback
                        from src.artist_utils import parse_collaboration
                        constituents = parse_collaboration(artist)

                        if len(constituents) > 1:
                            # It's a collaboration - fetch from constituents
                            inherited_genres = []
                            for constituent in constituents:
                                constituent_genres = self.musicbrainz.fetch_musicbrainz_artist_genres(constituent)
                                # Take top 5 per artist
                                inherited_genres.extend(constituent_genres[:5])
                                time.sleep(1.1)  # Rate limit each constituent fetch

                            if inherited_genres:
                                # Deduplicate and cap at 10 total
                                unique_genres = list(dict.fromkeys(inherited_genres))[:10]
                                self._store_artist_genres(artist, unique_genres, 'musicbrainz_artist_inherited')
                                stats['inherited'] += 1
                                logger.info(f"  + Inherited from {len(constituents)} artists: " +
                                          f"{', '.join(unique_genres[:3])}" +
                                          (f" (+{len(unique_genres)-3} more)" if len(unique_genres) > 3 else ""))
                            else:
                                # No genres found for constituents either
                                self._store_artist_genres(artist, ['__EMPTY__'], 'musicbrainz_artist')
                                stats['empty'] += 1
                                logger.info(f"  - No genres found (collaboration with no constituent genres)")
                        else:
                            # Solo artist with no genres
                            self._store_artist_genres(artist, ['__EMPTY__'], 'musicbrainz_artist')
                            stats['empty'] += 1
                            logger.info(f"  - MusicBrainz: No genres found (marked as checked)")

                time.sleep(1.1)  # Rate limiting for MusicBrainz

            except Exception as e:
                logger.error(f"  Error: {e}")
                stats['failed'] += 1

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"Artist Genres Complete: {total - stats['failed']}/{total} in {elapsed:.1f}s")
        logger.info(f"  Direct MusicBrainz: {stats['musicbrainz']}")
        logger.info(f"  Inherited (collab): {stats['inherited']}")
        logger.info(f"  No genres found: {stats['empty']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info("=" * 70)

    def update_album_genres(self, limit: Optional[int] = None):
        """Update album-level genres"""
        logger.info("=" * 70)
        logger.info("Updating Album Genres")
        logger.info("=" * 70)

        albums = self.get_albums_needing_genres(limit)
        total = len(albums)

        if total == 0:
            logger.info("No albums need genre updates!")
            return

        logger.info(f"Found {total} albums needing genres from one or more sources")

        stats = {'total': total, 'musicbrainz': 0, 'empty': 0, 'failed': 0}
        start_time = time.time()

        for i, album in enumerate(albums, 1):
            needs_musicbrainz = bool(album['needs_musicbrainz'])

            logger.info(f"[{i}/{total}] {album['artist']} - {album['title']}")

            try:
                # MusicBrainz release genres (if needed)
                if needs_musicbrainz:
                    mb_genres = self.musicbrainz.fetch_musicbrainz_release_genres(album['artist'], album['title'])
                    if mb_genres:
                        self._store_album_genres(album['album_id'], mb_genres, 'musicbrainz_release')
                        stats['musicbrainz'] += 1
                        logger.info(f"  + MusicBrainz: {', '.join(mb_genres[:3])}" +
                                  (f" (+{len(mb_genres)-3} more)" if len(mb_genres) > 3 else ""))
                    else:
                        # Store empty marker to indicate we checked
                        self._store_album_genres(album['album_id'], ['__EMPTY__'], 'musicbrainz_release')
                        stats['empty'] += 1
                        logger.info(f"  - MusicBrainz: No genres found (marked as checked)")

                time.sleep(1.1)  # Rate limiting for MusicBrainz

            except Exception as e:
                logger.error(f"  Error: {e}")
                stats['failed'] += 1

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"Album Genres Complete: {total - stats['failed']}/{total} in {elapsed:.1f}s")
        logger.info(f"  MusicBrainz: {stats['musicbrainz']} | Failed: {stats['failed']}")
        logger.info("=" * 70)

    def update_track_genres(self, limit: Optional[int] = None):
        """Track-level genre fetching from Last.FM is disabled; purge legacy data."""
        logger.info("=" * 70)
        logger.info("Track genre updates are disabled (Last.FM track tags removed). Purging legacy lastfm_track rows.")
        logger.info("=" * 70)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM track_genres WHERE source = 'lastfm_track'")
        self.conn.commit()
        logger.info("Removed legacy lastfm_track entries from track_genres.")
        return

    def _store_artist_genres(self, artist: str, genres: List[str], source: str):
        """Store genres for an artist (with normalization)"""
        cursor = self.conn.cursor()
        # Normalize genres before storing
        normalized = normalize_genre_list(genres, filter_broad=True)
        for genre in normalized:
            cursor.execute("""
                INSERT OR IGNORE INTO artist_genres (artist, genre, source)
                VALUES (?, ?, ?)
            """, (artist, genre, source))
        self.conn.commit()
        if len(normalized) != len(genres):
            logger.debug(f"Normalized {len(genres)} genres to {len(normalized)} for artist {artist}")

    def _store_album_genres(self, album_id: str, genres: List[str], source: str):
        """Store genres for an album (with normalization)"""
        cursor = self.conn.cursor()
        # Normalize genres before storing
        normalized = normalize_genre_list(genres, filter_broad=True)
        for genre in normalized:
            cursor.execute("""
                INSERT OR IGNORE INTO album_genres (album_id, genre, source)
                VALUES (?, ?, ?)
            """, (album_id, genre, source))
        self.conn.commit()

    def _store_track_genres(self, track_id: str, genres: List[str], source: str):
        """Store genres for a track (with normalization)"""
        cursor = self.conn.cursor()
        # Normalize genres before storing
        normalized = normalize_genre_list(genres, filter_broad=True)
        for genre in normalized:
            cursor.execute("""
                INSERT OR IGNORE INTO track_genres (track_id, genre, source)
                VALUES (?, ?, ?)
            """, (track_id, genre, source))
        self.conn.commit()

    def get_stats(self):
        """Show genre statistics"""
        cursor = self.conn.cursor()

        print("\nGenre Statistics (Normalized Schema):")
        print("=" * 70)

        # Artists
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM tracks WHERE artist IS NOT NULL")
        total_artists = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM artist_genres")
        artists_with_genres = cursor.fetchone()[0]
        print(f"Artists: {artists_with_genres:,}/{total_artists:,} have genres ({artists_with_genres/total_artists*100:.1f}%)")

        # Albums
        cursor.execute("SELECT COUNT(*) FROM albums")
        total_albums = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT album_id) FROM album_genres")
        albums_with_genres = cursor.fetchone()[0]
        print(f"Albums: {albums_with_genres:,}/{total_albums:,} have genres ({albums_with_genres/total_albums*100:.1f}%)")

        # Tracks: Last.FM track tags disabled; file tag counts only (if present)
        cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL")
        total_tracks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT track_id) FROM track_genres WHERE source = 'file'")
        tracks_with_genres = cursor.fetchone()[0]
        print(f"Tracks: {tracks_with_genres:,}/{total_tracks:,} have file tag genres (Last.FM track tags disabled)")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Update genres with normalized schema')
    parser.add_argument('--artists', action='store_true', help='Update only artist genres')
    parser.add_argument('--albums', action='store_true', help='Update only album genres')
    parser.add_argument('--tracks', action='store_true', help='(Disabled) Track genre updates are no-ops; retains file tags only')
    parser.add_argument('--limit', type=int, help='Maximum number of items to process')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    args = parser.parse_args()

    updater = NormalizedGenreUpdater()

    if args.stats:
        updater.get_stats()
    else:
        # If no specific type specified, update artists and albums (track updates disabled)
        if not (args.artists or args.albums or args.tracks):
            updater.update_artist_genres(limit=args.limit)
            updater.update_album_genres(limit=args.limit)
        else:
            if args.artists:
                updater.update_artist_genres(limit=args.limit)
            if args.albums:
                updater.update_album_genres(limit=args.limit)
            if args.tracks:
                updater.update_track_genres(limit=args.limit)

    updater.close()
