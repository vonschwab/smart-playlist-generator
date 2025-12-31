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
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Set
import logging

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.logging_utils import ProgressLogger
from src.config_loader import Config
from src.multi_source_genre_fetcher import MusicBrainzGenreFetcher
from src.genre_normalization import normalize_genre_list
from src.artist_key_db import ensure_artist_key_schema
from src.string_utils import normalize_artist_key

# Logging will be configured in main() - just get the logger here
logger = logging.getLogger('update_genres')

STATUS_UNKNOWN = "unknown"
STATUS_OK = "ok"
STATUS_NO_MATCH = "no_match"
STATUS_FAILED = "failed"


def ensure_enrichment_status_schema(conn: sqlite3.Connection, logger: logging.Logger) -> None:
    """Create lightweight enrichment status table for attempt markers."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_status (
            entity_type TEXT,
            entity_id TEXT,
            status TEXT,
            attempt_count INTEGER DEFAULT 0,
            attempted_at TEXT,
            last_error TEXT,
            PRIMARY KEY (entity_type, entity_id)
        )
        """
    )
    conn.commit()
    logger.debug("Ensured enrichment_status table exists")


def get_enrichment_status(conn: sqlite3.Connection, entity_type: str, entity_id: str) -> str:
    cur = conn.execute(
        "SELECT status FROM enrichment_status WHERE entity_type=? AND entity_id=?",
        (entity_type, entity_id),
    )
    row = cur.fetchone()
    if not row:
        return STATUS_UNKNOWN
    try:
        status = row["status"]
    except Exception:
        status = row[0]
    return status or STATUS_UNKNOWN


def set_enrichment_status(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    status: str,
    last_error: Optional[str] = None,
) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    cur = conn.execute(
        "SELECT attempt_count FROM enrichment_status WHERE entity_type=? AND entity_id=?",
        (entity_type, entity_id),
    )
    row = cur.fetchone()
    attempt_count = (row[0] if row else 0) or 0
    conn.execute(
        """
        INSERT OR REPLACE INTO enrichment_status
        (entity_type, entity_id, status, attempt_count, attempted_at, last_error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (entity_type, entity_id, status, attempt_count + 1, now, last_error),
    )
    conn.commit()


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
        ensure_enrichment_status_schema(self.conn, logger=logger)

    def get_artists_needing_genres(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get artists that need genre data from any source.

        Returns artists missing MusicBrainz data, deduplicated by artist_key.
        """
        ensure_artist_key_schema(self.conn, logger=logger)
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT artist_key, artist, COUNT(*) as track_count
            FROM tracks
            WHERE artist IS NOT NULL AND TRIM(artist) != ''
              AND file_path IS NOT NULL AND file_path != ''
            GROUP BY artist_key, artist
        """)
        display_counts = {}
        for row in cursor.fetchall():
            try:
                artist = row["artist"]
                artist_key = row["artist_key"]
                track_count = row["track_count"]
            except Exception:
                artist_key, artist, track_count = row
            key = artist_key or normalize_artist_key(artist or "")
            if not key:
                continue
            if key not in display_counts:
                display_counts[key] = Counter()
            display_counts[key][artist] += int(track_count or 0)

        cursor.execute("""
            SELECT DISTINCT artist
            FROM artist_genres
            WHERE source IN ('musicbrainz_artist', 'musicbrainz_artist_inherited')
        """)
        artists_with_genres = {row["artist"] for row in cursor.fetchall()}

        results = []
        for key, counter in display_counts.items():
            status = get_enrichment_status(self.conn, "artist", key)
            if status in (STATUS_OK, STATUS_NO_MATCH):
                continue
            if any(name in artists_with_genres for name in counter.keys()):     
                continue
            if len(counter) > 1:
                logger.debug("Artist key collision for %s: %s", key, list(counter.keys())[:3])
            display = counter.most_common(1)[0][0]
            results.append({
                "artist": display,
                "artist_key": key,
                "needs_musicbrainz": 1,
            })

        results.sort(key=lambda r: r["artist"].casefold())
        if limit:
            return results[:limit]
        return results

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
        albums = []
        for row in cursor.fetchall():
            album_id = row["album_id"]
            status = get_enrichment_status(self.conn, "album", album_id)
            if status in (STATUS_OK, STATUS_NO_MATCH):
                continue
            albums.append(dict(row))
        return albums

    def get_tracks_needing_genres(self, limit: Optional[int] = None) -> List[Dict]:
        """Track-level genre fetching from Last.FM is disabled; return empty list."""
        return []

    def update_artist_genres(
        self,
        limit: Optional[int] = None,
        progress: bool = True,
        progress_interval: float = 15.0,
        progress_every: int = 500,
        verbose_each: bool = False,
    ):
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

        prog = ProgressLogger(
            logger,
            total=total,
            label="artist_genres",
            unit="artists",
            interval_s=progress_interval,
            every_n=progress_every,
            verbose_each=verbose_each,
        ) if progress else None

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
            key = artist_data.get("artist_key") or normalize_artist_key(artist or "")
            needs_musicbrainz = bool(artist_data['needs_musicbrainz'])

            if prog:
                prog.update(detail=artist)
            logger.debug(f"[{i}/{total}] {artist}")

            try:
                # MusicBrainz artist genres (if needed)
                if needs_musicbrainz:
                    mb_genres = self.musicbrainz.fetch_musicbrainz_artist_genres(artist)
                    if mb_genres:
                        self._store_artist_genres(artist, mb_genres, 'musicbrainz_artist')
                        set_enrichment_status(self.conn, "artist", key, STATUS_OK, last_error=None)
                        stats['musicbrainz'] += 1
                        logger.debug(f"  + MusicBrainz: {', '.join(mb_genres[:3])}" +
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
                                set_enrichment_status(self.conn, "artist", key, STATUS_OK, last_error=None)
                                stats['inherited'] += 1
                                logger.debug(f"  + Inherited from {len(constituents)} artists: " +
                                          f"{', '.join(unique_genres[:3])}" +
                                          (f" (+{len(unique_genres)-3} more)" if len(unique_genres) > 3 else ""))
                            else:
                                # No genres found for constituents either
                                self._store_artist_genres(artist, ['__EMPTY__'], 'musicbrainz_artist')
                                set_enrichment_status(self.conn, "artist", key, STATUS_NO_MATCH, last_error=None)
                                stats['empty'] += 1
                                logger.debug(f"  - No genres found (collaboration with no constituent genres)")
                        else:
                            # Solo artist with no genres
                            self._store_artist_genres(artist, ['__EMPTY__'], 'musicbrainz_artist')
                            set_enrichment_status(self.conn, "artist", key, STATUS_NO_MATCH, last_error=None)
                            stats['empty'] += 1
                            logger.debug(f"  - MusicBrainz: No genres found (marked as checked)")

                time.sleep(1.1)  # Rate limiting for MusicBrainz

            except Exception as e:
                logger.error(f"  Error processing {artist}: {e}")
                set_enrichment_status(self.conn, "artist", key, STATUS_FAILED, last_error=e.__class__.__name__)
                stats['failed'] += 1

            # Progress report every 50 artists
            if i % 50 == 0 or i == total:
                logger.info(f"Progress: {i}/{total} artists ({i/total*100:.0f}%) - "
                          f"found={stats['musicbrainz']}, inherited={stats['inherited']}, "
                          f"empty={stats['empty']}, failed={stats['failed']}")

        if prog:
            prog.finish()

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"Artist Genres Complete: {total - stats['failed']}/{total} in {elapsed:.1f}s")
        logger.info(f"  Direct MusicBrainz: {stats['musicbrainz']}")
        logger.info(f"  Inherited (collab): {stats['inherited']}")
        logger.info(f"  No genres found: {stats['empty']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info("=" * 70)

    def update_album_genres(
        self,
        limit: Optional[int] = None,
        progress: bool = True,
        progress_interval: float = 15.0,
        progress_every: int = 500,
        verbose_each: bool = False,
    ):
        """Update album-level genres"""
        logger.info("=" * 70)
        logger.info("Updating Album Genres")
        logger.info("=" * 70)

        albums = self.get_albums_needing_genres(limit)
        total = len(albums)

        if total == 0:
            logger.info("No albums need genre updates!")
            return

        prog = ProgressLogger(
            logger,
            total=total,
            label="album_genres",
            unit="albums",
            interval_s=progress_interval,
            every_n=progress_every,
            verbose_each=verbose_each,
        ) if progress else None

        logger.info(f"Found {total} albums needing genres from one or more sources")

        stats = {'total': total, 'musicbrainz': 0, 'empty': 0, 'failed': 0}
        start_time = time.time()

        for i, album in enumerate(albums, 1):
            needs_musicbrainz = bool(album['needs_musicbrainz'])
            album_id = album["album_id"]

            if prog:
                prog.update(detail=f"{album['artist']} - {album['title']}")
            logger.debug(f"[{i}/{total}] {album['artist']} - {album['title']}")

            try:
                # MusicBrainz release genres (if needed)
                if needs_musicbrainz:
                    mb_genres = self.musicbrainz.fetch_musicbrainz_release_genres(album['artist'], album['title'])
                    if mb_genres:
                        self._store_album_genres(album['album_id'], mb_genres, 'musicbrainz_release')
                        set_enrichment_status(self.conn, "album", album_id, STATUS_OK, last_error=None)
                        stats['musicbrainz'] += 1
                        logger.debug(f"  + MusicBrainz: {', '.join(mb_genres[:3])}" +
                                  (f" (+{len(mb_genres)-3} more)" if len(mb_genres) > 3 else ""))
                    else:
                        # Store empty marker to indicate we checked
                        self._store_album_genres(album['album_id'], ['__EMPTY__'], 'musicbrainz_release')
                        set_enrichment_status(self.conn, "album", album_id, STATUS_NO_MATCH, last_error=None)
                        stats['empty'] += 1
                        logger.debug(f"  - MusicBrainz: No genres found")

                time.sleep(1.1)  # Rate limiting for MusicBrainz

            except Exception as e:
                logger.error(f"  Error processing {album['artist']} - {album['title']}: {e}")
                set_enrichment_status(self.conn, "album", album_id, STATUS_FAILED, last_error=e.__class__.__name__)
                stats['failed'] += 1

            # Progress report every 50 albums
            if i % 50 == 0 or i == total:
                logger.info(f"Progress: {i}/{total} albums ({i/total*100:.0f}%) - "
                          f"found={stats['musicbrainz']}, empty={stats['empty']}, failed={stats['failed']}")

        if prog:
            prog.finish()

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

        logger.info("Genre Statistics (Normalized Schema):")
        logger.info("=" * 70)

        # Artists
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM tracks WHERE artist IS NOT NULL")
        total_artists = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM artist_genres")
        artists_with_genres = cursor.fetchone()[0]
        logger.info("Artists: %s/%s have genres (%.1f%%)", f"{artists_with_genres:,}", f"{total_artists:,}", artists_with_genres/total_artists*100)

        # Albums
        cursor.execute("SELECT COUNT(*) FROM albums")
        total_albums = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT album_id) FROM album_genres")
        albums_with_genres = cursor.fetchone()[0]
        logger.info("Albums: %s/%s have genres (%.1f%%)", f"{albums_with_genres:,}", f"{total_albums:,}", albums_with_genres/total_albums*100)

        # Tracks: Last.FM track tags disabled; file tag counts only (if present)
        cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_path IS NOT NULL")
        total_tracks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT track_id) FROM track_genres WHERE source = 'file'")
        tracks_with_genres = cursor.fetchone()[0]
        logger.info("Tracks: %s/%s have file tag genres (Last.FM track tags disabled)", f"{tracks_with_genres:,}", f"{total_tracks:,}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import argparse
    from src.logging_utils import configure_logging, add_logging_args, resolve_log_level

    parser = argparse.ArgumentParser(description='Update genres with normalized schema')
    parser.add_argument('--artists', action='store_true', help='Update only artist genres')
    parser.add_argument('--albums', action='store_true', help='Update only album genres')
    parser.add_argument('--tracks', action='store_true', help='(Disabled) Track genre updates are no-ops; retains file tags only')
    parser.add_argument('--limit', type=int, help='Maximum number of items to process')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--progress', dest='progress', action='store_true', default=True,
                        help='Enable progress logging (default)')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable progress logging')
    parser.add_argument('--progress-interval', type=float, default=15.0,
                        help='Seconds between progress updates (default: 15)')
    parser.add_argument('--progress-every', type=int, default=500,
                        help='Items between progress updates (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose per-item progress (DEBUG)')
    add_logging_args(parser)
    args = parser.parse_args()

    # Configure logging
    log_level = resolve_log_level(args)
    if args.verbose and not args.debug and not args.quiet and args.log_level.upper() == "INFO":
        log_level = "DEBUG"
    log_file = getattr(args, 'log_file', None) or 'genre_update_v3.log'
    configure_logging(level=log_level, log_file=log_file)

    updater = NormalizedGenreUpdater()

    if args.stats:
        updater.get_stats()
    else:
        # If no specific type specified, update artists and albums (track updates disabled)
        if not (args.artists or args.albums or args.tracks):
            updater.update_artist_genres(
                limit=args.limit,
                progress=args.progress,
                progress_interval=args.progress_interval,
                progress_every=args.progress_every,
                verbose_each=args.verbose,
            )
            updater.update_album_genres(
                limit=args.limit,
                progress=args.progress,
                progress_interval=args.progress_interval,
                progress_every=args.progress_every,
                verbose_each=args.verbose,
            )
        else:
            if args.artists:
                updater.update_artist_genres(
                    limit=args.limit,
                    progress=args.progress,
                    progress_interval=args.progress_interval,
                    progress_every=args.progress_every,
                    verbose_each=args.verbose,
                )
            if args.albums:
                updater.update_album_genres(
                    limit=args.limit,
                    progress=args.progress,
                    progress_interval=args.progress_interval,
                    progress_every=args.progress_every,
                    verbose_each=args.verbose,
                )
            if args.tracks:
                updater.update_track_genres(limit=args.limit)

    updater.close()
