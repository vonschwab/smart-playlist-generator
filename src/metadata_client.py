"""
Metadata Client - Local metadata database interface

Manages a SQLite database of enriched metadata from MusicBrainz and file tags
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
from .artist_key_db import ensure_artist_key_schema
from .blacklist_db import ensure_blacklist_schema
from .string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


class MetadataClient:
    """Interface for local metadata database"""

    def __init__(self, db_path: str = "metadata.db"):
        """
        Initialize metadata client

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries

        cursor = self.conn.cursor()

        # Tracks table - maps tracks to external IDs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                musicbrainz_id TEXT,
                title TEXT,
                artist TEXT,
                artist_key TEXT,
                album TEXT,
                duration_ms INTEGER,
                is_blacklisted INTEGER NOT NULL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Artists table - stores artist metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artists (
                artist_name TEXT PRIMARY KEY,
                musicbrainz_id TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Track genres table - many-to-many relationship
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_genres (
                track_id TEXT,
                genre TEXT,
                source TEXT,  -- 'file', 'musicbrainz', 'manual'
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (track_id, genre, source),
                FOREIGN KEY (track_id) REFERENCES tracks(track_id)
            )
        """)

        # Album genres table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS album_genres (
                album_name TEXT,
                artist_name TEXT,
                genre TEXT,
                source TEXT,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (album_name, artist_name, genre, source)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist_key ON tracks(artist_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_mbid ON tracks(musicbrainz_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_genres_genre ON track_genres(genre)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artists_mbid ON artists(musicbrainz_id)")

        self.conn.commit()
        ensure_artist_key_schema(self.conn, logger=logger)
        ensure_blacklist_schema(self.conn, logger=logger)
        logger.info(f"Initialized metadata database: {self.db_path}")

    def add_track(self, track_id: str, title: str, artist: str, album: str,
                  duration_ms: int = 0, musicbrainz_id: Optional[str] = None):
        """
        Add or update track in database

        Args:
            track_id: track ID
            title: Track title
            artist: Artist name
            album: Album name
            duration_ms: Track duration in milliseconds
            musicbrainz_id: MusicBrainz recording ID
        """
        cursor = self.conn.cursor()
        artist_key = normalize_artist_key(artist or "")
        cursor.execute(
            """
            INSERT INTO tracks
            (track_id, musicbrainz_id, title, artist, artist_key, album, duration_ms, is_blacklisted, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(track_id) DO UPDATE SET
                musicbrainz_id=excluded.musicbrainz_id,
                title=excluded.title,
                artist=excluded.artist,
                artist_key=excluded.artist_key,
                album=excluded.album,
                duration_ms=excluded.duration_ms,
                last_updated=CURRENT_TIMESTAMP
            """,
            (track_id, musicbrainz_id, title, artist, artist_key, album, duration_ms),
        )
        self.conn.commit()

    def add_artist(self, artist_name: str, musicbrainz_id: Optional[str] = None):
        """
        Add or update artist in database

        Args:
            artist_name: Artist name
            musicbrainz_id: MusicBrainz artist ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO artists
            (artist_name, musicbrainz_id, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (
            artist_name,
            musicbrainz_id
        ))
        self.conn.commit()

    def add_track_genre(self, track_id: str, genre: str, source: str = 'file',
                       weight: float = 1.0):
        """
        Add genre for a track

        Args:
            track_id: track ID
            genre: Genre name
            source: Source of genre ('file', 'musicbrainz', 'manual')
            weight: Relevance weight (0.0-1.0)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO track_genres
            (track_id, genre, source, weight)
            VALUES (?, ?, ?, ?)
        """, (track_id, genre, source, weight))
        self.conn.commit()

    def get_track_genres(self, track_id: str, min_weight: float = 0.0) -> List[str]:
        """
        Get all genres for a track

        Args:
            track_id: track ID
            min_weight: Minimum weight threshold

        Returns:
            List of genre names sorted by weight (excludes __EMPTY__ markers)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT genre, MAX(weight) as max_weight
            FROM track_genres
            WHERE track_id = ? AND weight >= ? AND genre != '__EMPTY__'
            GROUP BY genre
            ORDER BY max_weight DESC
        """, (track_id, min_weight))

        return [row['genre'] for row in cursor.fetchall()]

    def get_track_genres_by_source(self, track_id: str) -> Dict[str, List[str]]:
        """
        Get genres for a track organized by source

        Args:
            track_id: track ID

        Returns:
            Dictionary mapping source to list of genres (excludes __EMPTY__ markers)
            e.g., {'file': ['rock', 'indie'], 'musicbrainz_artist': ['alternative']}
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT genre, source
            FROM track_genres
            WHERE track_id = ? AND genre != '__EMPTY__'
            ORDER BY source, genre
        """, (track_id,))

        genres_by_source = {}
        for row in cursor.fetchall():
            source = row['source']
            genre = row['genre']
            if source not in genres_by_source:
                genres_by_source[source] = []
            genres_by_source[source].append(genre)

        return genres_by_source

    def get_combined_track_genres(self, track_id: str,
                                   lastfm_weight: float = 0.0,
                                   musicbrainz_weight: float = 0.4) -> List[str]:
        """
        Get combined genres with prioritization:
        - MusicBrainz: release > artist (weighted by musicbrainz_weight)
        - File tags: track-level fallbacks

        Args:
            track_id: track ID
            lastfm_weight: Deprecated (ignored)
            musicbrainz_weight: Weight for MusicBrainz genres (0.0-1.0)

        Returns:
            Deduplicated list of genres with priority:
            1. MusicBrainz release-level
            2. MusicBrainz artist-level
            3. File tags
        """
        genres_by_source = self.get_track_genres_by_source(track_id)

        # Priority order for combining
        priority = [
            'musicbrainz_release',
            'musicbrainz_artist',
            'file'
        ]

        combined = []
        seen = set()

        for source in priority:
            if source in genres_by_source:
                for genre in genres_by_source[source]:
                    if genre not in seen:
                        combined.append(genre)
                        seen.add(genre)

        return combined

    def fetch_blacklisted_track_ids(self) -> set[str]:
        """Return blacklisted track ids."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT track_id FROM tracks WHERE is_blacklisted = 1")
        return {str(row["track_id"]) for row in cursor.fetchall()}

    def set_blacklisted(self, track_ids: List[str], value: bool) -> int:
        """Set blacklisted flag for a list of track ids."""
        if not track_ids:
            return 0
        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in track_ids)
        cursor.execute(
            f"""
            UPDATE tracks
            SET is_blacklisted = ?
            WHERE track_id IN ({placeholders})
            """,
            (1 if value else 0, *track_ids),
        )
        self.conn.commit()
        return cursor.rowcount or 0

    def fetch_blacklisted_tracks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch blacklisted tracks with basic metadata + genres."""
        cursor = self.conn.cursor()
        limit_clause = "LIMIT ?" if isinstance(limit, int) and limit > 0 else ""
        params: List[Any] = []
        if limit_clause:
            params.append(limit)
        query_with_path = f"""
            SELECT
                t.track_id,
                t.title,
                t.artist,
                t.album,
                t.duration_ms,
                t.file_path,
                GROUP_CONCAT(DISTINCT tg.genre) AS genres
            FROM tracks t
            LEFT JOIN track_genres tg
                ON t.track_id = tg.track_id AND tg.genre != '__EMPTY__'
            WHERE t.is_blacklisted = 1
            GROUP BY t.track_id, t.title, t.artist, t.album, t.duration_ms, t.file_path
            ORDER BY t.artist, t.album, t.title
            {limit_clause}
            """
        query_no_path = f"""
            SELECT
                t.track_id,
                t.title,
                t.artist,
                t.album,
                t.duration_ms,
                GROUP_CONCAT(DISTINCT tg.genre) AS genres
            FROM tracks t
            LEFT JOIN track_genres tg
                ON t.track_id = tg.track_id AND tg.genre != '__EMPTY__'
            WHERE t.is_blacklisted = 1
            GROUP BY t.track_id, t.title, t.artist, t.album, t.duration_ms
            ORDER BY t.artist, t.album, t.title
            {limit_clause}
            """
        try:
            cursor.execute(query_with_path, params)
        except sqlite3.OperationalError:
            cursor.execute(query_no_path, params)
        rows = []
        for row in cursor.fetchall():
            genres = []
            if row["genres"]:
                genres = [g for g in str(row["genres"]).split(",") if g]
            file_path = ""
            try:
                file_path = row["file_path"] or ""
            except Exception:
                file_path = ""
            rows.append(
                {
                    "track_id": row["track_id"],
                    "rating_key": row["track_id"],
                    "title": row["title"] or "",
                    "artist": row["artist"] or "",
                    "album": row["album"] or "",
                    "duration_ms": row["duration_ms"] or 0,
                    "file_path": file_path,
                    "genres": genres,
                }
            )
        return rows

    def fetch_track_durations(self, track_ids: List[str]) -> Dict[str, int]:
        """Fetch duration_ms for the given track_ids."""
        if not track_ids:
            return {}
        cursor = self.conn.cursor()
        durations: Dict[str, int] = {}
        chunk_size = 900
        for i in range(0, len(track_ids), chunk_size):
            chunk = [str(t) for t in track_ids[i:i + chunk_size]]
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(
                f"""
                SELECT track_id, duration_ms
                FROM tracks
                WHERE track_id IN ({placeholders})
                """,
                chunk,
            )
            for row in cursor.fetchall():
                durations[str(row["track_id"])] = int(row["duration_ms"] or 0)
        return durations

    def fetch_track_ids_by_duration_limits(
        self,
        *,
        min_ms: int,
        max_ms: int,
        cutoff_ms: int,
    ) -> set[str]:
        """Fetch track_ids that violate duration limits or have unknown durations."""
        cursor = self.conn.cursor()
        clauses = ["duration_ms IS NULL", "duration_ms <= 0"]
        params: List[Any] = []
        if min_ms > 0:
            clauses.append("duration_ms < ?")
            params.append(min_ms)
        if max_ms > 0:
            clauses.append("duration_ms > ?")
            params.append(max_ms)
        if cutoff_ms > 0:
            clauses.append("duration_ms > ?")
            params.append(cutoff_ms)
        if not clauses:
            return set()
        where_clause = " OR ".join(clauses)
        cursor.execute(
            f"""
            SELECT track_id
            FROM tracks
            WHERE {where_clause}
            """,
            params,
        )
        return {str(row["track_id"]) for row in cursor.fetchall()}

    def get_artist_metadata(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        Get all metadata for an artist

        Args:
            artist_name: Artist name

        Returns:
            Dictionary with artist metadata or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM artists WHERE artist_name = ?
        """, (artist_name,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'artist_name': row['artist_name'],
            'musicbrainz_id': row['musicbrainz_id'],
            'last_updated': row['last_updated']
        }

    def get_tracks_by_genre(self, genres: List[str], limit: int = 100,
                           min_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find tracks matching any of the given genres

        Args:
            genres: List of genre names to match
            limit: Maximum number of tracks to return
            min_weight: Minimum genre weight threshold

        Returns:
            List of track dictionaries with rating keys
        """
        if not genres:
            return []

        placeholders = ','.join('?' * len(genres))
        query = f"""
            SELECT DISTINCT t.track_id, t.title, t.artist, t.album,
                   GROUP_CONCAT(tg.genre, ', ') as genres,
                   MAX(tg.weight) as max_weight
            FROM tracks t
            JOIN track_genres tg ON t.track_id = tg.track_id
            WHERE tg.genre IN ({placeholders})
              AND tg.weight >= ?
              AND tg.genre != '__EMPTY__'
              AND t.is_blacklisted = 0
            GROUP BY t.track_id
            ORDER BY max_weight DESC
            LIMIT ?
        """

        cursor = self.conn.cursor()
        cursor.execute(query, (*genres, min_weight, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'track_id': row['track_id'],
                'title': row['title'],
                'artist': row['artist'],
                'album': row['album'],
                'genres': row['genres'].split(', ') if row['genres'] else [],
                'weight': row['max_weight']
            })

        return results

    def get_tracks_by_artist(self, artist_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find tracks by a specific artist

        Args:
            artist_name: Name of the artist
            limit: Maximum number of tracks to return

        Returns:
            List of track dictionaries with rating keys
        """
        artist_key = normalize_artist_key(artist_name or "")
        query = """
            SELECT track_id, title, artist, album
            FROM tracks
            WHERE artist_key = ?
              AND is_blacklisted = 0
            LIMIT ?
        """

        cursor = self.conn.cursor()
        cursor.execute(query, (artist_key, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'track_id': row['track_id'],
                'title': row['title'],
                'artist': row['artist'],
                'album': row['album'],
                'weight': 1.0
            })

        return results

    def get_stale_artists(self, days: int = 30) -> List[str]:
        """
        Get artists with metadata older than specified days

        Args:
            days: Number of days to consider stale

        Returns:
            List of artist names needing refresh
        """
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT artist_name FROM artists
            WHERE last_updated < ?
            ORDER BY last_updated ASC
        """, (cutoff.strftime('%Y-%m-%d %H:%M:%S'),))

        return [row['artist_name'] for row in cursor.fetchall()]

    def get_artists_without_metadata(self) -> List[str]:
        """
        Get artists that exist in tracks but not in artists table

        Returns:
            List of artist names missing metadata
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT artist FROM tracks
            WHERE artist NOT IN (SELECT artist_name FROM artists)
            ORDER BY artist
        """)

        return [row['artist'] for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics

        Returns:
            Dictionary with counts
        """
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) as count FROM tracks")
        stats['total_tracks'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM tracks WHERE musicbrainz_id IS NOT NULL")
        stats['tracks_with_mbid'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM artists")
        stats['total_artists'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(DISTINCT artist) as count FROM artist_genres")
        stats['artists_with_genres'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(DISTINCT track_id) as count FROM track_genres")
        stats['tracks_with_genres'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM track_genres")
        stats['total_genre_mappings'] = cursor.fetchone()['count']

        return stats

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed metadata database connection")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()
