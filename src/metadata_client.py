"""
Metadata Client - Local metadata database interface

Manages a SQLite database of enriched metadata from Last.FM and MusicBrainz
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

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
                album TEXT,
                duration_ms INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Artists table - stores artist metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artists (
                artist_name TEXT PRIMARY KEY,
                musicbrainz_id TEXT,
                lastfm_tags TEXT,  -- JSON array
                similar_artists TEXT,  -- JSON array
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Track genres table - many-to-many relationship
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_genres (
                track_id TEXT,
                genre TEXT,
                source TEXT,  -- 'lastfm', 'musicbrainz', 'manual'
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_mbid ON tracks(musicbrainz_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_genres_genre ON track_genres(genre)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artists_mbid ON artists(musicbrainz_id)")

        self.conn.commit()
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
        cursor.execute("""
            INSERT OR REPLACE INTO tracks
            (track_id, musicbrainz_id, title, artist, album, duration_ms, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (track_id, musicbrainz_id, title, artist, album, duration_ms))
        self.conn.commit()

    def add_artist(self, artist_name: str, musicbrainz_id: Optional[str] = None,
                   lastfm_tags: Optional[List[str]] = None,
                   similar_artists: Optional[List[str]] = None):
        """
        Add or update artist in database

        Args:
            artist_name: Artist name
            musicbrainz_id: MusicBrainz artist ID
            lastfm_tags: List of Last.FM tags/genres
            similar_artists: List of similar artist names
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO artists
            (artist_name, musicbrainz_id, lastfm_tags, similar_artists, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            artist_name,
            musicbrainz_id,
            json.dumps(lastfm_tags) if lastfm_tags else None,
            json.dumps(similar_artists) if similar_artists else None
        ))
        self.conn.commit()

    def add_track_genre(self, track_id: str, genre: str, source: str = 'lastfm',
                       weight: float = 1.0):
        """
        Add genre for a track

        Args:
            track_id: track ID
            genre: Genre name
            source: Source of genre ('lastfm', 'musicbrainz', 'manual')
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
            e.g., {'lastfm_track': ['rock', 'indie'], 'musicbrainz_artist': ['alternative']}
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
                                   lastfm_weight: float = 0.6,
                                   musicbrainz_weight: float = 0.4) -> List[str]:
        """
        Get combined genres with prioritization:
        - Last.FM: track > album > artist (weighted by lastfm_weight)
        - MusicBrainz: release > artist (weighted by musicbrainz_weight)

        Args:
            track_id: track ID
            lastfm_weight: Weight for Last.FM genres (0.0-1.0)
            musicbrainz_weight: Weight for MusicBrainz genres (0.0-1.0)

        Returns:
            Deduplicated list of genres with priority:
            1. Last.FM track-level (most specific)
            2. Last.FM album-level
            3. MusicBrainz release-level
            4. Last.FM artist-level
            5. MusicBrainz artist-level
        """
        genres_by_source = self.get_track_genres_by_source(track_id)

        # Priority order for combining
        priority = [
            'lastfm_track',      # Most specific
            'lastfm_album',
            'musicbrainz_release',
            'lastfm_artist',
            'musicbrainz_artist'
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

    def get_artist_genres(self, artist_name: str) -> List[str]:
        """
        Get genres for an artist from Last.FM tags

        Args:
            artist_name: Artist name

        Returns:
            List of genre tags
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT lastfm_tags FROM artists WHERE artist_name = ?
        """, (artist_name,))

        row = cursor.fetchone()
        if row and row['lastfm_tags']:
            return json.loads(row['lastfm_tags'])
        return []

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
            'lastfm_tags': json.loads(row['lastfm_tags']) if row['lastfm_tags'] else [],
            'similar_artists': json.loads(row['similar_artists']) if row['similar_artists'] else [],
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
            WHERE tg.genre IN ({placeholders}) AND tg.weight >= ? AND tg.genre != '__EMPTY__'
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
        query = """
            SELECT track_id, title, artist, album
            FROM tracks
            WHERE artist = ?
            LIMIT ?
        """

        cursor = self.conn.cursor()
        cursor.execute(query, (artist_name, limit))

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

        cursor.execute("SELECT COUNT(*) as count FROM artists WHERE lastfm_tags IS NOT NULL")
        stats['artists_with_tags'] = cursor.fetchone()['count']

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
