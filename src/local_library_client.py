"""
Local Library Client - Music library client using local database
Provides the same interface as LibraryClient but queries metadata.db instead
"""
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from .artist_key_db import ensure_artist_key_schema
from .blacklist_db import ensure_blacklist_schema
from .similarity_calculator import SimilarityCalculator
from .string_utils import normalize_artist_key

logger = logging.getLogger(__name__)


class LocalLibraryClient:
    """
    Local library client that mimics LibraryClient interface
    Uses metadata.db and SimilarityCalculator instead of local database
    """

    def __init__(self, db_path: str = "data/metadata.db"):
        """
        Initialize local library client

        Args:
            db_path: Path to metadata database
        """
        self.db_path = db_path
        self.conn = None
        self.similarity_calc = SimilarityCalculator(db_path)
        self._init_db_connection()
        logger.info("Initialized LocalLibraryClient (local library mode)")

    def _init_db_connection(self):
        """Initialize database connection"""
        # Allow use across threads (FastAPI worker threads)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        ensure_artist_key_schema(self.conn, logger=logger)
        ensure_blacklist_schema(self.conn, logger=logger)
        logger.debug(f"Connected to database: {self.db_path}")

    def get_all_tracks(self, library_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all tracks from local library

        Args:
            library_id: Ignored (for compatibility with LibraryClient)

        Returns:
            List of track dictionaries
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                track_id as rating_key,
                artist,
                artist_key,
                title,
                album,
                duration_ms,
                file_path,
                musicbrainz_id as mbid
            FROM tracks
            WHERE file_path IS NOT NULL
              AND is_blacklisted = 0
            ORDER BY artist, album, title
        """)

        tracks = []
        for row in cursor.fetchall():
            artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
            track = {
                'rating_key': row['rating_key'],
                'artist': row['artist'] or '',
                'artist_key': artist_key,
                'title': row['title'] or '',
                'album': row['album'] or '',
                'duration': row['duration_ms'],  # Keep as milliseconds
                'file_path': row['file_path'],
                'mbid': row['mbid']
            }
            tracks.append(track)

        logger.info(f"Retrieved {len(tracks)} tracks from local library")
        return tracks

    def get_track_by_key(self, rating_key: str) -> Optional[Dict[str, Any]]:
        """
        Get track by rating key

        Args:
            rating_key: Track ID (track_id)

        Returns:
            Track dictionary or None
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                track_id as rating_key,
                artist,
                artist_key,
                title,
                album,
                duration_ms,
                file_path,
                musicbrainz_id as mbid
            FROM tracks
            WHERE track_id = ?
              AND is_blacklisted = 0
        """, (rating_key,))

        row = cursor.fetchone()
        if not row:
            logger.warning(f"Track {rating_key} not found in local library")
            return None

        artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
        track = {
            'rating_key': row['rating_key'],
            'artist': row['artist'] or '',
            'artist_key': artist_key,
            'title': row['title'] or '',
            'album': row['album'] or '',
            'duration': row['duration_ms'],
            'file_path': row['file_path'],
            'mbid': row['mbid']
        }

        return track

    def get_tracks_by_ids(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch tracks by rating keys. Preserves input order for hits; missing tracks are skipped.
        """
        if not track_ids:
            return []
        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in track_ids)
        cursor.execute(
            f"""
            SELECT
                track_id as rating_key,
                artist,
                artist_key,
                title,
                album,
                duration_ms,
                file_path,
                musicbrainz_id as mbid
            FROM tracks
            WHERE track_id IN ({placeholders})
              AND is_blacklisted = 0
            """,
            track_ids,
        )
        rows = cursor.fetchall()
        lookup = {}
        for row in rows:
            artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
            track = {
                'rating_key': row['rating_key'],
                'artist': row['artist'] or '',
                'artist_key': artist_key,
                'title': row['title'] or '',
                'album': row['album'] or '',
                'duration': row['duration_ms'],
                'file_path': row['file_path'],
                'mbid': row['mbid']
            }
            lookup[str(row["rating_key"])] = track
        ordered = [lookup[k] for k in map(str, track_ids) if k in lookup]
        return ordered

    def get_similar_tracks(self, rating_key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get tracks similar to the given track using sonic analysis

        Args:
            rating_key: Seed track ID
            limit: Maximum number of similar tracks to return

        Returns:
            List of similar track dictionaries with 'similarity_score'
        """
        # Use SimilarityCalculator to find similar tracks
        similar_ids = self.similarity_calc.find_similar_tracks(
            rating_key,
            limit=limit,
            min_similarity=0.3  # Lower threshold to allow more candidates for diversification
        )

        if not similar_ids:
            logger.debug(f"No similar tracks found for {rating_key}")
            return []

        # Get full track data for similar tracks
        similar_tracks = []
        for track_id, similarity_score in similar_ids:
            track = self.get_track_by_key(track_id)
            if track:
                track['similarity_score'] = similarity_score
                similar_tracks.append(track)

        logger.debug(f"Found {len(similar_tracks)} similar tracks for {rating_key}")
        return similar_tracks

    def get_similar_tracks_sonic_only(self, rating_key: str, limit: int = 50,
                                      min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get sonically similar tracks (no genre weighting/filtering).

        Args:
            rating_key: Seed track ID
            limit: Maximum number of tracks to return
            min_similarity: Minimum sonic similarity threshold

        Returns:
            List of similar track dictionaries with 'similarity_score'
        """
        similar_ids = self.similarity_calc.find_similar_tracks_sonic_only(
            rating_key,
            limit=limit,
            min_similarity=min_similarity
        )

        if not similar_ids:
            logger.debug(f"No sonic-only similar tracks found for {rating_key}")
            return []

        similar_tracks = []
        for track_id, similarity_score in similar_ids:
            track = self.get_track_by_key(track_id)
            if track:
                track['similarity_score'] = similarity_score
                similar_tracks.append(track)

        logger.debug(f"Found {len(similar_tracks)} sonic-only similar tracks for {rating_key}")
        return similar_tracks

    def get_tracks_for_genre(self, genre: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get tracks matching a specific genre (normalized).

        Uses a single optimized UNION query to fetch from all genre sources.
        Genres are already normalized/lowercase in DB, so no LOWER() needed.

        Args:
            genre: Normalized genre string (already lowercase)
            limit: Maximum number of tracks to return

        Returns:
            List of track dictionaries
        """
        cursor = self.conn.cursor()

        # Single optimized UNION query - fetches from all sources in one round-trip
        # Priority: track_effective_genres > album_genres > artist_genres (via UNION order)
        cursor.execute("""
            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   1 as priority
            FROM tracks t
            JOIN track_effective_genres teg ON t.track_id = teg.track_id
            WHERE teg.genre = ?
            AND t.file_path IS NOT NULL
            AND t.is_blacklisted = 0

            UNION

            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   2 as priority
            FROM tracks t
            JOIN album_genres ag ON t.album_id = ag.album_id
            WHERE ag.genre = ?
            AND t.file_path IS NOT NULL
            AND t.is_blacklisted = 0

            UNION

            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   3 as priority
            FROM tracks t
            JOIN artist_genres ag ON t.artist = ag.artist
            WHERE ag.genre = ?
            AND t.file_path IS NOT NULL
            AND t.is_blacklisted = 0

            LIMIT ?
        """, (genre, genre, genre, limit))

        tracks = []
        for row in cursor.fetchall():
            artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
            track = {
                'rating_key': row['rating_key'],
                'artist': row['artist'] or '',
                'artist_key': artist_key,
                'title': row['title'] or '',
                'album': row['album'] or '',
                'duration': row['duration'],
                'file_path': row['file_path'],
                'mbid': row['mbid']
            }
            tracks.append(track)

        logger.debug(f"Found {len(tracks)} tracks for genre '{genre}' (optimized UNION query)")
        return tracks

    def suggest_similar_genres(self, partial: str, limit: int = 10) -> List[str]:
        """
        Suggest genres matching a partial string.
        Used when exact match fails to help user find the right genre.

        Args:
            partial: Partial genre string to match (will be normalized)
            limit: Maximum number of suggestions

        Returns:
            List of genre suggestions (ranked: exact > prefix > contains; then by frequency)
        """
        from src.genre_normalization import normalize_genre_token

        cursor = self.conn.cursor()

        # Normalize the input for consistent matching
        normalized_partial = normalize_genre_token(partial) or partial.lower()

        # Search across all genre tables for matches
        # Rank by: exact match > prefix match > contains match, then by frequency
        cursor.execute("""
            SELECT genre,
                   COUNT(*) as freq,
                   CASE
                       WHEN LOWER(genre) = LOWER(?) THEN 0
                       WHEN LOWER(genre) LIKE LOWER(?) THEN 1
                       ELSE 2
                   END as match_type
            FROM (
                SELECT DISTINCT genre FROM track_effective_genres WHERE LOWER(genre) LIKE LOWER(?)
                UNION
                SELECT DISTINCT genre FROM album_genres WHERE LOWER(genre) LIKE LOWER(?)
                UNION
                SELECT DISTINCT genre FROM artist_genres WHERE LOWER(genre) LIKE LOWER(?)
            )
            GROUP BY genre
            ORDER BY match_type, freq DESC, genre
            LIMIT ?
        """, (normalized_partial, f"{normalized_partial}%",
              f"%{normalized_partial}%", f"%{normalized_partial}%", f"%{normalized_partial}%", limit))

        suggestions = [row[0] for row in cursor.fetchall()]
        return suggestions

    def get_play_history(self, library_id: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get play history (PLACEHOLDER - returns empty list)

        Note: Local library doesn't track play history. Use Last.FM instead.

        Args:
            library_id: Ignored
            days: Ignored

        Returns:
            Empty list (use Last.FM for play history)
        """
        logger.info("Local library mode: Play history not available (use Last.FM only)")
        return []

    def get_playlists(self, name_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get playlists (PLACEHOLDER - returns empty list)

        Note: Local library mode doesn't manage playlists. Only M3U exports are used.

        Args:
            name_prefix: Ignored

        Returns:
            Empty list (no playlists in local mode)
        """
        logger.debug("Local library mode: No playlists to manage (M3U-only)")
        return []

    def get_track_file_path(self, rating_key: str) -> Optional[str]:
        """
        Get the file system path for a track

        Args:
            rating_key: Rating key of the track

        Returns:
            File path or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT file_path FROM tracks WHERE track_id = ? AND is_blacklisted = 0",
            (rating_key,),
        )
        row = cursor.fetchone()
        return row['file_path'] if row else None

    def get_tracks_by_artist(self, artist_name: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """
        Get all tracks by an artist

        Args:
            artist_name: Artist name to search for
            fuzzy: If True, use LIKE matching (case-insensitive)

        Returns:
            List of track dictionaries
        """
        cursor = self.conn.cursor()

        artist_key = normalize_artist_key(artist_name or "")
        if fuzzy:
            cursor.execute("""
                SELECT
                    track_id as rating_key,
                    artist,
                    artist_key,
                    title,
                    album,
                    duration_ms,
                    file_path,
                    musicbrainz_id as mbid
                FROM tracks
                WHERE artist_key LIKE ?
                  AND file_path IS NOT NULL
                  AND is_blacklisted = 0
                ORDER BY album, title
            """, (f"%{artist_key}%",))
        else:
            cursor.execute("""
                SELECT
                    track_id as rating_key,
                    artist,
                    artist_key,
                    title,
                    album,
                    duration_ms,
                    file_path,
                    musicbrainz_id as mbid
                FROM tracks
                WHERE artist_key = ?
                  AND file_path IS NOT NULL
                  AND is_blacklisted = 0
                ORDER BY album, title
            """, (artist_key,))

        tracks = []
        for row in cursor.fetchall():
            key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
            track = {
                'rating_key': row['rating_key'],
                'artist': row['artist'] or '',
                'artist_key': key,
                'title': row['title'] or '',
                'album': row['album'] or '',
                'duration': row['duration_ms'],
                'file_path': row['file_path'],
                'mbid': row['mbid']
            }
            tracks.append(track)

        logger.debug(f"Found {len(tracks)} tracks for artist '{artist_name}'")
        return tracks

    def close(self):
        """Close database connections"""
        if self.conn:
            self.conn.close()
        if self.similarity_calc:
            self.similarity_calc.close()
        logger.debug("Closed LocalLibraryClient connections")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: ensure resources are closed."""
        self.close()


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    client = LocalLibraryClient()

    # Test getting all tracks
    tracks = client.get_all_tracks()
    logger.info(f"Total tracks: {len(tracks)}")

    if tracks:
        # Test getting similar tracks
        sample_track = tracks[0]
        logger.info(f"Sample track: {sample_track['artist']} - {sample_track['title']}")

        similar = client.get_similar_tracks(sample_track['rating_key'], limit=10)
        logger.info(f"Similar tracks:")
        for i, track in enumerate(similar, 1):
            logger.info(f"  {i}. {track['artist']} - {track['title']}")
            logger.info(f"     Similarity: {track.get('similarity_score', 0):.3f}")

    client.close()
