"""
Track Matcher - Matches Last.FM tracks to library
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import sqlite3
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from .artist_utils import get_artist_variations
from .string_utils import normalize_match_string

logger = logging.getLogger(__name__)


class TrackMatcher:
    """Matches Last.FM listening history to local library tracks"""

    def __init__(self, library_client, library_id: str = None, db_path: str = "metadata.db", cache_expiry_days: int = 7):
        """
        Initialize track matcher

        Args:
            library_client: LibraryClient instance for library access
            library_id: Music library ID (optional, can be set later)
            db_path: Path to metadata database
            cache_expiry_days: Number of days before cache expires
        """
        self.library = library_client
        self.library_id = library_id
        self.db_path = db_path
        self.cache_expiry_days = cache_expiry_days
        self.conn = None
        self._init_db_connection()
        logger.info("Initialized TrackMatcher with metadata database")

    def _init_db_connection(self):
        """Initialize database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.debug(f"Connected to metadata database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def match_lastfm_to_library(self, lastfm_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match Last.FM tracks to library

        Args:
            lastfm_tracks: List of tracks from Last.FM

        Returns:
            List of matched tracks with play count and timestamp info
        """
        logger.info(f"Matching tracks to library")

        matched_tracks = []
        unmatched_count = 0
        match_stats = {'mbid': 0, 'exact': 0, 'fuzzy': 0}

        for lfm_track in lastfm_tracks:
            library_match, match_type = self._find_best_match(lfm_track)

            if library_match:
                # Track match type for statistics
                if match_type:
                    match_stats[match_type] = match_stats.get(match_type, 0) + 1

                # Add Last.FM metadata to library track
                matched_track = {
                    **library_match,
                    'lastfm_timestamp': lfm_track.get('timestamp', 0),
                    'lastfm_url': lfm_track.get('url', ''),
                    'match_type': match_type
                }
                matched_tracks.append(matched_track)
            else:
                unmatched_count += 1
                if unmatched_count <= 10:  # Only log first 10 to avoid spam
                    logger.debug(f"No match: {lfm_track['artist']} - {lfm_track['title']}")

        match_rate = (len(matched_tracks) / len(lastfm_tracks) * 100) if lastfm_tracks else 0
        logger.info(f"Matched {len(matched_tracks)}/{len(lastfm_tracks)} tracks ({match_rate:.1f}%)")
        logger.info(f"Match breakdown - MBID: {match_stats.get('mbid', 0)}, Exact: {match_stats.get('exact', 0)}, Fuzzy: {match_stats.get('fuzzy', 0)}")

        if unmatched_count > 10:
            logger.debug(f"... and {unmatched_count - 10} more unmatched tracks")

        return matched_tracks


    def _find_best_match(self, lastfm_track: Dict[str, Any]):
        """
        Find best matching track for a Last.FM track using prioritized strategies.
        """
        artist = lastfm_track.get('artist', '')
        title = lastfm_track.get('title', '')
        lfm_mbid = lastfm_track.get('mbid', '')

        if not artist or not title:
            return None, None

        cursor = self.conn.cursor()
        norm_artist = normalize_match_string(artist, is_artist=True)
        norm_title = normalize_match_string(title)

        strategies = [
            lambda: self._match_by_mbid(cursor, lfm_mbid),
            lambda: self._match_exact(cursor, norm_artist, norm_title),
            lambda: self._match_exact_with_variations(cursor, artist, norm_title),
            lambda: self._match_fuzzy_within_artist(cursor, norm_artist, norm_title, artist, title),
        ]

        for strategy in strategies:
            match = strategy()
            if match[0]:
                return match

        return None, None

    def _match_by_mbid(self, cursor, lfm_mbid: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Strategy 1: MusicBrainz ID matching (highest confidence)."""
        if not lfm_mbid:
            return None, None
        cursor.execute("""
            SELECT track_id as rating_key, title, artist, album,
                   duration_ms as duration, musicbrainz_id as mbid
            FROM tracks
            WHERE musicbrainz_id = ?
        """, (lfm_mbid,))
        row = cursor.fetchone()
        if row:
            return dict(row), 'mbid'
        return None, None

    def _match_exact(self, cursor, norm_artist: str, norm_title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Strategy 2: Exact match on normalized artist/title."""
        cursor.execute("""
            SELECT track_id as rating_key, title, artist, album,
                   duration_ms as duration, musicbrainz_id as mbid
            FROM tracks
            WHERE norm_artist = ? AND norm_title = ?
        """, (norm_artist, norm_title))
        row = cursor.fetchone()
        if row:
            return dict(row), 'exact'
        return None, None

    def _match_exact_with_variations(self, cursor, artist: str, norm_title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Strategy 3: Exact match using alternate artist normalizations."""
        for alt_artist in get_artist_variations(artist):
            alt_norm = normalize_match_string(alt_artist, is_artist=True)
            cursor.execute("""
                SELECT track_id as rating_key, title, artist, album,
                       duration_ms as duration, musicbrainz_id as mbid
                FROM tracks
                WHERE norm_artist = ? AND norm_title = ?
            """, (alt_norm, norm_title))
            row = cursor.fetchone()
            if row:
                return dict(row), 'exact'
        return None, None

    def _match_fuzzy_within_artist(self, cursor, norm_artist: str, norm_title: str, artist: str, title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Strategy 4: Fuzzy match within the same normalized artist."""
        cursor.execute("""
            SELECT track_id as rating_key, title, artist, album,
                   duration_ms as duration, musicbrainz_id as mbid, norm_title
            FROM tracks
            WHERE norm_artist = ?
        """, (norm_artist,))

        artist_tracks = cursor.fetchall()
        best_match = None
        best_score = 0.0

        for track in artist_tracks:
            norm_title_db = track['norm_title']
            score = self._similarity_score(norm_title, norm_title_db)

            if score > best_score and score >= 0.80:
                best_score = score
                best_match = dict(track)

        if best_match:
            best_match.pop('norm_title', None)
            logger.debug(f"Fuzzy match ({best_score:.2f}): {artist} - {title}")
            return best_match, 'fuzzy'

        return None, None

    def _similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate similarity score between two strings

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return SequenceMatcher(None, s1, s2).ratio()

    def aggregate_play_counts(self, matched_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate play counts for tracks that appear multiple times

        Args:
            matched_tracks: List of matched tracks with timestamps

        Returns:
            Deduplicated list with aggregated play counts
        """
        track_stats = {}

        for track in matched_tracks:
            key = track.get('rating_key')
            if not key:
                continue

            if key not in track_stats:
                track_stats[key] = {
                    **track,
                    'play_count': 0,
                    'first_played': track.get('lastfm_timestamp', 0),
                    'last_played': track.get('lastfm_timestamp', 0)
                }

            stats = track_stats[key]
            stats['play_count'] += 1

            timestamp = track.get('lastfm_timestamp', 0)
            if timestamp > 0:
                stats['first_played'] = min(stats['first_played'], timestamp)
                stats['last_played'] = max(stats['last_played'], timestamp)

        # Convert back to list and sort by play count
        aggregated = list(track_stats.values())
        aggregated.sort(key=lambda x: x['play_count'], reverse=True)

        logger.info(f"Aggregated {len(matched_tracks)} plays into {len(aggregated)} unique tracks")
        return aggregated

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.debug("Closed database connection")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Track Matcher module loaded")
