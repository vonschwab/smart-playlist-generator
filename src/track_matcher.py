"""
Track Matcher - Matches Last.FM tracks to library
"""
from typing import List, Dict, Any, Optional
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from difflib import SequenceMatcher

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
        Find best matching track for a Last.FM track using database queries

        Args:
            lastfm_track: Last.FM track data

        Returns:
            Tuple of (Matching library track or None, match type)
        """
        artist = lastfm_track.get('artist', '')
        title = lastfm_track.get('title', '')
        lfm_mbid = lastfm_track.get('mbid', '')

        if not artist or not title:
            return None, None

        cursor = self.conn.cursor()

        # Strategy 1: MusicBrainz ID matching (highest confidence)
        if lfm_mbid:
            cursor.execute("""
                SELECT track_id as rating_key, title, artist, album,
                       duration_ms as duration, musicbrainz_id as mbid
                FROM tracks
                WHERE musicbrainz_id = ?
            """, (lfm_mbid,))
            row = cursor.fetchone()
            if row:
                return dict(row), 'mbid'

        # Strategy 2: Exact match on normalized strings
        norm_artist = self._normalize_string(artist, is_artist=True)
        norm_title = self._normalize_string(title)

        cursor.execute("""
            SELECT track_id as rating_key, title, artist, album,
                   duration_ms as duration, musicbrainz_id as mbid
            FROM tracks
            WHERE norm_artist = ? AND norm_title = ?
        """, (norm_artist, norm_title))
        row = cursor.fetchone()
        if row:
            return dict(row), 'exact'

        # Strategy 3: Try alternate normalizations
        for alt_artist in self._get_artist_variations(artist):
            alt_norm = self._normalize_string(alt_artist, is_artist=True)
            cursor.execute("""
                SELECT track_id as rating_key, title, artist, album,
                       duration_ms as duration, musicbrainz_id as mbid
                FROM tracks
                WHERE norm_artist = ? AND norm_title = ?
            """, (alt_norm, norm_title))
            row = cursor.fetchone()
            if row:
                return dict(row), 'exact'

        # Strategy 4: Fuzzy matching within same artist
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

            # Require at least 80% similarity for fuzzy match
            if score > best_score and score >= 0.80:
                best_score = score
                best_match = dict(track)

        if best_match:
            # Remove norm_title from result (internal field)
            best_match.pop('norm_title', None)
            logger.debug(f"Fuzzy match ({best_score:.2f}): {artist} - {title}")
            return best_match, 'fuzzy'

        return None, None

    def _get_artist_variations(self, artist: str) -> List[str]:
        """Generate artist name variations for better matching"""
        variations = []

        # Handle "The Band" vs "Band, The"
        if artist.lower().startswith('the '):
            variations.append(artist[4:] + ', The')
        elif artist.lower().endswith(', the'):
            variations.append('The ' + artist[:-5])

        # Handle "&" vs "and"
        if ' & ' in artist:
            variations.append(artist.replace(' & ', ' and '))
        if ' and ' in artist.lower():
            variations.append(artist.replace(' and ', ' & ').replace(' And ', ' & '))

        return variations

    def _normalize_string(self, s: str, is_artist: bool = False) -> str:
        """
        Normalize string for matching (lowercase, remove special chars, etc.)

        Args:
            s: String to normalize
            is_artist: If True, applies artist-specific normalization (extracts primary artist)

        Returns:
            Normalized string
        """
        if not s:
            return ""

        # Lowercase
        s = s.lower()

        # Remove featuring/with artists
        s = re.sub(r'\s+(feat|ft|featuring|with|vs)[\.\s]+.*$', '', s, flags=re.IGNORECASE)

        # Remove content in parentheses/brackets (remixes, versions, etc.)
        s = re.sub(r'\([^)]*\)', '', s)
        s = re.sub(r'\[[^\]]*\]', '', s)

        # Remove common prefixes
        s = re.sub(r'^the\s+', '', s)

        # Normalize common variations
        s = s.replace('&', 'and')
        s = s.replace('+', 'and')

        # For artist names: extract primary artist if it's a collaboration
        # BUT preserve band names like "Sonny and The Sunsets"
        if is_artist:
            # Don't split if it contains "and the" (likely a band name)
            if not re.search(r'\s+and\s+the\s+', s, flags=re.IGNORECASE):
                # Split on common separators and take first artist
                s = re.split(r'\s*(?:and|,|;)\s+', s)[0]

        # Remove special characters but keep spaces
        s = re.sub(r'[^\w\s]', '', s)

        # Collapse multiple spaces
        s = re.sub(r'\s+', ' ', s)

        return s.strip()

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
