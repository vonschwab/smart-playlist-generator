"""
Similarity Calculator - Compares sonic features to find similar tracks
"""
import hashlib
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine, euclidean

from .genre_similarity_v2 import GenreSimilarityV2

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculates similarity between tracks using sonic features and genre"""

    def __init__(self, db_path: str = "metadata.db", config: Dict[str, Any] = None):
        """
        Initialize similarity calculator

        Args:
            db_path: Path to metadata database
            config: Configuration dictionary with genre similarity settings
        """
        self.db_path = db_path
        self.conn = None
        self.config = config or {}

        # Genre similarity settings
        genre_config = self.config.get('playlists', {}).get('genre_similarity', {})
        self.genre_enabled = genre_config.get('enabled', True)
        self.genre_weight = genre_config.get('weight', 0.4)
        self.sonic_weight = genre_config.get('sonic_weight', 0.6)
        self.min_genre_similarity = genre_config.get('min_genre_similarity', 0.3)

        # Genre similarity method (default: ensemble)
        self.genre_method = genre_config.get('method', 'ensemble')

        # Genre filtering and combination settings
        self.use_artist_tags = genre_config.get('use_artist_tags', True)
        default_broad = (
            ['__empty__', 'unknown', 'favorites', 'seen live'] +
            # decades
            ['50s', '60s', '70s', '80s', '90s', '00s', '2000s', '2010s', '2020s'] +
            # years
            [str(y) for y in range(1950, 2031)] +
            # noisy/meta tags
            [
                'best of 2009', 'best of 2010', 'best of 2011', 'best of 2012', 'best of 2013',
                'best of 2014', 'best of 2015', 'best of 2016', 'best of 2017', 'best of 2018',
                'best of 2019', 'best of 2020', 'best of 2021', 'best of 2022', 'best of 2023',
                'best of 2024', 'best of 2025', 'yes', 'sst', 'chicago', 'rock/pop', 'alternative & indie'
            ]
        )
        self.broad_filters = set(tag.lower() for tag in genre_config.get('broad_filters', default_broad))

        # Similar artists boost settings
        similar_artists_config = self.config.get('playlists', {}).get('similar_artists', {})
        self.similar_artists_enabled = similar_artists_config.get('enabled', True)
        self.similar_artists_boost = similar_artists_config.get('boost', 0.1)

        # Initialize genre similarity calculator (V2 with multiple methods)
        if self.genre_enabled:
            similarity_file = genre_config.get('similarity_file', 'data/genre_similarity.yaml')
            self.genre_calc = GenreSimilarityV2(similarity_file)
            logger.info(f"Genre similarity V2 enabled (method: {self.genre_method}, weight: {self.genre_weight}, min: {self.min_genre_similarity})")
        else:
            self.genre_calc = None
            logger.info("Genre similarity disabled")

        self._init_db_connection()
        logger.info("Initialized SimilarityCalculator")

    def _init_db_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to database: {self.db_path}")

    def calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate similarity score between two feature sets

        Args:
            features1: First track's features
            features2: Second track's features

        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        try:
            # Extract feature vectors
            scores = []
            weights = []

            # 1. MFCC similarity (60% weight - timbre/texture)
            if 'mfcc_mean' in features1 and 'mfcc_mean' in features2:
                mfcc1 = np.array(features1['mfcc_mean'])
                mfcc2 = np.array(features2['mfcc_mean'])

                if len(mfcc1) == len(mfcc2) and len(mfcc1) > 0:
                    # Check for zero-magnitude vectors
                    mag1 = np.linalg.norm(mfcc1)
                    mag2 = np.linalg.norm(mfcc2)

                    if mag1 > 1e-10 and mag2 > 1e-10:
                        # Cosine similarity (1 = same direction, -1 = opposite)
                        # Convert to 0-1 scale
                        mfcc_sim = (1 - cosine(mfcc1, mfcc2))
                        mfcc_sim = max(0, mfcc_sim)  # Clamp to 0-1
                        scores.append(mfcc_sim)
                        weights.append(0.6)

            # 2. Chroma/HPCP similarity (20% weight - harmonic content)
            chroma_key1 = 'hpcp_mean' if 'hpcp_mean' in features1 else 'chroma_mean'
            chroma_key2 = 'hpcp_mean' if 'hpcp_mean' in features2 else 'chroma_mean'

            if chroma_key1 in features1 and chroma_key2 in features2:
                chroma1 = np.array(features1[chroma_key1])
                chroma2 = np.array(features2[chroma_key2])

                if len(chroma1) == len(chroma2) and len(chroma1) > 0:
                    # Check for zero-magnitude vectors
                    mag1 = np.linalg.norm(chroma1)
                    mag2 = np.linalg.norm(chroma2)

                    if mag1 > 1e-10 and mag2 > 1e-10:
                        chroma_sim = (1 - cosine(chroma1, chroma2))
                        chroma_sim = max(0, chroma_sim)
                        scores.append(chroma_sim)
                        weights.append(0.2)

            # 3. Rhythm similarity (10% weight - tempo)
            if 'bpm' in features1 and 'bpm' in features2:
                bpm1 = features1['bpm']
                bpm2 = features2['bpm']

                if bpm1 > 0 and bpm2 > 0:
                    # Tempo similarity (allow for doubling/halving)
                    tempo_diff = abs(bpm1 - bpm2)
                    tempo_diff_half = min(tempo_diff, abs(bpm1 - bpm2/2), abs(bpm1 - bpm2*2))
                    tempo_diff_half = min(tempo_diff_half, abs(bpm1/2 - bpm2))

                    # Convert difference to similarity (smaller diff = higher similarity)
                    # Assume BPMs within 20 are very similar
                    tempo_sim = max(0, 1 - (tempo_diff_half / 40))
                    scores.append(tempo_sim)
                    weights.append(0.1)

            # 4. Spectral similarity (10% weight - brightness/texture)
            if 'spectral_centroid' in features1 and 'spectral_centroid' in features2:
                spec1 = features1['spectral_centroid']
                spec2 = features2['spectral_centroid']

                if spec1 > 0 and spec2 > 0:
                    # Normalize and compare
                    spec_diff = abs(spec1 - spec2)
                    max_centroid = max(spec1, spec2)
                    spec_sim = max(0, 1 - (spec_diff / max_centroid))
                    scores.append(spec_sim)
                    weights.append(0.1)

            # Calculate weighted average
            if scores:
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sim = sum(s * w for s, w in zip(scores, weights)) / total_weight
                    return weighted_sim

            # If no features could be compared, return 0
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def find_similar_tracks(self, track_id: str, limit: int = 50,
                          min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find tracks similar to the given track using hybrid sonic + genre similarity

        Args:
            track_id: ID of the seed track (track_id)
            limit: Maximum number of similar tracks to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity
        """
        # Get seed track features and artist
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT sonic_features, artist
            FROM tracks
            WHERE track_id = ?
        """, (track_id,))

        seed_row = cursor.fetchone()
        if not seed_row or not seed_row['sonic_features']:
            logger.debug(f"No features found for track {track_id}")
            return []

        seed_features = json.loads(seed_row['sonic_features'])
        if 'average' in seed_features:
            seed_features = seed_features['average']
        seed_artist = seed_row['artist']

        # Get seed track genres (if genre similarity enabled)
        seed_genres = []
        if self.genre_enabled and self.genre_calc:
            seed_genres = self._get_combined_genres(track_id)
            logger.info(f"Seed track combined genres: {seed_genres}")

        # Get all tracks with features
        cursor.execute("""
            SELECT track_id, sonic_features, artist
            FROM tracks
            WHERE sonic_features IS NOT NULL
              AND track_id != ?
        """, (track_id,))

        # Calculate similarity for each track
        similarities = []
        for row in cursor.fetchall():
            try:
                candidate_id = row['track_id']
                candidate_artist = row['artist']

                # Skip same-artist tracks here so the top-N results aren't monopolized
                if (candidate_artist or '').lower() == (seed_artist or '').lower():
                    continue
                candidate_features_raw = json.loads(row['sonic_features'])

                # Extract average segment for multi-segment features
                if 'average' in candidate_features_raw:
                    candidate_features = candidate_features_raw['average']
                else:
                    candidate_features = candidate_features_raw

                # Calculate sonic similarity
                sonic_sim = self.calculate_similarity(seed_features, candidate_features)

                # If genre similarity is enabled, calculate hybrid score
                if self.genre_enabled and self.genre_calc and seed_genres:
                    candidate_genres = self._get_combined_genres(candidate_id)

                    # Calculate genre similarity
                    genre_sim = self.genre_calc.calculate_similarity(seed_genres, candidate_genres)

                    # Filter by minimum genre similarity
                    if genre_sim < self.min_genre_similarity:
                        continue

                    # Combine using weighted average
                    final_sim = (sonic_sim * self.sonic_weight) + (genre_sim * self.genre_weight)
                else:
                    final_sim = sonic_sim

                # Apply similar artists boost if enabled
                if self.similar_artists_enabled and self._are_artists_similar(seed_artist, candidate_artist):
                    final_sim = min(1.0, final_sim + self.similar_artists_boost)
                    logger.debug(f"Similar artist boost applied: {seed_artist} <-> {candidate_artist} (new score: {final_sim:.3f})")

                if final_sim >= min_similarity:
                    similarities.append((candidate_id, final_sim))

            except Exception as e:
                logger.debug(f"Error processing track {row['track_id']}: {e}")
                continue

        # Sort by similarity (highest first) and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def find_similar_tracks_sonic_only(self, track_id: str, limit: int = 50,
                                       min_similarity: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find tracks similar to the given track using sonic features only.

        Args:
            track_id: ID of the seed track (track_id)
            limit: Maximum number of similar tracks to return
            min_similarity: Minimum sonic similarity threshold

        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT sonic_features, artist
            FROM tracks
            WHERE track_id = ?
        """, (track_id,))

        seed_row = cursor.fetchone()
        if not seed_row or not seed_row['sonic_features']:
            logger.debug(f"No features found for track {track_id}")
            return []

        seed_features = json.loads(seed_row['sonic_features'])
        if 'average' in seed_features:
            seed_features = seed_features['average']
        seed_artist = seed_row['artist'] or ''

        cursor.execute("""
            SELECT track_id, sonic_features, artist
            FROM tracks
            WHERE sonic_features IS NOT NULL
              AND track_id != ?
        """, (track_id,))

        similarities = []
        for row in cursor.fetchall():
            try:
                candidate_id = row['track_id']
                candidate_artist = row['artist'] or ''
                candidate_features_raw = json.loads(row['sonic_features'])

                # Extract average segment for multi-segment features
                if 'average' in candidate_features_raw:
                    candidate_features = candidate_features_raw['average']
                else:
                    candidate_features = candidate_features_raw

                sonic_sim = self.calculate_similarity(seed_features, candidate_features)

                if sonic_sim >= min_similarity:
                    similarities.append((candidate_id, sonic_sim))

            except Exception as e:
                logger.debug(f"Error processing track {row['track_id']}: {e}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def find_similar_to_multiple(self, track_ids: List[str], limit: int = 100,
                                min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find tracks similar to a collection of seed tracks (for playlist generation)

        Args:
            track_ids: List of seed track IDs
            limit: Maximum number of similar tracks to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (track_id, average_similarity_score) tuples
        """
        if not track_ids:
            return []

        # Get features for all seed tracks
        seed_features_list = []
        for track_id in track_ids:
            features = self._get_track_features(track_id)
            if features:
                seed_features_list.append(features)

        if not seed_features_list:
            logger.debug("No features found for any seed tracks")
            return []

        # Get all candidate tracks
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(track_ids))
        cursor.execute(f"""
            SELECT track_id, sonic_features
            FROM tracks
            WHERE sonic_features IS NOT NULL
              AND track_id NOT IN ({placeholders})
        """, track_ids)

        # Calculate average similarity to seed tracks
        similarities = []
        for row in cursor.fetchall():
            try:
                candidate_id = row['track_id']
                candidate_features = json.loads(row['sonic_features'])

                # Calculate similarity to each seed track
                sim_scores = []
                for seed_features in seed_features_list:
                    sim = self.calculate_similarity(seed_features, candidate_features)
                    sim_scores.append(sim)

                # Use average similarity
                avg_sim = np.mean(sim_scores)

                if avg_sim >= min_similarity:
                    similarities.append((candidate_id, avg_sim))

            except Exception as e:
                logger.debug(f"Error processing track {row['track_id']}: {e}")
                continue

        # Sort by similarity and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def _get_track_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get sonic features for a track from database

        Handles both multi-segment (new) and single-segment (legacy) formats.
        For multi-segment features, returns the 'average' segment.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT sonic_features
            FROM tracks
            WHERE track_id = ?
        """, (track_id,))

        row = cursor.fetchone()
        if row and row['sonic_features']:
            try:
                features = json.loads(row['sonic_features'])

                # Multi-segment format: extract 'average' for general similarity
                if 'average' in features:
                    return features['average']

                # Legacy single-segment format: use as-is
                return features

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in sonic_features for track {track_id}")
                return None
        return None

    def _get_track_segment(self, track_id: str, segment: str) -> Optional[Dict[str, Any]]:
        """
        Get specific segment features for a track

        Args:
            track_id: Track ID
            segment: Segment name ('beginning', 'middle', 'end', 'average')

        Returns:
            Segment features or None if not available
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT sonic_features
            FROM tracks
            WHERE track_id = ?
        """, (track_id,))

        row = cursor.fetchone()
        if row and row['sonic_features']:
            try:
                features = json.loads(row['sonic_features'])

                # Multi-segment format: extract requested segment
                if segment in features:
                    return features[segment]

                # If requesting specific segment but only have legacy format, use what we have
                if segment == 'average':
                    return features

                # Segment not available
                logger.debug(f"Segment '{segment}' not available for track {track_id}, using fallback")
                return None

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in sonic_features for track {track_id}")
                return None
        return None

    def _get_track_genres(self, track_id: str) -> List[str]:
        """
        DEPRECATED: Get genre tags for a track from database (excludes __EMPTY__ markers)

        This method only queries track_genres which are often empty.
        Use _get_combined_genres() instead for better coverage.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT genre
            FROM track_genres
            WHERE track_id = ? AND genre != '__EMPTY__'
        """, (track_id,))

        return [row['genre'] for row in cursor.fetchall()]

    def _normalize_genre(self, genre: str) -> str:
        """
        Normalize a genre tag for comparison and deduplication

        Args:
            genre: Raw genre string

        Returns:
            Normalized genre string (lowercase, stripped, punctuation normalized)
        """
        if not genre or not isinstance(genre, str):
            return ""

        # Lowercase and strip whitespace
        normalized = genre.lower().strip()

        # Replace multiple spaces with single space
        import re
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove leading/trailing punctuation but keep hyphens and slashes
        normalized = normalized.strip('.,;:!?\'"')

        # Normalize common separators without collapsing distinct styles
        normalized = normalized.replace(' | ', ' ')
        normalized = normalized.replace('|', ' ')
        normalized = normalized.replace(' & ', ' ')

        return normalized

    def _coerce_tag_to_str(self, tag: Any) -> str:
        """
        Convert a tag entry (possibly dict from API) to a string value.
        Accepts dicts with 'name' or 'tag' keys, otherwise returns empty string for non-strings.
        """
        if isinstance(tag, str):
            return tag
        if isinstance(tag, dict):
            # Common shapes: {'name': 'ambient'} or {'tag': 'ambient'}
            if 'name' in tag and isinstance(tag['name'], str):
                return tag['name']
            if 'tag' in tag and isinstance(tag['tag'], str):
                return tag['tag']
        return ""

    def _filter_broad_tags(self, genres: List[str]) -> List[str]:
        """
        Filter out broad/noisy genre tags

        Args:
            genres: List of genre tags

        Returns:
            Filtered list with broad tags removed
        """
        if not self.broad_filters:
            return genres

        filtered = []
        for genre in genres:
            normalized = self._normalize_genre(genre)
            if normalized and normalized not in self.broad_filters:
                filtered.append(genre)
            else:
                logger.debug(f"Filtered broad tag: {genre}")

        return filtered

    def _get_artist_data(self, artist_name: str) -> Dict[str, Any]:
        """
        Get artist metadata including tags and similar artists

        Args:
            artist_name: Artist name

        Returns:
            Dictionary with 'tags' and 'similar_artists' lists
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT lastfm_tags, similar_artists
            FROM artists
            WHERE artist_name = ?
        """, (artist_name,))

        row = cursor.fetchone()
        if not row:
            return {'tags': [], 'similar_artists': []}

        import json
        tags = json.loads(row['lastfm_tags']) if row['lastfm_tags'] else []
        similar = json.loads(row['similar_artists']) if row['similar_artists'] else []

        return {
            'tags': tags,
            'similar_artists': similar
        }

    def _are_artists_similar(self, artist1: str, artist2: str) -> bool:
        """
        Check if two artists are in each other's similar_artists list

        Args:
            artist1: First artist name
            artist2: Second artist name

        Returns:
            True if either artist appears in the other's similar_artists list
        """
        if not artist1 or not artist2:
            return False

        # Get similar_artists for both
        artist1_data = self._get_artist_data(artist1)
        artist2_data = self._get_artist_data(artist2)

        similar1 = set(a.lower() for a in artist1_data['similar_artists'])
        similar2 = set(a.lower() for a in artist2_data['similar_artists'])

        # Check if artist2 is in artist1's similar list, or vice versa
        is_similar = (artist2.lower() in similar1) or (artist1.lower() in similar2)

        if is_similar:
            logger.debug(f"Similar artist match: {artist1} <-> {artist2}")

        return is_similar

    def _get_combined_genres(self, track_id: str) -> List[str]:
        """
        Get combined genre tags from album, artist, and artist tags (NOT track_genres)

        This method:
        1. Fetches album_genres for the track's album
        2. Fetches artist_genres for the track's artist
        3. Optionally includes artist tags from artists.lastfm_tags
        4. Filters out broad/noisy tags
        5. Normalizes and deduplicates while preserving priority order

        Priority order: album_genres > artist_genres > artist_tags

        Args:
            track_id: Track ID

        Returns:
            Combined, filtered, deduplicated list of genres
        """
        cursor = self.conn.cursor()

        # Get track's artist, album, and album_id
        cursor.execute("""
            SELECT artist, album, album_id
            FROM tracks
            WHERE track_id = ?
        """, (track_id,))

        row = cursor.fetchone()
        if not row:
            logger.debug(f"Track {track_id} not found")
            return []

        artist_name = row['artist']
        album_name = row['album']
        album_id = row['album_id']

        combined_genres = []
        seen_normalized = set()

        # 1. Get album genres (highest priority)
        if album_id or album_name:
            album_ids_to_try = []

            # Prefer stored album_id
            if album_id:
                album_ids_to_try.append(album_id)

            # Fallback: compute md5 hash of artist|album (normalized) to match schema generation
            if album_name:
                computed_album_id = hashlib.md5(f"{artist_name}|{album_name}".lower().encode('utf-8')).hexdigest()[:16]
                album_ids_to_try.append(computed_album_id)

                # Last resort: legacy raw artist|album string
                album_ids_to_try.append(f"{artist_name}|{album_name}")

            album_genres = []
            for aid in album_ids_to_try:
                cursor.execute("""
                    SELECT genre
                    FROM album_genres
                    WHERE album_id = ?
                """, (aid,))
                rows = cursor.fetchall()
                if rows:
                    album_genres = [r['genre'] for r in rows]
                    logger.debug(f"Album genres for {album_name} (album_id={aid}): {album_genres}")
                    break

            for genre in album_genres:
                normalized = self._normalize_genre(genre)
                if normalized and normalized not in seen_normalized:
                    combined_genres.append(genre)
                    seen_normalized.add(normalized)

        # 2. Get artist genres
        cursor.execute("""
            SELECT genre
            FROM artist_genres
            WHERE artist = ?
        """, (artist_name,))

        artist_genres = [row['genre'] for row in cursor.fetchall()]
        logger.debug(f"Artist genres for {artist_name}: {artist_genres}")

        for genre in artist_genres:
            normalized = self._normalize_genre(genre)
            if normalized and normalized not in seen_normalized:
                combined_genres.append(genre)
                seen_normalized.add(normalized)

        # 3. Optionally get artist tags from artists.lastfm_tags
        if self.use_artist_tags:
            artist_data = self._get_artist_data(artist_name)
            artist_tags = artist_data['tags']
            logger.debug(f"Artist tags for {artist_name}: {artist_tags}")

            for tag in artist_tags:
                tag_str = self._coerce_tag_to_str(tag)
                normalized = self._normalize_genre(tag_str)
                if normalized and normalized not in seen_normalized:
                    combined_genres.append(tag_str)
                    seen_normalized.add(normalized)

        # 4. Filter broad/noisy tags
        filtered_genres = self._filter_broad_tags(combined_genres)

        logger.debug(f"Combined genres for track {track_id} ({artist_name} - {album_name}): {filtered_genres}")

        return filtered_genres

    def calculate_transition_similarity(self, from_track_id: str, to_track_id: str) -> Optional[float]:
        """
        Calculate transition similarity from one track to another
        Compares END segment of from_track to BEGINNING segment of to_track

        This creates smooth transitions by matching the outro of one track
        with the intro of the next track.

        Args:
            from_track_id: Track ID to transition from
            to_track_id: Track ID to transition to

        Returns:
            Transition similarity score (0.0-1.0) or None if calculation fails
        """
        # Get end segment of from_track
        from_features = self._get_track_segment(from_track_id, 'end')
        if not from_features:
            # Fallback to average if end segment not available
            from_features = self._get_track_features(from_track_id)
            logger.debug(f"Using average segment for {from_track_id} (end segment not available)")

        # Get beginning segment of to_track
        to_features = self._get_track_segment(to_track_id, 'beginning')
        if not to_features:
            # Fallback to average if beginning segment not available
            to_features = self._get_track_features(to_track_id)
            logger.debug(f"Using average segment for {to_track_id} (beginning segment not available)")

        if not from_features or not to_features:
            logger.warning(f"Cannot calculate transition similarity: {from_track_id} -> {to_track_id}")
            return None

        # Calculate sonic similarity between segments
        sonic_sim = self.calculate_similarity(from_features, to_features)

        # If genre similarity is disabled, return sonic only
        if not self.genre_enabled or not self.genre_calc:
            return sonic_sim

        # Get combined genres for both tracks
        from_genres = self._get_combined_genres(from_track_id)
        to_genres = self._get_combined_genres(to_track_id)

        # If genre data missing, use sonic only
        if not from_genres or not to_genres:
            logger.debug(f"No combined genres for transition {from_track_id} -> {to_track_id}, using sonic only")
            return sonic_sim

        # Calculate genre similarity
        genre_sim = self.genre_calc.calculate_similarity(from_genres, to_genres, method=self.genre_method)

        # If genre similarity below threshold but genres exist, allow sonic-only fallback
        if genre_sim < self.min_genre_similarity:
            logger.debug(f"Transition {from_track_id} -> {to_track_id} below min genre similarity ({genre_sim:.3f} < {self.min_genre_similarity}), using sonic only")
            return sonic_sim

        # Combine using weighted average (same weights as general similarity)
        final_sim = (sonic_sim * self.sonic_weight) + (genre_sim * self.genre_weight)
        return final_sim

    def calculate_hybrid_similarity(self, track1_id: str, track2_id: str) -> Optional[float]:
        """
        Calculate hybrid similarity combining sonic and genre similarity

        Args:
            track1_id: First track ID
            track2_id: Second track ID

        Returns:
            Hybrid similarity score (0.0-1.0) or None if calculation fails
        """
        # Get sonic features and artist names
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT track_id, sonic_features, artist
            FROM tracks
            WHERE track_id IN (?, ?)
        """, (track1_id, track2_id))

        rows = {row['track_id']: row for row in cursor.fetchall()}

        if track1_id not in rows or track2_id not in rows:
            return None

        features1 = self._get_track_features(track1_id)
        features2 = self._get_track_features(track2_id)

        if not features1 or not features2:
            return None

        artist1 = rows[track1_id]['artist']
        artist2 = rows[track2_id]['artist']

        # Calculate sonic similarity
        sonic_sim = self.calculate_similarity(features1, features2)

        # If genre similarity is disabled, return sonic only (with optional artist boost)
        if not self.genre_enabled or not self.genre_calc:
            hybrid_sim = sonic_sim
        else:
            # Get combined genres
            genres1 = self._get_combined_genres(track1_id)
            genres2 = self._get_combined_genres(track2_id)

            # Calculate genre similarity using configured method
            genre_sim = self.genre_calc.calculate_similarity(genres1, genres2, method=self.genre_method) if genres1 and genres2 else 0.0

            # If genre data is weak, fall back to sonic-only instead of hard zeroing
            if genre_sim < self.min_genre_similarity and (not genres1 or not genres2):
                hybrid_sim = sonic_sim
            elif genre_sim < self.min_genre_similarity:
                logger.debug(f"Tracks {track1_id} and {track2_id} filtered by genre similarity ({genre_sim:.3f} < {self.min_genre_similarity})")
                return 0.0
            else:
                # Combine using weighted average
                hybrid_sim = (sonic_sim * self.sonic_weight) + (genre_sim * self.genre_weight)
                logger.debug(f"Hybrid similarity: sonic={sonic_sim:.3f}, genre={genre_sim:.3f} ({self.genre_method}), final={hybrid_sim:.3f}")

        # Apply similar artists boost if enabled
        if self.similar_artists_enabled and self._are_artists_similar(artist1, artist2):
            hybrid_sim = min(1.0, hybrid_sim + self.similar_artists_boost)
            logger.debug(f"Similar artist boost applied: {artist1} <-> {artist2} (new score: {hybrid_sim:.3f})")

        return hybrid_sim

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about analyzed tracks"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM tracks WHERE file_path IS NOT NULL")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) as analyzed FROM tracks WHERE sonic_features IS NOT NULL")
        analyzed = cursor.fetchone()['analyzed']

        cursor.execute("""
            SELECT sonic_source, COUNT(*) as count
            FROM tracks
            WHERE sonic_source IS NOT NULL
            GROUP BY sonic_source
        """)

        sources = {row['sonic_source']: row['count'] for row in cursor.fetchall()}

        return {
            'total_tracks': total,
            'analyzed': analyzed,
            'pending': total - analyzed,
            'acousticbrainz': sources.get('acousticbrainz', 0),
            'librosa': sources.get('librosa', 0)
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure connection is closed on context exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    calc = SimilarityCalculator()

    # Show stats
    stats = calc.get_stats()
    print("Similarity Calculator Stats:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Analyzed: {stats['analyzed']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  AcousticBrainz: {stats['acousticbrainz']}")
    print(f"  Librosa: {stats['librosa']}")

    calc.close()
