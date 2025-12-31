"""
Similarity Calculator - Compares sonic features to find similar tracks
"""
import hashlib
import json
import logging
import math
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
        # Sonic feature vector layout (slices, weights, lengths) built lazily
        self._sonic_feature_layout: Optional[Dict[str, Dict[str, Any]]] = None
        # Track per-segment fallbacks (for diagnostics in experiments)
        self.segment_fallback_counts: Dict[str, int] = {"start": 0, "mid": 0, "end": 0}

        # Genre similarity settings
        genre_config = self.config.get('playlists', {}).get('genre_similarity', {})
        self.genre_enabled = genre_config.get('enabled', True)
        self.genre_weight = genre_config.get('weight', 0.4)
        self.sonic_weight = genre_config.get('sonic_weight', 0.6)
        self.min_genre_similarity = genre_config.get('min_genre_similarity', 0.3)
        source_weights = genre_config.get('source_weights', {}) or {}
        self.genre_source_weights = {
            "track": float(source_weights.get("track", 1.2)),
            "album": float(source_weights.get("album", 1.0)),
            "artist": float(source_weights.get("artist", 0.4)),
        }

        # Duration matching settings
        duration_cfg = self.config.get('playlists', {}).get('duration_match', {})
        self.duration_match_enabled = duration_cfg.get('enabled', True)
        self.duration_weight = float(duration_cfg.get('weight', 0.35))
        self.duration_window_frac = float(duration_cfg.get('window_frac', 0.25))
        self.duration_falloff = float(duration_cfg.get('falloff', 0.6))
        self.duration_min_target_seconds = float(duration_cfg.get('min_target_seconds', 40))

        # Genre similarity method (default: ensemble)
        self.genre_method = genre_config.get('method', 'ensemble')

        # Genre filtering and combination settings
        # Artist tag enrichment relied on Last.FM; disable now that genre tags are MusicBrainz/file-only
        self.use_artist_tags = False
        self.use_discogs_album = genre_config.get('use_discogs_album', True)
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

        # Similar artist boosts are disabled now that Last.FM genre data is removed
        self.similar_artists_enabled = False
        self.similar_artists_boost = 0.0

        # Initialize genre similarity calculator (V2 with multiple methods)
        if self.genre_enabled:
            similarity_file = genre_config.get('similarity_file', 'data/genre_similarity.yaml')
            self.genre_calc = GenreSimilarityV2(similarity_file)
            logger.info(f"Genre similarity V2 enabled (method: {self.genre_method}, weight: {self.genre_weight}, min: {self.min_genre_similarity})")
        else:
            self.genre_calc = None
            logger.info("Genre similarity disabled")

        self._init_db_connection()

    # --- Duration helpers -------------------------------------------------

    @staticmethod
    def duration_similarity(
        target_s: float,
        cand_s: float,
        *,
        window_frac: float,
        falloff: float,
    ) -> float:
        """
        Smooth, symmetric duration preference in [0,1].
        Uses log-ratio distance with a flat window, then Gaussian-style decay.
        """
        if target_s <= 0 or cand_s <= 0:
            return 1.0
        eps = 1e-6
        d = abs(math.log((cand_s + eps) / (target_s + eps)))
        w = math.log(1 + max(window_frac, 0))
        if d <= w:
            return 1.0
        if falloff <= 0:
            return 0.0
        val = math.exp(-((d - w) / falloff) ** 2)
        return max(0.0, min(1.0, val))

    def _init_db_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to database: {self.db_path}")

    @staticmethod
    def _normalize_sonic_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize beat3tower segment dictionaries into the legacy keys expected by
        the similarity vectorizer. Returns the original dict when no mapping applies.
        """
        if not isinstance(features, dict):
            return {}

        if any(key in features for key in ("timbre", "harmony", "rhythm")):
            timbre = features.get("timbre") or {}
            harmony = features.get("harmony") or {}
            rhythm = features.get("rhythm") or {}
            bpm_info = features.get("bpm_info") or {}

            mapped = {
                "mfcc_mean": timbre.get("mfcc_median") or timbre.get("mfcc_mean"),
                "chroma_mean": harmony.get("chroma_median") or harmony.get("chroma_mean"),
                "bpm": rhythm.get("bpm") or bpm_info.get("primary_bpm"),
                "spectral_centroid": timbre.get("spec_centroid_median") or timbre.get("spec_centroid_mean"),
            }
            return {key: value for key, value in mapped.items() if value is not None}

        return features

    def _vector_from_features(self, features: Dict[str, Any], update_layout: bool = True) -> np.ndarray:
        """
        Internal helper to turn a feature dict (single segment) into a vector using
        the cached layout. If update_layout is False, lengths will not change.
        """
        if features is None:
            return np.array([], dtype=float)

        features = self._normalize_sonic_features(features)

        if self._sonic_feature_layout is None:
            self._sonic_feature_layout = {
                'mfcc_mean': {'weight': 0.6, 'length': None, 'slice': slice(0, 0)},
                'chroma': {'weight': 0.2, 'length': None, 'slice': slice(0, 0), 'key': None},
                'bpm': {'weight': 0.1, 'length': 1, 'slice': slice(0, 0)},
                'spectral_centroid': {'weight': 0.1, 'length': 1, 'slice': slice(0, 0)},
            }

        layout = self._sonic_feature_layout
        vector_parts: List[np.ndarray] = []

        def _normalize_and_append(values: Optional[Any], name: str, preferred_length: Optional[int]) -> np.ndarray:
            if values is None:
                if preferred_length:
                    return np.zeros(preferred_length, dtype=float)
                return np.array([], dtype=float)

            arr = np.asarray(values, dtype=float).flatten()
            if preferred_length is None:
                return arr

            if arr.size == preferred_length:
                return arr
            if arr.size == 0:
                return np.zeros(preferred_length, dtype=float)

            if arr.size < preferred_length:
                padding = np.zeros(preferred_length - arr.size, dtype=float)
                return np.concatenate([arr, padding])
            return arr[:preferred_length]

        # MFCC block
        mfcc_len = layout['mfcc_mean']['length']
        mfcc_values = features.get('mfcc_mean')
        if update_layout and mfcc_values is not None:
            mfcc_array = np.asarray(mfcc_values, dtype=float).flatten()
            if mfcc_array.size > 0:
                layout['mfcc_mean']['length'] = mfcc_array.size if mfcc_len is None else mfcc_len
                mfcc_len = layout['mfcc_mean']['length']
        mfcc_vector = _normalize_and_append(mfcc_values, 'mfcc_mean', layout['mfcc_mean']['length'])
        vector_parts.append(mfcc_vector)

        # Chroma/HPCP block (prefer hpcp_mean if available)
        chroma_key = 'hpcp_mean' if 'hpcp_mean' in features else 'chroma_mean'
        if chroma_key in features:
            layout['chroma']['key'] = chroma_key
            chroma_values = features.get(chroma_key)
        else:
            chroma_values = None
        chroma_len = layout['chroma']['length']
        if update_layout and chroma_values is not None:
            chroma_array = np.asarray(chroma_values, dtype=float).flatten()
            if chroma_array.size > 0:
                layout['chroma']['length'] = chroma_array.size if chroma_len is None else chroma_len
        chroma_vector = _normalize_and_append(chroma_values, 'chroma', layout['chroma']['length'])
        vector_parts.append(chroma_vector)

        # BPM scalar
        bpm_values = [features.get('bpm')] if 'bpm' in features else None
        bpm_len = layout['bpm']['length']
        if bpm_len is None and update_layout:
            layout['bpm']['length'] = 1
            bpm_len = 1
        bpm_vector = _normalize_and_append(bpm_values, 'bpm', layout['bpm']['length'])
        vector_parts.append(bpm_vector)

        # Spectral centroid scalar
        spec_values = [features.get('spectral_centroid')] if 'spectral_centroid' in features else None
        spec_len = layout['spectral_centroid']['length']
        if spec_len is None and update_layout:
            layout['spectral_centroid']['length'] = 1
            spec_len = 1
        spec_vector = _normalize_and_append(spec_values, 'spectral_centroid', layout['spectral_centroid']['length'])
        vector_parts.append(spec_vector)

        # Update slices for downstream consumers
        offset = 0
        for name in ['mfcc_mean', 'chroma', 'bpm', 'spectral_centroid']:
            length = layout[name]['length'] or 0
            layout[name]['slice'] = slice(offset, offset + length)
            offset += length

        if not vector_parts:
            return np.array([], dtype=float)

        return np.concatenate(vector_parts)

    def build_sonic_feature_vector(self, sonic_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert the multi-segment sonic_features JSON for a single track into a 1D vector.

        The ordering and scaling mirror the existing sonic similarity logic:
        - MFCC mean vector (weight 0.6)
        - Chroma/HPCP mean vector (weight 0.2)
        - BPM scalar (weight 0.1)
        - Spectral centroid scalar (weight 0.1)

        Missing components are filled with zeros once a length is established, keeping
        the vector dimension consistent for downstream experiments. Lengths are learned
        lazily from the first available instance of each component.
        """
        if not sonic_features:
            return np.array([], dtype=float)

        features = None
        if isinstance(sonic_features, dict):
            avg = sonic_features.get('average')
            full = sonic_features.get('full')
            if isinstance(avg, dict):
                features = avg
            elif isinstance(full, dict):
                features = full
        if features is None:
            features = sonic_features
        return self._vector_from_features(features, update_layout=True)

    def build_sonic_feature_vector_by_segment(self, sonic_features: Dict[str, Any], segment: str) -> np.ndarray:
        """
        Return the sonic feature vector for a specific segment ('start', 'mid', 'end')
        using the exact same dimension and ordering as build_sonic_feature_vector.

        If the requested segment is missing, fall back to the aggregate/full vector and
        increment a per-segment fallback counter for diagnostics.
        """
        if not sonic_features:
            return np.array([], dtype=float)

        # Ensure layout is initialized based on the full/average vector
        full_vec = self.build_sonic_feature_vector(sonic_features)
        seg_key = segment.lower()
        seg_data = sonic_features.get(seg_key)
        if seg_data is None and seg_key == "mid":
            seg_data = sonic_features.get("middle")

        if seg_data is None:
            if seg_key in self.segment_fallback_counts:
                self.segment_fallback_counts[seg_key] += 1
            return full_vec

        vec = self._vector_from_features(seg_data, update_layout=False)
        if vec.size == 0:
            if seg_key in self.segment_fallback_counts:
                self.segment_fallback_counts[seg_key] += 1
            return full_vec
        return vec

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
            # Build aligned feature vectors (handling multi-segment data and normalization)
            vec1 = self.build_sonic_feature_vector(features1)
            vec2 = self.build_sonic_feature_vector(features2)

            if vec1.size == 0 or vec2.size == 0:
                return 0.0

            layout = self._sonic_feature_layout or {}
            scores = []
            weights = []

            # 1. MFCC similarity (60% weight - timbre/texture)
            mfcc_slice = layout.get('mfcc_mean', {}).get('slice')
            if mfcc_slice and (mfcc_slice.stop - mfcc_slice.start) > 0:
                mfcc1 = vec1[mfcc_slice]
                mfcc2 = vec2[mfcc_slice]

                mag1 = np.linalg.norm(mfcc1)
                mag2 = np.linalg.norm(mfcc2)

                if mag1 > 1e-10 and mag2 > 1e-10:
                    # Cosine similarity (1 = same direction, -1 = opposite) converted to 0-1
                    mfcc_sim = (1 - cosine(mfcc1, mfcc2))
                    mfcc_sim = max(0, mfcc_sim)
                    scores.append(mfcc_sim)
                    weights.append(layout['mfcc_mean']['weight'])

            # 2. Chroma/HPCP similarity (20% weight - harmonic content)
            chroma_slice = layout.get('chroma', {}).get('slice')
            if chroma_slice and (chroma_slice.stop - chroma_slice.start) > 0:
                chroma1 = vec1[chroma_slice]
                chroma2 = vec2[chroma_slice]

                mag1 = np.linalg.norm(chroma1)
                mag2 = np.linalg.norm(chroma2)

                if mag1 > 1e-10 and mag2 > 1e-10:
                    chroma_sim = (1 - cosine(chroma1, chroma2))
                    chroma_sim = max(0, chroma_sim)
                    scores.append(chroma_sim)
                    weights.append(layout['chroma']['weight'])

            # 3. Rhythm similarity (10% weight - tempo)
            bpm_slice = layout.get('bpm', {}).get('slice')
            if bpm_slice and (bpm_slice.stop - bpm_slice.start) > 0:
                bpm1 = vec1[bpm_slice][0] if bpm_slice.stop - bpm_slice.start > 0 else 0
                bpm2 = vec2[bpm_slice][0] if bpm_slice.stop - bpm_slice.start > 0 else 0

                if bpm1 > 0 and bpm2 > 0:
                    tempo_diff = abs(bpm1 - bpm2)
                    tempo_diff_half = min(tempo_diff, abs(bpm1 - bpm2 / 2), abs(bpm1 - bpm2 * 2))
                    tempo_diff_half = min(tempo_diff_half, abs(bpm1 / 2 - bpm2))

                    # Convert difference to similarity (smaller diff = higher similarity)
                    tempo_sim = max(0, 1 - (tempo_diff_half / 40))
                    scores.append(tempo_sim)
                    weights.append(layout['bpm']['weight'])

            # 4. Spectral similarity (10% weight - brightness/texture)
            spec_slice = layout.get('spectral_centroid', {}).get('slice')
            if spec_slice and (spec_slice.stop - spec_slice.start) > 0:
                spec1 = vec1[spec_slice][0] if spec_slice.stop - spec_slice.start > 0 else 0
                spec2 = vec2[spec_slice][0] if spec_slice.stop - spec_slice.start > 0 else 0

                if spec1 > 0 and spec2 > 0:
                    spec_diff = abs(spec1 - spec2)
                    max_centroid = max(spec1, spec2)
                    spec_sim = max(0, 1 - (spec_diff / max_centroid))
                    scores.append(spec_sim)
                    weights.append(layout['spectral_centroid']['weight'])

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
            SELECT sonic_features, artist, duration_ms
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
        seed_duration_s = None
        if self.genre_enabled and self.genre_calc:
            seed_genres = self._get_combined_genres(track_id)
            logger.info(f"Seed track combined genres: {seed_genres}")
        if "duration_ms" in seed_row.keys() and seed_row["duration_ms"]:
            seed_duration_s = (seed_row["duration_ms"] or 0) / 1000.0

        # Get all tracks with features
        cursor.execute("""
            SELECT track_id, sonic_features, artist, duration_ms
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
                cand_duration_s = None
                if "duration_ms" in row.keys() and row["duration_ms"]:
                    cand_duration_s = (row["duration_ms"] or 0) / 1000.0

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

                    # Calculate genre similarity with confidence
                    genre_result = self.genre_calc.calculate_similarity_with_confidence(
                        seed_genres, candidate_genres
                    )
                    genre_sim = genre_result.score

                    # Soft gating: low confidence matches get a penalty instead of rejection
                    if genre_result.confidence == "low":
                        # Low confidence: apply penalty but don't hard-reject
                        # This prevents fragmented/unknown genres from killing good sonic matches
                        genre_sim = max(genre_sim, 0.15)  # Floor at 0.15 for low-confidence
                        logger.debug(
                            f"Low confidence genre match for {candidate_id}: "
                            f"score={genre_result.score:.2f}, using floor 0.15"
                        )
                    elif genre_sim < self.min_genre_similarity:
                        # High/medium confidence but below threshold: skip
                        continue

                    # Combine using weighted average
                    final_sim = (sonic_sim * self.sonic_weight) + (genre_sim * self.genre_weight)
                else:
                    final_sim = sonic_sim

                # Apply duration preference (soft, multiplicative)
                if (
                    self.duration_match_enabled
                    and self.duration_weight > 0
                    and seed_duration_s
                    and seed_duration_s >= self.duration_min_target_seconds
                    and cand_duration_s
                ):
                    dur_sim = self.duration_similarity(
                        seed_duration_s,
                        cand_duration_s,
                        window_frac=self.duration_window_frac,
                        falloff=self.duration_falloff,
                    )
                    final_sim = final_sim * (dur_sim ** self.duration_weight)

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
            SELECT sonic_features, artist, duration_ms
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
        seed_duration_s = None
        if "duration_ms" in seed_row.keys() and seed_row["duration_ms"]:
            seed_duration_s = (seed_row["duration_ms"] or 0) / 1000.0

        cursor.execute("""
            SELECT track_id, sonic_features, artist, duration_ms
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
                final_sim = sonic_sim

                if (
                    self.duration_match_enabled
                    and self.duration_weight > 0
                    and seed_duration_s
                    and seed_duration_s >= self.duration_min_target_seconds
                    and "duration_ms" in row.keys()
                    and row["duration_ms"]
                ):
                    cand_duration_s = (row["duration_ms"] or 0) / 1000.0
                    if cand_duration_s:
                        dur_sim = self.duration_similarity(
                            seed_duration_s,
                            cand_duration_s,
                            window_frac=self.duration_window_frac,
                            falloff=self.duration_falloff,
                        )
                        final_sim = final_sim * (dur_sim ** self.duration_weight)

                if final_sim >= min_similarity:
                    similarities.append((candidate_id, final_sim))

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
        Get genre tags directly associated with a track (excludes __EMPTY__ markers).
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

    def _get_combined_genres_with_weights(self, track_id: str) -> List[Tuple[str, float]]:
        """
        Get combined genre tags with weights from track, album, and artist.

        This method:
        1. Fetches track_genres for the track
        2. Fetches album_genres for the track's album
        3. Fetches artist_genres for the track's artist
        4. Filters out broad/noisy tags
        5. Normalizes and deduplicates while preserving priority order

        Priority order: track_genres > album_genres > artist_genres

        Args:
            track_id: Track ID

        Returns:
            Combined, filtered, deduplicated list of (genre, weight)
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

        combined_order: List[str] = []
        seen: Dict[str, Tuple[str, float]] = {}

        # 1. Get track genres (highest priority)
        track_genres = self._get_track_genres(track_id)

        # 2. Get album genres
        album_genres: List[str] = []
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
                if self.use_discogs_album:
                    cursor.execute("""
                        SELECT genre
                        FROM album_genres
                        WHERE album_id = ?
                          AND genre != '__EMPTY__'
                    """, (aid,))
                else:
                    cursor.execute("""
                        SELECT genre
                        FROM album_genres
                        WHERE album_id = ?
                          AND genre != '__EMPTY__'
                          AND (source IS NULL OR source NOT LIKE 'discogs_%')
                    """, (aid,))
                rows = cursor.fetchall()
                if rows:
                    album_genres = [r['genre'] for r in rows]
                    break

        # 3. Get artist genres
        cursor.execute("""
            SELECT genre
            FROM artist_genres
            WHERE artist = ?
        """, (artist_name,))

        artist_genres = [row['genre'] for row in cursor.fetchall()]

        sources = [
            ("track", self.genre_source_weights.get("track", 1.2), track_genres),
            ("album", self.genre_source_weights.get("album", 1.0), album_genres),
            ("artist", self.genre_source_weights.get("artist", 0.4), artist_genres),
        ]

        for _source, weight, genres in sources:
            if not genres:
                continue
            filtered = self._filter_broad_tags(genres)
            for genre in filtered:
                normalized = self._normalize_genre(genre)
                if not normalized:
                    continue
                existing = seen.get(normalized)
                if existing is None:
                    seen[normalized] = (genre, weight)
                    combined_order.append(normalized)
                else:
                    if weight > existing[1]:
                        seen[normalized] = (genre, weight)

        return [(seen[n][0], seen[n][1]) for n in combined_order]

    def _get_combined_genres(self, track_id: str) -> List[str]:
        """
        Get combined genre tags from track, album, and artist.

        Returns:
            Combined, filtered, deduplicated list of genres
        """
        weighted = self._get_combined_genres_with_weights(track_id)
        return [genre for genre, _ in weighted]

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

        # Calculate genre similarity with confidence
        genre_result = self.genre_calc.calculate_similarity_with_confidence(
            from_genres, to_genres, method=self.genre_method
        )
        genre_sim = genre_result.score

        # Soft gating for transitions: low confidence uses sonic-only
        if genre_result.confidence == "low":
            logger.debug(
                f"Low confidence genre for transition {from_track_id} -> {to_track_id}, "
                f"using sonic only (genre_sim={genre_sim:.3f})"
            )
            return sonic_sim
        elif genre_sim < self.min_genre_similarity:
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

            if not genres1 or not genres2:
                # No genre data: use sonic only
                hybrid_sim = sonic_sim
            else:
                # Calculate genre similarity with confidence
                genre_result = self.genre_calc.calculate_similarity_with_confidence(
                    genres1, genres2, method=self.genre_method
                )
                genre_sim = genre_result.score

                # Soft gating based on confidence
                if genre_result.confidence == "low":
                    # Low confidence: use sonic only instead of hard zeroing
                    hybrid_sim = sonic_sim
                elif genre_sim < self.min_genre_similarity:
                    logger.debug(f"Tracks {track1_id} and {track2_id} filtered by genre similarity ({genre_sim:.3f} < {self.min_genre_similarity})")
                    return 0.0
                else:
                    # Combine using weighted average
                    hybrid_sim = (sonic_sim * self.sonic_weight) + (genre_sim * self.genre_weight)

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

    def get_filtered_combined_genres_for_track(self, track_id: str) -> List[str]:
        """
        Public helper to retrieve combined genres for a track using production filtering.

        Combines album/artist/track genres (and optional artist tags) and applies
        the same normalization and broad tag filtering as the main generator.

        Args:
            track_id: Track ID to retrieve genres for

        Returns:
            List of cleaned genre tokens (broad/meta tags removed).
        """
        return self._get_combined_genres(track_id)

    def get_weighted_genres_for_track(self, track_id: str) -> List[Tuple[str, float]]:
        """
        Public helper to retrieve weighted combined genres for a track.

        Returns:
            List of (genre, weight) tuples.
        """
        return self._get_combined_genres_with_weights(track_id)

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
    calc = SimilarityCalculator()

    # Show stats
    stats = calc.get_stats()
    logger.info("Similarity Calculator Stats:")
    logger.info(f"  Total tracks: {stats['total_tracks']}")
    logger.info(f"  Analyzed: {stats['analyzed']}")
    logger.info(f"  Pending: {stats['pending']}")
    logger.info(f"  AcousticBrainz: {stats['acousticbrainz']}")
    logger.info(f"  Librosa: {stats['librosa']}")

    calc.close()
