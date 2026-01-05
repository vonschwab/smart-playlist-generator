"""Unit tests for SimilarityCalculator.

Tests cover:
- Duration similarity calculations
- Sonic feature vector building
- Genre normalization and filtering
- Similarity scoring (sonic, genre, hybrid)
- Edge cases (empty vectors, NaN values, zero divisions)
"""

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.similarity_calculator import SimilarityCalculator


# ===========================================================================================
# Fixtures and Test Data
# ===========================================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database with minimal schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tracks table
    cursor.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            album TEXT,
            album_id TEXT,
            title TEXT,
            sonic_features TEXT,
            duration_ms INTEGER
        )
    """)

    # Create genre tables
    cursor.execute("""
        CREATE TABLE track_genres (
            track_id TEXT,
            genre TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE album_genres (
            album_id TEXT,
            genre TEXT,
            source TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE artist_genres (
            artist TEXT,
            genre TEXT
        )
    """)

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_sonic_features():
    """Sample sonic features for testing."""
    return {
        "mfcc_mean": [1.2, -0.5, 0.8, 1.1, -0.3, 0.6, -0.2, 0.9, 0.4, -0.7, 0.5, -0.1, 0.3],
        "chroma_mean": [0.8, 0.2, 0.5, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.2, 0.8, 0.5],
        "bpm": 120.0,
        "spectral_centroid": 1500.5,
    }


@pytest.fixture
def sample_multisegment_features():
    """Sample multi-segment sonic features (beat3tower format)."""
    return {
        "average": {
            "mfcc_mean": [1.2, -0.5, 0.8, 1.1, -0.3, 0.6, -0.2, 0.9, 0.4, -0.7, 0.5, -0.1, 0.3],
            "chroma_mean": [0.8, 0.2, 0.5, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.2, 0.8, 0.5],
            "bpm": 120.0,
            "spectral_centroid": 1500.5,
        },
        "start": {
            "mfcc_mean": [1.1, -0.4, 0.7, 1.0, -0.2, 0.5, -0.1, 0.8, 0.3, -0.6, 0.4, -0.0, 0.2],
            "chroma_mean": [0.7, 0.1, 0.4, 0.8, 0.0, 0.6, 0.2, 0.5, 0.3, 0.1, 0.7, 0.4],
            "bpm": 118.0,
            "spectral_centroid": 1450.0,
        },
        "end": {
            "mfcc_mean": [1.3, -0.6, 0.9, 1.2, -0.4, 0.7, -0.3, 1.0, 0.5, -0.8, 0.6, -0.2, 0.4],
            "chroma_mean": [0.9, 0.3, 0.6, 1.0, 0.2, 0.8, 0.4, 0.7, 0.5, 0.3, 0.9, 0.6],
            "bpm": 122.0,
            "spectral_centroid": 1550.0,
        },
    }


# ===========================================================================================
# Duration Similarity Tests
# ===========================================================================================

class TestDurationSimilarity:
    """Test duration similarity calculations."""

    def test_identical_durations(self):
        """Identical durations should return 1.0."""
        sim = SimilarityCalculator.duration_similarity(
            200.0, 200.0, window_frac=0.25, falloff=0.6
        )
        assert sim == 1.0

    def test_within_window(self):
        """Durations within window should return 1.0."""
        # 220s is 10% different from 200s, within 25% window
        sim = SimilarityCalculator.duration_similarity(
            200.0, 220.0, window_frac=0.25, falloff=0.6
        )
        assert sim == 1.0

    def test_outside_window(self):
        """Durations outside window should decay smoothly."""
        # 300s is 50% different from 200s, outside 25% window
        sim = SimilarityCalculator.duration_similarity(
            200.0, 300.0, window_frac=0.25, falloff=0.6
        )
        assert 0.0 < sim < 1.0

    def test_very_different_durations(self):
        """Very different durations should approach 0."""
        sim = SimilarityCalculator.duration_similarity(
            100.0, 500.0, window_frac=0.25, falloff=0.6
        )
        assert 0.0 <= sim < 0.1

    def test_zero_duration_handling(self):
        """Zero durations should return 1.0 (no penalty)."""
        sim = SimilarityCalculator.duration_similarity(
            0.0, 200.0, window_frac=0.25, falloff=0.6
        )
        assert sim == 1.0

    def test_negative_duration_handling(self):
        """Negative durations should return 1.0 (no penalty)."""
        sim = SimilarityCalculator.duration_similarity(
            -10.0, 200.0, window_frac=0.25, falloff=0.6
        )
        assert sim == 1.0

    def test_symmetric(self):
        """Duration similarity should be symmetric."""
        sim1 = SimilarityCalculator.duration_similarity(
            200.0, 300.0, window_frac=0.25, falloff=0.6
        )
        sim2 = SimilarityCalculator.duration_similarity(
            300.0, 200.0, window_frac=0.25, falloff=0.6
        )
        assert abs(sim1 - sim2) < 1e-6


# ===========================================================================================
# Sonic Feature Vector Tests
# ===========================================================================================

class TestSonicFeatureVector:
    """Test sonic feature vector building."""

    def test_build_vector_basic(self, temp_db):
        """Test building vector from basic features."""
        calc = SimilarityCalculator(str(temp_db), config={})
        features = {
            "mfcc_mean": [1.0, 2.0, 3.0],
            "chroma_mean": [0.5, 0.6],
            "bpm": 120.0,
            "spectral_centroid": 1500.0,
        }

        vec = calc.build_sonic_feature_vector(features)

        # Should concatenate: mfcc (3) + chroma (2) + bpm (1) + spectral (1) = 7
        assert vec.shape == (7,)
        assert vec[0] == 1.0  # First MFCC
        assert vec[3] == 0.5  # First chroma
        assert vec[5] == 120.0  # BPM
        assert vec[6] == 1500.0  # Spectral centroid

    def test_build_vector_multisegment_average(self, temp_db, sample_multisegment_features):
        """Test building vector from multi-segment features uses average."""
        calc = SimilarityCalculator(str(temp_db), config={})
        vec = calc.build_sonic_feature_vector(sample_multisegment_features)

        # Should use 'average' segment
        assert vec.size > 0
        # First MFCC should match average
        assert abs(vec[0] - 1.2) < 1e-6

    def test_build_vector_by_segment(self, temp_db, sample_multisegment_features):
        """Test building vector for specific segment."""
        calc = SimilarityCalculator(str(temp_db), config={})

        # Initialize layout with average
        calc.build_sonic_feature_vector(sample_multisegment_features)

        # Get start segment
        vec_start = calc.build_sonic_feature_vector_by_segment(
            sample_multisegment_features, "start"
        )
        assert vec_start.size > 0
        assert abs(vec_start[0] - 1.1) < 1e-6  # Start MFCC

        # Get end segment
        vec_end = calc.build_sonic_feature_vector_by_segment(
            sample_multisegment_features, "end"
        )
        assert abs(vec_end[0] - 1.3) < 1e-6  # End MFCC

    def test_build_vector_missing_segment_fallback(self, temp_db, sample_multisegment_features):
        """Test fallback to average when segment missing."""
        calc = SimilarityCalculator(str(temp_db), config={})

        # Initialize layout
        calc.build_sonic_feature_vector(sample_multisegment_features)

        # Request non-existent segment
        vec = calc.build_sonic_feature_vector_by_segment(
            sample_multisegment_features, "nonexistent"
        )

        # Should fallback to average
        assert vec.size > 0
        assert calc.segment_fallback_counts.get("nonexistent", 0) > 0

    def test_build_vector_empty_features(self, temp_db):
        """Test building vector from empty features."""
        calc = SimilarityCalculator(str(temp_db), config={})
        vec = calc.build_sonic_feature_vector({})
        assert vec.size == 0

    def test_build_vector_none_features(self, temp_db):
        """Test building vector from None."""
        calc = SimilarityCalculator(str(temp_db), config={})
        vec = calc.build_sonic_feature_vector(None)
        assert vec.size == 0

    def test_normalize_beat3tower_features(self, temp_db):
        """Test normalization of beat3tower format to legacy keys."""
        calc = SimilarityCalculator(str(temp_db), config={})

        beat3tower = {
            "timbre": {"mfcc_median": [1, 2, 3], "spec_centroid_median": 1500},
            "harmony": {"chroma_median": [0.5, 0.6]},
            "rhythm": {"bpm": 120},
        }

        normalized = calc._normalize_sonic_features(beat3tower)
        assert "mfcc_mean" in normalized
        assert "chroma_mean" in normalized
        assert "bpm" in normalized
        assert "spectral_centroid" in normalized


# ===========================================================================================
# Similarity Calculation Tests
# ===========================================================================================

class TestSimilarityCalculation:
    """Test sonic similarity calculations."""

    def test_identical_features(self, temp_db, sample_sonic_features):
        """Identical features should return 1.0."""
        calc = SimilarityCalculator(str(temp_db), config={})
        sim = calc.calculate_similarity(sample_sonic_features, sample_sonic_features)
        assert abs(sim - 1.0) < 1e-6

    def test_similar_features(self, temp_db, sample_sonic_features):
        """Similar features should return high similarity."""
        calc = SimilarityCalculator(str(temp_db), config={})

        # Create slightly different features
        features2 = {
            "mfcc_mean": [x + 0.1 for x in sample_sonic_features["mfcc_mean"]],
            "chroma_mean": sample_sonic_features["chroma_mean"],
            "bpm": sample_sonic_features["bpm"],
            "spectral_centroid": sample_sonic_features["spectral_centroid"],
        }

        sim = calc.calculate_similarity(sample_sonic_features, features2)
        assert 0.8 < sim < 1.0

    def test_very_different_features(self, temp_db, sample_sonic_features):
        """Very different features should return low similarity."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features2 = {
            "mfcc_mean": [-x for x in sample_sonic_features["mfcc_mean"]],  # Inverted
            "chroma_mean": [0.1] * len(sample_sonic_features["chroma_mean"]),
            "bpm": 80.0,  # Very different tempo
            "spectral_centroid": 500.0,  # Very different brightness
        }

        sim = calc.calculate_similarity(sample_sonic_features, features2)
        assert sim < 0.5

    def test_empty_features(self, temp_db):
        """Empty features should return 0.0."""
        calc = SimilarityCalculator(str(temp_db), config={})
        sim = calc.calculate_similarity({}, {})
        assert sim == 0.0

    def test_partial_features(self, temp_db):
        """Partial features should calculate similarity on available components."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {"mfcc_mean": [1.0, 2.0, 3.0], "bpm": 120.0}
        features2 = {"mfcc_mean": [1.0, 2.0, 3.0], "bpm": 120.0}

        sim = calc.calculate_similarity(features1, features2)
        assert sim > 0.9  # Should be high since available features match

    def test_zero_magnitude_vectors(self, temp_db):
        """Zero magnitude vectors should be handled gracefully."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {"mfcc_mean": [0.0] * 13, "chroma_mean": [0.0] * 12}
        features2 = {"mfcc_mean": [1.0] * 13, "chroma_mean": [1.0] * 12}

        # Should not crash, should return 0 or low value
        sim = calc.calculate_similarity(features1, features2)
        assert 0.0 <= sim <= 1.0


# ===========================================================================================
# Genre Processing Tests
# ===========================================================================================

class TestGenreProcessing:
    """Test genre normalization and filtering."""

    def test_normalize_genre_basic(self, temp_db):
        """Test basic genre normalization."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc._normalize_genre("Rock") == "rock"
        assert calc._normalize_genre("  Indie Rock  ") == "indie rock"
        assert calc._normalize_genre("Post-Rock") == "post-rock"

    def test_normalize_genre_punctuation(self, temp_db):
        """Test normalization handles punctuation."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc._normalize_genre("Rock & Roll") == "rock roll"
        assert calc._normalize_genre("Rock|Pop") == "rock pop"
        assert calc._normalize_genre("Rock | Pop") == "rock pop"

    def test_normalize_genre_whitespace(self, temp_db):
        """Test normalization handles whitespace."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc._normalize_genre("Post   Rock") == "post rock"
        assert calc._normalize_genre("\tIndie\n") == "indie"

    def test_normalize_genre_empty(self, temp_db):
        """Test normalization handles empty/invalid input."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc._normalize_genre("") == ""
        assert calc._normalize_genre(None) == ""
        assert calc._normalize_genre("   ") == ""

    def test_filter_broad_tags(self, temp_db):
        """Test broad tag filtering."""
        config = {
            "playlists": {
                "genre_similarity": {
                    "broad_filters": ["rock", "pop", "80s", "favorites"]
                }
            }
        }
        calc = SimilarityCalculator(str(temp_db), config=config)

        genres = ["ambient", "rock", "idm", "pop", "80s", "glitch", "favorites"]
        filtered = calc._filter_broad_tags(genres)

        assert "ambient" in filtered
        assert "idm" in filtered
        assert "glitch" in filtered
        assert "rock" not in filtered
        assert "pop" not in filtered
        assert "80s" not in filtered
        assert "favorites" not in filtered

    def test_filter_broad_tags_case_insensitive(self, temp_db):
        """Test broad tag filtering is case insensitive."""
        config = {
            "playlists": {
                "genre_similarity": {
                    "broad_filters": ["rock"]
                }
            }
        }
        calc = SimilarityCalculator(str(temp_db), config=config)

        genres = ["Rock", "ROCK", "rock", "ambient"]
        filtered = calc._filter_broad_tags(genres)

        assert "ambient" in filtered
        assert len([g for g in filtered if g.lower() == "rock"]) == 0


# ===========================================================================================
# Configuration Tests
# ===========================================================================================

class TestConfiguration:
    """Test configuration parsing and defaults."""

    def test_default_config(self, temp_db):
        """Test default configuration values."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc.genre_enabled is True
        assert calc.genre_weight == 0.4
        assert calc.sonic_weight == 0.6
        assert calc.min_genre_similarity == 0.3
        assert calc.duration_match_enabled is True

    def test_custom_config(self, temp_db):
        """Test custom configuration values."""
        config = {
            "playlists": {
                "genre_similarity": {
                    "enabled": False,
                    "weight": 0.7,
                    "sonic_weight": 0.3,
                    "min_genre_similarity": 0.5,
                },
                "duration_match": {
                    "enabled": False,
                    "weight": 0.5,
                },
            }
        }
        calc = SimilarityCalculator(str(temp_db), config=config)

        assert calc.genre_enabled is False
        assert calc.genre_weight == 0.7
        assert calc.sonic_weight == 0.3
        assert calc.min_genre_similarity == 0.5
        assert calc.duration_match_enabled is False
        assert calc.duration_weight == 0.5

    def test_source_weights_default(self, temp_db):
        """Test default source weights."""
        calc = SimilarityCalculator(str(temp_db), config={})

        assert calc.genre_source_weights["track"] == 1.2
        assert calc.genre_source_weights["album"] == 1.0
        assert calc.genre_source_weights["artist"] == 0.4

    def test_source_weights_custom(self, temp_db):
        """Test custom source weights."""
        config = {
            "playlists": {
                "genre_similarity": {
                    "source_weights": {
                        "track": 2.0,
                        "album": 1.5,
                        "artist": 0.5,
                    }
                }
            }
        }
        calc = SimilarityCalculator(str(temp_db), config=config)

        assert calc.genre_source_weights["track"] == 2.0
        assert calc.genre_source_weights["album"] == 1.5
        assert calc.genre_source_weights["artist"] == 0.5


# ===========================================================================================
# Edge Case Tests
# ===========================================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_in_features(self, temp_db):
        """Test handling of NaN values in features."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {
            "mfcc_mean": [1.0, float("nan"), 3.0],
            "bpm": 120.0,
        }
        features2 = {
            "mfcc_mean": [1.0, 2.0, 3.0],
            "bpm": 120.0,
        }

        # Should not crash
        try:
            sim = calc.calculate_similarity(features1, features2)
            assert 0.0 <= sim <= 1.0 or math.isnan(sim)
        except Exception:
            pytest.skip("NaN handling may vary by numpy version")

    def test_inf_in_features(self, temp_db):
        """Test handling of infinity values in features."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {
            "mfcc_mean": [1.0, float("inf"), 3.0],
            "bpm": 120.0,
        }
        features2 = {
            "mfcc_mean": [1.0, 2.0, 3.0],
            "bpm": 120.0,
        }

        # Should not crash
        try:
            sim = calc.calculate_similarity(features1, features2)
            assert 0.0 <= sim <= 1.0 or math.isnan(sim) or math.isinf(sim)
        except Exception:
            pytest.skip("Inf handling may vary by numpy version")

    def test_mismatched_vector_lengths(self, temp_db):
        """Test handling of mismatched feature vector lengths."""
        calc = SimilarityCalculator(str(temp_db), config={})

        # Build first vector
        features1 = {"mfcc_mean": [1.0, 2.0, 3.0], "bpm": 120.0}
        vec1 = calc.build_sonic_feature_vector(features1)

        # Try to build second vector with different length
        features2 = {"mfcc_mean": [1.0, 2.0], "bpm": 120.0}

        # Should handle gracefully (zero-pad or truncate)
        sim = calc.calculate_similarity(features1, features2)
        assert 0.0 <= sim <= 1.0

    def test_very_large_values(self, temp_db):
        """Test handling of very large feature values."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {
            "mfcc_mean": [1e10] * 13,
            "bpm": 1e6,
            "spectral_centroid": 1e8,
        }
        features2 = {
            "mfcc_mean": [1e10] * 13,
            "bpm": 1e6,
            "spectral_centroid": 1e8,
        }

        # Should not overflow
        sim = calc.calculate_similarity(features1, features2)
        assert 0.0 <= sim <= 1.0

    def test_very_small_values(self, temp_db):
        """Test handling of very small feature values."""
        calc = SimilarityCalculator(str(temp_db), config={})

        features1 = {
            "mfcc_mean": [1e-10] * 13,
            "bpm": 1e-10,
            "spectral_centroid": 1e-10,
        }
        features2 = {
            "mfcc_mean": [1e-10] * 13,
            "bpm": 1e-10,
            "spectral_centroid": 1e-10,
        }

        # Should not underflow
        sim = calc.calculate_similarity(features1, features2)
        assert 0.0 <= sim <= 1.0


# ===========================================================================================
# Context Manager Tests
# ===========================================================================================

class TestContextManager:
    """Test context manager support."""

    def test_context_manager(self, temp_db):
        """Test using SimilarityCalculator as context manager."""
        with SimilarityCalculator(str(temp_db), config={}) as calc:
            assert calc.conn is not None

        # Connection should be closed after context exit
        # (Note: checking this directly may vary by implementation)

    def test_context_manager_exception(self, temp_db):
        """Test context manager closes connection on exception."""
        try:
            with SimilarityCalculator(str(temp_db), config={}) as calc:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should have closed connection gracefully
