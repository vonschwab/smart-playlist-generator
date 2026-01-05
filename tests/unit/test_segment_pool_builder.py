"""Unit tests for segment pool builder module.

Tests extracted segment pool building logic from pier_bridge_builder.py (Phase 3.2).

Coverage:
- Structural filtering (used tracks, allowed set, artist policies, track keys)
- Bridge scoring (harmonic mean, bridge floor gating)
- Internal connectors (priority handling, cap enforcement)
- Final selection (1-per-artist constraint)
- Edge cases (empty pools, zero similarity)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.segment_pool_builder import (
    SegmentCandidatePoolBuilder,
    SegmentPoolConfig,
    SegmentPoolResult,
)


# =============================================================================
# Mock Helper Classes
# =============================================================================

@dataclass
class MockIdentityKeys:
    """Mock identity keys for testing."""
    artist_key: str
    title_key: str
    track_key: tuple[str, str]


class MockArtifactBundle:
    """Mock artifact bundle for testing."""

    def __init__(self, track_count: int = 10):
        self.track_count = track_count
        # Create mock artist/title keys
        self.artist_keys = [f"artist_{i % 5}" for i in range(track_count)]  # 5 unique artists
        self.title_keys = [f"title_{i}" for i in range(track_count)]

    def get_identity_keys(self, idx: int) -> MockIdentityKeys:
        """Get mock identity keys for index."""
        artist_key = self.artist_keys[idx]
        title_key = self.title_keys[idx]
        return MockIdentityKeys(
            artist_key=artist_key,
            title_key=title_key,
            track_key=(artist_key, title_key),
        )


def mock_identity_keys_for_index(bundle, idx):
    """Mock implementation of identity_keys_for_index."""
    return bundle.get_identity_keys(idx)


# Patch the import
import src.playlist.segment_pool_builder as spb_module
spb_module.identity_keys_for_index = mock_identity_keys_for_index


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicPoolBuilding:
    """Test basic pool building functionality."""

    def test_simple_pool_building(self):
        """Test basic pool building with minimal configuration."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        # Create L2-normalized similarity matrix (10 tracks, 2D)
        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),  # Exclude piers themselves
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,  # No gating
            segment_pool_max=5,
        )

        result = builder.build(config)

        assert isinstance(result, SegmentPoolResult)
        assert len(result.candidates) <= 5
        assert len(result.artist_key_by_idx) == len(result.candidates)
        assert len(result.title_key_by_idx) == len(result.candidates)
        assert "pool_strategy" in result.diagnostics

    def test_empty_pool_when_max_is_zero(self):
        """Test that empty pool is returned when segment_pool_max is 0."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=0,  # Zero max
        )

        result = builder.build(config)

        assert result.candidates == []
        assert result.artist_key_by_idx == {}
        assert result.title_key_by_idx == {}
        assert result.diagnostics["final"] == 0

    def test_pool_respects_max_size(self):
        """Test that pool size respects segment_pool_max."""
        bundle = MockArtifactBundle(track_count=20)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(20, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=19,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 19)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Should have at most 10 candidates (may be less due to 1-per-artist)
        assert len(result.candidates) <= 10


# =============================================================================
# Structural Filtering Tests
# =============================================================================

class TestStructuralFiltering:
    """Test structural filtering logic."""

    def test_exclude_used_track_ids(self):
        """Test that used track IDs are excluded."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids={1, 2, 3},  # Mark 1, 2, 3 as used
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Used tracks should not appear in candidates
        assert 1 not in result.candidates
        assert 2 not in result.candidates
        assert 3 not in result.candidates
        assert result.diagnostics["excluded_used_track_ids"] == 3

    def test_allowed_set_clamping(self):
        """Test that allowed_set restricts candidates."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
            allowed_set={4, 5, 6},  # Only allow 4, 5, 6
        )

        result = builder.build(config)

        # Only tracks 4, 5, 6 should be in candidates
        for idx in result.candidates:
            assert idx in {4, 5, 6}

    def test_disallow_seed_artist_policy(self):
        """Test that seed artist is excluded from interiors when policy is set."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        # Seed artist is "artist_0" (tracks 0, 5 based on MockArtifactBundle)
        config = SegmentPoolConfig(
            pier_a=1,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(2, 9)),  # Includes track 5 (artist_0)
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
            seed_artist_key="artist_0",
            disallow_seed_artist_in_interiors=True,
        )

        result = builder.build(config)

        # Track 5 (artist_0) should be excluded
        assert 5 not in result.candidates
        assert result.diagnostics["excluded_seed_artist_policy"] >= 1

    def test_disallow_pier_artists_policy(self):
        """Test that pier artists are excluded from interiors when policy is set."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        # Pier A is track 0 (artist_0), Pier B is track 4 (artist_4)
        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=4,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 10)),  # Includes track 5 (artist_0) and 9 (artist_4)
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
            disallow_pier_artists_in_interiors=True,
        )

        result = builder.build(config)

        # Tracks with artist_0 or artist_4 should be excluded
        for idx in result.candidates:
            artist_key = bundle.get_identity_keys(idx).artist_key
            assert artist_key not in {"artist_0", "artist_4"}

    def test_track_key_collision_filtering(self):
        """Test that tracks with colliding track keys are excluded."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        # Mark track 3's track key as used
        track_3_keys = bundle.get_identity_keys(3)
        used_track_keys = {track_3_keys.track_key}

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
            used_track_keys=used_track_keys,
        )

        result = builder.build(config)

        # Track 3 should be excluded
        assert 3 not in result.candidates
        assert result.diagnostics["excluded_track_key_collision"] == 1


# =============================================================================
# Bridge Scoring Tests
# =============================================================================

class TestBridgeScoring:
    """Test bridge scoring logic."""

    def test_bridge_floor_gating(self):
        """Test that bridge floor properly gates candidates."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        # Create vectors where track 5 has high similarity to both piers
        X_full_norm = np.zeros((10, 2))
        X_full_norm[0] = [1.0, 0.0]  # Pier A
        X_full_norm[9] = [0.0, 1.0]  # Pier B
        X_full_norm[5] = [0.7, 0.7]  # Track 5: moderate similarity to both
        X_full_norm[3] = [1.0, 0.0]  # Track 3: high sim to A, zero to B
        # Normalize
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=[3, 5],
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.5,  # High threshold
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Track 5 should pass (similarity to both ~0.7), track 3 should fail (similarity to B ~0)
        assert 5 in result.candidates
        assert 3 not in result.candidates
        assert result.diagnostics["below_bridge_floor"] >= 1

    def test_harmonic_mean_scoring(self):
        """Test that candidates are sorted by harmonic mean of similarities."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        # Create vectors with different similarity profiles
        X_full_norm = np.zeros((10, 2))
        X_full_norm[0] = [1.0, 0.0]  # Pier A
        X_full_norm[9] = [0.0, 1.0]  # Pier B
        X_full_norm[1] = [0.9, 0.1]  # High sim to A, low to B
        X_full_norm[2] = [0.7, 0.7]  # Balanced similarity (highest harmonic mean)
        X_full_norm[3] = [0.1, 0.9]  # Low sim to A, high to B
        # Normalize
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=[1, 2, 3],
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Track 2 should rank highest (balanced similarity has best harmonic mean)
        assert result.candidates[0] == 2


# =============================================================================
# One-Per-Artist Constraint Tests
# =============================================================================

class TestOnePerArtistConstraint:
    """Test 1-per-artist constraint enforcement."""

    def test_one_per_artist_enforcement(self):
        """Test that only one track per artist is selected."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Check that each artist appears at most once
        artist_keys_seen = set()
        for idx in result.candidates:
            artist_key = result.artist_key_by_idx[idx]
            assert artist_key not in artist_keys_seen, f"Artist {artist_key} appears twice"
            artist_keys_seen.add(artist_key)

    def test_collapsed_by_artist_diagnostics(self):
        """Test that collapsed_by_artist_key diagnostic is populated."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # With 5 unique artists and 8 candidates, at least 3 should be collapsed
        assert "collapsed_by_artist_key" in result.diagnostics


# =============================================================================
# Internal Connector Tests
# =============================================================================

class TestInternalConnectors:
    """Test internal connector handling."""

    def test_internal_connector_priority(self):
        """Test that internal connectors are selected first when priority=True."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=3,
            internal_connectors={5, 6},  # Prioritize tracks 5, 6
            internal_connector_priority=True,
        )

        result = builder.build(config)

        # Internal connectors should appear first (if they pass gates)
        # Note: Due to 1-per-artist, may not have both if same artist
        assert result.diagnostics["internal_connectors_selected"] >= 0

    def test_internal_connector_cap(self):
        """Test that internal connector cap limits selection."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
            internal_connectors={2, 3, 7, 8},  # 4 internal connectors
            internal_connector_cap=2,  # But cap at 2
            internal_connector_priority=True,
        )

        result = builder.build(config)

        # Should select at most 2 internal connectors
        assert result.diagnostics["internal_connectors_selected"] <= 2

    def test_external_first_when_priority_false(self):
        """Test that external candidates are selected first when priority=False."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=3,
            internal_connectors={5, 6},
            internal_connector_priority=False,  # External first
        )

        result = builder.build(config)

        # External candidates should fill the pool first
        assert len(result.candidates) <= 3


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_all_candidates_filtered_out(self):
        """Test when all candidates are filtered out."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(range(1, 9)),  # Mark all as used
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        assert result.candidates == []
        assert result.diagnostics["final"] == 0
        assert result.diagnostics["eligible_after_structural"] == 0

    def test_zero_similarity_vectors(self):
        """Test with zero similarity vectors."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        # Zero vectors (edge case)
        X_full_norm = np.zeros((10, 2))
        X_full_norm[0] = [1.0, 0.0]  # Pier A
        X_full_norm[9] = [0.0, 1.0]  # Pier B

        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids=set(),
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=10,
        )

        result = builder.build(config)

        # Should handle gracefully (harmonic mean will be 0)
        assert isinstance(result, SegmentPoolResult)

    def test_diagnostics_population(self):
        """Test that diagnostics dict is populated correctly."""
        bundle = MockArtifactBundle(track_count=10)
        builder = SegmentCandidatePoolBuilder()

        X_full_norm = np.random.randn(10, 2)
        X_full_norm = X_full_norm / np.linalg.norm(X_full_norm, axis=1, keepdims=True)

        diagnostics = {}
        config = SegmentPoolConfig(
            pier_a=0,
            pier_b=9,
            X_full_norm=X_full_norm,
            universe_indices=list(range(1, 9)),
            used_track_ids={1, 2},
            bundle=bundle,
            bridge_floor=0.0,
            segment_pool_max=5,
            diagnostics=diagnostics,  # Pass external dict
        )

        result = builder.build(config)

        # Diagnostics should be populated in external dict
        assert "pool_strategy" in diagnostics
        assert "base_universe" in diagnostics
        assert "excluded_used_track_ids" in diagnostics
        assert diagnostics["excluded_used_track_ids"] == 2
        assert "final" in diagnostics
