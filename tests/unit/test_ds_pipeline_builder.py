"""Unit tests for DS Pipeline Builder.

Tests builder pattern infrastructure extracted from pipeline.py (Phase 5.2).

Coverage:
- DSPipelineRequest dataclass
- DSPipelineBuilder fluent API
- Builder validation
- Parameter chaining
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.ds_pipeline_builder import (
    DSPipelineBuilder,
    DSPipelineRequest,
)
from src.playlist.pier_bridge_builder import PierBridgeConfig


# =============================================================================
# DSPipelineRequest Tests
# =============================================================================

class TestDSPipelineRequest:
    """Test DSPipelineRequest dataclass."""

    def test_create_minimal_request(self):
        """Test creating request with minimal required parameters."""
        request = DSPipelineRequest(
            artifact_path="data/artifacts/test.npz",
            seed_track_id="track_001",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
        )

        assert request.artifact_path == "data/artifacts/test.npz"
        assert request.seed_track_id == "track_001"
        assert request.num_tracks == 30
        assert request.mode == "dynamic"
        assert request.random_seed == 42

        # Optional parameters should be None/default
        assert request.overrides is None
        assert request.allowed_track_ids is None
        assert request.excluded_track_ids is None
        assert request.single_artist is False
        assert request.sonic_variant is None

    def test_create_request_with_hybrid_weights(self):
        """Test creating request with hybrid weights."""
        request = DSPipelineRequest(
            artifact_path="data/artifacts/test.npz",
            seed_track_id="track_001",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
            sonic_weight=0.8,
            genre_weight=0.2,
        )

        assert request.sonic_weight == 0.8
        assert request.genre_weight == 0.2

    def test_create_request_with_anchor_seeds(self):
        """Test creating request with anchor seeds."""
        request = DSPipelineRequest(
            artifact_path="data/artifacts/test.npz",
            seed_track_id="track_001",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
            anchor_seed_ids=["track_002", "track_003"],
        )

        assert request.anchor_seed_ids == ["track_002", "track_003"]

    def test_create_request_with_exclusions(self):
        """Test creating request with excluded tracks."""
        excluded = {"track_100", "track_101"}
        request = DSPipelineRequest(
            artifact_path="data/artifacts/test.npz",
            seed_track_id="track_001",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
            excluded_track_ids=excluded,
        )

        assert request.excluded_track_ids == excluded


# =============================================================================
# DSPipelineBuilder Tests
# =============================================================================

class TestDSPipelineBuilder:
    """Test DSPipelineBuilder."""

    def test_create_builder(self):
        """Test creating empty builder."""
        builder = DSPipelineBuilder()

        assert builder._artifact_path is None
        assert builder._seed_track_id is None
        assert builder._num_tracks == 30  # Default
        assert builder._mode == "dynamic"  # Default
        assert builder._random_seed == 42  # Default

    def test_build_minimal_request(self):
        """Test building minimal request."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .build()
        )

        assert request.artifact_path == "data/artifacts/test.npz"
        assert request.seed_track_id == "track_001"
        assert request.num_tracks == 30
        assert request.mode == "dynamic"
        assert request.random_seed == 42

    def test_build_with_custom_parameters(self):
        """Test building with custom parameters."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_num_tracks(50)
            .with_mode("narrow")
            .with_random_seed(123)
            .build()
        )

        assert request.num_tracks == 50
        assert request.mode == "narrow"
        assert request.random_seed == 123

    def test_build_with_hybrid_weights(self):
        """Test building with hybrid weights."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_sonic_weight(0.8)
            .with_genre_weight(0.2)
            .build()
        )

        assert request.sonic_weight == 0.8
        assert request.genre_weight == 0.2

    def test_build_with_genre_config(self):
        """Test building with genre configuration."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_min_genre_similarity(0.25)
            .with_genre_method("jaccard")
            .build()
        )

        assert request.min_genre_similarity == 0.25
        assert request.genre_method == "jaccard"

    def test_build_with_anchor_seeds(self):
        """Test building with anchor seeds."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_anchor_seeds(["track_002", "track_003", "track_004"])
            .build()
        )

        assert request.anchor_seed_ids == ["track_002", "track_003", "track_004"]

    def test_build_with_allowed_tracks(self):
        """Test building with allowed track IDs."""
        allowed = ["track_001", "track_002", "track_003"]
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_allowed_tracks(allowed)
            .build()
        )

        assert request.allowed_track_ids == allowed
        assert request.allowed_track_ids_set == set(allowed)

    def test_build_with_excluded_tracks(self):
        """Test building with excluded track IDs."""
        excluded = {"track_100", "track_101"}
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_excluded_tracks(excluded)
            .build()
        )

        assert request.excluded_track_ids == excluded

    def test_build_with_single_artist(self):
        """Test building with single-artist constraint."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_single_artist(True)
            .build()
        )

        assert request.single_artist is True

    def test_build_with_sonic_variant(self):
        """Test building with sonic variant."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_sonic_variant("tower_pca")
            .build()
        )

        assert request.sonic_variant == "tower_pca"

    def test_build_with_overrides(self):
        """Test building with configuration overrides."""
        overrides = {"candidate": {"max_pool_size": 5000}}
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_overrides(overrides)
            .build()
        )

        assert request.overrides == overrides

    def test_build_with_internal_connectors(self):
        """Test building with internal connectors."""
        connectors = ["connector_001", "connector_002"]
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_internal_connectors(
                connector_ids=connectors,
                max_per_segment=2,
                priority=False
            )
            .build()
        )

        assert request.internal_connector_ids == connectors
        assert request.internal_connector_max_per_segment == 2
        assert request.internal_connector_priority is False

    def test_build_with_audit_config(self):
        """Test building with audit configuration."""
        audit_context = {"source": "test", "user": "tester"}
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_dry_run(True)
            .with_artist_style(True)
            .with_artist_playlist(True)
            .with_audit_context(audit_context)
            .build()
        )

        assert request.dry_run is True
        assert request.artist_style_enabled is True
        assert request.artist_playlist is True
        assert request.audit_context_extra == audit_context

    def test_build_with_pier_bridge_config(self):
        """Test building with pier-bridge configuration."""
        pb_config = PierBridgeConfig(
            bridge_floor=0.05,
            transition_floor=0.40,
        )
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_pier_bridge_config(pb_config)
            .build()
        )

        assert request.pier_bridge_config == pb_config
        assert request.pier_bridge_config.bridge_floor == 0.05
        assert request.pier_bridge_config.transition_floor == 0.40

    def test_build_missing_artifact_path(self):
        """Test build fails without artifact path."""
        builder = DSPipelineBuilder().with_seed("track_001")

        with pytest.raises(ValueError, match="Artifact path is required"):
            builder.build()

    def test_build_missing_seed(self):
        """Test build fails without seed track ID."""
        builder = DSPipelineBuilder().with_artifacts("data/artifacts/test.npz")

        with pytest.raises(ValueError, match="Seed track ID is required"):
            builder.build()

    def test_chaining_order_independence(self):
        """Test that chaining order doesn't matter."""
        request1 = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_num_tracks(50)
            .with_mode("narrow")
            .build()
        )

        request2 = (
            DSPipelineBuilder()
            .with_mode("narrow")
            .with_num_tracks(50)
            .with_seed("track_001")
            .with_artifacts("data/artifacts/test.npz")
            .build()
        )

        assert request1.artifact_path == request2.artifact_path
        assert request1.seed_track_id == request2.seed_track_id
        assert request1.num_tracks == request2.num_tracks
        assert request1.mode == request2.mode


# =============================================================================
# Integration Tests
# =============================================================================

class TestBuilderIntegration:
    """Test builder integration with pipeline."""

    def test_build_request_to_dict(self):
        """Test converting request to dict for function call."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/test.npz")
            .with_seed("track_001")
            .with_num_tracks(30)
            .with_mode("dynamic")
            .with_random_seed(42)
            .build()
        )

        # Should be convertible to dict for **kwargs
        request_dict = request.__dict__

        assert request_dict["artifact_path"] == "data/artifacts/test.npz"
        assert request_dict["seed_track_id"] == "track_001"
        assert request_dict["num_tracks"] == 30
        assert request_dict["mode"] == "dynamic"
        assert request_dict["random_seed"] == 42

    def test_complex_builder_scenario(self):
        """Test complex builder scenario with many parameters."""
        request = (
            DSPipelineBuilder()
            .with_artifacts("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
            .with_seed("d97b2f9e9f9c6c56e09135ecf9c30876")
            .with_num_tracks(30)
            .with_mode("dynamic")
            .with_random_seed(42)
            .with_sonic_weight(0.8)
            .with_genre_weight(0.2)
            .with_min_genre_similarity(0.25)
            .with_anchor_seeds([
                "abc123",
                "def456",
                "ghi789"
            ])
            .with_excluded_tracks({
                "excluded_001",
                "excluded_002"
            })
            .with_sonic_variant("tower_pca")
            .with_internal_connectors(
                connector_ids=["connector_001"],
                max_per_segment=1,
                priority=True
            )
            .with_dry_run(False)
            .build()
        )

        # Verify all parameters set correctly
        assert request.artifact_path == "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
        assert request.seed_track_id == "d97b2f9e9f9c6c56e09135ecf9c30876"
        assert request.num_tracks == 30
        assert request.mode == "dynamic"
        assert request.random_seed == 42
        assert request.sonic_weight == 0.8
        assert request.genre_weight == 0.2
        assert request.min_genre_similarity == 0.25
        assert len(request.anchor_seed_ids) == 3
        assert len(request.excluded_track_ids) == 2
        assert request.sonic_variant == "tower_pca"
        assert len(request.internal_connector_ids) == 1
        assert request.internal_connector_max_per_segment == 1
        assert request.dry_run is False
