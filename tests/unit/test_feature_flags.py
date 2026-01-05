"""Unit tests for feature flags system.

Tests feature flag infrastructure for safe refactoring migrations (Phase 6.1).

Coverage:
- FeatureFlags initialization
- Individual flag getters
- Utility methods (is_any_enabled, get_active_flags, etc.)
- Default behavior (all flags False)
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.feature_flags import FeatureFlags


# =============================================================================
# FeatureFlags Initialization Tests
# =============================================================================

class TestFeatureFlagsInit:
    """Test FeatureFlags initialization."""

    def test_init_empty_config(self):
        """Test initializing with empty config."""
        flags = FeatureFlags({})

        assert not flags.is_any_enabled()
        assert flags.get_active_flags() == {}

    def test_init_no_experimental_section(self):
        """Test initializing with config missing experimental section."""
        config = {"playlists": {"count": 8}}
        flags = FeatureFlags(config)

        assert not flags.is_any_enabled()
        assert flags.get_active_flags() == {}

    def test_init_with_flags(self):
        """Test initializing with some flags enabled."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_unified_artist_normalization": False,
            }
        }
        flags = FeatureFlags(config)

        assert flags.is_any_enabled()
        assert len(flags.get_active_flags()) == 1
        assert flags.use_unified_genre_normalization() is True
        assert flags.use_unified_artist_normalization() is False


# =============================================================================
# Genre/Artist Normalization Flags Tests
# =============================================================================

class TestNormalizationFlags:
    """Test genre/artist normalization flags."""

    def test_unified_genre_normalization_default(self):
        """Test unified genre normalization defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_unified_genre_normalization() is False

    def test_unified_genre_normalization_enabled(self):
        """Test unified genre normalization when enabled."""
        config = {"experimental": {"use_unified_genre_normalization": True}}
        flags = FeatureFlags(config)

        assert flags.use_unified_genre_normalization() is True

    def test_unified_artist_normalization_default(self):
        """Test unified artist normalization defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_unified_artist_normalization() is False

    def test_unified_artist_normalization_enabled(self):
        """Test unified artist normalization when enabled."""
        config = {"experimental": {"use_unified_artist_normalization": True}}
        flags = FeatureFlags(config)

        assert flags.use_unified_artist_normalization() is True


# =============================================================================
# Similarity Calculation Flags Tests
# =============================================================================

class TestSimilarityFlags:
    """Test similarity calculation flags."""

    def test_unified_genre_similarity_default(self):
        """Test unified genre similarity defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_unified_genre_similarity() is False

    def test_unified_genre_similarity_enabled(self):
        """Test unified genre similarity when enabled."""
        config = {"experimental": {"use_unified_genre_similarity": True}}
        flags = FeatureFlags(config)

        assert flags.use_unified_genre_similarity() is True


# =============================================================================
# Pier-Bridge Refactoring Flags Tests
# =============================================================================

class TestPierBridgeFlags:
    """Test pier-bridge refactoring flags."""

    def test_extracted_pier_bridge_scoring_default(self):
        """Test extracted pier-bridge scoring defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_extracted_pier_bridge_scoring() is False

    def test_extracted_pier_bridge_scoring_enabled(self):
        """Test extracted pier-bridge scoring when enabled."""
        config = {"experimental": {"use_extracted_pier_bridge_scoring": True}}
        flags = FeatureFlags(config)

        assert flags.use_extracted_pier_bridge_scoring() is True

    def test_extracted_segment_pool_default(self):
        """Test extracted segment pool defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_extracted_segment_pool() is False

    def test_extracted_segment_pool_enabled(self):
        """Test extracted segment pool when enabled."""
        config = {"experimental": {"use_extracted_segment_pool": True}}
        flags = FeatureFlags(config)

        assert flags.use_extracted_segment_pool() is True

    def test_extracted_pier_bridge_diagnostics_default(self):
        """Test extracted pier-bridge diagnostics defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_extracted_pier_bridge_diagnostics() is False

    def test_extracted_pier_bridge_diagnostics_enabled(self):
        """Test extracted pier-bridge diagnostics when enabled."""
        config = {"experimental": {"use_extracted_pier_bridge_diagnostics": True}}
        flags = FeatureFlags(config)

        assert flags.use_extracted_pier_bridge_diagnostics() is True


# =============================================================================
# Playlist Generation Flags Tests
# =============================================================================

class TestPlaylistGenerationFlags:
    """Test playlist generation flags."""

    def test_new_candidate_generator_default(self):
        """Test new candidate generator defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_new_candidate_generator() is False

    def test_new_candidate_generator_enabled(self):
        """Test new candidate generator when enabled."""
        config = {"experimental": {"use_new_candidate_generator": True}}
        flags = FeatureFlags(config)

        assert flags.use_new_candidate_generator() is True

    def test_playlist_factory_default(self):
        """Test playlist factory defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_playlist_factory() is False

    def test_playlist_factory_enabled(self):
        """Test playlist factory when enabled."""
        config = {"experimental": {"use_playlist_factory": True}}
        flags = FeatureFlags(config)

        assert flags.use_playlist_factory() is True

    def test_filtering_pipeline_default(self):
        """Test filtering pipeline defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_filtering_pipeline() is False

    def test_filtering_pipeline_enabled(self):
        """Test filtering pipeline when enabled."""
        config = {"experimental": {"use_filtering_pipeline": True}}
        flags = FeatureFlags(config)

        assert flags.use_filtering_pipeline() is True

    def test_history_repository_default(self):
        """Test history repository defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_history_repository() is False

    def test_history_repository_enabled(self):
        """Test history repository when enabled."""
        config = {"experimental": {"use_history_repository": True}}
        flags = FeatureFlags(config)

        assert flags.use_history_repository() is True


# =============================================================================
# Pipeline Refactoring Flags Tests
# =============================================================================

class TestPipelineFlags:
    """Test pipeline refactoring flags."""

    def test_config_resolver_default(self):
        """Test config resolver defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_config_resolver() is False

    def test_config_resolver_enabled(self):
        """Test config resolver when enabled."""
        config = {"experimental": {"use_config_resolver": True}}
        flags = FeatureFlags(config)

        assert flags.use_config_resolver() is True

    def test_pipeline_builder_default(self):
        """Test pipeline builder defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_pipeline_builder() is False

    def test_pipeline_builder_enabled(self):
        """Test pipeline builder when enabled."""
        config = {"experimental": {"use_pipeline_builder": True}}
        flags = FeatureFlags(config)

        assert flags.use_pipeline_builder() is True

    def test_variant_cache_default(self):
        """Test variant cache defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_variant_cache() is False

    def test_variant_cache_enabled(self):
        """Test variant cache when enabled."""
        config = {"experimental": {"use_variant_cache": True}}
        flags = FeatureFlags(config)

        assert flags.use_variant_cache() is True


# =============================================================================
# Configuration Facades Flags Tests
# =============================================================================

class TestConfigFacadesFlags:
    """Test configuration facades flags."""

    def test_typed_config_default(self):
        """Test typed config defaults to False."""
        flags = FeatureFlags({})

        assert flags.use_typed_config() is False

    def test_typed_config_enabled(self):
        """Test typed config when enabled."""
        config = {"experimental": {"use_typed_config": True}}
        flags = FeatureFlags(config)

        assert flags.use_typed_config() is True


# =============================================================================
# Utility Methods Tests
# =============================================================================

class TestUtilityMethods:
    """Test utility methods."""

    def test_is_any_enabled_none(self):
        """Test is_any_enabled when no flags enabled."""
        flags = FeatureFlags({})

        assert flags.is_any_enabled() is False

    def test_is_any_enabled_some(self):
        """Test is_any_enabled when some flags enabled."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_playlist_factory": False,
            }
        }
        flags = FeatureFlags(config)

        assert flags.is_any_enabled() is True

    def test_get_active_flags_none(self):
        """Test get_active_flags when none active."""
        flags = FeatureFlags({})

        active = flags.get_active_flags()

        assert active == {}

    def test_get_active_flags_some(self):
        """Test get_active_flags when some active."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_playlist_factory": True,
                "use_config_resolver": False,
            }
        }
        flags = FeatureFlags(config)

        active = flags.get_active_flags()

        assert len(active) == 2
        assert active["use_unified_genre_normalization"] is True
        assert active["use_playlist_factory"] is True
        assert "use_config_resolver" not in active

    def test_get_all_flags_empty(self):
        """Test get_all_flags with empty experimental section."""
        flags = FeatureFlags({})

        all_flags = flags.get_all_flags()

        assert all_flags == {}

    def test_get_all_flags_some(self):
        """Test get_all_flags with some flags."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_playlist_factory": False,
            }
        }
        flags = FeatureFlags(config)

        all_flags = flags.get_all_flags()

        assert len(all_flags) == 2
        assert all_flags["use_unified_genre_normalization"] is True
        assert all_flags["use_playlist_factory"] is False

    def test_repr_no_flags(self):
        """Test __repr__ with no active flags."""
        flags = FeatureFlags({})

        repr_str = repr(flags)

        assert "no active flags" in repr_str

    def test_repr_with_flags(self):
        """Test __repr__ with active flags."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_playlist_factory": True,
            }
        }
        flags = FeatureFlags(config)

        repr_str = repr(flags)

        assert "active:" in repr_str
        assert "use_unified_genre_normalization" in repr_str
        assert "use_playlist_factory" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================

class TestFeatureFlagsIntegration:
    """Test feature flags integration scenarios."""

    def test_all_flags_disabled_by_default(self):
        """Test that all flags default to disabled (safe behavior)."""
        flags = FeatureFlags({})

        # Normalization
        assert flags.use_unified_genre_normalization() is False
        assert flags.use_unified_artist_normalization() is False

        # Similarity
        assert flags.use_unified_genre_similarity() is False

        # Pier-bridge
        assert flags.use_extracted_pier_bridge_scoring() is False
        assert flags.use_extracted_segment_pool() is False
        assert flags.use_extracted_pier_bridge_diagnostics() is False

        # Playlist generation
        assert flags.use_new_candidate_generator() is False
        assert flags.use_playlist_factory() is False
        assert flags.use_filtering_pipeline() is False
        assert flags.use_history_repository() is False

        # Pipeline
        assert flags.use_config_resolver() is False
        assert flags.use_pipeline_builder() is False
        assert flags.use_variant_cache() is False

        # Config
        assert flags.use_typed_config() is False

    def test_gradual_migration_scenario(self):
        """Test gradual migration scenario (enable flags one by one)."""
        # Start with one flag
        config = {"experimental": {"use_unified_genre_normalization": True}}
        flags = FeatureFlags(config)

        assert flags.is_any_enabled()
        assert len(flags.get_active_flags()) == 1

        # Enable another flag
        config["experimental"]["use_config_resolver"] = True
        flags = FeatureFlags(config)

        assert len(flags.get_active_flags()) == 2
        assert flags.use_unified_genre_normalization() is True
        assert flags.use_config_resolver() is True

    def test_full_migration_all_flags_enabled(self):
        """Test full migration with all flags enabled."""
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_unified_artist_normalization": True,
                "use_unified_genre_similarity": True,
                "use_extracted_pier_bridge_scoring": True,
                "use_extracted_segment_pool": True,
                "use_extracted_pier_bridge_diagnostics": True,
                "use_new_candidate_generator": True,
                "use_playlist_factory": True,
                "use_filtering_pipeline": True,
                "use_history_repository": True,
                "use_config_resolver": True,
                "use_pipeline_builder": True,
                "use_variant_cache": True,
                "use_typed_config": True,
            }
        }
        flags = FeatureFlags(config)

        assert flags.is_any_enabled()
        assert len(flags.get_active_flags()) == 14

        # All flags should be enabled
        assert flags.use_unified_genre_normalization() is True
        assert flags.use_unified_artist_normalization() is True
        assert flags.use_unified_genre_similarity() is True
        assert flags.use_extracted_pier_bridge_scoring() is True
        assert flags.use_extracted_segment_pool() is True
        assert flags.use_extracted_pier_bridge_diagnostics() is True
        assert flags.use_new_candidate_generator() is True
        assert flags.use_playlist_factory() is True
        assert flags.use_filtering_pipeline() is True
        assert flags.use_history_repository() is True
        assert flags.use_config_resolver() is True
        assert flags.use_pipeline_builder() is True
        assert flags.use_variant_cache() is True
        assert flags.use_typed_config() is True
