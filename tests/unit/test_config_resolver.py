"""Unit tests for configuration resolver.

Tests config resolution logic extracted from pipeline.py (Phase 5.1).

Coverage:
- ConfigSource creation
- ConfigResolver precedence rules
- Multi-source resolution
- Hybrid weight resolution
- Pipeline config builder
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.config_resolver import (
    ConfigSource,
    ConfigResolver,
    build_pipeline_config_resolver,
    resolve_hybrid_weights,
)


# =============================================================================
# ConfigSource Tests
# =============================================================================

class TestConfigSource:
    """Test ConfigSource dataclass."""

    def test_create_config_source(self):
        """Test creating config source."""
        source = ConfigSource(
            name="test",
            get_value=lambda k: {"foo": "bar"}.get(k),
            priority=1,
        )

        assert source.name == "test"
        assert source.get_value("foo") == "bar"
        assert source.get_value("missing") is None
        assert source.priority == 1


# =============================================================================
# ConfigResolver Tests
# =============================================================================

class TestConfigResolver:
    """Test ConfigResolver."""

    def test_create_empty_resolver(self):
        """Test creating empty resolver."""
        resolver = ConfigResolver()

        assert len(resolver.sources) == 0

    def test_add_single_source(self):
        """Test adding single config source."""
        resolver = ConfigResolver()
        resolver.add_source("test", lambda k: {"foo": "bar"}.get(k), priority=1)

        assert len(resolver.sources) == 1
        assert resolver.sources[0].name == "test"

    def test_resolve_from_single_source(self):
        """Test resolving value from single source."""
        resolver = ConfigResolver()
        resolver.add_source("test", lambda k: {"sonic_weight": 0.7}.get(k), priority=1)

        value = resolver.resolve("sonic_weight")

        assert value == 0.7

    def test_resolve_with_default(self):
        """Test resolving with default value."""
        resolver = ConfigResolver()
        resolver.add_source("test", lambda k: {}.get(k), priority=1)

        value = resolver.resolve("missing_key", default=0.5)

        assert value == 0.5

    def test_resolve_required_missing(self):
        """Test resolving required key that's missing."""
        resolver = ConfigResolver()
        resolver.add_source("test", lambda k: {}.get(k), priority=1)

        with pytest.raises(ValueError, match="Required config key"):
            resolver.resolve("missing_key", required=True)

    def test_precedence_lower_priority_wins(self):
        """Test that lower priority number takes precedence."""
        resolver = ConfigResolver()
        resolver.add_source("low_priority", lambda k: {"key": "low"}.get(k), priority=2)
        resolver.add_source("high_priority", lambda k: {"key": "high"}.get(k), priority=0)

        value = resolver.resolve("key")

        assert value == "high"  # Priority 0 beats priority 2

    def test_precedence_ordering(self):
        """Test complete precedence ordering."""
        resolver = ConfigResolver()
        resolver.add_source("runtime", lambda k: {"weight": 0.9}.get(k), priority=0)
        resolver.add_source("mode", lambda k: {"weight": 0.8}.get(k), priority=1)
        resolver.add_source("base", lambda k: {"weight": 0.7}.get(k), priority=2)

        value = resolver.resolve("weight")

        assert value == 0.9  # Runtime wins

    def test_fallback_through_chain(self):
        """Test falling back through source chain."""
        resolver = ConfigResolver()
        resolver.add_source("runtime", lambda k: {}.get(k), priority=0)
        resolver.add_source("mode", lambda k: {"weight": 0.8}.get(k), priority=1)
        resolver.add_source("base", lambda k: {}.get(k), priority=2)

        value = resolver.resolve("weight")

        assert value == 0.8  # Falls through to mode

    def test_chaining_add_source(self):
        """Test chaining add_source calls."""
        resolver = (
            ConfigResolver()
            .add_source("a", lambda k: {}.get(k), priority=0)
            .add_source("b", lambda k: {}.get(k), priority=1)
        )

        assert len(resolver.sources) == 2

    def test_resolve_many(self):
        """Test resolving multiple keys."""
        resolver = ConfigResolver()
        resolver.add_source(
            "test",
            lambda k: {"sonic_weight": 0.7, "genre_weight": 0.3}.get(k),
            priority=1,
        )

        result = resolver.resolve_many(
            ["sonic_weight", "genre_weight", "missing"],
            defaults={"missing": 0.5},
        )

        assert result["sonic_weight"] == 0.7
        assert result["genre_weight"] == 0.3
        assert result["missing"] == 0.5

    def test_clear(self):
        """Test clearing all sources."""
        resolver = ConfigResolver()
        resolver.add_source("test", lambda k: {}.get(k), priority=1)

        assert len(resolver.sources) == 1

        resolver.clear()

        assert len(resolver.sources) == 0


# =============================================================================
# Pipeline Config Builder Tests
# =============================================================================

class TestPipelineConfigBuilder:
    """Test build_pipeline_config_resolver."""

    def test_build_with_overrides_only(self):
        """Test building with only overrides."""
        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            overrides={"sonic_weight": 0.9},
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.9

    def test_build_with_mode_config(self):
        """Test building with mode-specific config."""
        base_config = {
            "dynamic": {"sonic_weight": 0.8},
            "narrow": {"sonic_weight": 0.9},
        }

        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            base_config=base_config,
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.8

    def test_build_with_base_config(self):
        """Test building with base config."""
        base_config = {"sonic_weight": 0.7}

        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            base_config=base_config,
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.7

    def test_build_full_precedence_chain(self):
        """Test complete precedence chain."""
        base_config = {
            "sonic_weight": 0.5,  # Lowest priority
            "dynamic": {
                "sonic_weight": 0.7,  # Middle priority
            },
        }

        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            overrides={"sonic_weight": 0.9},  # Highest priority
            base_config=base_config,
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.9  # Runtime overrides win

    def test_fallback_to_mode_config(self):
        """Test falling back to mode config when override not present."""
        base_config = {
            "sonic_weight": 0.5,
            "dynamic": {
                "sonic_weight": 0.7,
            },
        }

        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            overrides={},  # Empty overrides
            base_config=base_config,
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.7  # Mode config wins

    def test_fallback_to_base_config(self):
        """Test falling back to base config."""
        base_config = {
            "sonic_weight": 0.5,
            "dynamic": {},  # Empty mode config
        }

        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            overrides={},
            base_config=base_config,
        )

        value = resolver.resolve("sonic_weight")

        assert value == 0.5  # Base config wins

    def test_different_modes_different_values(self):
        """Test that different modes resolve different values."""
        base_config = {
            "dynamic": {"sonic_weight": 0.7},
            "narrow": {"sonic_weight": 0.9},
        }

        resolver_dynamic = build_pipeline_config_resolver(
            mode="dynamic",
            base_config=base_config,
        )

        resolver_narrow = build_pipeline_config_resolver(
            mode="narrow",
            base_config=base_config,
        )

        assert resolver_dynamic.resolve("sonic_weight") == 0.7
        assert resolver_narrow.resolve("sonic_weight") == 0.9


# =============================================================================
# Hybrid Weight Resolution Tests
# =============================================================================

class TestHybridWeightResolution:
    """Test resolve_hybrid_weights."""

    def test_explicit_weights(self):
        """Test with explicit sonic and genre weights."""
        sonic, genre = resolve_hybrid_weights(
            sonic_weight=0.8,
            genre_weight=0.2,
        )

        assert sonic == 0.8
        assert genre == 0.2

    def test_explicit_sonic_only(self):
        """Test with only sonic weight (genre auto-complemented)."""
        sonic, genre = resolve_hybrid_weights(
            sonic_weight=0.8,
        )

        assert sonic == 0.8
        assert abs(genre - 0.2) < 0.01  # Auto: 1.0 - 0.8 (allow floating-point tolerance)

    def test_mode_defaults_dynamic(self):
        """Test default weights for dynamic mode."""
        sonic, genre = resolve_hybrid_weights(mode="dynamic")

        assert sonic == 0.7
        assert genre == 0.3

    def test_mode_defaults_narrow(self):
        """Test default weights for narrow mode."""
        sonic, genre = resolve_hybrid_weights(mode="narrow")

        assert sonic == 0.8
        assert genre == 0.2

    def test_mode_defaults_sonic_only(self):
        """Test default weights for sonic_only mode."""
        sonic, genre = resolve_hybrid_weights(mode="sonic_only")

        assert sonic == 1.0
        assert genre == 0.0

    def test_overrides_precedence(self):
        """Test that explicit weights beat mode defaults."""
        sonic, genre = resolve_hybrid_weights(
            sonic_weight=0.6,
            mode="narrow",  # Would default to 0.8
        )

        assert sonic == 0.6
        assert genre == 0.4  # Auto-complemented

    def test_normalization_when_not_summing_to_one(self):
        """Test normalization when weights don't sum to 1.0."""
        sonic, genre = resolve_hybrid_weights(
            sonic_weight=0.6,
            genre_weight=0.6,  # Sum = 1.2
        )

        assert abs(sonic - 0.5) < 0.01  # 0.6 / 1.2 = 0.5
        assert abs(genre - 0.5) < 0.01  # 0.6 / 1.2 = 0.5
        assert abs((sonic + genre) - 1.0) < 0.01

    def test_overrides_dict(self):
        """Test resolving from overrides dictionary."""
        overrides = {"sonic_weight": 0.75}

        sonic, genre = resolve_hybrid_weights(
            overrides=overrides,
        )

        assert sonic == 0.75
        assert genre == 0.25  # Auto-complemented
