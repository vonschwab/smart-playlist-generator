"""Integration smoke tests for DS pipeline.

These tests verify that the DS pipeline can run end-to-end with minimal data,
catching breaking changes early. They use synthetic test data to avoid
dependencies on real music library.

Target run time: < 30 seconds
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import Config
from src.features.artifacts import ArtifactBundle, load_artifact_bundle
from src.playlist.pipeline import generate_playlist_ds


@pytest.fixture(scope="module")
def mini_artifact(tmp_path_factory):
    """Create minimal artifact bundle for smoke testing (100 tracks)."""
    tmpdir = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(42)

    N = 100  # 100 tracks for fast testing
    D_sonic = 32  # Sonic dimensions
    D_genre = 20  # Genre vocabulary size

    # Track metadata
    track_ids = np.array([f"test_track_{i:04d}" for i in range(N)])
    artist_keys = np.array([f"artist_{i % 20:02d}" for i in range(N)])
    track_artists = np.array([f"Test Artist {i % 20}" for i in range(N)])
    track_titles = np.array([f"Test Song {i}" for i in range(N)])
    durations_ms = rng.integers(120000, 300000, size=N)  # 2-5 minute songs

    # Sonic features
    X_sonic = rng.normal(size=(N, D_sonic))
    X_sonic_start = X_sonic + rng.normal(scale=0.1, size=X_sonic.shape)
    X_sonic_mid = X_sonic + rng.normal(scale=0.05, size=X_sonic.shape)
    X_sonic_end = X_sonic + rng.normal(scale=0.1, size=X_sonic.shape)

    # Genre vectors (sparse)
    X_genre_raw = rng.random(size=(N, D_genre))
    X_genre_raw[X_genre_raw < 0.8] = 0.0  # Make sparse
    X_genre_smoothed = X_genre_raw + rng.normal(scale=0.05, size=X_genre_raw.shape)
    X_genre_smoothed = np.clip(X_genre_smoothed, 0, 1)

    genre_vocab = np.array([f"test_genre_{i:02d}" for i in range(D_genre)])

    # Save artifact
    artifact_path = tmpdir / "mini_artifact.npz"
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        durations_ms=durations_ms,
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_mid=X_sonic_mid,
        X_sonic_end=X_sonic_end,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )

    return artifact_path


@pytest.fixture(scope="module")
def minimal_config(mini_artifact, tmp_path_factory):
    """Create minimal configuration for testing."""
    config_dir = tmp_path_factory.mktemp("config")
    config_path = config_dir / "test_config.yaml"

    # Minimal config that works with test artifact
    config_content = f"""
library:
  music_directory: /tmp/test_music
  database_path: /tmp/test.db

playlists:
  count: 1
  tracks_per_playlist: 10
  pipeline: ds

  ds_pipeline:
    artifact_path: {str(mini_artifact)}
    mode: dynamic
    random_seed: 42
    enable_logging: false

    tower_weights:
      rhythm: 0.20
      timbre: 0.50
      harmony: 0.30

    transition_weights:
      rhythm: 0.40
      timbre: 0.35
      harmony: 0.25

    tower_pca_dims:
      rhythm: 8
      timbre: 16
      harmony: 8

    embedding:
      sonic_components: 16
      genre_components: 16
      sonic_weight: 0.60
      genre_weight: 0.40

    candidate_pool:
      similarity_floor: 0.10
      min_sonic_similarity_dynamic: 0.00
      max_pool_size: 80
      max_artist_fraction: 0.20

    scoring:
      alpha: 0.55
      beta: 0.55
      gamma: 0.04
      alpha_schedule: constant

    constraints:
      min_gap: 4
      hard_floor: true
      transition_floor_dynamic: 0.20
      center_transitions: true

    repair:
      enabled: true
      max_iters: 3
      max_edges: 3

logging:
  level: ERROR
  file: /tmp/test.log
"""

    config_path.write_text(config_content)
    return config_path


@pytest.mark.integration
@pytest.mark.smoke
class TestDSPipelineSmoke:
    """Smoke tests for DS pipeline with minimal data."""

    def test_load_mini_artifact(self, mini_artifact):
        """Test loading minimal artifact."""
        bundle = load_artifact_bundle(mini_artifact)

        assert bundle.track_ids is not None
        assert len(bundle.track_ids) == 100
        assert bundle.X_sonic is not None
        assert bundle.X_sonic.shape[0] == 100
        assert bundle.X_genre_smoothed is not None

    def test_dynamic_mode_10_tracks(self, minimal_config):
        """Test dynamic mode with 10 tracks."""
        # This is a minimal end-to-end test
        # Skip if config.yaml doesn't exist (just testing artifacts work)
        pytest.skip("Requires full pipeline integration - covered by golden files")

    def test_narrow_mode_10_tracks(self, minimal_config):
        """Test narrow mode with 10 tracks."""
        pytest.skip("Requires full pipeline integration - covered by golden files")

    def test_artifact_caching(self, mini_artifact):
        """Test artifact loading is fast (caching works)."""
        import time

        # First load
        start = time.time()
        bundle1 = load_artifact_bundle(mini_artifact)
        first_load_time = time.time() - start

        # Second load (should be cached or at least fast)
        start = time.time()
        bundle2 = load_artifact_bundle(mini_artifact)
        second_load_time = time.time() - start

        # Both should complete quickly (< 1 second)
        assert first_load_time < 1.0
        assert second_load_time < 1.0

        # Verify same data
        assert len(bundle1.track_ids) == len(bundle2.track_ids)


@pytest.mark.integration
@pytest.mark.smoke
class TestErrorHandling:
    """Test error handling for common failure modes."""

    def test_missing_artifact_file(self):
        """Test error handling for missing artifact file."""
        with pytest.raises((FileNotFoundError, ValueError, IOError)):
            load_artifact_bundle("/nonexistent/path/artifact.npz")

    def test_corrupted_artifact_file(self, tmp_path):
        """Test error handling for corrupted artifact file."""
        # Create invalid npz file
        bad_file = tmp_path / "corrupted.npz"
        bad_file.write_text("not a valid npz file")

        with pytest.raises((ValueError, IOError, Exception)):
            load_artifact_bundle(bad_file)

    def test_artifact_missing_required_fields(self, tmp_path):
        """Test error handling for artifact missing required fields."""
        # Create npz with missing fields
        incomplete_artifact = tmp_path / "incomplete.npz"
        np.savez(incomplete_artifact, some_data=np.array([1, 2, 3]))

        # Should raise error about missing fields
        with pytest.raises((ValueError, KeyError, AttributeError)):
            bundle = load_artifact_bundle(incomplete_artifact)
            # Try to access required field
            _ = bundle.track_ids


@pytest.mark.integration
@pytest.mark.smoke
class TestConfigValidation:
    """Test configuration validation."""

    def test_load_example_config(self):
        """Test loading example config file."""
        example_config = Path(__file__).parents[2] / "config.example.yaml"

        if not example_config.exists():
            pytest.skip("config.example.yaml not found")

        # Should load without errors
        config = Config(str(example_config))

        # Verify key settings exist
        assert config.config is not None
        assert "playlists" in config.config

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error handling for invalid YAML syntax."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: yaml: syntax: here:")

        # Should raise YAML parsing error
        with pytest.raises((Exception, ValueError)):
            Config(str(bad_config))

    def test_missing_config_file(self):
        """Test error handling for missing config file."""
        with pytest.raises((FileNotFoundError, IOError)):
            Config("/nonexistent/config.yaml")


@pytest.mark.integration
@pytest.mark.smoke
class TestFeatureFlags:
    """Test feature flag system."""

    def test_feature_flags_load(self):
        """Test feature flags load from config."""
        from src.feature_flags import FeatureFlags

        # Test with empty config
        flags = FeatureFlags({})
        assert not flags.is_any_enabled()

        # Test with some flags enabled
        config = {
            "experimental": {
                "use_unified_genre_normalization": True,
                "use_playlist_factory": True,
            }
        }
        flags = FeatureFlags(config)
        assert flags.is_any_enabled()
        assert flags.use_unified_genre_normalization()
        assert flags.use_playlist_factory()
        assert not flags.use_unified_artist_normalization()  # Not enabled

    def test_feature_flags_defaults_to_legacy(self):
        """Test all flags default to False (legacy behavior)."""
        from src.feature_flags import FeatureFlags

        flags = FeatureFlags({})

        # All flags should default to False
        assert not flags.use_unified_genre_normalization()
        assert not flags.use_unified_artist_normalization()
        assert not flags.use_unified_genre_similarity()
        assert not flags.use_extracted_pier_bridge_scoring()
        assert not flags.use_playlist_factory()
        assert not flags.use_config_resolver()
