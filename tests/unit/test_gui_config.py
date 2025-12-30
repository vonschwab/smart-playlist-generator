"""
Tests for GUI configuration model and normalization logic.

Run with: pytest tests/unit/test_gui_config.py -v
"""
import pytest
import tempfile
import os
from pathlib import Path

import yaml


class TestConfigModel:
    """Tests for ConfigModel class."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_data = {
            "library": {
                "database_path": "data/metadata.db",
                "music_directory": "E:\\MUSIC"
            },
            "openai": {
                "api_key": "test-key"
            },
            "playlists": {
                "count": 8,
                "tracks_per_playlist": 30,
                "ds_pipeline": {
                    "tower_weights": {
                        "rhythm": 0.20,
                        "timbre": 0.50,
                        "harmony": 0.30
                    },
                    "embedding": {
                        "sonic_weight": 0.60,
                        "genre_weight": 0.40
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    def test_load_config(self, temp_config):
        """Test loading a config file."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        assert model.get("playlists.count") == 8
        assert model.get("playlists.tracks_per_playlist") == 30

    def test_get_nested_value(self, temp_config):
        """Test getting nested values by dot-path."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        assert model.get("playlists.ds_pipeline.tower_weights.rhythm") == 0.20
        assert model.get("playlists.ds_pipeline.tower_weights.timbre") == 0.50

    def test_set_override(self, temp_config):
        """Test setting override values."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        model.set("playlists.count", 10)

        # Override should be returned
        assert model.get("playlists.count") == 10

        # Base value should be unchanged
        assert model.get_base_value("playlists.count") == 8

    def test_get_overrides(self, temp_config):
        """Test getting only the override values."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        model.set("playlists.count", 10)
        model.set("playlists.ds_pipeline.tower_weights.rhythm", 0.30)

        overrides = model.get_overrides()
        assert overrides["playlists"]["count"] == 10
        assert overrides["playlists"]["ds_pipeline"]["tower_weights"]["rhythm"] == 0.30

    def test_reset_overrides(self, temp_config):
        """Test resetting overrides."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        model.set("playlists.count", 10)
        model.reset()

        # Should return to base value
        assert model.get("playlists.count") == 8

    def test_merged_config(self, temp_config):
        """Test getting merged config (base + overrides)."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)
        model.set("playlists.count", 10)

        merged = model.get_merged_config()
        assert merged["playlists"]["count"] == 10
        assert merged["playlists"]["tracks_per_playlist"] == 30

    def test_default_from_schema(self, temp_config):
        """Test that schema defaults are used when key is missing."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        # This key isn't in our temp config but has a schema default
        value = model.get("playlists.ds_pipeline.scoring.alpha", default=0.55)
        assert value == 0.55

    def test_is_modified(self, temp_config):
        """Test checking if a key has been modified."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        assert not model.is_modified("playlists.count")

        model.set("playlists.count", 10)
        assert model.is_modified("playlists.count")


class TestNormalization:
    """Tests for weight normalization logic."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_data = {
            "library": {
                "database_path": "data/metadata.db"
            },
            "openai": {
                "api_key": "test-key"
            },
            "playlists": {
                "ds_pipeline": {
                    "tower_weights": {
                        "rhythm": 0.20,
                        "timbre": 0.50,
                        "harmony": 0.30
                    },
                    "embedding": {
                        "sonic_weight": 0.60,
                        "genre_weight": 0.40
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_normalize_tower_weights(self, temp_config):
        """Test normalizing tower weights to sum to 1.0."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        # Change rhythm and normalize
        model.set("playlists.ds_pipeline.tower_weights.rhythm", 0.40)
        model.normalize_group("tower_weights", changed_key="playlists.ds_pipeline.tower_weights.rhythm")

        # Get new values
        rhythm = model.get("playlists.ds_pipeline.tower_weights.rhythm")
        timbre = model.get("playlists.ds_pipeline.tower_weights.timbre")
        harmony = model.get("playlists.ds_pipeline.tower_weights.harmony")

        # Should sum to 1.0
        total = rhythm + timbre + harmony
        assert abs(total - 1.0) < 0.01, f"Sum was {total}, expected 1.0"

        # Changed key should be preserved
        assert rhythm == 0.40

    def test_normalize_embedding_weights(self, temp_config):
        """Test normalizing embedding weights to sum to 1.0."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        # Change sonic weight and normalize
        model.set("playlists.ds_pipeline.embedding.sonic_weight", 0.80)
        model.normalize_group("embedding_weights", changed_key="playlists.ds_pipeline.embedding.sonic_weight")

        sonic = model.get("playlists.ds_pipeline.embedding.sonic_weight")
        genre = model.get("playlists.ds_pipeline.embedding.genre_weight")

        total = sonic + genre
        assert abs(total - 1.0) < 0.01, f"Sum was {total}, expected 1.0"
        assert sonic == 0.80

    def test_normalize_already_normalized(self, temp_config):
        """Test that normalization doesn't change already normalized values."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        # Values are already normalized (0.20 + 0.50 + 0.30 = 1.0)
        rhythm_before = model.get("playlists.ds_pipeline.tower_weights.rhythm")
        timbre_before = model.get("playlists.ds_pipeline.tower_weights.timbre")
        harmony_before = model.get("playlists.ds_pipeline.tower_weights.harmony")

        # Normalize without changing anything
        model.normalize_group("tower_weights")

        # Values should be unchanged
        assert model.get("playlists.ds_pipeline.tower_weights.rhythm") == rhythm_before
        assert model.get("playlists.ds_pipeline.tower_weights.timbre") == timbre_before
        assert model.get("playlists.ds_pipeline.tower_weights.harmony") == harmony_before

    def test_get_group_sum(self, temp_config):
        """Test getting the sum of a normalization group."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel(temp_config)

        # Tower weights should sum to 1.0
        total = model.get_group_sum("tower_weights")
        assert abs(total - 1.0) < 0.01


class TestSecretRedaction:
    """Tests for secret value redaction."""

    def test_redact_secrets_in_dict(self):
        """Test redacting secret values from a dictionary."""
        from src.playlist_gui.config.config_model import redact_secrets

        data = {
            "openai": {
                "api_key": "sk-secret-key-123",
                "model": "gpt-4"
            },
            "plex": {
                "token": "my-plex-token",
                "base_url": "http://localhost:32400"
            },
            "normal": "value"
        }

        redacted = redact_secrets(data)

        assert redacted["openai"]["api_key"] == "***REDACTED***"
        assert redacted["openai"]["model"] == "gpt-4"
        assert redacted["plex"]["token"] == "***REDACTED***"
        assert redacted["plex"]["base_url"] == "http://localhost:32400"
        assert redacted["normal"] == "value"

    def test_is_secret_key(self):
        """Test detecting secret key names."""
        from src.playlist_gui.config.config_model import is_secret_key

        assert is_secret_key("api_key") is True
        assert is_secret_key("openai.api_key") is True
        assert is_secret_key("token") is True
        assert is_secret_key("plex_token") is True
        assert is_secret_key("password") is True
        assert is_secret_key("secret") is True

        assert is_secret_key("model") is False
        assert is_secret_key("count") is False
        assert is_secret_key("base_url") is False


class TestSettingsSchema:
    """Tests for settings schema."""

    def test_schema_has_required_keys(self):
        """Test that schema includes all required key paths."""
        from src.playlist_gui.config.settings_schema import SETTINGS_SCHEMA, get_setting_by_key

        required_keys = [
            "playlists.count",
            "playlists.tracks_per_playlist",
            "playlists.ds_pipeline.embedding.sonic_weight",
            "playlists.ds_pipeline.tower_weights.rhythm",
            "playlists.ds_pipeline.constraints.min_gap",
        ]

        for key in required_keys:
            spec = get_setting_by_key(key)
            assert spec is not None, f"Missing required key: {key}"

    def test_normalize_groups_defined(self):
        """Test that normalization groups are properly defined."""
        from src.playlist_gui.config.settings_schema import get_normalize_groups

        groups = get_normalize_groups()

        assert "tower_weights" in groups
        assert "transition_weights" in groups
        assert "embedding_weights" in groups

        # Tower weights should have 3 members
        assert len(groups["tower_weights"]) == 3

        # Embedding weights should have 2 members
        assert len(groups["embedding_weights"]) == 2

    def test_settings_by_group(self):
        """Test grouping settings by UI group."""
        from src.playlist_gui.config.settings_schema import get_settings_by_group

        groups = get_settings_by_group()

        assert "Playlist Settings" in groups
        assert "Tower Weights (Candidate Selection)" in groups
        assert "Constraints" in groups


class TestMergeConfigWithOverrides:
    """Tests for the merge utility function."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file."""
        config_data = {
            "playlists": {
                "count": 8,
                "ds_pipeline": {
                    "mode": "dynamic",
                    "tower_weights": {
                        "rhythm": 0.20
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_merge_flat_override(self, temp_config):
        """Test merging with flat overrides."""
        from src.playlist_gui.config.config_model import merge_config_with_overrides

        overrides = {
            "playlists": {
                "count": 10
            }
        }

        merged = merge_config_with_overrides(temp_config, overrides)

        assert merged["playlists"]["count"] == 10
        assert merged["playlists"]["ds_pipeline"]["mode"] == "dynamic"

    def test_merge_nested_override(self, temp_config):
        """Test merging with nested overrides."""
        from src.playlist_gui.config.config_model import merge_config_with_overrides

        overrides = {
            "playlists": {
                "ds_pipeline": {
                    "tower_weights": {
                        "rhythm": 0.30
                    }
                }
            }
        }

        merged = merge_config_with_overrides(temp_config, overrides)

        assert merged["playlists"]["ds_pipeline"]["tower_weights"]["rhythm"] == 0.30
        assert merged["playlists"]["ds_pipeline"]["mode"] == "dynamic"
