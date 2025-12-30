"""
Tests for override diff logic, group reset, and secret filtering.

Run with: pytest tests/unit/test_override_logic.py -v
"""
import pytest
from unittest.mock import MagicMock, patch


class TestOverrideDiffUtilities:
    """Tests for ConfigModel override diff utilities."""

    def test_set_override_then_has_override(self):
        """Test that set_override makes has_override return True."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8}}

        assert model.has_override("playlists.count") is False

        model.set("playlists.count", 10)

        assert model.has_override("playlists.count") is True
        assert model.get("playlists.count") == 10

    def test_clear_override_removes_it(self):
        """Test that clear_override removes the override."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8}}

        model.set("playlists.count", 10)
        assert model.has_override("playlists.count") is True

        model.clear_override("playlists.count")
        assert model.has_override("playlists.count") is False
        assert model.get("playlists.count") == 8  # Back to base value

    def test_list_overrides_returns_flat_dict(self):
        """Test that list_overrides returns a flat key_path -> value dict."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {
                "count": 8,
                "ds_pipeline": {"scoring": {"alpha": 0.5}}
            }
        }

        model.set("playlists.count", 10)
        model.set("playlists.ds_pipeline.scoring.alpha", 0.7)

        overrides = model.list_overrides()

        assert overrides == {
            "playlists.count": 10,
            "playlists.ds_pipeline.scoring.alpha": 0.7
        }

    def test_override_count(self):
        """Test that override_count returns correct count."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8, "tracks": 30}}

        assert model.override_count() == 0

        model.set("playlists.count", 10)
        assert model.override_count() == 1

        model.set("playlists.tracks", 50)
        assert model.override_count() == 2

        model.clear_override("playlists.count")
        assert model.override_count() == 1

    def test_diff_summary_returns_correct_pairs(self):
        """Test that diff_summary returns (key_path, base, override) tuples."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8, "tracks": 30}}

        model.set("playlists.count", 10)

        diffs = model.diff_summary()

        assert len(diffs) == 1
        key_path, base_value, override_value = diffs[0]
        assert key_path == "playlists.count"
        assert base_value == 8
        assert override_value == 10

    def test_get_effective_value_is_alias_for_get(self):
        """Test that get_effective_value works like get."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8}}

        assert model.get_effective_value("playlists.count") == 8

        model.set("playlists.count", 10)
        assert model.get_effective_value("playlists.count") == 10

    def test_get_base_value_ignores_overrides(self):
        """Test that get_base_value returns base config value even when overridden."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8}}

        model.set("playlists.count", 10)

        assert model.get("playlists.count") == 10  # Override
        assert model.get_base_value("playlists.count") == 8  # Base


class TestGroupReset:
    """Tests for group-level reset functionality."""

    def test_clear_group_overrides_clears_only_group_keys(self):
        """Test that clear_group_overrides only clears keys in that group."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {
                "count": 8,
                "ds_pipeline": {
                    "scoring": {"alpha": 0.5, "beta": 0.5},
                    "constraints": {"min_gap": 6}
                }
            }
        }

        # Set overrides in different groups
        model.set("playlists.ds_pipeline.scoring.alpha", 0.7)
        model.set("playlists.ds_pipeline.scoring.beta", 0.8)
        model.set("playlists.ds_pipeline.constraints.min_gap", 10)

        assert model.override_count() == 3

        # Clear only Scoring group
        cleared = model.clear_group_overrides("Scoring")

        # Should have cleared 2 overrides (alpha and beta are in Scoring group)
        assert cleared == 2
        assert model.has_override("playlists.ds_pipeline.scoring.alpha") is False
        assert model.has_override("playlists.ds_pipeline.scoring.beta") is False
        # min_gap in Constraints group should still be overridden
        assert model.has_override("playlists.ds_pipeline.constraints.min_gap") is True

    def test_clear_group_overrides_returns_zero_for_unknown_group(self):
        """Test that clear_group_overrides returns 0 for unknown group."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8}}
        model.set("playlists.count", 10)

        cleared = model.clear_group_overrides("NonExistentGroup")
        assert cleared == 0

    def test_global_reset_clears_all_overrides(self):
        """Test that reset() clears all overrides."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {"playlists": {"count": 8, "tracks": 30}}

        model.set("playlists.count", 10)
        model.set("playlists.tracks", 50)

        assert model.override_count() == 2

        model.reset()

        assert model.override_count() == 0
        assert model.get("playlists.count") == 8
        assert model.get("playlists.tracks") == 30


class TestSecretFiltering:
    """Tests for secret filtering in schema and diff summaries."""

    def test_is_secret_key_detects_secret_patterns(self):
        """Test that is_secret_key detects secret patterns."""
        from src.playlist_gui.config.config_model import is_secret_key

        # Should be secret
        assert is_secret_key("api_key") is True
        assert is_secret_key("plex.api_key") is True
        assert is_secret_key("lastfm.token") is True
        assert is_secret_key("discogs.secret") is True
        assert is_secret_key("password") is True
        assert is_secret_key("credential") is True
        assert is_secret_key("bearer_token") is True

        # Should not be secret
        assert is_secret_key("playlists.count") is False
        assert is_secret_key("ds_pipeline.scoring.alpha") is False

    def test_diff_summary_excludes_secrets_by_default(self):
        """Test that diff_summary excludes secret key_paths by default."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {"count": 8},
            "plex": {"api_key": "secret123"}
        }

        model.set("playlists.count", 10)
        model.set("plex.api_key", "new_secret")

        diffs = model.diff_summary(include_secrets=False)

        # Only playlists.count should be in diffs
        assert len(diffs) == 1
        assert diffs[0][0] == "playlists.count"

    def test_diff_summary_includes_secrets_when_requested(self):
        """Test that diff_summary includes secrets when include_secrets=True."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {"count": 8},
            "plex": {"api_key": "secret123"}
        }

        model.set("playlists.count", 10)
        model.set("plex.api_key", "new_secret")

        diffs = model.diff_summary(include_secrets=True)

        # Both should be in diffs
        assert len(diffs) == 2
        key_paths = [d[0] for d in diffs]
        assert "playlists.count" in key_paths
        assert "plex.api_key" in key_paths

    def test_is_secret_setting_filters_schema(self):
        """Test that is_secret_setting correctly identifies secret settings."""
        from src.playlist_gui.config.settings_schema import (
            SettingSpec,
            SettingType,
            is_secret_setting
        )

        # Create test specs
        normal_spec = SettingSpec(
            key_path="playlists.count",
            label="Count",
            setting_type=SettingType.INT,
            group="Test"
        )

        secret_spec = SettingSpec(
            key_path="plex.api_key",
            label="API Key",
            setting_type=SettingType.STRING,
            group="Test"
        )

        assert is_secret_setting(normal_spec) is False
        assert is_secret_setting(secret_spec) is True

    def test_get_visible_settings_excludes_secrets(self):
        """Test that get_visible_settings excludes secret settings."""
        from src.playlist_gui.config.settings_schema import (
            get_visible_settings,
            is_secret_setting
        )

        visible = get_visible_settings()

        # No visible setting should be a secret
        for spec in visible:
            assert is_secret_setting(spec) is False


class TestCleanupEmptyDicts:
    """Tests for cleanup of empty nested dictionaries after clearing overrides."""

    def test_cleanup_removes_empty_nested_dicts(self):
        """Test that clearing overrides removes empty parent dicts."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {
                "ds_pipeline": {
                    "scoring": {"alpha": 0.5}
                }
            }
        }

        model.set("playlists.ds_pipeline.scoring.alpha", 0.7)

        # Verify override structure exists
        assert "playlists" in model._overrides

        model.clear_override("playlists.ds_pipeline.scoring.alpha")

        # After clearing, empty nested dicts should be removed
        assert model._overrides == {}

    def test_cleanup_preserves_sibling_keys(self):
        """Test that cleanup preserves sibling keys when clearing one override."""
        from src.playlist_gui.config.config_model import ConfigModel

        model = ConfigModel()
        model._base_config = {
            "playlists": {
                "ds_pipeline": {
                    "scoring": {"alpha": 0.5, "beta": 0.5}
                }
            }
        }

        model.set("playlists.ds_pipeline.scoring.alpha", 0.7)
        model.set("playlists.ds_pipeline.scoring.beta", 0.8)

        model.clear_override("playlists.ds_pipeline.scoring.alpha")

        # Beta should still be there
        assert model.has_override("playlists.ds_pipeline.scoring.beta") is True
        assert model.get("playlists.ds_pipeline.scoring.beta") == 0.8
