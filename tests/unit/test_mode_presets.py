"""Unit tests for mode preset system."""
import pytest
from src.playlist.mode_presets import (
    resolve_genre_mode,
    resolve_sonic_mode,
    resolve_quick_preset,
    validate_mode_combination,
    get_mode_description,
    compare_modes,
    list_available_modes,
    GENRE_MODE_PRESETS,
    SONIC_MODE_PRESETS,
    QUICK_PRESETS,
)


class TestGenreModePresets:
    """Test genre mode preset resolution."""

    def test_all_modes_exist(self):
        """Verify all expected genre modes are defined."""
        expected_modes = {"strict", "narrow", "dynamic", "discover", "off"}
        assert set(GENRE_MODE_PRESETS.keys()) == expected_modes

    def test_dynamic_mode_defaults(self):
        """Dynamic mode should be the balanced default."""
        settings = resolve_genre_mode("dynamic")
        assert settings["enabled"] is True
        assert settings["weight"] == 0.50
        assert settings["sonic_weight"] == 0.50
        assert settings["min_genre_similarity"] == 0.30

    def test_strict_mode_high_threshold(self):
        """Strict mode should have highest threshold and weight."""
        settings = resolve_genre_mode("strict")
        assert settings["weight"] == 0.80
        assert settings["min_genre_similarity"] == 0.50
        assert settings["weight"] > GENRE_MODE_PRESETS["narrow"]["weight"]

    def test_off_mode_disables_genre(self):
        """Off mode should disable genre completely."""
        settings = resolve_genre_mode("off")
        assert settings["enabled"] is False
        assert settings["weight"] == 0.0
        assert settings["sonic_weight"] == 1.0

    def test_mode_progression(self):
        """Genre weight should decrease from strict → narrow → dynamic → discover."""
        strict = resolve_genre_mode("strict")
        narrow = resolve_genre_mode("narrow")
        dynamic = resolve_genre_mode("dynamic")
        discover = resolve_genre_mode("discover")

        assert strict["weight"] > narrow["weight"] > dynamic["weight"] > discover["weight"]

    def test_invalid_mode_raises_error(self):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown genre mode"):
            resolve_genre_mode("invalid_mode")

    def test_mode_with_overrides(self):
        """Overrides should be applied to preset."""
        settings = resolve_genre_mode("dynamic", {"weight": 0.60})
        assert settings["weight"] == 0.60  # Overridden
        assert settings["min_genre_similarity"] == 0.30  # Unchanged

    def test_case_insensitive(self):
        """Mode names should be case-insensitive."""
        lower = resolve_genre_mode("dynamic")
        upper = resolve_genre_mode("DYNAMIC")
        mixed = resolve_genre_mode("Dynamic")

        assert lower["weight"] == upper["weight"] == mixed["weight"]


class TestSonicModePresets:
    """Test sonic mode preset resolution."""

    def test_all_modes_exist(self):
        """Verify all expected sonic modes are defined."""
        expected_modes = {"strict", "narrow", "dynamic", "discover", "off"}
        assert set(SONIC_MODE_PRESETS.keys()) == expected_modes

    def test_dynamic_mode_defaults(self):
        """Dynamic mode should be the balanced default."""
        settings = resolve_sonic_mode("dynamic")
        assert settings["enabled"] is True
        assert settings["weight"] == 0.50
        assert settings["candidate_pool_multiplier"] == 1.0

    def test_strict_mode_tightest(self):
        """Strict mode should have highest weight and smallest pool."""
        settings = resolve_sonic_mode("strict")
        assert settings["weight"] == 0.85
        assert settings["candidate_pool_multiplier"] == 0.6
        assert settings["weight"] > SONIC_MODE_PRESETS["narrow"]["weight"]

    def test_off_mode_disables_sonic(self):
        """Off mode should disable sonic completely (genre-only)."""
        settings = resolve_sonic_mode("off")
        assert settings["enabled"] is False
        assert settings["weight"] == 0.0

    def test_mode_progression(self):
        """Sonic weight should decrease from strict → narrow → dynamic → discover."""
        strict = resolve_sonic_mode("strict")
        narrow = resolve_sonic_mode("narrow")
        dynamic = resolve_sonic_mode("dynamic")
        discover = resolve_sonic_mode("discover")

        assert strict["weight"] > narrow["weight"] > dynamic["weight"] > discover["weight"]

    def test_pool_multiplier_progression(self):
        """Pool multiplier should increase from strict → narrow → dynamic → discover."""
        strict = resolve_sonic_mode("strict")
        narrow = resolve_sonic_mode("narrow")
        dynamic = resolve_sonic_mode("dynamic")
        discover = resolve_sonic_mode("discover")

        assert strict["candidate_pool_multiplier"] < narrow["candidate_pool_multiplier"]
        assert narrow["candidate_pool_multiplier"] < dynamic["candidate_pool_multiplier"]
        assert dynamic["candidate_pool_multiplier"] < discover["candidate_pool_multiplier"]

    def test_invalid_mode_raises_error(self):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown sonic mode"):
            resolve_sonic_mode("invalid_mode")


class TestQuickPresets:
    """Test quick preset combinations."""

    def test_all_presets_defined(self):
        """Verify expected quick presets exist."""
        expected = {"balanced", "tight", "exploratory", "sonic_only", "genre_only", "varied_sound", "sonic_thread"}
        assert set(QUICK_PRESETS.keys()) == expected

    def test_balanced_preset(self):
        """Balanced preset should use dynamic for both."""
        genre, sonic = resolve_quick_preset("balanced")
        assert genre == "dynamic"
        assert sonic == "dynamic"

    def test_tight_preset(self):
        """Tight preset should use strict for both."""
        genre, sonic = resolve_quick_preset("tight")
        assert genre == "strict"
        assert sonic == "strict"

    def test_sonic_only_preset(self):
        """Sonic-only preset should disable genre."""
        genre, sonic = resolve_quick_preset("sonic_only")
        assert genre == "off"
        assert sonic == "dynamic"

    def test_genre_only_preset(self):
        """Genre-only preset should disable sonic."""
        genre, sonic = resolve_quick_preset("genre_only")
        assert genre == "dynamic"
        assert sonic == "off"

    def test_invalid_preset_raises_error(self):
        """Unknown preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown quick preset"):
            resolve_quick_preset("invalid_preset")


class TestModeValidation:
    """Test mode combination validation."""

    def test_valid_combinations(self):
        """Most combinations should be valid."""
        assert validate_mode_combination("dynamic", "dynamic") is True
        assert validate_mode_combination("strict", "strict") is True
        assert validate_mode_combination("off", "dynamic") is True
        assert validate_mode_combination("dynamic", "off") is True
        assert validate_mode_combination("narrow", "discover") is True

    def test_invalid_both_off(self):
        """Both genre and sonic off is invalid."""
        assert validate_mode_combination("off", "off") is False


class TestModeUtilities:
    """Test utility functions."""

    def test_get_mode_description(self):
        """Should return combined description."""
        desc = get_mode_description("dynamic", "dynamic")
        assert "Balanced genre exploration" in desc
        assert "Balanced sonic flow" in desc

    def test_list_available_modes(self):
        """Should list all available modes."""
        modes = list_available_modes()
        assert "genre" in modes
        assert "sonic" in modes
        assert "quick_presets" in modes
        assert len(modes["genre"]) == 5
        assert len(modes["sonic"]) == 5

    def test_compare_modes(self):
        """Should provide detailed mode comparison."""
        comparison = compare_modes("dynamic", "narrow")
        assert comparison["genre_mode"] == "dynamic"
        assert comparison["sonic_mode"] == "narrow"
        assert "weights" in comparison
        assert comparison["weights"]["genre"] == 0.50
        assert "description" in comparison
        assert "playlist_character" in comparison

    def test_compare_modes_weight_sum(self):
        """Genre and sonic weights should sum to 1.0 for balanced modes."""
        comparison = compare_modes("dynamic", "dynamic")
        total = comparison["weights"]["total"]
        assert abs(total - 1.0) < 0.01  # Should be approximately 1.0


class TestModeCharacterInference:
    """Test playlist character inference."""

    def test_ultra_cohesive(self):
        """Strict + strict should be ultra-cohesive."""
        comparison = compare_modes("strict", "strict")
        assert "Ultra-cohesive" in comparison["playlist_character"]

    def test_sonic_only(self):
        """Genre off should be sonic-only."""
        comparison = compare_modes("off", "dynamic")
        assert "Sonic-only" in comparison["playlist_character"]

    def test_genre_only(self):
        """Sonic off should be genre-only."""
        comparison = compare_modes("dynamic", "off")
        assert "Genre-only" in comparison["playlist_character"]

    def test_balanced(self):
        """Dynamic + dynamic should be balanced."""
        comparison = compare_modes("dynamic", "dynamic")
        assert "Balanced" in comparison["playlist_character"]

    def test_maximum_exploration(self):
        """Discover + discover should be maximum exploration."""
        comparison = compare_modes("discover", "discover")
        assert "Maximum exploration" in comparison["playlist_character"]
