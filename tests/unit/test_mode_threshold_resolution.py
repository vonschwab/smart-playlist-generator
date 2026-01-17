"""
Test suite for mode threshold resolution (Phases 1, 2A, 2B, 3A).

Validates that strict/narrow/dynamic/discover modes resolve to correct
threshold values for sonic floors, genre floors, and bridge floors.
"""
import pytest

from src.playlist.config import default_ds_config, get_min_sonic_similarity, resolve_pier_bridge_tuning
from src.playlist.mode_presets import GENRE_MODE_PRESETS, SONIC_MODE_PRESETS, apply_mode_presets


class TestSonicFloorResolution:
    """Test sonic floor resolution for all modes."""

    def test_sonic_mode_presets_values(self):
        """Verify SONIC_MODE_PRESETS has correct values (Phase 2A)."""
        assert SONIC_MODE_PRESETS["strict"]["min_sonic_similarity"] == 0.20
        assert SONIC_MODE_PRESETS["narrow"]["min_sonic_similarity"] == 0.12  # Relaxed from 0.18
        assert SONIC_MODE_PRESETS["dynamic"]["min_sonic_similarity"] == 0.05  # Relaxed from 0.10
        assert SONIC_MODE_PRESETS["discover"]["min_sonic_similarity"] == 0.00  # Disabled from 0.02

    def test_get_min_sonic_similarity_defaults(self):
        """Verify get_min_sonic_similarity returns correct defaults."""
        assert get_min_sonic_similarity({}, "strict") == 0.20
        assert get_min_sonic_similarity({}, "narrow") == 0.12
        assert get_min_sonic_similarity({}, "dynamic") == 0.05
        assert get_min_sonic_similarity({}, "discover") == 0.00


class TestGenreFloorResolution:
    """Test genre floor resolution for all modes."""

    def test_genre_mode_presets_values(self):
        """Verify GENRE_MODE_PRESETS has correct values (Phase 2B)."""
        # Strict mode
        assert GENRE_MODE_PRESETS["strict"]["min_genre_similarity"] == 0.50
        assert GENRE_MODE_PRESETS["strict"]["min_genre_similarity_narrow"] == 0.60

        # Narrow mode
        assert GENRE_MODE_PRESETS["narrow"]["min_genre_similarity"] == 0.40
        assert GENRE_MODE_PRESETS["narrow"]["min_genre_similarity_narrow"] == 0.42  # Relaxed from 0.50

        # Dynamic mode
        assert GENRE_MODE_PRESETS["dynamic"]["min_genre_similarity"] == 0.25  # Relaxed from 0.30
        assert GENRE_MODE_PRESETS["dynamic"]["min_genre_similarity_narrow"] == 0.40

        # Discover mode
        assert GENRE_MODE_PRESETS["discover"]["min_genre_similarity"] == 0.20
        assert GENRE_MODE_PRESETS["discover"]["min_genre_similarity_narrow"] == 0.30


class TestBridgeFloorResolution:
    """Test bridge floor resolution for all modes."""

    def test_pier_bridge_tuning_defaults(self):
        """Verify resolve_pier_bridge_tuning returns correct bridge_floor values (Phase 3A)."""
        # Strict mode
        tuning_strict, _ = resolve_pier_bridge_tuning(mode="strict", similarity_floor=0.40)
        assert tuning_strict.bridge_floor == 0.10

        # Narrow mode
        tuning_narrow, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35)
        assert tuning_narrow.bridge_floor == 0.05  # Relaxed from 0.08

        # Dynamic mode
        tuning_dynamic, _ = resolve_pier_bridge_tuning(mode="dynamic", similarity_floor=0.28)
        assert tuning_dynamic.bridge_floor == 0.02  # Relaxed from 0.03

    def test_pier_bridge_tuning_with_config_overrides(self):
        """Verify config.yaml overrides work correctly."""
        overrides = {
            "pier_bridge": {
                "bridge_floor_strict": 0.10,
                "bridge_floor_narrow": 0.05,
                "bridge_floor_dynamic": 0.02,
            }
        }

        tuning_strict, _ = resolve_pier_bridge_tuning(mode="strict", similarity_floor=0.40, overrides=overrides)
        assert tuning_strict.bridge_floor == 0.10

        tuning_narrow, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
        assert tuning_narrow.bridge_floor == 0.05

        tuning_dynamic, _ = resolve_pier_bridge_tuning(mode="dynamic", similarity_floor=0.28, overrides=overrides)
        assert tuning_dynamic.bridge_floor == 0.02


class TestDSPipelineConfigResolution:
    """Test end-to-end DS pipeline config resolution."""

    def test_strict_mode_config(self):
        """Verify strict mode resolves correctly."""
        cfg = default_ds_config("strict", playlist_len=30)

        assert cfg.mode == "strict"
        assert cfg.candidate.min_sonic_similarity == 0.20
        assert cfg.candidate.max_pool_size == 600  # Smaller pool for strict
        assert cfg.construct.min_gap == 3
        assert cfg.candidate.similarity_floor == 0.40  # Higher floor for strict

    def test_narrow_mode_config(self):
        """Verify narrow mode resolves correctly (Phase 2A/3A relaxation)."""
        cfg = default_ds_config("narrow", playlist_len=30)

        assert cfg.mode == "narrow"
        assert cfg.candidate.min_sonic_similarity == 0.12  # Relaxed from 0.18
        assert cfg.candidate.max_pool_size == 800
        assert cfg.construct.min_gap == 3
        assert cfg.candidate.similarity_floor == 0.35

    def test_dynamic_mode_config(self):
        """Verify dynamic mode resolves correctly (Phase 2A/3A relaxation)."""
        cfg = default_ds_config("dynamic", playlist_len=30)

        assert cfg.mode == "dynamic"
        assert cfg.candidate.min_sonic_similarity == 0.05  # Relaxed from 0.10
        assert cfg.candidate.max_pool_size == 1200
        assert cfg.construct.min_gap == 6
        assert cfg.candidate.similarity_floor == 0.28

    def test_discover_mode_config(self):
        """Verify discover mode resolves correctly (Phase 2A)."""
        cfg = default_ds_config("discover", playlist_len=30)

        assert cfg.mode == "discover"
        assert cfg.candidate.min_sonic_similarity == 0.00  # Disabled
        assert cfg.candidate.max_pool_size == 2000
        assert cfg.construct.min_gap == 9
        assert cfg.candidate.similarity_floor == 0.22


class TestModePresetsApplication:
    """Test that apply_mode_presets correctly applies mode settings."""

    def test_apply_strict_mode_presets(self):
        """Verify apply_mode_presets works for strict mode."""
        playlists_cfg = {
            "genre_mode": "strict",
            "sonic_mode": "strict",
        }

        apply_mode_presets(playlists_cfg)

        # Genre settings
        genre_cfg = playlists_cfg["genre_similarity"]
        assert genre_cfg["enabled"] is True
        assert genre_cfg["min_genre_similarity"] == 0.50
        assert genre_cfg["min_genre_similarity_narrow"] == 0.60

        # Sonic settings
        candidate_pool = playlists_cfg["ds_pipeline"]["candidate_pool"]
        assert candidate_pool["min_sonic_similarity"] == 0.20

    def test_apply_narrow_mode_presets(self):
        """Verify apply_mode_presets works for narrow mode (Phase 2A/2B relaxation)."""
        playlists_cfg = {
            "genre_mode": "narrow",
            "sonic_mode": "narrow",
        }

        apply_mode_presets(playlists_cfg)

        # Genre settings
        genre_cfg = playlists_cfg["genre_similarity"]
        assert genre_cfg["enabled"] is True
        assert genre_cfg["min_genre_similarity"] == 0.40
        assert genre_cfg["min_genre_similarity_narrow"] == 0.42  # Relaxed from 0.50

        # Sonic settings
        candidate_pool = playlists_cfg["ds_pipeline"]["candidate_pool"]
        assert candidate_pool["min_sonic_similarity"] == 0.12  # Relaxed from 0.18

    def test_apply_dynamic_mode_presets(self):
        """Verify apply_mode_presets works for dynamic mode (Phase 2A/2B relaxation)."""
        playlists_cfg = {
            "genre_mode": "dynamic",
            "sonic_mode": "dynamic",
        }

        apply_mode_presets(playlists_cfg)

        # Genre settings
        genre_cfg = playlists_cfg["genre_similarity"]
        assert genre_cfg["enabled"] is True
        assert genre_cfg["min_genre_similarity"] == 0.25  # Relaxed from 0.30

        # Sonic settings
        candidate_pool = playlists_cfg["ds_pipeline"]["candidate_pool"]
        assert candidate_pool["min_sonic_similarity"] == 0.05  # Relaxed from 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
