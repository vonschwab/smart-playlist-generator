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
        """SONIC_MODE_PRESETS recalibrated to MERT's measured cosine scale (2026-06).

        Set as percentiles of the seed-relative max-sim distribution measured on
        the folded MERT artifact (docs/run_audits/mert_full/FLOOR_RECALIBRATION_DISTRIBUTIONS.md):
        strict p75, narrow p50, dynamic p25, discover p10. Supersedes the Phase 2A
        values, which were band-aided down for the (untrustworthy) towers.
        """
        assert SONIC_MODE_PRESETS["strict"]["min_sonic_similarity"] == 0.28   # p75
        assert SONIC_MODE_PRESETS["narrow"]["min_sonic_similarity"] == 0.18   # p50
        assert SONIC_MODE_PRESETS["dynamic"]["min_sonic_similarity"] == 0.08  # p25
        assert SONIC_MODE_PRESETS["discover"]["min_sonic_similarity"] == 0.00  # p10

    def test_get_min_sonic_similarity_defaults(self):
        """get_min_sonic_similarity returns None when nothing is configured.
        apply_mode_presets() is the sole writer for this setting; the values
        it writes are already verified by test_sonic_mode_presets_values.
        """
        assert get_min_sonic_similarity({}, "strict") is None
        assert get_min_sonic_similarity({}, "narrow") is None
        assert get_min_sonic_similarity({}, "dynamic") is None
        assert get_min_sonic_similarity({}, "discover") is None


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
        """Verify strict mode resolves correctly.

        min_sonic_similarity is None here — apply_mode_presets() is the sole
        writer; it sets 0.20 when sonic_mode='strict' in the normal load path.
        max_artist_fraction_final is the universal fallback 0.125; policy.py
        is the sole writer for per-artist caps.
        """
        cfg = default_ds_config("strict", playlist_len=30)

        assert cfg.mode == "strict"
        assert cfg.candidate.min_sonic_similarity is None
        assert cfg.candidate.max_artist_fraction_final == 0.125
        assert cfg.candidate.max_pool_size == 600  # Smaller pool for strict
        assert cfg.construct.min_gap == 3
        assert cfg.candidate.similarity_floor == 0.40  # Higher floor for strict

    def test_narrow_mode_config(self):
        """Verify narrow mode resolves correctly (Phase 2A/3A relaxation).

        min_sonic_similarity is None — apply_mode_presets() sets 0.18 for narrow.
        max_artist_fraction_final is the universal fallback 0.125.
        """
        cfg = default_ds_config("narrow", playlist_len=30)

        assert cfg.mode == "narrow"
        assert cfg.candidate.min_sonic_similarity is None
        assert cfg.candidate.max_artist_fraction_final == 0.125
        assert cfg.candidate.max_pool_size == 800
        assert cfg.construct.min_gap == 3
        assert cfg.candidate.similarity_floor == 0.35

    def test_dynamic_mode_config(self):
        """Verify dynamic mode resolves correctly (Phase 2A/3A relaxation).

        min_sonic_similarity is None — apply_mode_presets() sets 0.08 for dynamic.
        max_artist_fraction_final is the universal fallback 0.125.
        """
        cfg = default_ds_config("dynamic", playlist_len=30)

        assert cfg.mode == "dynamic"
        assert cfg.candidate.min_sonic_similarity is None
        assert cfg.candidate.max_artist_fraction_final == 0.125
        assert cfg.candidate.max_pool_size == 1200
        assert cfg.construct.min_gap == 6
        assert cfg.candidate.similarity_floor == 0.28

    def test_discover_mode_config(self):
        """Verify discover mode resolves correctly (Phase 2A).

        min_sonic_similarity is None — apply_mode_presets() sets 0.00 for discover.
        max_artist_fraction_final is the universal fallback 0.125.
        """
        cfg = default_ds_config("discover", playlist_len=30)

        assert cfg.mode == "discover"
        assert cfg.candidate.min_sonic_similarity is None
        assert cfg.candidate.max_artist_fraction_final == 0.125
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
        assert candidate_pool["min_sonic_similarity"] == 0.28  # MERT p75
        assert candidate_pool["broad_filters"] == ["rock", "indie", "alternative", "pop"]

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
        assert candidate_pool["min_sonic_similarity"] == 0.18  # MERT p50
        assert candidate_pool["broad_filters"] == ["rock", "indie", "alternative", "pop"]

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
        assert candidate_pool["min_sonic_similarity"] == 0.08  # MERT p25
        assert "broad_filters" not in candidate_pool

    def test_apply_mode_presets_preserves_custom_broad_filters(self):
        playlists_cfg = {
            "genre_mode": "narrow",
            "sonic_mode": "narrow",
            "ds_pipeline": {
                "candidate_pool": {
                    "broad_filters": ["rock"],
                },
            },
        }

        apply_mode_presets(playlists_cfg)

        candidate_pool = playlists_cfg["ds_pipeline"]["candidate_pool"]
        assert candidate_pool["broad_filters"] == ["rock"]


class TestGenreIdfEnabledPresets:
    """Test that genre_idf_enabled is wired correctly through mode presets."""

    def test_genre_mode_presets_idf_values(self):
        """discover disables IDF; strict/narrow/dynamic leave it on."""
        assert GENRE_MODE_PRESETS["strict"]["genre_idf_enabled"] is True
        assert GENRE_MODE_PRESETS["narrow"]["genre_idf_enabled"] is True
        assert GENRE_MODE_PRESETS["dynamic"]["genre_idf_enabled"] is True
        assert GENRE_MODE_PRESETS["discover"]["genre_idf_enabled"] is False

    def test_default_ds_config_idf_enabled_by_mode(self):
        """default_ds_config propagates genre_idf_enabled from mode defaults."""
        from src.playlist.config import default_ds_config

        assert default_ds_config("strict", playlist_len=20).candidate.genre_idf_enabled is True
        assert default_ds_config("narrow", playlist_len=20).candidate.genre_idf_enabled is True
        assert default_ds_config("dynamic", playlist_len=20).candidate.genre_idf_enabled is True
        assert default_ds_config("discover", playlist_len=20).candidate.genre_idf_enabled is False

    def test_apply_mode_presets_sets_genre_idf_enabled(self):
        """apply_mode_presets writes genre_idf_enabled into candidate_pool."""
        # discover -> False
        cfg_discover = {"genre_mode": "discover", "sonic_mode": "discover"}
        apply_mode_presets(cfg_discover)
        pool_discover = cfg_discover["ds_pipeline"]["candidate_pool"]
        assert pool_discover["genre_idf_enabled"] is False

        # strict -> True
        cfg_strict = {"genre_mode": "strict", "sonic_mode": "strict"}
        apply_mode_presets(cfg_strict)
        pool_strict = cfg_strict["ds_pipeline"]["candidate_pool"]
        assert pool_strict["genre_idf_enabled"] is True

        # narrow -> True
        cfg_narrow = {"genre_mode": "narrow", "sonic_mode": "narrow"}
        apply_mode_presets(cfg_narrow)
        pool_narrow = cfg_narrow["ds_pipeline"]["candidate_pool"]
        assert pool_narrow["genre_idf_enabled"] is True

        # dynamic -> True
        cfg_dynamic = {"genre_mode": "dynamic", "sonic_mode": "dynamic"}
        apply_mode_presets(cfg_dynamic)
        pool_dynamic = cfg_dynamic["ds_pipeline"]["candidate_pool"]
        assert pool_dynamic["genre_idf_enabled"] is True


class TestSearchWidthPlumbing:
    """The six beam search-width keys must flow config.yaml -> tuning -> PierBridgeConfig.

    Regression for the 2026-06-10 audit finding: these keys were silently never
    read, so every generation ran at dataclass-default (half) search width.
    """

    OVERRIDES = {
        "pier_bridge": {
            "initial_beam_width": 40,
            "max_beam_width": 200,
            "initial_neighbors_m": 200,
            "max_neighbors_m": 800,
            "initial_bridge_helpers": 100,
            "max_bridge_helpers": 400,
        }
    }
    FIELDS = (
        "initial_beam_width",
        "max_beam_width",
        "initial_neighbors_m",
        "max_neighbors_m",
        "initial_bridge_helpers",
        "max_bridge_helpers",
    )

    def test_search_width_keys_resolve_from_config(self):
        tuning, sources = resolve_pier_bridge_tuning(
            mode="dynamic", similarity_floor=0.2, overrides=self.OVERRIDES
        )
        assert tuning.initial_beam_width == 40
        assert tuning.max_beam_width == 200
        assert tuning.initial_neighbors_m == 200
        assert tuning.max_neighbors_m == 800
        assert tuning.initial_bridge_helpers == 100
        assert tuning.max_bridge_helpers == 400
        assert sources["initial_beam_width"] == "pier_bridge.initial_beam_width"

    def test_search_width_defaults_match_pier_bridge_config(self):
        from src.playlist.pier_bridge.config import PierBridgeConfig

        tuning, sources = resolve_pier_bridge_tuning(mode="dynamic", similarity_floor=0.2)
        cfg = PierBridgeConfig()
        for f in self.FIELDS:
            assert getattr(tuning, f) == getattr(cfg, f), f
            assert sources[f] == "default", f

    def test_search_width_flows_into_pier_bridge_config(self):
        from src.playlist.config import default_ds_config
        from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides
        from src.playlist.run_audit import parse_run_audit_config

        pb_cfg, _tuning, _sources, _tw = apply_pier_bridge_overrides(
            pier_bridge_config=None,
            cfg=default_ds_config("dynamic", playlist_len=30),
            overrides=self.OVERRIDES,
            pb_overrides=self.OVERRIDES["pier_bridge"],
            artist_playlist=False,
            dry_run=False,
            audit_cfg=parse_run_audit_config(None),
            resolved_variant="raw",
        )
        assert pb_cfg.initial_beam_width == 40
        assert pb_cfg.max_beam_width == 200
        assert pb_cfg.initial_neighbors_m == 200
        assert pb_cfg.max_neighbors_m == 800
        assert pb_cfg.initial_bridge_helpers == 100
        assert pb_cfg.max_bridge_helpers == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
