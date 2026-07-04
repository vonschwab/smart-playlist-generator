"""Integration test: confirm one writer per setting after the dedup."""
from __future__ import annotations

from src.playlist.config import default_ds_config
from src.playlist.mode_presets import apply_mode_presets


def _build_playlists_cfg(cohesion: str, sonic: str, genre: str) -> dict:
    return {
        "cohesion_mode": cohesion,
        "sonic_mode": sonic,
        "genre_mode": genre,
        "ds_pipeline": {
            "candidate_pool": {},
            "scoring": {},
            "constraints": {},
        },
    }


class TestSingleWriterEndToEnd:
    def test_sonic_mode_owns_min_sonic_similarity(self):
        """sonic_mode=narrow should set 0.18 regardless of cohesion_mode."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="narrow", genre="dynamic")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # Sonic-narrow preset writes 0.18 (MERT p50, recalibrated 2026-06).
        # Without dedup, cohesion=strict would overwrite it inside
        # default_ds_config(); confirm sonic_mode stays the single writer.
        assert ds_cfg.candidate.min_sonic_similarity == 0.18

    def test_genre_mode_owns_broad_filters(self):
        """genre_mode=dynamic means no broad_filters even if cohesion_mode=strict."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # Without dedup, cohesion=strict would set broad_filters to
        # ["rock","indie","alternative","pop"]. Confirm it does not.
        assert ds_cfg.candidate.broad_filters == ()

    def test_genre_strict_writes_broad_filters(self):
        """genre_mode=strict still writes broad_filters via apply_mode_presets."""
        cfg = _build_playlists_cfg(cohesion="dynamic", sonic="dynamic", genre="strict")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # apply_mode_presets writes broad_filters when genre_mode in {strict, narrow}.
        assert "rock" in ds_cfg.candidate.broad_filters

    def test_max_artist_fraction_uses_caller_override(self):
        """max_artist_fraction comes from caller (policy.py in normal flow), not per-mode dict."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        cfg["ds_pipeline"]["candidate_pool"]["max_artist_fraction"] = 0.10
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        assert ds_cfg.candidate.max_artist_fraction_final == 0.10

    def test_cohesion_still_drives_alpha_schedule(self):
        """Cohesion-owned settings (alpha schedule etc.) still change with cohesion_mode."""
        cfg_strict = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        cfg_dynamic = _build_playlists_cfg(cohesion="dynamic", sonic="dynamic", genre="dynamic")
        apply_mode_presets(cfg_strict)
        apply_mode_presets(cfg_dynamic)
        ds_strict = default_ds_config("strict", playlist_len=30, overrides=cfg_strict["ds_pipeline"])
        ds_dynamic = default_ds_config("dynamic", playlist_len=30, overrides=cfg_dynamic["ds_pipeline"])
        # Cohesion-owned: alpha_schedule changes
        assert ds_strict.construct.alpha_schedule == "constant"
        assert ds_dynamic.construct.alpha_schedule == "arc"
