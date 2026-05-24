"""Tests that default_ds_config() no longer writes the three duplicate-writer settings."""
from __future__ import annotations

import pytest

from src.playlist.config import default_ds_config, get_min_sonic_similarity


class TestMinSonicSimilarityDedup:
    def test_returns_none_when_no_override_set(self):
        # No per-mode default lookup; returns None when caller hasn't set anything
        assert get_min_sonic_similarity({}, "dynamic") is None
        assert get_min_sonic_similarity({}, "strict") is None
        assert get_min_sonic_similarity({}, "narrow") is None
        assert get_min_sonic_similarity({}, "discover") is None

    def test_respects_explicit_override(self):
        assert get_min_sonic_similarity({"min_sonic_similarity": 0.15}, "dynamic") == 0.15

    def test_respects_per_mode_override(self):
        cfg = {"min_sonic_similarity_strict": 0.30}
        assert get_min_sonic_similarity(cfg, "strict") == 0.30

    def test_per_mode_override_wins_over_base(self):
        cfg = {"min_sonic_similarity": 0.10, "min_sonic_similarity_strict": 0.30}
        assert get_min_sonic_similarity(cfg, "strict") == 0.30
        assert get_min_sonic_similarity(cfg, "dynamic") == 0.10

    def test_explicit_none_returns_none(self):
        cfg = {"min_sonic_similarity": None}
        assert get_min_sonic_similarity(cfg, "dynamic") is None


class TestMaxArtistFractionDedup:
    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_falls_back_to_single_default(self, mode):
        # No per-mode dict; falls back to 0.125 for every mode unless overridden
        cfg = default_ds_config(mode, playlist_len=30)
        assert cfg.candidate.max_artist_fraction_final == 0.125

    def test_respects_explicit_override(self):
        overrides = {"candidate_pool": {"max_artist_fraction": 0.20}}
        cfg = default_ds_config("dynamic", playlist_len=30, overrides=overrides)
        assert cfg.candidate.max_artist_fraction_final == 0.20


class TestBroadFiltersDedup:
    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_empty_by_default(self, mode):
        # No per-mode dict; empty tuple for every mode unless caller passes one
        cfg = default_ds_config(mode, playlist_len=30)
        assert cfg.candidate.broad_filters == ()

    def test_respects_explicit_override(self):
        overrides = {"candidate_pool": {"broad_filters": ["rock", "indie"]}}
        cfg = default_ds_config("dynamic", playlist_len=30, overrides=overrides)
        assert cfg.candidate.broad_filters == ("rock", "indie")
