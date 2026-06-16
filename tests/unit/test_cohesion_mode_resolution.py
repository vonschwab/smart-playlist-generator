"""Tests for resolve_cohesion_mode() helper and updated Mode Literal."""
from __future__ import annotations

import logging

import pytest

from src.playlist.config import default_ds_config, resolve_cohesion_mode


class TestResolveCohesionMode:
    def test_valid_strict(self):
        assert resolve_cohesion_mode({"cohesion_mode": "strict"}) == "strict"

    def test_valid_narrow(self):
        assert resolve_cohesion_mode({"cohesion_mode": "narrow"}) == "narrow"

    def test_valid_dynamic(self):
        assert resolve_cohesion_mode({"cohesion_mode": "dynamic"}) == "dynamic"

    def test_valid_discover(self):
        assert resolve_cohesion_mode({"cohesion_mode": "discover"}) == "discover"

    def test_missing_key_defaults_dynamic(self):
        assert resolve_cohesion_mode({}) == "dynamic"

    def test_none_cfg_defaults_dynamic(self):
        assert resolve_cohesion_mode(None) == "dynamic"  # type: ignore[arg-type]

    def test_invalid_value_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode({"cohesion_mode": "tight"})
        assert result == "dynamic"
        assert any("Invalid cohesion_mode" in r.message for r in caplog.records)

    def test_value_is_normalized(self):
        assert resolve_cohesion_mode({"cohesion_mode": "  Strict "}) == "strict"

    def test_stale_ds_pipeline_mode_warns(self, caplog):
        cfg = {"ds_pipeline": {"mode": "narrow"}, "cohesion_mode": "dynamic"}
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode(cfg)
        assert result == "dynamic"
        assert any("ds_pipeline.mode is no longer used" in r.message for r in caplog.records)

    def test_stale_ds_pipeline_mode_alone_ignored(self, caplog):
        cfg = {"ds_pipeline": {"mode": "narrow"}}
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode(cfg)
        assert result == "dynamic"
        assert any("ds_pipeline.mode is no longer used" in r.message for r in caplog.records)


class TestModeLiteralUpdate:
    def test_strict_is_valid_mode(self):
        # Should not raise
        cfg = default_ds_config("strict", playlist_len=30)
        assert cfg.mode == "strict"

    def test_sonic_only_is_no_longer_valid(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            default_ds_config("sonic_only", playlist_len=30)
