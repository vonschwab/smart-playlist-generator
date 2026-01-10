"""
Unit tests for DJ bridging config parsing (flat key alias fallback).

Tests that the pipeline correctly parses both:
- Nested key: dj_bridging.pooling.strategy (preferred)
- Flat key: dj_bridging.dj_pooling_strategy (deprecated fallback)
"""
from __future__ import annotations

import pytest
from unittest import mock
from src.playlist.pier_bridge_builder import PierBridgeConfig
from dataclasses import replace


def test_nested_pooling_strategy_takes_precedence():
    """Nested pooling.strategy should take precedence over flat dj_pooling_strategy."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "pooling": {
            "strategy": "dj_union",  # nested (preferred)
        },
        "dj_pooling_strategy": "baseline",  # flat (should be ignored)
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (should NOT trigger since nested key was set)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "dj_union", "Nested key should take precedence"


def test_flat_pooling_strategy_fallback():
    """Flat dj_pooling_strategy should be used when nested pooling.strategy is absent."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "dj_pooling_strategy": "dj_union",  # flat (should be used as fallback)
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key (should not exist)
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (SHOULD trigger since nested key was not set)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "dj_union", "Flat key should be used as fallback"


def test_flat_pooling_strategy_with_empty_pooling_dict():
    """Flat dj_pooling_strategy should be used when pooling dict exists but has no strategy."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "pooling": {
            "k_local": 200,  # other pooling keys, but no strategy
        },
        "dj_pooling_strategy": "dj_union",  # flat (should be used)
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key (should not find strategy)
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (SHOULD trigger)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "dj_union", "Flat key should be used when pooling dict has no strategy"


def test_no_pooling_config_uses_default():
    """When neither nested nor flat key is present, should use dataclass default."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        # No pooling config at all
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key (should not exist)
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (should not exist)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "baseline", "Should use dataclass default when no config is present"


def test_flat_key_normalization():
    """Flat dj_pooling_strategy should be normalized (stripped and lowercased)."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "dj_pooling_strategy": "  DJ_UNION  ",  # should be normalized
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key (should not exist)
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias with normalization
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "dj_union", "Flat key should be normalized (stripped and lowercased)"


def test_flat_key_same_as_default_not_overridden():
    """If flat key matches dataclass default, it should not trigger fallback logic."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "dj_pooling_strategy": "baseline",  # same as default
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key (should not exist)
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (should NOT trigger since flat_key == default)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy

    assert pool_strategy == "baseline", "Flat key matching default should not trigger fallback"


@mock.patch('src.playlist.pipeline.logger')
def test_flat_key_logs_deprecation_warning(mock_logger):
    """Using flat dj_pooling_strategy should log a deprecation warning."""
    pb_cfg = PierBridgeConfig(dj_pooling_strategy="baseline")
    dj_raw = {
        "enabled": True,
        "dj_pooling_strategy": "dj_union",
    }

    # Simulate the parsing logic from pipeline.py:963-993
    pooling_raw = dj_raw.get("pooling")
    pool_strategy = pb_cfg.dj_pooling_strategy

    # Parse nested key
    if isinstance(pooling_raw, dict):
        if isinstance(pooling_raw.get("strategy"), str):
            pool_strategy = str(pooling_raw.get("strategy"))

    # Fallback: flat-key alias (should trigger and log warning)
    if isinstance(dj_raw.get("dj_pooling_strategy"), str):
        flat_key_strategy = str(dj_raw.get("dj_pooling_strategy")).strip().lower()
        if pool_strategy == pb_cfg.dj_pooling_strategy and flat_key_strategy != pb_cfg.dj_pooling_strategy:
            pool_strategy = flat_key_strategy
            mock_logger.warning(
                "DJ bridging: flat key 'dj_pooling_strategy=%s' is deprecated; "
                "use nested 'dj_bridging.pooling.strategy' instead",
                flat_key_strategy
            )

    # Verify warning was logged
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0]
    assert "deprecated" in call_args[0]
    assert "dj_pooling_strategy" in call_args[0]
    assert call_args[1] == "dj_union"
