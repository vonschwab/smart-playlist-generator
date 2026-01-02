"""Unit tests for soft duration penalty in DS pipeline."""
import pytest
import numpy as np
from src.playlist.pier_bridge_builder import _compute_duration_penalty, PierBridgeConfig


class TestDurationPenaltyFunction:
    """Test the _compute_duration_penalty helper function."""

    def test_no_penalty_for_shorter_tracks(self):
        """Tracks shorter than or equal to reference should have zero penalty."""
        reference = 180000.0  # 180s = 3min
        weight = 0.10

        # Shorter track
        assert _compute_duration_penalty(120000.0, reference, weight) == 0.0

        # Equal length
        assert _compute_duration_penalty(180000.0, reference, weight) == 0.0

    def test_penalty_for_longer_tracks(self):
        """Tracks longer than reference should have positive penalty."""
        reference = 180000.0  # 180s = 3min
        weight = 0.10

        # Longer track should have penalty > 0
        penalty = _compute_duration_penalty(240000.0, reference, weight)
        assert penalty > 0.0

    def test_penalty_monotonicity(self):
        """Penalty should increase as duration increases."""
        reference = 180000.0  # 180s = 3min
        weight = 0.10

        # Get penalties for different durations
        penalty_200s = _compute_duration_penalty(200000.0, reference, weight)
        penalty_240s = _compute_duration_penalty(240000.0, reference, weight)
        penalty_300s = _compute_duration_penalty(300000.0, reference, weight)

        # Should be strictly increasing
        assert penalty_200s < penalty_240s < penalty_300s

    def test_penalty_quadratic_growth(self):
        """Penalty should grow quadratically with excess duration."""
        reference = 180000.0  # 180s = 3min
        weight = 0.10

        # 2x excess should have ~4x penalty (quadratic)
        excess_30s = _compute_duration_penalty(210000.0, reference, weight)  # +30s excess
        excess_60s = _compute_duration_penalty(240000.0, reference, weight)  # +60s excess (2x)

        # Ratio should be close to 4.0 (within floating point tolerance)
        ratio = excess_60s / excess_30s if excess_30s > 0 else 0
        assert 3.9 < ratio < 4.1

    def test_weight_scaling(self):
        """Penalty should scale linearly with weight."""
        reference = 180000.0
        candidate = 240000.0

        penalty_w01 = _compute_duration_penalty(candidate, reference, 0.1)
        penalty_w02 = _compute_duration_penalty(candidate, reference, 0.2)

        # Double weight should give double penalty
        assert abs(penalty_w02 - 2 * penalty_w01) < 1e-9

    def test_invalid_durations(self):
        """Invalid durations should return zero penalty."""
        reference = 180000.0
        weight = 0.10

        # Zero durations
        assert _compute_duration_penalty(0.0, reference, weight) == 0.0
        assert _compute_duration_penalty(200000.0, 0.0, weight) == 0.0

        # Negative durations
        assert _compute_duration_penalty(-100.0, reference, weight) == 0.0
        assert _compute_duration_penalty(200000.0, -100.0, weight) == 0.0

    def test_realistic_values(self):
        """Test with realistic track durations."""
        # Pier seeds: 3min (180s) and 4min (240s) -> reference = max = 240s
        reference = 240000.0
        weight = 0.10

        # Short track (2min): no penalty
        assert _compute_duration_penalty(120000.0, reference, weight) == 0.0

        # Exact match (4min): no penalty
        assert _compute_duration_penalty(240000.0, reference, weight) == 0.0

        # Slightly longer (5min = 300s): +60s excess, +25% over reference
        penalty_5min = _compute_duration_penalty(300000.0, reference, weight)
        # penalty = 0.10 * (60000 / 240000)^2 = 0.10 * 0.25^2 = 0.10 * 0.0625 = 0.00625
        assert abs(penalty_5min - 0.00625) < 1e-9

        # Much longer (10min = 600s): +360s excess, +150% over reference
        penalty_10min = _compute_duration_penalty(600000.0, reference, weight)
        # penalty = 0.10 * (360000 / 240000)^2 = 0.10 * 1.5^2 = 0.10 * 2.25 = 0.225
        assert abs(penalty_10min - 0.225) < 1e-9


class TestDurationPenaltyConfig:
    """Test that config fields exist and have correct defaults."""

    def test_config_fields_exist(self):
        """Config should have duration_penalty_enabled and duration_penalty_weight fields."""
        cfg = PierBridgeConfig()

        assert hasattr(cfg, 'duration_penalty_enabled')
        assert hasattr(cfg, 'duration_penalty_weight')

    def test_default_disabled(self):
        """Duration penalty should be disabled by default."""
        cfg = PierBridgeConfig()

        assert cfg.duration_penalty_enabled is False

    def test_default_weight(self):
        """Default weight should be reasonable (0.10)."""
        cfg = PierBridgeConfig()

        assert cfg.duration_penalty_weight == 0.10

    def test_config_override(self):
        """Should be able to override config values."""
        cfg = PierBridgeConfig(
            duration_penalty_enabled=True,
            duration_penalty_weight=0.15
        )

        assert cfg.duration_penalty_enabled is True
        assert cfg.duration_penalty_weight == 0.15
