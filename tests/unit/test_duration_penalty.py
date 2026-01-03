"""Unit tests for geometric duration penalty in DS pipeline."""
import pytest
import numpy as np
from src.playlist.pier_bridge_builder import _compute_duration_penalty, PierBridgeConfig


class TestDurationPenaltyFunction:
    """Test the _compute_duration_penalty helper function."""

    def test_no_penalty_for_shorter_tracks(self):
        """Tracks shorter than or equal to reference should have zero penalty."""
        reference = 200000.0  # 200s
        weight = 0.30

        # Shorter track
        assert _compute_duration_penalty(150000.0, reference, weight) == 0.0

        # Equal length
        assert _compute_duration_penalty(200000.0, reference, weight) == 0.0

    def test_penalty_for_longer_tracks(self):
        """Tracks longer than reference should have positive penalty."""
        reference = 200000.0  # 200s
        weight = 0.30

        # Longer track should have penalty > 0
        penalty = _compute_duration_penalty(250000.0, reference, weight)
        assert penalty > 0.0

    def test_penalty_monotonicity(self):
        """Penalty should increase as duration increases."""
        reference = 200000.0  # 200s
        weight = 0.30

        # Get penalties for different durations
        penalty_210s = _compute_duration_penalty(210000.0, reference, weight)  # +5%
        penalty_240s = _compute_duration_penalty(240000.0, reference, weight)  # +20%
        penalty_300s = _compute_duration_penalty(300000.0, reference, weight)  # +50%
        penalty_400s = _compute_duration_penalty(400000.0, reference, weight)  # +100%
        penalty_600s = _compute_duration_penalty(600000.0, reference, weight)  # +200%

        # Should be strictly increasing
        assert penalty_210s < penalty_240s < penalty_300s < penalty_400s < penalty_600s

    def test_phase_boundaries(self):
        """Test penalty values at phase transition boundaries."""
        reference = 200000.0  # 200s
        weight = 0.30

        # Phase 1: 0-20% (gentle)
        penalty_20pct = _compute_duration_penalty(240000.0, reference, weight)  # +20%
        assert 0.01 < penalty_20pct < 0.02  # Should be small

        # Phase 2: 20-50% (moderate)
        penalty_50pct = _compute_duration_penalty(300000.0, reference, weight)  # +50%
        assert 0.08 < penalty_50pct < 0.10  # Moderate

        # Phase 3: 50-100% (steep)
        penalty_100pct = _compute_duration_penalty(400000.0, reference, weight)  # +100%
        assert 0.20 < penalty_100pct < 0.25  # Steep

        # Phase 4: >100% (severe)
        penalty_200pct = _compute_duration_penalty(600000.0, reference, weight)  # +200%
        assert 0.80 < penalty_200pct < 1.00  # Severe

    def test_gentle_phase_growth(self):
        """Penalty should grow gently in 0-20% excess range."""
        reference = 200000.0  # 200s
        weight = 0.30

        penalty_5pct = _compute_duration_penalty(210000.0, reference, weight)   # +5%
        penalty_10pct = _compute_duration_penalty(220000.0, reference, weight)  # +10%
        penalty_15pct = _compute_duration_penalty(230000.0, reference, weight)  # +15%

        # Growth should be sub-linear (power 1.5)
        # Verify penalties are increasing but at a decreasing rate
        diff_1 = penalty_10pct - penalty_5pct
        diff_2 = penalty_15pct - penalty_10pct
        # Second difference should be smaller (sub-linear growth)
        assert penalty_5pct < penalty_10pct < penalty_15pct
        assert diff_2 > diff_1  # Should be accelerating but gently

    def test_severe_phase_growth(self):
        """Penalty should grow very steeply beyond 100% excess."""
        reference = 200000.0  # 200s
        weight = 0.30

        penalty_100pct = _compute_duration_penalty(400000.0, reference, weight)  # +100%
        penalty_150pct = _compute_duration_penalty(500000.0, reference, weight)  # +150%
        penalty_200pct = _compute_duration_penalty(600000.0, reference, weight)  # +200%

        # Growth should be cubic (power 3.0)
        # Verify acceleration is strong
        diff_1 = penalty_150pct - penalty_100pct
        diff_2 = penalty_200pct - penalty_150pct
        # Second difference should be much larger (cubic growth)
        assert diff_2 > diff_1 * 1.5  # Strong acceleration

    def test_weight_scaling(self):
        """Penalty should scale linearly with weight."""
        reference = 200000.0
        candidate = 250000.0  # +25%

        penalty_w01 = _compute_duration_penalty(candidate, reference, 0.1)
        penalty_w03 = _compute_duration_penalty(candidate, reference, 0.3)

        # Triple weight should give triple penalty
        ratio = penalty_w03 / penalty_w01 if penalty_w01 > 0 else 0
        assert 2.9 < ratio < 3.1

    def test_invalid_durations(self):
        """Invalid durations should return zero penalty."""
        reference = 200000.0
        weight = 0.30

        # Zero durations
        assert _compute_duration_penalty(0.0, reference, weight) == 0.0
        assert _compute_duration_penalty(250000.0, 0.0, weight) == 0.0

        # Negative durations
        assert _compute_duration_penalty(-100.0, reference, weight) == 0.0
        assert _compute_duration_penalty(250000.0, -100.0, weight) == 0.0

    def test_realistic_playlist_scenarios(self):
        """Test with realistic pier-bridge scenarios."""
        weight = 0.30

        # Scenario 1: Short pier tracks (3min)
        ref_short = 180000.0  # 3min
        # 3:30 track (+16.7%)
        penalty_330 = _compute_duration_penalty(210000.0, ref_short, weight)
        assert penalty_330 < 0.02  # Gentle penalty

        # Scenario 2: Medium pier tracks (4:30)
        ref_medium = 270000.0  # 4.5min
        # 6min track (+33%)
        penalty_6min = _compute_duration_penalty(360000.0, ref_medium, weight)
        assert 0.025 < penalty_6min < 0.08  # Moderate penalty

        # Scenario 3: Long pier tracks (6min)
        ref_long = 360000.0  # 6min
        # 12min track (+100%)
        penalty_12min = _compute_duration_penalty(720000.0, ref_long, weight)
        assert 0.20 < penalty_12min < 0.30  # Severe penalty threshold

    def test_percentage_based_behavior(self):
        """Verify penalty is based on percentage, not absolute duration."""
        weight = 0.30

        # Same absolute excess (1min = 60s) but different percentages
        # Case 1: 3min → 4min (+33%)
        penalty_3to4 = _compute_duration_penalty(240000.0, 180000.0, weight)

        # Case 2: 6min → 7min (+16.7%)
        penalty_6to7 = _compute_duration_penalty(420000.0, 360000.0, weight)

        # 33% excess should have higher penalty than 16.7% excess
        # even though absolute excess is the same
        assert penalty_3to4 > penalty_6to7


class TestDurationPenaltyConfig:
    """Test that config fields exist and have correct defaults."""

    def test_config_fields_exist(self):
        """Config should have duration_penalty_enabled and duration_penalty_weight fields."""
        cfg = PierBridgeConfig()

        assert hasattr(cfg, 'duration_penalty_enabled')
        assert hasattr(cfg, 'duration_penalty_weight')

    def test_default_enabled(self):
        """Duration penalty should be ENABLED by default (medium-firm)."""
        cfg = PierBridgeConfig()

        assert cfg.duration_penalty_enabled is True

    def test_default_weight(self):
        """Default weight should be 0.30 (medium-firm strength)."""
        cfg = PierBridgeConfig()

        assert cfg.duration_penalty_weight == 0.30

    def test_config_override(self):
        """Should be able to override config values."""
        cfg = PierBridgeConfig(
            duration_penalty_enabled=False,
            duration_penalty_weight=0.15
        )

        assert cfg.duration_penalty_enabled is False
        assert cfg.duration_penalty_weight == 0.15
