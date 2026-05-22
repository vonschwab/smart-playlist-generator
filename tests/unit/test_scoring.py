"""Unit tests for pier-bridge scoring modules.

Tests extracted scoring functions from pier_bridge_builder.py (Phase 3.1).

Coverage:
- Transition scoring
- Bridge scoring
- Constraint validation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.scoring import (
    compute_transition_score,
    compute_transition_score_raw_and_transformed,
    compute_bridgeability_score,
    TransitionWeights,
    ScoringConstraints,
    SeedOrderingConfig,
)


# =============================================================================
# Transition Scoring Tests
# =============================================================================

class TestTransitionScoring:
    """Test transition quality scoring."""

    def test_basic_transition_score(self):
        """Test basic transition scoring with full matrices only."""
        # Create simple L2-normalized vectors
        X_full = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.707, 0.707],  # 45-degree angle
        ])

        # Perfect similarity (same vector)
        score = compute_transition_score(
            idx_a=0, idx_b=0,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.4, weight_mid_mid=0.3, weight_full_full=0.3,
            center_transitions=False
        )
        assert abs(score - 1.0) < 0.01

        # Orthogonal vectors (no similarity)
        score = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.4, weight_mid_mid=0.3, weight_full_full=0.3,
            center_transitions=False
        )
        assert abs(score - 0.0) < 0.01

        # 45-degree angle (cos(45°) ≈ 0.707)
        score = compute_transition_score(
            idx_a=0, idx_b=2,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.4, weight_mid_mid=0.3, weight_full_full=0.3,
            center_transitions=False
        )
        assert abs(score - 0.707) < 0.01

    def test_transition_with_centering(self):
        """Test transition scoring with centering enabled."""
        X_full = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],  # Opposite direction
        ])

        # Without centering: cos(180°) = -1.0
        score_no_center = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.0, weight_mid_mid=0.0, weight_full_full=1.0,
            center_transitions=False
        )
        assert abs(score_no_center - (-1.0)) < 0.01

        # With centering: rescale [-1, 1] → [0, 1], so -1 → 0
        score_centered = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.0, weight_mid_mid=0.0, weight_full_full=1.0,
            center_transitions=True
        )
        assert abs(score_centered - 0.0) < 0.01

    def test_transition_with_segments(self):
        """Test transition with start/mid/end segments."""
        # Create segment matrices
        X_full = np.array([[1.0, 0.0], [0.8, 0.6]])  # L2-normalized
        X_start = np.array([[1.0, 0.0], [1.0, 0.0]])  # Perfect start alignment
        X_mid = np.array([[0.6, 0.8], [0.6, 0.8]])    # Perfect mid alignment
        X_end = np.array([[0.0, 1.0], [1.0, 0.0]])    # Orthogonal end-start

        score = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=X_start, X_mid=X_mid, X_end=X_end,
            weight_end_start=0.5, weight_mid_mid=0.3, weight_full_full=0.2,
            center_transitions=False
        )

        # Expected: 0.5 * 0.0 (end-start) + 0.3 * 1.0 (mid) + 0.2 * 0.8 (full)
        # Note: [1,0]·[0.8,0.6] = 0.8, not 0.96
        expected = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.8
        assert abs(score - expected) < 0.01

    def test_raw_and_transformed_scores(self):
        """Test raw vs transformed scoring."""
        X_full = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],  # Opposite
        ])

        raw, transformed = compute_transition_score_raw_and_transformed(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.0, weight_mid_mid=0.0, weight_full_full=1.0,
            center_transitions=True
        )

        # Raw should be -1.0, transformed should be 0.0
        assert abs(raw - (-1.0)) < 0.01
        assert abs(transformed - 0.0) < 0.01

    def test_raw_equals_transformed_without_centering(self):
        """Without centering, raw and transformed should be identical."""
        X_full = np.array([[1.0, 0.0], [0.707, 0.707]])

        raw, transformed = compute_transition_score_raw_and_transformed(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.4, weight_mid_mid=0.3, weight_full_full=0.3,
            center_transitions=False
        )

        assert abs(raw - transformed) < 0.001


# =============================================================================
# Bridge Scoring Tests
# =============================================================================

class TestBridgeScoring:
    """Test bridgeability scoring."""

    def test_perfect_bridgeability(self):
        """Test bridgeability with perfect alignment."""
        X_full = np.array([[1.0, 0.0], [1.0, 0.0]])

        score = compute_bridgeability_score(
            idx_a=0, idx_b=1,
            X_full_norm=X_full, X_start_norm=None, X_end_norm=None
        )

        # Both direct and full similarity are 1.0
        # Expected: 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert abs(score - 1.0) < 0.01

    def test_orthogonal_bridgeability(self):
        """Test bridgeability with orthogonal vectors."""
        X_full = np.array([[1.0, 0.0], [0.0, 1.0]])

        score = compute_bridgeability_score(
            idx_a=0, idx_b=1,
            X_full_norm=X_full, X_start_norm=None, X_end_norm=None
        )

        # Both similarities are 0.0
        assert abs(score - 0.0) < 0.01

    def test_bridgeability_with_segments(self):
        """Test bridgeability using end-start segments."""
        X_full = np.array([[1.0, 0.0], [0.8, 0.6]])   # Moderate similarity
        X_start = np.array([[1.0, 0.0], [1.0, 0.0]])  # Perfect start
        X_end = np.array([[0.0, 1.0], [1.0, 0.0]])    # Orthogonal end-start

        score = compute_bridgeability_score(
            idx_a=0, idx_b=1,
            X_full_norm=X_full, X_start_norm=X_start, X_end_norm=X_end
        )

        # Direct (end-start): 0.0, Full: 0.8
        # Note: [1,0]·[0.8,0.6] = 0.8, not 0.96
        # Expected: 0.6 * 0.0 + 0.4 * 0.8 = 0.32
        assert abs(score - 0.32) < 0.01

    def test_bridgeability_favors_direct_transition(self):
        """Test that direct transition is weighted more than full similarity."""
        X_full = np.array([[1.0, 0.0], [0.707, 0.707]])
        X_start = np.array([[1.0, 0.0], [1.0, 0.0]])   # Perfect direct
        X_end = np.array([[1.0, 0.0], [1.0, 0.0]])     # Perfect direct

        score = compute_bridgeability_score(
            idx_a=0, idx_b=1,
            X_full_norm=X_full, X_start_norm=X_start, X_end_norm=X_end
        )

        # Direct: 1.0, Full: ~0.707
        # Expected: 0.6 * 1.0 + 0.4 * 0.707 = 0.8828
        expected = 0.6 * 1.0 + 0.4 * 0.707
        assert abs(score - expected) < 0.01


# =============================================================================
# Constraint Tests
# =============================================================================

class TestTransitionWeights:
    """Test TransitionWeights dataclass."""

    def test_valid_weights(self):
        """Valid weights should sum to ~1.0."""
        weights = TransitionWeights(
            weight_end_start=0.4,
            weight_mid_mid=0.3,
            weight_full_full=0.3
        )
        assert weights.weight_end_start == 0.4
        assert weights.weight_mid_mid == 0.3
        assert weights.weight_full_full == 0.3

    def test_weights_sum_validation(self):
        """Weights should sum to ~1.0."""
        # Should raise ValueError if sum is too far from 1.0
        with pytest.raises(ValueError, match="should sum to"):
            TransitionWeights(
                weight_end_start=0.5,
                weight_mid_mid=0.5,
                weight_full_full=0.5  # Sum = 1.5, too high
            )

    def test_frozen_dataclass(self):
        """TransitionWeights should be immutable."""
        weights = TransitionWeights(0.4, 0.3, 0.3)
        with pytest.raises(Exception):  # FrozenInstanceError
            weights.weight_end_start = 0.5


class TestScoringConstraints:
    """Test ScoringConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = ScoringConstraints()
        assert constraints.min_gap == 6
        assert constraints.bridge_floor == 0.03
        assert constraints.transition_floor == 0.20
        assert constraints.center_transitions is True

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = ScoringConstraints(
            min_gap=10,
            bridge_floor=0.10,
            transition_floor=0.30
        )
        assert constraints.min_gap == 10
        assert constraints.bridge_floor == 0.10
        assert constraints.transition_floor == 0.30

    def test_invalid_bridge_floor(self):
        """bridge_floor must be in [0, 1]."""
        with pytest.raises(ValueError, match="bridge_floor"):
            ScoringConstraints(bridge_floor=1.5)

    def test_invalid_transition_floor(self):
        """transition_floor must be in [0, 1]."""
        with pytest.raises(ValueError, match="transition_floor"):
            ScoringConstraints(transition_floor=-0.1)

    def test_invalid_min_gap(self):
        """min_gap must be >= 0."""
        with pytest.raises(ValueError, match="min_gap"):
            ScoringConstraints(min_gap=-1)


class TestSeedOrderingConfig:
    """Test SeedOrderingConfig dataclass."""

    def test_default_config(self):
        """Test default seed ordering config."""
        config = SeedOrderingConfig()
        assert config.max_exhaustive_search == 6
        assert config.use_greedy_for_large is True

    def test_invalid_max_exhaustive_search(self):
        """max_exhaustive_search must be >= 1."""
        with pytest.raises(ValueError, match="max_exhaustive_search"):
            SeedOrderingConfig(max_exhaustive_search=0)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_zero_vectors(self):
        """Test scoring with zero vectors (edge case)."""
        # In practice, vectors should be L2-normalized, but test robustness
        X_full = np.array([[0.0, 0.0], [1.0, 0.0]])

        score = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.0, weight_mid_mid=0.0, weight_full_full=1.0,
            center_transitions=False
        )

        # Dot product with zero vector is 0
        assert score == 0.0

    def test_single_dimension(self):
        """Test with 1D vectors."""
        X_full = np.array([[1.0], [-1.0]])

        score = compute_transition_score(
            idx_a=0, idx_b=1,
            X_full=X_full, X_start=None, X_mid=None, X_end=None,
            weight_end_start=0.0, weight_mid_mid=0.0, weight_full_full=1.0,
            center_transitions=False
        )

        # Opposite directions: cos(180°) = -1
        assert abs(score - (-1.0)) < 0.01
