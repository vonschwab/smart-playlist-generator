"""
Unit tests for Beat3TowerNormalizer.
"""

import numpy as np
import pytest

from src.features.beat3tower_normalizer import (
    Beat3TowerNormalizer,
    NormalizerConfig,
    TowerStats,
    compute_tower_calibration_stats,
    l2_normalize,
)


class TestBeat3TowerNormalizer:
    """Tests for Beat3TowerNormalizer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample tower data."""
        np.random.seed(42)
        n_samples = 100

        # Simulate realistic feature distributions
        X_rhythm = np.random.randn(n_samples, 21) * 0.5 + 0.3
        X_timbre = np.random.randn(n_samples, 83) * 1.5 + 0.1
        X_harmony = np.random.randn(n_samples, 33) * 0.8 + 0.2

        return X_rhythm, X_timbre, X_harmony

    def test_fit_stores_statistics(self, sample_data):
        """Test that fit() stores median and IQR for each tower."""
        X_rhythm, X_timbre, X_harmony = sample_data

        normalizer = Beat3TowerNormalizer()
        normalizer.fit(X_rhythm, X_timbre, X_harmony)

        assert normalizer._fitted
        assert "rhythm" in normalizer.tower_stats
        assert "timbre" in normalizer.tower_stats
        assert "harmony" in normalizer.tower_stats

        # Check dimensions
        assert normalizer.tower_stats["rhythm"].median.shape == (21,)
        assert normalizer.tower_stats["timbre"].median.shape == (83,)
        assert normalizer.tower_stats["harmony"].median.shape == (33,)

    def test_transform_output_shape(self, sample_data):
        """Test that transform() produces correct output shapes."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(use_pca_whitening=False)
        normalizer = Beat3TowerNormalizer(config)
        normalizer.fit(X_rhythm, X_timbre, X_harmony)

        X_r_norm, X_t_norm, X_h_norm = normalizer.transform(
            X_rhythm, X_timbre, X_harmony
        )

        # Without PCA, dimensions should be preserved
        assert X_r_norm.shape == X_rhythm.shape
        assert X_t_norm.shape == X_timbre.shape
        assert X_h_norm.shape == X_harmony.shape

    def test_transform_with_pca_reduces_dims(self, sample_data):
        """Test that PCA reduces dimensions when enabled."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(
            use_pca_whitening=True,
            pca_variance_retain=0.80,
            pca_min_components=5,
        )
        normalizer = Beat3TowerNormalizer(config)
        normalizer.fit(X_rhythm, X_timbre, X_harmony)

        X_r_norm, X_t_norm, X_h_norm = normalizer.transform(
            X_rhythm, X_timbre, X_harmony
        )

        # With PCA, dimensions should be reduced
        assert X_r_norm.shape[0] == X_rhythm.shape[0]
        assert X_r_norm.shape[1] <= X_rhythm.shape[1]
        assert X_r_norm.shape[1] >= 5  # min_components

    def test_fit_transform_same_as_separate(self, sample_data):
        """Test that fit_transform gives same result as fit then transform."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(use_pca_whitening=True, random_seed=42)

        # Fit and transform separately
        normalizer1 = Beat3TowerNormalizer(config)
        normalizer1.fit(X_rhythm, X_timbre, X_harmony)
        result1 = normalizer1.transform(X_rhythm, X_timbre, X_harmony)

        # Fit_transform
        normalizer2 = Beat3TowerNormalizer(config)
        result2 = normalizer2.fit_transform(X_rhythm, X_timbre, X_harmony)

        for r1, r2 in zip(result1, result2):
            np.testing.assert_array_almost_equal(r1, r2)

    def test_l2_normalization(self, sample_data):
        """Test that L2 normalization produces unit vectors."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(l2_normalize=True)
        normalizer = Beat3TowerNormalizer(config)
        X_r_norm, X_t_norm, X_h_norm = normalizer.fit_transform(
            X_rhythm, X_timbre, X_harmony
        )

        # Check each row has unit norm
        r_norms = np.linalg.norm(X_r_norm, axis=1)
        np.testing.assert_array_almost_equal(r_norms, np.ones(len(X_rhythm)))

    def test_clipping_limits_outliers(self, sample_data):
        """Test that clipping limits extreme values."""
        X_rhythm, X_timbre, X_harmony = sample_data

        # Add outliers
        X_rhythm_outlier = X_rhythm.copy()
        X_rhythm_outlier[0, 0] = 1000  # Extreme outlier

        config = NormalizerConfig(
            clip_sigma=3.0,
            use_pca_whitening=False,
            l2_normalize=False,
        )
        normalizer = Beat3TowerNormalizer(config)
        X_r_norm, _, _ = normalizer.fit_transform(
            X_rhythm_outlier, X_timbre, X_harmony
        )

        # After standardization and clipping, max should be ~3
        assert np.abs(X_r_norm).max() <= 3.0 + 0.01

    def test_serialization_roundtrip(self, sample_data):
        """Test that normalizer can be serialized and reconstructed."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(use_pca_whitening=True)
        normalizer = Beat3TowerNormalizer(config)
        normalizer.fit(X_rhythm, X_timbre, X_harmony)

        # Serialize
        params = normalizer.get_params()

        # Reconstruct
        normalizer2 = Beat3TowerNormalizer.from_params(params)

        # Transform with both should give same result
        result1 = normalizer.transform(X_rhythm, X_timbre, X_harmony)
        result2 = normalizer2.transform(X_rhythm, X_timbre, X_harmony)

        for r1, r2 in zip(result1, result2):
            np.testing.assert_array_almost_equal(r1, r2)

    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        X_rhythm, X_timbre, X_harmony = sample_data

        normalizer = Beat3TowerNormalizer()

        with pytest.raises(RuntimeError, match="not fitted"):
            normalizer.transform(X_rhythm, X_timbre, X_harmony)

    def test_get_output_dims(self, sample_data):
        """Test get_output_dims returns correct dimensions."""
        X_rhythm, X_timbre, X_harmony = sample_data

        config = NormalizerConfig(
            use_pca_whitening=True,
            pca_min_components=8,
        )
        normalizer = Beat3TowerNormalizer(config)
        normalizer.fit(X_rhythm, X_timbre, X_harmony)

        dims = normalizer.get_output_dims()

        assert "rhythm" in dims
        assert "timbre" in dims
        assert "harmony" in dims
        assert all(d >= 8 for d in dims.values())


class TestTowerStats:
    """Tests for TowerStats dataclass."""

    def test_to_dict_without_pca(self):
        """Test serialization without PCA."""
        stats = TowerStats(
            tower_name="rhythm",
            n_features_input=21,
            n_features_output=21,
            median=np.array([0.1, 0.2, 0.3]),
            iqr=np.array([0.5, 0.6, 0.7]),
        )

        d = stats.to_dict()

        assert d["tower_name"] == "rhythm"
        assert d["n_features_input"] == 21
        assert "pca_components" not in d

    def test_to_dict_with_pca(self):
        """Test serialization with PCA."""
        stats = TowerStats(
            tower_name="timbre",
            n_features_input=83,
            n_features_output=20,
            median=np.zeros(83),
            iqr=np.ones(83),
            pca_components=np.eye(20, 83),
            pca_explained_variance=np.ones(20),
            pca_mean=np.zeros(83),
        )

        d = stats.to_dict()

        assert "pca_components" in d
        assert len(d["pca_components"]) == 20

    def test_from_dict_roundtrip(self):
        """Test roundtrip serialization."""
        original = TowerStats(
            tower_name="harmony",
            n_features_input=33,
            n_features_output=15,
            median=np.random.randn(33),
            iqr=np.abs(np.random.randn(33)) + 0.1,
            pca_components=np.random.randn(15, 33),
            pca_explained_variance=np.abs(np.random.randn(15)) + 0.1,
            pca_mean=np.random.randn(33),
        )

        d = original.to_dict()
        restored = TowerStats.from_dict(d)

        assert restored.tower_name == original.tower_name
        np.testing.assert_array_almost_equal(restored.median, original.median)
        np.testing.assert_array_almost_equal(restored.pca_components, original.pca_components)


class TestCalibrationStats:
    """Tests for calibration statistics computation."""

    def test_compute_calibration_returns_all_towers(self):
        """Test that calibration returns stats for all towers."""
        np.random.seed(42)
        n = 100

        # Create L2-normalized random data
        X_r = l2_normalize(np.random.randn(n, 10))
        X_t = l2_normalize(np.random.randn(n, 20))
        X_h = l2_normalize(np.random.randn(n, 15))

        calib = compute_tower_calibration_stats(X_r, X_t, X_h, n_pairs=1000)

        assert "rhythm" in calib
        assert "timbre" in calib
        assert "harmony" in calib

        for tower in ["rhythm", "timbre", "harmony"]:
            assert "random_mean" in calib[tower]
            assert "random_std" in calib[tower]

    def test_random_mean_near_zero_for_l2_normalized(self):
        """Test that random similarity mean is near zero for L2-normalized data."""
        np.random.seed(42)
        n = 500

        # Random L2-normalized vectors should have near-zero mean similarity
        X = l2_normalize(np.random.randn(n, 50))

        calib = compute_tower_calibration_stats(X, X, X, n_pairs=5000)

        # Mean should be near 0 for high-dimensional random vectors
        assert abs(calib["rhythm"]["random_mean"]) < 0.15

    def test_std_positive(self):
        """Test that random similarity std is positive."""
        np.random.seed(42)
        n = 100

        X = l2_normalize(np.random.randn(n, 20))
        calib = compute_tower_calibration_stats(X, X, X, n_pairs=1000)

        for tower in ["rhythm", "timbre", "harmony"]:
            assert calib[tower]["random_std"] > 0


class TestL2Normalize:
    """Tests for l2_normalize function."""

    def test_output_unit_norm(self):
        """Test that output has unit norm rows."""
        X = np.random.randn(10, 5)
        X_norm = l2_normalize(X)

        norms = np.linalg.norm(X_norm, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10))

    def test_handles_zero_vector(self):
        """Test that zero vectors are handled gracefully."""
        X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        X_norm = l2_normalize(X)

        # Zero vector should remain zero (or very small)
        assert np.linalg.norm(X_norm[0]) < 1e-8
        # Non-zero vectors should have unit norm
        np.testing.assert_almost_equal(np.linalg.norm(X_norm[1]), 1.0)

    def test_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        X = np.array([[3, 4, 0], [0, 5, 12]])
        X_norm = l2_normalize(X)

        # Direction should be preserved
        expected0 = np.array([0.6, 0.8, 0])
        expected1 = np.array([0, 5 / 13, 12 / 13])

        np.testing.assert_array_almost_equal(X_norm[0], expected0)
        np.testing.assert_array_almost_equal(X_norm[1], expected1)
