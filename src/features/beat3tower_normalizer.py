"""
Beat 3-Tower Feature Normalizer
================================

Per-tower normalization with robust statistics and optional PCA whitening.

This module provides:
- Robust standardization using median and IQR (resistant to outliers)
- Optional sigma clipping to limit extreme values
- Optional PCA whitening for decorrelated features
- Serialization of normalization parameters for reproducibility

Design Decisions:
- IQR scaling: Uses (value - median) / (IQR / 1.35) which approximates
  standard deviation for normal distributions but is robust to outliers
- Per-tower normalization: Each tower (rhythm/timbre/harmony) is normalized
  independently to prevent one tower from dominating
- Clip before PCA: Limits extreme outliers before PCA to prevent them from
  distorting the principal components
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class NormalizerConfig:
    """Configuration for 3-tower normalization."""

    # Clipping
    clip_sigma: float = 3.0  # Clip to +/- N sigma after standardization

    # PCA whitening
    use_pca_whitening: bool = True
    pca_variance_retain: float = 0.95  # Retain 95% of variance
    pca_min_components: int = 8  # Minimum components per tower
    pca_max_components: Optional[int] = None  # Cap on components (None = no cap)

    # L2 normalization
    l2_normalize: bool = True  # Apply L2 norm after all transformations

    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class TowerStats:
    """Statistics for a single tower's normalization."""

    tower_name: str
    n_features_input: int
    n_features_output: int

    # Robust statistics
    median: np.ndarray = field(default_factory=lambda: np.array([]))
    iqr: np.ndarray = field(default_factory=lambda: np.array([]))

    # PCA components (if whitening enabled)
    pca_components: Optional[np.ndarray] = None
    pca_explained_variance: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        result = {
            'tower_name': self.tower_name,
            'n_features_input': self.n_features_input,
            'n_features_output': self.n_features_output,
            'median': self.median.tolist(),
            'iqr': self.iqr.tolist(),
        }
        if self.pca_components is not None:
            result['pca_components'] = self.pca_components.tolist()
            result['pca_explained_variance'] = self.pca_explained_variance.tolist()
            result['pca_mean'] = self.pca_mean.tolist()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TowerStats':
        """Deserialize from dictionary."""
        stats = cls(
            tower_name=d['tower_name'],
            n_features_input=d['n_features_input'],
            n_features_output=d['n_features_output'],
            median=np.array(d['median']),
            iqr=np.array(d['iqr']),
        )
        if 'pca_components' in d:
            stats.pca_components = np.array(d['pca_components'])
            stats.pca_explained_variance = np.array(d['pca_explained_variance'])
            stats.pca_mean = np.array(d['pca_mean'])
        return stats


class Beat3TowerNormalizer:
    """
    Normalizes 3-tower features with robust statistics.

    Workflow:
    1. Fit: Compute per-tower median, IQR, and optionally PCA from training data
    2. Transform: Apply normalization to new data using fitted parameters

    Example:
        normalizer = Beat3TowerNormalizer(config)
        normalizer.fit(X_rhythm, X_timbre, X_harmony)
        X_r_norm, X_t_norm, X_h_norm = normalizer.transform(X_rhythm, X_timbre, X_harmony)
    """

    def __init__(self, config: Optional[NormalizerConfig] = None):
        """
        Initialize normalizer.

        Args:
            config: Normalization configuration. Uses defaults if None.
        """
        self.config = config or NormalizerConfig()
        self.tower_stats: Dict[str, TowerStats] = {}
        self._fitted = False

    def fit(
        self,
        X_rhythm: np.ndarray,
        X_timbre: np.ndarray,
        X_harmony: np.ndarray,
    ) -> 'Beat3TowerNormalizer':
        """
        Fit normalization parameters from training data.

        Args:
            X_rhythm: Rhythm tower features (n_samples, n_rhythm_features)
            X_timbre: Timbre tower features (n_samples, n_timbre_features)
            X_harmony: Harmony tower features (n_samples, n_harmony_features)

        Returns:
            Self for chaining
        """
        self.tower_stats['rhythm'] = self._fit_tower(X_rhythm, 'rhythm')
        self.tower_stats['timbre'] = self._fit_tower(X_timbre, 'timbre')
        self.tower_stats['harmony'] = self._fit_tower(X_harmony, 'harmony')
        self._fitted = True
        return self

    def transform(
        self,
        X_rhythm: np.ndarray,
        X_timbre: np.ndarray,
        X_harmony: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply normalization to feature matrices.

        Args:
            X_rhythm: Rhythm tower features
            X_timbre: Timbre tower features
            X_harmony: Harmony tower features

        Returns:
            Tuple of (normalized_rhythm, normalized_timbre, normalized_harmony)
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        X_r_norm = self._transform_tower(X_rhythm, 'rhythm')
        X_t_norm = self._transform_tower(X_timbre, 'timbre')
        X_h_norm = self._transform_tower(X_harmony, 'harmony')

        return X_r_norm, X_t_norm, X_h_norm

    def fit_transform(
        self,
        X_rhythm: np.ndarray,
        X_timbre: np.ndarray,
        X_harmony: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.

        Args:
            X_rhythm: Rhythm tower features
            X_timbre: Timbre tower features
            X_harmony: Harmony tower features

        Returns:
            Tuple of (normalized_rhythm, normalized_timbre, normalized_harmony)
        """
        self.fit(X_rhythm, X_timbre, X_harmony)
        return self.transform(X_rhythm, X_timbre, X_harmony)

    def _fit_tower(self, X: np.ndarray, tower_name: str) -> TowerStats:
        """
        Fit normalization parameters for a single tower.

        Args:
            X: Feature matrix (n_samples, n_features)
            tower_name: Name of tower ('rhythm', 'timbre', 'harmony')

        Returns:
            TowerStats with fitted parameters
        """
        n_samples, n_features = X.shape

        # Step 1: Compute robust statistics
        median = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25

        # Avoid division by zero for constant features
        iqr = np.maximum(iqr, 1e-10)

        # Initialize stats
        stats = TowerStats(
            tower_name=tower_name,
            n_features_input=n_features,
            n_features_output=n_features,  # Updated below if PCA
            median=median,
            iqr=iqr,
        )

        # Step 2: Optionally fit PCA
        if self.config.use_pca_whitening:
            # Apply robust standardization first
            X_robust = (X - median) / (iqr / 1.35)  # 1.35 converts IQR to ~std for normal
            X_clipped = np.clip(X_robust, -self.config.clip_sigma, self.config.clip_sigma)

            # Determine number of components
            n_components = self._determine_n_components(X_clipped, tower_name)

            # Fit PCA with whitening
            pca = PCA(n_components=n_components, whiten=True, random_state=self.config.random_seed)
            pca.fit(X_clipped)

            # Store PCA parameters
            stats.pca_components = pca.components_
            stats.pca_explained_variance = pca.explained_variance_
            stats.pca_mean = pca.mean_
            stats.n_features_output = n_components

        return stats

    def _transform_tower(self, X: np.ndarray, tower_name: str) -> np.ndarray:
        """
        Apply normalization to a single tower.

        Args:
            X: Feature matrix (n_samples, n_features)
            tower_name: Name of tower

        Returns:
            Normalized feature matrix
        """
        stats = self.tower_stats[tower_name]

        # Step 1: Robust standardization
        X_robust = (X - stats.median) / (stats.iqr / 1.35)

        # Step 2: Clip outliers
        X_clipped = np.clip(X_robust, -self.config.clip_sigma, self.config.clip_sigma)

        # Step 3: Optional PCA whitening
        if stats.pca_components is not None:
            # Apply PCA manually (sklearn PCA.transform would work but we store raw params)
            X_centered = X_clipped - stats.pca_mean
            X_pca = X_centered @ stats.pca_components.T
            # Apply whitening (divide by sqrt of explained variance)
            X_whitened = X_pca / (np.sqrt(stats.pca_explained_variance) + 1e-10)
            X_out = X_whitened.astype(np.float32)
        else:
            X_out = X_clipped.astype(np.float32)

        # Step 4: Optional L2 normalization
        if self.config.l2_normalize:
            norms = np.linalg.norm(X_out, axis=1, keepdims=True)
            X_out = X_out / (norms + 1e-10)

        return X_out

    def _determine_n_components(self, X: np.ndarray, tower_name: str) -> int:
        """
        Determine number of PCA components to retain.

        Uses config.pca_variance_retain to select components that explain
        at least that fraction of variance, with min/max constraints.
        """
        # First pass PCA to get explained variance ratios
        pca_full = PCA(random_state=self.config.random_seed)
        pca_full.fit(X)

        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_for_variance = np.searchsorted(cumsum, self.config.pca_variance_retain) + 1

        # Apply constraints
        n_components = max(n_for_variance, self.config.pca_min_components)
        n_components = min(n_components, X.shape[1])  # Can't exceed input dims

        if self.config.pca_max_components is not None:
            n_components = min(n_components, self.config.pca_max_components)

        return n_components

    def get_output_dims(self) -> Dict[str, int]:
        """Get output dimensions for each tower after normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")
        return {
            name: stats.n_features_output
            for name, stats in self.tower_stats.items()
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Get all parameters for serialization.

        Returns:
            Dictionary suitable for saving to NPZ or JSON
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")

        return {
            'config': {
                'clip_sigma': self.config.clip_sigma,
                'use_pca_whitening': self.config.use_pca_whitening,
                'pca_variance_retain': self.config.pca_variance_retain,
                'pca_min_components': self.config.pca_min_components,
                'pca_max_components': self.config.pca_max_components,
                'l2_normalize': self.config.l2_normalize,
                'random_seed': self.config.random_seed,
            },
            'tower_stats': {
                name: stats.to_dict()
                for name, stats in self.tower_stats.items()
            },
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'Beat3TowerNormalizer':
        """
        Reconstruct normalizer from saved parameters.

        Args:
            params: Dictionary from get_params()

        Returns:
            Fitted normalizer ready for transform()
        """
        config_dict = params['config']
        config = NormalizerConfig(
            clip_sigma=config_dict['clip_sigma'],
            use_pca_whitening=config_dict['use_pca_whitening'],
            pca_variance_retain=config_dict['pca_variance_retain'],
            pca_min_components=config_dict['pca_min_components'],
            pca_max_components=config_dict.get('pca_max_components'),
            l2_normalize=config_dict['l2_normalize'],
            random_seed=config_dict['random_seed'],
        )

        normalizer = cls(config)
        normalizer.tower_stats = {
            name: TowerStats.from_dict(stats_dict)
            for name, stats_dict in params['tower_stats'].items()
        }
        normalizer._fitted = True

        return normalizer


def compute_tower_calibration_stats(
    X_rhythm: np.ndarray,
    X_timbre: np.ndarray,
    X_harmony: np.ndarray,
    n_pairs: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration statistics for per-tower similarity.

    For each tower, sample random pairs and compute cosine similarity
    to get the mean and std of random similarities. This allows converting
    raw similarities to z-scores for calibrated combination.

    Args:
        X_rhythm: Normalized rhythm features (assumed L2-normalized)
        X_timbre: Normalized timbre features (assumed L2-normalized)
        X_harmony: Normalized harmony features (assumed L2-normalized)
        n_pairs: Number of random pairs to sample
        random_seed: For reproducibility

    Returns:
        Dictionary with calibration stats per tower:
        {
            'rhythm': {'random_mean': 0.05, 'random_std': 0.12},
            'timbre': {'random_mean': 0.03, 'random_std': 0.15},
            'harmony': {'random_mean': 0.02, 'random_std': 0.18},
        }
    """
    rng = np.random.default_rng(random_seed)
    n_samples = X_rhythm.shape[0]

    # Sample random pairs
    idx_a = rng.integers(0, n_samples, size=n_pairs)
    idx_b = rng.integers(0, n_samples, size=n_pairs)

    # Exclude identical pairs
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]

    calibration = {}

    for tower_name, X in [('rhythm', X_rhythm), ('timbre', X_timbre), ('harmony', X_harmony)]:
        # Compute cosine similarities for random pairs
        # Since X is L2-normalized, cosine = dot product
        sims = np.sum(X[idx_a] * X[idx_b], axis=1)

        calibration[tower_name] = {
            'random_mean': float(np.mean(sims)),
            'random_std': float(np.std(sims)),
        }

    return calibration


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-10)
