"""
Beat 3-Tower Feature Types
===========================

Data classes for the 3-tower beat-synchronous sonic features:
- Rhythm tower: onset, tempogram, beat stability
- Timbre tower: MFCC, spectral features
- Harmony tower: chroma, tonnetz

These types provide structured access to features and easy conversion to vectors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class BPMInfo:
    """BPM information with half/double tempo awareness."""
    primary_bpm: float
    half_tempo_likely: bool = False
    double_tempo_likely: bool = False
    tempo_stability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_bpm': self.primary_bpm,
            'half_tempo_likely': self.half_tempo_likely,
            'double_tempo_likely': self.double_tempo_likely,
            'tempo_stability': self.tempo_stability,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BPMInfo':
        return cls(
            primary_bpm=d.get('primary_bpm', 0.0),
            half_tempo_likely=d.get('half_tempo_likely', False),
            double_tempo_likely=d.get('double_tempo_likely', False),
            tempo_stability=d.get('tempo_stability', 1.0),
        )


# Class-level constant for rhythm feature ordering
_RHYTHM_FEATURE_ORDER: List[str] = [
    'onset_median', 'onset_iqr', 'onset_std', 'onset_p10', 'onset_p90',
    'tempo_peak1_lag', 'tempo_peak2_lag', 'tempo_peak_ratio',
    'tempo_acf_lag1', 'tempo_acf_lag2', 'tempo_acf_lag3', 'tempo_acf_lag4', 'tempo_acf_lag5',
    'beat_interval_cv', 'beat_interval_median',
    'beat_strength_median', 'beat_strength_iqr',
    'onset_rate', 'rhythm_entropy',
    'bpm', 'tempo_stability',
]


@dataclass
class RhythmTowerFeatures:
    """
    Rhythm tower features (~25-35 dimensions).

    Captures:
    - Onset envelope statistics
    - Tempogram peaks and autocorrelation
    - Beat stability metrics
    - Rhythmic complexity
    """
    # Onset envelope stats
    onset_median: float = 0.0
    onset_iqr: float = 0.0
    onset_std: float = 0.0
    onset_p10: float = 0.0
    onset_p90: float = 0.0

    # Tempogram peaks
    tempo_peak1_lag: float = 0.0
    tempo_peak2_lag: float = 0.0
    tempo_peak_ratio: float = 1.0

    # Tempogram ACF (autocorrelation lags)
    tempo_acf_lag1: float = 0.0
    tempo_acf_lag2: float = 0.0
    tempo_acf_lag3: float = 0.0
    tempo_acf_lag4: float = 0.0
    tempo_acf_lag5: float = 0.0

    # Beat interval stability
    beat_interval_cv: float = 0.0
    beat_interval_median: float = 0.0

    # Beat strength
    beat_strength_median: float = 0.0
    beat_strength_iqr: float = 0.0

    # Onset rate (busyness)
    onset_rate: float = 0.0

    # Rhythmic complexity
    rhythm_entropy: float = 0.0

    # BPM features (stored separately but included here for vector)
    bpm: float = 0.0
    tempo_stability: float = 1.0

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector in canonical order."""
        values = [getattr(self, name) for name in _RHYTHM_FEATURE_ORDER]
        return np.array(values, dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {name: getattr(self, name) for name in _RHYTHM_FEATURE_ORDER}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RhythmTowerFeatures':
        """Create from dictionary."""
        return cls(**{k: d.get(k, 0.0) for k in _RHYTHM_FEATURE_ORDER})

    @classmethod
    def feature_names(cls) -> List[str]:
        """Get ordered feature names."""
        return _RHYTHM_FEATURE_ORDER.copy()

    @classmethod
    def n_features(cls) -> int:
        """Get number of features."""
        return len(_RHYTHM_FEATURE_ORDER)


@dataclass
class TimbreTowerFeatures:
    """
    Timbre tower features (~60-80 dimensions).

    Captures:
    - MFCC (20 coefficients) with median/IQR
    - MFCC delta (dynamics)
    - Spectral contrast, rolloff, centroid, bandwidth, flux
    - Zero crossing rate
    """
    # MFCC features (20 coefficients x 2 stats = 40 dims)
    mfcc_median: np.ndarray = field(default_factory=lambda: np.zeros(20))
    mfcc_iqr: np.ndarray = field(default_factory=lambda: np.zeros(20))

    # MFCC delta (20 dims)
    mfcc_delta_median: np.ndarray = field(default_factory=lambda: np.zeros(20))

    # Spectral contrast (7 bands x 2 stats = 14 dims)
    spec_contrast_median: np.ndarray = field(default_factory=lambda: np.zeros(7))
    spec_contrast_iqr: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Other spectral features (2 dims each = 8 dims)
    spec_rolloff_median: float = 0.0
    spec_rolloff_iqr: float = 0.0
    spec_centroid_median: float = 0.0
    spec_centroid_iqr: float = 0.0
    spec_bandwidth_median: float = 0.0
    spec_bandwidth_iqr: float = 0.0
    spec_flux_median: float = 0.0
    spec_flux_iqr: float = 0.0

    # Zero crossing rate (1 dim)
    zcr_median: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector in canonical order."""
        parts = [
            self.mfcc_median.flatten(),
            self.mfcc_iqr.flatten(),
            self.mfcc_delta_median.flatten(),
            self.spec_contrast_median.flatten(),
            self.spec_contrast_iqr.flatten(),
            np.array([
                self.spec_rolloff_median, self.spec_rolloff_iqr,
                self.spec_centroid_median, self.spec_centroid_iqr,
                self.spec_bandwidth_median, self.spec_bandwidth_iqr,
                self.spec_flux_median, self.spec_flux_iqr,
                self.zcr_median,
            ]),
        ]
        return np.concatenate(parts).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            'mfcc_median': self.mfcc_median.tolist(),
            'mfcc_iqr': self.mfcc_iqr.tolist(),
            'mfcc_delta_median': self.mfcc_delta_median.tolist(),
            'spec_contrast_median': self.spec_contrast_median.tolist(),
            'spec_contrast_iqr': self.spec_contrast_iqr.tolist(),
            'spec_rolloff_median': self.spec_rolloff_median,
            'spec_rolloff_iqr': self.spec_rolloff_iqr,
            'spec_centroid_median': self.spec_centroid_median,
            'spec_centroid_iqr': self.spec_centroid_iqr,
            'spec_bandwidth_median': self.spec_bandwidth_median,
            'spec_bandwidth_iqr': self.spec_bandwidth_iqr,
            'spec_flux_median': self.spec_flux_median,
            'spec_flux_iqr': self.spec_flux_iqr,
            'zcr_median': self.zcr_median,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TimbreTowerFeatures':
        """Create from dictionary."""
        return cls(
            mfcc_median=np.array(d.get('mfcc_median', np.zeros(20))),
            mfcc_iqr=np.array(d.get('mfcc_iqr', np.zeros(20))),
            mfcc_delta_median=np.array(d.get('mfcc_delta_median', np.zeros(20))),
            spec_contrast_median=np.array(d.get('spec_contrast_median', np.zeros(7))),
            spec_contrast_iqr=np.array(d.get('spec_contrast_iqr', np.zeros(7))),
            spec_rolloff_median=d.get('spec_rolloff_median', 0.0),
            spec_rolloff_iqr=d.get('spec_rolloff_iqr', 0.0),
            spec_centroid_median=d.get('spec_centroid_median', 0.0),
            spec_centroid_iqr=d.get('spec_centroid_iqr', 0.0),
            spec_bandwidth_median=d.get('spec_bandwidth_median', 0.0),
            spec_bandwidth_iqr=d.get('spec_bandwidth_iqr', 0.0),
            spec_flux_median=d.get('spec_flux_median', 0.0),
            spec_flux_iqr=d.get('spec_flux_iqr', 0.0),
            zcr_median=d.get('zcr_median', 0.0),
        )

    @classmethod
    def n_features(cls) -> int:
        """Get number of features (20+20+20+7+7+9 = 83)."""
        return 20 + 20 + 20 + 7 + 7 + 9


@dataclass
class HarmonyTowerFeatures:
    """
    Harmony tower features (~30-40 dimensions).

    Captures:
    - Chroma CQT (12 bins) with median/IQR
    - Chroma entropy and statistics
    - Tonnetz (6 dims)
    """
    # Chroma CQT (12 bins x 2 stats = 24 dims)
    chroma_median: np.ndarray = field(default_factory=lambda: np.zeros(12))
    chroma_iqr: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Chroma statistics (3 dims)
    chroma_entropy: float = 0.0
    chroma_peak_count: float = 0.0
    key_strength: float = 0.0

    # Tonnetz (6 dims)
    tonnetz_median: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector in canonical order."""
        parts = [
            self.chroma_median.flatten(),
            self.chroma_iqr.flatten(),
            np.array([self.chroma_entropy, self.chroma_peak_count, self.key_strength]),
            self.tonnetz_median.flatten(),
        ]
        return np.concatenate(parts).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            'chroma_median': self.chroma_median.tolist(),
            'chroma_iqr': self.chroma_iqr.tolist(),
            'chroma_entropy': self.chroma_entropy,
            'chroma_peak_count': self.chroma_peak_count,
            'key_strength': self.key_strength,
            'tonnetz_median': self.tonnetz_median.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HarmonyTowerFeatures':
        """Create from dictionary."""
        return cls(
            chroma_median=np.array(d.get('chroma_median', np.zeros(12))),
            chroma_iqr=np.array(d.get('chroma_iqr', np.zeros(12))),
            chroma_entropy=d.get('chroma_entropy', 0.0),
            chroma_peak_count=d.get('chroma_peak_count', 0.0),
            key_strength=d.get('key_strength', 0.0),
            tonnetz_median=np.array(d.get('tonnetz_median', np.zeros(6))),
        )

    @classmethod
    def n_features(cls) -> int:
        """Get number of features (12+12+3+6 = 33)."""
        return 12 + 12 + 3 + 6


@dataclass
class Beat3TowerFeatures:
    """
    Complete 3-tower beat-synchronous features for a track/segment.

    Total dimensions: ~21 + 83 + 33 = 137 (before PCA)
    """
    rhythm: RhythmTowerFeatures
    timbre: TimbreTowerFeatures
    harmony: HarmonyTowerFeatures
    bpm_info: BPMInfo
    n_beats: int = 0
    extraction_method: str = 'beat3tower'

    def to_vector(self) -> np.ndarray:
        """Concatenate all towers into single vector."""
        return np.concatenate([
            self.rhythm.to_vector(),
            self.timbre.to_vector(),
            self.harmony.to_vector(),
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            'extraction_method': self.extraction_method,
            'n_beats': self.n_beats,
            'bpm_info': self.bpm_info.to_dict(),
            'rhythm': self.rhythm.to_dict(),
            'timbre': self.timbre.to_dict(),
            'harmony': self.harmony.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Beat3TowerFeatures':
        """Create from dictionary."""
        return cls(
            rhythm=RhythmTowerFeatures.from_dict(d.get('rhythm', {})),
            timbre=TimbreTowerFeatures.from_dict(d.get('timbre', {})),
            harmony=HarmonyTowerFeatures.from_dict(d.get('harmony', {})),
            bpm_info=BPMInfo.from_dict(d.get('bpm_info', {})),
            n_beats=d.get('n_beats', 0),
            extraction_method=d.get('extraction_method', 'beat3tower'),
        )

    @classmethod
    def n_features(cls) -> int:
        """Get total number of features."""
        return (
            RhythmTowerFeatures.n_features() +
            TimbreTowerFeatures.n_features() +
            HarmonyTowerFeatures.n_features()
        )

    @classmethod
    def tower_dims(cls) -> Dict[str, int]:
        """Get dimension counts per tower."""
        return {
            'rhythm': RhythmTowerFeatures.n_features(),
            'timbre': TimbreTowerFeatures.n_features(),
            'harmony': HarmonyTowerFeatures.n_features(),
        }


class InsufficientBeatsError(Exception):
    """Raised when audio has too few beats for beat-sync extraction."""
    pass
