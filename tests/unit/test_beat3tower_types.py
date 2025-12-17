"""
Unit tests for Beat3Tower data types.
"""

import numpy as np
import pytest

from src.features.beat3tower_types import (
    Beat3TowerFeatures,
    BPMInfo,
    HarmonyTowerFeatures,
    InsufficientBeatsError,
    RhythmTowerFeatures,
    TimbreTowerFeatures,
)


class TestBPMInfo:
    """Tests for BPMInfo dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        bpm = BPMInfo(
            primary_bpm=120.0,
            half_tempo_likely=True,
            double_tempo_likely=False,
            tempo_stability=0.95,
        )

        d = bpm.to_dict()

        assert d["primary_bpm"] == 120.0
        assert d["half_tempo_likely"] is True
        assert d["double_tempo_likely"] is False
        assert d["tempo_stability"] == 0.95

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "primary_bpm": 140.0,
            "half_tempo_likely": False,
            "double_tempo_likely": True,
            "tempo_stability": 0.85,
        }

        bpm = BPMInfo.from_dict(d)

        assert bpm.primary_bpm == 140.0
        assert bpm.half_tempo_likely is False
        assert bpm.double_tempo_likely is True
        assert bpm.tempo_stability == 0.85

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing fields uses defaults."""
        bpm = BPMInfo.from_dict({})

        assert bpm.primary_bpm == 0.0
        assert bpm.half_tempo_likely is False
        assert bpm.double_tempo_likely is False
        assert bpm.tempo_stability == 1.0


class TestRhythmTowerFeatures:
    """Tests for RhythmTowerFeatures dataclass."""

    def test_to_vector_shape(self):
        """Test that to_vector produces correct shape."""
        rhythm = RhythmTowerFeatures(
            onset_median=0.5,
            onset_iqr=0.2,
            bpm=120.0,
        )

        vec = rhythm.to_vector()

        assert vec.shape == (21,)  # 21 rhythm features
        assert vec.dtype == np.float32

    def test_to_dict_roundtrip(self):
        """Test dictionary serialization roundtrip."""
        original = RhythmTowerFeatures(
            onset_median=0.5,
            onset_iqr=0.2,
            onset_std=0.1,
            tempo_peak1_lag=10.0,
            bpm=120.0,
        )

        d = original.to_dict()
        restored = RhythmTowerFeatures.from_dict(d)

        assert restored.onset_median == original.onset_median
        assert restored.onset_iqr == original.onset_iqr
        assert restored.bpm == original.bpm

    def test_feature_names(self):
        """Test that feature_names returns expected fields."""
        names = RhythmTowerFeatures.feature_names()

        assert "onset_median" in names
        assert "bpm" in names
        assert "tempo_stability" in names
        assert len(names) == 21

    def test_n_features(self):
        """Test n_features class method."""
        assert RhythmTowerFeatures.n_features() == 21


class TestTimbreTowerFeatures:
    """Tests for TimbreTowerFeatures dataclass."""

    def test_to_vector_shape(self):
        """Test that to_vector produces correct shape."""
        timbre = TimbreTowerFeatures()
        vec = timbre.to_vector()

        # 20 mfcc_median + 20 mfcc_iqr + 20 mfcc_delta + 7 contrast_median
        # + 7 contrast_iqr + 9 scalar features = 83
        assert vec.shape == (83,)
        assert vec.dtype == np.float32

    def test_to_dict_roundtrip(self):
        """Test dictionary serialization roundtrip."""
        original = TimbreTowerFeatures(
            mfcc_median=np.arange(20, dtype=np.float32),
            mfcc_iqr=np.ones(20, dtype=np.float32) * 0.5,
            spec_centroid_median=2000.0,
        )

        d = original.to_dict()
        restored = TimbreTowerFeatures.from_dict(d)

        np.testing.assert_array_equal(restored.mfcc_median, original.mfcc_median)
        assert restored.spec_centroid_median == original.spec_centroid_median

    def test_n_features(self):
        """Test n_features class method."""
        assert TimbreTowerFeatures.n_features() == 83


class TestHarmonyTowerFeatures:
    """Tests for HarmonyTowerFeatures dataclass."""

    def test_to_vector_shape(self):
        """Test that to_vector produces correct shape."""
        harmony = HarmonyTowerFeatures()
        vec = harmony.to_vector()

        # 12 chroma_median + 12 chroma_iqr + 3 scalars + 6 tonnetz = 33
        assert vec.shape == (33,)
        assert vec.dtype == np.float32

    def test_to_dict_roundtrip(self):
        """Test dictionary serialization roundtrip."""
        original = HarmonyTowerFeatures(
            chroma_median=np.ones(12, dtype=np.float32) / 12,
            chroma_entropy=2.5,
            key_strength=1.8,
        )

        d = original.to_dict()
        restored = HarmonyTowerFeatures.from_dict(d)

        np.testing.assert_array_almost_equal(restored.chroma_median, original.chroma_median)
        assert restored.chroma_entropy == original.chroma_entropy

    def test_n_features(self):
        """Test n_features class method."""
        assert HarmonyTowerFeatures.n_features() == 33


class TestBeat3TowerFeatures:
    """Tests for Beat3TowerFeatures composite dataclass."""

    @pytest.fixture
    def sample_features(self):
        """Create sample 3-tower features."""
        return Beat3TowerFeatures(
            rhythm=RhythmTowerFeatures(bpm=120.0, onset_median=0.5),
            timbre=TimbreTowerFeatures(spec_centroid_median=2000.0),
            harmony=HarmonyTowerFeatures(chroma_entropy=2.5),
            bpm_info=BPMInfo(primary_bpm=120.0, tempo_stability=0.9),
            n_beats=200,
            extraction_method="beat3tower",
        )

    def test_to_vector_shape(self, sample_features):
        """Test that concatenated vector has correct shape."""
        vec = sample_features.to_vector()

        # 21 rhythm + 83 timbre + 33 harmony = 137
        assert vec.shape == (137,)

    def test_to_vector_concatenation_order(self, sample_features):
        """Test that vector concatenates in rhythm -> timbre -> harmony order."""
        vec = sample_features.to_vector()

        r_vec = sample_features.rhythm.to_vector()
        t_vec = sample_features.timbre.to_vector()
        h_vec = sample_features.harmony.to_vector()

        expected = np.concatenate([r_vec, t_vec, h_vec])
        np.testing.assert_array_equal(vec, expected)

    def test_to_dict_structure(self, sample_features):
        """Test dictionary structure."""
        d = sample_features.to_dict()

        assert "rhythm" in d
        assert "timbre" in d
        assert "harmony" in d
        assert "bpm_info" in d
        assert d["n_beats"] == 200
        assert d["extraction_method"] == "beat3tower"

    def test_from_dict_roundtrip(self, sample_features):
        """Test dictionary serialization roundtrip."""
        d = sample_features.to_dict()
        restored = Beat3TowerFeatures.from_dict(d)

        assert restored.n_beats == sample_features.n_beats
        assert restored.bpm_info.primary_bpm == sample_features.bpm_info.primary_bpm
        assert restored.rhythm.bpm == sample_features.rhythm.bpm

    def test_n_features(self):
        """Test total feature count."""
        assert Beat3TowerFeatures.n_features() == 137

    def test_tower_dims(self):
        """Test tower dimension dictionary."""
        dims = Beat3TowerFeatures.tower_dims()

        assert dims["rhythm"] == 21
        assert dims["timbre"] == 83
        assert dims["harmony"] == 33


class TestInsufficientBeatsError:
    """Tests for InsufficientBeatsError exception."""

    def test_raise_with_message(self):
        """Test that exception can be raised with message."""
        with pytest.raises(InsufficientBeatsError, match="too few beats"):
            raise InsufficientBeatsError("too few beats detected")

    def test_is_exception(self):
        """Test that it's a proper Exception subclass."""
        assert issubclass(InsufficientBeatsError, Exception)
