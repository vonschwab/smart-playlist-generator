import numpy as np
import librosa
import pytest

from src.features.beat3tower_extractor import Beat3TowerExtractor, Beat3TowerConfig
from src.features.beat3tower_types import TimbreTowerFeatures, HarmonyTowerFeatures


def _synth(sr=22050, duration=4.0, bpm=120.0):
    """Deterministic tone + percussive clicks at a known tempo (real librosa, no mocks)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    period = 60.0 / bpm
    for k in range(int(duration / period)):
        i = int(k * period * sr)
        y[i:i + 200] += 0.8
    return y.astype(np.float32), sr


def test_slim_pace_fields_match_direct_librosa():
    y, sr = _synth()
    ext = Beat3TowerExtractor(Beat3TowerConfig(sample_rate=sr))
    tempo, beat_frames, beat_times = ext._detect_beats(y)
    feats = ext._extract_with_beats(y, beat_frames, beat_times, tempo)

    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", hop_length=ext.hop_length)
    expected_onset_rate = len(onset_times) / (len(y) / sr)
    assert feats.rhythm.onset_rate == pytest.approx(expected_onset_rate)

    expected = ext._compute_bpm_info(tempo, beat_times)
    assert feats.bpm_info.primary_bpm == pytest.approx(expected.primary_bpm)
    assert feats.bpm_info.tempo_stability == pytest.approx(expected.tempo_stability)
    assert feats.bpm_info.half_tempo_likely == expected.half_tempo_likely
    assert feats.bpm_info.double_tempo_likely == expected.double_tempo_likely


def test_slim_towers_are_empty_and_marked():
    y, sr = _synth()
    ext = Beat3TowerExtractor(Beat3TowerConfig(sample_rate=sr))
    tempo, beat_frames, beat_times = ext._detect_beats(y)
    feats = ext._extract_with_beats(y, beat_frames, beat_times, tempo)

    # timbre/harmony are the empty defaults -> zero vectors (dimensions preserved)
    assert np.allclose(feats.timbre.to_vector(), TimbreTowerFeatures().to_vector())
    assert np.allclose(feats.harmony.to_vector(), HarmonyTowerFeatures().to_vector())
    assert feats.extraction_method == "beat3tower"

    # downstream contract: the JSON bpm_loader reads carries the 5 pace fields
    d = feats.to_dict()
    assert "onset_rate" in d["rhythm"]
    for k in ("primary_bpm", "half_tempo_likely", "double_tempo_likely", "tempo_stability"):
        assert k in d["bpm_info"]


def test_detect_beats_returns_scalar_bpm_without_deprecation():
    """librosa.beat.beat_track returns tempo as an ndim>0 array; float(tempo) on it is a
    numpy DeprecationWarning that becomes a hard error in future numpy -- and this is the
    BPM source for the pace axis. _detect_beats must return a plain float, warning-free."""
    import warnings

    y, sr = _synth()
    ext = Beat3TowerExtractor(Beat3TowerConfig(sample_rate=sr))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tempo, _beat_frames, _beat_times = ext._detect_beats(y)
    assert isinstance(tempo, float)
    leaked = [w for w in caught if "ndim > 0 to a scalar" in str(w.message)]
    assert not leaked, f"_detect_beats leaked a numpy ndim>0 deprecation: {[str(w.message) for w in leaked]}"
