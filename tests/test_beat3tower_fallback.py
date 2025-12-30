import numpy as np
from scipy.io import wavfile
import librosa
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:n_fft=.*too large for input signal.*:UserWarning"
)

from src.features.beat3tower_extractor import Beat3TowerExtractor, Beat3TowerConfig
from src.features.beat3tower_types import Beat3TowerFeatures


def _write_wav(path, y, sr):
    wavfile.write(path, sr, y.astype(np.float32))


def _assert_schema(result, expected_source):
    assert result is not None
    meta = result.get("metadata", {})
    assert meta.get("sonic_source") == expected_source
    assert result.get("source") == expected_source
    full = Beat3TowerFeatures.from_dict(result["full"])
    assert full.to_vector().shape[0] == Beat3TowerFeatures.n_features()


def test_beat3tower_beats_mode(monkeypatch, tmp_path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    beat_frames = np.array([0, 10, 20, 30, 40], dtype=int)

    def fake_beat_track(y, sr, hop_length):
        return 120.0, beat_frames

    def fake_load(*_args, **_kwargs):
        return y, sr

    monkeypatch.setattr(librosa.beat, "beat_track", fake_beat_track)
    monkeypatch.setattr(librosa, "load", fake_load)

    path = tmp_path / "beats.wav"
    _write_wav(path, y, sr)

    config = Beat3TowerConfig(sample_rate=sr, segment_duration=1.0)
    extractor = Beat3TowerExtractor(config)
    result = extractor.extract_from_file(str(path))

    _assert_schema(result, "beat3tower_beats")
    meta = result["metadata"]
    assert meta["beat_mode"] == "beats"
    assert meta["tempo_source"] == "beat_track"


def test_beat3tower_timegrid_mode(monkeypatch, tmp_path):
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 220.0 * t)

    def fake_beat_track(y, sr, hop_length):
        return 120.0, np.array([], dtype=int)

    def fake_tempo(*_args, **_kwargs):
        return np.array([80.0])

    def fake_load(*_args, **_kwargs):
        return y, sr

    monkeypatch.setattr(librosa.beat, "beat_track", fake_beat_track)
    monkeypatch.setattr(librosa.beat, "tempo", fake_tempo)
    monkeypatch.setattr(librosa, "load", fake_load)

    path = tmp_path / "timegrid.wav"
    _write_wav(path, y, sr)

    config = Beat3TowerConfig(sample_rate=sr, segment_duration=1.0)
    extractor = Beat3TowerExtractor(config)
    result = extractor.extract_from_file(str(path))

    _assert_schema(result, "beat3tower_timegrid")
    meta = result["metadata"]
    assert meta["beat_mode"] == "timegrid"
    assert meta["tempo_source"] in ("onset_tempo", "default")
    assert meta["timegrid_period_sec"] is not None


def test_beat3tower_stats_mode_for_silence(monkeypatch, tmp_path):
    sr = 22050
    duration = 1.0
    y = np.zeros(int(sr * duration), dtype=np.float32)

    def fake_beat_track(y, sr, hop_length):
        return 120.0, np.array([], dtype=int)

    monkeypatch.setattr(librosa.beat, "beat_track", fake_beat_track)

    def fake_load(*_args, **_kwargs):
        return y, sr

    monkeypatch.setattr(librosa, "load", fake_load)

    path = tmp_path / "silence.wav"
    _write_wav(path, y, sr)

    config = Beat3TowerConfig(sample_rate=sr, segment_duration=0.05)
    extractor = Beat3TowerExtractor(config)
    result = extractor.extract_from_file(str(path))

    _assert_schema(result, "beat3tower_stats")
    meta = result["metadata"]
    assert meta["beat_mode"] == "stats"
    assert meta["silence_flag"] is True
