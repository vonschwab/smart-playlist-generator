# tests/unit/test_pace_audition_serve.py
from scripts.pace_audition_serve import blinded_manifest, upsert_capture_entry


def test_blinded_manifest_strips_all_server_side_keys():
    manifest = {
        "type": "pace_edges",
        "provenance": {"arms": {"narrow": {}}},
        "playlists": [{"seed": "s", "arm": "narrow"}],
        "edges": [{"edge_id": "e0001", "a": "ta", "b": "tb"}],
        "edge_data": {"e0001": {"arm": "narrow"}},
        "file_paths": {"ta": "x"},
    }
    served = blinded_manifest(manifest)
    assert set(served.keys()) == {"type", "edges"}
    assert "narrow" not in json_dumps(served)


def json_dumps(o):
    import json
    return json.dumps(o)


def test_upsert_capture_entry_dedupes_by_edge_id():
    entries = [{"edge_id": "e1", "continuity": 3, "smoothness": 3}]
    upsert_capture_entry(entries, {"edge_id": "e1", "continuity": 5, "smoothness": 4})
    upsert_capture_entry(entries, {"edge_id": "e2", "continuity": 2, "smoothness": 2})
    assert len(entries) == 2
    assert entries[0]["continuity"] == 5  # overwritten in place


def test_read_window_wav_transcodes_24bit_flac_to_16bit_wav(tmp_path):
    import io
    import numpy as np
    import soundfile as sf
    from scripts.pace_audition_serve import read_window_wav

    sr = 44100
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(5 * sr) / sr)).astype("float32")
    src = tmp_path / "src.flac"
    sf.write(src, x, sr, format="FLAC", subtype="PCM_24")  # 24-bit source

    wav = read_window_wav(str(src), "head", window_sec=1.0)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with sf.SoundFile(io.BytesIO(wav)) as f:
        assert f.subtype == "PCM_16"      # transcoded to browser-playable 16-bit
        assert f.samplerate == sr
        assert abs(len(f) - sr) <= 2      # ~1 second window


def test_read_window_wav_tail_returns_end_of_file(tmp_path):
    import io
    import numpy as np
    import soundfile as sf
    from scripts.pace_audition_serve import read_window_wav

    sr = 22050
    x = np.linspace(0.0, 1.0, 3 * sr).astype("float32")  # ramp: tail != head
    src = tmp_path / "ramp.flac"
    sf.write(src, x, sr, format="FLAC", subtype="PCM_16")

    tail, _ = sf.read(io.BytesIO(read_window_wav(str(src), "tail", window_sec=0.5)), dtype="float32")
    head, _ = sf.read(io.BytesIO(read_window_wav(str(src), "head", window_sec=0.5)), dtype="float32")
    assert tail.max() > 0.8   # tail is near the end of the ramp (~1.0)
    assert head.max() < 0.3   # head is near the start (~0.0)
