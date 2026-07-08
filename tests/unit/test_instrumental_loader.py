import numpy as np
from src.playlist.instrumental_loader import load_voice_prob


def _write_sidecar(tmp_path, ids, probs):
    p = tmp_path / "instrumental_sidecar.npz"
    np.savez_compressed(p, track_ids=np.array(ids, dtype=object),
                        voice_prob=np.asarray(probs, dtype=np.float32))
    return str(p)


def test_load_voice_prob_aligns_to_requested_order(tmp_path):
    side = _write_sidecar(tmp_path, ["a", "b", "c"], [0.9, 0.1, 0.5])
    out = load_voice_prob(["c", "a", "zzz"], sidecar_path=side)
    assert abs(out[0] - 0.5) < 1e-6   # c
    assert abs(out[1] - 0.9) < 1e-6   # a
    assert np.isnan(out[2])           # unknown track -> NaN


def test_load_voice_prob_missing_sidecar_returns_all_nan(tmp_path, caplog):
    out = load_voice_prob(["a", "b"], sidecar_path=str(tmp_path / "nope.npz"))
    assert out.shape == (2,)
    assert np.isnan(out).all()
    assert any("instrumental" in r.message.lower() for r in caplog.records)
