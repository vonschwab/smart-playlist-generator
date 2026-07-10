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


def test_load_voice_prob_corrupt_but_openable_npz_returns_all_nan(tmp_path, caplog):
    # A larger payload so there is enough compressed-data mid-file to corrupt
    # without touching the trailing zip central directory / EOCD record.
    ids = [f"t{i}" for i in range(200)]
    probs = list(np.linspace(0.0, 1.0, 200))
    side = _write_sidecar(tmp_path, ids, probs)

    raw = bytearray(open(side, "rb").read())
    assert len(raw) > 200, "sidecar too small to safely corrupt mid-file"
    # Flip ~20 bytes in the middle of the file (inside a compressed member's
    # data, well clear of the trailing central directory / EOCD).
    mid = len(raw) // 2
    for offset in range(mid, mid + 20):
        raw[offset] ^= 0xFF
    with open(side, "wb") as f:
        f.write(raw)

    # Sanity: np.load() itself must still succeed (lazy zip open) — the bug
    # this test guards against is the *array access* raising, not np.load.
    npzfile = np.load(side, allow_pickle=True)
    assert set(npzfile.files) == {"track_ids", "voice_prob"}

    with caplog.at_level("WARNING"):
        out = load_voice_prob(ids, sidecar_path=side)

    assert out.shape == (200,)
    assert np.isnan(out).all()
    assert any("instrumental" in r.message.lower() for r in caplog.records)


def test_load_voice_prob_missing_required_key_returns_all_nan(tmp_path, caplog):
    p = tmp_path / "instrumental_sidecar_missing_key.npz"
    # track_ids present, voice_prob absent.
    np.savez_compressed(p, track_ids=np.array(["a", "b"], dtype=object))

    with caplog.at_level("WARNING"):
        out = load_voice_prob(["a", "b"], sidecar_path=str(p))

    assert out.shape == (2,)
    assert np.isnan(out).all()
    assert any("instrumental" in r.message.lower() for r in caplog.records)
