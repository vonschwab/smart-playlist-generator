"""SP-B loader contract: per-variant sonic keys only (no plain X_sonic)."""
import numpy as np
import pytest

from src.features.artifacts import load_artifact_bundle


def _write_npz(path, *, with_plain=False, with_muq=True):
    n, d = 4, 8
    arrs = {
        "track_ids": np.array([f"t{i}" for i in range(n)], dtype=object),
        "artist_keys": np.array([f"a{i}" for i in range(n)], dtype=object),
        "X_genre_raw": np.zeros((n, 3), dtype=np.float32),
        "X_genre_smoothed": np.zeros((n, 3), dtype=np.float32),
        "genre_vocab": np.array(["g1", "g2", "g3"], dtype=object),
        "X_sonic_variant": np.array("muq"),
    }
    if with_plain:
        arrs["X_sonic"] = np.random.rand(n, d).astype(np.float32)
    if with_muq:
        arrs["X_sonic_muq"] = np.random.rand(n, d).astype(np.float32)
    np.savez(path, **arrs)
    return path


def test_muq_only_artifact_loads_under_override(tmp_path):
    p = _write_npz(tmp_path / "art.npz", with_plain=False, with_muq=True)
    bundle = load_artifact_bundle(p, sonic_variant_override="muq")
    assert bundle.X_sonic.shape == (4, 8)
    assert bundle.sonic_variant == "muq"


def test_missing_variant_key_still_raises(tmp_path):
    p = _write_npz(tmp_path / "art2.npz", with_plain=True, with_muq=False)
    with pytest.raises(ValueError, match="X_sonic_muq"):
        load_artifact_bundle(p, sonic_variant_override="muq")


def test_legacy_artifact_without_override_still_requires_plain(tmp_path):
    p = _write_npz(tmp_path / "art3.npz", with_plain=False, with_muq=True)
    # no override: legacy contract, plain X_sonic required
    with pytest.raises(ValueError, match="X_sonic"):
        load_artifact_bundle(p, sonic_variant_override=None)
