import numpy as np
from src.playlist.energy_loader import load_energy_matrix


def _make_sidecar(tmp_path):
    p = tmp_path / "energy_sidecar.npz"
    np.savez(
        p,
        track_ids=np.array(["a", "b", "c", "d"], dtype=object),
        arousal_p50=np.array([2.0, 4.0, 6.0, np.nan], dtype=np.float32),
        danceability=np.array([0.1, 0.5, 0.9, 0.5], dtype=np.float32),
    )
    return str(p)


def test_zscored_shape_and_values(tmp_path):
    side = _make_sidecar(tmp_path)
    m = load_energy_matrix(["a", "b", "c"], sidecar_path=side, features=("arousal_p50",))
    assert m.shape == (3, 1)
    # arousal [2,4,6] over the library (a,b,c finite; d is NaN, ignored) -> mean 4, so b -> 0
    assert abs(float(m[1, 0])) < 1e-6
    assert float(m[0, 0]) < 0 < float(m[2, 0])


def test_missing_track_and_nan_feature_are_nan_rows(tmp_path):
    side = _make_sidecar(tmp_path)
    m = load_energy_matrix(["a", "zzz", "d"], sidecar_path=side, features=("arousal_p50", "danceability"))
    assert m.shape == (3, 2)
    assert np.all(np.isfinite(m[0]))          # a present
    assert np.all(np.isnan(m[1]))             # zzz absent -> NaN row
    assert np.isnan(m[2, 0]) and np.isfinite(m[2, 1])  # d: arousal NaN, dance finite
