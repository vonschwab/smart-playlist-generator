import numpy as np
from src.playlist.popularity_loader import load_popularity_vector


def test_load_popularity_aligns_and_nan_for_gaps(tmp_path):
    side = tmp_path / "popularity_sidecar.npz"
    np.savez(side, track_ids=np.array(["a", "b", "c"], dtype=object),
             popularity=np.array([1.0, np.nan, 0.5], dtype=np.float32))
    out = load_popularity_vector(["c", "a", "zzz"], sidecar_path=str(side))
    assert out.shape == (3,)
    assert out[0] == 0.5 and out[1] == 1.0
    assert np.isnan(out[2])            # not in sidecar -> NaN


def test_load_popularity_missing_file_is_all_nan(tmp_path):
    out = load_popularity_vector(["a", "b"], sidecar_path=str(tmp_path / "nope.npz"))
    assert out.shape == (2,) and np.all(np.isnan(out))
