import numpy as np

from src.features.artifacts import load_artifact_bundle


def test_tower_weighted_artifact_loads_pre_scaled(tmp_path):
    N = 4
    tw_full = np.random.default_rng(0).normal(size=(N, 86)).astype(np.float32)
    tw_start = np.random.default_rng(1).normal(size=(N, 86)).astype(np.float32)
    d = {
        "track_ids": np.array(["a", "b", "c", "d"], dtype=object),
        "artist_keys": np.array(["a", "b", "c", "d"], dtype=object),
        "track_artists": np.array(["A", "B", "C", "D"], dtype=object),
        "X_sonic": np.zeros((N, 86), np.float32),       # raw key (unselected)
        "X_sonic_tower_weighted": tw_full,              # variant key (selected)
        "X_sonic_start": tw_start,
        "X_sonic_mid": tw_start,
        "X_sonic_end": tw_start,
        "X_sonic_variant": np.array("tower_weighted"),
        "X_sonic_pre_scaled": np.array(True),
        "X_genre_raw": np.ones((N, 3), np.float32),
        "X_genre_smoothed": np.ones((N, 3), np.float32),
        "genre_vocab": np.array(["x", "y", "z"], dtype=object),
    }
    p = tmp_path / "twart.npz"
    np.savez(p, **d)

    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(p)

    assert b.sonic_variant == "tower_weighted"
    assert b.sonic_pre_scaled is True
    assert np.array_equal(b.X_sonic, tw_full)        # selected the variant key
    assert np.array_equal(b.X_sonic_start, tw_start)  # segment key used directly
