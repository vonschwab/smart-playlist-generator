import numpy as np

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


def test_artifact_exposes_tower_dims(tmp_path):
    """The per-tower blend split (rhythm/timbre/harmony) is loaded onto the bundle.

    This is the authoritative source for slicing X_sonic into perceptual axes —
    consumers must not have to infer it from the total blend width.
    """
    N = 3
    d = {
        "track_ids": np.array(["a", "b", "c"], dtype=object),
        "artist_keys": np.array(["a", "b", "c"], dtype=object),
        "X_sonic": np.zeros((N, 162), np.float32),
        "X_genre_raw": np.ones((N, 2), np.float32),
        "X_genre_smoothed": np.ones((N, 2), np.float32),
        "genre_vocab": np.array(["x", "y"], dtype=object),
        "tower_dims": np.array([9, 57, 96], dtype=np.int64),
    }
    p = tmp_path / "td.npz"
    np.savez(p, **d)

    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(p)

    assert b.tower_dims == (9, 57, 96)


def test_artifact_without_tower_dims_is_none(tmp_path):
    """Legacy artifacts lacking the tower_dims key load with tower_dims=None."""
    N = 3
    d = {
        "track_ids": np.array(["a", "b", "c"], dtype=object),
        "artist_keys": np.array(["a", "b", "c"], dtype=object),
        "X_sonic": np.zeros((N, 86), np.float32),
        "X_genre_raw": np.ones((N, 2), np.float32),
        "X_genre_smoothed": np.ones((N, 2), np.float32),
        "genre_vocab": np.array(["x", "y"], dtype=object),
    }
    p = tmp_path / "notd.npz"
    np.savez(p, **d)

    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(p)

    assert b.tower_dims is None


def test_artifact_loads_layered_genre_shadow_keys(tmp_path):
    N = 3
    d = {
        "track_ids": np.array(["a", "b", "c"], dtype=object),
        "artist_keys": np.array(["a", "b", "c"], dtype=object),
        "X_sonic": np.zeros((N, 86), np.float32),
        "X_genre_raw": np.ones((N, 2), np.float32),
        "X_genre_smoothed": np.ones((N, 2), np.float32),
        "genre_vocab": np.array(["x", "y"], dtype=object),
        "X_genre_leaf_idf": np.arange(N * 2, dtype=np.float32).reshape(N, 2),
        "X_genre_family": np.arange(N, dtype=np.float32).reshape(N, 1),
        "X_genre_bridge": np.ones((N, 2), np.float32),
        "X_facet": np.ones((N, 1), np.float32),
        "genre_leaf_vocab": np.array(["jangle pop", "synth-pop"], dtype=object),
        "genre_family_vocab": np.array(["pop"], dtype=object),
        "genre_bridge_vocab": np.array(["jangle pop", "synth-pop"], dtype=object),
        "facet_vocab": np.array(["synth-heavy"], dtype=object),
        "genre_graph_taxonomy_version": np.array("layered-genre-graph-v1", dtype=object),
        "genre_graph_sidecar_fingerprint": np.array("a" * 64, dtype=object),
    }
    p = tmp_path / "layered.npz"
    np.savez(p, **d)

    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(p)

    assert b.X_genre_leaf_idf is not None
    assert b.X_genre_leaf_idf.shape == (N, 2)
    assert b.X_genre_family is not None
    assert b.X_genre_bridge is not None
    assert b.X_facet is not None
    assert b.genre_leaf_vocab is not None
    assert b.genre_leaf_vocab.tolist() == ["jangle pop", "synth-pop"]
    assert b.genre_graph_sidecar_fingerprint is not None
    assert b.genre_graph_sidecar_fingerprint.item() == "a" * 64
