import numpy as np


from src.features.artifacts import load_artifact_bundle


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
