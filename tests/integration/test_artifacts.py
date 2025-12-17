from src.features.artifacts import get_sonic_matrix, load_artifact_bundle


def test_artifact_loader_round_trip(tmp_path):
    path = tmp_path / "artifact.npz"
    import numpy as np

    rng = np.random.default_rng(0)
    track_ids = np.array([f"id{i}" for i in range(5)])
    artist_keys = np.array([f"a{i}" for i in range(5)])
    X_sonic = rng.normal(size=(5, 4))
    X_genre = rng.random(size=(5, 3))
    np.savez(
        path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=np.array([f"Artist {i}" for i in range(5)]),
        track_titles=np.array([f"Song {i}" for i in range(5)]),
        X_sonic=X_sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=np.array(["g1", "g2", "g3"]),
    )

    bundle = load_artifact_bundle(path)
    for idx, tid in enumerate(track_ids):
        assert bundle.track_id_to_index[str(tid)] == idx
    assert get_sonic_matrix(bundle, "full").shape == X_sonic.shape

