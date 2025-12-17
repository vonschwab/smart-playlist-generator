from src.playlist.pipeline import generate_playlist_ds


def test_pipeline_deterministic(synthetic_artifact):
    path, bundle = synthetic_artifact
    overrides = {
        "candidate": {
            "similarity_floor": -1.0,
            "max_pool_size": 2000,
            "candidates_per_artist": 6,
            "target_artists": 12,
        },
        "construct": {
            "transition_floor": -1.0,
            "hard_floor": False,
            "min_gap": 2,
        },
    }
    result1 = generate_playlist_ds(
        artifact_path=path,
        seed_track_id=str(bundle.track_ids[0]),
        num_tracks=15,
        mode="dynamic",
        random_seed=123,
        overrides=overrides,
    )
    result2 = generate_playlist_ds(
        artifact_path=path,
        seed_track_id=str(bundle.track_ids[0]),
        num_tracks=15,
        mode="dynamic",
        random_seed=123,
        overrides=overrides,
    )
    assert result1.track_ids == result2.track_ids

