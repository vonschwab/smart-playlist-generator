from types import SimpleNamespace

import numpy as np


def test_replacement_cache_uses_ds_candidate_pool_track_ids(monkeypatch):
    from src.playlist_gui import worker
    import src.features.artifacts as artifacts
    import src.playlist.bpm_loader as bpm_loader
    import src.playlist.transition_metrics as transition_metrics

    track_ids = np.array([f"t{i}" for i in range(6)], dtype=object)
    bundle = SimpleNamespace(
        track_ids=track_ids,
        artist_keys=np.array([f"a{i}" for i in range(6)], dtype=object),
        X_sonic=np.eye(6, dtype=float),
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_smoothed=np.eye(6, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(6)], dtype=object),
        track_id_to_index={str(tid): idx for idx, tid in enumerate(track_ids)},
    )

    monkeypatch.setattr(artifacts, "load_artifact_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        bpm_loader,
        "load_bpm_arrays",
        lambda _track_ids, db_path: {
            "perceptual_bpm": np.full(6, np.nan),
            "tempo_stability": np.full(6, np.nan),
        },
    )
    monkeypatch.setattr(
        transition_metrics,
        "build_transition_metric_context",
        lambda **_kwargs: None,
    )

    generator = SimpleNamespace(
        _last_ds_report={
            "artifact_path": "fake.npz",
            "playlist_stats": {
                "candidate_pool": {
                    "seed_sonic_sim_track_ids": {
                        "t1": 0.9,
                        "t3": 0.8,
                        "t5": 0.7,
                    }
                },
                "playlist": {"transition_floor": 0.2},
            },
        }
    )
    playlist_result = {
        "name": "test",
        "tracks": [
            {"rating_key": "t1"},
            {"rating_key": "t2"},
            {"rating_key": "t3"},
        ],
    }

    worker._populate_last_generation_cache(
        generator=generator,
        playlist_result=playlist_result,
        config={"playlists": {"ds_pipeline": {"tower_pca_dims": [2, 2, 2]}}},
        db_path="metadata.db",
    )

    assert set(worker._LAST_GENERATION_CACHE.candidate_pool_indices.tolist()) == {1, 3, 5}
