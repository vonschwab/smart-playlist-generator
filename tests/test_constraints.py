import numpy as np

from src.playlist.pipeline import generate_playlist_ds


def test_constraints_enforced(synthetic_artifact):
    path, bundle = synthetic_artifact
    overrides = {
        "candidate": {"similarity_floor": -1.0, "max_pool_size": 1500, "candidates_per_artist": 5, "target_artists": 10},
        "construct": {"min_gap": 3, "transition_floor": 0.0},
    }
    result = generate_playlist_ds(
        artifact_path=path,
        seed_track_id=str(bundle.track_ids[1]),
        num_tracks=12,
        mode="dynamic",
        random_seed=99,
        overrides=overrides,
    )

    artists = [str(bundle.artist_keys[i]) for i in result.track_indices]
    # No adjacency
    for i in range(1, len(artists)):
        assert artists[i] != artists[i - 1]
    # Min-gap
    min_gap = overrides["construct"]["min_gap"]
    for i in range(len(artists)):
        recent = artists[max(0, i - min_gap) : i]
        assert artists[i] not in recent

    playlist_stats = result.stats["playlist"]
    transition_floor = result.params_effective["playlist"]["transition_floor"]
    if not playlist_stats.get("hard_floor_relaxed"):
        assert playlist_stats["min_transition"] >= transition_floor - 1e-6
    else:
        assert playlist_stats["hard_floor_relaxed"] is True


def test_floor_bookkeeping_soft_mode(synthetic_artifact):
    path, bundle = synthetic_artifact
    overrides = {
        "candidate": {"similarity_floor": -1.0, "max_pool_size": 1500, "candidates_per_artist": 4, "target_artists": 10},
        "construct": {"transition_floor": 0.1, "hard_floor": False, "min_gap": 2},
    }
    result = generate_playlist_ds(
        artifact_path=path,
        seed_track_id=str(bundle.track_ids[2]),
        num_tracks=10,
        mode="discover",
        random_seed=7,
        overrides=overrides,
    )
    playlist_stats = result.stats["playlist"]
    assert "below_floor_count" in playlist_stats
    assert "gap_sum" in playlist_stats

