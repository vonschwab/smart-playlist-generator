import numpy as np

from src.playlist.pipeline import generate_playlist_ds
from src.similarity.sonic_variant import compute_sonic_variant_norm


def test_variant_changes_similarity_ranking():
    # Construct dominant-dimension artifact: dim0 >> others
    base = np.array(
        [
            [10.0, 0.1],
            [9.9, 0.0],
            [10.0, -5.0],
            [1.0, 6.0],  # large in dim1 to surface after z
        ],
        dtype=float,
    )
    norm_raw, _ = compute_sonic_variant_norm(base, "raw")
    norm_z, _ = compute_sonic_variant_norm(base, "z")

    seed = 0
    others = [1, 2, 3]
    raw_sims = {i: float(norm_raw[seed] @ norm_raw[i]) for i in others}
    z_sims = {i: float(norm_z[seed] @ norm_z[i]) for i in others}

    raw_spread = max(raw_sims.values()) - min(raw_sims.values())
    z_spread = max(z_sims.values()) - min(z_sims.values())
    # Variant should change similarity distribution even if ranking ties
    assert z_spread > raw_spread
    assert any(abs(raw_sims[i] - z_sims[i]) > 1e-3 for i in others)


def test_variant_changes_playlist_order(tmp_path):
    # Build tiny artifact where dim0 dominates raw but dim1 separates after z
    track_ids = np.array([f"t{i}" for i in range(6)])
    artist_keys = np.array([f"a{i%3}" for i in range(6)])
    X_sonic = np.array(
        [
            [10.0, 0.0],
            [9.5, 0.1],
            [9.0, 0.2],
            [1.0, 5.0],
            [0.5, 4.5],
            [0.0, 4.0],
        ]
    )
    genre_vocab = np.array(["g"])
    X_genre = np.zeros((6, 1), dtype=float)
    art = tmp_path / "art.npz"
    np.savez(
        art,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=np.array([f"A{i}" for i in range(6)]),
        track_titles=np.array([f"S{i}" for i in range(6)]),
        X_sonic=X_sonic,
        X_sonic_start=X_sonic,
        X_sonic_end=X_sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=genre_vocab,
    )
    overrides = {
        "candidate": {
            "similarity_floor": -1.0,
            "max_pool_size": 6,
            "target_artists": 6,
            "candidates_per_artist": 6,
            "seed_artist_bonus": 0,
        },
        "construct": {
            "transition_floor": -1.0,
            "hard_floor": False,
            "min_gap": 0,
            "max_artist_fraction_final": 1.0,
        },
    }
    raw_res = generate_playlist_ds(
        artifact_path=art,
        seed_track_id="t0",
        num_tracks=5,
        mode="dynamic",
        random_seed=0,
        overrides=overrides,
        sonic_variant="raw",
    )
    z_res = generate_playlist_ds(
        artifact_path=art,
        seed_track_id="t0",
        num_tracks=5,
        mode="dynamic",
        random_seed=0,
        overrides=overrides,
        sonic_variant="z",
    )
    raw_order = raw_res.track_ids
    z_order = z_res.track_ids
    jaccard = len(set(raw_order) & set(z_order)) / max(1, len(set(raw_order) | set(z_order)))
    assert raw_order != z_order
    assert jaccard < 1.0
