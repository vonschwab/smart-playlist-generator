from __future__ import annotations

import numpy as np

from src.playlist.pipeline import generate_playlist_ds
from src.playlist.pier_bridge_builder import PierBridgeConfig


def _write_layered_artifact(path):
    n = 8
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    artist_keys = np.array([f"artist-{i}" for i in range(n)], dtype=object)
    track_artists = np.array([f"Artist {i}" for i in range(n)], dtype=object)
    track_titles = np.array([f"Track {i}" for i in range(n)], dtype=object)

    # Keep all tracks sonically reachable; this smoke is about graph plumbing.
    X_sonic = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.80, 0.20, 0.00, 0.00],
            [0.75, 0.25, 0.00, 0.00],
            [0.70, 0.30, 0.00, 0.00],
            [0.65, 0.35, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    # Legacy vectors are intentionally present but non-informative. In layered
    # mode they must not be used as active genre gates.
    X_genre_raw = np.array(
        [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(n)],
        dtype=np.float32,
    )
    X_genre_smoothed = X_genre_raw.copy()
    genre_vocab = np.array(["legacy-a", "legacy-b"], dtype=object)

    # Leaf dimensions: [slowcore, dream pop].
    X_leaf = np.array(
        [
            [1.0, 0.0],  # seed
            [1.0, 0.0],
            [0.8, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.7, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    X_family = np.ones((n, 1), dtype=np.float32)
    # Bridge affordance is leaf-aligned; seed can bridge toward dream pop, and
    # dream-pop candidates can bridge back toward slowcore.
    X_bridge = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    X_facet = np.ones((n, 1), dtype=np.float32)

    np.savez(
        path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        X_genre_leaf_idf=X_leaf,
        X_genre_family=X_family,
        X_genre_bridge=X_bridge,
        X_facet=X_facet,
        genre_leaf_vocab=np.array(["slowcore", "dream pop"], dtype=object),
        genre_family_vocab=np.array(["rock"], dtype=object),
        genre_bridge_vocab=np.array(["slowcore", "dream pop"], dtype=object),
        facet_vocab=np.array(["reverb-heavy"], dtype=object),
    )


def test_ds_pipeline_can_generate_with_layered_graph_source(tmp_path):
    artifact = tmp_path / "layered-artifact.npz"
    _write_layered_artifact(artifact)

    pb_cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        initial_neighbors_m=10,
        initial_bridge_helpers=5,
        max_neighbors_m=10,
        max_bridge_helpers=5,
        initial_beam_width=4,
        max_beam_width=4,
        max_expansion_attempts=1,
        segment_pool_strategy="segment_scored",
        segment_pool_max=8,
        max_segment_pool_max=8,
        progress_enabled=False,
        edge_repair_enabled=False,
        genre_steering_enabled=False,
        weight_genre=0.0,
        dj_bridging_enabled=False,
        layered_transition_scoring_enabled=True,
        layered_transition_weight=0.15,
        layered_transition_mode="dynamic",
    )

    result = generate_playlist_ds(
        artifact_path=str(artifact),
        seed_track_id="t0",
        num_tracks=4,
        mode="dynamic",
        pace_mode="off",
        random_seed=0,
        overrides={
            "genre_graph": {"source": "layered", "transition_weight": 0.15},
            "candidate": {
                "similarity_floor": -1.0,
                "min_sonic_similarity": None,
                "max_pool_size": 8,
                "target_artists": 4,
                "candidates_per_artist": 2,
                "seed_artist_bonus": 0,
                "title_hard_exclude_flags": [],
            },
            "pier_bridge": {
                "audit_run": {"enabled": False},
                "infeasible_handling": {"enabled": False},
            },
        },
        pier_bridge_config=pb_cfg,
        dry_run=True,
    )

    admission = result.stats["candidate_pool"]["layered_genre_admission"]
    transition = result.stats["playlist"]["layered_transition_diagnostics"]

    assert len(result.track_ids) == 4
    assert admission["applied"] is True
    assert admission["source"] == "layered"
    assert admission["legacy_flat_genre_gate_applied"] is False
    assert admission["admitted_count"] >= 4
    assert result.params_effective["candidate_pool"]["genre_graph_source"] == "layered"
    assert result.stats["playlist"]["success"] is True
    assert transition["enabled"] is True
    assert transition["edge_count"] == 3
