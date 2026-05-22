from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.artist_identity_resolver import ArtistIdentityConfig
from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides
from src.playlist.repair.edge_repair import repair_playlist_edges
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge


def _repair_bundle(
    *,
    titles: list[str] | None = None,
    artists: list[str] | None = None,
    X: list[list[float]] | None = None,
) -> ArtifactBundle:
    X_arr = np.array(
        X
        if X is not None
        else [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    n = int(X_arr.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array(artists or [f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array(artists or [f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array(titles or [f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_arr,
        X_sonic_start=X_arr,
        X_sonic_mid=X_arr,
        X_sonic_end=X_arr,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _context(bundle: ArtifactBundle):
    return build_transition_metric_context(
        X_sonic=bundle.X_sonic,
        X_start=bundle.X_sonic_start,
        X_mid=bundle.X_sonic_mid,
        X_end=bundle.X_sonic_end,
        X_genre=bundle.X_genre_smoothed,
        center_transitions=False,
        sonic_variant="raw",
    )


def test_edge_repair_noops_when_edges_are_clean():
    bundle = _repair_bundle()
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 3, 2],
        candidate_indices=[1, 4],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 2},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
    )

    assert result.indices == [0, 3, 2]
    assert result.swap_log == []


def test_edge_repair_replaces_destination_interior_track():
    bundle = _repair_bundle()
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 2],
        candidate_indices=[3],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 2},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
    )

    assert result.indices == [0, 3, 2]
    assert result.swap_log[0]["position"] == 1
    assert result.swap_log[0]["old_idx"] == 1
    assert result.swap_log[0]["new_idx"] == 3
    assert min(
        score_transition_edge(ctx, result.indices[i - 1], result.indices[i])["T"]
        for i in range(1, len(result.indices))
    ) >= 0.2


def test_edge_repair_for_bad_edge_into_pier_replaces_source_not_pier():
    bundle = _repair_bundle()
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 2],
        candidate_indices=[3],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 2},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
        repair_edge_position=2,
    )

    assert result.indices == [0, 3, 2]
    assert result.swap_log[0]["position"] == 1
    assert result.swap_log[0]["reason"] == "source_before_pier"


def test_edge_repair_refuses_seed_duplicate_artist_allowed_title_and_pier_violations():
    bundle = _repair_bundle(
        titles=[
            "Seed A",
            "Broken",
            "Seed B",
            "Seed A",
            "Candidate (Live)",
        ],
        artists=["Pier A", "Broken Artist", "Pier B", "Pier A", "Other Artist"],
    )
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 2],
        candidate_indices=[0, 2, 3, 4],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 2},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
        allowed_indices={0, 1, 2, 3, 4},
        disallowed_artist_keys={"pier a"},
    )

    assert result.indices == [0, 1, 2]
    refusal_reasons = {entry["reason"] for entry in result.swap_log}
    assert {
        "candidate_is_seed",
        "candidate_is_pier",
        "duplicate_track_key",
        "disallowed_artist",
        "title_artifact",
    } <= refusal_reasons


def test_edge_repair_refuses_candidate_that_exceeds_non_seed_artist_cap():
    bundle = _repair_bundle(
        X=[
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        artists=["Pier A", "Broken Artist", "Pier B", "Duplicate Artist", "Duplicate Artist"],
    )
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 3, 2],
        candidate_indices=[4],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 3},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
        max_non_seed_tracks_per_artist=1,
    )

    assert result.indices == [0, 1, 3, 2]
    assert "max_non_seed_artist_cap" in {entry["reason"] for entry in result.swap_log}


def test_edge_repair_refuses_collaboration_secondary_artist_that_exceeds_non_seed_artist_cap():
    bundle = _repair_bundle(
        X=[
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        artists=["Pier A", "Broken Artist", "Pier B", "Guest Artist", "Main Artist & Guest Artist"],
    )
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 3, 2],
        candidate_indices=[4],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 3},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
        max_non_seed_tracks_per_artist=1,
        artist_identity_cfg=ArtistIdentityConfig(enabled=True),
    )

    assert result.indices == [0, 1, 3, 2]
    assert "max_non_seed_artist_cap" in {entry["reason"] for entry in result.swap_log}


def test_edge_repair_nested_config_overrides_are_parsed():
    pb_cfg, _tuning, _sources, _weights = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides={},
        pb_overrides={
            "edge_repair": {
                "enabled": True,
                "centered_cos_floor": -0.6,
                "margin": 0.08,
                "variety_guard": {
                    "enabled": True,
                    "threshold": 0.9,
                },
            }
        },
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
        resolved_variant="raw",
    )

    assert pb_cfg.edge_repair_enabled is True
    assert pb_cfg.edge_repair_centered_cos_floor == -0.6
    assert pb_cfg.edge_repair_margin == 0.08
    assert pb_cfg.edge_repair_variety_guard_enabled is True
    assert pb_cfg.edge_repair_variety_guard_threshold == 0.9
