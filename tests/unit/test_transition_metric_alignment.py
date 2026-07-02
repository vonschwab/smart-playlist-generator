from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist import reporter
from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist
from src.playlist.transition_metrics import (
    build_transition_metric_context,
    is_broken_transition,
    score_transition_edge,
)


def _bundle(
    *,
    X_sonic: np.ndarray,
    X_start: np.ndarray | None = None,
    X_mid: np.ndarray | None = None,
    X_end: np.ndarray | None = None,
) -> ArtifactBundle:
    n = int(X_sonic.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_sonic,
        X_sonic_start=X_start,
        X_sonic_mid=X_mid,
        X_sonic_end=X_end,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _matrix_sonic(rows: list[tuple[float, float, float]]) -> np.ndarray:
    """Build sparse muq-shaped (512-dim) rows with three feature-axis sentinels.

    Width intentionally != 137: the beam builder (pier_bridge_builder.py)
    still calls the legacy `apply_transition_weights`, which special-cases
    exactly-137-dim beat3tower matrices and re-applies tower reweighting that
    `build_transition_metric_context` no longer does (SP-B Task 3). A 137-dim
    fixture would re-introduce that legacy path and break the beam-vs-reporter
    alignment this test guards, pending its removal from pier_bridge_builder.py
    in a later SP-B task.
    """
    mat = np.zeros((len(rows), 512), dtype=float)
    for i, (rhythm, timbre, harmony) in enumerate(rows):
        mat[i, 0] = rhythm
        mat[i, 21] = timbre
        mat[i, 104] = harmony
    return mat


def test_beam_trans_score_matches_reporter_t_for_centered_weighted_edge(monkeypatch):
    weights = (0.2, 0.5, 0.3)
    X_full = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.8, 0.5, 0.0),
        (0.0, 1.0, 0.4),
    ])
    X_start = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.7, 0.6, 0.0),
        (0.0, 1.0, 0.5),
    ])
    X_mid = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.6, 0.7, 0.0),
        (0.1, 1.0, 0.4),
    ])
    X_end = _matrix_sonic([
        (1.0, 0.1, 0.0),
        (0.6, 0.7, 0.1),
        (0.0, 0.9, 0.6),
    ])
    bundle = _bundle(X_sonic=X_full, X_start=X_start, X_mid=X_mid, X_end=X_end)
    monkeypatch.setattr(reporter, "load_artifact_bundle", lambda _path: bundle)

    ctx = build_transition_metric_context(
        X_sonic=X_full,
        X_start=X_start,
        X_mid=X_mid,
        X_end=X_end,
        X_genre=bundle.X_genre_smoothed,
        center_transitions=True,
    )
    cfg = PierBridgeConfig(
        center_transitions=True,
        transition_floor=0.0,
        transition_weights=weights,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    out = {}
    path, _hits, _scored, err = _beam_search_segment(
        0,
        2,
        1,
        [1],
        ctx.X_full,
        ctx.X_sonic_norm,
        ctx.X_start,
        ctx.X_mid,
        ctx.X_end,
        ctx.X_genre_norm,
        cfg,
        3,
        transition_metric_context=ctx,
        edge_components_out=out,
    )
    assert err is None
    assert path == [1]

    reporter_edges = reporter.compute_edge_scores_from_artifact(
        tracks=[{"rating_key": "t0"}, {"rating_key": "t1"}, {"rating_key": "t2"}],
        artifact_path="fake.npz",
        config_sonic_variant="raw",
        center_transitions=True,
        transition_weights=weights,
        sonic_variant="raw",
    )
    beam_edges = out["components"]
    assert len(beam_edges) == len(reporter_edges) == 2
    for beam_edge, reporter_edge in zip(beam_edges, reporter_edges):
        assert beam_edge["trans_score_in_beam"] == reporter_edge["T"]
        assert beam_edge["T_centered_cos"] == reporter_edge["T_centered_cos"]


def test_catastrophic_centered_cos_is_broken_even_when_rescaled_t_clears_floor():
    X = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    ctx = build_transition_metric_context(
        X_sonic=X,
        X_start=X,
        X_mid=X,
        X_end=X,
        center_transitions=True,
    )
    edge = score_transition_edge(ctx, 0, 1)

    assert edge["T_centered_cos"] < -0.5
    assert edge["T"] >= 0.0
    assert is_broken_transition(edge, transition_floor=0.0, centered_cos_floor=-0.5)


def test_builder_edge_scores_match_final_reporter_edges(monkeypatch):
    weights = (0.2, 0.5, 0.3)
    X_full = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.8, 0.5, 0.0),
        (0.0, 1.0, 0.4),
    ])
    X_start = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.7, 0.6, 0.0),
        (0.0, 1.0, 0.5),
    ])
    X_mid = _matrix_sonic([
        (1.0, 0.0, 0.0),
        (0.6, 0.7, 0.0),
        (0.1, 1.0, 0.4),
    ])
    X_end = _matrix_sonic([
        (1.0, 0.1, 0.0),
        (0.6, 0.7, 0.1),
        (0.0, 0.9, 0.6),
    ])
    bundle = _bundle(X_sonic=X_full, X_start=X_start, X_mid=X_mid, X_end=X_end)
    monkeypatch.setattr(reporter, "load_artifact_bundle", lambda _path: bundle)

    cfg = PierBridgeConfig(
        center_transitions=False,
        transition_floor=-1.0,
        bridge_floor=-1.0,
        transition_weights=weights,
        progress_enabled=False,
        collapse_segment_pool_by_artist=False,
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["t0", "t2"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[1],
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=bundle.X_genre_smoothed,
    )

    assert result.success
    assert result.track_ids == ["t0", "t1", "t2"]
    reporter_edges = reporter.compute_edge_scores_from_artifact(
        tracks=[{"rating_key": tid} for tid in result.track_ids],
        artifact_path="fake.npz",
        config_sonic_variant="raw",
        center_transitions=False,
        transition_weights=weights,
        sonic_variant="raw",
    )
    builder_edges = result.stats["edge_scores"]
    assert len(builder_edges) == len(reporter_edges) == 2
    for builder_edge, reporter_edge in zip(builder_edges, reporter_edges):
        assert builder_edge["T"] == reporter_edge["T"]
        assert builder_edge["T_centered_cos"] == reporter_edge["T_centered_cos"]
