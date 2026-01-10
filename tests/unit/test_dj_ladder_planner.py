import numpy as np

from src.playlist.pier_bridge_builder import (
    _genre_vocab_map,
    _label_to_genre_vector,
    _label_to_smoothed_vector,
    _shortest_genre_path,
    _build_genre_targets,
    PierBridgeConfig,
)
from src.playlist.pier_bridge_diagnostics import SegmentDiagnostics


def test_shortest_genre_path_simple_graph():
    graph = {
        "a": [("b", 0.9), ("c", 0.2)],
        "b": [("c", 0.9)],
        "c": [("b", 0.9)],
    }
    path = _shortest_genre_path(graph, "a", "c", max_steps=2)
    assert path == ["a", "b", "c"]


def test_label_to_vector_onehot_maps_vocab():
    vocab = np.array(["rock", "pop"])
    vocab_map = _genre_vocab_map(vocab)
    vec = _label_to_genre_vector("pop", genre_vocab=vocab, genre_vocab_map=vocab_map)
    assert vec is not None
    assert vec.tolist() == [0.0, 1.0]


def test_smoothed_vector_maps_and_normalizes():
    vocab = np.array(["rock", "alt rock", "jazz"])
    vocab_map = _genre_vocab_map(vocab)

    def similarity_fn(a: str, b: str) -> float:
        scores = {
            ("rock", "rock"): 1.0,
            ("rock", "alt rock"): 0.8,
            ("rock", "jazz"): 0.1,
        }
        return scores.get((a, b), 0.0)

    vec, stats = _label_to_smoothed_vector(
        "rock",
        genre_vocab=vocab,
        genre_vocab_map=vocab_map,
        top_k=2,
        min_sim=0.2,
        similarity_fn=similarity_fn,
    )
    assert vec is not None
    assert stats["nonzero"] == 2
    assert np.isclose(float(np.linalg.norm(vec)), 1.0)
    assert vec[vocab_map["rock"]] > 0
    assert vec[vocab_map["alt rock"]] > 0


def test_smoothed_vector_fallbacks_to_onehot_when_empty():
    vocab = np.array(["rock", "pop"])
    vocab_map = _genre_vocab_map(vocab)

    def similarity_fn(a: str, b: str) -> float:
        return 0.0

    vec, stats = _label_to_smoothed_vector(
        "rock",
        genre_vocab=vocab,
        genre_vocab_map=vocab_map,
        top_k=2,
        min_sim=0.5,
        similarity_fn=similarity_fn,
    )
    assert vec is None
    assert stats["nonzero"] == 0

    fallback = _label_to_genre_vector("rock", genre_vocab=vocab, genre_vocab_map=vocab_map)
    assert fallback is not None
    assert fallback.tolist() == [1.0, 0.0]


def test_ladder_diagnostics_emitted_and_plumbed():
    vocab = np.array(["a", "c"])
    genre_graph = {
        "a": [("b", 0.9)],
        "b": [("c", 0.9)],
        "c": [("b", 0.9)],
    }
    X_genre_norm = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    X_full_norm = np.eye(2, dtype=float)
    warnings: list[dict[str, object]] = []
    ladder_diag: dict[str, object] = {}
    cfg = PierBridgeConfig(
        dj_route_shape="ladder",
        dj_ladder_top_labels=1,
        dj_ladder_min_label_weight=0.0,
        dj_ladder_max_steps=3,
    )

    g_targets = _build_genre_targets(
        pier_a=0,
        pier_b=1,
        interior_length=2,
        X_full_norm=X_full_norm,
        X_genre_norm=X_genre_norm,
        genre_vocab=vocab,
        genre_graph=genre_graph,
        cfg=cfg,
        warnings=warnings,
        ladder_diag=ladder_diag,
    )

    assert g_targets is not None
    assert ladder_diag.get("route_shape") == "ladder"
    assert ladder_diag.get("ladder_waypoint_count") == 3
    labels = ladder_diag.get("ladder_waypoint_labels") or []
    assert "a" in labels and "c" in labels
    assert any(w.get("type") == "genre_ladder_label_unmapped" for w in warnings)

    diag = SegmentDiagnostics(
        pier_a_id="track_a",
        pier_b_id="track_b",
        target_length=2,
        actual_length=2,
        pool_size_initial=10,
        pool_size_final=5,
        expansions=1,
        beam_width_used=4,
        worst_edge_score=0.2,
        mean_edge_score=0.3,
        success=True,
        route_shape=str(ladder_diag.get("route_shape")),
        ladder_waypoint_labels=list(ladder_diag.get("ladder_waypoint_labels") or []),
        ladder_waypoint_count=int(ladder_diag.get("ladder_waypoint_count") or 0),
        ladder_waypoint_vector_mode=str(ladder_diag.get("ladder_waypoint_vector_mode") or "onehot"),
    )
    playlist_stats = {"playlist": {"segment_diagnostics": [diag.__dict__]}}
    seg0 = playlist_stats["playlist"]["segment_diagnostics"][0]
    assert seg0["route_shape"] == "ladder"
    assert seg0["ladder_waypoint_labels"]
