from types import SimpleNamespace

import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _sonic_tie_matrix(rows: int) -> np.ndarray:
    X = np.ones((rows, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _layered_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        track_ids=np.array(["pier-a", "bridge-ok", "genre-jump", "pier-b"], dtype=object),
        X_genre_leaf_idf=np.array(
            [
                [1.0, 0.0, 0.0],  # pier A: jangle pop
                [0.0, 1.0, 0.0],  # bridge-supported candidate: synth-pop
                [0.0, 0.0, 1.0],  # unexplained jump candidate: death metal
                [0.0, 1.0, 0.0],  # pier B: synth-pop
            ],
            dtype=float,
        ),
        X_genre_family=np.array(
            [
                [1.0, 0.0],  # pop
                [1.0, 0.0],  # pop
                [0.0, 1.0],  # metal
                [1.0, 0.0],  # pop
            ],
            dtype=float,
        ),
        X_genre_bridge=np.array(
            [
                [0.0, 1.0, 0.0],  # jangle pop can bridge to synth-pop
                [1.0, 0.0, 0.0],  # synth-pop can bridge back to jangle pop
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        X_facet=np.ones((4, 1), dtype=float),
    )


def test_beam_layered_transition_scoring_prefers_explained_bridge_candidate():
    Xn = _sonic_tie_matrix(4)
    edge_components: dict[str, object] = {}

    path, _hits, _edges, err = _beam_search_segment(
        0,
        3,
        1,
        [2, 1],
        Xn,
        Xn,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            bridge_floor=-1.0,
            transition_floor=-1.0,
            progress_enabled=False,
            layered_transition_scoring_enabled=True,
            layered_transition_weight=0.75,
            layered_transition_mode="dynamic",
        ),
        5,
        bundle=_layered_bundle(),
        edge_components_out=edge_components,
    )

    assert err is None
    assert path == [1]
    first_edge = edge_components["components"][0]
    assert first_edge["layered_transition_reason"] == "bridge_supported"
    assert first_edge["layered_transition_score"] > 0


def test_beam_layered_transition_scoring_default_off_preserves_legacy_tie_order():
    Xn = _sonic_tie_matrix(4)

    path, _hits, _edges, err = _beam_search_segment(
        0,
        3,
        1,
        [2, 1],
        Xn,
        Xn,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            bridge_floor=-1.0,
            transition_floor=-1.0,
            progress_enabled=False,
        ),
        5,
        bundle=_layered_bundle(),
    )

    assert err is None
    assert path == [2]


def test_beam_layered_transition_scoring_ignores_flat_genre_penalty():
    Xn = _sonic_tie_matrix(4)
    legacy_flat_genres = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    path, hits, _edges, err = _beam_search_segment(
        0,
        3,
        1,
        [1],
        Xn,
        Xn,
        None,
        None,
        None,
        legacy_flat_genres,
        PierBridgeConfig(
            bridge_floor=-1.0,
            transition_floor=-1.0,
            progress_enabled=False,
            genre_penalty_threshold=0.5,
            genre_penalty_strength=0.75,
            layered_transition_scoring_enabled=True,
            layered_transition_weight=0.75,
            layered_transition_mode="dynamic",
        ),
        5,
        bundle=_layered_bundle(),
    )

    assert err is None
    assert path == [1]
    assert hits == 0
