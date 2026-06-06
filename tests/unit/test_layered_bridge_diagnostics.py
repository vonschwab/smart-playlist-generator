from types import SimpleNamespace

import numpy as np

from src.playlist.layered_bridge_diagnostics import build_layered_transition_diagnostics


def _bundle():
    return SimpleNamespace(
        track_ids=np.array(["t0", "t1", "t2"], dtype=object),
        X_genre_leaf_idf=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        X_genre_family=np.ones((3, 1), dtype=float),
        X_genre_bridge=np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        X_facet=np.ones((3, 1), dtype=float),
        genre_leaf_vocab=np.array(["jangle pop", "synth-pop"], dtype=object),
        genre_family_vocab=np.array(["pop"], dtype=object),
        genre_bridge_vocab=np.array(["jangle pop", "synth-pop"], dtype=object),
        facet_vocab=np.array(["reverb-heavy"], dtype=object),
    )


def test_layered_transition_diagnostics_summarize_selected_edges():
    diagnostics = build_layered_transition_diagnostics(
        bundle=_bundle(),
        track_indices=[0, 1, 2],
        edge_scores=[
            {"S": 0.82, "T": 0.84},
            {"S": 0.75, "T": 0.76},
        ],
        mode="dynamic",
        enabled=True,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["edge_count"] == 2
    assert diagnostics["explained_count"] == 2
    assert diagnostics["reason_counts"] == {
        "bridge_supported": 1,
        "leaf_continuity": 1,
    }
    assert diagnostics["samples"][0]["from_track_id"] == "t0"
    assert diagnostics["samples"][0]["to_track_id"] == "t1"
    assert diagnostics["samples"][0]["reason"] == "leaf_continuity"
    assert diagnostics["samples"][0]["shared_leaf_terms"] == ["jangle pop"]
    assert diagnostics["samples"][0]["shared_family_terms"] == ["pop"]
    assert diagnostics["samples"][1]["reason"] == "bridge_supported"
    assert diagnostics["samples"][1]["from_bridge_terms"] == []
    assert diagnostics["samples"][1]["to_leaf_terms"] == ["synth-pop"]
    assert diagnostics["samples"][1]["to_bridge_terms"] == ["jangle pop"]
    assert diagnostics["samples"][1]["shared_facet_terms"] == ["reverb-heavy"]


def test_layered_transition_diagnostics_handles_missing_matrices():
    bundle = SimpleNamespace(track_ids=np.array(["t0", "t1"], dtype=object))

    diagnostics = build_layered_transition_diagnostics(
        bundle=bundle,
        track_indices=[0, 1],
        edge_scores=[{"S": 0.8, "T": 0.8}],
        mode="dynamic",
        enabled=True,
    )

    assert diagnostics == {
        "enabled": False,
        "reason": "missing_layered_matrices",
    }


def test_layered_transition_diagnostics_disabled_is_noop():
    diagnostics = build_layered_transition_diagnostics(
        bundle=_bundle(),
        track_indices=[0, 1],
        edge_scores=[{"S": 0.8, "T": 0.8}],
        mode="dynamic",
        enabled=False,
    )

    assert diagnostics == {
        "enabled": False,
        "reason": "disabled",
    }
