"""Tests for src/genre/graph_similarity.py — graph-derived genre similarity matrix.

Stage 2 of the graph-taxonomy integration. The matrix must match the NPZ
contract of src/analyze/genre_similarity.py ({genre_vocab, S, stats}) so the
artifact builder's _smooth_genres consumes it unchanged in Stage 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.genre.graph_adapter import load_graph_adapter
from src.genre.graph_similarity import (
    GraphSimilarityParams,
    build_graph_similarity,
    export_neighbor_yaml,
    save_graph_similarity_npz,
)

# Fixture graph (records format). Expected direct similarities with default params:
#   scene_adjacent a<->b: wc = 0.5*0.8 = 0.40 -> 0.25 + 0.8*0.40 = 0.57
#   is_a c->b:           wc = 0.75*0.85 = 0.6375 -> 0.30 + 0.8*0.6375 = 0.81
#   family_context d->dance, e->dance: wc = 0.40 -> 0.6*0.40 = 0.24
#   is_a g->dance:       wc = 0.68 -> 0.844 -> capped at broad_pair_cap 0.40
FIXTURE = {
    "taxonomy_version": "test-sim-0.1",
    "records": [
        {"name": "rock", "kind": "family", "status": "active", "specificity_score": 0.2},
        {"name": "dance", "kind": "umbrella", "status": "active", "specificity_score": 0.3},
        {
            "name": "a",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "rock", "edge_type": "family_context", "weight": 0.5, "confidence": 0.8},
                {"target": "b", "edge_type": "scene_adjacent", "weight": 0.5, "confidence": 0.8},
            ],
        },
        {"name": "b", "kind": "genre", "status": "active", "specificity_score": 0.6},
        {
            "name": "c",
            "kind": "subgenre",
            "status": "active",
            "specificity_score": 0.7,
            "parent_edges": [
                {"target": "b", "edge_type": "is_a", "weight": 0.75, "confidence": 0.85},
            ],
        },
        {
            "name": "d",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "dance", "edge_type": "family_context", "weight": 0.5, "confidence": 0.8},
            ],
        },
        {
            "name": "e",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "dance", "edge_type": "family_context", "weight": 0.5, "confidence": 0.8},
            ],
        },
        {
            "name": "g",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "dance", "edge_type": "is_a", "weight": 0.8, "confidence": 0.85},
            ],
        },
        # Review-status connector: usable as a path intermediate, not a dimension.
        {
            "name": "rev",
            "kind": "genre",
            "status": "review",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "x", "edge_type": "scene_adjacent", "weight": 0.5, "confidence": 0.8},
                {"target": "y", "edge_type": "scene_adjacent", "weight": 0.5, "confidence": 0.8},
            ],
        },
        {"name": "x", "kind": "genre", "status": "active", "specificity_score": 0.6},
        {"name": "y", "kind": "genre", "status": "active", "specificity_score": 0.6},
        {"name": "lo-fi", "kind": "facet", "facet_type": "production", "status": "active"},
        {"name": "seen live", "kind": "reject", "status": "rejected", "reject_reason": "source_noise"},
        {"name": "a prime", "kind": "alias", "status": "alias_only", "canonical_target": "a"},
    ],
}


@pytest.fixture(scope="module")
def result():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "taxonomy.yaml"
        path.write_text(yaml.safe_dump(FIXTURE, sort_keys=False), encoding="utf-8")
        adapter = load_graph_adapter(path)
        return build_graph_similarity(adapter)


def _sim(result, g1: str, g2: str) -> float:
    index = {g: i for i, g in enumerate(result.genre_vocab)}
    return float(result.S[index[g1], index[g2]])


# --- matrix structure --------------------------------------------------------


def test_matrix_shape_symmetry_diagonal_bounds(result):
    G = len(result.genre_vocab)
    assert result.S.shape == (G, G)
    assert result.S.dtype == np.float32
    assert np.allclose(result.S, result.S.T)
    assert np.allclose(np.diag(result.S), 1.0)
    off = result.S[~np.eye(G, dtype=bool)]
    assert off.min() >= 0.0
    assert off.max() <= 0.95


def test_vocab_is_sorted_active_genres_only(result):
    vocab = result.genre_vocab
    assert vocab == sorted(vocab)
    assert "lo-fi" not in vocab  # facet
    assert "seen live" not in vocab  # reject
    assert "a prime" not in vocab  # alias
    assert "rev" not in vocab  # review status excluded by default
    assert {"a", "b", "c", "rock", "dance"} <= set(vocab)


# --- direct edge similarity --------------------------------------------------


def test_scene_adjacent_direct_similarity(result):
    assert _sim(result, "a", "b") == pytest.approx(0.57, abs=1e-3)


def test_is_a_direct_similarity(result):
    assert _sim(result, "c", "b") == pytest.approx(0.81, abs=1e-3)


def test_family_context_is_damped(result):
    assert _sim(result, "d", "dance") == pytest.approx(0.24, abs=1e-3)


def test_broad_pair_cap_applies(result):
    # g --is_a--> dance (umbrella) would score 0.844; capped because dance is broad.
    assert _sim(result, "g", "dance") == pytest.approx(0.40, abs=1e-3)


# --- multi-hop paths ---------------------------------------------------------


def test_two_hop_path_with_decay(result):
    # a-b (0.57) * b-c (0.81) * hop_decay (0.85), b is not broad -> no hub damping
    assert _sim(result, "a", "c") == pytest.approx(0.57 * 0.81 * 0.85, abs=1e-3)


def test_path_through_broad_hub_is_damped(result):
    # d-dance (0.24) * dance-e (0.24) * 0.85 * hub_mult(dance)= 0.3/0.6 = 0.5
    assert _sim(result, "d", "e") == pytest.approx(0.24 * 0.24 * 0.85 * 0.5, abs=1e-3)
    assert _sim(result, "d", "e") < 0.05


def test_review_node_bridges_paths_but_is_not_a_dimension(result):
    assert "rev" not in result.genre_vocab
    # x-rev (0.57) * rev-y (0.57) * 0.85
    assert _sim(result, "x", "y") == pytest.approx(0.57 * 0.57 * 0.85, abs=1e-3)


def test_unconnected_pair_is_zero(result):
    assert _sim(result, "x", "a") == 0.0


# --- persistence -------------------------------------------------------------


def test_npz_round_trip_matches_legacy_contract(result, tmp_path: Path):
    out = tmp_path / "graph_sim.npz"
    save_graph_similarity_npz(result, out)
    data = np.load(out, allow_pickle=True)
    assert set(["genre_vocab", "S", "stats"]) <= set(data.files)
    vocab = [str(g) for g in data["genre_vocab"]]
    assert vocab == result.genre_vocab
    assert np.allclose(data["S"], result.S)
    stats = data["stats"].item()
    assert stats["taxonomy_version"] == "test-sim-0.1"


def test_neighbor_yaml_export_is_loadable_by_legacy_readers(result, tmp_path: Path):
    out = tmp_path / "graph_sim.yaml"
    export_neighbor_yaml(result, out, min_sim=0.10, top_k=5)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["a"]["b"] == pytest.approx(0.57, abs=1e-3)
    for src, neighbors in data.items():
        assert src not in neighbors  # no self edges
        for sim in neighbors.values():
            assert isinstance(sim, float) and sim >= 0.10


# --- live taxonomy goldens ---------------------------------------------------


@pytest.fixture(scope="module")
def live():
    return build_graph_similarity(load_graph_adapter())


def test_live_golden_neighbors(live):
    assert _sim(live, "shoegaze", "dream pop") >= 0.55
    assert _sim(live, "jangle pop", "twee pop") >= 0.50
    assert _sim(live, "post-punk", "dance-punk") >= 0.50
    assert _sim(live, "acid techno", "techno") >= 0.70  # is_a parent/child
    assert _sim(live, "drone music", "ambient") >= 0.15  # family membership: present, modest


def test_live_negative_controls(live):
    assert _sim(live, "post-rock", "post-punk") < 0.20  # the modifier-only trap
    assert _sim(live, "indie pop", "techno") < 0.15
    assert "lo-fi" not in live.genre_vocab
    assert "seen live" not in live.genre_vocab


def test_live_broad_rows_are_capped(live):
    params = GraphSimilarityParams()
    index = {g: i for i, g in enumerate(live.genre_vocab)}
    adapter = load_graph_adapter()
    for name in ("rock", "pop", "electronic"):
        node = adapter.node(name)
        if node is None or not node.is_broad or name not in index:
            continue
        row = live.S[index[name]].copy()
        row[index[name]] = 0.0
        assert row.max() <= params.broad_pair_cap + 1e-6, name
