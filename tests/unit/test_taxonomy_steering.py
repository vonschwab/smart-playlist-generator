"""Tests for taxonomy-backed genre-arc steering (re-pointing steering off the
dense PMI-SVD embedding and onto the SP3a taxonomy graph).

The provider wraps the build-time graph similarity (hub-damped S over canonical
genres) for two runtime uses:
  - ladder routing  : canonical-name adjacency consumed by _shortest_genre_path
  - arc-vote scoring : taxonomy similarity(a, b) used to build smoothed waypoint
                       vectors over the artifact genre vocabulary
"""
from __future__ import annotations

import numpy as np

import pytest

from src.playlist.pier_bridge.taxonomy_steering import (
    TaxonomySteering,
    _filter_path_by_mass,
    build_taxonomy_genre_targets,
    get_taxonomy_steering,
)


# --- provider basics ---------------------------------------------------------


def test_provider_loads_live_taxonomy():
    steering = get_taxonomy_steering()
    assert len(steering.vocab) > 100
    # canonical names round-trip
    assert steering.canonical_label("shoegaze") == "shoegaze"


def test_canonical_label_unknown_is_none():
    steering = get_taxonomy_steering()
    assert steering.canonical_label("xyzzy not a real genre") is None


def test_similarity_matches_graph_golden_pairs():
    steering = get_taxonomy_steering()
    assert steering.similarity("shoegaze", "shoegaze") == 1.0
    # golden neighbors from the graph similarity matrix
    assert steering.similarity("shoegaze", "dream pop") >= 0.55
    assert steering.similarity("acid techno", "techno") >= 0.70


def test_similarity_unknown_is_zero():
    steering = get_taxonomy_steering()
    assert steering.similarity("xyzzy nonsense", "shoegaze") == 0.0


def test_similarity_modifier_trap_stays_low():
    # post-rock and post-punk share a modifier but are not close in the taxonomy
    steering = get_taxonomy_steering()
    assert steering.similarity("post-rock", "post-punk") < 0.20


def test_arc_adjacency_is_canonical_and_sorted():
    steering = get_taxonomy_steering()
    adj = steering.arc_adjacency()
    assert "shoegaze" in adj
    nbrs = adj["shoegaze"]
    assert all(isinstance(n, str) and isinstance(w, float) for n, w in nbrs)
    sims = [w for _, w in nbrs]
    assert sims == sorted(sims, reverse=True)
    assert "dream pop" in {n for n, _ in nbrs}


# --- target construction -----------------------------------------------------


def _vocab_fixture():
    vocab = np.array(
        ["twee pop", "jangle pop", "indie pop", "synth-pop", "techno"], dtype=object
    )
    # pier_a = twee pop track, pier_b = synth-pop track
    X = np.zeros((2, len(vocab)), dtype=np.float32)
    X[0, 0] = 1.0  # twee pop
    X[1, 3] = 1.0  # synth-pop
    return X, vocab


def test_build_targets_returns_arc_over_artifact_vocab():
    steering = get_taxonomy_steering()
    X, vocab = _vocab_fixture()
    targets = build_taxonomy_genre_targets(
        pier_a=0,
        pier_b=1,
        interior_length=3,
        X_genre_raw=X,
        genre_vocab=vocab,
        steering=steering,
    )
    assert targets is not None
    assert len(targets) == 3
    for t in targets:
        assert t.shape == (len(vocab),)
        assert np.isfinite(t).all()
        assert float(np.linalg.norm(t)) > 0.0

    twee = 0
    synth = 3
    # arc moves from the twee-pop end toward the synth-pop end
    assert targets[0][twee] >= targets[-1][twee]
    assert targets[-1][synth] >= targets[0][synth]


def test_is_broad_flags_umbrella_not_subgenre():
    steering = get_taxonomy_steering()
    assert steering.is_broad("rock") is True
    assert steering.is_broad("shoegaze") is False


def test_canonical_pier_labels_prefer_specific_over_broad():
    from src.playlist.pier_bridge.taxonomy_steering import _canonical_pier_labels

    steering = get_taxonomy_steering()
    vocab = np.array(["rock", "shoegaze"], dtype=object)
    g = np.array([1.0, 0.5], dtype=np.float32)  # broad "rock" has the higher raw weight
    labels = _canonical_pier_labels(
        g, vocab, steering, top_labels=5, min_label_weight=0.0
    )
    assert "shoegaze" in labels and "rock" in labels
    # specific genre ranked ahead of the broad umbrella despite lower raw weight,
    # so the arc endpoint does not collapse to "rock"
    assert labels.index("shoegaze") < labels.index("rock")


def test_build_targets_returns_none_when_no_canonical_genres():
    steering = get_taxonomy_steering()
    vocab = np.array(["xyzzy aaa", "xyzzy bbb"], dtype=object)
    X = np.zeros((2, 2), dtype=np.float32)
    X[0, 0] = 1.0
    X[1, 1] = 1.0
    targets = build_taxonomy_genre_targets(
        pier_a=0,
        pier_b=1,
        interior_length=3,
        X_genre_raw=X,
        genre_vocab=vocab,
        steering=steering,
    )
    assert targets is None  # caller falls back to dense steering


# --- routing graph truncation bug fix ----------------------------------------


def _build_synthetic_steering() -> TaxonomySteering:
    """17-genre synthetic S matrix that reproduces the new_wave→rock truncation bug.

    new_wave has 12 "decoy" neighbors at sim=0.20.  The backbone hub edge
    new_wave→rock sits at sim=0.15, which ranks 14th — outside the old top_k=12
    window.  Dijkstra must route synth-pop→city_pop→j_rock→rock (cost 1.48) when
    the backbone is invisible, but synth-pop→new_wave→rock (cost 1.15) when it
    isn't.
    """
    genres = ["synth-pop", "new_wave", "city_pop", "j_rock", "rock"] + [
        f"d{i:02d}" for i in range(12)
    ]
    idx = {g: i for i, g in enumerate(genres)}
    N = len(genres)
    S = np.eye(N, dtype=np.float32)

    def _s(a: str, b: str, v: float) -> None:
        S[idx[a], idx[b]] = v
        S[idx[b], idx[a]] = v

    _s("synth-pop", "new_wave", 0.70)
    _s("synth-pop", "city_pop", 0.50)
    _s("city_pop", "j_rock", 0.42)
    _s("j_rock", "rock", 0.60)
    _s("new_wave", "rock", 0.15)  # backbone hub edge; hub-damped
    for i in range(12):
        _s("new_wave", f"d{i:02d}", 0.20)  # fills new_wave's top-12, displacing rock

    return TaxonomySteering(genres, S, adapter=None)


def test_arc_adjacency_backbone_edge_not_truncated():
    """Default arc_adjacency() must preserve hub backbone edges.

    12 decoy neighbors at sim=0.20 push the backbone new_wave→rock edge (sim=0.15)
    to rank 14, outside the old top_k=12 window.  Dijkstra then routes via city_pop
    (cost 1.48) instead of the cheaper new_wave path (cost 1.15).

    Fix: remove the top_k cap so min_sim alone gates the graph.
    """
    from src.playlist.pier_bridge.genre import _shortest_genre_path

    steering = _build_synthetic_steering()
    adj = steering.arc_adjacency()  # default — must include the backbone edge

    new_wave_nbr_names = {n for n, _ in adj.get("new_wave", [])}
    assert "rock" in new_wave_nbr_names, (
        "backbone edge new_wave→rock must survive; raise/remove the top_k cap"
    )

    path = _shortest_genre_path(adj, "synth-pop", "rock", max_steps=6)
    assert path is not None
    assert "new_wave" in path, "cheaper 2-hop path via new_wave must be chosen"
    assert "city_pop" not in path, "scenic route via city_pop must not be chosen"


# --- library-mass waypoint filter ---------------------------------------------


def test_filter_path_by_mass_removes_low_mass_intermediates():
    path = ["synth-pop", "new_wave", "city_pop", "rock"]
    counts: dict[str, int] = {"synth-pop": 2000, "new_wave": 50, "city_pop": 20, "rock": 20000}
    result = _filter_path_by_mass(path, counts, min_mass=100)
    assert result == ["synth-pop", "rock"]


def test_filter_path_by_mass_keeps_endpoints_even_if_below_threshold():
    path = ["a", "b", "c"]
    counts: dict[str, int] = {"a": 0, "b": 999, "c": 0}
    result = _filter_path_by_mass(path, counts, min_mass=100)
    assert result[0] == "a"
    assert result[-1] == "c"
    assert "b" in result


def test_filter_path_by_mass_two_node_path_unchanged():
    path = ["a", "b"]
    result = _filter_path_by_mass(path, {}, min_mass=100)
    assert result == ["a", "b"]


def test_filter_path_by_mass_none_counts_returns_original():
    path = ["a", "b", "c"]
    result = _filter_path_by_mass(path, None, min_mass=100)
    assert result == ["a", "b", "c"]


def test_build_targets_filters_low_mass_intermediates():
    """genre_track_counts + min_waypoint_mass prune sparse waypoints from the arc path."""
    steering = get_taxonomy_steering()
    X, vocab = _vocab_fixture()

    # Baseline path (no mass filter)
    diag_base: dict = {}
    build_taxonomy_genre_targets(
        pier_a=0, pier_b=1, interior_length=5,
        X_genre_raw=X, genre_vocab=vocab, steering=steering,
        ladder_diag=diag_base,
    )
    baseline = diag_base.get("taxonomy_waypoint_labels", [])

    if len(baseline) <= 2:
        pytest.skip("no intermediate waypoints in baseline path to filter")

    # Zero every taxonomy genre's count — all intermediates must be stripped
    zero_counts: dict[str, int] = {g: 0 for g in steering.vocab}

    diag_filt: dict = {}
    targets_filt = build_taxonomy_genre_targets(
        pier_a=0, pier_b=1, interior_length=5,
        X_genre_raw=X, genre_vocab=vocab, steering=steering,
        genre_track_counts=zero_counts,
        min_waypoint_mass=1,
        ladder_diag=diag_filt,
    )
    assert targets_filt is not None  # falls back to 2-rung; still produces targets

    filtered = diag_filt.get("taxonomy_waypoint_labels", [])
    # No intermediate waypoints survive a min_mass=1 filter when all counts are 0
    assert len(filtered) <= 2, (
        f"expected ≤2 waypoints after zeroing all counts; got {filtered}"
    )
