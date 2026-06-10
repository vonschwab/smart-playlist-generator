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

from src.playlist.pier_bridge.taxonomy_steering import (
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
