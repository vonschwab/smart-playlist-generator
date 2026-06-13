# tests/unit/test_pace_audition_build.py
import numpy as np

from scripts.pace_audition_build import (
    genre_cosine,
    edge_metrics,
    extract_interior_edges,
)


def test_genre_cosine_is_l2_normalized_dot():
    u = np.array([3.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    assert genre_cosine(u, v) == 1.0
    w = np.array([0.0, 5.0, 0.0])
    assert abs(genre_cosine(u, w)) < 1e-9


def test_genre_cosine_zero_vector_is_zero():
    assert genre_cosine(np.zeros(4), np.ones(4)) == 0.0


def test_edge_metrics_uses_log2_distance_and_genre_cos():
    m = edge_metrics(
        a_onset=2.0, b_onset=4.0, a_bpm=90.0, b_bpm=90.0,
        a_genre=np.array([1.0, 0.0]), b_genre=np.array([1.0, 0.0]),
    )
    assert abs(m["onset_log_dist"] - 1.0) < 1e-9   # 4/2 = one octave
    assert abs(m["bpm_log_dist"] - 0.0) < 1e-9
    assert abs(m["genre_cos"] - 1.0) < 1e-9


def test_extract_interior_edges_excludes_pier_adjacent():
    # positions:        0(pier) 1 2 3 4(pier) 5
    track_ids = ["p0", "a", "b", "c", "p1", "d"]
    piers = {"p0", "p1"}
    # interior edges = consecutive pairs with NEITHER endpoint a pier
    assert extract_interior_edges(track_ids, piers) == [(1, 2), (2, 3)]
