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


from scripts.pace_audition_build import sample_edges, synthesize_decoy_edges


def test_sample_edges_deterministic_and_bounded():
    edges = [(i, i + 1) for i in range(10)]
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = sample_edges(edges, k=3, rng=rng1)
    b = sample_edges(edges, k=3, rng=rng2)
    assert a == b              # same seed → same sample
    assert len(a) == 3
    assert all(e in edges for e in a)


def test_sample_edges_returns_all_when_fewer_than_k():
    edges = [(0, 1), (1, 2)]
    out = sample_edges(edges, k=5, rng=np.random.default_rng(0))
    assert sorted(out) == sorted(edges)


def test_synthesize_decoy_prefers_pace_distant_genre_close_pairs():
    # 3 tracks, all same genre. Onsets chosen so EXACTLY ONE pair exceeds the
    # 1.0-octave pace threshold: t0-t1=log2(1.5)=0.585 and t1-t2=log2(1.5)=0.585
    # are both <=1.0; only t0-t2=log2(2.25)=1.170 qualifies. So the single decoy
    # is {t0,t2} regardless of the RNG stream (unambiguous by construction).
    tids = ["t0", "t1", "t2"]
    onset = {"t0": 2.0, "t1": 3.0, "t2": 4.5}
    bpm = {"t0": 90.0, "t1": 91.0, "t2": 90.0}
    genre = {"t0": np.array([1.0, 0.0]), "t1": np.array([1.0, 0.0]),
             "t2": np.array([1.0, 0.0])}           # all identical genre
    decoys = synthesize_decoy_edges(
        tids, onset=onset, bpm=bpm, genre_vecs=genre,
        k=1, rng=np.random.default_rng(0), min_onset_dist=1.0,
    )
    assert len(decoys) == 1
    a, b = decoys[0]
    assert {a, b} == {"t0", "t2"}                  # the only pace-distant pair
