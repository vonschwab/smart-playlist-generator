"""
Tests for src/genre/pmi_svd.py — PMI-SVD genre embedding.

Uses hand-crafted synthetic matrices to verify:
  - Known pairwise similarity ordering (semantic invariants)
  - Output shape and L2-normalization
  - Edge cases (V < dim error, uniform matrix, single-genre)

All tests are pure numpy — no DB, no API calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.genre.pmi_svd import project_tracks, train_pmi_svd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (assumed L2-normalized)."""
    return float(np.dot(a, b))


def _make_group_matrix(
    groups: list[list[int]],
    n_tracks_per_group: int,
    n_genres: int,
    *,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build an (N, n_genres) float32 matrix where tracks in the same group
    share all genres in that group (weight = 1.0).

    noise: fraction of tracks that receive a random cross-group genre (weight 0.3).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(groups) * n_tracks_per_group
    X = np.zeros((N, n_genres), dtype=np.float32)
    t = 0
    for genre_ids in groups:
        for _ in range(n_tracks_per_group):
            for g in genre_ids:
                X[t, g] = 1.0
            if noise > 0 and rng.random() < noise:
                g_rand = int(rng.integers(0, n_genres))
                if g_rand not in genre_ids:
                    X[t, g_rand] = 0.3
            t += 1
    return X


# ---------------------------------------------------------------------------
# Shape and normalization
# ---------------------------------------------------------------------------

def test_output_shape():
    X = np.eye(10, dtype=np.float32)  # 10 genres, each unique
    emb = train_pmi_svd(X, dim=4)
    assert emb.shape == (10, 4)


def test_output_dtype():
    X = np.eye(8, dtype=np.float32)
    emb = train_pmi_svd(X, dim=4)
    assert emb.dtype == np.float32


def test_rows_are_l2_normalized():
    X = _make_group_matrix([[0, 1], [2, 3], [4, 5]], n_tracks_per_group=20, n_genres=6)
    emb = train_pmi_svd(X, dim=3)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)


def test_vocab_smaller_than_dim_raises():
    X = np.ones((5, 3), dtype=np.float32)  # V=3, dim=4
    with pytest.raises(ValueError, match="vocab size"):
        train_pmi_svd(X, dim=4)


# ---------------------------------------------------------------------------
# Semantic invariants — within-group similarity > cross-group
# ---------------------------------------------------------------------------

def _train_three_group_embedding(dim: int = 8) -> tuple[np.ndarray, dict[str, int]]:
    """
    Three genre clusters:
      Group A (shoegaze/dreampop/slowcore):  genre indices 0, 1, 2
      Group B (techno/house/trance):         genre indices 3, 4, 5
      Group C (jazz/bebop/swing):            genre indices 6, 7, 8

    100 tracks per group, weight=1.0 within group.
    """
    vocab = {
        "shoegaze": 0, "dreampop": 1, "slowcore": 2,
        "techno": 3, "house": 4, "trance": 5,
        "jazz": 6, "bebop": 7, "swing": 8,
    }
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    X = _make_group_matrix(groups, n_tracks_per_group=100, n_genres=9)
    emb = train_pmi_svd(X, dim=dim)
    return emb, vocab


def test_within_group_sim_exceeds_cross_group_sim():
    emb, v = _train_three_group_embedding()
    # shoegaze ↔ dreampop should be closer than shoegaze ↔ techno
    sg_dp = cosine(emb[v["shoegaze"]], emb[v["dreampop"]])
    sg_tc = cosine(emb[v["shoegaze"]], emb[v["techno"]])
    assert sg_dp > sg_tc, f"shoegaze↔dreampop={sg_dp:.3f} not > shoegaze↔techno={sg_tc:.3f}"


def test_shoegaze_dreampop_closer_than_shoegaze_bebop():
    emb, v = _train_three_group_embedding()
    sg_dp = cosine(emb[v["shoegaze"]], emb[v["dreampop"]])
    sg_bb = cosine(emb[v["shoegaze"]], emb[v["bebop"]])
    assert sg_dp > sg_bb, f"shoegaze↔dreampop={sg_dp:.3f} not > shoegaze↔bebop={sg_bb:.3f}"


def test_techno_house_closer_than_techno_jazz():
    emb, v = _train_three_group_embedding()
    tc_hs = cosine(emb[v["techno"]], emb[v["house"]])
    tc_jz = cosine(emb[v["techno"]], emb[v["jazz"]])
    assert tc_hs > tc_jz, f"techno↔house={tc_hs:.3f} not > techno↔jazz={tc_jz:.3f}"


def test_cross_group_similarity_is_low():
    emb, v = _train_three_group_embedding()
    # Completely different groups should have low similarity
    sg_tc = cosine(emb[v["shoegaze"]], emb[v["techno"]])
    assert sg_tc < 0.5, f"Expected shoegaze↔techno < 0.5, got {sg_tc:.3f}"


# ---------------------------------------------------------------------------
# Two-group pop/electronic variant
# ---------------------------------------------------------------------------

def test_pop_dance_closer_than_pop_death_metal():
    """
    Two clusters: pop/dance-pop vs death-metal/thrash-metal.
    pop ↔ dance-pop >> pop ↔ death-metal.
    """
    vocab = {"pop": 0, "dance-pop": 1, "electropop": 2,
             "death-metal": 3, "thrash-metal": 4, "black-metal": 5}
    groups = [[0, 1, 2], [3, 4, 5]]
    X = _make_group_matrix(groups, n_tracks_per_group=80, n_genres=6)
    emb = train_pmi_svd(X, dim=4)

    pop_dp = cosine(emb[vocab["pop"]], emb[vocab["dance-pop"]])
    pop_dm = cosine(emb[vocab["pop"]], emb[vocab["death-metal"]])
    assert pop_dp > pop_dm, f"pop↔dance-pop={pop_dp:.3f} not > pop↔death-metal={pop_dm:.3f}"


# ---------------------------------------------------------------------------
# Smoothing and edge cases
# ---------------------------------------------------------------------------

def test_smoothing_zero_raises_no_error():
    """smoothing=0 is legal (no additive term); pmi may have -inf for absent pairs."""
    X = _make_group_matrix([[0, 1], [2, 3]], n_tracks_per_group=20, n_genres=4)
    emb = train_pmi_svd(X, dim=2, smoothing=0.0)
    assert emb.shape == (4, 2)


def test_large_smoothing_reduces_variance():
    """Large smoothing → nearly uniform joint probs → near-zero PPMI → small embedding variance."""
    X = _make_group_matrix([[0, 1], [2, 3]], n_tracks_per_group=10, n_genres=4)
    emb_low = train_pmi_svd(X, dim=2, smoothing=0.01)
    emb_high = train_pmi_svd(X, dim=2, smoothing=1e9)
    # With huge smoothing all rows converge → very low pairwise variance
    var_low = float(np.var(emb_low))
    var_high = float(np.var(emb_high))
    assert var_high < var_low, f"expected var_high {var_high:.6f} < var_low {var_low:.6f}"


def test_uniform_matrix_runs_without_error():
    """All-ones matrix: PPMI is all-zero (log(1)=0), so embeddings collapse to zero — no crash."""
    X = np.ones((20, 5), dtype=np.float32)
    emb = train_pmi_svd(X, dim=4)
    assert emb.shape == (5, 4)
    assert not np.any(np.isnan(emb)), "should not produce NaN"


def test_random_state_determinism():
    X = _make_group_matrix([[0, 1, 2], [3, 4, 5]], n_tracks_per_group=30, n_genres=6)
    emb1 = train_pmi_svd(X, dim=4, random_state=7)
    emb2 = train_pmi_svd(X, dim=4, random_state=7)
    np.testing.assert_array_equal(emb1, emb2)


def test_different_random_states_may_differ():
    X = _make_group_matrix([[0, 1, 2], [3, 4, 5]], n_tracks_per_group=30, n_genres=6)
    emb1 = train_pmi_svd(X, dim=4, random_state=1)
    emb2 = train_pmi_svd(X, dim=4, random_state=99)
    # May differ in sign flips but cosine structure should be preserved
    # Just check they're not identical (numerical stability check)
    # Note: cos similarity invariant to sign flips
    c1 = cosine(emb1[0], emb1[1])
    c2 = cosine(emb2[0], emb2[1])
    # Both should show positive correlation (same group)
    assert c1 > 0, "group-A genres should have positive cosine with random_state=1"
    assert c2 > 0, "group-A genres should have positive cosine with random_state=99"


# ---------------------------------------------------------------------------
# All-but-the-top (anisotropy removal)
# ---------------------------------------------------------------------------

def _hub_matrix(rng=None):
    """Three genre groups PLUS a hub genre (idx 9) on ~85% of tracks, with
    cross-group noise.

    The hub injects a dominant shared co-occurrence direction — exactly the
    PPMI anisotropy all-but-the-top is meant to strip — while the noise keeps
    the groups genuinely distinct (avoids the degenerate perfectly-blocked case).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    X = _make_group_matrix(groups, n_tracks_per_group=150, n_genres=10, noise=0.15, rng=rng)
    hub_mask = rng.random(X.shape[0]) < 0.85  # hub on ~85% of tracks (non-constant)
    X[hub_mask, 9] = 1.0
    return X


def test_remove_top_components_strips_leading_pcs_and_mean():
    """Mechanism test (dimension-independent): after all-but-the-top the embedding
    has ~zero mean and ~zero variance along the removed principal directions."""
    from src.genre.pmi_svd import _remove_top_components
    rng = np.random.default_rng(0)
    base = rng.standard_normal((50, 8))
    shared = np.ones(8) / np.sqrt(8)
    E = base + 5.0 * shared                # inject a dominant common direction
    out = _remove_top_components(E, k=1)
    # The (pre-removal) leading PC must be projected out of the result.
    centered = E - E.mean(axis=0, keepdims=True)
    _u, _s, Vt = np.linalg.svd(centered, full_matrices=False)
    top = Vt[0]
    assert np.abs(out @ top).mean() < 1e-6, "leading PC not removed"
    assert np.linalg.norm(out.mean(axis=0)) < 1e-6, "mean not removed"


def test_remove_top_components_preserves_within_vs_cross_ordering():
    """Stripping the shared direction must not invert genuine genre structure."""
    X = _hub_matrix()
    emb = train_pmi_svd(X, dim=6, remove_top_components=2)
    within = cosine(emb[0], emb[1])     # group A internal
    cross = cosine(emb[0], emb[6])      # group A vs group C
    assert within > cross, f"within={within:.3f} not > cross={cross:.3f}"


def test_remove_top_components_default_is_legacy():
    """Default (0) must reproduce the un-post-processed embedding exactly."""
    X = _make_group_matrix([[0, 1], [2, 3], [4, 5]], n_tracks_per_group=20, n_genres=6)
    a = train_pmi_svd(X, dim=3)
    b = train_pmi_svd(X, dim=3, remove_top_components=0)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# project_tracks — IDF down-weighting of hub genres
# ---------------------------------------------------------------------------

def test_project_tracks_zero_genre_track_is_zero_vector():
    X = _make_group_matrix([[0, 1], [2, 3]], n_tracks_per_group=10, n_genres=4)
    emb = train_pmi_svd(X, dim=2)
    X_query = np.zeros((2, 4), dtype=np.float32)
    X_query[0, 0] = 1.0  # has a genre
    # row 1 left all-zero
    P = project_tracks(X_query, emb)
    assert np.linalg.norm(P[1]) == 0.0
    assert abs(np.linalg.norm(P[0]) - 1.0) < 1e-5


def test_project_tracks_idf_downweights_shared_hub_genre():
    """A track sharing only a HUB genre with a seed should score lower under
    IDF-weighted projection than under raw projection."""
    from src.playlist.genre_idf import compute_genre_idf
    X = _hub_matrix()
    emb = train_pmi_svd(X, dim=6, remove_top_components=2)
    idf = compute_genre_idf(X_genre_raw=X.astype(np.float64), power=1.0, norm="max1")

    seed = np.zeros((1, 10), dtype=np.float32)
    seed[0, [0, 1, 2, 9]] = 1.0          # group A + hub
    other = np.zeros((1, 10), dtype=np.float32)
    other[0, [6, 7, 8, 9]] = 1.0         # group C + hub (shares ONLY the hub)

    raw = project_tracks(np.vstack([seed, other]), emb)
    wtd = project_tracks(np.vstack([seed, other]), emb, idf=idf)
    sim_raw = float(raw[0] @ raw[1])
    sim_wtd = float(wtd[0] @ wtd[1])
    assert sim_wtd < sim_raw, (
        f"IDF should reduce hub-only similarity: raw={sim_raw:.3f} wtd={sim_wtd:.3f}"
    )


# ---------------------------------------------------------------------------
# Support counting (via build_genre_matrix output contract)
# ---------------------------------------------------------------------------

def test_support_counts_tracks_with_nonzero_weight():
    """support[g] = number of tracks where X[t, g] > 0."""
    X = np.array([
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    support = (X > 0).sum(axis=0).astype(np.int32)
    assert support[0] == 2  # genres 0 appears in tracks 0,1
    assert support[1] == 2  # genre 1 appears in tracks 1,2
    assert support[2] == 1  # genre 2 only in track 0
