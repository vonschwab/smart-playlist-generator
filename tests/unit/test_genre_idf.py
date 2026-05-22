"""Tests for the shared genre IDF computation."""
import numpy as np
import pytest

from src.playlist.genre_idf import compute_genre_idf


def _toy_matrix() -> np.ndarray:
    # 5 tracks, 4 genres. Genre 0 appears in all 5 (common).
    # Genre 1 in 4. Genre 2 in 2 (rare). Genre 3 in 1 (rarest).
    return np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=float)


def test_rare_genres_get_higher_weights():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    assert idf[3] > idf[2] > idf[1] > idf[0]


def test_max1_normalization_caps_at_one():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    assert float(np.max(idf)) == pytest.approx(1.0)
    assert float(np.min(idf)) > 0.0


def test_sum1_normalization_sums_to_one():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="sum1")
    assert float(np.sum(idf)) == pytest.approx(1.0)


def test_none_normalization_returns_raw_idf_values():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="none")
    assert float(np.max(idf)) > 1.0


def test_power_zero_collapses_to_uniform_weights():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=0.0, norm="none")
    assert np.allclose(idf, 1.0)


def test_power_two_amplifies_rare_more_than_power_one():
    idf_p1 = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    idf_p2 = compute_genre_idf(X_genre_raw=_toy_matrix(), power=2.0, norm="max1")
    assert idf_p2[0] < idf_p1[0]


def test_empty_matrix_returns_empty_array():
    idf = compute_genre_idf(
        X_genre_raw=np.zeros((0, 4), dtype=float),
        power=1.0,
        norm="max1",
    )
    assert idf.shape == (4,)
