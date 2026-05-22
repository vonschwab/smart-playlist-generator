import numpy as np
import pytest

from src.playlist.sonic_axes import (
    axis_cosine_similarity,
    extract_axis_vectors,
    interpolate_axis_vector,
)


def test_extract_axis_vectors_returns_correct_slices():
    X = np.arange(32 * 3, dtype=float).reshape(3, 32)
    axes = extract_axis_vectors(X, tower_pca_dims=(8, 16, 8))

    assert axes["rhythm"].shape == (3, 8)
    assert axes["timbre"].shape == (3, 16)
    assert axes["harmony"].shape == (3, 8)
    assert axes["color"].shape == (3, 24)
    np.testing.assert_array_equal(axes["rhythm"], X[:, :8])
    np.testing.assert_array_equal(axes["timbre"], X[:, 8:24])
    np.testing.assert_array_equal(axes["harmony"], X[:, 24:32])
    np.testing.assert_array_equal(axes["color"], X[:, 8:32])


def test_extract_axis_vectors_respects_custom_dims():
    X = np.arange(20 * 2, dtype=float).reshape(2, 20)
    axes = extract_axis_vectors(X, tower_pca_dims=(4, 12, 4))

    assert axes["rhythm"].shape == (2, 4)
    assert axes["timbre"].shape == (2, 12)
    assert axes["harmony"].shape == (2, 4)


def test_extract_axis_vectors_validates_dims():
    X = np.zeros((1, 32))
    with pytest.raises(ValueError, match="sum of tower_pca_dims"):
        extract_axis_vectors(X, tower_pca_dims=(8, 16, 16))


def test_axis_cosine_similarity_normalizes():
    v_a = np.array([[1.0, 0.0, 0.0]])
    v_b = np.array([[5.0, 0.0, 0.0]])

    sim = axis_cosine_similarity(v_a, v_b)

    np.testing.assert_allclose(sim, [[1.0]], atol=1e-9)


def test_axis_cosine_similarity_orthogonal_is_zero():
    v_a = np.array([[1.0, 0.0]])
    v_b = np.array([[0.0, 1.0]])

    sim = axis_cosine_similarity(v_a, v_b)

    np.testing.assert_allclose(sim, [[0.0]], atol=1e-9)


def test_interpolate_axis_vector_endpoints():
    R_a = np.array([1.0, 0.0, 0.0])
    R_b = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(interpolate_axis_vector(R_a, R_b, 0.0), R_a)
    np.testing.assert_allclose(interpolate_axis_vector(R_a, R_b, 1.0), R_b)


def test_interpolate_axis_vector_midpoint():
    R_a = np.array([1.0, 0.0])
    R_b = np.array([0.0, 1.0])

    mid = interpolate_axis_vector(R_a, R_b, 0.5)

    np.testing.assert_allclose(mid, [0.5, 0.5])
