import numpy as np
import pytest
from src.playlist.bpm_axis import (
    resolve_perceptual_bpm,
    bpm_log_distance,
    interpolate_log_bpm,
)


def test_resolve_perceptual_bpm_unambiguous_passes_through():
    assert resolve_perceptual_bpm(120.0, half_tempo_likely=False, double_tempo_likely=False) == 120.0


def test_resolve_perceptual_bpm_half_tempo_likely_doubles():
    assert resolve_perceptual_bpm(60.0, half_tempo_likely=True, double_tempo_likely=False) == 120.0


def test_resolve_perceptual_bpm_double_tempo_likely_halves():
    assert resolve_perceptual_bpm(160.0, half_tempo_likely=False, double_tempo_likely=True) == 80.0


def test_resolve_perceptual_bpm_both_flags_falls_back_to_primary():
    assert resolve_perceptual_bpm(70.0, half_tempo_likely=True, double_tempo_likely=True) == 70.0


def test_bpm_log_distance_identical():
    assert bpm_log_distance(120.0, 120.0) == 0.0


def test_bpm_log_distance_octave():
    np.testing.assert_allclose(bpm_log_distance(60.0, 120.0), 1.0, atol=1e-9)
    np.testing.assert_allclose(bpm_log_distance(120.0, 60.0), 1.0, atol=1e-9)


def test_bpm_log_distance_handles_zeros_safely():
    assert bpm_log_distance(0.0, 120.0) == float("inf")
    assert bpm_log_distance(120.0, -5.0) == float("inf")


def test_bpm_log_distance_vector_broadcasts():
    a = np.array([60.0, 120.0, 90.0])
    b = 120.0
    expected = np.array([1.0, 0.0, np.log2(120 / 90)])
    np.testing.assert_allclose(bpm_log_distance(a, b), expected, atol=1e-9)


def test_interpolate_log_bpm_endpoints():
    np.testing.assert_allclose(interpolate_log_bpm(60.0, 120.0, t=0.0), 60.0)
    np.testing.assert_allclose(interpolate_log_bpm(60.0, 120.0, t=1.0), 120.0)


def test_interpolate_log_bpm_midpoint_is_geometric_mean():
    result = interpolate_log_bpm(60.0, 240.0, t=0.5)
    np.testing.assert_allclose(result, 120.0, atol=1e-9)
