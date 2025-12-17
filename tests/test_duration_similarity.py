import math

from src.similarity_calculator import SimilarityCalculator


def test_duration_similarity_exact_match():
    sim = SimilarityCalculator.duration_similarity(
        120, 120, window_frac=0.25, falloff=0.6
    )
    assert math.isclose(sim, 1.0, rel_tol=1e-6)


def test_duration_similarity_within_window():
    # 10% deviation within 25% window -> ~1.0
    sim = SimilarityCalculator.duration_similarity(
        100, 110, window_frac=0.25, falloff=0.6
    )
    assert sim == 1.0


def test_duration_similarity_far_penalized():
    sim_short = SimilarityCalculator.duration_similarity(
        100, 50, window_frac=0.25, falloff=0.6
    )
    sim_long = SimilarityCalculator.duration_similarity(
        100, 200, window_frac=0.25, falloff=0.6
    )
    assert sim_short < 0.8
    assert sim_long < 0.8
    assert math.isclose(sim_short, sim_long, rel_tol=0.2)


def test_duration_similarity_missing_values_neutral():
    assert (
        SimilarityCalculator.duration_similarity(
            0, 120, window_frac=0.25, falloff=0.6
        )
        == 1.0
    )
    assert (
        SimilarityCalculator.duration_similarity(
            120, 0, window_frac=0.25, falloff=0.6
        )
        == 1.0
    )
