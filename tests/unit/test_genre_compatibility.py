import numpy as np

import numpy as np

from src.playlist.genre_compatibility import compute_raw_genre_compatibility


def test_sparse_candidate_with_single_compatible_tag_is_not_penalized():
    vocab = ["indie pop", "rnb", "house", "punk"]
    seed_raw = np.array([1.0, 0.0, 0.0, 1.0])
    candidates_raw = np.array([
        [1.0, 0.0, 0.0, 0.0],
    ])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
    )

    assert result.compatible_mass[0] > 0
    assert result.conflict_mass[0] == 0
    assert result.confidence[0] == 1.0


def test_one_overlap_plus_many_conflicting_tags_has_low_confidence():
    vocab = ["indie pop", "rnb", "house", "soul", "funk", "punk"]
    seed_raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    candidates_raw = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
    )

    assert result.compatible_mass[0] > 0
    assert result.conflict_mass[0] > result.compatible_mass[0]
    assert result.confidence[0] < 0.5


def test_missing_candidate_raw_tags_is_uncertain_not_bad():
    vocab = ["indie pop", "punk"]
    seed_raw = np.array([1.0, 1.0])
    candidates_raw = np.array([[0.0, 0.0]])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
    )

    assert result.compatible_mass[0] == 0
    assert result.conflict_mass[0] == 0
    assert result.confidence[0] == 1.0
    assert result.missing_or_sparse[0]


def test_zero_denominator_candidates_do_not_emit_runtime_warning(recwarn):
    vocab = ["indie pop", "rnb", "house"]
    seed_raw = np.array([1.0, 0.0, 0.0])
    candidates_raw = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
        penalty_strength=0.2,
    )

    assert result.confidence.tolist() == [1.0, 1.0]
    assert result.penalty.tolist() == [0.0, 0.0]
    assert not recwarn
