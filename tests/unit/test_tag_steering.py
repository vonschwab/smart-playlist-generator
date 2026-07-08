"""Tag-name -> dense steering target resolution."""
import logging

import numpy as np

from src.playlist.tag_steering import resolve_tag_steering_target, sonic_prototype_from_rows, sonic_global_mean

VOCAB = ["jazz-funk", "post-bop", "soul"]
EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_maps_case_insensitively_and_normalizes():
    target, mapped, unmapped = resolve_tag_steering_target(
        ["Jazz-Funk", " post-bop "], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert mapped == ["Jazz-Funk", "post-bop"]  # resolver stores stripped inputs
    assert unmapped == []
    np.testing.assert_allclose(target, np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5]))
    assert abs(np.linalg.norm(target) - 1.0) < 1e-9


def test_unmapped_tags_warn_and_are_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk", "vaporwave"], genre_vocab=VOCAB, genre_emb=EMB
        )
    assert unmapped == ["vaporwave"]
    assert target is not None
    assert any("not in the artifact genre vocabulary" in r.message for r in caplog.records)


def test_missing_genre_emb_warns_and_disables(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk"], genre_vocab=VOCAB, genre_emb=None
        )
    assert target is None and mapped == [] and unmapped == ["jazz-funk"]
    assert any("genre_emb" in r.message for r in caplog.records)


def test_no_tags_is_silent_none():
    target, mapped, unmapped = resolve_tag_steering_target(
        [], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert target is None and mapped == [] and unmapped == []


def test_no_tags_emits_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_tag_steering_target([], genre_vocab=VOCAB, genre_emb=EMB)
    assert caplog.records == []


def test_degenerate_zero_norm_target_returns_none_and_warns(caplog):
    # Two selected tags whose embedding rows cancel to a ~zero mean vector.
    vocab = ["cancel-a", "cancel-b"]
    emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["cancel-a", "cancel-b"], genre_vocab=vocab, genre_emb=emb
        )
    assert target is None
    assert mapped == ["cancel-a", "cancel-b"]
    assert unmapped == []
    assert any(
        "degenerate" in r.message.lower() or "zero-norm" in r.message.lower()
        for r in caplog.records
    )


def test_sonic_prototype_points_at_member_mean_direction():
    A = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1)) + 1e-3
    B = np.tile(np.array([0.0, 1.0, 0.0]), (10, 1)) + 1e-3
    M = np.vstack([A, B])
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(10)))
    assert n == 10
    assert cohesion > 0.9
    assert proto @ np.array([1.0, 0.0, 0.0]) > 0.8
    assert proto @ np.array([0.0, 1.0, 0.0]) < 0.2


def test_sonic_prototype_low_cohesion_for_scattered_rows():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((50, 16))
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(50)))
    assert proto is not None
    assert cohesion < 0.5


def test_sonic_prototype_empty_rows_returns_none():
    M = np.eye(4)
    proto, cohesion, n = sonic_prototype_from_rows(M, [])
    assert proto is None and n == 0


def test_global_mean_centering_subtracts_common_component():
    common = np.array([5.0, 0.0, 0.0])
    M = np.vstack([common + np.array([0, 1.0, 0]), common + np.array([0, 0, 1.0])])
    gm = sonic_global_mean(M)
    proto_centered, _, _ = sonic_prototype_from_rows(M, [0], global_mean=gm)
    assert abs(proto_centered[0]) < abs(proto_centered[1]) + abs(proto_centered[2])
