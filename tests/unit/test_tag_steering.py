"""Tag-name -> dense steering target resolution."""
import logging

import numpy as np

from src.playlist.tag_steering import resolve_tag_steering_target

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
