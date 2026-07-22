from __future__ import annotations

import numpy as np
import pytest

from src.genre.blend import alpha_schedule, blend_with_prior


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha_at_zero": -0.01}, "alpha_at_zero"),
        ({"alpha_at_zero": 1.01}, "alpha_at_zero"),
        ({"alpha_at_zero": np.nan}, "alpha_at_zero"),
        ({"alpha_at_zero": np.inf}, "alpha_at_zero"),
        ({"alpha_at_zero": -np.inf}, "alpha_at_zero"),
        ({"half_life": 0}, "half_life"),
        ({"half_life": -1}, "half_life"),
        ({"half_life": np.nan}, "half_life"),
        ({"half_life": np.inf}, "half_life"),
        ({"half_life": -np.inf}, "half_life"),
    ],
)
@pytest.mark.parametrize("function", [blend_with_prior, alpha_schedule])
def test_blend_functions_reject_invalid_parameters(function, kwargs, match):
    support = np.array([0, 1], dtype=np.float32)
    if function is blend_with_prior:
        corpus = np.eye(2, dtype=np.float32)
        prior = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match=match):
            function(corpus, prior, support, **kwargs)
    else:
        with pytest.raises(ValueError, match=match):
            function(support, **kwargs)


@pytest.mark.parametrize(
    "support",
    [
        np.array(1),
        np.array([[1, 2]]),
        np.array([0, -1]),
        np.array([0, np.nan]),
        np.array([0, np.inf]),
        np.array([0, -np.inf]),
    ],
)
@pytest.mark.parametrize("function", [blend_with_prior, alpha_schedule])
def test_blend_functions_reject_invalid_support(function, support):
    if function is blend_with_prior:
        corpus = np.eye(2, dtype=np.float32)
        prior = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match="support"):
            function(corpus, prior, support)
    else:
        with pytest.raises(ValueError, match="support"):
            function(support)


def test_alpha_schedule_rejects_empty_support_with_actionable_error():
    with pytest.raises(ValueError, match="support must not be empty"):
        alpha_schedule(np.array([], dtype=np.float32))


def test_blend_with_prior_rejects_empty_vocabulary_with_actionable_error():
    empty_embedding = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="embeddings must contain vocabulary rows"):
        blend_with_prior(
            empty_embedding,
            empty_embedding,
            np.array([], dtype=np.float32),
        )


@pytest.mark.parametrize("embedding_name", ["corpus", "prior"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_blend_with_prior_rejects_non_finite_embeddings(embedding_name, bad_value):
    corpus = np.eye(2, dtype=np.float32)
    prior = np.eye(2, dtype=np.float32)
    embedding = corpus if embedding_name == "corpus" else prior
    embedding[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match=rf"{embedding_name}_emb must contain only finite values",
    ):
        blend_with_prior(corpus, prior, np.array([0, 1], dtype=np.float32))
