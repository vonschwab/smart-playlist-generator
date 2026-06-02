"""Focused tests for the LLM-backed genre embedding prior."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.genre.llm_prior import _make_prior_prompt, build_llm_prior


class RecordingClient:
    provider = "test"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, *, max_retries: int = 2):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _identity_fixture():
    vocab = ["rock", "jazz", "ambient"]
    corpus_emb = np.eye(3, dtype=np.float32)
    support = np.array([30, 20, 1])
    return vocab, corpus_emb, support


def test_anchor_identity_uses_corpus_vectors_without_llm_calls():
    vocab, corpus_emb, support = _identity_fixture()
    client = RecordingClient([])

    prior = build_llm_prior(vocab, corpus_emb, support, client, n_anchors=3)

    np.testing.assert_allclose(prior, corpus_emb)
    assert client.prompts == []


def test_failed_batch_is_not_cached_and_is_retried(tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    failing = RecordingClient([RuntimeError("temporary failure")])

    build_llm_prior(
        vocab, corpus_emb, support, failing, cache_path, n_anchors=2, batch_size=1
    )

    cache = json.loads(cache_path.read_text())
    assert "ambient" not in cache["scores"]

    succeeding = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": [10.0, 0.0]}]}]
    )
    prior = build_llm_prior(
        vocab, corpus_emb, support, succeeding, cache_path, n_anchors=2, batch_size=1
    )

    assert len(succeeding.prompts) == 1
    np.testing.assert_allclose(prior[2], corpus_emb[0])


def test_cache_is_invalidated_when_ordered_anchors_change(tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "anchors": ["jazz", "rock"],
                "scores": {"ambient": [10.0, 0.0]},
            }
        )
    )
    client = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": [10.0, 0.0]}]}]
    )

    prior = build_llm_prior(
        vocab, corpus_emb, support, client, cache_path, n_anchors=2, batch_size=1
    )

    assert len(client.prompts) == 1
    np.testing.assert_allclose(prior[2], corpus_emb[0])


def test_legacy_cache_is_safely_invalidated(tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    cache_path.write_text(json.dumps({"ambient": [0.0, 10.0]}))
    client = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": [10.0, 0.0]}]}]
    )

    prior = build_llm_prior(
        vocab, corpus_emb, support, client, cache_path, n_anchors=2, batch_size=1
    )

    assert len(client.prompts) == 1
    np.testing.assert_allclose(prior[2], corpus_emb[0])


def test_malformed_response_falls_back_without_caching(tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    client = RecordingClient([{"items": ["not an object"]}])

    prior = build_llm_prior(
        vocab, corpus_emb, support, client, cache_path, n_anchors=2, batch_size=1
    )

    np.testing.assert_allclose(prior[2], corpus_emb[2])
    cache = json.loads(cache_path.read_text())
    assert "ambient" not in cache["scores"]


def test_mismatched_genre_response_falls_back_without_caching_and_retries(tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    mismatched = RecordingClient(
        [{"items": [{"genre": "not ambient", "scores": [10.0, 0.0]}]}]
    )

    prior = build_llm_prior(
        vocab, corpus_emb, support, mismatched, cache_path, n_anchors=2, batch_size=1
    )

    np.testing.assert_allclose(prior[2], corpus_emb[2])
    cache = json.loads(cache_path.read_text())
    assert "ambient" not in cache["scores"]

    succeeding = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": [10.0, 0.0]}]}]
    )
    prior = build_llm_prior(
        vocab, corpus_emb, support, succeeding, cache_path, n_anchors=2, batch_size=1
    )

    assert len(succeeding.prompts) == 1
    np.testing.assert_allclose(prior[2], corpus_emb[0])


def test_prior_prompt_requests_object_wrapped_items():
    prompt = _make_prior_prompt(["ambient"], ["rock"])

    assert '{"items": [' in prompt


@pytest.mark.parametrize(
    "scores",
    [
        [True, 0.0],
        ["10", 0.0],
        [np.nan, 0.0],
        [np.inf, 0.0],
        [-0.1, 0.0],
        [10.1, 0.0],
    ],
)
def test_invalid_response_scores_fall_back_without_caching(scores, tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    client = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": scores}]}]
    )

    prior = build_llm_prior(
        vocab, corpus_emb, support, client, cache_path, n_anchors=2, batch_size=1
    )

    np.testing.assert_allclose(prior[2], corpus_emb[2])
    cache = json.loads(cache_path.read_text())
    assert "ambient" not in cache["scores"]


@pytest.mark.parametrize(
    "scores",
    [
        [False, 0.0],
        ["10", 0.0],
        [np.nan, 0.0],
        [np.inf, 0.0],
        [-0.1, 0.0],
        [10.1, 0.0],
    ],
)
def test_invalid_cached_scores_are_retried(scores, tmp_path):
    vocab, corpus_emb, support = _identity_fixture()
    cache_path = tmp_path / "prior.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "anchors": ["rock", "jazz"],
                "scores": {"ambient": scores},
            }
        )
    )
    client = RecordingClient(
        [{"items": [{"genre": "ambient", "scores": [10.0, 0.0]}]}]
    )

    prior = build_llm_prior(
        vocab, corpus_emb, support, client, cache_path, n_anchors=2, batch_size=1
    )

    assert len(client.prompts) == 1
    np.testing.assert_allclose(prior[2], corpus_emb[0])
