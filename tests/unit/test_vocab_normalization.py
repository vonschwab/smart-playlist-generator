"""
Tests for src/genre/vocab_normalization.py and src/genre/llm_client.py.

All tests use DryRunLLMClient — no API calls, no env vars required.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.genre.llm_client import DryRunLLMClient, make_llm_client
from src.genre.vocab_normalization import (
    CanonicalizationDecision,
    _pick_canonical_form,
    _strip_punctuation,
    classify_cluster_batch,
    cluster_by_form,
    cluster_by_similarity,
    collect_raw_vocab,
    decisions_from_form_clusters,
    normalize_vocab,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def in_memory_db():
    """Minimal metadata DB with genre tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        INSERT INTO track_genres VALUES ('t1', 'trip hop', 'lastfm', 1.0);
        INSERT INTO track_genres VALUES ('t1', 'trip-hop', 'musicbrainz', 1.0);
        INSERT INTO track_genres VALUES ('t2', 'post-punk', 'lastfm', 1.0);
        INSERT INTO track_genres VALUES ('t3', 'indie rock', 'lastfm', 1.0);
        INSERT INTO track_genres VALUES ('t4', 'shoegaze', 'lastfm', 1.0);
        INSERT INTO album_genres VALUES ('a1', 'dark wave', 'discogs');
        INSERT INTO album_genres VALUES ('a1', 'darkwave', 'musicbrainz');
        INSERT INTO artist_genres VALUES ('Low', 'slowcore', 'lastfm');
    """)
    return conn


@pytest.fixture()
def dry_run_client() -> DryRunLLMClient:
    return DryRunLLMClient()


def _make_same_genre_response(pairs: list[list[str]]) -> dict:
    """Build a response where all pairs are classified as same_genre=True."""
    return [
        {
            "same_genre": True,
            "canonical": pair[0],
            "confidence": 0.95,
            "reasoning": "Test: confirmed same genre.",
        }
        for pair in pairs
    ]


def _make_different_genre_response(pairs: list[list[str]]) -> dict:
    """Build a response where all pairs are classified as same_genre=False."""
    return [
        {
            "same_genre": False,
            "canonical": None,
            "confidence": 0.95,
            "reasoning": "Test: different genres.",
        }
        for pair in pairs
    ]


# ---------------------------------------------------------------------------
# _strip_punctuation
# ---------------------------------------------------------------------------

def test_strip_punctuation_hyphen():
    assert _strip_punctuation("trip-hop") == "triphop"


def test_strip_punctuation_space():
    assert _strip_punctuation("trip hop") == "triphop"


def test_strip_punctuation_mixed():
    assert _strip_punctuation("chill-out") == "chillout"
    assert _strip_punctuation("chill out") == "chillout"
    assert _strip_punctuation("chillout") == "chillout"


# ---------------------------------------------------------------------------
# cluster_by_form
# ---------------------------------------------------------------------------

def test_cluster_by_form_finds_hyphenation_pair():
    tokens = ["trip hop", "trip-hop", "post-punk", "shoegaze"]
    clusters = cluster_by_form(tokens)
    assert any(sorted(c) == ["trip hop", "trip-hop"] for c in clusters)


def test_cluster_by_form_finds_three_way():
    tokens = ["chill out", "chill-out", "chillout"]
    clusters = cluster_by_form(tokens)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["chill out", "chill-out", "chillout"]


def test_cluster_by_form_no_cluster_for_unique():
    tokens = ["shoegaze", "slowcore", "post-punk"]
    clusters = cluster_by_form(tokens)
    assert clusters == []


def test_cluster_by_form_dark_wave():
    tokens = ["dark wave", "darkwave"]
    clusters = cluster_by_form(tokens)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["dark wave", "darkwave"]


# ---------------------------------------------------------------------------
# cluster_by_similarity
# ---------------------------------------------------------------------------

def test_cluster_by_similarity_finds_afrobeat():
    tokens = ["afrobeat", "afrobeats", "jazz", "punk"]
    pairs = cluster_by_similarity(tokens)
    assert any(sorted(p) == ["afrobeat", "afrobeats"] for p in pairs)


def test_cluster_by_similarity_excludes_already_form_clustered():
    # "trip hop" / "trip-hop" should be found by form clustering already
    # similarity clustering on the FULL list should not re-include them
    tokens = ["trip hop", "trip-hop", "post-punk", "punk rock", "funk rock"]
    pairs = cluster_by_similarity(tokens)
    # "trip hop" / "trip-hop" should NOT appear in similarity clusters
    for pair in pairs:
        assert not (set(pair) == {"trip hop", "trip-hop"})


def test_cluster_by_similarity_finds_gothic_rock():
    tokens = ["goth rock", "gothic rock", "jazz"]
    pairs = cluster_by_similarity(tokens)
    assert any(sorted(p) == ["goth rock", "gothic rock"] for p in pairs)


# ---------------------------------------------------------------------------
# _pick_canonical_form
# ---------------------------------------------------------------------------

def test_pick_canonical_prefers_hyphenated():
    # "post-punk" preferred over "post punk"
    assert _pick_canonical_form(["post punk", "post-punk"]) == "post-punk"


def test_pick_canonical_solid_britpop():
    assert _pick_canonical_form(["brit pop", "britpop"]) == "britpop"


def test_pick_canonical_solid_bossanova():
    assert _pick_canonical_form(["bossa nova", "bossanova"]) == "bossanova"


# ---------------------------------------------------------------------------
# decisions_from_form_clusters
# ---------------------------------------------------------------------------

def test_form_cluster_decisions_are_rule_sourced():
    clusters = [["trip hop", "trip-hop"], ["dark wave", "darkwave"]]
    decisions = decisions_from_form_clusters(clusters)
    assert all(d.source == "rule" for d in decisions)
    assert all(d.is_same_genre for d in decisions)
    assert all(d.confidence == 1.0 for d in decisions)


def test_form_cluster_decisions_canonical_in_cluster():
    clusters = [["trip hop", "trip-hop"]]
    decisions = decisions_from_form_clusters(clusters)
    assert decisions[0].canonical in decisions[0].tokens


# ---------------------------------------------------------------------------
# classify_cluster_batch (with DryRunLLMClient overrides)
# ---------------------------------------------------------------------------

def test_classify_batch_same_genre(dry_run_client):
    import hashlib, json
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    clusters = [["goth rock", "gothic rock"]]
    pairs_json = json.dumps([[c[0], c[1]] for c in clusters], indent=2)
    prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)
    h = dry_run_client.prompt_hash(prompt)
    dry_run_client._overrides[h] = _make_same_genre_response(clusters)

    decisions = classify_cluster_batch(clusters, dry_run_client)
    assert len(decisions) == 1
    assert decisions[0].is_same_genre is True
    assert decisions[0].canonical in ["goth rock", "gothic rock"]


def test_classify_batch_different_genre(dry_run_client):
    import hashlib, json
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    clusters = [["funk rock", "punk rock"]]
    pairs_json = json.dumps([[c[0], c[1]] for c in clusters], indent=2)
    prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)
    h = dry_run_client.prompt_hash(prompt)
    dry_run_client._overrides[h] = _make_different_genre_response(clusters)

    decisions = classify_cluster_batch(clusters, dry_run_client)
    assert len(decisions) == 1
    assert decisions[0].is_same_genre is False


def test_classify_batch_uses_cache(dry_run_client):
    """Second call with same cluster key uses cache, not LLM."""
    import json
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    clusters = [["goth rock", "gothic rock"]]
    pairs_json = json.dumps([[c[0], c[1]] for c in clusters], indent=2)
    prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)
    h = dry_run_client.prompt_hash(prompt)
    dry_run_client._overrides[h] = _make_same_genre_response(clusters)

    cache: dict = {}
    classify_cluster_batch(clusters, dry_run_client, cache=cache)

    # Remove override — second call MUST use cache
    dry_run_client._overrides.clear()
    decisions2 = classify_cluster_batch(clusters, dry_run_client, cache=cache)
    assert decisions2[0].is_same_genre is True


def test_classify_batch_idempotent(dry_run_client):
    """Running classify twice with same cache produces identical decisions."""
    import json
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    clusters = [["goth rock", "gothic rock"]]
    pairs_json = json.dumps([[c[0], c[1]] for c in clusters], indent=2)
    prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)
    h = dry_run_client.prompt_hash(prompt)
    dry_run_client._overrides[h] = _make_same_genre_response(clusters)

    cache: dict = {}
    d1 = classify_cluster_batch(clusters, dry_run_client, cache=cache)[0]
    d2 = classify_cluster_batch(clusters, dry_run_client, cache=cache)[0]
    assert d1.canonical == d2.canonical
    assert d1.is_same_genre == d2.is_same_genre


def test_classify_batch_failed_call_is_not_cached():
    class FailingClient:
        provider = "test"

        def complete_json(self, prompt, *, max_retries=2):
            raise RuntimeError("temporary failure")

    cache: dict = {}
    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], FailingClient(), cache=cache
    )

    assert decisions[0].is_same_genre is False
    assert cache == {}


def test_classify_batch_dry_run_fallback_is_not_cached():
    cache: dict = {}

    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], DryRunLLMClient(), cache=cache
    )

    assert decisions[0].source == "dry_run"
    assert cache == {}


def test_classify_batch_malformed_response_degrades_gracefully():
    class MalformedClient:
        provider = "test"

        def complete_json(self, prompt, *, max_retries=2):
            return {"items": ["not an object"]}

    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], MalformedClient()
    )

    assert decisions[0].is_same_genre is False
    assert decisions[0].reasoning == "LLM response malformed item"


@pytest.mark.parametrize(
    "item",
    [
        {"canonical": "goth rock", "confidence": 0.9, "reasoning": "missing bool"},
        {"same_genre": 1, "canonical": "goth rock", "confidence": 0.9, "reasoning": "bad bool"},
        {"same_genre": True, "canonical": None, "confidence": 0.9, "reasoning": "bad canonical"},
        {"same_genre": True, "canonical": "", "confidence": 0.9, "reasoning": "bad canonical"},
        {"same_genre": False, "canonical": 3, "confidence": 0.9, "reasoning": "bad canonical"},
        {"same_genre": True, "canonical": "goth rock", "confidence": "high", "reasoning": "bad confidence"},
        {"same_genre": True, "canonical": "goth rock", "confidence": math.nan, "reasoning": "bad confidence"},
        {"same_genre": True, "canonical": "goth rock", "confidence": 1.1, "reasoning": "bad confidence"},
        {"same_genre": True, "canonical": "goth rock", "confidence": 0.9, "reasoning": ["bad reasoning"]},
    ],
)
def test_classify_batch_malformed_item_falls_back_without_caching(item):
    class MalformedClient:
        provider = "test"

        def complete_json(self, prompt, *, max_retries=2):
            return {"items": [item]}

    cache: dict = {}
    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], MalformedClient(), cache=cache
    )

    assert decisions[0].is_same_genre is False
    assert decisions[0].reasoning == "LLM response malformed item"
    assert cache == {}


def test_classify_batch_malformed_legacy_cache_entry_is_retried():
    class ValidClient:
        provider = "test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt, *, max_retries=2):
            self.calls += 1
            return {
                "items": [
                    {
                        "same_genre": True,
                        "canonical": "goth rock",
                        "confidence": 0.9,
                        "reasoning": "same genre",
                    }
                ]
            }

    client = ValidClient()
    cache = {"goth rock||gothic rock": {"broken": "legacy entry"}}

    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], client, cache=cache
    )

    assert client.calls == 1
    assert decisions[0].is_same_genre is True
    assert cache["goth rock||gothic rock"]["canonical"] == "goth rock"


def test_classify_batch_legacy_dry_run_cache_entry_is_retried():
    class ValidClient:
        provider = "test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt, *, max_retries=2):
            self.calls += 1
            return {
                "items": [
                    {
                        "same_genre": True,
                        "canonical": "goth rock",
                        "confidence": 0.9,
                        "reasoning": "same genre",
                    }
                ]
            }

    client = ValidClient()
    cache = {
        "goth rock||gothic rock": {
            "tokens": ["goth rock", "gothic rock"],
            "canonical": "goth rock",
            "confidence": 0.0,
            "is_same_genre": False,
            "reasoning": "dry-run: LLM not called",
            "source": "dry_run",
        }
    }

    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], client, cache=cache
    )

    assert client.calls == 1
    assert decisions[0].is_same_genre is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical", "unrelated token"),
        ("source", "rule"),
        ("source", "dry_run"),
        ("source", "untrusted"),
    ],
)
def test_classify_batch_unsafe_cached_decision_is_retried(field, value):
    class ValidClient:
        provider = "test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt, *, max_retries=2):
            self.calls += 1
            return {
                "items": [
                    {
                        "same_genre": True,
                        "canonical": "goth rock",
                        "confidence": 0.9,
                        "reasoning": "same genre",
                    }
                ]
            }

    client = ValidClient()
    cached = {
        "tokens": ["goth rock", "gothic rock"],
        "canonical": "goth rock",
        "confidence": 0.9,
        "is_same_genre": True,
        "reasoning": "cached",
        "source": "llm",
    }
    cached[field] = value
    cache = {"goth rock||gothic rock": cached}

    decisions = classify_cluster_batch(
        [["goth rock", "gothic rock"]], client, cache=cache
    )

    assert client.calls == 1
    assert decisions[0].canonical == "goth rock"
    assert cache["goth rock||gothic rock"]["source"] == "llm"


def test_classify_batch_accepts_object_wrapped_items(dry_run_client):
    import json
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    clusters = [["goth rock", "gothic rock"]]
    pairs_json = json.dumps([[c[0], c[1]] for c in clusters], indent=2)
    prompt = _BATCH_PROMPT_TEMPLATE.format(pairs_json=pairs_json)
    h = dry_run_client.prompt_hash(prompt)
    dry_run_client._overrides[h] = {"items": _make_same_genre_response(clusters)}

    decisions = classify_cluster_batch(clusters, dry_run_client)

    assert decisions[0].is_same_genre is True


def test_vocab_prompt_requests_object_wrapped_items():
    from src.genre.vocab_normalization import _BATCH_PROMPT_TEMPLATE

    assert '{"items": [' in _BATCH_PROMPT_TEMPLATE.format(pairs_json="[]")


# ---------------------------------------------------------------------------
# collect_raw_vocab
# ---------------------------------------------------------------------------

def test_collect_raw_vocab_deduplicates(in_memory_db):
    vocab = collect_raw_vocab(in_memory_db)
    # "trip hop" and "trip-hop" are both present in track_genres
    assert "trip hop" in vocab
    assert "trip-hop" in vocab
    # No duplicates
    assert len(vocab) == len(set(vocab))


def test_collect_raw_vocab_includes_all_sources(in_memory_db):
    vocab = collect_raw_vocab(in_memory_db)
    # From album_genres
    assert "dark wave" in vocab
    assert "darkwave" in vocab
    # From artist_genres
    assert "slowcore" in vocab


def test_collect_raw_vocab_sorted(in_memory_db):
    vocab = collect_raw_vocab(in_memory_db)
    assert vocab == sorted(vocab)


# ---------------------------------------------------------------------------
# normalize_vocab (integration, no LLM calls)
# ---------------------------------------------------------------------------

def test_normalize_vocab_handles_form_clusters(dry_run_client):
    tokens = ["trip hop", "trip-hop", "dark wave", "darkwave", "shoegaze"]
    decisions = normalize_vocab(tokens, dry_run_client)
    # Rule-based decisions for form clusters
    merged_tokens: set[str] = set()
    for d in decisions:
        if d.is_same_genre:
            merged_tokens.update(d.tokens)
    assert "trip hop" in merged_tokens
    assert "trip-hop" in merged_tokens
    assert "dark wave" in merged_tokens
    assert "darkwave" in merged_tokens


def test_normalize_vocab_cache_file_written(dry_run_client, tmp_path):
    tokens = ["trip hop", "trip-hop", "afrobeat", "afrobeats"]
    cache_file = tmp_path / "test_cache.json"
    normalize_vocab(tokens, dry_run_client, cache_path=cache_file)
    # Cache file should exist (even if empty for dry-run)
    assert cache_file.exists()
    assert json.loads(cache_file.read_text()) == {}


def test_normalize_vocab_malformed_top_level_cache_is_retried(tmp_path):
    class ValidClient:
        provider = "test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt, *, max_retries=2):
            self.calls += 1
            return {
                "items": [
                    {
                        "same_genre": True,
                        "canonical": "afrobeat",
                        "confidence": 0.9,
                        "reasoning": "same genre",
                    }
                ]
            }

    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps(["malformed legacy cache"]))
    client = ValidClient()

    decisions = normalize_vocab(
        ["afrobeat", "afrobeats"], client, cache_path=cache_file
    )

    assert client.calls == 1
    assert decisions[0].is_same_genre is True


def test_normalize_vocab_idempotent(dry_run_client, tmp_path):
    """Running normalize_vocab twice with the same cache file is idempotent."""
    tokens = ["trip hop", "trip-hop", "dark wave", "darkwave"]
    cache_file = tmp_path / "cache.json"

    d1 = normalize_vocab(tokens, dry_run_client, cache_path=cache_file)
    d2 = normalize_vocab(tokens, dry_run_client, cache_path=cache_file)

    # Same number of decisions
    assert len(d1) == len(d2)
    # Same canonical forms
    canon1 = {frozenset(d.tokens): d.canonical for d in d1}
    canon2 = {frozenset(d.tokens): d.canonical for d in d2}
    assert canon1 == canon2


# ---------------------------------------------------------------------------
# make_llm_client
# ---------------------------------------------------------------------------

def test_make_llm_client_dry_run():
    client = make_llm_client(dry_run=True)
    assert client.provider == "dry-run"


def test_make_llm_client_dry_run_via_provider():
    client = make_llm_client(provider="dry-run")
    assert client.provider == "dry-run"


def test_make_llm_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        make_llm_client(provider="unknown_provider_xyz")


def test_dry_run_client_deterministic():
    client = DryRunLLMClient()
    prompt = "Test prompt for determinism."
    r1 = client.complete_json(prompt)
    r2 = client.complete_json(prompt)
    assert r1 == r2


def test_dry_run_client_override():
    client = DryRunLLMClient()
    prompt = "Test prompt."
    h = client.prompt_hash(prompt)
    client._overrides[h] = {"test": "value"}
    assert client.complete_json(prompt) == {"test": "value"}
