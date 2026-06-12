"""Tests for src/genre/granularity.py — granularity ordering of genre tags.

Display-path helper: canonicalize raw tags through the SP3a taxonomy and order
most-specific first. Pinned against a fixture taxonomy so production growth
doesn't break the tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.genre.graph_adapter import GenreGraphAdapter, load_graph_adapter

FIXTURE = {
    "taxonomy_version": "granularity-test-0.1",
    "records": [
        {"name": "rock", "kind": "family", "status": "active", "specificity_score": 0.05},
        {"name": "indie rock", "kind": "genre", "status": "active", "specificity_score": 0.55},
        {"name": "shoegaze", "kind": "genre", "status": "active", "specificity_score": 0.86},
        {"name": "dream pop", "kind": "genre", "status": "active", "specificity_score": 0.78},
        # Same specificity as dream pop — tie-order test.
        {"name": "noise pop", "kind": "genre", "status": "active", "specificity_score": 0.78},
        # Review-status node: must NOT appear in display output.
        {"name": "chillwave", "kind": "genre", "status": "review", "specificity_score": 0.70},
        {"name": "nu gaze", "kind": "alias", "status": "alias_only", "canonical_target": "shoegaze"},
        {"name": "lo-fi", "kind": "facet", "facet_type": "production", "status": "active"},
        {"name": "seen live", "kind": "reject", "status": "rejected", "reject_reason": "source_noise"},
    ],
}


@pytest.fixture()
def adapter(tmp_path: Path) -> GenreGraphAdapter:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(FIXTURE, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_graph_adapter(path)


def test_orders_most_specific_first(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(["rock", "shoegaze", "indie rock"], adapter=adapter)
    assert result == ["shoegaze", "indie rock", "rock"]


def test_drops_noise_facets_unknown_and_review(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(
        ["seen live", "lo-fi", "no-such-genre", "chillwave", "shoegaze"], adapter=adapter
    )
    assert result == ["shoegaze"]


def test_alias_resolves_and_dedups_with_canonical(adapter):
    from src.genre.granularity import order_genres_by_granularity

    # "nu gaze" is an alias of shoegaze; the canonical name appears once.
    result = order_genres_by_granularity(["nu gaze", "shoegaze", "rock"], adapter=adapter)
    assert result == ["shoegaze", "rock"]


def test_variant_spellings_dedup(adapter):
    from src.genre.granularity import order_genres_by_granularity

    result = order_genres_by_granularity(["Dream-Pop", "dream pop"], adapter=adapter)
    assert result == ["dream pop"]


def test_ties_preserve_input_order(adapter):
    from src.genre.granularity import order_genres_by_granularity

    # dream pop and noise pop share specificity 0.78 — input order wins.
    assert order_genres_by_granularity(["noise pop", "dream pop"], adapter=adapter) == [
        "noise pop", "dream pop",
    ]
    assert order_genres_by_granularity(["dream pop", "noise pop"], adapter=adapter) == [
        "dream pop", "noise pop",
    ]


def test_empty_and_blank_input(adapter):
    from src.genre.granularity import order_genres_by_granularity

    assert order_genres_by_granularity([], adapter=adapter) == []
    assert order_genres_by_granularity(["", "  "], adapter=adapter) == []


def test_nothing_canonicalizes_returns_empty(adapter):
    from src.genre.granularity import order_genres_by_granularity

    assert order_genres_by_granularity(["seen live", "no-such-genre"], adapter=adapter) == []


def test_display_fallback_shows_raw_when_nothing_canonicalizes(adapter):
    from src.genre.granularity import order_genres_for_display

    raw = ["seen live", "no-such-genre"]
    assert order_genres_for_display(raw, adapter=adapter) == raw


def test_display_orders_when_canonicalization_succeeds(adapter):
    from src.genre.granularity import order_genres_for_display

    assert order_genres_for_display(["rock", "shoegaze"], adapter=adapter) == ["shoegaze", "rock"]


def test_adapter_load_failure_degrades_to_raw(monkeypatch):
    """Taxonomy unavailable -> raw tags pass through unchanged, no raise."""
    import src.genre.granularity as granularity

    def _boom():
        raise FileNotFoundError("taxonomy missing")

    monkeypatch.setattr(granularity, "load_graph_adapter", _boom)
    monkeypatch.setattr(granularity, "_ADAPTER_WARNED", False)
    raw = ["shoegaze", "rock"]
    assert granularity.order_genres_by_granularity(raw) == raw
    assert granularity.order_genres_for_display(raw) == raw
