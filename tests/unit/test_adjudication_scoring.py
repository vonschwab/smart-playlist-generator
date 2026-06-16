"""Tests for the Phase-2 adjudication scorer (metrics.md protocol)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ai_genre_enrichment.adjudication_scoring import (
    distribution,
    match_keys,
    preservation,
    set_metrics,
)
from src.ai_genre_enrichment.tag_classification import normalize_source_tag


def test_set_metrics_precision_recall_noise():
    m = set_metrics({"a", "b", "c"}, {"a", "b", "d"})
    assert m["n_correct"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["noise_rate"] == pytest.approx(1 / 3)  # 'c' not in gold
    assert m["f1"] == pytest.approx(2 / 3)


def test_set_metrics_empty_proposed_is_zero_not_error():
    m = set_metrics(set(), {"a"})
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["noise_rate"] == 0.0
    assert m["f1"] == 0.0


def test_preservation_is_fraction_of_must_preserve_kept():
    assert preservation({"a", "b"}, {"a", "b"}) == 1.0
    assert preservation({"a"}, {"a", "b"}) == 0.5
    assert preservation({"x"}, set()) == 1.0  # nothing required -> perfect


def test_match_keys_canonicalizes_and_falls_back_to_normalized():
    table = {
        "soul-jazz": ("canonical", "soul jazz"),
        "soul jazz": ("canonical", "soul jazz"),
        "ethio-jazz": ("unknown", None),
    }

    def fake(term: str):
        res, canon = table.get(term, ("unknown", None))
        return SimpleNamespace(resolution=res, canonical=canon)

    keys = match_keys(["soul-jazz", "ethio-jazz", "soul jazz"], fake)
    # canonical-equivalent terms collapse; gaps fall back to a normalized key
    assert keys == {"soul jazz", normalize_source_tag("ethio-jazz")}


def test_distribution_reports_min_p10_p50_p90():
    d = distribution([0.0, 0.5, 1.0, 0.5, 0.5])
    assert d["min"] == 0.0
    assert d["p50"] == 0.5
    assert d["max"] == 1.0
    assert "p10" in d and "p90" in d
