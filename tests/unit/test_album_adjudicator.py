"""Tests for the album-adjudicator-v1 contract (Phase 1)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ai_genre_enrichment.album_adjudicator import (
    ADJUDICATOR_PROMPT_VERSION,
    ADJUDICATOR_SCHEMA_VERSION,
    build_adjudicator_payload,
    canonicalize_proposed,
    validate_adjudicator_response,
)


def _wellformed() -> dict:
    return {
        "genres": [
            {"term": "Math Rock", "confidence": 0.9, "layer": "core", "rationale": "angular guitars"},
            {"term": "noise rock", "confidence": 0.6, "layer": "secondary", "rationale": "feedback"},
        ],
        "facets": [{"term": "instrumental", "facet_type": "instrumentation"}],
        "escalate": False,
        "escalate_reason": "",
        "overall_confidence": 0.85,
        "warnings": [],
    }


def test_validate_normalizes_a_well_formed_response():
    out = validate_adjudicator_response(_wellformed())
    assert [g["term"] for g in out["genres"]] == ["math rock", "noise rock"]
    assert out["genres"][0]["layer"] == "core"
    assert out["facets"][0]["term"] == "instrumental"
    assert out["facets"][0]["facet_type"] == "instrumentation"
    assert out["escalate"] is False
    assert out["overall_confidence"] == 0.85
    assert out["warnings"] == []


@pytest.mark.parametrize("mutate", [
    lambda d: d["genres"][0].update(confidence=1.5),     # out of [0,1]
    lambda d: d["genres"][0].update(confidence=-0.1),
    lambda d: d["genres"][0].update(layer="primary"),    # not in {core, secondary}
    lambda d: d["facets"][0].update(facet_type="bogus"), # not a taxonomy facet_type
    lambda d: d.pop("escalate"),                          # missing required key
    lambda d: d["genres"][0].update(term=""),            # empty term
], ids=["conf_hi", "conf_lo", "bad_layer", "bad_facet", "missing_key", "empty_term"])
def test_validate_rejects_malformed(mutate):
    d = _wellformed()
    mutate(d)
    with pytest.raises(ValueError):
        validate_adjudicator_response(d)


def test_build_payload_assembles_dedups_truncates_and_stamps_versions():
    payload = build_adjudicator_payload({
        "artist": "Marnie Stern",
        "album": "The Chronicles of Marnia",
        "album_id": "abc",
        "year": 2013,
        "identifiers": {"mbid": "x"},
        "track_titles": [f"t{i}" for i in range(12)],
        "existing_genres_by_source": {"file": ["math rock", "rock"], "lastfm": ["noise pop", "math rock"]},
        "current_observed_leaf": ["math rock", "experimental rock", "noise pop"],
    })
    assert payload["artist"] == "Marnie Stern"
    assert payload["year"] == 2013
    assert payload["identifiers"] == {"mbid": "x"}
    assert len(payload["track_titles"]) == 8  # truncated
    assert payload["known_tags"] == ["math rock", "noise pop", "rock"]  # union, deduped, sorted
    assert payload["current_observed_leaf"] == ["experimental rock", "math rock", "noise pop"]
    assert payload["prompt_version"] == ADJUDICATOR_PROMPT_VERSION
    assert payload["schema_version"] == ADJUDICATOR_SCHEMA_VERSION


def test_canonicalize_splits_canonical_alias_and_gaps_preserving_order():
    table = {
        "soul-jazz": ("canonical", "soul jazz"),
        "funk": ("canonical", "funk"),
        "jazz fusion": ("alias", "jazz fusion"),
        "ethio-jazz": ("unknown", None),
    }

    def fake_canonicalize(term: str):
        res, canon = table.get(term, ("unknown", None))
        return SimpleNamespace(resolution=res, canonical=canon)

    out = canonicalize_proposed(
        ["soul-jazz", "funk", "ethio-jazz", "soul-jazz", "jazz fusion"],
        fake_canonicalize,
    )
    # canonical names, deduped, first-seen order; aliases resolve to their canonical
    assert out["canonical"] == ["soul jazz", "funk", "jazz fusion"]
    assert out["gaps"] == ["ethio-jazz"]
