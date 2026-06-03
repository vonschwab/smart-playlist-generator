from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def test_request_structured_uses_supplied_validator(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    seen = []

    class Response:
        output_text = '{"genres":[],"warnings":[]}'
        usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", api_key="test", max_retries=0)
    monkeypatch.setattr(client, "_call_openai", lambda *_args, **_kwargs: Response())

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: seen.append(value) or value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "complete"
    assert result.response_json == {"genres": [], "warnings": []}
    assert seen == [{"genres": [], "warnings": []}]


def test_request_structured_dry_run_does_not_call_openai(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", dry_run=True)
    monkeypatch.setattr(
        client,
        "_call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "skipped"
    assert result.response_json["dry_run"] is True
    assert result.response_json["estimated_output_tokens"] == 300


def test_validate_model_prior_response_normalizes_terms():
    from src.ai_genre_enrichment.model_prior import validate_model_prior_response

    result = validate_model_prior_response({
        "genres": [{
            "term": "  Ambient Americana ",
            "confidence": 0.82,
            "specificity": "subgenre",
            "taxonomy_role": "core_style",
            "notes": "Taxonomic fit.",
        }],
        "warnings": [],
    })
    assert result["genres"][0]["term"] == "ambient americana"


def test_validate_model_prior_response_rejects_source_claims():
    from src.ai_genre_enrichment.model_prior import validate_model_prior_response

    with pytest.raises(ValueError, match="source authority"):
        validate_model_prior_response({
            "genres": [{
                "term": "slowcore", "confidence": 0.9, "specificity": "subgenre",
                "taxonomy_role": "core_style", "notes": "Bandcamp says this is slowcore.",
            }],
            "warnings": [],
        })


def test_map_model_prior_terms_accepts_known_style_and_rejects_descriptor():
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.model_prior import map_model_prior_terms

    mapped = map_model_prior_terms(
        [
            {"term": "slowcore", "confidence": 0.9, "specificity": "subgenre", "taxonomy_role": "core_style", "notes": ""},
            {"term": "instrumental", "confidence": 0.8, "specificity": "broad", "taxonomy_role": "secondary_style", "notes": ""},
        ],
        GenreVocabulary(),
    )

    assert mapped[0]["mapping_status"] == "mapped"
    assert mapped[0]["accepted_for_shadow"] == 1
    assert mapped[0]["auto_apply_eligible"] == 0
    assert mapped[1]["mapping_status"] == "descriptor"
    assert mapped[1]["accepted_for_shadow"] == 0
