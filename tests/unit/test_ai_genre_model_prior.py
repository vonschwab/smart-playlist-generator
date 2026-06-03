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


def test_store_records_and_reuses_model_prior_cache(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    prior_id = store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster",
        normalized_album="stratosphere", album_id="a1", provider="openai",
        model="gpt-4o-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="hash-1",
        status="complete", response_json={"genres": [], "warnings": []},
        warnings=[], error_message=None, token_usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        estimated_cost_usd=0.00001, mapped_terms=[],
    )

    cached = store.find_model_prior(
        release_key="duster::stratosphere", provider="openai", model="gpt-4o-mini",
        prompt_version="album-model-prior-v1", taxonomy_version="genre-vocabulary-v1",
        schema_version="album-model-prior-response-v1", enrichment_policy_version="genre-enrichment-v2",
        input_hash="hash-1",
    )

    assert prior_id > 0
    assert cached["status"] == "complete"


def test_model_prior_one_dry_run_is_api_free_and_sidecar_free(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )
    sidecar = tmp_path / "sidecar.db"
    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db), "--sidecar-db", str(sidecar),
        "model-prior-one", "--artist", "Duster", "--album", "Stratosphere", "--dry-run",
    ])

    assert rc == 0
    assert not sidecar.exists()
    assert '"dry_run": true' in capsys.readouterr().out


def test_model_prior_report_counts_mapping_statuses(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster", normalized_album="stratosphere",
        album_id="a1", provider="openai", model="gpt-4o-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="h", status="complete",
        response_json={"genres": [], "warnings": []}, warnings=[], error_message=None, token_usage={},
        estimated_cost_usd=None, mapped_terms=[
            {"raw_term": "slowcore", "normalized_term": "slowcore", "canonical_slug": "slowcore",
             "confidence": 0.9, "specificity": "subgenre", "taxonomy_role": "core_style",
             "mapping_status": "mapped", "accepted_for_shadow": 1, "auto_apply_eligible": 0, "notes": ""},
        ],
    )

    report = store.model_prior_report()
    assert report["mapping_status_counts"] == {"mapped": 1}
    assert report["accepted_for_shadow"] == 1


def test_model_prior_missing_only_skips_before_api_call(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    release = ai_genre_enrich._discover(type("Args", (), {
        "metadata_db": metadata_db,
        "limit": 1,
        "artist": "Duster",
        "album": "Stratosphere",
        "generic_only": False,
        "min_existing_specific_genres": None,
    })())[0]

    payload = ai_genre_enrich.build_model_prior_payload(release)
    store.record_model_prior(
        release_key=release.release_key,
        normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album,
        album_id=release.album_id,
        provider="openai",
        model="gpt-4o-mini",
        prompt_version=ai_genre_enrich.MODEL_PRIOR_PROMPT_VERSION,
        taxonomy_version=ai_genre_enrich.MODEL_PRIOR_TAXONOMY_VERSION,
        schema_version=ai_genre_enrich.MODEL_PRIOR_SCHEMA_VERSION,
        enrichment_policy_version=ai_genre_enrich.STABILIZED_POLICY_VERSION,
        input_hash=ai_genre_enrich.stable_input_hash(payload),
        status="complete",
        response_json={"genres": [], "warnings": []},
        warnings=[],
        error_message=None,
        token_usage={},
        estimated_cost_usd=None,
        mapped_terms=[],
    )

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called before cache check")),
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "model-prior",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--missing-only",
    ])

    assert rc == 0
    assert "existing-model-prior" in capsys.readouterr().out


def test_model_prior_subcommands_accept_model_after_command():
    from scripts.ai_genre_enrich import build_parser

    args = build_parser().parse_args([
        "model-prior-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--model", "gpt-4.1-mini",
    ])

    assert args.model == "gpt-4.1-mini"
