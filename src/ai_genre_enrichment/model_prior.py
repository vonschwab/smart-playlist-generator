"""No-web album-level model-prior contract and taxonomy mapping."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .tag_classification import normalize_source_tag

MODEL_PRIOR_PROMPT_VERSION = "album-model-prior-v1"
MODEL_PRIOR_SCHEMA_VERSION = "album-model-prior-response-v1"
MODEL_PRIOR_TAXONOMY_VERSION = "genre-vocabulary-v1"
MODEL_PRIOR_INSTRUCTIONS = (
    "Classify the album into a compact multi-genre signature using only the supplied payload. "
    "Do not use web search. Do not claim that any external source says something. "
    "Return taxonomic hypotheses, not authoritative evidence."
)
SPECIFICITIES = {"broad", "genre", "subgenre", "microgenre"}
TAXONOMY_ROLES = {"parent", "core_style", "secondary_style", "edge_case"}
SOURCE_CLAIM_MARKERS = (
    "bandcamp says",
    "discogs says",
    "musicbrainz says",
    "last.fm says",
    "official source",
)

MODEL_PRIOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "genres": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "confidence": {"type": "number"},
                    "specificity": {"type": "string", "enum": sorted(SPECIFICITIES)},
                    "taxonomy_role": {"type": "string", "enum": sorted(TAXONOMY_ROLES)},
                    "notes": {"type": "string"},
                },
                "required": ["term", "confidence", "specificity", "taxonomy_role", "notes"],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["genres", "warnings"],
    "additionalProperties": False,
}


def model_prior_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ai_genre_album_model_prior",
        "strict": True,
        "schema": deepcopy(MODEL_PRIOR_RESPONSE_SCHEMA),
    }


def validate_model_prior_response(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"genres", "warnings"}:
        raise ValueError("model prior response must contain genres and warnings")
    if not isinstance(data["genres"], list) or not isinstance(data["warnings"], list):
        raise ValueError("model prior genres and warnings must be lists")
    normalized: list[dict[str, Any]] = []
    for item in data["genres"]:
        term = normalize_source_tag(str(item.get("term", "")))
        confidence = item.get("confidence")
        specificity = item.get("specificity")
        role = item.get("taxonomy_role")
        notes = str(item.get("notes", "")).strip()
        if not term or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("model prior term and confidence are invalid")
        if specificity not in SPECIFICITIES or role not in TAXONOMY_ROLES:
            raise ValueError("model prior specificity or taxonomy role is invalid")
        if any(marker in notes.casefold() for marker in SOURCE_CLAIM_MARKERS):
            raise ValueError("model prior notes must not claim source authority")
        normalized.append({
            "term": term,
            "confidence": float(confidence),
            "specificity": specificity,
            "taxonomy_role": role,
            "notes": notes,
        })
    return {"genres": normalized, "warnings": [str(v) for v in data["warnings"]]}


def map_model_prior_terms(items: list[dict[str, Any]], vocabulary: Any) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in items:
        term = normalize_source_tag(item["term"])
        non_genre = vocabulary.classify_non_genre(term)
        genre = vocabulary.classify_genre(term)
        conditional = item["taxonomy_role"] == "edge_case" or item["confidence"] < 0.70
        if non_genre:
            status, slug, accepted = non_genre, None, 0
        elif genre and conditional:
            status, slug, accepted = "conditional", genre.genre, 0
        elif genre:
            status, slug, accepted = "mapped", genre.genre, 1
        else:
            status, slug, accepted = "unmapped", None, 0
        mapped.append({
            **item,
            "raw_term": item["term"],
            "normalized_term": term,
            "canonical_slug": slug,
            "mapping_status": status,
            "accepted_for_shadow": accepted,
            "auto_apply_eligible": 0,
        })
    return mapped


def stable_input_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_model_prior_payload(release: Any) -> dict[str, Any]:
    baseline = release.existing_genres_by_source
    return {
        "release_key": release.release_key,
        "artist": release.normalized_artist,
        "album": release.normalized_album,
        "album_id": release.album_id,
        "identifiers": getattr(release, "identifiers", {}) or {},
        "year": getattr(release, "year", None),
        "track_titles": list(release.track_titles[:8]),
        "baseline_genres_by_source": baseline,
        "known_tags": sorted({tag for tags in baseline.values() for tag in tags}),
        "prompt_version": MODEL_PRIOR_PROMPT_VERSION,
        "taxonomy_version": MODEL_PRIOR_TAXONOMY_VERSION,
        "schema_version": MODEL_PRIOR_SCHEMA_VERSION,
    }


def build_model_prior_prompt(payload: dict[str, Any]) -> str:
    return (
        "Return a compact album genre prior for this local payload:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
