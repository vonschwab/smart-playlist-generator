from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

RESPONSE_SCHEMA_VERSION = "ai-genre-response-v2"
EVIDENCE_QUALITIES = {"high", "medium", "low", "unknown"}
WEB_SEARCH_QUALITIES = {"none", "weak", "adequate", "strong"}
IDENTITY_STATUSES = {"confirmed", "probable", "ambiguous", "unknown"}
SOURCE_TYPES = {
    "official_release",
    "official_artist",
    "official_label",
    "bandcamp_release",
    "label_catalog",
    "press_release",
    "liner_notes",
    "official_distributor",
    "local_metadata",
    "local_payload",
    "review_context",
    "model_knowledge",
    "lastfm_tags",
}
SOURCE_RELIABILITIES = {"high", "medium", "low"}
RECOMMENDATION_BASES = {"authoritative_source", "hybrid", "local_metadata", "model_knowledge", "review_context"}
AUTHORITATIVE_SOURCE_TYPES = {
    "official_release",
    "official_artist",
    "official_label",
    "bandcamp_release",
    "label_catalog",
    "press_release",
    "liner_notes",
    "official_distributor",
}
EXCLUDED_AUTHORITY_DOMAINS = {
    "allaboutjazz.com",
    "allmusic.com",
    "amoeba.com",
    "audiomack.com",
    "deezer.com",
    "discogs.com",
    "last.fm",
    "musicbrainz.org",
    "pitchfork.com",
    "qobuz.com",
    "soundcloud.com",
    "spotify.com",
    "tidal.com",
    "wikidata.org",
    "wikipedia.org",
}
BROAD_PARENT_GENRES = {
    "alternative rock",
    "electronic",
    "experimental",
    "folk",
    "hip hop",
    "indie rock",
    "jazz",
    "pop",
    "rock",
}
NON_GENRE_TAGS = {
    "oakland",
    "saxophone",
    "meditation",
    "improvisation",
}
BASELINE_SOURCE_HINTS = {"discogs", "musicbrainz", "last.fm", "lastfm"}
PRUNE_BROAD_PARENT_REASON_HINTS = (
    "too broad",
    "more specific",
    "overlaps with",
    "overlap with",
    "is more specific",
)
DESCRIPTOR_OR_GENRE = {"genre", "descriptor"}
PRUNE_TYPES = {"incorrect", "too_broad", "descriptor", "duplicate", "source_error", "malformed", "noise", "other"}


def _source_index_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "integer", "minimum": 0}}


def _recommendation_common_schema(name_key: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    props: dict[str, Any] = {
        name_key: {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "recommendation_basis": {"type": "string", "enum": sorted(RECOMMENDATION_BASES)},
        "supporting_source_indexes": _source_index_schema(),
    }
    required = [name_key, "confidence", "reason", "recommendation_basis", "supporting_source_indexes"]
    if extra:
        props.update(extra)
        required.extend(extra)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


AI_GENRE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "release_identity": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": sorted(IDENTITY_STATUSES)},
                "canonical_artist": {"type": "string"},
                "canonical_album": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["status", "canonical_artist", "canonical_album", "notes"],
            "additionalProperties": False,
        },
        "source_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string"},
                    "source_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "source_type": {"type": "string", "enum": sorted(SOURCE_TYPES)},
                    "reliability": {"type": "string", "enum": sorted(SOURCE_RELIABILITIES)},
                    "release_specific": {"type": "boolean"},
                    "extracted_genres_or_styles": {"type": "array", "items": {"type": "string"}},
                    "evidence_summary": {"type": "string"},
                },
                "required": [
                    "source_name",
                    "source_url",
                    "source_type",
                    "reliability",
                    "release_specific",
                    "extracted_genres_or_styles",
                    "evidence_summary",
                ],
                "additionalProperties": False,
            },
        },
        "source_conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "conflict": {"type": "string"},
                    "resolution": {"type": "string"},
                },
                "required": ["conflict", "resolution"],
                "additionalProperties": False,
            },
        },
        "release_level_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_quality": {"type": "string", "enum": sorted(EVIDENCE_QUALITIES)},
        "web_search_used": {"type": "boolean"},
        "web_search_quality": {"type": "string", "enum": sorted(WEB_SEARCH_QUALITIES)},
        "model_knowledge_used": {"type": "boolean"},
        "existing_genres_to_keep": {"type": "array", "items": _recommendation_common_schema("genre")},
        "existing_genres_to_prune": {
            "type": "array",
            "items": _recommendation_common_schema(
                "genre",
                {
                    "prune_type": {"type": "string", "enum": sorted(PRUNE_TYPES)},
                    "descriptor_or_genre": {"type": "string", "enum": sorted(DESCRIPTOR_OR_GENRE)},
                },
            ),
        },
        "new_genres_to_add": {
            "type": "array",
            "items": _recommendation_common_schema(
                "genre",
                {
                    "auto_apply_eligible": {"type": "boolean"},
                    "descriptor_or_genre": {"type": "string", "enum": sorted(DESCRIPTOR_OR_GENRE)},
                },
            ),
        },
        "descriptor_tags": {
            "type": "array",
            "items": _recommendation_common_schema(
                "tag",
                {"descriptor_or_genre": {"type": "string", "enum": sorted(DESCRIPTOR_OR_GENRE)}},
            ),
        },
        "review_only_suggestions": {
            "type": "array",
            "items": _recommendation_common_schema(
                "tag",
                {"descriptor_or_genre": {"type": "string", "enum": sorted(DESCRIPTOR_OR_GENRE)}},
            ),
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        "should_escalate": {"type": "boolean"},
    },
    "required": [
        "release_identity",
        "source_evidence",
        "source_conflicts",
        "release_level_confidence",
        "evidence_quality",
        "web_search_used",
        "web_search_quality",
        "model_knowledge_used",
        "existing_genres_to_keep",
        "existing_genres_to_prune",
        "new_genres_to_add",
        "descriptor_tags",
        "review_only_suggestions",
        "warnings",
        "uncertainty_notes",
        "should_escalate",
    ],
    "additionalProperties": False,
}


def response_format_schema() -> dict[str, Any]:
    """Return the Responses API strict structured-output format."""
    return {
        "type": "json_schema",
        "name": "ai_genre_release_recommendation",
        "description": "Authoritative-source genre refinement recommendations for one canonical artist and album release.",
        "strict": True,
        "schema": deepcopy(AI_GENRE_RESPONSE_SCHEMA),
    }


def validate_ai_response(data: dict[str, Any]) -> dict[str, Any]:
    """Small dependency-free validator for the response contract used by tests and CLI."""
    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object")
    data = _normalize_source_evidence_domains(data)
    data = _infer_keep_source_indexes(data)
    data = _normalize_baseline_keep_recommendations(data)
    data = _normalize_source_based_recommendation_indexes(data)
    data = _normalize_prunes_against_local_payload(data)
    data = _demote_non_genre_additions(data)
    missing = [key for key in AI_GENRE_RESPONSE_SCHEMA["required"] if key not in data]
    if missing:
        raise ValueError(f"AI response missing required keys: {', '.join(missing)}")

    identity = data["release_identity"]
    if not isinstance(identity, dict) or identity.get("status") not in IDENTITY_STATUSES:
        raise ValueError("release_identity.status is invalid")

    source_evidence = data["source_evidence"]
    if not isinstance(source_evidence, list):
        raise ValueError("source_evidence must be a list")
    for source in source_evidence:
        _validate_source_evidence(source)

    _validate_confidence(data["release_level_confidence"], "release_level_confidence")
    if data["evidence_quality"] not in EVIDENCE_QUALITIES:
        raise ValueError("evidence_quality must be high, medium, low, or unknown")
    if data["web_search_quality"] not in WEB_SEARCH_QUALITIES:
        raise ValueError("web_search_quality is invalid")
    if not isinstance(data["web_search_used"], bool) or not isinstance(data["model_knowledge_used"], bool):
        raise ValueError("web_search_used and model_knowledge_used must be booleans")
    if not isinstance(data["should_escalate"], bool):
        raise ValueError("should_escalate must be a boolean")

    for key in ["source_conflicts", "warnings", "uncertainty_notes"]:
        if not isinstance(data[key], list):
            raise ValueError(f"{key} must be a list")

    max_index = len(source_evidence) - 1
    for item in data["existing_genres_to_keep"]:
        _validate_recommendation_item(item, "existing_genres_to_keep", "genre", max_index, source_evidence)
    for item in data["existing_genres_to_prune"]:
        _validate_recommendation_item(item, "existing_genres_to_prune", "genre", max_index, source_evidence)
        if item["prune_type"] not in PRUNE_TYPES:
            raise ValueError("existing_genres_to_prune prune_type is invalid")
        if item["descriptor_or_genre"] not in DESCRIPTOR_OR_GENRE:
            raise ValueError("existing_genres_to_prune descriptor_or_genre is invalid")
        if _is_broad_parent_prune(item):
            raise ValueError("existing_genres_to_prune cannot prune broad parent genres merely because a child is specific")
    for item in data["new_genres_to_add"]:
        _validate_recommendation_item(item, "new_genres_to_add", "genre", max_index, source_evidence)
        if not isinstance(item["auto_apply_eligible"], bool):
            raise ValueError("new_genres_to_add auto_apply_eligible must be boolean")
        if item["descriptor_or_genre"] != "genre":
            raise ValueError("new_genres_to_add items must be real genres/styles")
    for item in data["descriptor_tags"]:
        _validate_recommendation_item(item, "descriptor_tags", "tag", max_index, source_evidence)
        if item["descriptor_or_genre"] != "descriptor":
            raise ValueError("descriptor_tags items must be descriptors")
    for item in data["review_only_suggestions"]:
        _validate_recommendation_item(item, "review_only_suggestions", "tag", max_index, source_evidence)
    return data


def _validate_source_evidence(source: Any) -> None:
    if not isinstance(source, dict):
        raise ValueError("source_evidence items must be objects")
    required = {
        "source_name",
        "source_url",
        "source_type",
        "reliability",
        "release_specific",
        "extracted_genres_or_styles",
        "evidence_summary",
    }
    if not required <= source.keys():
        raise ValueError("source_evidence item missing required keys")
    if source["source_type"] not in SOURCE_TYPES:
        raise ValueError("source_evidence source_type is invalid")
    if source["reliability"] not in SOURCE_RELIABILITIES:
        raise ValueError("source_evidence reliability is invalid")
    if not isinstance(source["release_specific"], bool):
        raise ValueError("source_evidence release_specific must be boolean")
    if not isinstance(source["extracted_genres_or_styles"], list):
        raise ValueError("source_evidence extracted_genres_or_styles must be a list")


def _validate_recommendation_item(
    item: Any,
    field_name: str,
    name_key: str,
    max_source_index: int,
    source_evidence: list[dict[str, Any]] | None = None,
) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{field_name} items must be objects")
    required = {name_key, "confidence", "reason", "recommendation_basis", "supporting_source_indexes"}
    if not required <= item.keys():
        raise ValueError(f"{field_name} item missing required keys")
    if not isinstance(item[name_key], str) or not item[name_key].strip():
        raise ValueError(f"{field_name} {name_key} must be a non-empty string")
    _validate_confidence(item["confidence"], f"{field_name} confidence")
    if item["recommendation_basis"] not in RECOMMENDATION_BASES:
        raise ValueError(f"{field_name} recommendation_basis is invalid")
    indexes = item["supporting_source_indexes"]
    if not isinstance(indexes, list) or not all(isinstance(index, int) for index in indexes):
        raise ValueError(f"{field_name} supporting_source_indexes must be integer indexes")
    if any(index < 0 or index > max_source_index for index in indexes):
        raise ValueError(f"{field_name} supporting_source_indexes contains an invalid source index")
    if item["recommendation_basis"] in {"authoritative_source", "hybrid", "review_context"} and not indexes:
        raise ValueError(f"{field_name} source-based recommendations need supporting source indexes")
    reason = str(item.get("reason") or "")
    if item["recommendation_basis"] == "authoritative_source" and _mentions_baseline_source(reason):
        raise ValueError(f"{field_name} baseline source names cannot justify authoritative_source")
    if item["recommendation_basis"] == "authoritative_source" and source_evidence is not None:
        if any(source_evidence[index].get("source_type") == "review_context" for index in indexes):
            raise ValueError(f"{field_name} authoritative recommendations cannot cite review_context sources")
        has_authoritative_source = any(
            0 <= index <= max_source_index and source_evidence[index].get("source_type") in AUTHORITATIVE_SOURCE_TYPES
            for index in indexes
        )
        all_authoritative_sources = all(
            0 <= index <= max_source_index and source_evidence[index].get("source_type") in AUTHORITATIVE_SOURCE_TYPES
            for index in indexes
        )
        if not has_authoritative_source or not all_authoritative_sources:
            raise ValueError(f"{field_name} authoritative recommendations need authoritative source indexes")
    if item["recommendation_basis"] == "hybrid" and source_evidence is not None:
        if any(source_evidence[index].get("source_type") == "review_context" for index in indexes):
            raise ValueError(f"{field_name} hybrid recommendations cannot cite review_context sources")
        has_authoritative_source = any(
            0 <= index <= max_source_index and source_evidence[index].get("source_type") in AUTHORITATIVE_SOURCE_TYPES
            for index in indexes
        )
        if not has_authoritative_source:
            raise ValueError(f"{field_name} authoritative recommendations need authoritative source indexes")
    if not isinstance(item["reason"], str):
        raise ValueError(f"{field_name} reason must be a string")


def _validate_confidence(value: Any, name: str) -> None:
    if not isinstance(value, int | float) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")


def _normalize_source_evidence_domains(data: dict[str, Any]) -> dict[str, Any]:
    source_evidence = data.get("source_evidence")
    if not isinstance(source_evidence, list):
        return data
    for source in source_evidence:
        if not isinstance(source, dict):
            continue
        domain = _source_domain(source.get("source_url"))
        if not domain:
            if source.get("source_type") == "review_context":
                source["reliability"] = "low"
                source["release_specific"] = False
            continue
        if domain == "bandcamp.com" or domain.endswith(".bandcamp.com"):
            source["source_type"] = "bandcamp_release"
        elif domain in EXCLUDED_AUTHORITY_DOMAINS or any(domain.endswith(f".{root}") for root in EXCLUDED_AUTHORITY_DOMAINS):
            source["source_type"] = "review_context"
        if source.get("source_type") == "review_context":
            source["reliability"] = "low"
            source["release_specific"] = False
    return data


def _normalize_baseline_keep_recommendations(data: dict[str, Any]) -> dict[str, Any]:
    source_evidence = data.get("source_evidence")
    keep_items = data.get("existing_genres_to_keep")
    if not isinstance(source_evidence, list) or not isinstance(keep_items, list):
        return data
    local_indexes = [
        index for index, source in enumerate(source_evidence)
        if isinstance(source, dict) and source.get("source_type") == "local_payload"
    ]
    if not local_indexes:
        return data
    for item in keep_items:
        if not isinstance(item, dict):
            continue
        if item.get("recommendation_basis") == "authoritative_source" and _mentions_baseline_source(
            str(item.get("reason") or "")
        ):
            item["recommendation_basis"] = "local_metadata"
            item["supporting_source_indexes"] = local_indexes
    return data


def _normalize_source_based_recommendation_indexes(data: dict[str, Any]) -> dict[str, Any]:
    source_evidence = data.get("source_evidence")
    if not isinstance(source_evidence, list):
        return data
    for field in [
        "existing_genres_to_keep",
        "existing_genres_to_prune",
        "new_genres_to_add",
        "descriptor_tags",
        "review_only_suggestions",
    ]:
        items = data.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            _normalize_source_based_recommendation_item(item, source_evidence)
    return data


def _normalize_source_based_recommendation_item(item: Any, source_evidence: list[dict[str, Any]]) -> None:
    if not isinstance(item, dict):
        return
    basis = item.get("recommendation_basis")
    if basis not in {"authoritative_source", "hybrid"}:
        return
    indexes = item.get("supporting_source_indexes")
    if not isinstance(indexes, list) or not all(isinstance(index, int) for index in indexes):
        return
    valid_indexes = [index for index in indexes if 0 <= index < len(source_evidence)]
    if not valid_indexes:
        return
    if basis == "authoritative_source" and _mentions_baseline_source(str(item.get("reason") or "")):
        return
    authoritative_indexes = [
        index for index in valid_indexes
        if source_evidence[index].get("source_type") in AUTHORITATIVE_SOURCE_TYPES
    ]
    item_name = _recommendation_item_name(item)
    matching_authoritative_indexes = [
        index for index in authoritative_indexes
        if item_name and _source_supports_genre(source_evidence[index], item_name, authoritative_only=True)
    ]
    if matching_authoritative_indexes:
        item["supporting_source_indexes"] = matching_authoritative_indexes
        return
    if authoritative_indexes and not item_name:
        item["supporting_source_indexes"] = authoritative_indexes
        return
    review_context_indexes = [
        index for index in valid_indexes
        if source_evidence[index].get("source_type") == "review_context"
    ]
    if review_context_indexes:
        item["recommendation_basis"] = "review_context"
        item["supporting_source_indexes"] = review_context_indexes
        if "auto_apply_eligible" in item:
            item["auto_apply_eligible"] = False
        return
    matching_local_indexes = _source_indexes_supporting_genre(
        source_evidence,
        item_name,
        source_types={"local_payload"},
    )
    if matching_local_indexes:
        item["recommendation_basis"] = "local_metadata"
        item["supporting_source_indexes"] = matching_local_indexes
        if "auto_apply_eligible" in item:
            item["auto_apply_eligible"] = False
        return
    if item_name:
        item["recommendation_basis"] = "model_knowledge"
        item["supporting_source_indexes"] = []
        if "auto_apply_eligible" in item:
            item["auto_apply_eligible"] = False


def _recommendation_item_name(item: dict[str, Any]) -> str:
    return str(item.get("genre") or item.get("tag") or "").casefold()


def _normalize_prunes_against_local_payload(data: dict[str, Any]) -> dict[str, Any]:
    source_evidence = data.get("source_evidence")
    prunes = data.get("existing_genres_to_prune")
    if not isinstance(source_evidence, list) or not isinstance(prunes, list):
        return data
    local_genres = _source_genres_for_types(source_evidence, {"local_payload"})
    if not local_genres:
        return data

    additions = data.setdefault("new_genres_to_add", [])
    descriptor_tags = data.setdefault("descriptor_tags", [])
    existing_additions = {
        str(item.get("genre") or "").casefold()
        for item in additions
        if isinstance(item, dict)
    }
    existing_descriptors = {
        str(item.get("tag") or "").casefold()
        for item in descriptor_tags
        if isinstance(item, dict)
    }

    retained = []
    for item in prunes:
        if not isinstance(item, dict):
            retained.append(item)
            continue
        genre_key = str(item.get("genre") or "").casefold()
        if not genre_key or genre_key in local_genres:
            retained.append(item)
            continue

        authoritative_indexes = _source_indexes_supporting_genre(
            source_evidence,
            genre_key,
            source_types=AUTHORITATIVE_SOURCE_TYPES,
        )
        if not authoritative_indexes:
            continue
        if _is_non_genre_tag(genre_key) or item.get("descriptor_or_genre") == "descriptor":
            if genre_key not in existing_descriptors:
                descriptor_tags.append(
                    {
                        "tag": item.get("genre"),
                        "confidence": item.get("confidence", 0.0),
                        "reason": item.get("reason") or "Source tag is a descriptor, not an existing genre to prune.",
                        "recommendation_basis": item.get("recommendation_basis", "hybrid"),
                        "supporting_source_indexes": authoritative_indexes,
                        "descriptor_or_genre": "descriptor",
                    }
                )
                existing_descriptors.add(genre_key)
            continue
        if genre_key not in existing_additions:
            additions.append(
                {
                    "genre": item.get("genre"),
                    "confidence": item.get("confidence", 0.0),
                    "reason": "Release-specific source tag; not an existing local genre to prune.",
                    "recommendation_basis": item.get("recommendation_basis", "hybrid"),
                    "supporting_source_indexes": authoritative_indexes,
                    "auto_apply_eligible": False,
                    "descriptor_or_genre": "genre",
                }
            )
            existing_additions.add(genre_key)
    data["existing_genres_to_prune"] = retained
    return data


def _source_genres_for_types(source_evidence: list[Any], source_types: set[str]) -> set[str]:
    genres: set[str] = set()
    for source in source_evidence:
        if not isinstance(source, dict) or source.get("source_type") not in source_types:
            continue
        tags = source.get("extracted_genres_or_styles")
        if isinstance(tags, list):
            genres.update(str(tag).casefold() for tag in tags)
    return genres


def _source_indexes_supporting_genre(
    source_evidence: list[Any],
    genre_key: str,
    *,
    source_types: set[str],
) -> list[int]:
    return [
        index for index, source in enumerate(source_evidence)
        if (
            isinstance(source, dict)
            and source.get("source_type") in source_types
            and _source_supports_genre(source, genre_key, authoritative_only=False)
        )
    ]


def _demote_non_genre_additions(data: dict[str, Any]) -> dict[str, Any]:
    additions = data.get("new_genres_to_add")
    if not isinstance(additions, list):
        return data
    descriptor_tags = data.setdefault("descriptor_tags", [])
    review_only = data.setdefault("review_only_suggestions", [])
    retained = []
    for item in additions:
        if not isinstance(item, dict) or not _is_non_genre_tag(item.get("genre")):
            retained.append(item)
            continue
        demoted = {
            "tag": item.get("genre"),
            "confidence": item.get("confidence", 0.0),
            "reason": item.get("reason") or "Demoted from genre addition because this source tag is not a genre/style.",
            "recommendation_basis": item.get("recommendation_basis", "authoritative_source"),
            "supporting_source_indexes": item.get("supporting_source_indexes", []),
            "descriptor_or_genre": "descriptor",
        }
        if _is_descriptor_like_tag(item.get("genre")):
            descriptor_tags.append(demoted)
        else:
            demoted["descriptor_or_genre"] = "genre"
            review_only.append(demoted)
    data["new_genres_to_add"] = retained
    return data


def _infer_keep_source_indexes(data: dict[str, Any]) -> dict[str, Any]:
    source_evidence = data.get("source_evidence")
    keep_items = data.get("existing_genres_to_keep")
    if not isinstance(source_evidence, list) or not isinstance(keep_items, list):
        return data
    for item in keep_items:
        if not isinstance(item, dict) or item.get("supporting_source_indexes"):
            continue
        if item.get("recommendation_basis") not in {"authoritative_source", "hybrid"}:
            continue
        genre_key = str(item.get("genre") or "").casefold()
        inferred = [
            index for index, source in enumerate(source_evidence)
            if _source_supports_genre(source, genre_key, authoritative_only=True)
        ]
        if inferred:
            item["supporting_source_indexes"] = inferred
    return data


def _source_supports_genre(source: Any, genre_key: str, *, authoritative_only: bool) -> bool:
    if not isinstance(source, dict):
        return False
    if authoritative_only and source.get("source_type") not in AUTHORITATIVE_SOURCE_TYPES:
        return False
    tags = source.get("extracted_genres_or_styles")
    if not isinstance(tags, list):
        return False
    return any(str(tag).casefold() == genre_key for tag in tags)


def _source_domain(url: Any) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.") or None


def _mentions_baseline_source(reason: str) -> bool:
    lowered = reason.casefold()
    return any(hint in lowered for hint in BASELINE_SOURCE_HINTS)


def _is_broad_parent_prune(item: dict[str, Any]) -> bool:
    if item.get("prune_type") != "too_broad":
        return False
    genre = str(item.get("genre") or "").casefold()
    reason = str(item.get("reason") or "").casefold()
    return genre in BROAD_PARENT_GENRES and any(hint in reason for hint in PRUNE_BROAD_PARENT_REASON_HINTS)


def _is_non_genre_tag(value: Any) -> bool:
    return str(value or "").casefold() in NON_GENRE_TAGS


def _is_descriptor_like_tag(value: Any) -> bool:
    return str(value or "").casefold() in {"oakland", "saxophone", "meditation"}
