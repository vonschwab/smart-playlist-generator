"""album-adjudicator-v1: Claude proposes a tight, facet-separated genre identity.

Evolves model_prior.py (keeps its evidence-richness confidence cap and anti-noise
guards) but inverts the old refinement contract's inclusive philosophy: the output is
the SPECIFIC observed-leaf set the record actually is — broad parents are derived by the
graph downstream, not stored here — with facets split out of genres. File tags are ground
truth (never silently dropped; escalate instead). Post-processing canonicalizes proposed
terms against the taxonomy (canonical / alias / gap), it is not the LLM's job.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from .tag_classification import normalize_source_tag

ADJUDICATOR_PROMPT_VERSION = "album-adjudicator-v1"
ADJUDICATOR_PROMPT_VERSION_THOROUGH = "album-adjudicator-v1-thorough"
ADJUDICATOR_SCHEMA_VERSION = "album-adjudicator-response-v1"

LAYERS = {"core", "secondary"}
# facet_type enum from data/layered_genre_taxonomy.yaml (taxonomy-growth skill)
FACET_TYPES = {
    "mood", "texture", "instrumentation", "production", "era",
    "region", "function", "vocal", "scene", "format", "rhythm",
}
_REQUIRED_KEYS = {"genres", "facets", "escalate", "overall_confidence", "warnings"}

ADJUDICATOR_INSTRUCTIONS = """You are adjudicating the genre identity of a single music release for a local library.

Return ONE JSON object only — no prose, no markdown.

Goal: state what this release ACTUALLY IS as a TIGHT, SPECIFIC set — the 3-6 genres a knowledgeable listener would name, not an exhaustive list. Broad parent genres (e.g. rock, pop, jazz, electronic, hip hop, folk, indie rock, alternative rock, experimental) are derived elsewhere from the genre graph; do NOT include them. Prefer the specific style: shoegaze not "rock"; ethio-jazz not "world music"; trip-hop not "downtempo".

Separate genres from facets. Mood, texture, instrumentation, production, era, region, function, vocal, scene, format, and rhythm descriptors (instrumental, lo-fi, acoustic, orchestral, 1970s, japanese, live, female vocals, drone) are FACETS — put them in `facets`, never in `genres`.

Each genre gets a `layer`: "core" (primary identity, keep to ~2-4) or "secondary" (a real but lesser element). Total genres ~3-6; fewer for focused releases.

The payload's `user_file_tags` are the USER'S OWN embedded file tags — ground truth. Every SPECIFIC `user_file_tags` genre (anything that is not a broad parent) MUST appear in your `genres`, OR you MUST set `escalate` true and name the omitted tag in `escalate_reason`. Silently dropping a specific user file tag is the single worst error — never do it. (You MAY drop a broad-parent file tag like "rock" without escalating, since broad parents are derived elsewhere.)

Source tags are often applied at the ARTIST level, identical across an artist's albums. Use your own music knowledge to give THIS release its specific identity, distinct from the artist's other work. But never infer genre from artist name, nationality, language, album-title aesthetics, or demographic cues alone.

Do not use web search. Do not claim any external source says anything.

`confidence` (0-1) per genre and `overall_confidence`: lower for sparse evidence. Set `escalate` true when the release identity is ambiguous, evidence is thin and you are guessing, or a correct file tag would be dropped. Do NOT output per-genre rationale (omit the field) and no chain-of-thought — keep output minimal. Use canonical genre names where known — spelling is normalized downstream.
"""

ADJUDICATOR_INSTRUCTIONS_THOROUGH = """You are adjudicating the genre identity of a single music release for a local library on a SECOND PASS.

Return ONE JSON object only — no prose, no markdown.

Goal: give this release a COMPLETE, SPECIFIC genre picture. A prior automated pass returned a minimal result (1-2 genres). Draw on your full knowledge of this specific album — its production style, era, critical reception, scene, and how it is discussed by listeners — to provide every genre it genuinely warrants. If you know this record well, do not leave out secondary or complementary genres out of excessive caution. Lean toward completeness: a release with a primary style almost always has 1-3 secondary styles worth naming. If the release is genuinely focused and minimal in identity, keep it tight — but only if that reflects reality, not uncertainty.

Same rules as always:
- Broad parent genres (rock, pop, jazz, electronic, hip hop, folk, indie rock, alternative rock, experimental) are derived elsewhere; do NOT include them. Prefer the specific: shoegaze not "rock"; ethio-jazz not "world music".
- Separate genres from facets. Mood, texture, instrumentation, production, era, region, function, vocal, scene, format, and rhythm descriptors (instrumental, lo-fi, acoustic, orchestral, 1970s, japanese, live, female vocals, drone) are FACETS — put them in `facets`, never in `genres`.
- Each genre gets a `layer`: "core" (primary identity, keep to ~2-4) or "secondary" (a real but lesser element). Total genres typically 3-6.
- The payload's `user_file_tags` are the USER'S OWN embedded file tags — ground truth. Every SPECIFIC `user_file_tags` genre (anything that is not a broad parent) MUST appear in your `genres`, OR you MUST set `escalate` true and name the omitted tag in `escalate_reason`. Silently dropping a specific user file tag is the single worst error.
- Use your own music knowledge for THIS specific release, not the artist in general. Never infer genre from name, nationality, language, or demographic cues alone.
- Do not use web search. Do not claim any external source says anything.
- `confidence` (0-1) per genre and `overall_confidence`: lower for sparse evidence. Set `escalate` true when genuinely uncertain, evidence is thin, or a correct file tag would be dropped.
- Do NOT output per-genre rationale (omit the field) and no chain-of-thought — keep output minimal.
"""

ADJUDICATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "genres": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "confidence": {"type": "number"},
                    "layer": {"type": "string", "enum": sorted(LAYERS)},
                    "rationale": {"type": "string"},
                },
                "required": ["term", "confidence", "layer"],
                "additionalProperties": False,
            },
        },
        "facets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "facet_type": {"type": "string", "enum": sorted(FACET_TYPES)},
                },
                "required": ["term", "facet_type"],
                "additionalProperties": False,
            },
        },
        "escalate": {"type": "boolean"},
        "escalate_reason": {"type": "string"},
        "overall_confidence": {"type": "number"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["genres", "facets", "escalate", "escalate_reason", "overall_confidence", "warnings"],
    "additionalProperties": False,
}


def adjudicator_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "album_adjudicator_v1",
        "strict": True,
        "schema": deepcopy(ADJUDICATOR_RESPONSE_SCHEMA),
    }


def build_adjudicator_prompt(payload: dict[str, Any]) -> str:
    return (
        "Adjudicate the genre identity of this release. Use the payload only.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    )


def _confidence(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} out of [0,1]: {value}")
    return float(value)


def validate_adjudicator_response(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("adjudicator response must be a JSON object")
    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"adjudicator response missing keys: {sorted(missing)}")
    for key in ("genres", "facets", "warnings"):
        if not isinstance(data[key], list):
            raise ValueError(f"{key} must be a list")

    genres: list[dict[str, Any]] = []
    for item in data["genres"]:
        term = normalize_source_tag(str(item.get("term", "")))
        if not term:
            raise ValueError("genre term must be non-empty")
        layer = item.get("layer")
        if layer not in LAYERS:
            raise ValueError(f"genre layer must be one of {sorted(LAYERS)}: {layer!r}")
        genres.append({
            "term": term,
            "confidence": _confidence(item.get("confidence"), "genre confidence"),
            "layer": layer,
            "rationale": str(item.get("rationale", "")).strip(),
        })

    facets: list[dict[str, Any]] = []
    for facet in data["facets"]:
        fterm = normalize_source_tag(str(facet.get("term", "")))
        if not fterm:
            raise ValueError("facet term must be non-empty")
        ftype = facet.get("facet_type")
        if ftype not in FACET_TYPES:
            raise ValueError(f"facet_type must be one of {sorted(FACET_TYPES)}: {ftype!r}")
        facets.append({"term": fterm, "facet_type": ftype})

    return {
        "genres": genres,
        "facets": facets,
        "escalate": bool(data["escalate"]),
        "escalate_reason": str(data.get("escalate_reason", "")).strip(),
        "overall_confidence": _confidence(data["overall_confidence"], "overall_confidence"),
        "warnings": [str(w) for w in data["warnings"]],
    }


def build_adjudicator_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    """Assemble the per-release payload the adjudicator reasons over.

    Dedups the raw tags across sources into `known_tags`, truncates track titles,
    sorts the current observed-leaf set (what the adjudicator is asked to tighten),
    and stamps the contract versions.
    """
    by_source = evidence.get("existing_genres_by_source") or {}
    known = sorted({
        normalize_source_tag(str(tag))
        for tags in by_source.values()
        for tag in tags
        if str(tag).strip()
    })
    observed = sorted({
        normalize_source_tag(str(genre))
        for genre in (evidence.get("current_observed_leaf") or [])
        if str(genre).strip()
    })
    file_tags = sorted({
        normalize_source_tag(str(tag))
        for tag in (evidence.get("file_tags") or [])
        if str(tag).strip()
    })
    return {
        "artist": evidence.get("artist"),
        "album": evidence.get("album"),
        "album_id": evidence.get("album_id"),
        "year": evidence.get("year"),
        "identifiers": evidence.get("identifiers") or {},
        "track_titles": list(evidence.get("track_titles") or [])[:8],
        "existing_genres_by_source": by_source,
        "known_tags": known,
        "user_file_tags": file_tags,
        "current_observed_leaf": observed,
        "prompt_version": ADJUDICATOR_PROMPT_VERSION,
        "schema_version": ADJUDICATOR_SCHEMA_VERSION,
    }


def enforce_file_tag_floor(
    response: dict[str, Any],
    *,
    file_tags: list[str],
    canonicalize_fn: Callable[[str], Any],
    is_broad_fn: Callable[[str], bool],
) -> dict[str, Any]:
    """Deterministic floor: a SPECIFIC (resolved, non-broad) user file-tag genre that
    is absent from the proposed set forces `escalate` (the prompt can't be trusted on
    the tail). Production-valid — uses the user's file tags, not gold.
    """
    proposed = set(canonicalize_proposed([g["term"] for g in response["genres"]], canonicalize_fn)["canonical"])
    required: set[str] = set()
    for tag in file_tags:
        result = canonicalize_fn(tag)
        name = getattr(result, "canonical", None)
        if getattr(result, "resolution", None) in ("canonical", "alias") and name and not is_broad_fn(name):
            required.add(name)
    dropped = sorted(required - proposed)
    out = dict(response)
    out["dropped_file_tags"] = dropped
    if dropped and not response.get("escalate"):
        note = f"file-tag floor: specific file tag(s) dropped: {dropped}"
        prev = str(response.get("escalate_reason", "")).strip()
        out["escalate"] = True
        out["escalate_reason"] = f"{prev}; {note}".strip("; ") if prev else note
        out["warnings"] = list(response.get("warnings", [])) + [note]
    return out


def canonicalize_proposed(
    terms: list[str],
    canonicalize_fn: Callable[[str], Any],
) -> dict[str, list[str]]:
    """Split proposed genre terms into taxonomy-canonical names vs gaps.

    `canonicalize_fn(term)` returns an object with `.resolution`
    ("canonical"/"alias"/...) and `.canonical` (the canonical name or None) —
    in production this is `graph_adapter.canonicalize_tag`. Canonical/alias hits
    are emitted as their canonical name; everything else is a gap. Both lists are
    deduped, first-seen order preserved.
    """
    canonical: list[str] = []
    gaps: list[str] = []
    seen_canon: set[str] = set()
    seen_gap: set[str] = set()
    for term in terms:
        result = canonicalize_fn(term)
        name = getattr(result, "canonical", None)
        if getattr(result, "resolution", None) in ("canonical", "alias") and name:
            if name not in seen_canon:
                seen_canon.add(name)
                canonical.append(name)
        else:
            raw = str(term)
            if raw not in seen_gap:
                seen_gap.add(raw)
                gaps.append(raw)
    return {"canonical": canonical, "gaps": gaps}
