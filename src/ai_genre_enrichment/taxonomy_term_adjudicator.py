"""Claude contract for adjudicating a single taxonomy-vocabulary term.

One on-demand call per term. Claude returns an add/alias/reject verdict;
validate_response() maps it onto graph_growth's proposal helpers (the same
records the Apply engine writes). Modeled on album_adjudicator.py, but the
concern is *what genres exist in the vocabulary*, not *which genres an album gets*.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .graph_growth import (
    GrowthProposal, _alias_target_error, _build_taxonomy_context,
    _parent_target_error,
)
from .layered_taxonomy import LayeredTaxonomy

# Mirrors enums.reject_reason in data/layered_genre_taxonomy.yaml.
REJECT_REASONS = {
    "label", "artist_name", "release_title", "place", "format", "era",
    "user_list", "malformed", "joke_tag", "negative_tag", "retail_bucket",
    "source_noise", "unknown_noise",
}

ADJUDICATOR_PROMPT_VERSION = "taxonomy-term-adjudicator-v1"

TAXONOMY_ADJUDICATOR_INSTRUCTIONS = """You place ONE candidate genre term into an existing hierarchical music-genre taxonomy, or reject it.

Return ONE JSON object only — no prose, no markdown.

Decide a `verdict`:
- "add": the term is a real GENRE or subgenre that belongs in the taxonomy. Give `kind` (umbrella/genre/subgenre), `status`, `specificity_score`, `parent_edges` (1-2, each `target` an EXISTING taxonomy name exactly as given), optional `similar_to` (existing names), optional `alias_variants`, and a `rationale`.
- "alias": the term is a spelling/naming variant of an EXISTING canonical genre. Give `canonical_target` (an existing canonical name) and `rationale`. Do NOT collapse a genuinely distinct genre into an alias (uk garage is NOT garage rock).
- "reject": the term is not a genre (a label, artist name, release title, place, format, era, user list, malformed/joke/negative tag, retail bucket, or source noise). Give a `reject_reason` from the allowed list and a `rationale`.

Placement guardrails (hard rules):
- Umbrellas are broad context, LOW specificity (~0.24-0.42), with spread parentage — no single child branch gets a strong parent weight.
- Instrument-led terms (piano jazz, jazz guitar) are FACETS, not genre leaves — reject them here (reject_reason "format" or "source_noise") unless there's a real scene/style tradition beyond the instrument.
- Specificity ladder: umbrella 0.24-0.42 · genre 0.48-0.66 · subgenre 0.62-0.82.
- A leaf (genre/subgenre) needs at least one parent edge to an existing taxonomy name.
- Broad or noisy-but-real terms get status "review", not "active".
- parent_edges / similar_to targets MUST be names present in the provided `existing_taxonomy_names`. Do not invent edges to names not given.

Allowed reject_reason values: label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise.
"""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "name", "kind", "status", "specificity_score",
                 "parent_edges", "similar_to", "alias_variants",
                 "canonical_target", "reject_reason", "rationale"],
    "properties": {
        "verdict": {"type": "string", "enum": ["add", "alias", "reject"]},
        "name": {"type": "string"},
        "kind": {"type": "string",
                 "enum": ["umbrella", "genre", "subgenre", "alias", "reject"]},
        "status": {"type": "string",
                   "enum": ["active", "review", "alias_only", "rejected"]},
        "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "parent_edges": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["target", "edge_type", "weight", "confidence"],
                "properties": {
                    "target": {"type": "string"},
                    "edge_type": {"type": "string",
                                  "enum": ["is_a", "family_context"]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "similar_to": {"type": "array", "items": {"type": "string"}},
        "alias_variants": {"type": "array", "items": {"type": "string"}},
        "canonical_target": {"type": "string"},
        "reject_reason": {"type": "string"},
        "rationale": {"type": "string"},
    },
}


def taxonomy_adjudicator_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "name": "taxonomy_term_adjudicator_v1",
            "strict": True, "schema": deepcopy(_SCHEMA)}


@dataclass
class RejectVerdict:
    term: str
    reject_reason: str
    rationale: str = ""


def build_payload(candidate, taxonomy: LayeredTaxonomy) -> dict[str, Any]:
    """A *relevant slice* of the taxonomy (candidate parents/aliases by token +
    co-occurrence, plus family/umbrella anchors) so Claude places against the
    real graph, not the whole file. `candidate` is a TaxonomyCandidate-like
    object (.term/.raw_term/.album_frequency/.cooccurring_tags/.examples/.variants)."""
    context_names = _build_taxonomy_context(taxonomy, candidate)
    return {
        "candidate_term": candidate.raw_term,
        "normalized_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": list(candidate.cooccurring_tags),
        "spelling_variants": list(candidate.variants),
        "examples": list(candidate.examples),
        "existing_taxonomy_names": context_names,
        "prompt_version": ADJUDICATOR_PROMPT_VERSION,
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return ("Adjudicate this candidate genre term. Use the payload only.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def validate_response(data: dict[str, Any], *, term: str,
                      taxonomy: LayeredTaxonomy) -> "GrowthProposal | RejectVerdict":
    """Map a Claude verdict onto the existing proposal helpers, enforcing the same
    loader-resolution rules the Apply step uses. Raises ValueError on any contract
    violation (so the GUI surfaces a clean error instead of a silent bad write)."""
    if not isinstance(data, dict):
        raise ValueError("adjudicator response must be a JSON object")
    verdict = str(data.get("verdict") or "").strip()
    if verdict not in ("add", "alias", "reject"):
        raise ValueError(f"unknown verdict: {verdict!r}")

    if verdict == "reject":
        reason = str(data.get("reject_reason") or "").strip()
        if reason not in REJECT_REASONS:
            raise ValueError(f"unsupported reject_reason: {reason!r}")
        return RejectVerdict(term=term, reject_reason=reason,
                             rationale=str(data.get("rationale") or ""))

    if verdict == "alias":
        target = str(data.get("canonical_target") or "").strip()
        err = _alias_target_error(taxonomy, target)
        if err is not None:
            raise ValueError(err)
        return GrowthProposal(
            name=str(data.get("name") or term), kind="alias", status="alias_only",
            specificity_score=0.0, parent_edges=[], similar_to=[], alias_variants=[],
            term_kind_confirm="genre", canonical_target=target,
            rationale=str(data.get("rationale") or ""))

    # verdict == "add"
    kind = str(data.get("kind") or "genre")
    if kind not in ("umbrella", "genre", "subgenre"):
        raise ValueError(f"add verdict has non-leaf/umbrella kind: {kind!r}")
    parent_edges = list(data.get("parent_edges") or [])
    similar_to = list(data.get("similar_to") or [])
    for e in parent_edges:
        err = _parent_target_error(taxonomy, str(e.get("target") or ""))
        if err is not None:
            raise ValueError(err)
    for t in similar_to:
        err = _parent_target_error(taxonomy, str(t))
        if err is not None:
            raise ValueError(err)
    if kind in ("genre", "subgenre") and not parent_edges:
        raise ValueError("an add leaf needs at least one parent edge")
    return GrowthProposal(
        name=str(data.get("name") or term), kind=kind,
        status=str(data.get("status") or "active"),
        specificity_score=float(data.get("specificity_score") or 0.5),
        parent_edges=parent_edges, similar_to=similar_to,
        alias_variants=list(data.get("alias_variants") or []),
        term_kind_confirm="genre", rationale=str(data.get("rationale") or ""))


def adjudicate_term(candidate, taxonomy: LayeredTaxonomy, *,
                    client) -> "GrowthProposal | RejectVerdict":
    """One structured Claude call. `client` exposes call_structured (see
    provider.create_enrichment_client)."""
    payload = build_payload(candidate, taxonomy)
    data = client.call_structured(
        build_prompt(payload), taxonomy_adjudicator_response_format(),
        instructions=TAXONOMY_ADJUDICATOR_INSTRUCTIONS)
    return validate_response(data, term=candidate.term, taxonomy=taxonomy)
