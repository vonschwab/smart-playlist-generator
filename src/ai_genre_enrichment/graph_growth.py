"""Graph growth pre-pass: propose + ingest new genres into the layered taxonomy.

See docs/superpowers/specs/2026-06-06-sp3a-graph-growth-design.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .layered_assignment import classify_layered_term
from .layered_taxonomy import FAMILY_KIND, LayeredTaxonomy, _record_id, normalize_taxonomy_name
from .routing import WebMode


@dataclass
class GrowthCandidate:
    term: str
    album_frequency: int
    cooccurring_tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)


def gather_growth_candidates(
    store,
    taxonomy: LayeredTaxonomy,
    *,
    min_album_freq: int = 3,
    max_examples: int = 3,
    max_cooccurring: int = 8,
) -> list[GrowthCandidate]:
    """Distinct genuinely-unmapped genres at/above the album-frequency threshold.

    A candidate is a tag whose classify_layered_term yields term_kind == 'review'
    AND canonical_id is None (genuinely unknown). Mapped genres, aliases, facets,
    rejects, and review-but-known terms are excluded. Ranked by album_frequency.
    """
    rows = store.all_collected_tags()

    # release -> set(tags); tag -> set(releases); tag -> example releases
    tags_by_release: dict[str, set[str]] = defaultdict(set)
    releases_by_tag: dict[str, set[str]] = defaultdict(set)
    example_by_tag: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        tag = r["normalized_tag"]
        rk = r["release_key"]
        tags_by_release[rk].add(tag)
        releases_by_tag[tag].add(rk)
        if len(example_by_tag[tag]) < max_examples:
            label = f"{r['normalized_artist']} — {r['normalized_album']}"
            if label not in example_by_tag[tag]:
                example_by_tag[tag].append(label)

    candidates: list[GrowthCandidate] = []
    for tag, releases in releases_by_tag.items():
        freq = len(releases)
        if freq < min_album_freq:
            continue
        classification = classify_layered_term(taxonomy, tag)
        if classification.term_kind != "review" or classification.canonical_id is not None:
            continue  # mapped / alias / facet / reject / known-review -> not a candidate

        cooccur: Counter[str] = Counter()
        for rk in releases:
            for other in tags_by_release[rk]:
                if other != tag:
                    cooccur[other] += 1
        candidates.append(GrowthCandidate(
            term=tag,
            album_frequency=freq,
            cooccurring_tags=[t for t, _ in cooccur.most_common(max_cooccurring)],
            examples=list(example_by_tag[tag]),
        ))

    candidates.sort(key=lambda c: (-c.album_frequency, c.term))
    return candidates


def _variant_key(term: str) -> str:
    """Spacing/separator-insensitive key: 'synth wave' and 'synthwave' match."""
    return "".join(ch for ch in term.casefold() if ch.isalnum())


def collapse_variants(candidates: list[GrowthCandidate]) -> list[GrowthCandidate]:
    """Merge spacing/separator variants into the highest-frequency representative.

    The representative keeps its term + the union of examples/co-occurring tags;
    the merged-away spellings are recorded in `variants` (alias suggestions).
    Combined album_frequency is summed (upper bound; exact de-dup of overlapping
    releases is unnecessary for ranking).
    """
    groups: dict[str, list[GrowthCandidate]] = defaultdict(list)
    for cand in candidates:
        groups[_variant_key(cand.term)].append(cand)

    merged: list[GrowthCandidate] = []
    for members in groups.values():
        members.sort(key=lambda c: (-c.album_frequency, c.term))
        rep = members[0]
        variants = [m.term for m in members[1:]]
        cooccur: list[str] = list(rep.cooccurring_tags)
        examples: list[str] = list(rep.examples)
        for m in members[1:]:
            for t in m.cooccurring_tags:
                if t not in cooccur:
                    cooccur.append(t)
            for e in m.examples:
                if e not in examples:
                    examples.append(e)
        merged.append(GrowthCandidate(
            term=rep.term,
            album_frequency=sum(m.album_frequency for m in members),
            cooccurring_tags=cooccur,
            examples=examples,
            variants=variants,
        ))
    merged.sort(key=lambda c: (-c.album_frequency, c.term))
    return merged


GROWTH_PROPOSAL_INSTRUCTIONS = """
You place a new music genre into an existing hierarchical genre taxonomy.
Given a candidate genre term, its evidence (how often it appears and which
genres co-occur with it), and the relevant existing taxonomy names, propose
where it belongs.

Rules:
- Only propose a placement if the term is a real GENRE/subgenre. If it is a
  descriptor/facet (mood, instrument, era, region, format) set term_kind_confirm
  to "facet"; if it is noise/non-music set it to "noise".
- parent_edges must reference EXISTING taxonomy names exactly as given. Choose
  1-2 parents (a family via "family_context", or a broader genre via "is_a").
- specificity_score: ~0.05 for broad families, ~0.5 for mid genres, ~0.8-0.9 for
  narrow microgenres.
- Do not invent edges to names not in the provided context.
""".strip()


_GROWTH_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "kind", "status", "specificity_score", "parent_edges",
                 "similar_to", "alias_variants", "term_kind_confirm", "rationale"],
    "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": ["genre", "subgenre"]},
        "status": {"type": "string", "enum": ["active", "review"]},
        "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "parent_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
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
        "term_kind_confirm": {"type": "string",
                              "enum": ["genre", "facet", "noise"]},
        "rationale": {"type": "string"},
    },
}


def growth_proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "graph_growth_proposal",
        "schema": deepcopy(_GROWTH_PROPOSAL_SCHEMA),
        "strict": True,
    }


@dataclass
class GrowthProposal:
    name: str
    kind: str
    status: str
    specificity_score: float
    parent_edges: list[dict] = field(default_factory=list)
    similar_to: list[str] = field(default_factory=list)
    alias_variants: list[str] = field(default_factory=list)
    term_kind_confirm: str = "genre"
    rationale: str = ""


def _build_taxonomy_context(taxonomy: LayeredTaxonomy, candidate: GrowthCandidate) -> list[str]:
    """Bounded context: all families/umbrellas + genres sharing a token with the
    candidate or its co-occurring tags. Keeps the prompt small but relevant."""
    tokens = set(normalize_taxonomy_name(candidate.term).split())
    for t in candidate.cooccurring_tags:
        tokens.update(normalize_taxonomy_name(t).split())
    names: list[str] = []
    for genre in taxonomy.genres:
        if genre.kind in {FAMILY_KIND, "umbrella"}:
            names.append(genre.name)
        elif tokens & set(normalize_taxonomy_name(genre.name).split()):
            names.append(genre.name)
    return sorted(dict.fromkeys(names))


def propose_placement(
    candidate: GrowthCandidate,
    taxonomy: LayeredTaxonomy,
    *,
    client,
    web_mode: WebMode | str = WebMode.OFF,
) -> GrowthProposal:
    """Ask the model to place one candidate. `client` exposes `_call_openai`."""
    from .client import _extract_response_json

    context_names = _build_taxonomy_context(taxonomy, candidate)
    prompt = json.dumps({
        "candidate_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": candidate.cooccurring_tags,
        "spelling_variants": candidate.variants,
        "examples": candidate.examples,
        "existing_taxonomy_names": context_names,
    }, ensure_ascii=False, sort_keys=True)
    raw = client._call_openai(
        prompt, growth_proposal_response_format(),
        instructions=GROWTH_PROPOSAL_INSTRUCTIONS,
    )
    data = _extract_response_json(raw)
    return GrowthProposal(
        name=str(data["name"]),
        kind=str(data["kind"]),
        status=str(data.get("status") or "active"),
        specificity_score=float(data["specificity_score"]),
        parent_edges=list(data.get("parent_edges") or []),
        similar_to=list(data.get("similar_to") or []),
        alias_variants=list(data.get("alias_variants") or candidate.variants),
        term_kind_confirm=str(data.get("term_kind_confirm") or "genre"),
        rationale=str(data.get("rationale") or ""),
    )


@dataclass
class ProposalEntry:
    term: str
    album_frequency: int
    cooccurring_tags: list[str]
    examples: list[str]
    decision: str
    proposal: GrowthProposal


def write_proposals(path, items: list[tuple[GrowthCandidate, GrowthProposal]]) -> None:
    """Write candidate+proposal pairs to an editable review YAML."""
    entries = []
    for cand, proposal in items:
        entries.append({
            "term": cand.term,
            "album_frequency": cand.album_frequency,
            "cooccurring_tags": list(cand.cooccurring_tags),
            "examples": list(cand.examples),
            "decision": "pending",
            "proposal": asdict(proposal),
        })
    Path(path).write_text(
        yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def read_proposals(path) -> list[ProposalEntry]:
    """Read a (possibly user-edited) proposal YAML back into entries."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    entries: list[ProposalEntry] = []
    for row in raw:
        p = row.get("proposal") or {}
        entries.append(ProposalEntry(
            term=str(row.get("term") or ""),
            album_frequency=int(row.get("album_frequency") or 0),
            cooccurring_tags=list(row.get("cooccurring_tags") or []),
            examples=list(row.get("examples") or []),
            decision=str(row.get("decision") or "pending"),
            proposal=GrowthProposal(
                name=str(p.get("name") or row.get("term") or ""),
                kind=str(p.get("kind") or "subgenre"),
                status=str(p.get("status") or "active"),
                specificity_score=float(p.get("specificity_score") or 0.5),
                parent_edges=list(p.get("parent_edges") or []),
                similar_to=list(p.get("similar_to") or []),
                alias_variants=list(p.get("alias_variants") or []),
                term_kind_confirm=str(p.get("term_kind_confirm") or "genre"),
                rationale=str(p.get("rationale") or ""),
            ),
        ))
    return entries


def _name_exists(taxonomy: LayeredTaxonomy, name: str) -> bool:
    norm = normalize_taxonomy_name(name)
    if taxonomy.genre_by_name(norm) is not None:
        return True
    if taxonomy.facet_by_name(norm) is not None:
        return True
    return taxonomy.genre_by_id(_record_id(name)) is not None


def validate_proposal(taxonomy: LayeredTaxonomy, proposal: GrowthProposal) -> list[str]:
    """Return a list of structural errors ([] means the proposal is safe to add)."""
    errors: list[str] = []
    name = (proposal.name or "").strip()
    if not name:
        errors.append("Proposal has an empty name.")
        return errors
    if proposal.term_kind_confirm != "genre":
        errors.append(f"Not a genre (term_kind_confirm={proposal.term_kind_confirm}); skip.")
    if proposal.kind not in {"genre", "subgenre"}:
        errors.append(f"Unsupported kind: {proposal.kind}")
    if not (0.0 <= float(proposal.specificity_score) <= 1.0):
        errors.append(f"specificity_score out of range: {proposal.specificity_score}")
    if _name_exists(taxonomy, name):
        errors.append(f"A taxonomy record named/sluged like '{name}' already exists.")
    if not proposal.parent_edges:
        errors.append("A new leaf genre needs at least one parent edge.")
    for edge in proposal.parent_edges:
        target = str(edge.get("target") or "").strip()
        norm = normalize_taxonomy_name(target)
        if taxonomy.genre_by_name(norm) is None and taxonomy.facet_by_name(norm) is None:
            errors.append(f"parent edge target does not exist: '{target}'")
    for target in proposal.similar_to:
        norm = normalize_taxonomy_name(target)
        if taxonomy.genre_by_name(norm) is None and taxonomy.facet_by_name(norm) is None:
            errors.append(f"similar_to target does not exist: '{target}'")
    return errors
