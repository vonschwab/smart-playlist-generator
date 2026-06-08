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
    facet_type: str | None = None
    canonical_target: str | None = None


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
                facet_type=(str(p["facet_type"]) if p.get("facet_type") else None),
                canonical_target=(str(p["canonical_target"]) if p.get("canonical_target") else None),
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


def _parent_target_error(taxonomy: LayeredTaxonomy, target: str) -> str | None:
    """Return an error message if `target` cannot serve as a parent-edge target.

    The loader (`layered_taxonomy._structured_taxonomy_from_data`) resolves
    `parent_edges[].target` strings against a canonical-genre-name-only dict:
    aliases are not in that dict (so an alias name resolves to nothing, which
    either silently drops the edge or raises `ValueError` on load), and facet
    matches are explicitly skipped (`continue`) rather than turned into edges.
    `genre_by_name` resolves aliases, so it is too permissive for this check —
    we need to confirm the target IS a genre's own canonical name.
    """
    norm = normalize_taxonomy_name(target)
    genre = taxonomy.genre_by_name(norm)
    if genre is not None:
        if normalize_taxonomy_name(genre.name) != norm:
            return (
                f"parent target '{target}' is an alias, not a canonical genre "
                f"name — use '{genre.name}' instead"
            )
        return None
    if taxonomy.facet_by_name(norm) is not None:
        return (
            f"parent target '{target}' is a facet, not a genre — facets cannot "
            "be parent-edge targets"
        )
    return f"parent edge target does not exist: '{target}'"


def _alias_target_error(taxonomy: LayeredTaxonomy, target: str) -> str | None:
    """Return an error message if `target` cannot serve as an alias `canonical_target`.

    Unlike parent-edge targets, alias targets MAY be facets or umbrellas (the
    loader's alias resolution covers `genre_by_name`'s full dict, which includes
    facets/umbrellas — confirmed against `_structured_taxonomy_from_data`). The
    one thing an alias must never point at is *another alias* — that produces a
    chain the loader doesn't follow, silently orphaning the new alias.
    """
    norm = normalize_taxonomy_name(target)
    genre = taxonomy.genre_by_name(norm)
    if genre is not None:
        if normalize_taxonomy_name(genre.name) != norm:
            return (
                f"alias target '{target}' is itself an alias — point directly "
                f"at '{genre.name}' instead"
            )
        return None
    if taxonomy.facet_by_name(norm) is not None:
        return None
    return f"alias target does not exist: '{target}'"


# Kinds the growth pipeline can append as canonical records. `genre`/`subgenre`
# are leaves (need >=1 resolvable parent edge); `umbrella` is broad context
# (parent edges optional, mirrors e.g. `world music` having none); `facet` is
# a descriptor/modifier (no parent edges — facets are never genre-hierarchy
# nodes; see `lo-fi`/`female vocals` in the taxonomy); `alias` is a plain
# spelling/naming variant pointing at an EXISTING canonical record (for new
# canonical records, use `alias_variants` on the genre/umbrella/facet proposal
# instead — that's the batch-safe path; see `append_approved_to_taxonomy`).
_LEAF_KINDS = {"genre", "subgenre"}
_GROWTH_PROPOSABLE_KINDS = _LEAF_KINDS | {"umbrella", "facet", "alias"}

# Mirrors `enums.facet_type` in `data/layered_genre_taxonomy.yaml` (excluding
# the `None` sentinel, which only applies to non-facet records). Kept here
# because `LayeredTaxonomy` doesn't carry the raw enum table at runtime.
_VALID_FACET_TYPES = {
    "mood", "texture", "instrumentation", "production", "era", "region",
    "function", "vocal", "scene", "format", "rhythm",
}


def validate_proposal(taxonomy: LayeredTaxonomy, proposal: GrowthProposal) -> list[str]:
    """Return a list of structural errors ([] means the proposal is safe to add)."""
    errors: list[str] = []
    name = (proposal.name or "").strip()
    if not name:
        errors.append("Proposal has an empty name.")
        return errors
    # `term_kind_confirm` is an AI-proposal sanity field with a closed schema
    # enum (genre/facet/noise — no "alias" entry); plain alias proposals are
    # human-authored (no AI placement step), so "genre" is the expected value.
    expected_confirm = "facet" if proposal.kind == "facet" else "genre"
    if proposal.term_kind_confirm != expected_confirm:
        errors.append(
            f"term_kind_confirm mismatch: expected '{expected_confirm}' for "
            f"kind={proposal.kind}, got '{proposal.term_kind_confirm}'."
        )
    if proposal.kind not in _GROWTH_PROPOSABLE_KINDS:
        errors.append(f"Unsupported kind: {proposal.kind}")
    if not (0.0 <= float(proposal.specificity_score) <= 1.0):
        errors.append(f"specificity_score out of range: {proposal.specificity_score}")
    if _name_exists(taxonomy, name):
        errors.append(f"A taxonomy record named/sluged like '{name}' already exists.")

    if proposal.kind == "alias":
        target = (proposal.canonical_target or "").strip()
        if not target:
            errors.append("An alias proposal needs a canonical_target.")
        else:
            err = _alias_target_error(taxonomy, target)
            if err is not None:
                errors.append(err)
        if proposal.parent_edges:
            errors.append("Alias proposals cannot have parent_edges (aliases are pure name redirects).")
        if proposal.similar_to:
            errors.append("Alias proposals cannot have similar_to (no bridge edges for aliases).")
        if proposal.alias_variants:
            errors.append("Alias proposals cannot have alias_variants (an alias cannot itself have aliases).")
        return errors

    if proposal.kind == "facet":
        facet_type = (proposal.facet_type or "").strip()
        if not facet_type:
            errors.append("A facet proposal needs a facet_type.")
        elif facet_type not in _VALID_FACET_TYPES:
            errors.append(f"Unsupported facet_type: {facet_type}")
        if proposal.parent_edges:
            errors.append("Facet proposals cannot have parent_edges (facets are modifiers, not hierarchy nodes).")
        if proposal.similar_to:
            errors.append("Facet proposals cannot have similar_to (no bridge edges for modifiers).")
        return errors

    if proposal.kind in _LEAF_KINDS and not proposal.parent_edges:
        errors.append("A new leaf genre needs at least one parent edge.")
    for edge in proposal.parent_edges:
        target = str(edge.get("target") or "").strip()
        err = _parent_target_error(taxonomy, target)
        if err is not None:
            errors.append(err)
    # similar_to entries become bridge_to parent_edges in `_proposal_record`, so
    # they are subject to the same loader-side canonical-name-only resolution.
    for target in proposal.similar_to:
        err = _parent_target_error(taxonomy, str(target).strip())
        if err is not None:
            errors.append(err.replace("parent target", "similar_to target")
                          .replace("parent edge target", "similar_to target"))
    return errors


@dataclass
class AppendResult:
    appended: int
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _proposal_record(proposal: GrowthProposal) -> dict:
    """Build the taxonomy record dict for an approved proposal of any growable kind.

    `genre`/`subgenre` -> leaf (parent edges define identity); `umbrella` ->
    context (parent edges optional, mirrors e.g. `world music`/`avant-garde`);
    `facet` -> modifier (no parent edges — descriptors sit outside the genre
    hierarchy, mirrors `lo-fi`/`female vocals`); `alias` -> pure name redirect
    to an existing canonical record (built via the same `_alias_record` helper
    `alias_variants` uses, just targeting an existing name instead of a sibling
    record created in this batch).
    """
    if proposal.kind == "alias":
        return _alias_record(proposal.name, proposal.canonical_target)

    if proposal.kind == "facet":
        return {
            "name": proposal.name,
            "kind": "facet",
            "role": "modifier",
            "status": proposal.status or "active",
            "facet_type": proposal.facet_type,
            "specificity_score": float(proposal.specificity_score),
            "canonical_target": None,
            "parent_edges": [],
            "secondary_roles": [],
            "reject_reason": None,
            "alias_policy": None,
            "source_policy": "growth",
            "possible_context_target": None,
            "notes": proposal.rationale or "Added via SP3a graph growth.",
        }

    parent_edges = [
        {
            "target": e["target"],
            "edge_type": e.get("edge_type", "family_context"),
            "weight": float(e.get("weight", 0.55)),
            "confidence": float(e.get("confidence", 0.8)),
            "notes": None,
        }
        for e in proposal.parent_edges
    ]
    # similar_to becomes bridge_to edges in the same parent_edges channel the
    # loader reads (it resolves any parent_edges target by name).
    for target in proposal.similar_to:
        parent_edges.append({
            "target": target, "edge_type": "bridge_to",
            "weight": 0.4, "confidence": 0.6, "notes": "similar_to (growth)",
        })
    role = "context" if proposal.kind == "umbrella" else "leaf"
    return {
        "name": proposal.name,
        "kind": proposal.kind,
        "role": role,
        "status": proposal.status or "active",
        "facet_type": None,
        "specificity_score": float(proposal.specificity_score),
        "canonical_target": None,
        "parent_edges": parent_edges,
        "secondary_roles": [],
        "reject_reason": None,
        "alias_policy": None,
        "source_policy": "growth",
        "possible_context_target": None,
        "notes": proposal.rationale or "Added via SP3a graph growth.",
    }


def _alias_record(variant: str, canonical_name: str) -> dict:
    return {
        "name": variant,
        "kind": "alias",
        "role": "alias",
        "status": "alias_only",
        "facet_type": None,
        "specificity_score": None,
        "canonical_target": canonical_name,
        "parent_edges": [],
        "secondary_roles": [],
        "reject_reason": None,
        "alias_policy": {"type": "plain"},
        "source_policy": None,
        "possible_context_target": None,
        "notes": "Spelling variant (growth).",
    }


def append_approved_to_taxonomy(taxonomy_path, approved: list[GrowthProposal],
                                *, new_version: str) -> AppendResult:
    """Append approved proposals as records to the taxonomy YAML and bump version.

    Caller is responsible for having validated each proposal first. New genre
    records are appended before their alias records so name targets resolve.
    """
    path = Path(taxonomy_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = data.setdefault("records", [])
    appended = 0
    for proposal in approved:
        records.append(_proposal_record(proposal))
        for variant in proposal.alias_variants:
            if variant and normalize_taxonomy_name(variant) != normalize_taxonomy_name(proposal.name):
                records.append(_alias_record(variant, proposal.name))
        appended += 1
    data["taxonomy_version"] = new_version
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return AppendResult(appended=appended)
