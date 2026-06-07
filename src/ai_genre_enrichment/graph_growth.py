"""Graph growth pre-pass: propose + ingest new genres into the layered taxonomy.

See docs/superpowers/specs/2026-06-06-sp3a-graph-growth-design.md.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .layered_assignment import classify_layered_term
from .layered_taxonomy import LayeredTaxonomy


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
