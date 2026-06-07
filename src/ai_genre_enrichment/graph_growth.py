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
