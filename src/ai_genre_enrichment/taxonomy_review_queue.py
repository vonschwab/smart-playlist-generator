"""Merged/deduped candidate queue for taxonomy-term adjudication.

Surfaces genre terms present in the library but absent from the taxonomy graph,
annotated with reach (distinct albums), co-occurring tags, and example releases.
The queue is a DERIVED view: candidates are computed from the sidecar's collected
tags via graph_growth.gather_growth_candidates; staged decisions
(taxonomy_decision_store) are joined in to split untriaged vs decided. The read
path is mode=ro (reader-thread safe) and paginated (the page line stays small).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph_growth import collapse_variants, gather_growth_candidates
from .layered_taxonomy import (
    LayeredTaxonomy, load_layered_taxonomy, normalize_taxonomy_name,
)
from .storage import SidecarStore
from .tag_classification import classify_source_tag
from .taxonomy_decision_store import list_decisions

# Only these deterministic-classifier buckets warrant graph-level adjudication:
# `review_only` (genuinely unknown — might be a real new genre or genuine noise
# Claude should reject) and `genre_style` (a known genre missing from the graph).
# The other buckets — place, instrument, format, mood_function, descriptor
# (years, labels) — are non-genres/facets the noise policy already owns at
# enrichment time; surfacing them here would flood the queue with junk like
# "new york" / "2016" / "piano" that needs no per-term Claude call.
_QUEUE_KEEP_CLASSIFICATIONS = frozenset({"review_only", "genre_style"})


@dataclass
class TaxonomyCandidate:
    term: str
    raw_term: str
    album_frequency: int
    cooccurring_tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    source: str = "growth"


def build_candidate_index(
    store, taxonomy: LayeredTaxonomy, *, min_album_freq: int = 3,
) -> dict[str, TaxonomyCandidate]:
    """Normalized-name -> candidate.

    Spine = gather_growth_candidates (taxonomy-aware unmapped detection over the
    sidecar's collected tags) + collapse_variants (spacing-variant dedup). Impact
    count is the album_frequency = distinct releases where the term appears as an
    observed/legacy collected tag.

    Variants are collapsed BEFORE the frequency threshold so a candidate's reach
    reflects its combined spellings (``vapor wave`` + ``vaporwave``) and the
    minority spelling isn't silently dropped below threshold. We therefore gather
    at freq 1 and apply ``min_album_freq`` to the merged reach.
    """
    candidates = collapse_variants(
        gather_growth_candidates(store, taxonomy, min_album_freq=1)
    )
    index: dict[str, TaxonomyCandidate] = {}
    for c in candidates:
        if c.album_frequency < min_album_freq:
            continue
        # Apply the deterministic noise policy: drop place/year/instrument/format/
        # mood/label tags the system already resolves without graph adjudication.
        if classify_source_tag(c.term).classification not in _QUEUE_KEEP_CLASSIFICATIONS:
            continue
        key = normalize_taxonomy_name(c.term)
        index[key] = TaxonomyCandidate(
            term=key,
            raw_term=c.term,
            album_frequency=c.album_frequency,
            cooccurring_tags=list(c.cooccurring_tags),
            examples=list(c.examples),
            variants=list(c.variants),
            source="growth",
        )
    return index


def _open_store_readonly(sidecar_db_path) -> SidecarStore:
    """Indirection seam so tests can inject a fake collected-tags store.

    SidecarStore.all_collected_tags() opens its own connection; this path never
    calls initialize()/DDL, so it stays read-only by construction.
    """
    return SidecarStore(sidecar_db_path)


def list_page(
    sidecar_db_path, taxonomy_path, *,
    status: str = "untriaged", search: "str | None" = None,
    limit: int = 50, offset: int = 0,
) -> dict:
    """Read-only page of candidate terms joined with staged decisions.

    status: 'untriaged' (no staged decision) or 'decided' (has one).
    Never writes; safe on the worker reader thread.
    """
    taxonomy = load_layered_taxonomy(taxonomy_path)
    store = _open_store_readonly(sidecar_db_path)
    index = build_candidate_index(store, taxonomy)

    decisions = {d["term"]: d for d in list_decisions(sidecar_db_path, status="pending")}
    decisions.update(
        {d["term"]: d for d in list_decisions(sidecar_db_path, status="applied")})

    rows: list[dict] = []
    for key, cand in index.items():
        decision = decisions.get(key)
        is_decided = decision is not None
        if status == "untriaged" and is_decided:
            continue
        if status == "decided" and not is_decided:
            continue
        if search and search.lower() not in key.lower():
            continue
        rows.append({
            "term": cand.term, "raw_term": cand.raw_term,
            "album_frequency": cand.album_frequency,
            "cooccurring_tags": cand.cooccurring_tags,
            "examples": cand.examples, "variants": cand.variants,
            "source": cand.source, "decision": decision,
        })

    rows.sort(key=lambda r: (-r["album_frequency"], r["term"]))
    untriaged_total = sum(1 for k in index if k not in decisions)
    decided_total = sum(1 for k in index if k in decisions)
    page = rows[offset:offset + limit]
    return {"terms": page, "untriaged_terms": untriaged_total,
            "decided_terms": decided_total}
