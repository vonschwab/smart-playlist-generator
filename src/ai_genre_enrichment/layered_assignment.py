from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hybrid_evidence import FusedGenreDecision, HybridGenreReport
from .layered_taxonomy import FAMILY_KIND, CanonicalGenre, LayeredTaxonomy


SOURCE_RELIABILITY: dict[str, float] = {
    "human": 1.00,
    "official_release": 0.95,
    "bandcamp_release": 0.90,
    "discogs": 0.75,
    "musicbrainz": 0.75,
    "local_metadata": 0.70,
    "model_prior": 0.60,
    "lastfm_tags": 0.45,
    "lastfm": 0.45,
}


@dataclass(frozen=True)
class LayeredAssignmentSummary:
    release_id: str
    genre_assignment_count: int
    facet_assignment_count: int
    rejected_term_count: int = 0
    review_term_count: int = 0


@dataclass(frozen=True)
class LayeredTermClassification:
    raw_term: str
    normalized_term: str
    term_kind: str
    canonical_id: str | None = None
    canonical_name: str | None = None
    canonical_kind: str | None = None
    reason: str = ""


def classify_layered_term(taxonomy: LayeredTaxonomy, term: str) -> LayeredTermClassification:
    normalized = " ".join(str(term or "").casefold().replace("_", " ").replace("-", " ").split())
    rejected = taxonomy.rejected_term_by_name(normalized)
    if rejected is not None:
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=normalized,
            term_kind="reject",
            reason=rejected.reason,
        )

    alias_target = taxonomy.exact_alias_target_for_name(term)
    if alias_target is not None:
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=alias_target.name,
            term_kind="alias",
            canonical_id=alias_target.genre_id,
            canonical_name=alias_target.name,
            canonical_kind=_genre_term_kind(alias_target),
            reason="Known alias for canonical genre.",
        )

    genre = taxonomy.genre_by_name(normalized)
    if genre is not None:
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=genre.name,
            term_kind=_genre_term_kind(genre),
            canonical_id=genre.genre_id,
            canonical_name=genre.name,
            canonical_kind=genre.kind,
            reason="Known canonical genre.",
        )

    facet = taxonomy.facet_by_name(normalized)
    if facet is not None:
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=facet.name,
            term_kind="facet",
            canonical_id=facet.facet_id,
            canonical_name=facet.name,
            canonical_kind=facet.facet_type,
            reason="Known canonical facet.",
        )

    return LayeredTermClassification(
        raw_term=term,
        normalized_term=normalized,
        term_kind="review",
        reason="Unknown layered taxonomy term.",
    )


def materialize_layered_assignments(
    store: Any,
    *,
    release_id: str,
    artist: str,
    album: str,
    report: HybridGenreReport,
    taxonomy: LayeredTaxonomy,
) -> LayeredAssignmentSummary:
    genre_rows: dict[tuple[str, str], dict[str, Any]] = {}
    facet_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_term_count = len(report.rejected_noise)
    review_term_count = len(report.needs_review)

    for decision in report.accepted_genres:
        classification = classify_layered_term(taxonomy, decision.term)
        if classification.term_kind == "reject":
            rejected_term_count += 1
            continue
        if classification.term_kind == "review":
            review_term_count += 1
            continue
        if classification.term_kind == "facet":
            source = _assignment_source(decision)
            facet_rows[(classification.canonical_id, source)] = {
                "facet_id": classification.canonical_id,
                "confidence": decision.confidence,
                "source": source,
                "provenance": _provenance(decision, classification.normalized_term),
            }
            continue

        genre = taxonomy.genre_by_id(classification.canonical_id or "")
        if genre is None:
            continue
        if classification.term_kind == "family":
            _put_genre_row(genre_rows, genre, "inferred_family", decision)
            continue

        _put_genre_row(genre_rows, genre, "observed_leaf", decision)
        for parent in taxonomy.parents_for_genre(genre.genre_id):
            _put_genre_row(genre_rows, parent, "inferred_parent", decision)
        for family in taxonomy.families_for_genre(genre.genre_id):
            _put_genre_row(genre_rows, family, "inferred_family", decision)

    counts = store.replace_layered_assignments_for_release(
        release_id=release_id,
        artist=artist,
        album=album,
        genre_assignments=list(genre_rows.values()),
        facet_assignments=list(facet_rows.values()),
    )
    return LayeredAssignmentSummary(
        release_id=release_id,
        genre_assignment_count=counts["genre_assignment_count"],
        facet_assignment_count=counts["facet_assignment_count"],
        rejected_term_count=rejected_term_count,
        review_term_count=review_term_count,
    )


def _put_genre_row(
    rows: dict[tuple[str, str], dict[str, Any]],
    genre: CanonicalGenre,
    assignment_layer: str,
    decision: FusedGenreDecision,
) -> None:
    key = (genre.genre_id, assignment_layer)
    candidate = {
        "genre_id": genre.genre_id,
        "assignment_layer": assignment_layer,
        "confidence": decision.confidence,
        "source_reliability": _source_reliability(decision.sources),
        "evidence_count": len(set(decision.sources)),
        "rejected_by_user": False,
        "provenance": _provenance(decision, genre.name),
    }
    existing = rows.get(key)
    if existing is None or candidate["confidence"] > existing["confidence"]:
        rows[key] = candidate


def _source_reliability(sources: list[str]) -> float:
    if not sources:
        return 0.0
    return max(SOURCE_RELIABILITY.get(source, 0.40) for source in sources)


def _assignment_source(decision: FusedGenreDecision) -> str:
    if len(decision.sources) == 1:
        return decision.sources[0]
    return "+".join(sorted(decision.sources))


def _provenance(decision: FusedGenreDecision, normalized_term: str) -> dict[str, Any]:
    return {
        "term": decision.term,
        "normalized_term": normalized_term,
        "basis": decision.basis,
        "sources": sorted(decision.sources),
        "reason": decision.reason,
    }


def _genre_term_kind(genre: CanonicalGenre) -> str:
    if genre.kind == FAMILY_KIND:
        return "family"
    return "leaf"
