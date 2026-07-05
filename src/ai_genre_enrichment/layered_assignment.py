from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .hybrid_evidence import EvidenceTerm, FusedGenreDecision, HybridGenreReport, collect_hybrid_evidence, fuse_hybrid_evidence
from .layered_taxonomy import FAMILY_KIND, CanonicalFacet, CanonicalGenre, LayeredTaxonomy


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


def classify_layered_term(
    taxonomy: LayeredTaxonomy,
    term: str,
    *,
    context_terms: list[str] | tuple[str, ...] | None = None,
) -> LayeredTermClassification:
    normalized = " ".join(str(term or "").casefold().replace("_", " ").replace("-", " ").split())
    rejected = taxonomy.rejected_term_by_name(normalized)
    if rejected is not None:
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=normalized,
            term_kind="reject",
            reason=rejected.reason,
        )

    alias_target = taxonomy.exact_alias_target_for_name(term, context_terms=context_terms)
    if alias_target is not None:
        if isinstance(alias_target, CanonicalFacet):
            return LayeredTermClassification(
                raw_term=term,
                normalized_term=alias_target.name,
                term_kind="alias",
                canonical_id=alias_target.facet_id,
                canonical_name=alias_target.name,
                canonical_kind=alias_target.facet_type,
                reason="Known alias for canonical facet.",
            )
        return LayeredTermClassification(
            raw_term=term,
            normalized_term=alias_target.name,
            term_kind="alias",
            canonical_id=alias_target.genre_id,
            canonical_name=alias_target.name,
            canonical_kind=_genre_term_kind(alias_target),
            reason="Known alias for canonical genre.",
        )

    genre = taxonomy.genre_by_name(normalized, context_terms=context_terms)
    if genre is not None:
        if genre.status == "review":
            return LayeredTermClassification(
                raw_term=term,
                normalized_term=genre.name,
                term_kind="review",
                canonical_id=genre.genre_id,
                canonical_name=genre.name,
                canonical_kind=genre.kind,
                reason="Known taxonomy term marked for review.",
            )
        if genre.status in {"deprecated", "alias_only"}:
            return LayeredTermClassification(
                raw_term=term,
                normalized_term=genre.name,
                term_kind="review",
                canonical_id=genre.genre_id,
                canonical_name=genre.name,
                canonical_kind=genre.kind,
                reason=f"Known taxonomy term has status {genre.status}.",
            )
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


def compute_layered_assignment_rows(
    report: HybridGenreReport, taxonomy: LayeredTaxonomy
) -> dict[str, Any]:
    """Pure: turn a fused report into the genre/facet rows the materializer writes.

    Extracted so callers can preview the exact assignment a re-materialization
    would produce (e.g. the delta-migration dry-run) without touching the
    sidecar. ``materialize_layered_assignments`` is this plus the write.
    """
    genre_rows: dict[tuple[str, str], dict[str, Any]] = {}
    facet_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_term_count = len(report.rejected_noise)
    review_term_count = 0
    context_terms = [
        decision.term
        for decision in (
            list(report.accepted_genres)
            + list(report.provisional_genres)
            + list(report.rejected_noise)
        )
    ]

    for decision in list(report.accepted_genres) + list(report.provisional_genres):
        classification = classify_layered_term(taxonomy, decision.term, context_terms=context_terms)
        if classification.term_kind == "reject":
            rejected_term_count += 1
            continue
        if classification.term_kind == "review":
            review_term_count += 1
            continue
        if classification.term_kind == "facet" or _classification_targets_facet(classification, taxonomy):
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

    return {
        "genre_rows": list(genre_rows.values()),
        "facet_rows": list(facet_rows.values()),
        "rejected_term_count": rejected_term_count,
        "review_term_count": review_term_count,
    }


def materialize_layered_assignments(
    store: Any,
    *,
    release_id: str,
    artist: str,
    album: str,
    report: HybridGenreReport,
    taxonomy: LayeredTaxonomy,
) -> LayeredAssignmentSummary:
    computed = compute_layered_assignment_rows(report, taxonomy)
    counts = store.replace_layered_assignments_for_release(
        release_id=release_id,
        artist=artist,
        album=album,
        genre_assignments=computed["genre_rows"],
        facet_assignments=computed["facet_rows"],
    )
    return LayeredAssignmentSummary(
        release_id=release_id,
        genre_assignment_count=counts["genre_assignment_count"],
        facet_assignment_count=counts["facet_assignment_count"],
        rejected_term_count=computed["rejected_term_count"],
        review_term_count=computed["review_term_count"],
    )


def build_layered_release_diagnostics(
    store: Any,
    *,
    release_id: str,
    taxonomy: LayeredTaxonomy,
    sparse_release: bool = False,
) -> dict[str, Any]:
    evidence = collect_hybrid_evidence(store, release_id)
    report = fuse_hybrid_evidence(release_key=release_id, evidence=evidence, sparse_release=sparse_release)
    context_terms = [item.term for item in evidence]
    assignment_summary = store.layered_release_summary(release_id)
    accepted_leaf_terms: list[dict[str, Any]] = []
    accepted_broad_terms: list[dict[str, Any]] = []
    accepted_facets: list[dict[str, Any]] = []
    review_terms: dict[str, dict[str, Any]] = {}
    rejected_terms: dict[str, dict[str, Any]] = {}

    for item in evidence:
        classification = classify_layered_term(taxonomy, item.term, context_terms=context_terms)
        if classification.term_kind == "reject":
            rejected_terms.setdefault(classification.normalized_term, _classification_evidence_row(item, classification))
        elif classification.term_kind == "review":
            review_terms.setdefault(classification.normalized_term, _classification_evidence_row(item, classification))
        elif classification.term_kind == "facet" or _classification_targets_facet(classification, taxonomy):
            accepted_facets.append(_classification_evidence_row(item, classification))

    for decision in report.rejected_noise:
        _merge_decision_row(rejected_terms, decision, "hybrid_fusion", taxonomy, context_terms=context_terms)
    for decision in report.provisional_genres:
        # A provisional decision whose term is UNKNOWN to the layered taxonomy
        # (term_kind == "review") is a coverage gap, not a published term: it
        # never reaches genre_graph_release_genre_assignments (see
        # compute_layered_assignment_rows). Its queue basis must stay
        # "layered_taxonomy" so it isn't double-counted as a published term
        # by the queue-stats split (pending_published_terms vs
        # pending_coverage_terms) — matching the pass-1 seed from
        # _classification_evidence_row instead of clobbering it.
        decision_classification = classify_layered_term(taxonomy, decision.term, context_terms=context_terms)
        provisional_basis = (
            "layered_taxonomy" if decision_classification.term_kind == "review" else "hybrid_provisional"
        )
        _merge_decision_row(review_terms, decision, provisional_basis, taxonomy, context_terms=context_terms)
    for decision in report.accepted_genres:
        row = _decision_row(decision, taxonomy, context_terms=context_terms)
        if row["term_kind"] == "family":
            accepted_broad_terms.append(row)
        elif row["term_kind"] == "facet" or taxonomy.facet_by_id(str(row.get("canonical_id") or "")) is not None:
            accepted_facets.append(row)
        elif row["term_kind"] == "reject":
            rejected_terms.setdefault(row["term"], row)
        elif row["term_kind"] == "review":
            review_terms.setdefault(row["term"], row)
        else:
            accepted_leaf_terms.append(row)

    inferred_terms = [
        _annotate_inferred_assignment(row, taxonomy)
        for rows in assignment_summary.get("genres_by_layer", {}).values()
        for row in rows
        if str(row.get("assignment_layer")) in {"inferred_parent", "inferred_family"}
    ]
    assignment_count = int(assignment_summary.get("genre_assignment_count") or 0) + int(
        assignment_summary.get("facet_assignment_count") or 0
    )
    return {
        **assignment_summary,
        "raw_evidence": [_evidence_row(item) for item in evidence],
        "raw_evidence_terms": [_evidence_row(item) for item in evidence],
        "normalized_evidence": _normalized_evidence(evidence, taxonomy, context_terms=context_terms),
        "normalized_evidence_terms": _normalized_evidence(evidence, taxonomy, context_terms=context_terms),
        "accepted_leaf_terms": _dedupe_rows_by_term(accepted_leaf_terms),
        "accepted_broad_terms": _dedupe_rows_by_term(accepted_broad_terms),
        "accepted_facets": _dedupe_rows_by_term(accepted_facets),
        "inferred_terms": inferred_terms,
        "review_terms": sorted(review_terms.values(), key=lambda row: row["term"]),
        "rejected_terms": sorted(rejected_terms.values(), key=lambda row: row["term"]),
        "model_prior_exists": any(item.source_type == "model_prior" for item in evidence),
        "model_prior_presence": any(item.source_type == "model_prior" for item in evidence),
        "evidence_status": _evidence_status(evidence=evidence, assignment_count=assignment_count),
        "zero_assignment_status": _evidence_status(evidence=evidence, assignment_count=assignment_count),
        "missing_taxonomy_terms": sorted(
            row["term"] for row in review_terms.values() if row.get("reason") == "Unknown layered taxonomy term."
        ),
    }


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


def _dedupe_rows_by_term(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_term: dict[str, dict[str, Any]] = {}
    for row in rows:
        term = str(row.get("term") or "")
        existing = by_term.get(term)
        if existing is None or float(row.get("confidence") or 0.0) > float(existing.get("confidence") or 0.0):
            by_term[term] = row
    return sorted(by_term.values(), key=lambda row: row["term"])


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
    if genre.kind == FAMILY_KIND or genre.role in {"family", "umbrella", "context"}:
        return "family"
    return "leaf"


def _classification_targets_facet(classification: LayeredTermClassification, taxonomy: LayeredTaxonomy) -> bool:
    return bool(classification.canonical_id and taxonomy.facet_by_id(classification.canonical_id) is not None)


def _classification_evidence_row(item: EvidenceTerm, classification: LayeredTermClassification) -> dict[str, Any]:
    return {
        "term": classification.normalized_term,
        "raw_terms": [item.term],
        "term_kind": classification.term_kind,
        "canonical_id": classification.canonical_id,
        "canonical_name": classification.canonical_name,
        "canonical_kind": classification.canonical_kind,
        "confidence": item.confidence,
        "source_basis": "layered_taxonomy",
        "sources": [item.source_type],
        "reason": classification.reason,
        "reject_reason": classification.reason if classification.term_kind == "reject" else None,
    }


def _evidence_row(item: EvidenceTerm) -> dict[str, Any]:
    return {
        "term": item.term,
        "source_type": item.source_type,
        "confidence": item.confidence,
        "canonical_slug": item.canonical_slug,
        "mapping_status": item.mapping_status,
        "classifier": item.classifier,
        "notes": item.notes,
    }


def _normalized_evidence(
    evidence: list[EvidenceTerm],
    taxonomy: LayeredTaxonomy,
    *,
    context_terms: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in evidence:
        classification = classify_layered_term(taxonomy, item.term, context_terms=context_terms)
        row = rows.setdefault(
            classification.normalized_term,
            {
                "term": classification.normalized_term,
                "term_kind": classification.term_kind,
                "canonical_id": classification.canonical_id,
                "canonical_name": classification.canonical_name,
                "canonical_kind": classification.canonical_kind,
                "sources": set(),
                "mapping_statuses": set(),
                "classifiers": set(),
                "max_confidence": 0.0,
                "reason": classification.reason,
            },
        )
        row["sources"].add(item.source_type)
        row["mapping_statuses"].add(item.mapping_status)
        row["classifiers"].add(item.classifier)
        row["max_confidence"] = max(float(row["max_confidence"]), item.confidence)
    return [
        {
            **row,
            "sources": sorted(row["sources"]),
            "mapping_statuses": sorted(row["mapping_statuses"]),
            "classifiers": sorted(row["classifiers"]),
        }
        for row in sorted(rows.values(), key=lambda item: item["term"])
    ]


def _decision_row(
    decision: FusedGenreDecision,
    taxonomy: LayeredTaxonomy,
    *,
    context_terms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    classification = classify_layered_term(taxonomy, decision.term, context_terms=context_terms)
    reason = decision.reason
    if classification.term_kind in {"reject", "review"}:
        reason = classification.reason
    return {
        "term": classification.normalized_term,
        "raw_term": decision.term,
        "term_kind": classification.term_kind,
        "canonical_id": classification.canonical_id,
        "canonical_name": classification.canonical_name,
        "canonical_kind": classification.canonical_kind,
        "confidence": decision.confidence,
        "basis": decision.basis,
        "sources": sorted(decision.sources),
        "reason": reason,
        "reject_reason": classification.reason if classification.term_kind == "reject" else None,
    }


def _merge_decision_row(
    rows: dict[str, dict[str, Any]],
    decision: FusedGenreDecision,
    source_basis: str,
    taxonomy: LayeredTaxonomy,
    *,
    context_terms: list[str] | tuple[str, ...] | None = None,
) -> None:
    row = _decision_row(decision, taxonomy, context_terms=context_terms)
    existing = rows.setdefault(row["term"], row)
    existing["confidence"] = max(float(existing.get("confidence") or 0.0), decision.confidence)
    existing["sources"] = sorted(set(existing.get("sources", [])) | set(decision.sources))
    existing["source_basis"] = source_basis
    if row["term_kind"] == "review" and row["reason"] == "Unknown layered taxonomy term.":
        existing["reason"] = row["reason"]


def _annotate_inferred_assignment(row: dict[str, Any], taxonomy: LayeredTaxonomy) -> dict[str, Any]:
    provenance = row.get("provenance") or {}
    source_term = str(provenance.get("term") or "")
    source_classification = classify_layered_term(taxonomy, source_term)
    edge = None
    if source_classification.canonical_id:
        edge = taxonomy.edge_for_genre(source_classification.canonical_id, str(row.get("genre_id")))
    return {
        "term": row.get("name"),
        "genre_id": row.get("genre_id"),
        "assignment_layer": row.get("assignment_layer"),
        "source_term": source_term,
        "confidence": row.get("confidence"),
        "basis": provenance.get("basis"),
        "sources": provenance.get("sources", []),
        "reason": provenance.get("reason"),
        "inference_edge": asdict(edge) if edge is not None else None,
    }


def _evidence_status(*, evidence: list[EvidenceTerm], assignment_count: int) -> str:
    if not evidence:
        return "no_evidence"
    if assignment_count == 0:
        return "evidence_present_no_assignments"
    return "assignments_present"
