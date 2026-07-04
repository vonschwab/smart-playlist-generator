"""Materialize an album-adjudicator response into the sidecar's layered genre store.

The adjudicator's tight observed-leaf set is expanded through the graph (parents →
inferred_parent, families → inferred_family) exactly as the hybrid materializer does,
then written for ONE album's release_key — superseding the prior hybrid rows. Facet
terms route to the facet table; unknown/review terms are recorded as skipped, never
invented. Writes the sidecar only; `publish()` remains the sole authority writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .layered_assignment import classify_layered_term
from .layered_taxonomy import CanonicalGenre, LayeredTaxonomy
from .normalization import make_release_key

ADJUDICATOR_SOURCE = "claude_adjudicator"
ADJUDICATOR_SOURCE_RELIABILITY = 0.85


@dataclass(frozen=True)
class AdjudicationMaterializeSummary:
    release_id: str
    observed_leaf: tuple[str, ...]
    inferred_count: int
    facet_count: int
    skipped_terms: tuple[str, ...]


def _add_genre(rows: dict[tuple[str, str], dict], genre: CanonicalGenre, layer: str, conf: float, *, prompt_version: str, model: str, term: str) -> None:
    key = (genre.genre_id, layer)
    candidate = {
        "genre_id": genre.genre_id,
        "assignment_layer": layer,
        "confidence": float(conf),
        "source_reliability": ADJUDICATOR_SOURCE_RELIABILITY,
        "evidence_count": 1,
        "rejected_by_user": False,
        "provenance": {
            "source": ADJUDICATOR_SOURCE, "prompt_version": prompt_version,
            "model": model, "term": term, "genre": genre.name,
        },
    }
    existing = rows.get(key)
    if existing is None or candidate["confidence"] > existing["confidence"]:
        rows[key] = candidate


def compute_adjudication_rows(
    response: dict[str, Any], taxonomy: LayeredTaxonomy, *, prompt_version: str, model: str
) -> tuple[list[dict], list[dict], list[str]]:
    genre_rows: dict[tuple[str, str], dict] = {}
    facet_rows: dict[tuple[str, str], dict] = {}
    skipped: list[str] = []
    proposed = [g["term"] for g in response.get("genres", [])]

    def _facet(canonical_id: str, conf: float, term: str) -> None:
        facet_rows[(canonical_id, ADJUDICATOR_SOURCE)] = {
            "facet_id": canonical_id, "confidence": float(conf),
            "source": ADJUDICATOR_SOURCE,
            "provenance": {"source": ADJUDICATOR_SOURCE, "term": term,
                           "prompt_version": prompt_version, "model": model},
        }

    for g in response.get("genres", []):
        term = g["term"]
        conf = g.get("confidence", response.get("overall_confidence", 0.0))
        cls = classify_layered_term(taxonomy, term, context_terms=proposed)
        if cls.term_kind in ("reject", "review") or cls.canonical_id is None:
            skipped.append(term)
            continue
        if cls.term_kind == "facet":
            _facet(cls.canonical_id, conf, term)
            continue
        # alias term_kind: route to genre or facet based on canonical target
        if cls.term_kind == "alias":
            # check if the alias resolves to a facet
            if taxonomy.facet_by_id(cls.canonical_id) is not None:
                _facet(cls.canonical_id, conf, term)
                continue
            # falls through to genre handling below
        genre = taxonomy.genre_by_id(cls.canonical_id)
        if genre is None:
            skipped.append(term)
            continue
        if cls.term_kind == "family":
            _add_genre(genre_rows, genre, "inferred_family", conf,
                       prompt_version=prompt_version, model=model, term=term)
            continue
        _add_genre(genre_rows, genre, "observed_leaf", conf,
                   prompt_version=prompt_version, model=model, term=term)
        for parent in taxonomy.parents_for_genre(genre.genre_id):
            _add_genre(genre_rows, parent, "inferred_parent", conf,
                       prompt_version=prompt_version, model=model, term=term)
        for family in taxonomy.families_for_genre(genre.genre_id):
            _add_genre(genre_rows, family, "inferred_family", conf,
                       prompt_version=prompt_version, model=model, term=term)

    for f in response.get("facets", []):
        raw = f.get("term", "")
        atoms = [a.strip() for a in raw.split(",") if a.strip()] or [raw]
        for term in atoms:
            cls = classify_layered_term(taxonomy, term)
            if cls.term_kind in ("facet", "alias") and cls.canonical_id and taxonomy.facet_by_id(cls.canonical_id):
                _facet(cls.canonical_id, response.get("overall_confidence", 0.8), term)

    return list(genre_rows.values()), list(facet_rows.values()), skipped


def materialize_adjudication(
    store: Any,
    *,
    album_id: str,
    artist: str,
    album: str,
    response: dict[str, Any],
    taxonomy: LayeredTaxonomy,
    prompt_version: str,
    model: str,
) -> AdjudicationMaterializeSummary:
    release_id = make_release_key(artist, album)
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        response, taxonomy, prompt_version=prompt_version, model=model)
    store.replace_layered_assignments_for_release(
        release_id=release_id, artist=artist, album=album,
        genre_assignments=genre_rows, facet_assignments=facet_rows,
    )
    observed = tuple(sorted(
        taxonomy.genre_by_id(r["genre_id"]).name  # type: ignore[union-attr]
        for r in genre_rows if r["assignment_layer"] == "observed_leaf"
    ))
    inferred = sum(1 for r in genre_rows if r["assignment_layer"] != "observed_leaf")
    return AdjudicationMaterializeSummary(
        release_id=release_id, observed_leaf=observed,
        inferred_count=inferred, facet_count=len(facet_rows), skipped_terms=tuple(skipped),
    )
