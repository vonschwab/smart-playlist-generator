"""Deterministic fusion policy for hybrid album genre evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DecisionKind = Literal["accepted", "provisional", "rejected_noise", "needs_review"]

SOURCE_WEIGHTS: dict[str, float] = {
    "bandcamp_release": 0.95,
    "official_release": 0.95,
    "discogs": 0.78,
    "musicbrainz": 0.76,
    "local_metadata": 0.70,
    "model_prior": 0.68,
    "lastfm_tags": 0.25,
}

LASTFM_SOURCE_TYPES = {"lastfm_tags", "lastfm"}
STRONG_SOURCE_TYPES = {"bandcamp_release", "official_release"}
MEDIUM_SOURCE_TYPES = {"local_metadata", "discogs", "musicbrainz"}


@dataclass(frozen=True)
class EvidenceTerm:
    term: str
    source_type: str
    confidence: float
    canonical_slug: str | None = None
    mapping_status: str = "mapped"
    notes: str = ""


@dataclass(frozen=True)
class FusedGenreDecision:
    term: str
    confidence: float
    basis: str
    sources: list[str]
    reason: str


@dataclass(frozen=True)
class HybridGenreReport:
    release_key: str
    accepted_genres: list[FusedGenreDecision]
    provisional_genres: list[FusedGenreDecision]
    rejected_noise: list[FusedGenreDecision]
    needs_review: list[FusedGenreDecision]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fuse_hybrid_evidence(
    *,
    release_key: str,
    evidence: list[EvidenceTerm],
    sparse_release: bool,
) -> HybridGenreReport:
    grouped: dict[str, list[EvidenceTerm]] = {}
    for item in evidence:
        term = _decision_term(item)
        if term:
            grouped.setdefault(term, []).append(item)

    accepted: list[FusedGenreDecision] = []
    provisional: list[FusedGenreDecision] = []
    rejected: list[FusedGenreDecision] = []
    review: list[FusedGenreDecision] = []

    for term in sorted(grouped):
        items = grouped[term]
        sources = sorted({item.source_type for item in items})
        score = _score(items)

        if all(source in LASTFM_SOURCE_TYPES for source in sources):
            rejected.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="lastfm_only",
                sources=sources,
                reason="Last.fm-only signal is treated as noisy corroboration, not accepted evidence.",
            ))
            continue

        if any(source in STRONG_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.90),
                basis=_basis(sources),
                sources=sources,
                reason="Strong release-specific source evidence supports this mapped genre.",
            ))
            continue

        if "model_prior" in sources and any(source in MEDIUM_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.78),
                basis=_basis(sources),
                sources=sources,
                reason="Model taxonomy agrees with existing non-Last.fm metadata.",
            ))
            continue

        if sources == ["model_prior"] and sparse_release and score >= 0.58:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="model_prior+taxonomy",
                sources=sources,
                reason="Sparse release has a high-confidence mapped model taxonomy signal.",
            ))
            continue

        review.append(FusedGenreDecision(
            term=term,
            confidence=score,
            basis=_basis(sources),
            sources=sources,
            reason="Evidence is mapped but not strong enough for automatic acceptance.",
        ))

    return HybridGenreReport(
        release_key=release_key,
        accepted_genres=accepted,
        provisional_genres=provisional,
        rejected_noise=rejected,
        needs_review=review,
    )


def _decision_term(item: EvidenceTerm) -> str:
    if item.mapping_status not in {"mapped", "canonical", "alias"}:
        return ""
    return item.canonical_slug or item.term.strip().casefold()


def _score(items: list[EvidenceTerm]) -> float:
    weighted = 0.0
    total_weight = 0.0
    for item in items:
        weight = SOURCE_WEIGHTS.get(item.source_type, 0.40)
        weighted += weight * max(0.0, min(1.0, item.confidence))
        total_weight += weight
    if total_weight == 0:
        return 0.0
    agreement_bonus = min(0.10, 0.03 * max(0, len({item.source_type for item in items}) - 1))
    return min(1.0, (weighted / total_weight) + agreement_bonus)


def _basis(sources: list[str]) -> str:
    ordered = [
        source
        for source in [
            "bandcamp_release",
            "official_release",
            "discogs",
            "musicbrainz",
            "local_metadata",
            "model_prior",
            "lastfm_tags",
        ]
        if source in sources
    ]
    extra = sorted(source for source in sources if source not in ordered)
    return "+".join(ordered + extra + ["taxonomy"])
