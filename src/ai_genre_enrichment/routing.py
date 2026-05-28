from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .discovery import GENERIC_OR_DESCRIPTOR_TAGS, ReleasePayload, count_specific_existing_genres

DESCRIPTOR_ONLY_TAGS = {
    "compilation",
    "demo",
    "female vocalist",
    "instrumental",
    "japanese",
    "live",
    "remastered",
    "soundtrack",
}


class WebMode(StrEnum):
    OFF = "off"
    AUTO = "auto"
    REQUIRED = "required"


class EnrichmentLane(StrEnum):
    SKIP_WELL_TAGGED = "skip_well_tagged"
    NO_WEB_ADJUDICATION = "no_web_adjudication"
    AUTHORITATIVE_SOURCE_ENRICHMENT = "authoritative_source_enrichment"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class RouteDecision:
    lane: EnrichmentLane
    web_mode: WebMode
    reasons: list[str]


def route_release(payload: ReleasePayload, web_mode: WebMode | str) -> RouteDecision:
    mode = WebMode(web_mode)
    reasons: list[str] = []
    existing = {genre.casefold() for genres in payload.existing_genres_by_source.values() for genre in genres}
    specific_count = count_specific_existing_genres(payload)
    has_descriptors = any(tag in DESCRIPTOR_ONLY_TAGS for tag in existing)
    generic_only = bool(existing) and all(tag in GENERIC_OR_DESCRIPTOR_TAGS or tag in DESCRIPTOR_ONLY_TAGS for tag in existing)
    descriptor_only = bool(existing) and all(tag in DESCRIPTOR_ONLY_TAGS for tag in existing)
    has_conflict = _has_material_source_conflict(payload)
    thin_identity = len(payload.track_titles) <= 1 and not payload.identifiers

    if 3 <= specific_count <= 8 and not has_descriptors and not has_conflict:
        return RouteDecision(
            lane=EnrichmentLane.SKIP_WELL_TAGGED,
            web_mode=WebMode.OFF,
            reasons=[f"already has {specific_count} specific usable genres"],
        )

    if not existing:
        reasons.append("no existing genres")
    if generic_only:
        reasons.append("only generic genres")
    if descriptor_only:
        reasons.append("only descriptor tags")
    if has_conflict:
        reasons.append("source genres conflict materially")
    if thin_identity:
        reasons.append("thin release identity evidence")

    wants_web = mode in {WebMode.AUTO, WebMode.REQUIRED} and (
        not existing or generic_only or descriptor_only or has_conflict or thin_identity
    )
    if wants_web:
        return RouteDecision(lane=EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT, web_mode=mode, reasons=reasons)
    if mode == WebMode.REQUIRED:
        return RouteDecision(
            lane=EnrichmentLane.AUTHORITATIVE_SOURCE_ENRICHMENT,
            web_mode=mode,
            reasons=reasons or ["authoritative source lookup required"],
        )
    return RouteDecision(
        lane=EnrichmentLane.NO_WEB_ADJUDICATION,
        web_mode=WebMode.OFF,
        reasons=reasons or ["local metadata appears sufficient for no-web adjudication"],
    )


def _has_material_source_conflict(payload: ReleasePayload) -> bool:
    source_sets = [
        {genre.casefold() for genre in genres if genre and genre != "__EMPTY__"}
        for genres in payload.existing_genres_by_source.values()
    ]
    source_sets = [genres for genres in source_sets if genres]
    if len(source_sets) < 2:
        return False
    union = set().union(*source_sets)
    intersection = set.intersection(*source_sets)
    return bool(union) and not intersection and len(union) >= 4
