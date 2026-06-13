"""Surgical delta migration for legacy graph release assignments.

Why this exists (2026-06-12): re-deriving the whole library through the strict
fusion policy un-decides good past decisions along with the bad ones — much of
the published authority's evidence ancestry is Last.fm tags or graduated
acceptances that the strict policy now (correctly, for NEW decisions) refuses
to auto-accept, yet the resulting tags are right (Duster's "dream pop" is
lastfm-only AND true). Both wholesale re-materializations were rejected at the
publish dry-run gate.

So the legacy state transition is delta-based, and everything else is
grandfathered:

  Delta A (additive):    taxonomy-leaf local file tags missing from a covered
                         release's observed set are added — the never-drop
                         class (VV Torso's hardcore/post-punk/punk).
  Delta B (subtractive): observed leaves whose entire attributable evidence is
                         the storefront echo chamber {bandcamp_label,
                         bandcamp_unknown, ai_enriched_accepted} are removed —
                         the LPVV "indie rock" class. Any independent source
                         (local, MusicBrainz, Discogs, Last.fm, artist-run
                         bandcamp, web check) protects the term; terms with no
                         attributable evidence are left alone; user-added
                         terms are never removed.

The strict fusion policy in `hybrid_evidence.py` governs newly collected
evidence going forward; this module is the one-time bridge for state created
under the old policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STOREFRONT_ONLY_SOURCES = frozenset(
    {"bandcamp_label", "bandcamp_unknown", "ai_enriched_accepted"}
)


@dataclass(frozen=True)
class ObservedTermEvidence:
    """An observed_leaf assignment plus every evidence source attributable to it."""

    genre_id: str
    evidence_sources: frozenset
    user_added: bool = False


def select_storefront_removals(terms: list[ObservedTermEvidence]) -> list[str]:
    """Delta B: genre_ids whose entire attributable evidence is storefront-class."""
    return [
        term.genre_id
        for term in terms
        if term.evidence_sources
        and term.evidence_sources <= STOREFRONT_ONLY_SOURCES
        and not term.user_added
    ]


def select_local_additions(
    local_leaf_genre_ids: set[str], *, observed_genre_ids: set[str]
) -> list[str]:
    """Delta A: taxonomy-leaf local-tag genre_ids missing from observed."""
    return sorted(set(local_leaf_genre_ids) - set(observed_genre_ids))


def select_misidentified_lastfm_removals(
    terms: list[ObservedTermEvidence], *, artist_lastfm_contradicted: bool
) -> list[str]:
    """Delta C: lastfm-only observed leaves on a wrong-identity artist.

    Last.fm is matched by artist-name string, so a name collision imports a
    different act's genres (Green-House the LA ambient project -> a Ukrainian
    "green house" hip-hop act). We only strip a lastfm-only tag when the
    artist's Last.fm tag set is *contradicted* — i.e. shares nothing with the
    artist's other evidence, which itself must be substantive (the caller's
    bar). A genre with any non-lastfm source, or a user-added one, is never an
    identity artifact and is kept.
    """
    if not artist_lastfm_contradicted:
        return []
    return [
        term.genre_id
        for term in terms
        if term.evidence_sources == frozenset({"lastfm_tags"})
        and not term.user_added
    ]


def plan_release_delta(
    *,
    observed_terms: list[ObservedTermEvidence],
    local_leaf_genre_ids: set[str],
    has_local_tags: bool,
    artist_lastfm_contradicted: bool = False,
) -> tuple[list[str], list[str]]:
    """Compute the surgical (additions, removals) of observed-leaf genre_ids.

    additions: missing taxonomy-leaf local file tags (Delta A, purely additive).
    removals:
      - Delta C (lastfm misidentification): UNCONDITIONAL — a wrong-identity tag
        goes even if it empties the observed set (the release keeps its other
        layers; sonic-only beats a mislabel).
      - Delta B (storefront): only when the user has local tags to take over,
        and guarded so storefront removal alone never empties the observed set
        (reissue-label releases without local curation are grandfathered).

    The caller applies exactly these to the observed-leaf rows — no full
    re-materialization, so correct non-target tags (Beach Fossils' dream pop)
    are never collateral.
    """
    observed_ids = {term.genre_id for term in observed_terms}
    additions = select_local_additions(
        local_leaf_genre_ids, observed_genre_ids=observed_ids
    )
    removals: set[str] = set(
        select_misidentified_lastfm_removals(
            observed_terms, artist_lastfm_contradicted=artist_lastfm_contradicted
        )
    )
    if has_local_tags:
        storefront = set(select_storefront_removals(observed_terms))
        # Guard storefront removal (only) against emptying the observed set.
        surviving = (observed_ids - removals - storefront) | set(additions)
        if surviving:
            removals |= storefront
    return additions, sorted(removals)


def apply_surgical_delta(
    existing_rows: list[dict[str, Any]],
    *,
    additions: list[str],
    removals: list[str],
    taxonomy: Any,
) -> list[dict[str, Any]]:
    """Apply observed-leaf additions/removals to a release's existing row set.

    Removals drop every row (observed AND inferred) whose genre_id is removed,
    so a dropped leaf takes its self-genre inferred chip with it (Green-House's
    hip_hop family chip) while unrelated inferred rows (ambient) are preserved.
    Additions insert an observed leaf plus its taxonomy parents/families,
    merged by (genre_id, layer) keeping the higher confidence. Everything not
    named is left exactly as-is — no re-derivation, no collateral.
    """
    remove = set(removals)
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (r["genre_id"], r["assignment_layer"]): r
        for r in existing_rows
        if r["genre_id"] not in remove
    }

    def _merge(genre_id: str, layer: str, confidence: float, provenance: dict) -> None:
        key = (genre_id, layer)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = {
                "genre_id": genre_id,
                "assignment_layer": layer,
                "confidence": confidence,
                "source_reliability": 0.80,
                "evidence_count": 1,
                "rejected_by_user": False,
                "provenance": provenance,
            }
        elif confidence > existing["confidence"]:
            existing["confidence"] = confidence

    for gid in additions:
        prov = {
            "basis": "local_metadata+delta_migration",
            "sources": ["local_metadata"],
            "reason": "User-curated local file tag restored by the 2026-06-12 delta migration.",
        }
        _merge(gid, "observed_leaf", 0.85, prov)
        for parent in taxonomy.parents_for_genre(gid):
            _merge(parent.genre_id, "inferred_parent", 0.85 * 0.5, prov)
        for family in taxonomy.families_for_genre(gid):
            _merge(family.genre_id, "inferred_family", 0.85 * 0.5, prov)

    return list(by_key.values())
