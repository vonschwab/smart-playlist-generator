"""Tests for the surgical delta migration of legacy graph assignments.

Context (2026-06-12): wholesale re-derivation through the strict fusion policy
un-decides good past decisions (Duster's lastfm-only "dream pop" is correct).
The selection functions decide WHICH releases the migration re-materializes
through the real pipeline, grandfathering everything else:
  Delta A: add taxonomy-leaf local file tags missing from observed.
  Delta B: remove observed leaves whose entire evidence is the storefront echo
           chamber {bandcamp_label, bandcamp_unknown, ai_enriched_accepted}.
  Delta C: remove lastfm-only observed leaves on a wrong-identity artist.
"""
from __future__ import annotations

from src.ai_genre_enrichment.assignment_migration import (
    ObservedTermEvidence,
    select_local_additions,
    select_misidentified_lastfm_removals,
    select_storefront_removals,
)


class TestSelectStorefrontRemovals:
    def test_storefront_only_evidence_is_removed(self):
        terms = [
            ObservedTermEvidence(
                genre_id="indie_rock",
                evidence_sources=frozenset({"bandcamp_unknown", "ai_enriched_accepted"}),
            ),
        ]
        assert select_storefront_removals(terms) == ["indie_rock"]

    def test_any_independent_source_protects_the_term(self):
        terms = [
            ObservedTermEvidence(  # Duster: lastfm-only AND correct — keep
                genre_id="dream_pop",
                evidence_sources=frozenset({"lastfm_tags"}),
            ),
            ObservedTermEvidence(  # storefront + local — keep
                genre_id="slowcore",
                evidence_sources=frozenset({"bandcamp_label", "local_metadata"}),
            ),
            ObservedTermEvidence(  # artist-run bandcamp is self-reported — keep
                genre_id="shoegaze",
                evidence_sources=frozenset({"bandcamp_artist"}),
            ),
            ObservedTermEvidence(  # storefront + musicbrainz — keep
                genre_id="post_punk",
                evidence_sources=frozenset({"bandcamp_unknown", "musicbrainz"}),
            ),
        ]
        assert select_storefront_removals(terms) == []

    def test_no_attributable_evidence_is_left_alone(self):
        # Conservative: a term we cannot attribute (older flows, evidence not
        # in the sidecar) is grandfathered, not removed.
        terms = [
            ObservedTermEvidence(genre_id="krautrock", evidence_sources=frozenset()),
        ]
        assert select_storefront_removals(terms) == []

    def test_user_added_terms_are_never_removed(self):
        terms = [
            ObservedTermEvidence(
                genre_id="heartland_rock",
                evidence_sources=frozenset({"bandcamp_label"}),
                user_added=True,
            ),
        ]
        assert select_storefront_removals(terms) == []


class TestSelectMisidentifiedLastfmRemovals:
    def test_lastfm_only_term_removed_when_artist_contradicted(self):
        # Green-House: lastfm name-collision (Ukrainian hip-hop) contradicted by
        # the artist's own bandcamp (ambient). lastfm-only observed -> remove.
        terms = [
            ObservedTermEvidence(
                genre_id="underground_hip_hop",
                evidence_sources=frozenset({"lastfm_tags"}),
            ),
            ObservedTermEvidence(
                genre_id="ambient",
                evidence_sources=frozenset({"bandcamp_artist"}),
            ),
        ]
        assert select_misidentified_lastfm_removals(
            terms, artist_lastfm_contradicted=True
        ) == ["underground_hip_hop"]

    def test_no_contradiction_grandfathers_lastfm(self):
        # Duster: dream pop is lastfm-only but the artist's lastfm is coherent
        # with its other evidence — never second-guess it.
        terms = [
            ObservedTermEvidence(
                genre_id="dream_pop", evidence_sources=frozenset({"lastfm_tags"})
            ),
        ]
        assert select_misidentified_lastfm_removals(
            terms, artist_lastfm_contradicted=False
        ) == []

    def test_corroborated_term_is_protected_even_when_contradicted(self):
        # A genre with a non-lastfm source is not a lastfm-identity artifact.
        terms = [
            ObservedTermEvidence(
                genre_id="ambient",
                evidence_sources=frozenset({"lastfm_tags", "bandcamp_artist"}),
            ),
        ]
        assert select_misidentified_lastfm_removals(
            terms, artist_lastfm_contradicted=True
        ) == []

    def test_user_added_lastfm_term_never_removed(self):
        terms = [
            ObservedTermEvidence(
                genre_id="hip_hop",
                evidence_sources=frozenset({"lastfm_tags"}),
                user_added=True,
            ),
        ]
        assert select_misidentified_lastfm_removals(
            terms, artist_lastfm_contradicted=True
        ) == []


class TestPlanReleaseDelta:
    def test_greenhouse_lastfm_removed_even_when_it_empties(self):
        # Delta C is unconditional: a wrong-identity lastfm tag goes even if it
        # leaves zero observed leaves (the ambient inferred-family survives in
        # the row set; sonic-only beats a hip-hop mislabel).
        from src.ai_genre_enrichment.assignment_migration import plan_release_delta

        additions, removals = plan_release_delta(
            observed_terms=[
                ObservedTermEvidence(
                    genre_id="hip_hop", evidence_sources=frozenset({"lastfm_tags"})
                ),
                ObservedTermEvidence(
                    genre_id="underground_hip_hop",
                    evidence_sources=frozenset({"lastfm_tags"}),
                ),
            ],
            local_leaf_genre_ids=set(),
            has_local_tags=False,
            artist_lastfm_contradicted=True,
        )
        assert additions == []
        assert removals == ["hip_hop", "underground_hip_hop"]

    def test_lpvv_adds_local_and_drops_storefront(self):
        from src.ai_genre_enrichment.assignment_migration import plan_release_delta

        additions, removals = plan_release_delta(
            observed_terms=[
                ObservedTermEvidence(
                    genre_id="indie_rock",
                    evidence_sources=frozenset({"bandcamp_unknown", "ai_enriched_accepted"}),
                ),
            ],
            local_leaf_genre_ids={"hardcore_punk", "post_punk"},
            has_local_tags=True,
        )
        assert additions == ["hardcore_punk", "post_punk"]
        assert removals == ["indie_rock"]

    def test_storefront_grandfathered_without_local_tags(self):
        # Nigeria 70 reissue-label class: storefront is the only evidence, no
        # local curation to take over -> never strip.
        from src.ai_genre_enrichment.assignment_migration import plan_release_delta

        additions, removals = plan_release_delta(
            observed_terms=[
                ObservedTermEvidence(
                    genre_id="juju",
                    evidence_sources=frozenset({"bandcamp_label", "ai_enriched_accepted"}),
                ),
            ],
            local_leaf_genre_ids=set(),
            has_local_tags=False,
        )
        assert (additions, removals) == ([], [])

    def test_storefront_removal_guarded_against_emptying(self):
        # Storefront removal alone would empty the release (local tag maps only
        # to a family, no leaf addition) -> grandfather the storefront tag.
        from src.ai_genre_enrichment.assignment_migration import plan_release_delta

        additions, removals = plan_release_delta(
            observed_terms=[
                ObservedTermEvidence(
                    genre_id="indie_rock",
                    evidence_sources=frozenset({"bandcamp_label"}),
                ),
            ],
            local_leaf_genre_ids=set(),
            has_local_tags=True,
        )
        assert (additions, removals) == ([], [])

    def test_addition_only_release(self):
        from src.ai_genre_enrichment.assignment_migration import plan_release_delta

        additions, removals = plan_release_delta(
            observed_terms=[
                ObservedTermEvidence(
                    genre_id="slowcore", evidence_sources=frozenset({"local_metadata"})
                ),
            ],
            local_leaf_genre_ids={"slowcore", "dream_pop"},
            has_local_tags=True,
        )
        assert additions == ["dream_pop"]
        assert removals == []


class _FakeTaxonomy:
    """genre_by_id / parents_for_genre / families_for_genre stub."""

    from types import SimpleNamespace as _NS

    def __init__(self, parents=None, families=None, known=None):
        self._parents = parents or {}
        self._families = families or {}
        self._known = known if known is not None else set(self._parents) | set(self._families)

    def genre_by_id(self, genre_id):
        if genre_id in self._known:
            return self._NS(genre_id=genre_id, name=genre_id.replace("_", " "))
        return None

    def parents_for_genre(self, genre_id):
        return [self._NS(genre_id=g, name=g) for g in self._parents.get(genre_id, [])]

    def families_for_genre(self, genre_id):
        return [self._NS(genre_id=g, name=g) for g in self._families.get(genre_id, [])]


def _row(genre_id, layer="observed_leaf", confidence=0.9):
    return {
        "genre_id": genre_id,
        "assignment_layer": layer,
        "confidence": confidence,
        "source_reliability": 0.7,
        "evidence_count": 1,
        "rejected_by_user": False,
        "provenance": {"basis": "existing"},
    }


class TestApplySurgicalDelta:
    def test_removal_drops_observed_and_inferred_of_that_genre(self):
        # Green-House: drop hip_hop (observed + inferred) but KEEP ambient.
        from src.ai_genre_enrichment.assignment_migration import apply_surgical_delta

        existing = [
            _row("hip_hop", "observed_leaf"),
            _row("hip_hop", "inferred_family"),
            _row("underground_hip_hop", "observed_leaf"),
            _row("ambient", "inferred_family"),
        ]
        out = apply_surgical_delta(
            existing,
            additions=[],
            removals=["hip_hop", "underground_hip_hop"],
            taxonomy=_FakeTaxonomy(),
        )
        keys = {(r["genre_id"], r["assignment_layer"]) for r in out}
        assert keys == {("ambient", "inferred_family")}

    def test_addition_inserts_leaf_and_ancestors_preserving_existing(self):
        from src.ai_genre_enrichment.assignment_migration import apply_surgical_delta

        tax = _FakeTaxonomy(
            parents={"hardcore_punk": ["punk_rock"]},
            families={"hardcore_punk": ["punk"]},
            known={"hardcore_punk", "punk_rock", "punk"},
        )
        existing = [_row("dream_pop", "observed_leaf")]
        out = apply_surgical_delta(
            existing, additions=["hardcore_punk"], removals=[], taxonomy=tax
        )
        keys = {(r["genre_id"], r["assignment_layer"]) for r in out}
        assert keys == {
            ("dream_pop", "observed_leaf"),       # preserved, untouched
            ("hardcore_punk", "observed_leaf"),   # added
            ("punk_rock", "inferred_parent"),
            ("punk", "inferred_family"),
        }

    def test_addition_does_not_duplicate_existing_genre(self):
        from src.ai_genre_enrichment.assignment_migration import apply_surgical_delta

        existing = [_row("slowcore", "observed_leaf", confidence=0.95)]
        out = apply_surgical_delta(
            existing, additions=["slowcore"], removals=[], taxonomy=_FakeTaxonomy(known={"slowcore"})
        )
        slow = [r for r in out if r["genre_id"] == "slowcore"]
        assert len(slow) == 1
        assert slow[0]["confidence"] == 0.95  # keeps the stronger existing row


class TestSelectLocalAdditions:
    def test_missing_local_leaves_are_added_sorted(self):
        adds = select_local_additions(
            {"hardcore_punk", "post_punk", "indie_rock"},
            observed_genre_ids={"indie_rock"},
        )
        assert adds == ["hardcore_punk", "post_punk"]

    def test_already_observed_yields_nothing(self):
        assert select_local_additions(
            {"slowcore"}, observed_genre_ids={"slowcore"}
        ) == []
