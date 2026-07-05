"""Tests for the additive publish-backfill planner/applier (zero-touch M1, Task 5).

The hard invariant under test: re-running the *current* fusion policy against
a release that already has published rows may only ADD rows for (genre_id,
assignment_layer) keys not already stored. It must never remove a row, never
lower an existing row's confidence, and never touch facet rows or user
decisions -- the 2026-06-12 lesson (wholesale re-derivation un-decides good
past calls) applied to a policy that only got MORE permissive.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
from src.ai_genre_enrichment.layered_assignment import compute_layered_assignment_rows
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


class _FakeFusionStore:
    """Minimal store exposing only the four hybrid-evidence collector methods
    fuse_release_evidence needs, plus the Task-4 reader. Lets fusion run for
    real without seeding the full sidecar schema (source tags, checks, etc.)."""

    def __init__(self, source_terms, existing):
        self._source_terms = source_terms
        self._existing = existing

    def hybrid_source_terms_for_release(self, release_key):
        return self._source_terms

    def accepted_enriched_genres_for_release(self, release_key):
        return []

    def latest_check_suggestions_for_release(self, release_key):
        return []

    def latest_model_prior_terms_for_release(self, release_key):
        return []

    def layered_assignment_rows_for_release(self, release_key):
        return self._existing


def _release(release_key: str, artist: str, album: str) -> SimpleNamespace:
    return SimpleNamespace(
        release_key=release_key,
        normalized_artist=artist,
        normalized_album=album,
        album_id="alb1",
        existing_genres_by_source={},
    )


def _lastfm_term(term: str) -> dict:
    return {
        "term": term,
        "source_type": "lastfm_tags",
        "confidence": 0.9,
        "mapping_status": "mapped",
        "canonical_slug": term,
    }


def test_plan_is_additive_only_and_targets_new_terms():
    from src.ai_genre_enrichment.publish_backfill import plan_release_backfill

    taxonomy = load_default_layered_taxonomy()
    release = _release("duster::stratosphere", "duster", "stratosphere")

    # Seed "already published" state: release already has observed_leaf
    # 'slowcore' @0.9. Materialized through the same compute_layered_assignment_rows
    # the backfill re-runs, so the seed can't drift from what a fresh
    # computation would produce for this exact term.
    seed_report = HybridGenreReport(
        release_key=release.release_key,
        accepted_genres=[
            FusedGenreDecision(
                term="slowcore",
                confidence=0.9,
                basis="local_metadata+taxonomy",
                sources=["local_metadata"],
                reason="User-curated local file tag retained.",
            )
        ],
        provisional_genres=[],
        rejected_noise=[],
    )
    existing = compute_layered_assignment_rows(seed_report, taxonomy)
    existing_keys = {(row["genre_id"], row["assignment_layer"]) for row in existing["genre_rows"]}
    assert ("slowcore", "observed_leaf") in existing_keys  # sanity: seed landed

    # Fresh evidence: 'slowcore' (lastfm) + 'dub techno' (lastfm, canonical in
    # the real taxonomy). Both are lastfm-only -> fuse to provisional @<=0.40.
    store = _FakeFusionStore(
        source_terms=[_lastfm_term("slowcore"), _lastfm_term("dub techno")],
        existing={"genre_rows": existing["genre_rows"], "facet_rows": existing["facet_rows"]},
    )

    plan = plan_release_backfill(store, taxonomy=taxonomy, release=release)

    addition_keys = {(row["genre_id"], row["assignment_layer"]) for row in plan.additions}
    # Invariant: no addition ever repeats an already-stored (genre_id, layer) key.
    assert addition_keys.isdisjoint(existing_keys)
    # 'slowcore' itself (already published, at higher confidence) is untouched.
    assert all(row["genre_id"] != "slowcore" for row in plan.additions)

    # 'dub techno' is new -> its observed_leaf row (+ inferred ancestors) is added.
    new_leaves = [row for row in plan.additions if row["assignment_layer"] == "observed_leaf"]
    assert len(new_leaves) == 1
    assert new_leaves[0]["genre_id"] == "dub_techno"
    assert new_leaves[0]["confidence"] <= 0.40  # lastfm-only publish cap (zero-touch M1)

    # added_observed_terms mirrors only the observed_leaf additions.
    assert len(plan.added_observed_terms) == 1
    added = plan.added_observed_terms[0]
    assert added["genre_id"] == "dub_techno"
    assert added["term"] == "dub techno"
    assert added["confidence"] <= 0.40
    assert added["basis"] == "lastfm_only"
    assert added["sources"] == ["lastfm_tags"]


def test_plan_skips_release_with_no_new_terms():
    from src.ai_genre_enrichment.publish_backfill import plan_release_backfill

    taxonomy = load_default_layered_taxonomy()
    release = _release("duster::stratosphere", "duster", "stratosphere")

    # Evidence fuses to exactly the decision that is already published
    # (same term, same lastfm-only capped confidence) -> nothing new to add.
    seed_report = HybridGenreReport(
        release_key=release.release_key,
        accepted_genres=[
            FusedGenreDecision(
                term="slowcore",
                confidence=0.40,
                basis="lastfm_only",
                sources=["lastfm_tags"],
                reason="Last.fm-only mapped signal published at capped confidence pending corroboration.",
            )
        ],
        provisional_genres=[],
        rejected_noise=[],
    )
    existing = compute_layered_assignment_rows(seed_report, taxonomy)

    store = _FakeFusionStore(
        source_terms=[_lastfm_term("slowcore")],
        existing={"genre_rows": existing["genre_rows"], "facet_rows": existing["facet_rows"]},
    )

    plan = plan_release_backfill(store, taxonomy=taxonomy, release=release)

    assert plan.additions == []
    assert plan.added_observed_terms == []


def test_apply_merges_without_removing_or_lowering(tmp_path):
    from src.ai_genre_enrichment.publish_backfill import (
        ReleaseBackfillPlan,
        apply_release_backfill,
        plan_release_backfill,
    )

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    release = _release("duster::stratosphere", "duster", "stratosphere")

    existing_genre_rows = [
        {
            "genre_id": "slowcore",
            "assignment_layer": "observed_leaf",
            "confidence": 0.9,
            "source_reliability": 0.70,
            "evidence_count": 1,
            "rejected_by_user": False,
            "provenance": {"basis": "local_metadata+taxonomy", "sources": ["local_metadata"]},
        },
        {
            "genre_id": "rock",
            "assignment_layer": "inferred_family",
            "confidence": 0.9,
            "source_reliability": 0.70,
            "evidence_count": 1,
            "rejected_by_user": False,
            "provenance": {"basis": "local_metadata+taxonomy", "sources": ["local_metadata"]},
        },
    ]
    existing_facet_rows = [
        {
            "facet_id": "lo_fi",
            "confidence": 0.86,
            "source": "bandcamp_release",
            "provenance": {"basis": "bandcamp_release+taxonomy"},
        },
    ]
    store.replace_layered_assignments_for_release(
        release_id=release.release_key,
        artist=release.normalized_artist,
        album=release.normalized_album,
        genre_assignments=existing_genre_rows,
        facet_assignments=existing_facet_rows,
    )

    addition_row = {
        "genre_id": "dub_techno",
        "assignment_layer": "observed_leaf",
        "confidence": 0.40,
        "source_reliability": 0.25,
        "evidence_count": 1,
        "rejected_by_user": False,
        "provenance": {"basis": "lastfm_only", "sources": ["lastfm_tags"]},
    }
    plan = ReleaseBackfillPlan(
        release_key=release.release_key,
        additions=[addition_row],
        added_observed_terms=[
            {
                "term": "dub techno",
                "genre_id": "dub_techno",
                "confidence": 0.40,
                "basis": "lastfm_only",
                "sources": ["lastfm_tags"],
                "reason": "Last.fm-only mapped signal published at capped confidence pending corroboration.",
            }
        ],
    )

    added = apply_release_backfill(store, release=release, plan=plan)
    assert added == 1

    after = store.layered_assignment_rows_for_release(release.release_key)
    by_key = {(row["genre_id"], row["assignment_layer"]): row for row in after["genre_rows"]}

    # (a) Original rows byte-identical -- confidence (and everything else) unchanged.
    assert by_key[("slowcore", "observed_leaf")] == existing_genre_rows[0]
    assert by_key[("rock", "inferred_family")] == existing_genre_rows[1]
    # (b) New row present.
    assert by_key[("dub_techno", "observed_leaf")]["confidence"] == 0.40
    assert len(after["genre_rows"]) == 3
    # (c) Facet rows untouched.
    assert after["facet_rows"] == existing_facet_rows

    # (d) Idempotent: re-running plan+apply against the now-merged state adds
    # nothing. apply_release_backfill itself does not re-plan (it trusts the
    # plan handed to it), so idempotency across repeat runs is enforced by
    # plan_release_backfill's existing-keys filter -- re-plan, then re-apply,
    # exactly as the Task-6 CLI would on a second run.
    taxonomy = load_default_layered_taxonomy()
    refusion_store = _FakeFusionStore(source_terms=[], existing=after)
    replan = plan_release_backfill(refusion_store, taxonomy=taxonomy, release=release)
    assert replan.additions == []
    assert replan.added_observed_terms == []

    second_added = apply_release_backfill(store, release=release, plan=replan)
    assert second_added == 0

    still_after = store.layered_assignment_rows_for_release(release.release_key)
    assert still_after == after
