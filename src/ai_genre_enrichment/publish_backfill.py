"""Additive backfill for the 2026-07-04 always-publish policy flip (zero-touch M1).

Re-runs the current fusion policy per release and merges ONLY new assignment
rows. Never removes a row, never lowers a confidence, never touches
rejected_by_user or user overrides -- the 2026-06-12 lesson (wholesale
re-derivation un-decides good past calls; see assignment_migration.py) applied
to a policy that only got MORE permissive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hybrid_evidence import fuse_release_evidence
from .layered_assignment import compute_layered_assignment_rows


@dataclass(frozen=True)
class ReleaseBackfillPlan:
    release_key: str
    additions: list[dict[str, Any]] = field(default_factory=list)
    added_observed_terms: list[dict[str, Any]] = field(default_factory=list)


def plan_release_backfill(store: Any, *, taxonomy: Any, release: Any) -> ReleaseBackfillPlan:
    report = fuse_release_evidence(store, release)
    computed = compute_layered_assignment_rows(report, taxonomy)
    existing = store.layered_assignment_rows_for_release(release.release_key)
    existing_keys = {
        (row["genre_id"], row["assignment_layer"]) for row in existing["genre_rows"]
    }
    additions = [
        row
        for row in computed["genre_rows"]
        if (row["genre_id"], row["assignment_layer"]) not in existing_keys
    ]
    added_observed_terms = [
        {
            "term": row["provenance"].get("term", ""),
            "genre_id": row["genre_id"],
            "confidence": row["confidence"],
            "basis": row["provenance"].get("basis", ""),
            "sources": row["provenance"].get("sources", []),
            "reason": row["provenance"].get("reason", ""),
        }
        for row in additions
        if row["assignment_layer"] == "observed_leaf"
    ]
    return ReleaseBackfillPlan(
        release_key=release.release_key,
        additions=additions,
        added_observed_terms=added_observed_terms,
    )


def apply_release_backfill(store: Any, *, release: Any, plan: ReleaseBackfillPlan) -> int:
    if not plan.additions:
        return 0
    existing = store.layered_assignment_rows_for_release(release.release_key)
    # Self-guard against a stale plan: re-filter against a FRESH read of what's
    # actually stored right now (same (genre_id, assignment_layer) set the
    # planner uses), so applying the same plan object twice -- e.g. a caller
    # re-applying a dry-run report whose additions already landed -- is a
    # natural no-op instead of a duplicate-PK IntegrityError.
    existing_keys = {
        (row["genre_id"], row["assignment_layer"]) for row in existing["genre_rows"]
    }
    fresh_additions = [
        row
        for row in plan.additions
        if (row["genre_id"], row["assignment_layer"]) not in existing_keys
    ]
    if not fresh_additions:
        return 0
    store.replace_layered_assignments_for_release(
        release_id=release.release_key,
        artist=release.normalized_artist,
        album=release.normalized_album,
        genre_assignments=existing["genre_rows"] + fresh_additions,
        facet_assignments=existing["facet_rows"],
    )
    return len(fresh_additions)
