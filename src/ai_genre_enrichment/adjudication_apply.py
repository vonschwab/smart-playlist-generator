"""Deterministic apply: checkpoint best-results -> sidecar (non-escalated) + queue (escalated).

No LLM calls. Idempotent (materialize is replace-by-release-key). Safe to re-run after a
taxonomy-growth pass to pick up new canonical mappings.
"""
from __future__ import annotations

from dataclasses import dataclass

from .adjudication_materializer import materialize_adjudication
from .album_adjudicator import canonicalize_proposed
from .album_evidence import build_evidence
from .normalization import normalize_release_artist, normalize_release_name


def best_results(rows, *, thorough_pv) -> dict[str, dict]:
    """Return the best adjudication response per album_id.

    The thorough-pv response wins over any standard-pv response for the same
    album.  Exactly one result per album is expected; if duplicate thorough rows
    exist for the same album the last one seen wins.
    """
    best: dict = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


# Escalated albums whose proposed genres are materialized as a provisional fallback get
# their confidence scaled by this factor — present enough to beat legacy raw tags, but
# weighted below a confirmed adjudication and still flagged in the review queue.
PROVISIONAL_CONFIDENCE_SCALE = 0.6


@dataclass
class ApplySummary:
    materialized: int
    escalated: int
    provisional: int = 0


def apply_adjudications(*, rows, thorough_pv, std_pv, meta_conn, id2name, taxonomy, adapter,
                        sidecar_store, queue, model: str = "sonnet") -> ApplySummary:
    best = best_results(rows, thorough_pv=thorough_pv)
    materialized = 0
    escalated = 0
    provisional = 0
    for album_id, resp in best.items():
        ev = build_evidence(meta_conn, album_id, id2name)
        if resp.get("escalate"):
            proposed = resp.get("genres", [])
            release_key = f"{normalize_release_artist(ev['artist'])}::{normalize_release_name(ev['album'])}"
            # Provisional fallback: an escalated album with proposed genres but NO prior
            # assignment is materialized at reduced confidence so it is never worse than
            # legacy, while staying queued for human confirmation. Never clobbers an
            # existing (confirmed) assignment.
            if proposed and not sidecar_store.has_genre_assignments(release_key):
                provisional_resp = {
                    **resp,
                    "genres": [
                        {**g, "confidence": float(g.get("confidence", 0.0)) * PROVISIONAL_CONFIDENCE_SCALE}
                        for g in proposed
                    ],
                }
                materialize_adjudication(
                    sidecar_store, album_id=album_id, artist=ev["artist"], album=ev["album"],
                    response=provisional_resp, taxonomy=taxonomy,
                    prompt_version=std_pv, model="provisional",
                )
                provisional += 1
            canon = canonicalize_proposed(
                [g["term"] for g in proposed], adapter.canonicalize_tag)["canonical"]
            queue.enqueue(
                album_id=album_id, release_key=release_key, artist=ev["artist"], album=ev["album"],
                prior_observed_leaf=ev["current_observed_leaf"],
                proposed_genres=[{"term": t, "confidence": 0.8} for t in canon],
                escalate_reason=resp.get("escalate_reason", ""),
                dropped_file_tags=resp.get("dropped_file_tags", []),
                prompt_version=std_pv, model=model, input_hash=resp.get("input_hash", ""),
            )
            escalated += 1
            continue
        materialize_adjudication(
            sidecar_store, album_id=album_id, artist=ev["artist"], album=ev["album"],
            response=resp, taxonomy=taxonomy, prompt_version=std_pv, model=model,
        )
        materialized += 1
    return ApplySummary(materialized=materialized, escalated=escalated, provisional=provisional)
