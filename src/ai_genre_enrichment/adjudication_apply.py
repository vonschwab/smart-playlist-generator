"""Deterministic apply: checkpoint best-results -> sidecar (non-escalated) + queue (escalated).

No LLM calls. Idempotent (materialize is replace-by-release-key). Safe to re-run after a
taxonomy-growth pass to pick up new canonical mappings.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from .adjudication_materializer import materialize_adjudication
from .album_adjudicator import canonicalize_proposed
from .album_evidence import build_evidence
from .normalization import make_release_key

logger = logging.getLogger(__name__)


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
    skipped_orphan: int = 0


def prune_orphaned_adjudications(
    enrichment_conn: sqlite3.Connection, meta_conn: sqlite3.Connection
) -> int:
    """Cascade album deletion into the adjudication cache.

    Deletes every ``adjudications`` row whose ``album_id`` no longer has a matching
    row in ``albums`` (the album was pruned by scan orphan-cleanup). This keeps the
    apply stage from re-processing — and choking on — a dead album's cached
    adjudication on every subsequent run.

    Scoped to the ``adjudications`` cache only. The ``genre_graph_release_*``
    assignments are keyed by ``release_key`` with a 1:many album fan-out, so a dead
    album can share a release with a live one; pruning those here could delete a
    live album's genres, so it is intentionally left to publish. Returns the number
    of distinct orphaned albums removed.
    """
    if not enrichment_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='adjudications'"
    ).fetchone():
        return 0
    live = {r[0] for r in meta_conn.execute("SELECT album_id FROM albums")}
    orphaned = sorted(
        aid for (aid,) in enrichment_conn.execute("SELECT DISTINCT album_id FROM adjudications")
        if aid not in live
    )
    if orphaned:
        enrichment_conn.executemany(
            "DELETE FROM adjudications WHERE album_id = ?", [(a,) for a in orphaned]
        )
        enrichment_conn.commit()
        logger.info(
            "Pruned %d orphaned adjudication album(s) (album deleted since adjudication): %s",
            len(orphaned), ", ".join(orphaned[:10]) + (" …" if len(orphaned) > 10 else ""),
        )
    return len(orphaned)


def apply_adjudications(*, rows, thorough_pv, std_pv, meta_conn, id2name, taxonomy, adapter,
                        sidecar_store, queue, model: str = "sonnet") -> ApplySummary:
    best = best_results(rows, thorough_pv=thorough_pv)
    materialized = 0
    escalated = 0
    provisional = 0
    skipped_orphan = 0
    for album_id, resp in best.items():
        ev = build_evidence(meta_conn, album_id, id2name)
        if not ev.get("artist"):
            # Album row is gone (orphaned metadata — deleted by scan cleanup after this
            # adjudication was cached). Its artist/album are unresolvable, so there is
            # nothing to materialize; skip loudly rather than crash the whole stage on a
            # NOT-NULL artist insert. The stale row is pruned by prune_orphaned_adjudications.
            logger.warning(
                "apply: skipping adjudication for album_id=%s — no matching `albums` row "
                "(orphaned metadata); nothing to materialize", album_id)
            skipped_orphan += 1
            continue
        if resp.get("escalate"):
            proposed = resp.get("genres", [])
            release_key = make_release_key(ev['artist'], ev['album'])
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
    return ApplySummary(materialized=materialized, escalated=escalated,
                        provisional=provisional, skipped_orphan=skipped_orphan)
