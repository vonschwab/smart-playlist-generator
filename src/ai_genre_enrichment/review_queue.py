"""Scan and decision logic for the human genre-review queue.

The queue persists hybrid-evidence review terms (uncertain fusion decisions and
taxonomy-unknown terms) per release in ai_genre_review_queue. Decisions are
written through the existing user-override mechanism, which the publish stage
already applies.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .layered_assignment import build_layered_release_diagnostics

DiagnosticsFn = Callable[..., dict[str, Any]]

_DECISION_TO_STATUS = {"accept": "accepted", "reject": "rejected", "revert": "pending"}


def compute_review_terms(
    store: Any,
    *,
    taxonomy: Any,
    release_key: str,
    diagnostics_fn: DiagnosticsFn = build_layered_release_diagnostics,
) -> list[dict[str, Any]]:
    """Review-term rows for one release, excluding override-settled terms."""
    diag = diagnostics_fn(store, release_id=release_key, taxonomy=taxonomy)
    override = store.get_user_override(release_key) or {"genres_add": [], "genres_remove": []}
    settled = {
        g.casefold()
        for g in list(override["genres_add"]) + list(override["genres_remove"])
    }
    terms: list[dict[str, Any]] = []
    for row in diag.get("review_terms") or []:
        term = str(row.get("term") or "").strip()
        if not term or term.casefold() in settled:
            continue
        terms.append({
            "term": term,
            "confidence": float(row.get("confidence") or 0.0),
            "basis": str(row.get("source_basis") or row.get("basis") or "hybrid_fusion"),
            "sources": sorted(row.get("sources") or []),
            "reason": str(row.get("reason") or ""),
        })
    return terms


def scan_review_queue(
    store: Any,
    *,
    taxonomy: Any,
    diagnostics_fn: DiagnosticsFn = build_layered_release_diagnostics,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], None]] = None,
) -> dict[str, int]:
    """Reconcile the review queue against fresh diagnostics for every release.

    cancel_cb is invoked at release boundaries only; each release's sync is a
    committed transaction, so a cancelled scan keeps its partial results.
    """
    releases = store.list_review_scan_releases()
    total = len(releases)
    summary = {"releases_scanned": 0, "new_terms": 0, "pruned_terms": 0, "pending_terms": 0}
    for i, rel in enumerate(releases):
        if cancel_cb is not None:
            cancel_cb()
        terms = compute_review_terms(
            store, taxonomy=taxonomy, release_key=rel["release_key"],
            diagnostics_fn=diagnostics_fn,
        )
        counts = store.sync_review_queue_for_release(
            release_key=rel["release_key"],
            normalized_artist=rel["normalized_artist"],
            normalized_album=rel["normalized_album"],
            terms=terms,
        )
        summary["releases_scanned"] += 1
        summary["new_terms"] += counts["inserted"]
        summary["pruned_terms"] += counts["pruned"]
        if progress_cb is not None:
            progress_cb(i + 1, total, f"{rel['normalized_artist']} – {rel['normalized_album']}")
    summary["pending_terms"] = store.get_review_queue_page(limit=1)["pending_terms"]
    return summary


def apply_review_decision(
    store: Any,
    *,
    release_key: str,
    term: str,
    decision: str,
) -> dict[str, Any]:
    """Apply accept/reject/revert for one queue row.

    Merges into the release's user override (set_user_override REPLACES the
    row, so we must read-merge-write), re-bakes the enriched signature, and
    updates the queue row status.
    """
    status = _DECISION_TO_STATUS.get(decision)
    if status is None:
        raise ValueError(f"invalid decision: {decision!r} (use accept/reject/revert)")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT normalized_artist, normalized_album FROM ai_genre_review_queue "
            "WHERE release_key = ? AND term = ?",
            (release_key, term),
        ).fetchone()
    if row is None:
        raise ValueError(f"no review queue row for {release_key!r} / {term!r}")

    override = store.get_user_override(release_key) or {"genres_add": [], "genres_remove": []}
    add = {g.casefold() for g in override["genres_add"]}
    remove = {g.casefold() for g in override["genres_remove"]}
    key = term.casefold()
    add.discard(key)
    remove.discard(key)
    if decision == "accept":
        add.add(key)
    elif decision == "reject":
        remove.add(key)

    store.set_user_override(
        release_key=release_key,
        normalized_artist=row["normalized_artist"],
        normalized_album=row["normalized_album"],
        genres_add=sorted(add),
        genres_remove=sorted(remove),
    )
    store.rebuild_enriched_genres_for_release(release_key)
    store.set_review_queue_status(release_key=release_key, term=term, status=status)
    return {"release_key": release_key, "term": term, "decision": decision, "status": status}
