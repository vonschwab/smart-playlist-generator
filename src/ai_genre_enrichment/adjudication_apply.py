"""Deterministic apply: checkpoint best-results -> sidecar (non-escalated) + queue (escalated).

No LLM calls. Idempotent (materialize is replace-by-release-key). Safe to re-run after a
taxonomy-growth pass to pick up new canonical mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adjudication_materializer import materialize_adjudication
from .album_adjudicator import canonicalize_proposed
from .album_evidence import build_evidence
from .normalization import normalize_release_artist, normalize_release_name


def best_results(rows, *, thorough_pv) -> dict:
    best: dict = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


@dataclass
class ApplySummary:
    materialized: int
    escalated: int


def apply_adjudications(*, rows, thorough_pv, std_pv, meta_conn, id2name, taxonomy, adapter,
                        sidecar_store, queue, model: str = "sonnet") -> ApplySummary:
    best = best_results(rows, thorough_pv=thorough_pv)
    materialized = 0
    escalated = 0
    for album_id, resp in best.items():
        ev = build_evidence(meta_conn, album_id, id2name)
        if resp.get("escalate"):
            canon = canonicalize_proposed(
                [g["term"] for g in resp.get("genres", [])], adapter.canonicalize_tag)["canonical"]
            release_key = f"{normalize_release_artist(ev['artist'])}::{normalize_release_name(ev['album'])}"
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
    return ApplySummary(materialized=materialized, escalated=escalated)
