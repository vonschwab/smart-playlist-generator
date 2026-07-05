"""Compose the per-generation receipt (GUI dials spec 2026-07-04).

Honor + confess: numbers come only from stats the run actually produced;
notes state every limitation/intervention explicitly, in dial vocabulary.
"""
from __future__ import annotations

from typing import Any, Optional


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def compose_receipt(playlist_stats: dict, pool_stats: dict) -> dict:
    playlist_stats = playlist_stats or {}
    pool_stats = pool_stats or {}
    bpm = playlist_stats.get("bpm_summary") or {}

    notes: list[str] = []
    # Confessions in LISTENER vocabulary ONLY — never relay a raw warning
    # message (they carry engine terms like bridge_floor / X_genre_raw that
    # must never reach the GUI; spec hard-constraint). Only the relaxation
    # cascade is a user-facing confession; other warning types are internal
    # diagnostics and are intentionally NOT surfaced here.
    # NOTE: the exact phrase below is provisional copy — product owner
    # finalizes wording later.
    warnings = playlist_stats.get("warnings") or []
    if any(isinstance(w, dict) and w.get("type") == "relaxation" for w in warnings):
        notes.append("relaxed the match to fill out the playlist")
    rescued = _i(pool_stats.get("genre_rescued")) or 0
    if rescued:
        notes.append(f"kept {rescued} sound-alike connectors past the Range gate")
    n, total = _i(bpm.get("n")), _i(bpm.get("total"))
    if n is not None and total and n < total / 2:
        notes.append(f"tempo data sparse ({n}/{total} tracks) — Pace applied where possible")

    return {
        "range": {"pool": _i(pool_stats.get("admitted")), "considered": _i(pool_stats.get("considered"))},
        "flow": {"worst": _f(playlist_stats.get("min_transition")), "mean": _f(playlist_stats.get("mean_transition"))},
        "pace": {"bpm_mean": _f(bpm.get("mean")), "bpm_std": _f(bpm.get("std")), "n": n, "total": total},
        "notes": notes,
    }
