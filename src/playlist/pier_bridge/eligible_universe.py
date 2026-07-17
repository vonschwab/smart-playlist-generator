"""Eligible-universe assembly -- Phase 1 Task 2 of
docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.

Computes, ONCE per generation, the index set + aligned L2-normalized sonic
rows surviving all standing hard exclusions, plus the rehomed rank-penalty
vector that Task 3's per-segment corridor builder (`build_corridor`) consumes
as a multiplicative factor on `rank_scores`. Replaces the per-segment
re-application of these exclusions/penalties that `candidate_pool.py` does
today (once per pier pair) with a single pass over the whole bundle.

Hard exclusions (index-set membership, no bundle slicing):
  - recency/blacklist (`excluded_track_ids`, same pre-pipeline source as
    `playlist_generator.py:778`)
  - duration hard cutoff (same math as `candidate_pool.py:678`)
  - title hygiene (same check as `candidate_pool.py:1084-1089`, via
    `src.playlist.title_quality.detect_title_artifacts`)
  - relevance mask (Task 4 wires the mode-keyed mask; here it is applied
    verbatim when given, ignored when `None`)

Seeds/piers (`seed_indices`) are always exempt from every exclusion above,
mirroring `src/playlist/pipeline/bundle_restrict.py:66-128`'s exemption
semantics -- and from the soft rank penalty below, mirroring
`candidate_pool.py`'s `seed_mask[i]: continue` skip ahead of both penalty
loops (:671, :707).

C1 rehome (duration soft penalty) + C10-pool rehome (instrumental-lean pool
demotion): both were previously *subtracted* from a cosine-similarity score
in `candidate_pool.py` (additive, similarity-space). Here they are reframed
as multiplicative factors in `[0, 1]` on the corridor's `rank_scores`
(`1.0 - penalty`, clamped at 0 so an extreme duration overshoot can never
flip the sign of a downstream score), then combined multiplicatively:

    duration_rank_penalty[i] = max(0, 1 - duration_penalty(i))
                              * max(0, 1 - instrumental_penalty(i))

`1.0` means "no penalty" for both hard-exempt seeds and untouched rows.
C10's beam half (`beam.py:1256-1257`) is untouched by this module --
preserving the never-fail contract stays the beam's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.playlist.candidate_pool import compute_duration_penalty
from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
from src.playlist.pier_bridge.vec import _l2_normalize_rows  # noqa: F401 -- reused private helper
from src.playlist.title_quality import detect_title_artifacts


@dataclass(frozen=True)
class EligibleUniverse:
    indices: np.ndarray  # bundle indices surviving all standing hard exclusions, ascending
    X_norm: np.ndarray  # aligned L2-normalized sonic rows
    duration_rank_penalty: np.ndarray  # aligned multiplicative penalty (C1 + C10), 1.0 = none
    stats: dict[str, Any] = field(default_factory=dict)  # counts per exclusion class


def build_eligible_universe(
    *,
    bundle: Any,
    seed_indices: list[int],
    excluded_track_ids: set[str],
    relevance_mask: "np.ndarray | None",  # (N,) bool over the full bundle; None = genre_mode off
    duration_reference_ms: "float | None",
    duration_cutoff_multiplier: float,
    duration_penalty_weight: float,
    title_hard_exclude_flags: "frozenset[str]",  # see deviation note below
    instrumental_enabled: bool,
    instrumental_penalty_weight: float,
    voice_prob: "np.ndarray | None",
) -> EligibleUniverse:
    """Assemble the eligible universe once per generation.

    Deviation from the task brief's literal signature (documented, matching
    Task 1's precedent for flagged deviations): the brief types
    `title_hard_exclude_flags` as `int` ("bitmask"), but the real reference
    (`candidate_pool.py:1085-1086`, `config.py:58`) is a `frozenset[str]` of
    flag names intersected against `detect_title_artifacts`'s returned set --
    there is no integer bitmask anywhere in this codebase's title-hygiene
    path. Implemented against the real type so this module's exclusion is
    byte-identical to the reference, not a reinterpretation of it.

    `bundle` is duck-typed (not type-hinted as `ArtifactBundle`) so tests can
    pass a minimal stub carrying only the fields this function reads:
    `track_ids`, `X_sonic`, `track_titles`, `durations_ms`.
    """
    n = int(len(bundle.track_ids))
    seed_set = {int(i) for i in seed_indices}

    track_ids = bundle.track_ids
    track_titles = bundle.track_titles
    durations_ms = bundle.durations_ms

    excluded_ids_str = {str(t) for t in excluded_track_ids} if excluded_track_ids else set()

    # Mirrors candidate_pool.py's `duration_penalty_active` gate: no reference
    # duration (or a non-positive one), or no durations column at all -> the
    # whole duration mechanism (cutoff + penalty) is inert.
    duration_active = (
        durations_ms is not None
        and duration_reference_ms is not None
        and float(duration_reference_ms) > 0.0
    )
    cutoff_ms = (
        float(duration_reference_ms) * float(duration_cutoff_multiplier)  # type: ignore[arg-type]
        if duration_active
        else None
    )

    title_flags = title_hard_exclude_flags or frozenset()

    excluded_mask = np.zeros(n, dtype=bool)
    n_recency = 0
    n_duration_cutoff = 0
    n_title = 0
    n_relevance = 0

    for i in range(n):
        if i in seed_set:
            continue
        row_excluded = False

        if excluded_ids_str and str(track_ids[i]) in excluded_ids_str:
            n_recency += 1
            row_excluded = True

        if duration_active and cutoff_ms is not None:
            dur = float(durations_ms[i])
            # candidate_pool.py:676 -- unknown/zero duration is never
            # cutoff-excluded or penalized (unknown is never punished).
            if dur > 0.0 and dur > cutoff_ms:
                n_duration_cutoff += 1
                row_excluded = True

        if title_flags and track_titles is not None:
            flags = detect_title_artifacts(str(track_titles[i]))
            if flags & title_flags:
                n_title += 1
                row_excluded = True

        if relevance_mask is not None and not bool(relevance_mask[i]):
            n_relevance += 1
            row_excluded = True

        if row_excluded:
            excluded_mask[i] = True

    indices = np.nonzero(~excluded_mask)[0].astype(np.int64)

    X_full_norm = _l2_normalize_rows(bundle.X_sonic)
    X_norm = X_full_norm[indices]

    penalty = np.ones(len(indices), dtype=np.float64)
    for pos in range(len(indices)):
        idx = int(indices[pos])
        if idx in seed_set:
            continue  # seeds exempt from the soft penalty too (candidate_pool seed_mask skip)

        factor = 1.0
        if duration_active and durations_ms is not None:
            dur = float(durations_ms[idx])
            if dur > 0.0 and dur > float(duration_reference_ms):  # type: ignore[arg-type]
                dp = compute_duration_penalty(
                    dur, float(duration_reference_ms), float(duration_penalty_weight)  # type: ignore[arg-type]
                )
                if dp > 0:
                    factor *= max(0.0, 1.0 - dp)

        if instrumental_enabled and instrumental_penalty_weight > 0.0 and voice_prob is not None:
            ip = compute_instrumental_penalty(voice_prob, cand=idx, weight=float(instrumental_penalty_weight))
            if ip > 0:
                factor *= max(0.0, 1.0 - ip)

        penalty[pos] = factor

    stats: dict[str, Any] = {
        "universe_size_total": n,
        "seed_exempt_count": len(seed_set),
        "excluded_recency_blacklist": n_recency,
        "excluded_duration_cutoff": n_duration_cutoff,
        "excluded_title_hygiene": n_title,
        "excluded_relevance_mask": n_relevance,
        "excluded_total": int(np.count_nonzero(excluded_mask)),
        "eligible_count": int(len(indices)),
    }

    return EligibleUniverse(indices=indices, X_norm=X_norm, duration_rank_penalty=penalty, stats=stats)
