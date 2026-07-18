"""Eligible-universe assembly -- Phase 1 Task 2 of
docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.

Computes, ONCE per generation, the index set + aligned L2-normalized sonic
rows surviving all standing hard exclusions, plus a `duration_rank_penalty`
placeholder vector (see the C1/C10 section below -- `build_corridor` does
NOT consume it; nothing does any more). Replaces the per-segment
re-application of these exclusions that `candidate_pool.py` does today
(once per pier pair) with a single pass over the whole bundle.

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
semantics.

C1 rehome (duration soft penalty) + C10-pool rehome (instrumental-lean pool
demotion): both were previously *subtracted* from a cosine-similarity score
in `candidate_pool.py` (additive, similarity-space). An earlier version of
this module reframed them as multiplicative factors in `[0, 1]`
(`duration_rank_penalty[i] = max(0, 1 - duration_penalty(i)) * max(0, 1 -
instrumental_penalty(i))`) intended for `build_corridor`'s `rank_scores` --
but the beam ended up owning BOTH effects' actual selection impact instead
(`beam.py`'s "Duration soft penalty" / "Instrumental lean" terms, additive
in transition-score space, matching C10's existing pattern) to avoid
double-applying the penalty at the segment_pool_max cap margin
(single-enforcement; see `pier_bridge_builder.py`'s C1-rehome comment and
CLAUDE.md Layer 3 item 18). `build_corridor` never reads
`duration_rank_penalty` -- it never did, once the beam rehome landed.

`duration_rank_penalty` is therefore now a structural placeholder: always
`1.0` (no penalty) for every eligible row, computed with no per-row work.
The field/param stays on `EligibleUniverse` (interface compatibility -- any
future consumer that indexes it by position still gets a valid, aligned
array) but nothing currently reads it: looping the real C1/C10 math over the
WHOLE eligible universe (tens of thousands of rows) purely to feed the
`mean_duration_penalty` diagnostic (which only ever reads a handful of
accepted-corridor members, ~2-30 per segment) was a dead full-universe pass
(final-review finding, corridor-phase1-pooling, 2026-07-18). The real
per-track math now lives in `pier_bridge_builder.py`, computed directly over
each segment's small accepted corridor at diagnostic time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.playlist.pier_bridge.vec import _l2_normalize_rows  # noqa: F401 -- reused private helper
from src.playlist.title_quality import detect_title_artifacts


@dataclass(frozen=True)
class EligibleUniverse:
    indices: np.ndarray  # bundle indices surviving all standing hard exclusions, ascending
    X_norm: np.ndarray  # aligned L2-normalized sonic rows
    duration_rank_penalty: np.ndarray  # structural placeholder, always 1.0 -- see module docstring
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

    # Structural placeholder only -- see the module docstring's C1/C10
    # section. No per-row work: the real duration/instrumental penalty math
    # is computed on demand, over each segment's small accepted corridor, by
    # pier_bridge_builder.py's diagnostic code, not by looping the whole
    # (tens-of-thousands-of-row) eligible universe here.
    penalty = np.ones(len(indices), dtype=np.float64)

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
