"""Pure corridor builder — spec section 1 of
docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md.

Membership: for a candidate x, corridor_score(x) = min(cos(A, x), cos(B, x)); a
candidate is a corridor member when corridor_score(x) >= quantile(min_sims,
width_percentile) — self-calibrating per anchor pair (no fixed global floor).

Ranking: harmonic mean of (sim_a, sim_b), optionally blended with a genre
harmonic mean. The hmean formula (and the blend combination) are copied
VERBATIM from src/playlist/segment_pool_builder.py so this module's scoring
semantics are byte-identical to the pre-existing segment-pool builder:

  - Base hmean: segment_pool_builder.py:334-337
    (SegmentPoolBuilder._compute_bridge_score, non-experimental branch)
        denom = sim_a + sim_b
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
  - Genre blend: segment_pool_builder.py:592-598
    (SegmentPoolBuilder._compute_bridge_scores, per-candidate loop)
        g_a_i = max(0.0, float(genre_to_a[i]))
        g_b_i = max(0.0, float(genre_to_b[i]))
        denom = g_a_i + g_b_i
        genre_hmean = 0.0 if denom <= 1e-9 else (2.0 * g_a_i * g_b_i) / denom
        score = (1.0 - genre_w) * score + genre_w * genre_hmean

This module is pure: numpy only, no config objects, no bundle, no logging (the
caller logs) and deterministic for fixed inputs.

Deviation from the Task 1 brief's literal signature (documented, not tested by
the 8 required unit tests, all of which pass genre_blend_weight=0.0): the
brief's interface block gives `X_genre_dense` as the only genre-blend input,
aligned with `X_norm` (the universe rows). The reference implementation this
module must byte-match also needs the *anchors'* genre vectors (it reads
`X_genre_dense[pier_a]` / `[pier_b]` from a library-wide dense genre matrix).
Since piers are not guaranteed to be part of the eligible universe passed as
`X_norm`/`X_genre_dense` here, this module accepts two additional *optional*
keyword-only parameters, `genre_vec_a`/`genre_vec_b` (the anchors' own dense
genre vectors), required only when `genre_blend_weight > 0`. This is additive
(default `None`, inert unless genre blending is requested) and does not change
any name or behavior the brief's 8 tests exercise. Flagged for the task-2+
author to confirm or correct against the real caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorridorResult:
    indices: np.ndarray          # eligible-universe indices, ranked (hmean desc)
    min_sims: np.ndarray         # min(sim_a, sim_b) per index, aligned
    rank_scores: np.ndarray      # hmean (+ optional genre blend), aligned
    threshold: float             # the min-sim cutoff actually applied
    width_percentile: float      # requested percentile
    capped: bool                 # segment_pool_max truncation applied
    stats: dict[str, Any]        # size_before_cap, universe_size, anchor_support_a/b


def _hmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise harmonic mean; 0.0 where a + b <= 1e-9 (never NaN).

    Verbatim (vectorized) from src/playlist/segment_pool_builder.py:334-337 --
    see module docstring for the scalar source and citation.
    """
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float64)
    safe = denom > 1e-9
    out[safe] = (2.0 * a[safe] * b[safe]) / denom[safe]
    return out


def _empty_result(width_percentile: float) -> CorridorResult:
    return CorridorResult(
        indices=np.array([], dtype=np.int64),
        min_sims=np.array([], dtype=np.float64),
        rank_scores=np.array([], dtype=np.float64),
        threshold=0.0,
        width_percentile=float(width_percentile),
        capped=False,
        stats={
            "size_before_cap": 0,
            "universe_size": 0,
            "anchor_support_a": 0.0,
            "anchor_support_b": 0.0,
        },
    )


def build_corridor(
    *,
    vec_a: np.ndarray,           # L2-normalized anchor A (D,)
    vec_b: np.ndarray,           # L2-normalized anchor B (D,)
    X_norm: np.ndarray,          # (N, D) L2-normalized library slice = eligible universe rows
    universe_indices: np.ndarray,  # (N,) bundle indices aligned with X_norm rows
    width_percentile: float,     # e.g. 0.90 -> keep top 10% by min-sim
    segment_pool_max: int,
    genre_blend_weight: float = 0.0,
    X_genre_dense: np.ndarray | None = None,   # aligned with X_norm when blending
    force_include: np.ndarray | None = None,   # bundle indices force-admitted (tag guarantee)
    genre_vec_a: np.ndarray | None = None,     # anchor A's dense genre vector (see module docstring)
    genre_vec_b: np.ndarray | None = None,     # anchor B's dense genre vector (see module docstring)
) -> CorridorResult:
    n = int(X_norm.shape[0])
    if n == 0:
        return _empty_result(width_percentile)

    sim_a = X_norm @ vec_a
    sim_b = X_norm @ vec_b
    min_sims = np.minimum(sim_a, sim_b)
    threshold = float(np.quantile(min_sims, width_percentile))

    rank_scores = _hmean(sim_a, sim_b)

    if genre_blend_weight > 0.0:
        if X_genre_dense is None or genre_vec_a is None or genre_vec_b is None:
            raise ValueError(
                "genre_blend_weight > 0 requires X_genre_dense, genre_vec_a, and "
                "genre_vec_b -- a configured blend that can't act is a bug, not a "
                "silent no-op."
            )
        genre_w = max(0.0, min(1.0, float(genre_blend_weight)))
        genre_a = np.maximum(0.0, X_genre_dense @ genre_vec_a)
        genre_b = np.maximum(0.0, X_genre_dense @ genre_vec_b)
        genre_hmean = _hmean(genre_a, genre_b)
        rank_scores = (1.0 - genre_w) * rank_scores + genre_w * genre_hmean

    # Anchor support: fraction of each anchor's own top-100 universe neighbors
    # (by raw similarity to that anchor) whose corridor min-sim clears the
    # threshold -- the validated Phase-0a coverage metric.
    k = min(100, n)
    top_a = np.argsort(sim_a)[::-1][:k]
    top_b = np.argsort(sim_b)[::-1][:k]
    anchor_support_a = float(np.mean(min_sims[top_a] >= threshold))
    anchor_support_b = float(np.mean(min_sims[top_b] >= threshold))

    member_local = np.nonzero(min_sims >= threshold)[0]
    order = np.argsort(-rank_scores[member_local], kind="stable")
    ranked_local = member_local[order]

    forced_local: np.ndarray = np.array([], dtype=np.int64)
    if force_include is not None and len(force_include) > 0:
        lookup = {int(g): i for i, g in enumerate(universe_indices)}
        ranked_set = set(int(i) for i in ranked_local)
        seen: set[int] = set()
        forced_candidates = []
        for g in force_include:
            local = lookup.get(int(g))
            if local is None or local in ranked_set or local in seen:
                continue
            seen.add(local)
            forced_candidates.append(local)
        if forced_candidates:
            forced_arr = np.array(forced_candidates, dtype=np.int64)
            forced_order = np.argsort(-rank_scores[forced_arr], kind="stable")
            forced_local = forced_arr[forced_order]

    size_before_cap = int(len(ranked_local) + len(forced_local))
    cap = int(segment_pool_max)
    capped = size_before_cap > cap
    if capped:
        # Truncate only the ranked (threshold-passing) portion; force-included
        # rows are guaranteed admission and are never evicted by the cap.
        ranked_final = ranked_local[: max(0, cap)]
    else:
        ranked_final = ranked_local
    final_local = np.concatenate([ranked_final, forced_local]).astype(np.int64)

    return CorridorResult(
        indices=universe_indices[final_local],
        min_sims=min_sims[final_local],
        rank_scores=rank_scores[final_local],
        threshold=threshold,
        width_percentile=float(width_percentile),
        capped=capped,
        stats={
            "size_before_cap": size_before_cap,
            "universe_size": n,
            "anchor_support_a": anchor_support_a,
            "anchor_support_b": anchor_support_b,
        },
    )
