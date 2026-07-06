# src/playlist/pier_bridge/mini_pier_select.py
"""SP3 mini-pier v2 selection (pure functions, unit-testable).

Pick a waypoint to pin inside a long bridge: relative smoothness floor (candidates
within `margin` of the best available min-sim to BOTH piers, so the pick is genuinely
between them and adapts to close vs cross-niche) -> anti-center within (the least
central relative to the local between-region, so it's on-character, not the wallpaper).
See docs/superpowers/specs/2026-06-30-sp3-mini-piers-design.md.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def select_waypoint(
    pier_a: int,
    pier_b: int,
    candidate_indices: Sequence[int],
    X_full_norm: np.ndarray,
    *,
    margin: float = 0.12,
    k_broad: int = 150,
    exclude: frozenset[int] = frozenset(),
) -> Optional[int]:
    piers = {int(pier_a), int(pier_b)}
    cand = np.array(
        [int(c) for c in candidate_indices if int(c) not in piers and int(c) not in exclude],
        dtype=int,
    )
    if cand.size == 0:
        return None
    simA = X_full_norm[cand] @ X_full_norm[int(pier_a)]
    simB = X_full_norm[cand] @ X_full_norm[int(pier_b)]
    minsim = np.minimum(simA, simB)
    # between-region = the k_broad most-between candidates (stable local center + floor)
    k = int(min(max(1, k_broad), cand.size))
    broad_local = np.argpartition(-minsim, k - 1)[:k]
    broad = cand[broad_local]
    best = float(minsim[broad_local].max())
    smooth_mask = minsim[broad_local] >= best - float(margin)
    smooth = broad[smooth_mask]
    if smooth.size == 0:
        return int(broad[int(np.argmax(minsim[broad_local]))])
    center = X_full_norm[smooth].mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm < 1e-12:
        return int(smooth[0])
    center = center / norm
    cent = X_full_norm[smooth] @ center
    return int(smooth[int(np.argmin(cent))])


def _even_split_lengths(total_interior: int, num_segments: int) -> list[int]:
    base, rem = divmod(total_interior, num_segments)
    return [base + (1 if i < rem else 0) for i in range(num_segments)]


def plan_pier_sequence(
    ordered_seeds,
    total_tracks: int,
    candidate_indices,
    X_full_norm: np.ndarray,
    *,
    max_interior: int,
    margin: float,
    k_broad: int,
    exclude_base: frozenset[int] = frozenset(),
    max_waypoints: int = 8,
    balance_gaps: bool = False,
) -> list[int]:
    """Greedily split the longest segment (by even-split interior) by inserting a
    waypoint between its two piers, until every segment's interior <= max_interior,
    no feasible waypoint remains, or max_waypoints is reached. Returns the augmented
    pier list; identical to ordered_seeds when nothing needs splitting.

    ``balance_gaps`` (option 2): once subdivision has begun, keep inserting until
    every ORIGINAL seed-gap holds the SAME number of waypoints, so the seed anchors
    stay evenly spaced. Without it, W waypoints over M gaps (W not a multiple of M)
    leaves the trailing gap(s) unsplit and the seed anchors bunch at the end -- e.g.
    4 piers / 30 tracks needs 2 waypoints to hit interior<=5, both land in the first
    two gaps, and the last two anchors sit ~5 apart while the rest are ~12 apart.
    Balancing tops that up to 3 waypoints (one per gap) so anchors land ~evenly."""
    piers = [int(s) for s in ordered_seeds]
    used = set(piers) | {int(e) for e in exclude_base}
    num_seed_gaps = len(piers) - 1
    if num_seed_gaps < 1:
        return piers
    wp_per_gap = [0] * num_seed_gaps      # waypoints inserted per ORIGINAL seed-gap
    seg_gap = list(range(num_seed_gaps))  # current-segment index -> its origin seed-gap
    for _ in range(int(max_waypoints)):
        num_seg = len(piers) - 1
        interior = int(total_tracks) - len(piers)
        if num_seg < 1 or interior < 1:
            break
        lengths = _even_split_lengths(interior, num_seg)
        satisfied = int(max(lengths)) <= int(max_interior)
        # With balancing on, "done" also requires every seed-gap to hold an equal
        # waypoint count -- otherwise the trailing gap stays a single long segment
        # and the anchors bunch. With it off, behaviour is unchanged (byte-identical).
        balanced = (not balance_gaps) or (min(wp_per_gap) == max(wp_per_gap))
        if satisfied and balanced:
            break
        # Round-robin across ORIGIN seed-gaps: split the gap with the fewest waypoints
        # so far (ties -> leftmost). Even-split makes every sub-segment ~equal length
        # regardless of which gap we split, so the split-choice only controls seed
        # spacing -- distribute it, don't (as the old argmax did) hammer segment 0.
        # NB: do NOT pre-filter to `lengths[i] > max_interior`. `_even_split_lengths`
        # piles the remainder onto the EARLIEST segments, so at the tail only gap-0
        # segments look "over-long" and waypoints re-concentrate there (the bug found
        # in review). `max(lengths)` above is the stop condition; selection ranges over
        # ALL segments.
        # When the interior target is already met and we're only balancing, restrict
        # the split to the under-served gaps so we top them up to the max count
        # without over-inserting into gaps that are already deep enough.
        if balance_gaps and satisfied:
            target = max(wp_per_gap)
            eligible = {g for g in range(num_seed_gaps) if wp_per_gap[g] < target}
        else:
            eligible = set(range(num_seed_gaps))
        order = sorted(
            (i for i in range(num_seg) if seg_gap[i] in eligible),
            key=lambda i: (wp_per_gap[seg_gap[i]], i),
        )
        seg = wp = None
        for i in order:
            cand = select_waypoint(
                piers[i], piers[i + 1], candidate_indices, X_full_norm,
                margin=margin, k_broad=k_broad, exclude=frozenset(used),
            )
            if cand is not None:
                seg, wp = i, int(cand)
                break
        if seg is None or wp is None:
            break
        g = seg_gap[seg]
        piers.insert(seg + 1, wp)
        seg_gap.insert(seg + 1, g)
        wp_per_gap[g] += 1
        used.add(wp)
    return piers
