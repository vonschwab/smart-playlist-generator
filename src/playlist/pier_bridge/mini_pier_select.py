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
