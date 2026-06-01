"""Distribution-relative (percentile) floors for adaptive genre gating.

A floor at percentile p of a similarity distribution admits roughly the top
(1 - p) fraction. Because it is relative to the distribution, it survives
embedding rebuilds and adapts to sparse vs dense seeds / disparate vs similar
piers.
"""
from __future__ import annotations

import numpy as np


def floor_at_percentile(sims, p: float) -> float:
    """Return the similarity value at percentile p (0..1) of `sims`.

    A candidate clears the floor iff sim >= floor, so this admits ~ the top
    (1 - p) fraction of the distribution.
    """
    arr = np.asarray(sims, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("-inf")
    p = float(min(max(p, 0.0), 1.0))
    return float(np.quantile(arr, p))


def relax_percentile(p: float, p_min: float, step: float = 0.15) -> list[float]:
    """Descending percentile sequence p -> p_min (admit progressively more)."""
    p = float(p)
    p_min = float(p_min)
    if p_min >= p - 1e-9:
        return [p]
    out = [p]
    cur = round(p - step, 4)
    while cur > p_min + 1e-9 and len(out) < 6:
        out.append(cur)
        cur = round(cur - step, 4)
    if not any(abs(x - p_min) < 1e-9 for x in out):
        out.append(p_min)
    return out
