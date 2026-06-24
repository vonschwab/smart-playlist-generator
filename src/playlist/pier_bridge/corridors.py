"""Corridor math: how far a candidate strays from the on-manifold reference, and
the soft width-controlled penalty for straying. Pure; no engine deps.

Used by the roam-corridors engine: the builder produces per-candidate deviation
arrays (geodesic detour for sonic/genre, band deviation for energy) and the beam
turns them into a soft penalty bounded by the corridor width.
"""
from __future__ import annotations

import numpy as np


def geodesic_detour(d_a: np.ndarray, d_b: np.ndarray, pier_b: int) -> np.ndarray:
    """Per-candidate detour vs the direct geodesic between pier_a and pier_b.

    `d_a`/`d_b` are single-source shortest-path distances from pier_a / pier_b over
    the kNN graph. detour = d_a[c] + d_b[c] - geodesic(a,b); 0 on the path, larger
    off it. +inf if a candidate is unreachable from a pier; and if pier_a and pier_b
    are themselves disconnected (geodesic = inf) EVERY entry becomes +inf, so the
    corridor goes inert (uniform penalty) for that segment rather than starving it.
    """
    geo = float(d_a[int(pier_b)])
    det = np.asarray(d_a, dtype=np.float64) + np.asarray(d_b, dtype=np.float64) - geo
    return np.where(np.isfinite(det), np.maximum(det, 0.0), np.inf)


def corridor_penalty(deviation: np.ndarray, width: float, *, slope: float = 1.0) -> np.ndarray:
    """Soft penalty for straying beyond the corridor width.

    Free (0) while deviation <= width; smooth and increasing beyond it. Unreachable
    (inf) deviations get a large but finite penalty so the beam never sees -inf and
    the pool never hard-starves.
    """
    dev = np.asarray(deviation, dtype=np.float64)
    over = dev - float(width)
    over = np.where(np.isfinite(over), over, 1e6)          # unreachable -> large finite
    pen = np.where(over <= 0.0, 0.0, over)                  # linear-beyond-boundary, exactly 0 inside
    return pen * float(slope)


def band_deviation(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Distance of each value outside [lo, hi]; 0 inside the band (energy corridor)."""
    v = np.asarray(values, dtype=np.float64)
    return np.maximum(0.0, np.maximum(lo - v, v - hi))
