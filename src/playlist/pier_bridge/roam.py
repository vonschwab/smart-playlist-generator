"""Per-segment roam-corridor assembly: compose the on-manifold kNN graph
(manifold.py) and the corridor formulas (corridors.py) into the global-indexed
deviation arrays the beam consumes.

Scope note: the sonic kNN graph is built over the *segment's* node set
([pier_a, pier_b] + candidates) — hundreds of nodes — NOT the full ~32k library.
A library-wide N×N similarity matrix would be ~8 GB; the on-manifold reference we
need is the geodesic between the two piers *through the segment's own candidates*,
so the small per-segment graph is both correct and cheap (honors the 90 s budget).
If the segment graph is disconnected (pier_b unreachable from pier_a), every
candidate's detour is +inf, the penalty is uniform, and the corridor is inert for
that segment — never a crash, never a starved pool.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.playlist.pier_bridge.manifold import build_knn_graph, geodesic_from_source
from src.playlist.pier_bridge.corridors import geodesic_detour, band_deviation


def segment_sonic_detour(
    pier_a: int,
    pier_b: int,
    candidates: Sequence[int],
    X_full_norm: np.ndarray,
    *,
    k: int,
    mutual_proximity: bool,
) -> np.ndarray:
    """Global-indexed geodesic-detour array over the segment's on-manifold graph.

    Returns an array of length ``X_full_norm.shape[0]``; entries for the segment's
    nodes carry their detour from the pier_a→pier_b geodesic (0 on the path), all
    other entries are +inf (not in this segment's corridor).
    """
    n_total = int(X_full_norm.shape[0])
    nodes = [int(pier_a), int(pier_b)]
    seen = {int(pier_a), int(pier_b)}
    for c in candidates:
        ci = int(c)
        if ci not in seen:
            seen.add(ci)
            nodes.append(ci)
    out = np.full(n_total, np.inf, dtype=np.float64)
    if len(nodes) < 3:
        # No interior candidates distinct from the piers — nothing to shape.
        out[int(pier_a)] = 0.0
        out[int(pier_b)] = 0.0
        return out
    local = {g: i for i, g in enumerate(nodes)}
    graph = build_knn_graph(X_full_norm[nodes], int(k), mutual_proximity_approx=bool(mutual_proximity))
    d_a = geodesic_from_source(graph, local[int(pier_a)])
    d_b = geodesic_from_source(graph, local[int(pier_b)])
    det_local = geodesic_detour(d_a, d_b, local[int(pier_b)])
    for g, i in local.items():
        out[g] = det_local[i]
    return out


def energy_band_deviation(
    energy: Optional[np.ndarray],
    seed_indices: Sequence[int],
) -> Optional[np.ndarray]:
    """Global-indexed deviation outside the seed-defined arousal band; None if no energy."""
    if energy is None:
        return None
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    seeds = [int(s) for s in seed_indices if 0 <= int(s) < e.shape[0]]
    if not seeds:
        return None
    vals = e[seeds]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    return band_deviation(e, lo, hi)
