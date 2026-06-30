"""SP2 seed-character anti-collapse scoring (pure functions, unit-testable).

Genre-blur collapse: bridge interiors drift off the seeds' specific character into the
dense generic neighborhood ([[project_collapse_attack_design]]). Two alternative
bridge-candidate adjustments, compared head-to-head via the collapse harness:

  A "hubness"     — deflate a candidate's pier-similarities by its k-NN in-degree
                    within the segment pool. Hubs (tracks that sit in everyone's
                    neighborhood) are the generic-blur attractor; the harmonic-mean
                    bridge score over-rewards them because they're close to *both*
                    piers. Deflating by hubness pushes the beam toward genuine
                    pier-neighbors over central wallpaper.
  B "anti_center" — penalize a candidate by how much closer it sits to the local pool
                    centroid than to its own piers. This is the scoring twin of the
                    within-bridge sag metric (cos(t,center) - cos(t,piers)): it
                    directly demotes interiors that sag toward the average.

Off (strength 0 / mode "off") => the caller leaves the score untouched (byte-identical).
"""
from __future__ import annotations

import numpy as np


def pool_hubness(X_pool_norm: np.ndarray, k: int) -> np.ndarray:
    """k-NN in-degree per pool track, normalized to [0, 1] (1 = biggest hub).

    ``X_pool_norm`` is (n, d) unit rows. Each track votes for its top-k neighbors
    (excluding itself); a track's hubness = (votes received) / (max votes). A hub
    sits in many neighborhoods => high; a peripheral track => ~0. Rank-based, so it
    is robust to the near-isotropy that makes a centroid-distance hub measure
    degenerate in well-conditioned spaces like MuQ.
    """
    X = np.asarray(X_pool_norm, dtype=np.float64)
    n = int(X.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)            # never a neighbor of itself
    kk = int(min(max(int(k), 1), n - 1))
    nbr = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    indeg = np.bincount(nbr.reshape(-1), minlength=n).astype(np.float64)
    peak = float(indeg.max())
    return indeg / peak if peak > 0.0 else indeg


def hubness_deflated_bridge(sim_a: float, sim_b: float, hub: float, strength: float) -> float:
    """SP2-A bridge score: harmonic mean of the pier-sims after deflating BOTH by
    ``strength * hub`` (hubs lose more, clamped at 0). ``strength`` 0 => the plain
    harmonic mean the beam computes today."""
    a = max(0.0, float(sim_a) - float(strength) * float(hub))
    b = max(0.0, float(sim_b) - float(strength) * float(hub))
    denom = a + b
    return 0.0 if denom <= 1e-9 else (2.0 * a * b) / denom


def anti_center_penalty(cand_center_sim: float, bridge_score: float, strength: float) -> float:
    """SP2-B penalty (>= 0) to SUBTRACT from combined_score: ``strength`` times how
    much closer the candidate sits to the local pool centroid than to its own piers
    (the bridge score). 0 when the candidate is more pier-like than central."""
    return float(strength) * max(0.0, float(cand_center_sim) - float(bridge_score))
