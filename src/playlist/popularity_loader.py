"""Load a per-track popularity vector from the popularity sidecar.

Runtime-only consumer (no Last.fm import). Mirrors energy_loader: returns values
aligned to the requested track_ids, NaN for gaps. Popularity is already a
per-artist rank score in [0,1] — no z-scoring.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_popularity_vector(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray:
    """Return (len(track_ids),) popularity aligned to track_ids; NaN for gaps."""
    n = len(track_ids)
    out = np.full(n, np.nan, dtype=float)
    if not Path(sidecar_path).exists():
        logger.info("popularity_loader: sidecar missing at %s; all-NaN (neutral)", sidecar_path)
        return out
    z = np.load(sidecar_path, allow_pickle=True)
    if "popularity" not in z or "track_ids" not in z:
        logger.warning("popularity_loader: sidecar missing expected keys; all-NaN")
        return out
    pos = {str(t): i for i, t in enumerate(z["track_ids"])}
    col = np.asarray(z["popularity"], dtype=float)
    for ti, tid in enumerate(track_ids):
        j = pos.get(str(tid))
        if j is not None and np.isfinite(col[j]):
            out[ti] = col[j]
    return out
