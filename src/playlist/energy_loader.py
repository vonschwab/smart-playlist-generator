"""Load a z-scored per-track energy matrix from the energy sidecar.

Runtime-only consumer of the Essentia energy sidecar (no essentia import).
Mirrors bpm_loader: returns arrays aligned to the requested track_ids, NaN for
gaps. Library-wide z-score so distances are in std units (matches the eval).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _zscore_params(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    mean = float(finite.mean())
    std = float(finite.std())
    return (mean, std if std > 0 else 1.0)


def load_energy_matrix(
    track_ids: Sequence[str],
    *,
    sidecar_path: str,
    features: Sequence[str] = ("arousal_p50",),
) -> np.ndarray:
    """Return (len(track_ids), len(features)) z-scored energy matrix; NaN rows for gaps."""
    track_ids = [str(t) for t in track_ids]
    n = len(track_ids)
    feats = list(features)
    out = np.full((n, len(feats)), np.nan, dtype=float)

    z = np.load(sidecar_path, allow_pickle=True)
    side_ids = [str(t) for t in z["track_ids"]]
    pos = {t: i for i, t in enumerate(side_ids)}
    for fi, feat in enumerate(feats):
        if feat not in z:
            logger.warning("energy_loader: feature %r absent from sidecar; column left NaN", feat)
            continue
        col = np.asarray(z[feat], dtype=float)
        mean, std = _zscore_params(col)            # library-wide
        for ti, tid in enumerate(track_ids):
            j = pos.get(tid)
            if j is not None and np.isfinite(col[j]):
                out[ti, fi] = (col[j] - mean) / std
    return out
