"""Read path for the isolated instrumental sidecar (per-track voice_prob).

Mirrors src/playlist/energy_loader.py's index-map alignment, but degrades to an
all-NaN array (+ WARNING) on a missing sidecar instead of raising — the
Instrumental guard must be inert-but-safe when the data is absent, never fatal.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_voice_prob(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray:
    n = len(track_ids)
    out = np.full(n, np.nan, dtype=np.float64)
    if not os.path.exists(sidecar_path):
        logger.warning(
            "instrumental_loader: sidecar %s absent; voice_prob all-NaN (Instrumental guard inert)",
            sidecar_path,
        )
        return out
    try:
        data = np.load(sidecar_path, allow_pickle=True)
    except Exception as exc:
        logger.warning(
            "instrumental_loader: failed to read sidecar %s (%s); voice_prob all-NaN", sidecar_path, exc
        )
        return out
    if "voice_prob" not in data or "track_ids" not in data:
        logger.warning("instrumental_loader: sidecar %s missing keys; voice_prob all-NaN", sidecar_path)
        return out
    side_ids = [str(t) for t in data["track_ids"]]
    side_vp = np.asarray(data["voice_prob"], dtype=np.float64)
    pos = {t: i for i, t in enumerate(side_ids)}
    for i, t in enumerate(track_ids):
        j = pos.get(str(t))
        if j is not None:
            out[i] = side_vp[j]
    return out
