"""Load BPM arrays from `metadata.db` aligned to an artifact's track_ids.

Reads `sonic_features.full.bpm_info` for each track via JSON extraction.
Returns numpy arrays aligned with the input `track_ids` (NaN for missing).

This is a runtime fallback path until `build_beat3tower_artifacts.py`
bakes BPM arrays directly into the .npz (deferred task).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict

import numpy as np

from src.playlist.bpm_axis import resolve_perceptual_bpm

logger = logging.getLogger(__name__)

SQL_VARIABLE_BATCH_SIZE = 900


def _track_id_batches(track_ids: list[str], batch_size: int = SQL_VARIABLE_BATCH_SIZE):
    for start in range(0, len(track_ids), max(1, int(batch_size))):
        yield track_ids[start : start + max(1, int(batch_size))]


def load_bpm_arrays(
    track_ids: np.ndarray,
    *,
    db_path: str,
) -> Dict[str, np.ndarray]:
    """Return BPM arrays aligned with track_ids.

    Keys: primary_bpm, perceptual_bpm, tempo_stability,
          half_tempo_likely (bool), double_tempo_likely (bool).
    Missing tracks and null sonic_features return NaN / False.
    """
    n = int(track_ids.shape[0])
    primary = np.full(n, np.nan, dtype=float)
    perceptual = np.full(n, np.nan, dtype=float)
    stability = np.full(n, np.nan, dtype=float)
    half_flags = np.zeros(n, dtype=bool)
    double_flags = np.zeros(n, dtype=bool)

    id_to_pos = {str(tid): i for i, tid in enumerate(track_ids)}
    if not id_to_pos:
        return {
            "primary_bpm": primary,
            "perceptual_bpm": perceptual,
            "tempo_stability": stability,
            "half_tempo_likely": half_flags,
            "double_tempo_likely": double_flags,
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        for batch in _track_id_batches(list(id_to_pos.keys())):
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"""
                SELECT track_id,
                       json_extract(sonic_features, '$.full.bpm_info.primary_bpm') AS primary_bpm,
                       json_extract(sonic_features, '$.full.bpm_info.half_tempo_likely') AS half_t,
                       json_extract(sonic_features, '$.full.bpm_info.double_tempo_likely') AS double_t,
                       json_extract(sonic_features, '$.full.bpm_info.tempo_stability') AS stability
                FROM tracks
                WHERE track_id IN ({placeholders})
                """,
                tuple(batch),
            )
            for row in cur.fetchall():
                pos = id_to_pos.get(str(row["track_id"]))
                if pos is None:
                    continue
                bpm = row["primary_bpm"]
                if bpm is None:
                    continue
                half = bool(row["half_t"]) if row["half_t"] is not None else False
                dbl = bool(row["double_t"]) if row["double_t"] is not None else False
                stab = float(row["stability"]) if row["stability"] is not None else 0.0
                primary[pos] = float(bpm)
                half_flags[pos] = half
                double_flags[pos] = dbl
                stability[pos] = stab
                perceptual[pos] = resolve_perceptual_bpm(
                    float(bpm), half_tempo_likely=half, double_tempo_likely=dbl
                )
    finally:
        conn.close()

    missing_count = int(np.sum(np.isnan(perceptual)))
    if missing_count:
        logger.warning(
            "BPM data missing for %d/%d tracks (will skip BPM gate for them)",
            missing_count,
            n,
        )

    return {
        "primary_bpm": primary,
        "perceptual_bpm": perceptual,
        "tempo_stability": stability,
        "half_tempo_likely": half_flags,
        "double_tempo_likely": double_flags,
    }
