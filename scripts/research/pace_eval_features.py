"""Candidate pace-representation feature vectors for the eval.

Reads the prebuilt energy sidecar + the beat3tower artifact + metadata.db.
Does NOT import essentia. Reuses bpm_loader for perceptual_bpm + onset_rate.
"""
from __future__ import annotations

import sqlite3

import numpy as np

from scripts.research.pace_eval_metrics import apply_zscore, zscore_params

SCALAR_KEYS = [
    "log_bpm_trusted", "onset_rate", "beat_strength",
    "arousal_p10", "arousal_p50", "arousal_p90", "danceability",
]

CANDIDATES: dict[str, list[str]] = {
    "rhythm_tower": ["rhythm_tower"],
    "perceptual_bpm": ["log_bpm_trusted"],
    "onset_rate": ["onset_rate"],
    "beat_strength": ["beat_strength"],
    "arousal_p50": ["arousal_p50"],
    "danceability": ["danceability"],
    "energy_pair": ["arousal_p50", "danceability"],
    "energy_dist": ["arousal_p10", "arousal_p50", "arousal_p90", "danceability"],
    "energy_onset": ["arousal_p50", "danceability", "onset_rate"],
    "pace_full": ["arousal_p50", "danceability", "onset_rate", "log_bpm_trusted", "beat_strength"],
}


def zscore_features(raw_scalars: dict[str, np.ndarray], raw_tower: np.ndarray):
    zscored = {}
    for key, arr in raw_scalars.items():
        mean, std = zscore_params(arr)
        zscored[key] = apply_zscore(arr, mean, std)
    ztower = np.full_like(raw_tower, np.nan, dtype=float)
    for d in range(raw_tower.shape[1]):
        mean, std = zscore_params(raw_tower[:, d])
        ztower[:, d] = apply_zscore(raw_tower[:, d], mean, std)
    return zscored, ztower


def candidate_vector(name: str, idx: int, zscored_scalars: dict[str, np.ndarray],
                     ztower: np.ndarray) -> np.ndarray:
    if name == "rhythm_tower":
        return np.asarray(ztower[idx], dtype=float)
    keys = CANDIDATES[name]
    return np.array([float(zscored_scalars[k][idx]) for k in keys], dtype=float)


def load_raw_features(track_ids, *, db_path: str, artifact_path: str,
                      sidecar_path: str, bpm_trust_min_onset: float = 0.5):
    """Return (index, raw_scalars, raw_tower) aligned to track_ids order."""
    track_ids = [str(t) for t in track_ids]
    index = {tid: i for i, tid in enumerate(track_ids)}
    n = len(track_ids)

    # perceptual_bpm + onset via the production loader (reuse, DRY).
    from src.playlist.bpm_loader import load_bpm_arrays
    bpm = load_bpm_arrays(np.array(track_ids, dtype=object), db_path=db_path)
    perceptual_bpm = np.asarray(bpm["perceptual_bpm"], dtype=float)
    onset_rate = np.asarray(bpm["onset_rate"], dtype=float)
    log_bpm = np.log(np.where(perceptual_bpm > 0, perceptual_bpm, np.nan))
    untrusted = ~(onset_rate >= float(bpm_trust_min_onset))
    log_bpm_trusted = np.where(untrusted, np.nan, log_bpm)

    # beat_strength_median from DB.
    beat_strength = np.full(n, np.nan, dtype=float)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for start in range(0, n, 900):
            batch = track_ids[start:start + 900]
            ph = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"SELECT track_id, json_extract(sonic_features,"
                f"'$.full.rhythm.beat_strength_median') AS bs "
                f"FROM tracks WHERE track_id IN ({ph})", tuple(batch)
            ):
                pos = index.get(str(row["track_id"]))
                if pos is not None and row["bs"] is not None:
                    beat_strength[pos] = float(row["bs"])
    finally:
        conn.close()

    # rhythm_tower (9-dim) from the artifact, mapped to track_ids order.
    art = np.load(artifact_path, allow_pickle=True)
    art_ids = [str(t) for t in art["track_ids"]]
    art_pos = {tid: i for i, tid in enumerate(art_ids)}
    tower_src = np.asarray(art["X_sonic_rhythm"], dtype=float)
    raw_tower = np.full((n, tower_src.shape[1]), np.nan, dtype=float)
    for tid, i in index.items():
        j = art_pos.get(tid)
        if j is not None:
            raw_tower[i] = tower_src[j]

    # arousal/danceability from the sidecar, mapped to track_ids order.
    side = np.load(sidecar_path, allow_pickle=True)
    side_ids = [str(t) for t in side["track_ids"]]
    side_pos = {tid: i for i, tid in enumerate(side_ids)}

    def _side(col):
        src = np.asarray(side[col], dtype=float)
        out = np.full(n, np.nan, dtype=float)
        for tid, i in index.items():
            j = side_pos.get(tid)
            if j is not None:
                out[i] = src[j]
        return out

    raw_scalars = {
        "log_bpm_trusted": log_bpm_trusted,
        "onset_rate": onset_rate,
        "beat_strength": beat_strength,
        "arousal_p10": _side("arousal_p10"),
        "arousal_p50": _side("arousal_p50"),
        "arousal_p90": _side("arousal_p90"),
        "danceability": _side("danceability"),
    }
    return index, raw_scalars, raw_tower
