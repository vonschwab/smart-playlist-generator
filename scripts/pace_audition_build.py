# scripts/pace_audition_build.py
"""Build the blinded pace-audition manifest.

Generates narrow/dynamic/off playlists per seed via the production fidelity
harness, extracts interior bridge edges, synthesizes pace-bad decoy edges,
blinds everything, and writes pace_manifest.json + index.json under
docs/run_audits/pace_audition/. Read-only against metadata.db and the artifact.

Usage:
    python scripts/pace_audition_build.py
    python scripts/pace_audition_build.py --seeds "Green-House" "J Dilla"
    python scripts/pace_audition_build.py --edges-per-arm 5 --decoy-per-seed 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.playlist.bpm_axis import bpm_log_distance  # noqa: E402


def genre_cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def edge_metrics(
    *, a_onset: float, b_onset: float, a_bpm: float, b_bpm: float,
    a_genre: np.ndarray, b_genre: np.ndarray,
) -> Dict[str, float]:
    return {
        "onset_log_dist": float(bpm_log_distance(a_onset, b_onset)),
        "bpm_log_dist": float(bpm_log_distance(a_bpm, b_bpm)),
        "genre_cos": genre_cosine(a_genre, b_genre),
    }


def extract_interior_edges(
    track_ids: Sequence[str], pier_ids: set
) -> List[Tuple[int, int]]:
    """Consecutive (i, i+1) index pairs where NEITHER endpoint is a pier."""
    out: List[Tuple[int, int]] = []
    for i in range(len(track_ids) - 1):
        if str(track_ids[i]) in pier_ids or str(track_ids[i + 1]) in pier_ids:
            continue
        out.append((i, i + 1))
    return out
