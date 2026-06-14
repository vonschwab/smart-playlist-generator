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


def sample_edges(
    edges: List[Tuple[int, int]], k: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Seeded sample of up to k edges (all of them if fewer than k)."""
    if len(edges) <= k:
        return list(edges)
    idx = rng.choice(len(edges), size=k, replace=False)
    return [edges[int(i)] for i in sorted(idx)]


def synthesize_decoy_edges(
    context_tids: Sequence[str],
    *,
    onset: Dict[str, float],
    bpm: Dict[str, float],
    genre_vecs: Dict[str, np.ndarray],
    k: int,
    rng: np.random.Generator,
    min_onset_dist: float = 1.0,
) -> List[Tuple[str, str]]:
    """Pairs that are pace-distant (onset log-dist > min) but genre-close
    (genre cos >= median over qualifying pairs). Falls back to the 25th
    percentile genre floor if fewer than k qualify. Returns (a, b) track-id pairs."""
    tids = [str(t) for t in context_tids]
    cand: List[Tuple[str, str, float]] = []  # (a, b, genre_cos)
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a, b = tids[i], tids[j]
            if not np.isfinite(float(bpm_log_distance(onset.get(a, np.nan), onset.get(b, np.nan)))):
                continue
            od = float(bpm_log_distance(onset.get(a, np.nan), onset.get(b, np.nan)))
            if od <= float(min_onset_dist):
                continue
            gc = genre_cosine(genre_vecs.get(a, np.zeros(1)), genre_vecs.get(b, np.zeros(1)))
            cand.append((a, b, gc))
    if not cand:
        return []
    gcs = np.array([c[2] for c in cand])
    floor = float(np.median(gcs))
    qualifying = [(a, b) for (a, b, gc) in cand if gc >= floor]
    if len(qualifying) < k:
        floor = float(np.percentile(gcs, 25))
        qualifying = [(a, b) for (a, b, gc) in cand if gc >= floor]
    if len(qualifying) <= k:
        return qualifying
    idx = rng.choice(len(qualifying), size=k, replace=False)
    return [qualifying[int(i)] for i in sorted(idx)]
