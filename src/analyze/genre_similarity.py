from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config_loader import Config
from src.similarity_calculator import SimilarityCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenreSimilarityResult:
    genre_vocab: np.ndarray
    cooc: np.ndarray
    S: np.ndarray
    stats: dict


def _jaccard_sim(cooc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Compute Jaccard similarity matrix from co-occurrence and counts."""
    g = cooc.shape[0]
    S = np.zeros((g, g), dtype=np.float32)
    for i in range(g):
        for j in range(i, g):
            inter = cooc[i, j]
            denom = counts[i] + counts[j] - inter
            sim = float(inter / denom) if denom > 0 else 0.0
            S[i, j] = sim
            S[j, i] = sim
    np.fill_diagonal(S, 1.0)
    return S


def build_genre_similarity_matrix(
    *,
    db_path: str,
    config_path: str,
    out_path: str,
    min_count: int = 2,
    max_genres: int = 0,
) -> GenreSimilarityResult:
    """
    - Read normalized genre tables
    - Apply DS cleaning rules / broad/meta filters (reuse config where possible)
    - Build canonical vocab
    - Compute co-occurrence and similarity S
    - Save npz with keys: genre_vocab, cooc, S, stats
    """
    cfg = Config(config_path)
    calc = SimilarityCalculator(db_path=db_path, config=cfg.config)

    cursor = calc.conn.cursor()
    try:
        cursor.execute("SELECT track_id FROM tracks WHERE file_path IS NOT NULL")
    except Exception:
        cursor.execute("SELECT track_id FROM tracks")
    track_ids = [row["track_id"] for row in cursor.fetchall()]

    genre_lists: List[List[str]] = []
    for tid in track_ids:
        genres = calc.get_filtered_combined_genres_for_track(tid) or []
        if genres:
            genre_lists.append(list(dict.fromkeys(genres)))  # dedupe per track

    # Build counts
    counts: Dict[str, int] = {}
    for gl in genre_lists:
        for g in gl:
            counts[g] = counts.get(g, 0) + 1

    # Filter by min_count and cap
    filtered = [(g, c) for g, c in counts.items() if c >= min_count]
    filtered.sort(key=lambda t: (-t[1], t[0]))
    if max_genres and len(filtered) > max_genres:
        filtered = filtered[:max_genres]
    vocab = [g for g, _ in filtered]
    if not vocab:
        raise RuntimeError("No genres meet the min_count threshold; cannot build similarity matrix.")

    index = {g: i for i, g in enumerate(vocab)}
    cooc = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    track_used = 0
    for gl in genre_lists:
        filtered_gl = [g for g in gl if g in index]
        if not filtered_gl:
            continue
        track_used += 1
        for i, gi in enumerate(filtered_gl):
            idx_i = index[gi]
            cooc[idx_i, idx_i] += 1
            for gj in filtered_gl[i + 1 :]:
                idx_j = index[gj]
                cooc[idx_i, idx_j] += 1
                cooc[idx_j, idx_i] += 1

    counts_vec = np.diag(cooc).copy()
    S = _jaccard_sim(cooc, counts_vec)

    stats = {
        "tracks_seen": len(track_ids),
        "tracks_with_genres": track_used,
        "genres_kept": len(vocab),
        "min_count": min_count,
        "max_genres": max_genres,
    }

    np.savez(out_path, genre_vocab=np.array(vocab), cooc=cooc, S=S, stats=stats)
    logger.info("Saved genre similarity matrix to %s (G=%d, tracks=%d)", out_path, len(vocab), track_used)

    calc.close()
    return GenreSimilarityResult(
        genre_vocab=np.array(vocab),
        cooc=cooc,
        S=S,
        stats=stats,
    )
