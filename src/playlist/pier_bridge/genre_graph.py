"""Niche genre-adjacency graph derived from the dense per-genre embedding.

Two genres are adjacent when their genre_emb cosine is high. Broad "hub" genres
are excluded as nodes so ladder paths cannot collapse into them. Output format
matches what _shortest_genre_path expects: {label: [(neighbor_label, weight), ...]}.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def build_genre_graph(
    genre_emb: np.ndarray,
    genre_vocab: list[str] | np.ndarray,
    *,
    k: int = 8,
    min_cos: float = 0.35,
    hub_labels: Optional[Iterable[str]] = None,
) -> dict[str, list[tuple[str, float]]]:
    """Build a kNN adjacency graph over genre embeddings.

    Args:
        genre_emb: (V, dim) per-genre embedding (rows need not be normalized).
        genre_vocab: V genre labels aligned to genre_emb rows.
        k: max neighbors per genre.
        min_cos: minimum cosine to create an edge.
        hub_labels: labels to exclude as nodes/neighbors (broad genres).
    Returns:
        {label: [(neighbor_label, cos), ...]} for non-hub labels only.
    """
    labels = [str(g) for g in list(genre_vocab)]
    hubs = {str(h).strip().lower() for h in (hub_labels or set())}
    M = np.asarray(genre_emb, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = M / np.maximum(norms, 1e-12)
    keep = [i for i, lab in enumerate(labels) if lab.strip().lower() not in hubs]
    graph: dict[str, list[tuple[str, float]]] = {}
    for i in keep:
        sims = Mn[keep] @ Mn[i]            # cosine to all kept genres
        order = np.argsort(-sims)
        nbrs: list[tuple[str, float]] = []
        for j_local in order:
            j = keep[int(j_local)]
            if j == i:
                continue
            c = float(sims[int(j_local)])
            if c < min_cos:
                break
            nbrs.append((labels[j], c))
            if len(nbrs) >= k:
                break
        graph[labels[i]] = nbrs
    return graph
