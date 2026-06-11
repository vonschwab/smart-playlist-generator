"""Taxonomy-backed genre-arc steering provider.

Re-points pier-bridge genre steering off the dense PMI-SVD embedding and onto the
curated SP3a taxonomy graph. The taxonomy's hub damping / broad-pair caps let
broad genres act as connective tissue in the arc without collapsing onto them.

This wraps the build-time graph similarity (`src/genre/graph_similarity.py`) for
two runtime uses:
  - `arc_adjacency()` : canonical-name adjacency `{genre: [(neighbor, sim), ...]}`
    consumed by `_shortest_genre_path` for ladder routing.
  - `similarity(a, b)`: hub-damped taxonomy similarity, injected as the scorer
    that builds smoothed waypoint vectors over the artifact genre vocabulary.

Requires NO per-track taxonomy assignments (distinct from the dormant
`genre_graph.source: layered` runtime, which needs leaf/family/bridge matrices
baked into the artifact and is blocked on SP2/SP3 coverage).
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np

from src.genre.graph_adapter import load_graph_adapter
from src.genre.graph_similarity import build_graph_similarity
from src.playlist.pier_bridge.genre import (
    _genre_vocab_map,
    _label_to_smoothed_vector,
    _normalize_vec,
    _select_top_genre_labels,
    _shortest_genre_path,
)
from src.playlist.pier_bridge.metrics import _step_fraction

logger = logging.getLogger(__name__)


class TaxonomySteering:
    """Runtime view over the graph-derived genre similarity for arc steering."""

    def __init__(self, genre_vocab: list[str], S: np.ndarray, adapter) -> None:
        self._vocab = list(genre_vocab)
        self._index = {g: i for i, g in enumerate(self._vocab)}
        self._S = np.asarray(S, dtype=np.float32)
        self._adapter = adapter
        self._canon_cache: dict[str, Optional[str]] = {}
        self._canon_idx_cache: dict[str, Optional[int]] = {}
        self._adjacency: Optional[dict[str, list[tuple[str, float]]]] = None

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    def canonical_label(self, token: str) -> Optional[str]:
        """Canonical taxonomy genre name for a raw tag, or None (facet/reject/unknown)."""
        key = str(token)
        if key in self._canon_cache:
            return self._canon_cache[key]
        res = self._adapter.canonicalize_tag(key)
        canon = res.canonical if res.resolution in ("canonical", "alias") else None
        self._canon_cache[key] = canon
        return canon

    def _canon_index(self, token: str) -> Optional[int]:
        key = str(token)
        if key in self._canon_idx_cache:
            return self._canon_idx_cache[key]
        canon = self.canonical_label(key)
        idx = self._index.get(canon) if canon is not None else None
        self._canon_idx_cache[key] = idx
        return idx

    def is_broad(self, canonical: str) -> bool:
        """True if the canonical genre is a broad node (family/umbrella)."""
        node = self._adapter.node(str(canonical))
        return bool(node.is_broad) if node is not None else False

    def similarity(self, a: str, b: str) -> float:
        """Hub-damped taxonomy similarity in [0, 1]; 0.0 if either tag is uncovered."""
        ia = self._canon_index(a)
        ib = self._canon_index(b)
        if ia is None or ib is None:
            return 0.0
        return float(self._S[ia, ib])

    def arc_adjacency(
        self, *, min_sim: float = 0.05, top_k: int = 0
    ) -> dict[str, list[tuple[str, float]]]:
        """Canonical-name adjacency for ladder routing (cached).

        top_k=0 (default) means no cap — include every neighbor above min_sim.
        The hub-damped S matrix already assigns low similarity to broad genres,
        so Dijkstra avoids them via cost naturally.  Truncating at top_k also
        removes backbone edges (e.g. new_wave→rock at sim=0.178) when 12 tight
        cluster neighbors rank above them, causing scenic routing.
        """
        if self._adjacency is not None:
            return self._adjacency
        adj: dict[str, list[tuple[str, float]]] = {}
        S = self._S
        _top_k = int(top_k)
        for i, g in enumerate(self._vocab):
            row = S[i]
            order = np.argsort(-row)
            nbrs: list[tuple[str, float]] = []
            for j in order:
                jj = int(j)
                if jj == i:
                    continue
                w = float(row[jj])
                if w < float(min_sim):
                    break
                nbrs.append((self._vocab[jj], w))
                if _top_k > 0 and len(nbrs) >= _top_k:
                    break
            adj[g] = nbrs
        self._adjacency = adj
        return adj


@lru_cache(maxsize=4)
def get_taxonomy_steering(taxonomy_path: Optional[str] = None) -> TaxonomySteering:
    """Load + build the taxonomy steering provider once per process (cached)."""
    adapter = load_graph_adapter(taxonomy_path)
    result = build_graph_similarity(adapter)
    logger.info(
        "Taxonomy steering ready: %d canonical genres (taxonomy v%s)",
        len(result.genre_vocab),
        result.stats.get("taxonomy_version"),
    )
    return TaxonomySteering(list(result.genre_vocab), result.S, adapter)


def _canonical_pier_labels(
    g_vec: np.ndarray,
    genre_vocab: np.ndarray,
    steering: TaxonomySteering,
    *,
    top_labels: int,
    min_label_weight: float,
) -> list[str]:
    """Top genre tags of a pier, canonicalized + restricted to routable nodes."""
    raw = _select_top_genre_labels(
        g_vec, genre_vocab, top_n=top_labels, min_weight=min_label_weight
    )
    # Prefer SPECIFIC genres over broad umbrellas/families as arc endpoints, so the
    # arc does not collapse to a hub like "rock" just because it has the highest raw
    # weight. Within each group, preserve raw-weight order.
    specific: list[str] = []
    broad: list[str] = []
    seen: set[str] = set()
    for lab in raw:
        canon = steering.canonical_label(lab)
        if canon is None or canon not in steering._index:
            continue  # uncovered / not routable in the taxonomy graph
        if canon in seen:
            continue
        seen.add(canon)
        (broad if steering.is_broad(canon) else specific).append(canon)
    return specific + broad


def _filter_path_by_mass(
    path: list[str],
    track_counts: Optional[dict[str, int]],
    min_mass: int,
) -> list[str]:
    """Keep endpoints always; strip intermediate nodes with fewer than min_mass tracks."""
    if track_counts is None or min_mass <= 0 or len(path) <= 2:
        return path
    filtered = [path[0]]
    for node in path[1:-1]:
        if track_counts.get(node, 0) >= min_mass:
            filtered.append(node)
    filtered.append(path[-1])
    return filtered


def build_taxonomy_genre_targets(
    *,
    pier_a: int,
    pier_b: int,
    interior_length: int,
    X_genre_raw: np.ndarray,
    genre_vocab: np.ndarray,
    steering: TaxonomySteering,
    top_labels: int = 5,
    min_label_weight: float = 0.05,
    smooth_top_k: int = 10,
    smooth_min_sim: float = 0.2,
    max_steps: int = 6,
    genre_track_counts: Optional[dict[str, int]] = None,
    min_waypoint_mass: int = 0,
    ladder_diag: Optional[dict] = None,
) -> Optional[list[np.ndarray]]:
    """Per-step genre-arc targets routed through the taxonomy graph.

    Walks a damped taxonomy shortest path from pier_a's genre to pier_b's, then
    converts each waypoint to a taxonomy-smoothed vector over the *artifact* genre
    vocabulary (so the beam arc vote scores candidates by taxonomy similarity).

    Returns None when neither pier has a canonicalizable genre (caller should fall
    back to dense steering for the segment).
    """
    if interior_length <= 0 or X_genre_raw is None:
        return None
    vocab_arr = np.asarray(genre_vocab, dtype=object)

    canon_a = _canonical_pier_labels(
        X_genre_raw[pier_a], vocab_arr, steering,
        top_labels=top_labels, min_label_weight=min_label_weight,
    )
    canon_b = _canonical_pier_labels(
        X_genre_raw[pier_b], vocab_arr, steering,
        top_labels=top_labels, min_label_weight=min_label_weight,
    )
    if not canon_a or not canon_b:
        return None

    adjacency = steering.arc_adjacency()
    path: Optional[list[str]] = None
    for la in canon_a:
        for lb in canon_b:
            path = _shortest_genre_path(adjacency, la, lb, max_steps=max_steps)
            if path:
                break
        if path:
            break
    if not path:
        # No taxonomy path between the piers' genres: use a direct two-rung ladder.
        path = [canon_a[0], canon_b[0]]

    path = _filter_path_by_mass(path, genre_track_counts, min_waypoint_mass)

    vocab_map = _genre_vocab_map(vocab_arr)
    waypoint_vecs: list[np.ndarray] = []
    used_labels: list[str] = []
    for label in path:
        vec, _stats = _label_to_smoothed_vector(
            label,
            genre_vocab=vocab_arr,
            genre_vocab_map=vocab_map,
            top_k=int(smooth_top_k),
            min_sim=float(smooth_min_sim),
            similarity_fn=steering.similarity,
        )
        if vec is not None and float(np.linalg.norm(vec)) > 1e-12:
            waypoint_vecs.append(_normalize_vec(vec))
            used_labels.append(str(label))

    if not waypoint_vecs:
        return None

    if ladder_diag is not None:
        ladder_diag["taxonomy_waypoint_labels"] = list(used_labels[:12])
        ladder_diag["taxonomy_waypoint_count"] = int(len(used_labels))

    n = len(waypoint_vecs)
    targets: list[np.ndarray] = []
    for i in range(int(interior_length)):
        frac = _step_fraction(i, int(interior_length))
        scaled = frac * float(n - 1)
        idx = int(math.floor(scaled))
        if idx >= n - 1:
            targets.append(waypoint_vecs[-1])
        else:
            local = scaled - float(idx)
            g = (1.0 - local) * waypoint_vecs[idx] + local * waypoint_vecs[idx + 1]
            targets.append(_normalize_vec(g))
    return targets
