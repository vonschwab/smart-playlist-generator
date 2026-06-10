"""Graph-derived genre-to-genre similarity matrix (Stage 2 of the taxonomy integration).

Consumes only the Stage 1 adapter (src/genre/graph_adapter.py) and emits the
same NPZ contract as src/analyze/genre_similarity.py ({genre_vocab, S, stats})
so artifact building can swap the co-occurrence Jaccard matrix for this one
behind a config flag in Stage 3, with every downstream shape unchanged.

Similarity recipe:
- Direct edges: sim = base[edge_type] + span[edge_type] * (weight * confidence),
  symmetrized by max. Bases/spans are calibrated so the taxonomy's standard
  edge shapes land near the legacy structural-scorer constants (is_a parent ~0.8,
  scene/bridge laterals ~0.5-0.7, family membership ~0.2-0.35).
- Multi-hop: best-path max-product with per-hop decay, up to max_hops edges.
  Review-status nodes participate as intermediates but are not output dimensions.
- Hub guard: paths routed *through* a broad node (family/umbrella) are scaled by
  that node's specificity, and any pair involving a broad endpoint is capped, so
  hub genres cannot glue the matrix together (the IDF lesson, applied to the graph).

Read-only: loads the taxonomy YAML, writes only the requested output files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml

from src.genre.graph_adapter import GenreGraphAdapter

logger = logging.getLogger(__name__)

DEFAULT_EDGE_BASE: dict[str, float] = {
    "is_a": 0.30,
    "family_context": 0.0,
    "scene_adjacent": 0.25,
    "bridge_to": 0.20,
    "fusion_of": 0.25,
    "style_modifier": 0.10,
}
DEFAULT_EDGE_SPAN: dict[str, float] = {
    "is_a": 0.80,
    "family_context": 0.60,
    "scene_adjacent": 0.80,
    "bridge_to": 0.80,
    "fusion_of": 0.80,
    "style_modifier": 0.70,
}


@dataclass(frozen=True)
class GraphSimilarityParams:
    """Tunable knobs for the edge->similarity mapping. Defaults are calibrated
    against the SP3a edge-shape conventions (see taxonomy-growth skill)."""

    edge_base: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_EDGE_BASE))
    edge_span: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_EDGE_SPAN))
    hop_decay: float = 0.85
    max_hops: int = 2
    broad_pair_cap: float = 0.40
    hub_specificity_floor: float = 0.6
    max_offdiag: float = 0.95


@dataclass(frozen=True)
class GraphSimilarityResult:
    genre_vocab: list[str]
    S: np.ndarray
    stats: dict


def build_graph_similarity(
    adapter: GenreGraphAdapter,
    params: Optional[GraphSimilarityParams] = None,
    *,
    include_review: bool = False,
) -> GraphSimilarityResult:
    """Build the genre-to-genre similarity matrix from the taxonomy graph.

    The path computation runs over active + review nodes (review nodes can
    legitimately connect actives); the emitted vocabulary is active-only
    unless include_review is set.
    """
    params = params or GraphSimilarityParams()

    path_vocab = adapter.active_genre_vocabulary(include_review=True)
    index = {name: i for i, name in enumerate(path_vocab)}
    n = len(path_vocab)

    A = np.zeros((n, n), dtype=np.float64)
    skipped_edge_types: dict[str, int] = {}
    edges_used = 0
    for edge in adapter.edges():
        i = index.get(edge.source)
        j = index.get(edge.target)
        if i is None or j is None or i == j:
            continue
        base = params.edge_base.get(edge.edge_type)
        span = params.edge_span.get(edge.edge_type)
        if base is None or span is None:
            skipped_edge_types[edge.edge_type] = skipped_edge_types.get(edge.edge_type, 0) + 1
            continue
        sim = base + span * (edge.weight * edge.confidence)
        sim = float(np.clip(sim, 0.0, params.max_offdiag))
        if sim > A[i, j]:
            A[i, j] = sim
            A[j, i] = sim
        edges_used += 1

    hub_mult = np.ones(n, dtype=np.float64)
    for name, i in index.items():
        node = adapter.node(name)
        if node is not None and node.is_broad:
            floor = max(params.hub_specificity_floor, 1e-9)
            hub_mult[i] = min(1.0, max(0.0, node.specificity_score) / floor)

    best = A.copy()
    current = A
    for _ in range(max(0, int(params.max_hops) - 1)):
        nxt = np.zeros_like(A)
        for k in range(n):
            via_k = current[:, k] * (hub_mult[k] * params.hop_decay)
            if not via_k.any():
                continue
            np.maximum(nxt, np.outer(via_k, A[k, :]), out=nxt)
        current = nxt
        np.maximum(best, current, out=best)

    broad_mask = np.array(
        [bool(adapter.node(name) and adapter.node(name).is_broad) for name in path_vocab]
    )
    if broad_mask.any():
        capped = np.minimum(best, params.broad_pair_cap)
        best[broad_mask, :] = capped[broad_mask, :]
        best[:, broad_mask] = capped[:, broad_mask]

    np.clip(best, 0.0, params.max_offdiag, out=best)
    np.fill_diagonal(best, 1.0)

    out_vocab = adapter.active_genre_vocabulary(include_review=include_review)
    keep = [index[name] for name in out_vocab]
    S = best[np.ix_(keep, keep)].astype(np.float32)

    off_diag = S[~np.eye(len(out_vocab), dtype=bool)] if len(out_vocab) > 1 else np.zeros(0)
    stats = {
        "source": "graph",
        "taxonomy_version": adapter.taxonomy_version,
        "genres_kept": len(out_vocab),
        "path_nodes": n,
        "edges_used": edges_used,
        "skipped_edge_types": skipped_edge_types,
        "include_review": include_review,
        "nonzero_offdiag_fraction": float((off_diag > 0).mean()) if off_diag.size else 0.0,
        "params": {
            "edge_base": dict(params.edge_base),
            "edge_span": dict(params.edge_span),
            "hop_decay": params.hop_decay,
            "max_hops": params.max_hops,
            "broad_pair_cap": params.broad_pair_cap,
            "hub_specificity_floor": params.hub_specificity_floor,
            "max_offdiag": params.max_offdiag,
        },
    }
    logger.info(
        "Graph similarity: %d genres (%d path nodes), %d edges used, %.1f%% off-diag nonzero",
        len(out_vocab), n, edges_used, 100.0 * stats["nonzero_offdiag_fraction"],
    )
    return GraphSimilarityResult(genre_vocab=list(out_vocab), S=S, stats=stats)


def npz_similarity_source(path: str | Path) -> Optional[str]:
    """Read which generator produced a genre-similarity NPZ.

    Returns "graph" for graph-derived matrices, "cooccurrence" for matrices
    without a source stamp (the legacy Jaccard builder), or None when the file
    does not exist or cannot be read.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        stats = data["stats"].item() if "stats" in data.files else {}
    except Exception:
        logger.warning("Could not read similarity provenance from %s", path, exc_info=True)
        return None
    return str(stats.get("source") or "cooccurrence")


def save_graph_similarity_npz(result: GraphSimilarityResult, out_path: str | Path) -> Path:
    """Write the NPZ in the same key layout as build_genre_similarity_matrix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        genre_vocab=np.array(result.genre_vocab, dtype=object),
        S=result.S,
        stats=result.stats,
    )
    logger.info("Saved graph genre similarity to %s (G=%d)", out_path, len(result.genre_vocab))
    return out_path


def export_neighbor_yaml(
    result: GraphSimilarityResult,
    out_path: str | Path,
    *,
    min_sim: float = 0.05,
    top_k: int = 12,
) -> Path:
    """Write a {genre: {neighbor: sim}} YAML in the data/genre_similarity.yaml
    schema, loadable by the DJ-bridge label scorer and ladder-graph reader."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = result.genre_vocab
    data: dict[str, dict[str, float]] = {}
    for i, name in enumerate(vocab):
        row = result.S[i].astype(float)
        order = np.argsort(-row)
        neighbors: dict[str, float] = {}
        for j in order:
            if int(j) == i:
                continue
            sim = float(row[int(j)])
            if sim < min_sim or len(neighbors) >= top_k:
                break
            neighbors[vocab[int(j)]] = round(sim, 4)
        if neighbors:
            data[name] = neighbors
    header = (
        f"# GENERATED by scripts/build_graph_genre_similarity.py — do not hand-edit.\n"
        f"# taxonomy_version: {result.stats.get('taxonomy_version')}\n"
        f"# min_sim={min_sim} top_k={top_k}\n"
    )
    out_path.write_text(
        header + yaml.safe_dump(data, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Saved graph genre-neighbor YAML to %s (%d genres)", out_path, len(data))
    return out_path
