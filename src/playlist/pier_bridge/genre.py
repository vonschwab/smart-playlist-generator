"""Genre helper functions extracted from pier_bridge_builder.py (Tier-3.1 PR-1).

All functions here are pure (or near-pure) helpers for genre vector math and
genre-graph traversal. None of them mutate global state or carry a class
instance — they take primitives in, return primitives out.

Extracted verbatim from src/playlist/pier_bridge_builder.py lines 753-1100,
with one signature change: _compute_genre_idf previously took a
PierBridgeConfig instance for two fields (dj_genre_idf_power, dj_genre_idf_norm);
it now takes those two values as primitives so this module has no back-
reference to pier_bridge_builder.PierBridgeConfig (avoiding circular import).
The pier_bridge_builder call sites pass cfg.dj_genre_idf_power and
cfg.dj_genre_idf_norm explicitly.
"""
from __future__ import annotations

import heapq
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from src.genre.similarity import load_yaml_overrides, pairwise_genre_similarity

logger = logging.getLogger(__name__)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 1e-12:
        return vec
    return vec / norm


def _genre_vocab_map(genre_vocab: np.ndarray) -> dict[str, int]:
    return {str(g).strip().lower(): int(i) for i, g in enumerate(genre_vocab)}


def _compute_genre_idf(
    X_genre_raw: np.ndarray,
    *,
    idf_power: float,
    idf_norm: str,
) -> np.ndarray:
    """
    Compute IDF (inverse document frequency) for each genre.

    Formula:
        df[g] = count(tracks where genre[g] > 0)
        idf[g] = log((N + 1) / (df[g] + 1))  # +1 smoothing
        idf = idf ** idf_power
        idf = normalize(idf, method=idf_norm)

    Returns:
        idf: (G,) array where idf[g] ∈ [0, 1] (after normalization)
             High values = rare genres, low values = common genres.
    """
    N, G = X_genre_raw.shape

    # Count tracks per genre (document frequency)
    df = (X_genre_raw > 0).sum(axis=0)  # (G,)

    # Compute raw IDF
    idf = np.log((N + 1) / (df + 1))  # +1 smoothing

    # Apply power scaling
    power = float(idf_power)
    if power != 1.0 and power > 0:
        idf = idf ** power

    # Normalize
    norm_method = str(idf_norm).strip().lower()
    if norm_method == "max1":
        max_val = np.max(idf)
        if max_val > 0:
            idf = idf / max_val  # Scale to [0, 1]
    elif norm_method == "sum1":
        sum_val = np.sum(idf)
        if sum_val > 0:
            idf = idf / sum_val  # Sum to 1.0
    # else: "none" - keep raw values

    return idf


def _apply_idf_weighting(
    genre_vec: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    """
    Apply IDF weighting element-wise and normalize.

    For 1D vector: result = normalize(genre_vec * idf)
    For 2D matrix: result = normalize_rows(genre_vec * idf)
    """
    if genre_vec.ndim == 1:
        # 1D vector
        weighted = genre_vec * idf
        return _normalize_vec(weighted)
    else:
        # 2D matrix (N, G)
        weighted = genre_vec * idf[np.newaxis, :]  # Broadcasting
        # Normalize rows
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return weighted / norms


def _extract_top_genres(
    genre_vec: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Extract top-K genres by weight.

    Args:
        genre_vec: (G,) genre vector (post-IDF if applicable)
        top_k: Number of top genres to extract

    Returns:
        List of (genre_idx, weight) tuples, sorted descending by weight.
    """
    if top_k <= 0 or genre_vec.size == 0:
        return []

    indices = np.argsort(-genre_vec)[:top_k]
    return [(int(i), float(genre_vec[i])) for i in indices if genre_vec[i] > 0]


def _compute_coverage(
    candidate_genre_vec: np.ndarray,
    topk_genres: list[tuple[int, float]],
    threshold: float,
    mode: str = "binary",
) -> float:
    """
    Compute coverage of top-K genres in candidate.

    Phase 3 modes:
    - binary (legacy): fraction of genres "present" (weight >= threshold)
    - weighted: mean of genre weights for top-K genres

    Args:
        candidate_genre_vec: (G,) candidate's genre vector
        topk_genres: List of (genre_idx, weight) from anchor
        threshold: Minimum weight to count as "present" (binary mode only)
        mode: "binary" or "weighted"

    Returns:
        coverage ∈ [0, 1]: coverage score
    """
    if not topk_genres:
        return 0.0

    if mode == "weighted":
        # Weighted mode: mean of genre weights
        weights_sum = 0.0
        for g_idx, _ in topk_genres:
            weights_sum += float(candidate_genre_vec[g_idx])
        return weights_sum / float(len(topk_genres))
    else:  # binary (legacy)
        # Binary mode: fraction of genres present
        present_count = 0
        for g_idx, _ in topk_genres:
            if candidate_genre_vec[g_idx] >= threshold:
                present_count += 1
        return float(present_count) / float(len(topk_genres))


def _compute_coverage_bonus(
    step: int,
    interior_length: int,
    coverage_A: float,
    coverage_B: float,
    coverage_weight: float,
    coverage_power: float,
) -> float:
    """
    Compute coverage bonus with decay schedule.

    Schedule:
        s = step / (interior_length + 1)  # Progress ∈ [0, 1]
        wA = (1 - s) ** power              # Strong near A (s=0)
        wB = s ** power                    # Strong near B (s=1)
        bonus = weight * (wA * coverage_A + wB * coverage_B)

    Args:
        step: Current step in interior (0-indexed)
        interior_length: Total interior length
        coverage_A: Coverage score relative to anchor A
        coverage_B: Coverage score relative to anchor B
        coverage_weight: Multiplier for bonus
        coverage_power: Schedule decay exponent

    Returns:
        bonus ∈ [0, weight] (additive score adjustment)
    """
    if interior_length == 0:
        return 0.0

    s = float(step) / float(interior_length + 1)
    power = float(coverage_power)

    wA = (1.0 - s) ** power
    wB = s ** power

    bonus = float(coverage_weight) * (
        wA * float(coverage_A) + wB * float(coverage_B)
    )

    return bonus


def _select_top_genre_labels(
    g_vec: np.ndarray,
    genre_vocab: np.ndarray,
    *,
    top_n: int,
    min_weight: float,
) -> list[str]:
    if top_n <= 0:
        return []
    if g_vec.size == 0:
        return []
    weights = np.array(g_vec, dtype=float)
    if weights.ndim != 1:
        weights = weights.reshape(-1)
    if not np.isfinite(weights).any():
        return []
    order = np.argsort(-weights)
    labels: list[str] = []
    for idx in order:
        w = float(weights[int(idx)])
        if w < float(min_weight):
            break
        label = str(genre_vocab[int(idx)])
        if label:
            labels.append(label)
        if len(labels) >= int(top_n):
            break
    return labels


def _load_genre_similarity_graph(
    path: Path,
    *,
    min_similarity: float,
) -> dict[str, list[tuple[str, float]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("Failed to load genre similarity graph from %s", path, exc_info=True)
        return {}
    graph: dict[str, list[tuple[str, float]]] = {}
    for src, neighbors in data.items():
        if not isinstance(neighbors, dict):
            continue
        edges: list[tuple[str, float]] = []
        for dst, score in neighbors.items():
            try:
                sim = float(score)
            except Exception:
                continue
            if sim < float(min_similarity):
                continue
            edges.append((str(dst), float(sim)))
        if edges:
            graph[str(src)] = edges
    return graph


def _ensure_genre_similarity_overrides_loaded(path: Path) -> None:
    try:
        load_yaml_overrides(str(path))
    except Exception:
        logger.warning(
            "Failed to load genre similarity YAML overrides from %s", path, exc_info=True
        )


def _shortest_genre_path(
    graph: dict[str, list[tuple[str, float]]],
    start: str,
    goal: str,
    *,
    max_steps: int,
) -> Optional[list[str]]:
    start = str(start)
    goal = str(goal)
    if start == goal:
        return [start]
    if start not in graph or goal not in graph:
        return None
    max_steps = max(1, int(max_steps))
    pq: list[tuple[float, str, list[str]]] = [(0.0, start, [start])]
    best_cost: dict[str, float] = {start: 0.0}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal:
            return path
        if len(path) - 1 >= max_steps:
            continue
        for neighbor, sim in graph.get(node, []):
            edge_cost = 1.0 - float(sim)
            next_cost = cost + edge_cost
            if next_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = next_cost
            heapq.heappush(pq, (next_cost, neighbor, path + [neighbor]))
    return None


def _label_to_genre_vector(
    label: str,
    *,
    genre_vocab: np.ndarray,
    genre_vocab_map: dict[str, int],
) -> Optional[np.ndarray]:
    idx = genre_vocab_map.get(str(label).strip().lower())
    if idx is None:
        return None
    vec = np.zeros((len(genre_vocab),), dtype=float)
    vec[int(idx)] = 1.0
    return vec


def _genre_similarity_score(label_a: str, label_b: str) -> float:
    result = pairwise_genre_similarity(label_a, label_b, use_yaml_overrides=True)
    score = result.score if result.score is not None else 0.0
    return float(score)


def _label_to_smoothed_vector(
    label: str,
    *,
    genre_vocab: np.ndarray,
    genre_vocab_map: dict[str, int],
    top_k: int,
    min_sim: float,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    if top_k <= 0:
        return None, {"nonzero": 0, "top_labels": []}
    scorer = similarity_fn or _genre_similarity_score
    scores: list[tuple[int, float, str]] = []
    for raw in genre_vocab:
        vocab_label = str(raw)
        try:
            sim = float(scorer(label, vocab_label))
        except Exception:
            continue
        if not math.isfinite(sim) or sim < float(min_sim):
            continue
        idx = genre_vocab_map.get(vocab_label.strip().lower())
        if idx is None:
            continue
        scores.append((int(idx), float(sim), vocab_label))
    if not scores:
        return None, {"nonzero": 0, "top_labels": []}
    scores.sort(key=lambda t: t[1], reverse=True)
    scores = scores[: int(top_k)]
    vec = np.zeros((len(genre_vocab),), dtype=float)
    weights = [float(s[1]) for s in scores]
    total = sum(weights)
    top_labels = []
    for idx, sim, vocab_label in scores:
        vec[int(idx)] = float(sim)
        if len(top_labels) < 3:
            weight = float(sim / total) if total > 0 else float(sim)
            top_labels.append({"label": str(vocab_label), "weight": weight})
    return _normalize_vec(vec), {
        "nonzero": int(len(scores)),
        "top_labels": top_labels,
    }
