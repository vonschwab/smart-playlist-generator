"""
Tier-3.1 PR-8: genre waypoint target builder.

Extracted verbatim from pier_bridge_builder.py:
  _fallback_genre_vector  — nearest-neighbour genre vector for a missing pier
  _build_genre_targets    — builds per-step genre target vectors for a segment
                            (vector / ladder / linear modes)
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from src.playlist.pier_bridge.config import PierBridgeConfig, _compute_genre_idf
from src.playlist.pier_bridge.genre import (
    _apply_idf_weighting,
    _genre_vocab_map,
    _label_to_genre_vector,
    _label_to_smoothed_vector,
    _normalize_vec,
    _select_top_genre_labels,
    _shortest_genre_path,
)
from src.playlist.pier_bridge.metrics import _progress_target_curve, _step_fraction

logger = logging.getLogger(__name__)


def _fallback_genre_vector(
    pier_idx: int,
    *,
    X_full_norm: np.ndarray,
    X_genre_norm: np.ndarray,
    k: int,
) -> Optional[np.ndarray]:
    if k <= 0:
        return None
    if X_genre_norm is None:
        return None
    sims = np.dot(X_full_norm, X_full_norm[int(pier_idx)])
    order = np.argsort(-sims)
    collected = []
    for idx in order:
        if int(idx) == int(pier_idx):
            continue
        vec = X_genre_norm[int(idx)]
        if float(np.linalg.norm(vec)) <= 1e-8:
            continue
        collected.append(vec)
        if len(collected) >= int(k):
            break
    if not collected:
        return None
    avg = np.mean(np.stack(collected, axis=0), axis=0)
    return _normalize_vec(avg)


def _build_genre_targets(
    *,
    pier_a: int,
    pier_b: int,
    interior_length: int,
    X_full_norm: np.ndarray,
    X_genre_norm: np.ndarray,
    genre_vocab: Optional[np.ndarray],  # Phase 3 fix: Optional for vector mode
    genre_graph: Optional[dict[str, list[tuple[str, float]]]],
    cfg: PierBridgeConfig,
    warnings: list[dict[str, Any]],
    ladder_diag: Optional[dict[str, Any]] = None,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_idf: Optional[np.ndarray] = None,
) -> Optional[list[np.ndarray]]:
    route_shape = str(cfg.dj_route_shape or "linear").strip().lower()
    if route_shape not in {"linear", "arc", "ladder"}:
        route_shape = "linear"
    if ladder_diag is not None:
        ladder_diag["route_shape"] = route_shape
        ladder_diag.setdefault("ladder_waypoint_labels", [])
        ladder_diag.setdefault("ladder_waypoint_count", 0)
        ladder_diag.setdefault("ladder_waypoint_vector_mode", "onehot")
        ladder_diag.setdefault("ladder_waypoint_vector_stats", [])
    if interior_length <= 0:
        return None
    if X_genre_norm is None:
        return None
    g_a = X_genre_norm[pier_a]
    g_b = X_genre_norm[pier_b]
    missing = []
    if float(np.linalg.norm(g_a)) <= 1e-8:
        fallback = _fallback_genre_vector(
            pier_a, X_full_norm=X_full_norm, X_genre_norm=X_genre_norm, k=int(cfg.dj_waypoint_fallback_k)
        )
        if fallback is not None:
            g_a = fallback
            warnings.append({
                "type": "genre_fallback",
                "scope": "anchor",
                "anchor_id": int(pier_a),
                "fallback": "neighbor_avg",
                "k": int(cfg.dj_waypoint_fallback_k),
            })
        else:
            missing.append(int(pier_a))
    if float(np.linalg.norm(g_b)) <= 1e-8:
        fallback = _fallback_genre_vector(
            pier_b, X_full_norm=X_full_norm, X_genre_norm=X_genre_norm, k=int(cfg.dj_waypoint_fallback_k)
        )
        if fallback is not None:
            g_b = fallback
            warnings.append({
                "type": "genre_fallback",
                "scope": "anchor",
                "anchor_id": int(pier_b),
                "fallback": "neighbor_avg",
                "k": int(cfg.dj_waypoint_fallback_k),
            })
        else:
            missing.append(int(pier_b))
    if missing:
        warnings.append({
            "type": "genre_missing",
            "scope": "segment",
            "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
            "missing_anchor_indices": missing,
        })
        return None

    # === VECTOR MODE (Phase 2): Direct multi-genre interpolation ===
    target_mode = str(cfg.dj_ladder_target_mode or "onehot").strip().lower()
    if target_mode == "vector":
        # Select source matrix
        vector_source = str(cfg.dj_genre_vector_source or "smoothed").strip().lower()
        if vector_source == "raw" and X_genre_raw is not None:
            X_genre_base = X_genre_raw
        elif X_genre_smoothed is not None:
            X_genre_base = X_genre_smoothed
        else:
            # Fall back to X_genre_norm (already normalized smoothed)
            X_genre_base = X_genre_norm

        # Extract anchor vectors
        vA = X_genre_base[pier_a].copy()
        vB = X_genre_base[pier_b].copy()

        # Apply IDF weighting (optional)
        if bool(cfg.dj_genre_use_idf):
            if genre_idf is None and X_genre_raw is not None:
                # Compute IDF on-demand
                genre_idf = _compute_genre_idf(X_genre_raw, cfg)
            if genre_idf is not None:
                vA = _apply_idf_weighting(vA, genre_idf)
                vB = _apply_idf_weighting(vB, genre_idf)
                if ladder_diag is not None:
                    ladder_diag["idf_enabled"] = True
                    ladder_diag["idf_stats"] = {
                        "min": float(np.min(genre_idf)),
                        "median": float(np.median(genre_idf)),
                        "max": float(np.max(genre_idf)),
                    }
            else:
                if ladder_diag is not None:
                    ladder_diag["idf_enabled"] = False
                warnings.append({
                    "type": "genre_idf_unavailable",
                    "scope": "segment",
                    "message": "IDF enabled but X_genre_raw missing; using base weights.",
                })
        else:
            # Normalize without IDF
            vA = _normalize_vec(vA)
            vB = _normalize_vec(vB)
            if ladder_diag is not None:
                ladder_diag["idf_enabled"] = False

        # Interpolate step targets
        g_targets: list[np.ndarray] = []
        for i in range(int(interior_length)):
            if route_shape == "arc":
                frac = _progress_target_curve(i, interior_length, "arc")
            else:
                frac = _step_fraction(i, interior_length)
            g = (1.0 - frac) * vA + frac * vB
            g_targets.append(_normalize_vec(g))

        if ladder_diag is not None:
            ladder_diag["route_shape"] = route_shape
            ladder_diag["ladder_waypoint_vector_mode"] = "vector"
            ladder_diag["vector_source"] = vector_source

        return g_targets

    # === LEGACY MODES: onehot/smoothed (shortest path) ===
    # Phase 3 fix: If genre_vocab is None, fall back to simple interpolation
    if route_shape != "ladder" or not genre_graph or genre_vocab is None:
        g_targets: list[np.ndarray] = []
        for i in range(int(interior_length)):
            if route_shape == "arc":
                frac = _progress_target_curve(i, interior_length, "arc")
            else:
                frac = _step_fraction(i, interior_length)
            g = (1.0 - frac) * g_a + frac * g_b
            g_targets.append(_normalize_vec(g))
        if route_shape == "ladder" and genre_vocab is None:
            warnings.append({
                "type": "genre_vocab_missing",
                "scope": "segment",
                "message": "Genre ladder disabled; genre_vocab missing, falling back to linear drift.",
            })
        return g_targets

    vocab_map = _genre_vocab_map(genre_vocab)
    labels_a = _select_top_genre_labels(
        g_a, genre_vocab, top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight)
    )
    labels_b = _select_top_genre_labels(
        g_b, genre_vocab, top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight)
    )
    if not labels_a or not labels_b:
        warnings.append({
            "type": "genre_ladder_unavailable",
            "scope": "segment",
            "message": "Genre ladder disabled; falling back to linear drift.",
        })
        return _build_genre_targets(
            pier_a=pier_a,
            pier_b=pier_b,
            interior_length=interior_length,
            X_full_norm=X_full_norm,
            X_genre_norm=X_genre_norm,
            genre_vocab=genre_vocab,
            genre_graph=None,
            cfg=replace(cfg, dj_route_shape="linear"),
            warnings=warnings,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
        )

    path_labels = None
    for la in labels_a:
        for lb in labels_b:
            path_labels = _shortest_genre_path(
                genre_graph,
                la,
                lb,
                max_steps=int(cfg.dj_ladder_max_steps),
            )
            if path_labels:
                break
        if path_labels:
            break
    if not path_labels:
        warnings.append({
            "type": "genre_ladder_unavailable",
            "scope": "segment",
            "message": "Genre ladder disabled; falling back to linear drift.",
        })
        return _build_genre_targets(
            pier_a=pier_a,
            pier_b=pier_b,
            interior_length=interior_length,
            X_full_norm=X_full_norm,
            X_genre_norm=X_genre_norm,
            genre_vocab=genre_vocab,
            genre_graph=None,
            cfg=replace(cfg, dj_route_shape="linear"),
            warnings=warnings,
            ladder_diag=ladder_diag,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
        )

    if ladder_diag is not None:
        ladder_diag["ladder_waypoint_labels"] = [str(l) for l in path_labels[:12]]
        ladder_diag["ladder_waypoint_count"] = int(len(path_labels))

    waypoint_vecs: list[np.ndarray] = []
    waypoint_stats: list[dict[str, Any]] = []
    missing_vocab_labels: set[str] = set()
    smoothed_fallback_labels: list[str] = []
    smoothed_used = 0
    for label in path_labels:
        label_str = str(label)
        if label_str.strip().lower() not in vocab_map:
            missing_vocab_labels.add(label_str)
        vec = None
        stats_entry: dict[str, Any] = {"label": label_str, "mode": "onehot"}
        if bool(cfg.dj_ladder_use_smoothed_waypoint_vectors):
            vec, stats = _label_to_smoothed_vector(
                label_str,
                genre_vocab=genre_vocab,
                genre_vocab_map=vocab_map,
                top_k=int(cfg.dj_ladder_smooth_top_k),
                min_sim=float(cfg.dj_ladder_smooth_min_sim),
            )
            if vec is not None:
                stats_entry.update(stats)
                stats_entry["mode"] = "smoothed"
                smoothed_used += 1
            else:
                smoothed_fallback_labels.append(label_str)
        if vec is None:
            vec = _label_to_genre_vector(
                label_str, genre_vocab=genre_vocab, genre_vocab_map=vocab_map
            )
        if vec is None:
            missing_vocab_labels.add(label_str)
            continue
        if stats_entry.get("mode") == "smoothed":
            waypoint_stats.append(stats_entry)
        waypoint_vecs.append(_normalize_vec(vec))
    if missing_vocab_labels:
        missing_list = sorted(missing_vocab_labels)
        warnings.append({
            "type": "genre_ladder_label_unmapped",
            "scope": "segment",
            "message": "Waypoint labels missing from genre_vocab; mapping loss.",
            "missing_labels": missing_list[:12],
            "missing_count": int(len(missing_list)),
        })
    if smoothed_fallback_labels:
        warnings.append({
            "type": "genre_ladder_smoothed_fallback",
            "scope": "segment",
            "message": "Smoothed waypoint vector empty; falling back to one-hot.",
            "labels": smoothed_fallback_labels[:12],
            "fallback_count": int(len(smoothed_fallback_labels)),
        })
    if len(waypoint_vecs) < 2:
        warnings.append({
            "type": "genre_ladder_unavailable",
            "scope": "segment",
            "message": "Genre ladder disabled; falling back to linear drift.",
        })
        return _build_genre_targets(
            pier_a=pier_a,
            pier_b=pier_b,
            interior_length=interior_length,
            X_full_norm=X_full_norm,
            X_genre_norm=X_genre_norm,
            genre_vocab=genre_vocab,
            genre_graph=None,
            cfg=replace(cfg, dj_route_shape="linear"),
            warnings=warnings,
            ladder_diag=ladder_diag,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
        )

    if ladder_diag is not None:
        if bool(cfg.dj_ladder_use_smoothed_waypoint_vectors) and smoothed_used == len(waypoint_vecs):
            ladder_diag["ladder_waypoint_vector_mode"] = "smoothed"
            ladder_diag["ladder_waypoint_vector_stats"] = waypoint_stats
        else:
            ladder_diag["ladder_waypoint_vector_mode"] = "onehot"
            ladder_diag["ladder_waypoint_vector_stats"] = waypoint_stats

    g_targets = []
    steps = int(interior_length)
    for i in range(steps):
        frac = _step_fraction(i, steps)
        scaled = frac * float(len(waypoint_vecs) - 1)
        idx = int(math.floor(scaled))
        if idx >= len(waypoint_vecs) - 1:
            g = waypoint_vecs[-1]
        else:
            local = scaled - float(idx)
            g = (1.0 - local) * waypoint_vecs[idx] + local * waypoint_vecs[idx + 1]
            g = _normalize_vec(g)
        g_targets.append(g)
    return g_targets
