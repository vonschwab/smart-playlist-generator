"""Progress + diagnostic metrics (Tier-3.1 PR-2).

Statistical helpers and per-segment progress/source-attribution counters.
All pure — no PierBridgeConfig coupling, no shared state.

Extracted from pier_bridge_builder.py.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def _dist(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _step_fraction(step_idx: int, steps: int) -> float:
    """Shared step fraction convention for progress + waypoint targets."""
    if steps <= 0:
        return 0.0
    return float(step_idx + 1) / float(steps + 1)


def _progress_target_curve(step_idx: int, steps: int, shape: str) -> float:
    if steps <= 0:
        return 0.0
    shape = str(shape or "linear").strip().lower()
    if shape not in {"linear", "arc"}:
        shape = "linear"
    base = _step_fraction(step_idx, steps)
    if shape == "arc":
        return 0.5 - 0.5 * math.cos(math.pi * base)
    return base


def _progress_arc_loss_value(err: float, loss: str, huber_delta: float) -> float:
    err = float(max(0.0, err))
    loss = str(loss or "abs").strip().lower()
    if loss == "squared":
        return float(err * err)
    if loss == "huber":
        delta = float(huber_delta) if math.isfinite(float(huber_delta)) else 0.1
        if delta <= 0:
            delta = 0.1
        if err <= delta:
            return float(0.5 * err * err)
        return float(delta * (err - 0.5 * delta))
    return float(err)


def _compute_progress_tracking_metrics(
    *,
    path: list[int],
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    shape: str,
) -> dict[str, Optional[float]]:
    if not path:
        return {
            "mean_abs_dev": None,
            "p50_abs_dev": None,
            "p90_abs_dev": None,
            "max_progress_jump": None,
        }

    vec_a_full = X_full_norm[pier_a]
    vec_b_full = X_full_norm[pier_b]
    d = vec_b_full - vec_a_full
    denom = float(np.dot(d, d))
    if (not math.isfinite(denom)) or denom <= 1e-12:
        return {
            "mean_abs_dev": None,
            "p50_abs_dev": None,
            "p90_abs_dev": None,
            "max_progress_jump": None,
        }

    steps = len(path)
    devs: list[float] = []
    max_jump = None
    last_t = None
    for idx, track_idx in enumerate(path):
        t_raw = float(np.dot((X_full_norm[int(track_idx)] - vec_a_full), d) / denom)
        if not math.isfinite(t_raw):
            continue
        t = float(max(0.0, min(1.0, t_raw)))
        target_t = _progress_target_curve(idx, steps, shape)
        devs.append(abs(t - target_t))
        if last_t is not None:
            jump = t - last_t
            if max_jump is None or jump > max_jump:
                max_jump = jump
        last_t = t

    if not devs:
        return {
            "mean_abs_dev": None,
            "p50_abs_dev": None,
            "p90_abs_dev": None,
            "max_progress_jump": None,
        }

    dev_arr = np.array(devs, dtype=float)
    return {
        "mean_abs_dev": float(np.mean(dev_arr)),
        "p50_abs_dev": float(np.percentile(dev_arr, 50)),
        "p90_abs_dev": float(np.percentile(dev_arr, 90)),
        "max_progress_jump": (float(max_jump) if max_jump is not None else None),
    }


def _compute_pool_overlap_metrics(
    baseline_candidates: List[int],
    union_candidates: List[int],
) -> Dict[str, Any]:
    baseline_set = set(int(i) for i in baseline_candidates)
    union_set = set(int(i) for i in union_candidates)
    intersection = baseline_set & union_set
    union_all = baseline_set | union_set
    jaccard = float(len(intersection) / len(union_all)) if union_all else 0.0
    return {
        "pool_overlap_jaccard": float(jaccard),
        "pool_overlap_baseline_only": int(len(baseline_set - union_set)),
        "pool_overlap_union_only": int(len(union_set - baseline_set)),
        "pool_overlap_intersection": int(len(intersection)),
        "pool_overlap_baseline_size": int(len(baseline_set)),
        "pool_overlap_union_size": int(len(union_set)),
    }


def _compute_chosen_source_counts(
    path: List[int],
    *,
    sources: Optional[Dict[str, Set[int]]] = None,
    baseline_pool: Optional[Set[int]] = None,
    log_per_track: bool = False,
) -> Dict[str, int]:
    """
    Compute source counts for chosen tracks (Phase 3: membership tracking).

    Phase 3 enhancement: Track all pool memberships (not just priority-based).
    Returns both exclusive counts (for backward compat) and membership flags.

    For each track:
    - in_local: track is in local pool
    - in_toward: track is in toward pool
    - in_genre: track is in genre pool

    Exclusive counts (legacy):
    - Priority order: genre > toward > local > baseline_only
    """
    sources = sources or {}
    local = sources.get("local", set())
    toward = sources.get("toward", set())
    genre = sources.get("genre", set())

    # Legacy exclusive counts (for backward compatibility)
    counts = {
        "chosen_from_local_count": 0,
        "chosen_from_toward_count": 0,
        "chosen_from_genre_count": 0,
        "chosen_from_baseline_only_count": 0,
    }

    # Phase 3: Membership-based counts (all overlaps tracked)
    membership_counts = {
        "local_only": 0,
        "toward_only": 0,
        "genre_only": 0,
        "local+toward": 0,
        "local+genre": 0,
        "toward+genre": 0,
        "local+toward+genre": 0,
        "baseline_only": 0,
    }

    for step, idx in enumerate(path):
        idx = int(idx)

        # Check membership in each pool
        in_local = idx in local
        in_toward = idx in toward
        in_genre = idx in genre

        # Task C: Per-track membership logging
        if log_per_track:
            memberships = []
            if in_local:
                memberships.append("L")
            if in_toward:
                memberships.append("T")
            if in_genre:
                memberships.append("G")
            if not memberships and baseline_pool is not None and idx in baseline_pool:
                memberships.append("B")
            logger.info(
                "    [Track %d] idx=%d pools=%s",
                step, idx, "+".join(memberships) if memberships else "NONE",
            )

        # Legacy exclusive assignment (priority-based)
        if in_genre:
            counts["chosen_from_genre_count"] += 1
        elif in_toward:
            counts["chosen_from_toward_count"] += 1
        elif in_local:
            counts["chosen_from_local_count"] += 1
        elif baseline_pool is not None and idx in baseline_pool:
            counts["chosen_from_baseline_only_count"] += 1

        # Phase 3: Membership-based (all overlaps)
        if in_local and in_toward and in_genre:
            membership_counts["local+toward+genre"] += 1
        elif in_local and in_toward:
            membership_counts["local+toward"] += 1
        elif in_local and in_genre:
            membership_counts["local+genre"] += 1
        elif in_toward and in_genre:
            membership_counts["toward+genre"] += 1
        elif in_local:
            membership_counts["local_only"] += 1
        elif in_toward:
            membership_counts["toward_only"] += 1
        elif in_genre:
            membership_counts["genre_only"] += 1
        elif baseline_pool is not None and idx in baseline_pool:
            membership_counts["baseline_only"] += 1

    # Merge into single dict
    counts.update(membership_counts)
    return counts
