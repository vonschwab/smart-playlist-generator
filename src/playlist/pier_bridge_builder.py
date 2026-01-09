"""
Pier + Bridge Playlist Builder
==============================

A new playlist ordering strategy where:
- Each seed track is a fixed "pier"
- Bridge segments connect consecutive piers
- No repair pass after ordering

Key features:
- Candidate pool deduped BEFORE ordering (no duplicate songs by normalized artist+title)
- Genre gating stays enabled with hard floors (no relaxation)
- Global used_track_ids prevents duplicates across segments
- One track per artist per segment enforced during beam search
- Cross-segment min_gap enforced during generation via boundary-aware constraints
- No post-order filtering or dropping (guarantees exact length)
- Single seed mode: seed acts as both start AND end pier, creating an arc structure
- Seed artist is allowed in bridges with same constraints as other artists
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from src.genre.similarity import pairwise_genre_similarity, load_yaml_overrides
from src.features.artifacts import ArtifactBundle, get_sonic_matrix
from src.title_dedupe import normalize_title_for_dedupe, normalize_artist_key
from src.string_utils import sanitize_for_logging
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    resolve_artist_identity_keys,
    format_identity_keys_for_logging,
)
from src.playlist.config import resolve_pier_bridge_tuning as _resolve_pier_bridge_tuning_cfg
from src.playlist.run_audit import InfeasibleHandlingConfig, RunAuditConfig, RunAuditEvent, now_utc_iso

# Phase 3 extracted modules
from src.playlist.scoring import (
    compute_transition_score as _compute_transition_score_extracted,
    compute_bridgeability_score as _compute_bridgeability_score_extracted,
)
from src.playlist.segment_pool_builder import (
    SegmentCandidatePoolBuilder,
    SegmentPoolConfig,
)
from src.playlist.pier_bridge_diagnostics import (
    SegmentDiagnostics as _SegmentDiagnosticsExtracted,
    PierBridgeDiagnosticsCollector,
)

logger = logging.getLogger(__name__)


@dataclass
class PierBridgeConfig:
    """Configuration for pier + bridge playlist builder."""
    # NOTE: Defaults represent the recommended "dynamic" mode behavior. Narrow
    # mode defaults are resolved by the DS pipeline config layer.
    transition_floor: float = 0.35
    bridge_floor: float = 0.03  # min(simA, simB) for bridge candidates
    center_transitions: bool = False  # if True, mean-center transition mats and rescale sims to [0,1]
    transition_weights: Optional[tuple[float, float, float]] = None  # (rhythm, timbre, harmony)
    sonic_variant: Optional[str] = None  # sonic sim space for bridge gating/endpoint sims
    initial_neighbors_m: int = 100
    initial_bridge_helpers: int = 50
    max_neighbors_m: int = 400
    max_bridge_helpers: int = 200
    initial_beam_width: int = 20
    max_beam_width: int = 100
    max_expansion_attempts: int = 4
    eta_destination_pull: float = 0.10
    # Transition scoring weights
    weight_end_start: float = 0.70
    weight_mid_mid: float = 0.15
    weight_full_full: float = 0.15
    # Bridge scoring weights
    weight_bridge: float = 0.6
    weight_transition: float = 0.4
    genre_tiebreak_weight: float = 0.05
    # Soft genre penalty (does not gate candidates): if edge_genre < threshold,
    # multiply the edge score by (1 - strength).
    genre_penalty_threshold: float = 0.20
    genre_penalty_strength: float = 0.10
    # Medium-firm duration penalty: asymmetric penalty for candidates longer than pier reference
    # (does not gate candidates, but significantly reduces score for long tracks)
    duration_penalty_enabled: bool = True
    duration_penalty_weight: float = 0.30
    # Segment candidate pool strategy:
    # - "segment_scored": score candidates jointly vs (pierA,pierB) and take top-K
    # - "legacy": neighbors(A) ∪ neighbors(B) ∪ helpers (debug/compat only)
    segment_pool_strategy: str = "segment_scored"
    segment_pool_max: int = 400
    max_segment_pool_max: int = 1200
    # Progress model (A→B) to avoid "teleporting" / bouncing.
    progress_enabled: bool = True
    progress_monotonic_epsilon: float = 0.05
    progress_penalty_weight: float = 0.15
    # Interior artist policies (configured/wired by pipeline for legacy --artist runs).
    disallow_pier_artists_in_interiors: bool = False
    disallow_seed_artist_in_interiors: bool = False
    # Experiment-only bridge scoring (dry-run/audit only; production disabled).
    experiment_bridge_scoring_enabled: bool = False
    experiment_bridge_min_weight: float = 0.25
    experiment_bridge_balance_weight: float = 0.15
    # Progress arc scoring (feature-flagged; default disabled).
    progress_arc_enabled: bool = False
    progress_arc_weight: float = 0.25
    progress_arc_shape: str = "linear"
    progress_arc_tolerance: float = 0.0
    progress_arc_loss: str = "abs"
    progress_arc_huber_delta: float = 0.10
    progress_arc_max_step: Optional[float] = None
    progress_arc_max_step_mode: str = "penalty"
    progress_arc_max_step_penalty: float = 0.25
    progress_arc_autoscale_enabled: bool = False
    progress_arc_autoscale_min_distance: float = 0.05
    progress_arc_autoscale_distance_scale: float = 0.50
    progress_arc_autoscale_per_step_scale: bool = False
    # Optional genre tie-break band for penalty application (default off).
    genre_tie_break_band: Optional[float] = None
    # DJ-style genre bridging (opt-in; default disabled).
    dj_bridging_enabled: bool = False
    dj_seed_ordering: str = "auto"  # auto | fixed
    dj_anchors_must_include_all: bool = True
    dj_route_shape: str = "linear"  # linear | arc | ladder (MVP uses linear)
    dj_waypoint_weight: float = 0.15
    dj_waypoint_floor: float = 0.20
    dj_waypoint_penalty: float = 0.10
    dj_waypoint_tie_break_band: Optional[float] = None
    dj_waypoint_cap: float = 0.05
    dj_seed_ordering_weight_sonic: float = 0.60
    dj_seed_ordering_weight_genre: float = 0.20
    dj_seed_ordering_weight_bridge: float = 0.20
    dj_pooling_strategy: str = "baseline"  # baseline | dj_union
    dj_pooling_k_local: int = 200
    dj_pooling_k_toward: int = 80
    dj_pooling_k_genre: int = 80
    dj_pooling_k_union_max: int = 900
    dj_pooling_step_stride: int = 1
    dj_pooling_cache_enabled: bool = True
    dj_pooling_debug_compare_baseline: bool = False
    dj_allow_detours_when_far: bool = True
    dj_far_threshold_sonic: float = 0.45
    dj_far_threshold_genre: float = 0.60
    dj_far_threshold_connector_scarcity: float = 0.10
    dj_connector_bias_enabled: bool = True
    dj_connector_max_per_segment_linear: int = 1
    dj_connector_max_per_segment_adventurous: int = 3
    dj_ladder_top_labels: int = 5
    dj_ladder_min_label_weight: float = 0.05
    dj_ladder_min_similarity: float = 0.20
    dj_ladder_max_steps: int = 6
    dj_ladder_use_smoothed_waypoint_vectors: bool = False
    dj_ladder_smooth_top_k: int = 10
    dj_ladder_smooth_min_sim: float = 0.20
    dj_waypoint_fallback_k: int = 25
    # Genre vector mode + IDF + Coverage (Phase 2)
    dj_ladder_target_mode: str = "onehot"  # "onehot" | "vector"
    dj_genre_vector_source: str = "smoothed"  # "smoothed" | "raw"
    dj_genre_use_idf: bool = False
    dj_genre_idf_power: float = 1.0
    dj_genre_idf_norm: str = "max1"  # "max1" | "sum1" | "none"
    dj_genre_use_coverage: bool = False
    dj_genre_coverage_top_k: int = 8
    dj_genre_coverage_weight: float = 0.15
    dj_genre_coverage_power: float = 2.0
    dj_genre_presence_threshold: float = 0.01
    dj_micro_piers_enabled: bool = False
    dj_micro_piers_max: int = 1
    dj_micro_piers_topk: int = 5
    dj_micro_piers_candidate_source: str = "union_pool"
    dj_micro_piers_selection_metric: str = "max_min_sim"
    dj_relaxation_enabled: bool = False
    dj_relaxation_max_attempts: int = 4
    dj_relaxation_emit_warnings: bool = True
    dj_relaxation_allow_floor_relaxation: bool = False
    # DJ Bridging Diagnostics (opt-in, no behavior change)
    dj_diagnostics_waypoint_rank_impact_enabled: bool = False
    dj_diagnostics_waypoint_rank_sample_steps: int = 3
    dj_diagnostics_pool_verbose: bool = False  # Phase 3 fix: Verbose pool breakdown logging
    # Phase 3: Waypoint delta mode + squashing
    dj_waypoint_delta_mode: str = "absolute"  # "absolute" (legacy) | "centered" (Phase 3)
    dj_waypoint_centered_baseline: str = "median"  # "median" | "mean" (for centered mode)
    dj_waypoint_squash: str = "none"  # "none" (hard cap) | "tanh" (smooth squashing)
    dj_waypoint_squash_alpha: float = 4.0  # Alpha for tanh squashing
    # Phase 3: Coverage enhancements
    dj_coverage_presence_source: str = "same"  # "same" (use scoring matrix) | "raw" (use raw genres)
    dj_coverage_mode: str = "binary"  # "binary" (0/1 count) | "weighted" (mean weights)


# Backward compatibility: SegmentDiagnostics now imported from extracted module
# Kept here as alias for existing code
SegmentDiagnostics = _SegmentDiagnosticsExtracted


@dataclass
class PierBridgeResult:
    """Result of pier + bridge playlist construction."""
    track_ids: List[str]
    track_indices: List[int]
    seed_positions: List[int]  # positions of seeds in final playlist
    segment_diagnostics: List[SegmentDiagnostics]
    stats: Dict[str, Any]
    success: bool = True
    failure_reason: Optional[str] = None
    bridge_debug: list = field(default_factory=list)


def resolve_pier_bridge_tuning(
    overrides: Optional[dict],
    mode: str,
) -> dict:
    """Backward-compatible wrapper around the canonical resolver in `src.playlist.config`."""
    similarity_floor = 0.0
    if isinstance(overrides, dict):
        cand = overrides.get("candidate_pool", {}) or {}
        if isinstance(cand, dict) and isinstance(cand.get("similarity_floor"), (int, float)):
            similarity_floor = float(cand.get("similarity_floor"))

    tuning, _ = _resolve_pier_bridge_tuning_cfg(
        mode=str(mode).strip().lower(),  # type: ignore[arg-type]
        similarity_floor=float(similarity_floor),
        overrides=overrides if isinstance(overrides, dict) else None,
    )
    return {
        "transition_floor": float(tuning.transition_floor),
        "bridge_floor": float(tuning.bridge_floor),
        "weight_bridge": float(tuning.weight_bridge),
        "weight_transition": float(tuning.weight_transition),
        "genre_tiebreak_weight": float(tuning.genre_tiebreak_weight),
        "genre_penalty_threshold": float(tuning.genre_penalty_threshold),
        "genre_penalty_strength": float(tuning.genre_penalty_strength),
    }


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2 normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_transition_score(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> float:
    """
    Compute multi-segment transition score from track A to track B.

    score = w_end_start * cos(end(A), start(B))
          + w_mid_mid * cos(mid(A), mid(B))
          + w_full_full * cos(full(A), full(B))
    """
    # NOTE: X_* matrices are expected to be row L2-normalized so dot() == cosine.
    sim_full = float(np.dot(X_full[idx_a], X_full[idx_b]))

    # End-start similarity (use full as fallback)
    if X_end is not None and X_start is not None:
        sim_end_start = float(np.dot(X_end[idx_a], X_start[idx_b]))
    else:
        sim_end_start = sim_full

    # Mid-mid similarity (use full as fallback)
    if X_mid is not None:
        sim_mid = float(np.dot(X_mid[idx_a], X_mid[idx_b]))
    else:
        sim_mid = sim_full

    if cfg.center_transitions:
        # When centering is enabled, rescale cosine sims from [-1,1] to [0,1]
        sim_full = (sim_full + 1.0) / 2.0
        sim_end_start = (sim_end_start + 1.0) / 2.0
        sim_mid = (sim_mid + 1.0) / 2.0

    return (
        cfg.weight_end_start * sim_end_start
        + cfg.weight_mid_mid * sim_mid
        + cfg.weight_full_full * sim_full
    )


def _compute_transition_score_raw_and_transformed(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> tuple[float, float]:
    """
    Return (raw, transformed) transition scores where "transformed" matches
    `_compute_transition_score()`, and "raw" is before optional centering/rescale.
    """
    sim_full_raw = float(np.dot(X_full[idx_a], X_full[idx_b]))
    if X_end is not None and X_start is not None:
        sim_end_start_raw = float(np.dot(X_end[idx_a], X_start[idx_b]))
    else:
        sim_end_start_raw = sim_full_raw
    if X_mid is not None:
        sim_mid_raw = float(np.dot(X_mid[idx_a], X_mid[idx_b]))
    else:
        sim_mid_raw = sim_full_raw

    raw = (
        cfg.weight_end_start * sim_end_start_raw
        + cfg.weight_mid_mid * sim_mid_raw
        + cfg.weight_full_full * sim_full_raw
    )

    if not cfg.center_transitions:
        return raw, raw

    sim_full = (sim_full_raw + 1.0) / 2.0
    sim_end_start = (sim_end_start_raw + 1.0) / 2.0
    sim_mid = (sim_mid_raw + 1.0) / 2.0
    transformed = (
        cfg.weight_end_start * sim_end_start
        + cfg.weight_mid_mid * sim_mid
        + cfg.weight_full_full * sim_full
    )
    return raw, transformed


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

    for idx in path:
        idx = int(idx)

        # Check membership in each pool
        in_local = idx in local
        in_toward = idx in toward
        in_genre = idx in genre

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


def _summarize_candidates_for_audit(
    *,
    candidates: list[int],
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    X_full_tr_norm: np.ndarray,
    X_start_tr_norm: Optional[np.ndarray],
    X_mid_tr_norm: Optional[np.ndarray],
    X_end_tr_norm: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    bundle: ArtifactBundle,
    internal_connector_indices: Optional[Set[int]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Optional[float]]]]:
    if not candidates:
        return [], {}

    cand_sorted = sorted(set(int(i) for i in candidates))
    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    # Progress diagnostics: projection onto the AB direction in the same sonic
    # similarity space used for endpoint sims.
    vec_a_full = X_full_norm[pier_a]
    vec_b_full = X_full_norm[pier_b]
    d = vec_b_full - vec_a_full
    denom_progress = float(np.dot(d, d))
    progress_active = bool(math.isfinite(denom_progress) and denom_progress > 1e-12)

    sim_a_vals: list[float] = []
    sim_b_vals: list[float] = []
    hmean_vals: list[float] = []
    progress_vals: list[float] = []
    tmin_vals: list[float] = []
    t_a_raw_vals: list[float] = []
    t_b_raw_vals: list[float] = []
    t_a_vals: list[float] = []
    t_b_vals: list[float] = []
    gmin_vals: list[float] = []
    g_a_vals: list[float] = []
    g_b_vals: list[float] = []

    genre_vec_a = X_genre_norm[pier_a] if X_genre_norm is not None else None
    genre_vec_b = X_genre_norm[pier_b] if X_genre_norm is not None else None

    rows: list[dict[str, Any]] = []
    for cand in cand_sorted:
        keys = identity_keys_for_index(bundle, int(cand))
        sim_a = float(sim_to_a[cand])
        sim_b = float(sim_to_b[cand])
        denom = sim_a + sim_b
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom

        progress_t = None
        if progress_active:
            t_raw = float(np.dot((X_full_norm[cand] - vec_a_full), d) / denom_progress)
            if math.isfinite(t_raw):
                progress_t = float(max(0.0, min(1.0, t_raw)))
                progress_vals.append(float(progress_t))

        t_a_raw, t_a = _compute_transition_score_raw_and_transformed(
            pier_a, cand, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm, X_end_tr_norm, cfg
        )
        t_b_raw, t_b = _compute_transition_score_raw_and_transformed(
            cand, pier_b, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm, X_end_tr_norm, cfg
        )
        t_min = min(t_a, t_b)

        g_a = float("nan")
        g_b = float("nan")
        g_min = float("nan")
        if genre_vec_a is not None and genre_vec_b is not None:
            g_a = float(np.dot(genre_vec_a, X_genre_norm[cand]))  # type: ignore[index]
            g_b = float(np.dot(X_genre_norm[cand], genre_vec_b))  # type: ignore[index]
            g_min = min(g_a, g_b) if math.isfinite(g_a) and math.isfinite(g_b) else float("nan")

        final = cfg.weight_bridge * hmean + cfg.weight_transition * t_min
        if math.isfinite(g_min) and cfg.genre_tiebreak_weight:
            final += float(cfg.genre_tiebreak_weight) * float(g_min)
        if (
            cfg.genre_penalty_strength > 0
            and math.isfinite(g_min)
            and float(g_min) < float(cfg.genre_penalty_threshold)
        ):
            final *= (1.0 - float(cfg.genre_penalty_strength))

        artist = (
            str(bundle.track_artists[cand])
            if bundle.track_artists is not None
            else (str(bundle.artist_keys[cand]) if bundle.artist_keys is not None else "")
        )
        title = str(bundle.track_titles[cand]) if bundle.track_titles is not None else ""
        rows.append(
            {
                "track_id": str(bundle.track_ids[cand]),
                "artist": sanitize_for_logging(artist),
                "title": sanitize_for_logging(title),
                "artist_key": keys.artist_key,
                "title_key": keys.title_key,
                "progress_t": (round(float(progress_t), 3) if progress_t is not None else None),
                "simA": round(sim_a, 3),
                "simB": round(sim_b, 3),
                "hmean": round(hmean, 3),
                "bridge_sim": round(hmean, 3),
                "T_min": round(float(t_min), 3),
                "G_min": (round(float(g_min), 3) if math.isfinite(g_min) else None),
                "final": round(float(final), 3),
                "internal": bool(internal_connector_indices and cand in internal_connector_indices),
            }
        )

        sim_a_vals.append(sim_a)
        sim_b_vals.append(sim_b)
        hmean_vals.append(hmean)
        tmin_vals.append(float(t_min))
        t_a_raw_vals.append(float(t_a_raw))
        t_b_raw_vals.append(float(t_b_raw))
        t_a_vals.append(float(t_a))
        t_b_vals.append(float(t_b))
        if math.isfinite(g_min):
            gmin_vals.append(float(g_min))
        if math.isfinite(g_a):
            g_a_vals.append(float(g_a))
        if math.isfinite(g_b):
            g_b_vals.append(float(g_b))

    rows = sorted(rows, key=lambda r: (-float(r.get("final") or 0.0), str(r.get("track_id", ""))))[: max(0, int(top_k))]

    dists: dict[str, dict[str, Optional[float]]] = {
        "simA": _dist(sim_a_vals),
        "simB": _dist(sim_b_vals),
        "hmean": _dist(hmean_vals),
        "progress_t": _dist(progress_vals),
        "T_min": _dist(tmin_vals),
        "T_raw_pierA_to_cand": _dist(t_a_raw_vals),
        "T_raw_cand_to_pierB": _dist(t_b_raw_vals),
        "T_pierA_to_cand": _dist(t_a_vals),
        "T_cand_to_pierB": _dist(t_b_vals),
    }
    if X_genre_norm is not None:
        dists["G_min"] = _dist(gmin_vals)
        dists["G_pierA_to_cand"] = _dist(g_a_vals)
        dists["G_cand_to_pierB"] = _dist(g_b_vals)
    return rows, dists


def _compute_bridgeability_score(
    idx_a: int,
    idx_b: int,
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> float:
    """
    Cheap heuristic for how well two seeds can be bridged.
    Uses direct transition similarity plus a term for the distance between them.
    """
    # Direct transition similarity
    if X_end_norm is not None and X_start_norm is not None:
        direct_sim = float(np.dot(X_end_norm[idx_a], X_start_norm[idx_b]))
    else:
        direct_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Full similarity (for overall coherence)
    full_sim = float(np.dot(X_full_norm[idx_a], X_full_norm[idx_b]))

    # Combine: favor pairs with good direct transitions
    return 0.6 * direct_sim + 0.4 * full_sim


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 1e-12:
        return vec
    return vec / norm


def _genre_vocab_map(genre_vocab: np.ndarray) -> dict[str, int]:
    return {str(g).strip().lower(): int(i) for i, g in enumerate(genre_vocab)}


def _compute_genre_idf(
    X_genre_raw: np.ndarray,
    cfg: PierBridgeConfig,
) -> np.ndarray:
    """
    Compute IDF (inverse document frequency) for each genre.

    Formula:
        df[g] = count(tracks where genre[g] > 0)
        idf[g] = log((N + 1) / (df[g] + 1))  # +1 smoothing
        idf = idf ** cfg.dj_genre_idf_power
        idf = normalize(idf, method=cfg.dj_genre_idf_norm)

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
    power = float(cfg.dj_genre_idf_power)
    if power != 1.0 and power > 0:
        idf = idf ** power

    # Normalize
    norm_method = str(cfg.dj_genre_idf_norm).strip().lower()
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


def _build_dj_relaxation_attempts(cfg: PierBridgeConfig) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    attempts.append({
        "label": "baseline",
        "cfg": cfg,
        "changes": [],
        "force_allow_detours": False,
    })

    relaxed_weight = float(cfg.dj_waypoint_weight) * 0.5
    attempts.append({
        "label": "relax_waypoint",
        "cfg": replace(
            cfg,
            dj_waypoint_weight=float(relaxed_weight),
            dj_waypoint_floor=0.0,
            dj_waypoint_penalty=0.0,
        ),
        "changes": [
            f"waypoint_weight*0.5->{relaxed_weight:.3f}",
            "waypoint_floor->0",
            "waypoint_penalty->0",
        ],
        "force_allow_detours": False,
    })

    pool_scale = 1.25
    effort_cfg = replace(
        cfg,
        segment_pool_max=min(
            int(cfg.max_segment_pool_max),
            int(max(1, round(float(cfg.segment_pool_max) * pool_scale))),
        ),
        dj_pooling_k_local=int(max(1, round(float(cfg.dj_pooling_k_local) * pool_scale))),
        dj_pooling_k_toward=int(max(1, round(float(cfg.dj_pooling_k_toward) * pool_scale))),
        dj_pooling_k_genre=int(max(1, round(float(cfg.dj_pooling_k_genre) * pool_scale))),
        dj_pooling_k_union_max=int(
            max(1, round(float(cfg.dj_pooling_k_union_max) * pool_scale))
        ),
        initial_beam_width=min(
            int(cfg.max_beam_width),
            int(max(1, round(float(cfg.initial_beam_width) * 1.5))),
        ),
    )
    attempts.append({
        "label": "relax_effort",
        "cfg": effort_cfg,
        "changes": [
            "segment_pool_max*1.25",
            "dj_pooling_k_* *1.25",
            "initial_beam_width*1.5",
        ],
        "force_allow_detours": False,
    })

    connector_cfg = replace(
        cfg,
        dj_connector_bias_enabled=True,
        dj_connector_max_per_segment_linear=int(cfg.dj_connector_max_per_segment_linear) + 1,
        dj_connector_max_per_segment_adventurous=int(cfg.dj_connector_max_per_segment_adventurous) + 1,
    )
    attempts.append({
        "label": "relax_connectors",
        "cfg": connector_cfg,
        "changes": [
            "connector_bias_enabled->true",
            "connector_max_per_segment+1",
            "force_allow_detours",
        ],
        "force_allow_detours": True,
    })

    if bool(cfg.dj_relaxation_allow_floor_relaxation):
        relaxed_floor = max(0.0, float(cfg.transition_floor) - 0.02)
        attempts.append({
            "label": "relax_transition_floor",
            "cfg": replace(cfg, transition_floor=float(relaxed_floor)),
            "changes": [f"transition_floor-0.02->{relaxed_floor:.3f}"],
            "force_allow_detours": False,
        })

    max_attempts = max(1, int(cfg.dj_relaxation_max_attempts))
    return attempts[:max_attempts]


def _score_micro_pier_candidates(
    candidates: list[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
) -> list[tuple[int, float]]:
    if not candidates:
        return []
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]
    cand_list = [int(i) for i in candidates]
    sims_a = np.dot(X_full_norm[cand_list], vec_a)
    sims_b = np.dot(X_full_norm[cand_list], vec_b)
    scores = np.minimum(sims_a, sims_b)
    return [(int(cand_list[i]), float(scores[i])) for i in range(len(cand_list))]


def _select_micro_pier_candidates(
    candidates: list[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
    top_k: int,
) -> list[tuple[int, float]]:
    scored = _score_micro_pier_candidates(candidates, X_full_norm, pier_a, pier_b)
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, int(top_k))]


def _micro_pier_candidate_pool(
    source: str,
    last_segment_candidates: list[int],
    pool_cache: Optional[Dict[str, Any]],
) -> list[int]:
    source = str(source or "union_pool").strip().lower()
    connectors: list[int] = []
    if pool_cache is not None:
        cached = pool_cache.get("dj_connectors")
        if cached:
            connectors = [int(i) for i in cached]
    if source == "connectors":
        return connectors
    if source == "both":
        combined = list(dict.fromkeys(connectors + list(last_segment_candidates)))
        return combined
    return list(last_segment_candidates)


def _should_attempt_micro_pier(
    *,
    relaxation_enabled: bool,
    segment_path: Optional[list[int]],
) -> bool:
    return bool(relaxation_enabled) and segment_path is None


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


def _attempt_micro_pier_split(
    *,
    pier_a: int,
    pier_b: int,
    interior_length: int,
    candidates: list[int],
    X_full: np.ndarray,
    X_full_norm: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    beam_width: int,
    artist_key_by_idx: Optional[Dict[int, str]],
    seed_artist_key: Optional[str],
    recent_global_artists: Optional[List[str]],
    durations_ms: Optional[np.ndarray],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    bundle: Optional[ArtifactBundle],
    warnings: list[dict[str, Any]],
    X_genre_vocab: Optional[np.ndarray],
    genre_graph: Optional[dict[str, list[tuple[str, float]]]],
    micro_diag: Optional[dict[str, Any]] = None,
    X_genre_norm_idf: Optional[np.ndarray] = None,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_idf: Optional[np.ndarray] = None,
) -> Optional[list[int]]:
    if interior_length < 2 or not candidates:
        return None
    max_micro = max(1, int(cfg.dj_micro_piers_max))
    topk = max(1, int(cfg.dj_micro_piers_topk))

    cand_list = [int(i) for i in candidates]
    metric = str(cfg.dj_micro_piers_selection_metric or "max_min_sim").strip().lower()
    if metric != "max_min_sim":
        metric = "max_min_sim"
    scored = _select_micro_pier_candidates(
        candidates,
        X_full_norm,
        pier_a,
        pier_b,
        top_k=topk,
    )
    micro_candidates = [idx for idx, _ in scored][:topk]

    left_len = interior_length // 2
    right_len = interior_length - left_len - 1
    if right_len < 0:
        return None

    for micro_idx in micro_candidates[:max_micro]:
        if micro_diag is not None:
            micro_diag.update({
                "micro_pier_index": int(micro_idx),
                "micro_pier_metric": "max_min_sim",
                "micro_pier_metric_value": float(
                    next((score for idx, score in scored if idx == micro_idx), 0.0)
                ),
                "left_success": False,
                "right_success": False,
            })
        left_g_targets = None
        right_g_targets = None
        if X_genre_norm is not None and X_genre_vocab is not None and bool(cfg.dj_bridging_enabled):
            left_g_targets = _build_genre_targets(
                pier_a=pier_a,
                pier_b=micro_idx,
                interior_length=left_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=X_genre_vocab,
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                X_genre_raw=None,
                X_genre_smoothed=None,
                genre_idf=None,
            )
            right_g_targets = _build_genre_targets(
                pier_a=micro_idx,
                pier_b=pier_b,
                interior_length=right_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=X_genre_vocab,
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                X_genre_raw=None,
                X_genre_smoothed=None,
                genre_idf=None,
            )

        keys_map = dict(artist_key_by_idx or {})
        if bundle is not None:
            try:
                keys_map[int(pier_a)] = identity_keys_for_index(bundle, int(pier_a)).artist_key
                keys_map[int(pier_b)] = identity_keys_for_index(bundle, int(pier_b)).artist_key
                keys_map[int(micro_idx)] = identity_keys_for_index(bundle, int(micro_idx)).artist_key
            except Exception:
                pass

        left_path, _, _, _ = _beam_search_segment(
            pier_a,
            micro_idx,
            left_len,
            cand_list,
            X_full,
            X_full_norm,
            X_start,
            X_mid,
            X_end,
            X_genre_norm,
            cfg,
            beam_width,
            X_genre_norm_idf=X_genre_norm_idf,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
            genre_vocab=X_genre_vocab,
            artist_key_by_idx=(keys_map if keys_map else None),
            seed_artist_key=seed_artist_key,
            recent_global_artists=recent_global_artists,
            durations_ms=durations_ms,
            artist_identity_cfg=artist_identity_cfg,
            bundle=bundle,
            g_targets_override=left_g_targets,
        )
        if left_path is None:
            continue
        if micro_diag is not None:
            micro_diag["left_success"] = True

        used_left = set(int(i) for i in left_path)
        right_candidates = [int(i) for i in cand_list if int(i) not in used_left and int(i) != int(micro_idx)]

        right_path, _, _, _ = _beam_search_segment(
            micro_idx,
            pier_b,
            right_len,
            right_candidates,
            X_full,
            X_full_norm,
            X_start,
            X_mid,
            X_end,
            X_genre_norm,
            cfg,
            beam_width,
            X_genre_norm_idf=X_genre_norm_idf,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
            genre_vocab=X_genre_vocab,
            artist_key_by_idx=(keys_map if keys_map else None),
            seed_artist_key=seed_artist_key,
            recent_global_artists=recent_global_artists,
            durations_ms=durations_ms,
            artist_identity_cfg=artist_identity_cfg,
            bundle=bundle,
            g_targets_override=right_g_targets,
        )
        if right_path is None:
            continue
        if micro_diag is not None:
            micro_diag["right_success"] = True

        warnings.append({
            "type": "micro_pier_used",
            "scope": "segment",
            "message": "Inserted a micro-pier connector to bridge a difficult segment.",
            "micro_pier_index": int(micro_idx),
        })
        return left_path + [int(micro_idx)] + right_path

    return None


def _segment_far_stats(
    *,
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    X_genre_norm: Optional[np.ndarray],
    universe: list[int],
    used_track_ids: Set[int],
    bridge_floor: float,
) -> dict[str, Optional[float]]:
    sim_sonic = float(np.dot(X_full_norm[pier_a], X_full_norm[pier_b]))
    sim_genre = None
    if X_genre_norm is not None:
        sim_genre = float(np.dot(X_genre_norm[pier_a], X_genre_norm[pier_b]))
    available = [int(i) for i in universe if int(i) not in used_track_ids]
    scarcity = None
    if available:
        vec_a = X_full_norm[pier_a]
        vec_b = X_full_norm[pier_b]
        sims_a = np.dot(X_full_norm[available], vec_a)
        sims_b = np.dot(X_full_norm[available], vec_b)
        gate = np.minimum(sims_a, sims_b) >= float(bridge_floor)
        scarcity = float(np.mean(gate)) if gate.size > 0 else None
    return {
        "sonic_sim": sim_sonic,
        "genre_sim": sim_genre,
        "connector_scarcity": scarcity,
    }


def _select_connector_candidates(
    available: List[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
    cap: int,
) -> List[int]:
    if cap <= 0 or not available:
        return []
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]
    sims_a = np.dot(X_full_norm[available], vec_a)
    sims_b = np.dot(X_full_norm[available], vec_b)
    scores = np.minimum(sims_a, sims_b)
    order = np.argsort(-scores)
    return [int(available[int(i)]) for i in order[:cap]]


def _order_seeds_by_bridgeability(
    seed_indices: List[int],
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray] = None,
    *,
    weight_sonic: float = 0.0,
    weight_genre: float = 0.0,
    weight_bridge: float = 1.0,
) -> List[int]:
    """
    Order seed indices to maximize total bridgeability.
    For <=6 seeds, evaluates all permutations.
    For >6 seeds, uses greedy nearest-neighbor heuristic.
    """
    n = len(seed_indices)
    if n <= 1:
        return seed_indices

    weight_sonic = float(weight_sonic) if math.isfinite(float(weight_sonic)) else 0.0
    weight_genre = float(weight_genre) if math.isfinite(float(weight_genre)) else 0.0
    weight_bridge = float(weight_bridge) if math.isfinite(float(weight_bridge)) else 0.0
    weight_sonic = max(0.0, weight_sonic)
    weight_genre = max(0.0, weight_genre)
    weight_bridge = max(0.0, weight_bridge)
    total_weight = weight_sonic + weight_genre + weight_bridge
    if total_weight <= 1e-9:
        weight_bridge = 1.0
        total_weight = 1.0
    weight_sonic /= total_weight
    weight_genre /= total_weight
    weight_bridge /= total_weight

    def _pair_score(a: int, b: int) -> float:
        score = 0.0
        if weight_bridge > 0:
            score += weight_bridge * _compute_bridgeability_score(
                a, b, X_full_norm, X_start_norm, X_end_norm
            )
        if weight_sonic > 0:
            score += weight_sonic * float(np.dot(X_full_norm[a], X_full_norm[b]))
        if weight_genre > 0 and X_genre_norm is not None:
            score += weight_genre * float(np.dot(X_genre_norm[a], X_genre_norm[b]))
        return score

    if n <= 6:
        # Exhaustive search for small seed counts
        best_order = None
        best_score = -float('inf')

        for perm in itertools.permutations(seed_indices):
            total_score = 0.0
            for i in range(len(perm) - 1):
                total_score += _pair_score(perm[i], perm[i + 1])
            if total_score > best_score:
                best_score = total_score
                best_order = list(perm)

        logger.info("Seed ordering: evaluated %d permutations, best_score=%.4f",
                   math.factorial(n), best_score)
        return best_order or seed_indices
    else:
        # Greedy nearest-neighbor for larger seed counts
        remaining = set(seed_indices)
        # Start with the first seed
        ordered = [seed_indices[0]]
        remaining.remove(seed_indices[0])

        while remaining:
            current = ordered[-1]
            best_next = None
            best_score = -float('inf')

            for candidate in remaining:
                score = _pair_score(current, candidate)
                if score > best_score:
                    best_score = score
                    best_next = candidate

            if best_next is not None:
                ordered.append(best_next)
                remaining.remove(best_next)

        logger.info("Seed ordering: greedy heuristic for %d seeds", n)
        return ordered


def _dedupe_candidate_pool(
    pool_indices: List[int],
    bundle: ArtifactBundle,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Deduplicate candidate pool by normalized artist+title.
    Returns deduplicated indices and mapping of norm_key -> chosen index.

    Prefers canonical versions based on version preference scoring.
    """
    from src.title_dedupe import calculate_version_preference_score

    seen: Dict[str, Tuple[int, int]] = {}  # norm_key -> (index, preference_score)

    for idx in pool_indices:
        keys = identity_keys_for_index(bundle, int(idx))
        key = f"{keys.artist_key}|||{keys.title_key}"

        # Compute preference score (higher = more canonical)
        title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""
        pref_score = calculate_version_preference_score(title)

        if key not in seen or pref_score > seen[key][1]:
            seen[key] = (idx, pref_score)

    deduped = [idx for idx, _ in seen.values()]
    norm_to_idx = {key: idx for key, (idx, _) in seen.items()}

    logger.debug("Deduped candidate pool: %d -> %d tracks", len(pool_indices), len(deduped))
    return deduped, norm_to_idx


def _build_segment_candidate_pool_legacy(
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    universe_indices: List[int],
    used_track_ids: Set[int],
    neighbors_m: int,
    bridge_helpers: int,
    artist_keys: Optional[np.ndarray] = None,
    bridge_floor: float = 0.0,
    allowed_set: Optional[Set[int]] = None,
    internal_connectors: Optional[Set[int]] = None,
    internal_connector_cap: int = 0,
    internal_connector_priority: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """
    Legacy segment pool builder (debug/compat).

    Builds candidates via a union of:
    - Top M neighbors of pier_a by full similarity
    - Top M neighbors of pier_b by full similarity
    - Top B "bridge helper" tracks by two-sided bridge score
    Then dedupes to 1-per-artist and applies the bridge_floor gate.

    Includes:
    - Top M neighbors of pier_a by full similarity
    - Top M neighbors of pier_b by full similarity
    - Top B "bridge helper" tracks by two-sided bridge score

    Only ONE track per artist is allowed in the segment.
    This prevents artist clustering without needing min_gap constraints.
    All artists (including seed artist) follow the same one-per-segment rule.
    """
    # Filter out used tracks
    available = [idx for idx in universe_indices if idx not in used_track_ids]
    if not available:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "available": 0,
                    "neighbors_a": 0,
                    "neighbors_b": 0,
                    "helpers": 0,
                    "combined": 0,
                    "combined_allowed": 0,
                    "deduped": 0,
                    "after_bridge_gate": 0,
                    "internal_connectors_candidates": 0,
                    "internal_connectors_pass_gate": 0,
                    "internal_connectors_selected": 0,
                    "final": 0,
                }
            )
        return []
    if diagnostics is not None:
        diagnostics["available"] = len(available)

    # Compute similarities to both piers
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]

    sim_to_a = {}
    sim_to_b = {}
    bridge_score = {}

    for idx in available:
        sim_a = float(np.dot(X_full_norm[idx], vec_a))
        sim_b = float(np.dot(X_full_norm[idx], vec_b))
        sim_to_a[idx] = sim_a
        sim_to_b[idx] = sim_b
        # Bridge score: geometric mean of similarities to both piers
        bridge_score[idx] = math.sqrt(max(0, sim_a) * max(0, sim_b))

    # Top M neighbors of pier_a
    sorted_by_a = sorted(available, key=lambda i: sim_to_a[i], reverse=True)
    neighbors_a = set(sorted_by_a[:neighbors_m])
    if diagnostics is not None:
        diagnostics["neighbors_a"] = len(neighbors_a)

    # Top M neighbors of pier_b
    sorted_by_b = sorted(available, key=lambda i: sim_to_b[i], reverse=True)
    neighbors_b = set(sorted_by_b[:neighbors_m])
    if diagnostics is not None:
        diagnostics["neighbors_b"] = len(neighbors_b)

    # Top B bridge helpers
    sorted_by_bridge = sorted(available, key=lambda i: bridge_score[i], reverse=True)
    helpers = set(sorted_by_bridge[:bridge_helpers])
    if diagnostics is not None:
        diagnostics["helpers"] = len(helpers)

    # Combine all candidates
    combined = neighbors_a | neighbors_b | helpers
    if diagnostics is not None:
        diagnostics["combined"] = len(combined)

    # Internal connectors (optional priority/cap)
    connector_selected: List[int] = []
    connector_candidates = 0
    connector_pass_gate = 0
    if internal_connectors:
        for idx in internal_connectors:
            if idx in used_track_ids or (allowed_set is not None and idx not in allowed_set):
                continue
            connector_candidates += 1
            sim_a = float(np.dot(X_full_norm[idx], vec_a))
            sim_b = float(np.dot(X_full_norm[idx], vec_b))
            if min(sim_a, sim_b) < bridge_floor:
                continue
            connector_pass_gate += 1
            connector_selected.append(idx)
        connector_selected = connector_selected[: internal_connector_cap if internal_connector_cap > 0 else len(connector_selected)]
    if diagnostics is not None:
        diagnostics["internal_connectors_candidates"] = int(connector_candidates)
        diagnostics["internal_connectors_pass_gate"] = int(connector_pass_gate)
        diagnostics["internal_connectors_selected"] = int(len(connector_selected))

    # Dedupe to ONE track per artist (all artists treated equally, including seed artist)
    if artist_keys is not None:
        artist_best: Dict[str, Tuple[int, float]] = {}  # artist -> (idx, score)
        combined_allowed = 0
        for idx in combined:
            if allowed_set is not None and idx not in allowed_set:
                continue
            combined_allowed += 1
            artist = normalize_artist_key(str(artist_keys[idx]))
            score = bridge_score.get(idx, 0.0)

            if artist not in artist_best or score > artist_best[artist][1]:
                artist_best[artist] = (idx, score)

        # Build final pool: one track per artist (normalized)
        deduped: List[int] = [idx for idx, _ in artist_best.values()]
        if diagnostics is not None:
            diagnostics["combined_allowed"] = int(combined_allowed)
            diagnostics["deduped"] = int(len(deduped))

        logger.debug("Segment pool: %d combined -> %d after 1-per-artist dedupe",
                     len(combined), len(deduped))
        filtered = [
            idx for idx in deduped
            if min(float(np.dot(X_full_norm[idx], vec_a)), float(np.dot(X_full_norm[idx], vec_b))) >= bridge_floor
        ]
        if diagnostics is not None:
            diagnostics["after_bridge_gate"] = int(len(filtered))
    else:
        filtered = [
            idx for idx in combined
            if min(float(np.dot(X_full_norm[idx], vec_a)), float(np.dot(X_full_norm[idx], vec_b))) >= bridge_floor
            and (allowed_set is None or idx in allowed_set)
        ]
        if diagnostics is not None:
            diagnostics["combined_allowed"] = int(
                sum(1 for idx in combined if (allowed_set is None or idx in allowed_set))
            )
            diagnostics["deduped"] = int(len(combined))
            diagnostics["after_bridge_gate"] = int(len(filtered))

    if internal_connector_priority:
        filtered = list(dict.fromkeys(connector_selected + filtered))
    else:
        filtered = list(dict.fromkeys(filtered + connector_selected))
    if diagnostics is not None:
        diagnostics["final"] = int(len(filtered))
    return filtered


def _compute_bridge_score(
    sim_a: float,
    sim_b: float,
    *,
    experiment_enabled: bool,
    experiment_min_weight: float,
    experiment_balance_weight: float,
) -> float:
    denom = sim_a + sim_b
    hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
    if not experiment_enabled:
        return float(hmean)

    min_weight = max(0.0, min(1.0, float(experiment_min_weight)))
    balance_weight = max(0.0, min(1.0 - min_weight, float(experiment_balance_weight)))
    hmean_weight = max(0.0, 1.0 - min_weight - balance_weight)

    min_sim = min(sim_a, sim_b)
    balance = 1.0 - abs(sim_a - sim_b)
    if balance < 0.0:
        balance = 0.0
    elif balance > 1.0:
        balance = 1.0

    return float(
        (hmean_weight * hmean)
        + (min_weight * min_sim)
        + (balance_weight * balance)
    )


def _build_segment_candidate_pool_scored(
    *,
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    universe_indices: List[int],
    used_track_ids: Set[int],
    bundle: ArtifactBundle,
    bridge_floor: float,
    segment_pool_max: int,
    allowed_set: Optional[Set[int]] = None,
    internal_connectors: Optional[Set[int]] = None,
    internal_connector_cap: int = 0,
    internal_connector_priority: bool = True,
    seed_artist_key: Optional[str] = None,
    disallow_pier_artists_in_interiors: bool = False,
    disallow_seed_artist_in_interiors: bool = False,
    used_track_keys: Optional[Set[tuple[str, str]]] = None,
    seed_track_keys: Optional[Set[tuple[str, str]]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    experiment_bridge_scoring_enabled: bool = False,
    experiment_bridge_min_weight: float = 0.25,
    experiment_bridge_balance_weight: float = 0.15,
    pool_strategy: str = "segment_scored",
    interior_length: int = 0,
    progress_arc_enabled: bool = False,
    progress_arc_shape: str = "linear",
    X_genre_norm: Optional[np.ndarray] = None,
    X_genre_norm_idf: Optional[np.ndarray] = None,
    genre_targets: Optional[List[np.ndarray]] = None,
    pool_k_local: int = 0,
    pool_k_toward: int = 0,
    pool_k_genre: int = 0,
    pool_k_union_max: int = 0,
    pool_step_stride: int = 1,
    pool_cache_enabled: bool = True,
    pooling_cache: Optional[Dict[str, Any]] = None,
    pool_verbose: bool = False,  # Phase 3 fix: verbose pool logging
) -> tuple[List[int], Dict[int, str], Dict[int, str]]:
    """
    Segment-local candidate pool builder ("segment_scored").

    Builds a segment candidate pool by scoring candidates jointly vs BOTH endpoints
    (pier A and pier B), applying structural exclusions (used ids, allowed-set
    clamp, artist policies, track_key collisions), gating by bridge_floor, then
    taking top-K by harmonic_mean(simA, simB).

    Returns:
      - candidates: list[int] indices for beam search
      - artist_key_by_idx: mapping for candidates (robust identity key)
      - title_key_by_idx: mapping for candidates (normalized title key)
    """
    pool_cfg = SegmentPoolConfig(
        pier_a=int(pier_a),
        pier_b=int(pier_b),
        X_full_norm=X_full_norm,
        universe_indices=list(universe_indices),
        used_track_ids=set(int(i) for i in used_track_ids),
        bundle=bundle,
        bridge_floor=float(bridge_floor),
        segment_pool_max=int(segment_pool_max),
        allowed_set=allowed_set,
        internal_connectors=internal_connectors,
        internal_connector_cap=int(internal_connector_cap),
        internal_connector_priority=bool(internal_connector_priority),
        seed_artist_key=seed_artist_key,
        disallow_pier_artists_in_interiors=bool(disallow_pier_artists_in_interiors),
        disallow_seed_artist_in_interiors=bool(disallow_seed_artist_in_interiors),
        used_track_keys=used_track_keys,
        seed_track_keys=seed_track_keys,
        diagnostics=diagnostics,
        experiment_bridge_scoring_enabled=bool(experiment_bridge_scoring_enabled),
        experiment_bridge_min_weight=float(experiment_bridge_min_weight),
        experiment_bridge_balance_weight=float(experiment_bridge_balance_weight),
        pool_strategy=str(pool_strategy),
        pool_k_local=int(pool_k_local),
        pool_k_toward=int(pool_k_toward),
        pool_k_genre=int(pool_k_genre),
        pool_k_union_max=int(pool_k_union_max),
        pool_step_stride=int(pool_step_stride),
        pool_cache_enabled=bool(pool_cache_enabled),
        interior_length=int(interior_length),
        progress_arc_enabled=bool(progress_arc_enabled),
        progress_arc_shape=str(progress_arc_shape),
        X_genre_norm=X_genre_norm,
        X_genre_norm_idf=X_genre_norm_idf,
        genre_targets=genre_targets,
        pooling_cache=pooling_cache,
        pool_verbose=bool(pool_verbose),  # Phase 3 fix
    )
    result = SegmentCandidatePoolBuilder().build(pool_cfg)
    return result.candidates, result.artist_key_by_idx, result.title_key_by_idx


@dataclass
class BeamState:
    """State for beam search."""
    path: List[int]
    score: float
    used: Set[int]
    used_artists: Set[str] = field(default_factory=set)
    last_progress: float = 0.0


def _compute_duration_penalty(
    candidate_duration_ms: float,
    reference_duration_ms: float,
    weight: float,
) -> float:
    """
    Compute asymmetric duration penalty based on percentage excess over reference track.

    Uses a three-phase geometric curve:
    - 0-20% excess: Gentle penalties (barely noticeable)
    - 20-50% excess: Moderate increasing penalties
    - 50-100% excess: Steep penalties
    - >100% excess: Severe penalties (track is 2x+ longer than reference)

    Args:
        candidate_duration_ms: Candidate track duration in milliseconds
        reference_duration_ms: Reference duration (max of two piers) in milliseconds
        weight: Penalty weight (default 0.30, range typically 0.10-0.50)

    Returns:
        Penalty value (>= 0) to subtract from combined_score

    Examples:
        With weight=0.30 and reference=200s:
        - 210s (+5% = +10s): penalty ≈ 0.003 (negligible)
        - 240s (+20% = +40s): penalty ≈ 0.015 (gentle)
        - 280s (+40% = +80s): penalty ≈ 0.10 (moderate)
        - 360s (+80% = +160s): penalty ≈ 0.45 (steep)
        - 400s (+100% = +200s): penalty ≈ 0.75 (severe threshold)
        - 600s (+200% = +400s): penalty ≈ 3.0 (very severe!)
    """
    if candidate_duration_ms <= 0 or reference_duration_ms <= 0:
        return 0.0

    if candidate_duration_ms <= reference_duration_ms:
        return 0.0  # No penalty for shorter or equal-length tracks

    # Calculate excess as percentage of reference
    excess_ratio = (candidate_duration_ms - reference_duration_ms) / reference_duration_ms

    # Three-phase geometric penalty curve
    if excess_ratio <= 0.20:
        # Phase 1 (0-20%): Gentle - power 1.5 for sub-linear growth
        # At 20%: penalty = weight * 0.05
        penalty = weight * 0.05 * (excess_ratio / 0.20) ** 1.5

    elif excess_ratio <= 0.50:
        # Phase 2 (20-50%): Moderate - power 2.0 for quadratic growth
        # At 20%: ~0.015, At 50%: ~0.30
        phase_ratio = (excess_ratio - 0.20) / 0.30
        penalty = weight * 0.05 + weight * 0.25 * (phase_ratio ** 2.0)

    elif excess_ratio <= 1.00:
        # Phase 3 (50-100%): Steep - power 2.5 for accelerating growth
        # At 50%: ~0.30, At 100%: ~0.75
        phase_ratio = (excess_ratio - 0.50) / 0.50
        penalty = weight * 0.30 + weight * 0.45 * (phase_ratio ** 2.5)

    else:
        # Phase 4 (>100%): Severe - power 3.0 for very steep growth
        # At 100%: 0.75, At 200%: 3.0, At 300%: 9.0
        phase_ratio = excess_ratio - 1.00
        penalty = weight * 0.75 + weight * 2.25 * (phase_ratio ** 3.0)

    return penalty


def _beam_search_segment(
    pier_a: int,
    pier_b: int,
    interior_length: int,
    candidates: List[int],
    X_full: np.ndarray,
    X_full_norm: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    beam_width: int,
    *,
    X_genre_norm_idf: Optional[np.ndarray] = None,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_idf: Optional[np.ndarray] = None,
    genre_vocab: Optional[np.ndarray] = None,
    artist_key_by_idx: Optional[Dict[int, str]] = None,
    seed_artist_key: Optional[str] = None,
    recent_global_artists: Optional[List[str]] = None,
    durations_ms: Optional[np.ndarray] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
    bundle: Optional[ArtifactBundle] = None,
    arc_stats: Optional[Dict[str, Any]] = None,
    genre_cache_stats: Optional[Dict[str, int]] = None,
    g_targets_override: Optional[List[np.ndarray]] = None,
    waypoint_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[int]], int, int, Optional[str]]:
    """
    Constrained beam search to find path from pier_a to pier_b.

    Returns interior track indices (not including piers) or None if no path found.

    Args:
        recent_global_artists: Artist keys from the last min_gap positions of the
            global playlist prefix (from previous segments). Used to enforce
            cross-segment min_gap constraints during generation. If artist_identity_cfg
            is enabled, these are identity keys (collapsing ensemble variants).
        durations_ms: Optional array of track durations in milliseconds (parallel to bundle.track_ids)
        artist_identity_cfg: Optional config for artist identity resolution (ensemble collapsing)
        bundle: Optional bundle for resolving artist strings to identity keys
        waypoint_stats: Optional dict to populate with waypoint influence diagnostics
    """

    genre_penalty_hits = 0
    edges_scored = 0
    penalty_strength = float(cfg.genre_penalty_strength)
    if not math.isfinite(penalty_strength):
        penalty_strength = 0.0
    penalty_strength = float(max(0.0, min(1.0, penalty_strength)))
    penalty_threshold = float(cfg.genre_penalty_threshold)
    genre_tie_break_band = cfg.genre_tie_break_band
    if isinstance(genre_tie_break_band, (int, float)) and math.isfinite(float(genre_tie_break_band)):
        genre_tie_break_band = float(genre_tie_break_band)
        if genre_tie_break_band <= 0:
            genre_tie_break_band = None
    else:
        genre_tie_break_band = None

    waypoint_enabled = bool(cfg.dj_bridging_enabled) and X_genre_norm is not None
    waypoint_weight = float(cfg.dj_waypoint_weight)
    if not math.isfinite(waypoint_weight):
        waypoint_weight = 0.0
    waypoint_floor = float(cfg.dj_waypoint_floor)
    if not math.isfinite(waypoint_floor):
        waypoint_floor = 0.0
    waypoint_penalty = float(cfg.dj_waypoint_penalty)
    if not math.isfinite(waypoint_penalty):
        waypoint_penalty = 0.0
    waypoint_cap = float(cfg.dj_waypoint_cap)
    if not math.isfinite(waypoint_cap):
        waypoint_cap = 0.0
    waypoint_cap = float(max(0.0, waypoint_cap))
    waypoint_tie_break_band = cfg.dj_waypoint_tie_break_band
    if isinstance(waypoint_tie_break_band, (int, float)) and math.isfinite(float(waypoint_tie_break_band)):
        waypoint_tie_break_band = float(waypoint_tie_break_band)
        if waypoint_tie_break_band <= 0:
            waypoint_tie_break_band = None
    else:
        waypoint_tie_break_band = None

    g_targets: Optional[List[np.ndarray]] = None
    if waypoint_enabled and interior_length > 0:
        if g_targets_override is not None and len(g_targets_override) == int(interior_length):
            g_targets = g_targets_override
        else:
            g_a = X_genre_norm[pier_a]
            g_b = X_genre_norm[pier_b]
            if float(np.linalg.norm(g_a)) <= 1e-8 or float(np.linalg.norm(g_b)) <= 1e-8:
                waypoint_enabled = False
            else:
                g_targets = []
                for i in range(interior_length):
                    frac = _step_fraction(i, interior_length)
                    g = (1.0 - frac) * g_a + frac * g_b
                    norm = float(np.linalg.norm(g))
                    if norm <= 1e-12:
                        g = g_a
                    else:
                        g = g / norm
                    g_targets.append(g)
    if waypoint_weight <= 0 and waypoint_penalty <= 0:
        waypoint_enabled = False

    # Phase 3: Waypoint delta mode configuration
    waypoint_delta_mode = str(cfg.dj_waypoint_delta_mode or "absolute").strip().lower()
    waypoint_centered_baseline_method = str(cfg.dj_waypoint_centered_baseline or "median").strip().lower()
    waypoint_squash_mode = str(cfg.dj_waypoint_squash or "none").strip().lower()
    waypoint_squash_alpha = float(cfg.dj_waypoint_squash_alpha)

    def _waypoint_delta(sim: Optional[float], sim0: float = 0.0) -> float:
        """
        Compute waypoint delta with optional centering and squashing.

        Phase 3 modes:
        - absolute (legacy): delta = weight * sim, clamped to [-cap, +cap]
        - centered: delta = weight * (sim - sim0), clamped to [-cap, +cap]

        Squashing (optional):
        - none (legacy): hard clamp at waypoint_cap
        - tanh: smooth squashing delta = cap * tanh(alpha * raw / cap)

        Args:
            sim: Waypoint similarity (candidate vs target)
            sim0: Baseline similarity for centered mode (median/mean of step candidates)

        Returns:
            delta ∈ [-waypoint_cap, +waypoint_cap]
        """
        if sim is None or not math.isfinite(float(sim)):
            return 0.0

        # Compute raw delta (centered or absolute)
        if waypoint_delta_mode == "centered":
            raw = float(waypoint_weight) * (float(sim) - float(sim0))
        else:  # absolute (legacy)
            raw = float(waypoint_weight) * float(sim)

        # Apply squashing or hard clamp
        if waypoint_squash_mode == "tanh" and waypoint_cap > 0:
            # Smooth squashing: delta = cap * tanh(alpha * raw / cap)
            normalized_raw = waypoint_squash_alpha * raw / waypoint_cap if waypoint_cap > 0 else 0.0
            delta = waypoint_cap * math.tanh(normalized_raw)
            # Clamp for safety (should be redundant with tanh)
            delta = float(max(-waypoint_cap, min(waypoint_cap, delta)))
        else:  # none (hard cap, legacy)
            delta = raw
            if waypoint_cap > 0:
                delta = float(max(-waypoint_cap, min(waypoint_cap, delta)))

        # Legacy penalty (only for absolute mode with floor)
        if waypoint_penalty > 0 and waypoint_floor > 0 and float(sim) < waypoint_floor:
            penalty = waypoint_penalty * (waypoint_floor - float(sim))
            if waypoint_cap > 0:
                penalty = min(waypoint_cap, penalty)
            delta -= float(penalty)

        return float(delta)

    # Coverage bonus setup (Phase 2 + Phase 3 enhancements)
    coverage_enabled = bool(cfg.dj_genre_use_coverage) and bool(cfg.dj_bridging_enabled)
    coverage_mode = str(cfg.dj_coverage_mode or "binary").strip().lower()
    coverage_presence_source = str(cfg.dj_coverage_presence_source or "same").strip().lower()
    topk_A: list[tuple[int, float]] = []
    topk_B: list[tuple[int, float]] = []
    X_genre_for_coverage: Optional[np.ndarray] = None
    X_genre_for_coverage_presence: Optional[np.ndarray] = None

    if coverage_enabled and interior_length > 0:
        # Determine which matrix to use for scoring (anchor topK extraction)
        if X_genre_norm_idf is not None:
            X_genre_for_coverage = X_genre_norm_idf
        elif X_genre_norm is not None:
            X_genre_for_coverage = X_genre_norm

        # Phase 3: Determine presence source (for coverage computation)
        if coverage_presence_source == "raw" and X_genre_raw is not None:
            # Use raw genres for presence checking (avoids smoothed inflation)
            if genre_idf is not None and bool(cfg.dj_genre_use_idf):
                # Apply IDF to raw for consistency
                X_genre_for_coverage_presence = X_genre_raw * genre_idf
                # Normalize rows
                row_norms = np.linalg.norm(X_genre_for_coverage_presence, axis=1, keepdims=True)
                X_genre_for_coverage_presence = np.divide(
                    X_genre_for_coverage_presence,
                    row_norms,
                    out=np.zeros_like(X_genre_for_coverage_presence),
                    where=row_norms > 1e-12
                )
            else:
                X_genre_for_coverage_presence = X_genre_raw
        else:
            # Use same matrix as scoring (default, Phase 2 behavior)
            X_genre_for_coverage_presence = X_genre_for_coverage

        # Extract anchor vectors in the same space as candidates will be scored
        if X_genre_for_coverage is not None:
            vA_cov = X_genre_for_coverage[pier_a]
            vB_cov = X_genre_for_coverage[pier_b]

            # Extract top-K genres from each anchor
            topk_A = _extract_top_genres(vA_cov, int(cfg.dj_genre_coverage_top_k))
            topk_B = _extract_top_genres(vB_cov, int(cfg.dj_genre_coverage_top_k))

            if not topk_A and not topk_B:
                coverage_enabled = False  # No genres to track
        else:
            coverage_enabled = False

    # Phase 2 diagnostic logging: segment-level config
    if cfg.dj_bridging_enabled and interior_length > 0:
        target_mode = str(cfg.dj_ladder_target_mode or "onehot").strip().lower()
        logger.info("[Phase2] Segment %d→%d: mode=%s, interior_length=%d",
                    pier_a, pier_b, target_mode, interior_length)

        # Phase 3: Log genre space being used for genre_sim
        genre_space_name = "IDF" if X_genre_norm_idf is not None else "normalized"
        logger.info("  Genre space for genre_sim: %s", genre_space_name)

        # Log IDF stats if enabled
        if bool(cfg.dj_genre_use_idf) and genre_idf is not None:
            logger.info("  IDF enabled: min=%.3f median=%.3f max=%.3f (power=%.2f norm=%s)",
                        float(np.min(genre_idf)), float(np.median(genre_idf)), float(np.max(genre_idf)),
                        float(cfg.dj_genre_idf_power), str(cfg.dj_genre_idf_norm))

        # Log anchor top-K genres if coverage enabled
        if coverage_enabled and genre_vocab is not None:
            def _format_topk(topk_list: list[tuple[int, float]], label: str) -> None:
                if topk_list:
                    genre_names = [f"{genre_vocab[idx]}={weight:.3f}" for idx, weight in topk_list[:5]]
                    logger.info("  %s topK genres: %s", label, ", ".join(genre_names))
            _format_topk(topk_A, "Anchor A")
            _format_topk(topk_B, "Anchor B")

    if interior_length == 0:
        # Check if direct transition meets floor
        direct_score = _compute_transition_score(
            pier_a, pier_b, X_full, X_start, X_mid, X_end, cfg
        )
        edges_scored = 1
        if direct_score >= cfg.transition_floor:
            return [], 0, edges_scored, None
        else:
            return None, 0, edges_scored, f"direct transition below floor ({direct_score:.3f} < {cfg.transition_floor:.3f})"

    # Progress model (A→B) in sonic similarity space (X_full_norm).
    progress_active = bool(cfg.progress_enabled)
    progress_eps = float(cfg.progress_monotonic_epsilon) if math.isfinite(float(cfg.progress_monotonic_epsilon)) else 0.0
    progress_eps = float(max(0.0, progress_eps))
    progress_weight = float(cfg.progress_penalty_weight) if math.isfinite(float(cfg.progress_penalty_weight)) else 0.0
    progress_weight = float(max(0.0, progress_weight))

    progress_arc_enabled = bool(cfg.progress_arc_enabled)
    progress_arc_weight = float(cfg.progress_arc_weight)
    if not math.isfinite(progress_arc_weight):
        progress_arc_weight = 0.0
    progress_arc_weight = float(max(0.0, progress_arc_weight))
    progress_arc_shape = str(cfg.progress_arc_shape or "linear").strip().lower()
    if progress_arc_shape not in {"linear", "arc"}:
        progress_arc_shape = "linear"
    progress_arc_tolerance = float(cfg.progress_arc_tolerance)
    if not math.isfinite(progress_arc_tolerance):
        progress_arc_tolerance = 0.0
    progress_arc_tolerance = float(max(0.0, progress_arc_tolerance))
    progress_arc_loss = str(cfg.progress_arc_loss or "abs").strip().lower()
    if progress_arc_loss not in {"abs", "squared", "huber"}:
        progress_arc_loss = "abs"
    progress_arc_huber_delta = float(cfg.progress_arc_huber_delta)
    if not math.isfinite(progress_arc_huber_delta) or progress_arc_huber_delta <= 0:
        progress_arc_huber_delta = 0.10
    progress_arc_max_step = cfg.progress_arc_max_step
    if isinstance(progress_arc_max_step, (int, float)) and math.isfinite(float(progress_arc_max_step)):
        progress_arc_max_step = float(progress_arc_max_step)
        if progress_arc_max_step <= 0:
            progress_arc_max_step = None
    else:
        progress_arc_max_step = None
    progress_arc_max_step_mode = str(cfg.progress_arc_max_step_mode or "penalty").strip().lower()
    if progress_arc_max_step_mode not in {"penalty", "gate"}:
        progress_arc_max_step_mode = "penalty"
    progress_arc_max_step_penalty = float(cfg.progress_arc_max_step_penalty)
    if not math.isfinite(progress_arc_max_step_penalty):
        progress_arc_max_step_penalty = 0.0
    progress_arc_max_step_penalty = float(max(0.0, progress_arc_max_step_penalty))
    progress_arc_autoscale_enabled = bool(cfg.progress_arc_autoscale_enabled)
    progress_arc_autoscale_min_distance = float(cfg.progress_arc_autoscale_min_distance)
    if not math.isfinite(progress_arc_autoscale_min_distance):
        progress_arc_autoscale_min_distance = 0.0
    progress_arc_autoscale_min_distance = float(max(0.0, progress_arc_autoscale_min_distance))
    progress_arc_autoscale_distance_scale = float(cfg.progress_arc_autoscale_distance_scale)
    if not math.isfinite(progress_arc_autoscale_distance_scale) or progress_arc_autoscale_distance_scale <= 0:
        progress_arc_autoscale_distance_scale = 0.0
    progress_arc_autoscale_per_step_scale = bool(cfg.progress_arc_autoscale_per_step_scale)
    progress_arc_effective_weight = 0.0

    progress_by_idx: Dict[int, float] = {}
    ab_distance: Optional[float] = None
    if progress_active:
        vec_a_full = X_full_norm[pier_a]
        vec_b_full = X_full_norm[pier_b]
        d = vec_b_full - vec_a_full
        denom = float(np.dot(d, d))
        if (not math.isfinite(denom)) or denom <= 1e-12:
            progress_active = False
        else:
            ab_distance = float(math.sqrt(denom))
            progress_arc_effective_weight = float(progress_arc_weight)
            if progress_arc_enabled and progress_arc_autoscale_enabled:
                if ab_distance < progress_arc_autoscale_min_distance:
                    progress_arc_effective_weight = 0.0
                elif progress_arc_autoscale_distance_scale > 0:
                    scale = min(1.0, ab_distance / progress_arc_autoscale_distance_scale)
                    progress_arc_effective_weight *= float(scale)
            if progress_arc_enabled and progress_arc_autoscale_per_step_scale:
                progress_arc_effective_weight *= 1.0 / float(max(1, interior_length))

            progress_by_idx[pier_a] = 0.0
            progress_by_idx[pier_b] = 1.0
            for cand in candidates:
                i = int(cand)
                t_raw = float(np.dot((X_full_norm[i] - vec_a_full), d) / denom)
                t = 0.0 if not math.isfinite(t_raw) else float(max(0.0, min(1.0, t_raw)))
                progress_by_idx[i] = t
    if arc_stats is not None:
        arc_stats.update(
            {
                "enabled": bool(progress_arc_enabled and progress_active),
                "shape": str(progress_arc_shape),
                "base_weight": float(progress_arc_weight),
                "effective_weight": float(progress_arc_effective_weight),
                "tolerance": float(progress_arc_tolerance),
                "loss": str(progress_arc_loss),
                "huber_delta": float(progress_arc_huber_delta),
                "max_step": (float(progress_arc_max_step) if progress_arc_max_step is not None else None),
                "max_step_mode": str(progress_arc_max_step_mode),
                "max_step_penalty": float(progress_arc_max_step_penalty),
                "autoscale": {
                    "enabled": bool(progress_arc_autoscale_enabled),
                    "min_distance": float(progress_arc_autoscale_min_distance),
                    "distance_scale": float(progress_arc_autoscale_distance_scale),
                    "per_step_scale": bool(progress_arc_autoscale_per_step_scale),
                },
                "ab_distance": (float(ab_distance) if ab_distance is not None else None),
                "steps": int(interior_length),
            }
        )

    vec_b_full = X_full_norm[pier_b]
    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    genre_cache: Dict[tuple[int, int], float] = {}
    genre_cache_hits = 0
    genre_cache_misses = 0

    # Phase 3: Use correct genre matrix for genre_sim (IDF if enabled)
    X_genre_for_sim = X_genre_norm_idf if X_genre_norm_idf is not None else X_genre_norm

    def _get_genre_sim(a_idx: int, b_idx: int) -> Optional[float]:
        nonlocal genre_cache_hits, genre_cache_misses
        if X_genre_for_sim is None:
            return None
        key = (int(a_idx), int(b_idx))
        if key in genre_cache:
            genre_cache_hits += 1
            return genre_cache[key]
        genre_cache_misses += 1
        val = float(np.dot(X_genre_for_sim[a_idx], X_genre_for_sim[b_idx]))
        genre_cache[key] = val
        return val

    def _progress_arc_loss(cand_t: float, target_t: float) -> float:
        err0 = abs(float(cand_t) - float(target_t))
        err = max(0.0, err0 - progress_arc_tolerance)
        return _progress_arc_loss_value(err, progress_arc_loss, progress_arc_huber_delta)

    def _record_genre_cache_stats() -> None:
        if genre_cache_stats is None:
            return
        total = int(genre_cache_hits + genre_cache_misses)
        hit_rate = float(genre_cache_hits) / float(total) if total > 0 else None
        genre_cache_stats.update(
            {
                "hits": int(genre_cache_hits),
                "misses": int(genre_cache_misses),
                "hit_rate": float(hit_rate) if hit_rate is not None else None,
                "entries": int(len(genre_cache)),
            }
        )

    # Initialize beam with pier_a
    used_artists_init: Set[str] = set()

    # Add boundary context from previous segments (cross-segment min_gap enforcement)
    if recent_global_artists:
        for artist_key in recent_global_artists:
            if artist_key:
                used_artists_init.add(str(artist_key))

    if artist_key_by_idx is not None:
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

        if cfg.disallow_pier_artists_in_interiors:
            # Add pier artist keys to used_artists_init
            for pier_idx in [pier_a, pier_b]:
                pier_artist_str = str(artist_key_by_idx.get(int(pier_idx), "") or "")
                if pier_artist_str:
                    if use_identity:
                        # Identity mode: add all identity keys for pier artist
                        pier_identity_keys = resolve_artist_identity_keys(pier_artist_str, artist_identity_cfg)
                        used_artists_init.update(pier_identity_keys)
                    else:
                        # Legacy mode: single artist key
                        used_artists_init.add(pier_artist_str)

        if cfg.disallow_seed_artist_in_interiors and seed_artist_key:
            if use_identity:
                # Identity mode: resolve seed artist to identity keys
                seed_identity_keys = resolve_artist_identity_keys(seed_artist_key, artist_identity_cfg)
                used_artists_init.update(seed_identity_keys)
            else:
                # Legacy mode: single seed artist key
                used_artists_init.add(str(seed_artist_key))

    initial_state = BeamState(
        path=[pier_a],
        score=0.0,
        used={pier_a, pier_b},
        used_artists=used_artists_init,
        last_progress=0.0,
    )
    beam = [initial_state]

    # Waypoint rank impact diagnostic (opt-in, default disabled)
    rank_impact_enabled = bool(
        cfg.dj_diagnostics_waypoint_rank_impact_enabled
        and waypoint_enabled
        and g_targets is not None
    )
    rank_impact_sampled_steps: Set[int] = set()
    rank_impact_results: List[Dict[str, Any]] = []
    if rank_impact_enabled:
        sample_count = int(cfg.dj_diagnostics_waypoint_rank_sample_steps)
        if sample_count > 0 and interior_length > 0:
            # Evenly-spaced sampling: e.g., [0, 7, 14] for length=15, sample_count=3
            if sample_count >= interior_length:
                rank_impact_sampled_steps = set(range(interior_length))
            else:
                step_interval = float(interior_length) / float(sample_count)
                rank_impact_sampled_steps = {
                    int(round(i * step_interval)) for i in range(sample_count)
                }
                # Ensure we don't exceed interior_length-1
                rank_impact_sampled_steps = {s for s in rank_impact_sampled_steps if s < interior_length}

    # TASK A: Track unique candidates that pass gates on first step
    pool_after_gating_candidates: Set[int] = set()

    # Phase 3 fix: Track actual applied waypoint deltas during beam search for correct stats
    # (Stats were using wrong sim0=0.0 instead of per-step baseline)
    chosen_waypoint_deltas: List[float] = []  # Actual applied deltas per step
    chosen_waypoint_sims: List[float] = []    # Raw sims per step (for context)
    chosen_waypoint_sim0s: List[float] = []   # Baselines per step (for debugging)

    for step in range(interior_length):
        next_beam: List[BeamState] = []
        target_t = _step_fraction(step, interior_length)
        experiment_target_t = (
            _progress_target_curve(step, interior_length, progress_arc_shape)
            if progress_arc_enabled
            else target_t
        )
        g_target = g_targets[step] if waypoint_enabled and g_targets is not None else None

        # Phase 3: Centered waypoint mode - collect all candidate waypoint sims for this step
        waypoint_sim0 = 0.0  # Baseline for centered mode (median or mean)
        # Phase 3 fix: Store waypoint info per candidate for this step (for stats tracking)
        step_waypoint_info: Dict[int, tuple[float, float]] = {}  # cand_idx -> (sim, delta)
        if waypoint_enabled and waypoint_delta_mode == "centered" and g_target is not None and X_genre_norm is not None:
            # Collect waypoint sims for all valid candidates
            step_waypoint_sims: List[float] = []
            for state in beam:
                current = state.path[-1]
                for cand in candidates:
                    if cand in state.used:
                        continue
                    # Compute waypoint sim
                    waypoint_sim = float(np.dot(X_genre_norm[cand], g_target))
                    if math.isfinite(waypoint_sim):
                        step_waypoint_sims.append(waypoint_sim)

            # Compute baseline (sim0) from distribution
            if step_waypoint_sims:
                if waypoint_centered_baseline_method == "mean":
                    waypoint_sim0 = float(np.mean(step_waypoint_sims))
                else:  # median (default)
                    waypoint_sim0 = float(np.median(step_waypoint_sims))

        # Rank impact diagnostic: collect candidates for sampled steps
        step_is_sampled = rank_impact_enabled and step in rank_impact_sampled_steps
        step_candidates_for_ranking: List[Tuple[int, float, float, float, float]] = []  # (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)

        for state in beam:
            current = state.path[-1]
            apply_tie_break = (
                (genre_tie_break_band is not None and X_genre_norm is not None and penalty_strength > 0)
                or (waypoint_enabled and waypoint_tie_break_band is not None)
            )
            cand_entries: list[tuple[int, float, float, float, Optional[float], Optional[float]]] = []
            best_score = -float("inf")

            for cand in candidates:
                if cand in state.used:
                    continue

                # Artist diversity: check if candidate artist already used
                if artist_key_by_idx is not None:
                    use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

                    if use_identity:
                        # Identity mode: resolve to identity keys and check if ANY key is already used
                        cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                        if cand_artist_str:
                            cand_identity_keys = resolve_artist_identity_keys(cand_artist_str, artist_identity_cfg)
                            # Reject if ANY identity key overlaps with used_artists
                            if any(key in state.used_artists for key in cand_identity_keys):
                                continue
                    else:
                        # Legacy mode: single artist key
                        cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                        if cand_artist and cand_artist in state.used_artists:
                            continue

                if min(sim_to_a[cand], sim_to_b[cand]) < cfg.bridge_floor:
                    continue

                cand_t = 0.0
                if progress_active:
                    cand_t = float(progress_by_idx.get(int(cand), 0.0))
                    if cand_t < float(state.last_progress) - progress_eps:
                        continue
                    if progress_arc_enabled and progress_arc_max_step is not None:
                        step_jump = float(cand_t) - float(state.last_progress)
                        if progress_arc_max_step_mode == "gate":
                            if step_jump > float(progress_arc_max_step) + progress_eps:
                                continue

                # Compute transition score
                trans_score = _compute_transition_score(
                    current, cand, X_full, X_start, X_mid, X_end, cfg
                )

                # Hard floors: transition + bridge-local
                if trans_score < cfg.transition_floor:
                    continue

                # TASK A: Track candidates that pass all gates (step 0 only for pool_after_gating)
                if step == 0:
                    pool_after_gating_candidates.add(int(cand))

                sim_a = float(sim_to_a[cand])
                sim_b = float(sim_to_b[cand])
                denom = sim_a + sim_b
                bridge_score = 0.0 if denom <= 1e-9 else (2 * sim_a * sim_b) / denom

                # Add heuristic pull toward destination
                dest_pull = cfg.eta_destination_pull * float(np.dot(X_full_norm[cand], vec_b_full))

                combined_score = (
                    cfg.weight_bridge * bridge_score +
                    cfg.weight_transition * trans_score
                )
                if progress_active and progress_weight > 0:
                    combined_score -= progress_weight * abs(float(cand_t) - target_t)
                if progress_active and progress_arc_enabled and progress_arc_effective_weight > 0:
                    combined_score -= progress_arc_effective_weight * _progress_arc_loss(
                        float(cand_t),
                        experiment_target_t,
                    )
                if progress_active and progress_arc_enabled and progress_arc_max_step is not None:
                    step_jump = float(cand_t) - float(state.last_progress)
                    if step_jump > float(progress_arc_max_step):
                        if progress_arc_max_step_mode == "penalty" and progress_arc_max_step_penalty > 0:
                            combined_score -= progress_arc_max_step_penalty * (step_jump - float(progress_arc_max_step))

                genre_sim = None
                if X_genre_norm is not None:
                    genre_sim = _get_genre_sim(int(current), int(cand))
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if cfg.genre_tiebreak_weight:
                            combined_score += cfg.genre_tiebreak_weight * genre_sim

                waypoint_sim = None
                if waypoint_enabled and g_target is not None and X_genre_norm is not None:
                    waypoint_sim = float(np.dot(X_genre_norm[cand], g_target))

                edges_scored += 1

                if apply_tie_break:
                    cand_entries.append((int(cand), float(cand_t), float(combined_score), float(dest_pull), genre_sim, waypoint_sim))
                    if combined_score > best_score:
                        best_score = float(combined_score)
                else:
                    base_score_for_rank = float(combined_score)
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if penalty_strength > 0 and genre_sim < penalty_threshold:
                            combined_score *= (1.0 - penalty_strength)
                            genre_penalty_hits += 1
                    waypoint_delta_val = 0.0
                    if waypoint_enabled:
                        waypoint_delta_val = _waypoint_delta(waypoint_sim, waypoint_sim0)
                        combined_score += waypoint_delta_val
                        # Phase 3 fix: Store for stats tracking
                        if waypoint_sim is not None:
                            step_waypoint_info[int(cand)] = (float(waypoint_sim), float(waypoint_delta_val))

                    # Coverage bonus (Phase 2 + Phase 3): reward matching anchor top-K genres with schedule decay
                    coverage_bonus_val = 0.0
                    if coverage_enabled and X_genre_for_coverage_presence is not None:
                        cand_genre_vec = X_genre_for_coverage_presence[cand]
                        coverage_A = _compute_coverage(
                            cand_genre_vec, topk_A, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_B = _compute_coverage(
                            cand_genre_vec, topk_B, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_bonus_val = _compute_coverage_bonus(
                            step, interior_length, coverage_A, coverage_B,
                            float(cfg.dj_genre_coverage_weight), float(cfg.dj_genre_coverage_power)
                        )
                        combined_score += coverage_bonus_val

                    # Rank impact: collect (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)
                    if step_is_sampled:
                        step_candidates_for_ranking.append((
                            int(cand),
                            float(base_score_for_rank),
                            float(waypoint_delta_val),
                            float(coverage_bonus_val),
                            float(combined_score)
                        ))

                    new_score = state.score + combined_score + dest_pull
                    new_path = state.path + [cand]
                    new_used = state.used | {cand}
                    new_used_artists = state.used_artists
                    if artist_key_by_idx is not None:
                        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
                        if use_identity:
                            # Identity mode: add ALL identity keys to used_artists
                            cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist_str:
                                cand_identity_keys = resolve_artist_identity_keys(cand_artist_str, artist_identity_cfg)
                                new_used_artists = state.used_artists | cand_identity_keys
                        else:
                            # Legacy mode: add single artist key
                            cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist:
                                new_used_artists = state.used_artists | {cand_artist}
                    new_last_progress = float(state.last_progress)
                    if progress_active:
                        new_last_progress = float(progress_by_idx.get(int(cand), 0.0))

                    next_beam.append(BeamState(
                        path=new_path,
                        score=new_score,
                        used=new_used,
                        used_artists=new_used_artists,
                        last_progress=new_last_progress,
                    ))

            if apply_tie_break and cand_entries:
                for cand, cand_t, base_score, dest_pull, genre_sim, waypoint_sim in cand_entries:
                    base_score_for_rank = float(base_score)
                    combined_score = float(base_score)
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if genre_tie_break_band is not None:
                            if (best_score - combined_score) <= float(genre_tie_break_band):
                                if penalty_strength > 0 and genre_sim < penalty_threshold:
                                    combined_score *= (1.0 - penalty_strength)
                                    genre_penalty_hits += 1
                        else:
                            if penalty_strength > 0 and genre_sim < penalty_threshold:
                                combined_score *= (1.0 - penalty_strength)
                                genre_penalty_hits += 1
                    waypoint_delta_val = 0.0
                    if waypoint_enabled:
                        # Always apply waypoint scoring (tie-break band removed in Phase 2)
                        waypoint_delta_val = _waypoint_delta(waypoint_sim, waypoint_sim0)
                        combined_score += waypoint_delta_val
                        # Phase 3 fix: Store for stats tracking
                        if waypoint_sim is not None:
                            step_waypoint_info[int(cand)] = (float(waypoint_sim), float(waypoint_delta_val))

                    # Coverage bonus (Phase 2 + Phase 3): reward matching anchor top-K genres with schedule decay
                    coverage_bonus_val = 0.0
                    if coverage_enabled and X_genre_for_coverage_presence is not None:
                        cand_genre_vec = X_genre_for_coverage_presence[cand]
                        coverage_A = _compute_coverage(
                            cand_genre_vec, topk_A, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_B = _compute_coverage(
                            cand_genre_vec, topk_B, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_bonus_val = _compute_coverage_bonus(
                            step, interior_length, coverage_A, coverage_B,
                            float(cfg.dj_genre_coverage_weight), float(cfg.dj_genre_coverage_power)
                        )
                        combined_score += coverage_bonus_val

                    # Rank impact: collect (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)
                    if step_is_sampled:
                        step_candidates_for_ranking.append((
                            int(cand),
                            float(base_score_for_rank),
                            float(waypoint_delta_val),
                            float(coverage_bonus_val),
                            float(combined_score)
                        ))

                    new_score = state.score + combined_score + float(dest_pull)
                    new_path = state.path + [cand]
                    new_used = state.used | {cand}
                    new_used_artists = state.used_artists
                    if artist_key_by_idx is not None:
                        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
                        if use_identity:
                            cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist_str:
                                cand_identity_keys = resolve_artist_identity_keys(cand_artist_str, artist_identity_cfg)
                                new_used_artists = state.used_artists | cand_identity_keys
                        else:
                            cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist:
                                new_used_artists = state.used_artists | {cand_artist}
                    new_last_progress = float(state.last_progress)
                    if progress_active:
                        new_last_progress = float(progress_by_idx.get(int(cand), 0.0))

                    next_beam.append(BeamState(
                        path=new_path,
                        score=new_score,
                        used=new_used,
                        used_artists=new_used_artists,
                        last_progress=new_last_progress,
                    ))

        if not next_beam:
            _record_genre_cache_stats()
            return None, genre_penalty_hits, edges_scored, f"no valid continuations at step={step}"

        # Keep top beam_width states
        next_beam.sort(key=lambda s: s.score, reverse=True)
        beam = next_beam[:beam_width]

        # Phase 3 fix: Track waypoint info for the chosen candidate (top beam state's last track)
        if beam and waypoint_enabled:
            chosen_cand = beam[0].path[-1]  # Last track in best path
            if chosen_cand in step_waypoint_info:
                sim, delta = step_waypoint_info[chosen_cand]
                chosen_waypoint_sims.append(sim)
                chosen_waypoint_deltas.append(delta)
                chosen_waypoint_sim0s.append(waypoint_sim0)
            else:
                # No waypoint info (shouldn't happen if waypoint_enabled, but handle gracefully)
                chosen_waypoint_sims.append(0.0)
                chosen_waypoint_deltas.append(0.0)
                chosen_waypoint_sim0s.append(waypoint_sim0)

        # Rank impact: compute metrics for sampled steps
        if step_is_sampled and step_candidates_for_ranking:
            # Get unique candidates (may have duplicates from different beam states)
            cand_scores_dict: Dict[int, Tuple[float, float, float, float]] = {}
            for cand_idx, base_score, waypoint_delta, coverage_bonus, full_score in step_candidates_for_ranking:
                if cand_idx not in cand_scores_dict:
                    cand_scores_dict[cand_idx] = (base_score, waypoint_delta, coverage_bonus, full_score)
                else:
                    # If duplicate, keep the one with higher full_score
                    existing_full = cand_scores_dict[cand_idx][3]
                    if full_score > existing_full:
                        cand_scores_dict[cand_idx] = (base_score, waypoint_delta, coverage_bonus, full_score)

            # Convert to list for ranking
            cand_list = [(idx, base, delta, cov, full) for idx, (base, delta, cov, full) in cand_scores_dict.items()]

            # Rank by base_score (descending)
            cand_list_by_base = sorted(cand_list, key=lambda x: x[1], reverse=True)
            # Rank by full_score (descending)
            cand_list_by_full = sorted(cand_list, key=lambda x: x[4], reverse=True)

            # Create rank maps: cand_idx -> rank (1-indexed)
            base_rank_map = {cand_idx: rank + 1 for rank, (cand_idx, _, _, _, _) in enumerate(cand_list_by_base)}
            full_rank_map = {cand_idx: rank + 1 for rank, (cand_idx, _, _, _, _) in enumerate(cand_list_by_full)}

            # Compute metrics
            winner_changed = (cand_list_by_base[0][0] != cand_list_by_full[0][0])
            topK = min(10, len(cand_list))
            topK_base_set = {cand_idx for cand_idx, _, _, _, _ in cand_list_by_base[:topK]}
            topK_reordered_count = sum(
                1 for cand_idx in topK_base_set
                if base_rank_map[cand_idx] != full_rank_map[cand_idx]
            )
            rank_deltas = [abs(base_rank_map[cand_idx] - full_rank_map[cand_idx]) for cand_idx in cand_scores_dict]
            mean_abs_rank_delta = float(np.mean(rank_deltas)) if rank_deltas else 0.0
            max_rank_jump = int(max(rank_deltas)) if rank_deltas else 0

            # Store step result
            step_result = {
                "step": step,
                "interior_length": interior_length,
                "winner_changed": winner_changed,
                "topK_reordered_count": topK_reordered_count,
                "topK": topK,
                "mean_abs_rank_delta": mean_abs_rank_delta,
                "max_rank_jump": max_rank_jump,
                "total_candidates": len(cand_list),
                # Top-10 table for detailed logging (optional)
                "top10_table": [
                    {
                        "cand_idx": cand_idx,
                        "base_score": base,
                        "waypoint_delta": delta,
                        "coverage_bonus": cov,
                        "full_score": full,
                        "base_rank": base_rank_map[cand_idx],
                        "full_rank": full_rank_map[cand_idx],
                        "rank_delta": base_rank_map[cand_idx] - full_rank_map[cand_idx],
                    }
                    for cand_idx, base, delta, cov, full in cand_list_by_base[:topK]
                ]
            }
            rank_impact_results.append(step_result)

            # Phase 2 per-step logging: target genres and top candidates
            if cfg.dj_bridging_enabled and g_target is not None and genre_vocab is not None:
                # Log target genre distribution (top 5-8)
                target_topk = _extract_top_genres(g_target, 8)
                if target_topk:
                    genre_names = [f"{genre_vocab[idx]}={weight:.3f}" for idx, weight in target_topk[:5]]
                    logger.info("  [Step %d/%d] Target genres: %s", step, interior_length, ", ".join(genre_names))

                # Log best 3 candidates by full_score (with genre alignment)
                if X_genre_for_coverage is not None and len(cand_list_by_full) >= 3:
                    logger.info("  [Step %d/%d] Top-3 candidates by full_score:", step, interior_length)
                    for rank, (cand_idx, base, delta, cov, full) in enumerate(cand_list_by_full[:3], start=1):
                        # Compute genre alignment to target
                        cand_genre_vec = X_genre_for_coverage[cand_idx]
                        genre_sim = float(np.dot(cand_genre_vec, g_target))
                        logger.info("    #%d: idx=%d base=%.3f waypoint=%.3f coverage=%.3f full=%.3f genre_sim=%.3f",
                                    rank, cand_idx, base, delta, cov, full, genre_sim)

    # Final step: connect to pier_b
    best_final: Optional[BeamState] = None
    best_final_score = -float('inf')

    for state in beam:
        last = state.path[-1]
        final_trans = _compute_transition_score(
            last, pier_b, X_full, X_start, X_mid, X_end, cfg
        )

        # Hard floor on final transition
        if final_trans < cfg.transition_floor:
            continue

        final_edge_score = final_trans
        edges_scored += 1
        if X_genre_norm is not None:
            genre_sim = _get_genre_sim(int(last), int(pier_b))
            if genre_sim is not None and math.isfinite(genre_sim):
                if cfg.genre_tiebreak_weight:
                    final_edge_score += cfg.genre_tiebreak_weight * genre_sim
                if (
                    penalty_strength > 0
                    and genre_sim < penalty_threshold
                ):
                    final_edge_score *= (1.0 - penalty_strength)
                    genre_penalty_hits += 1

        total_score = state.score + final_edge_score
        if total_score > best_final_score:
            best_final_score = total_score
            best_final = state

    if best_final is None:
        _record_genre_cache_stats()
        return None, genre_penalty_hits, edges_scored, "no valid final connection to destination"

    # Compute waypoint diagnostics for chosen path
    if waypoint_stats is not None and waypoint_enabled and g_targets is not None and X_genre_norm is not None:
        # Phase 3 fix: Use tracked actual applied deltas instead of recomputing with wrong sim0
        # The old code recomputed deltas with sim0=0.0 (default), which is wrong for centered mode.
        # Now we use the actual deltas that were applied during beam search.
        if chosen_waypoint_sims and chosen_waypoint_deltas:
            waypoint_sims = chosen_waypoint_sims
            waypoint_deltas = chosen_waypoint_deltas
        else:
            # Fallback: recompute (shouldn't happen if waypoint_enabled, but handle gracefully)
            full_path = best_final.path  # includes pier_a at start
            interior_path = full_path[1:]  # interior tracks only
            waypoint_sims: List[float] = []
            waypoint_deltas: List[float] = []
            for step_idx, track_idx in enumerate(interior_path):
                if step_idx < len(g_targets):
                    g_target = g_targets[step_idx]
                    waypoint_sim = float(np.dot(X_genre_norm[track_idx], g_target))
                    # Note: This fallback still has the sim0=0.0 issue, but it's rare
                    waypoint_delta = _waypoint_delta(waypoint_sim)
                    waypoint_sims.append(waypoint_sim)
                    waypoint_deltas.append(waypoint_delta)

        if waypoint_sims:
            waypoint_stats["waypoint_enabled"] = True
            waypoint_stats["waypoint_sims"] = waypoint_sims
            waypoint_stats["waypoint_deltas"] = waypoint_deltas
            waypoint_stats["mean_waypoint_sim"] = float(np.mean(waypoint_sims))
            waypoint_stats["p50_waypoint_sim"] = float(np.percentile(waypoint_sims, 50))
            waypoint_stats["p90_waypoint_sim"] = float(np.percentile(waypoint_sims, 90))
            waypoint_stats["min_waypoint_sim"] = float(np.min(waypoint_sims))
            waypoint_stats["max_waypoint_sim"] = float(np.max(waypoint_sims))
            waypoint_stats["waypoint_delta_applied_count"] = int(sum(1 for d in waypoint_deltas if abs(d) > 1e-9))
            waypoint_stats["mean_waypoint_delta"] = float(np.mean(waypoint_deltas))
        else:
            waypoint_stats["waypoint_enabled"] = True
            waypoint_stats["waypoint_sims"] = []
            waypoint_stats["mean_waypoint_sim"] = 0.0

        # Add rank impact results if computed
        if rank_impact_results:
            waypoint_stats["rank_impact_results"] = rank_impact_results
    elif waypoint_stats is not None:
        waypoint_stats["waypoint_enabled"] = False

    # TASK A: Add pool_after_gating count to waypoint_stats
    if waypoint_stats is not None:
        waypoint_stats["pool_after_gating_count"] = len(pool_after_gating_candidates)

    # Return interior tracks (exclude pier_a which is path[0])
    _record_genre_cache_stats()
    return best_final.path[1:], genre_penalty_hits, edges_scored, None


def _compute_edge_scores(
    path: List[int],
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> Tuple[float, float]:
    """Compute worst and mean edge scores for a path."""
    if len(path) < 2:
        return (1.0, 1.0)

    scores = []
    for i in range(len(path) - 1):
        score = _compute_transition_score(
            path[i], path[i + 1], X_full, X_start, X_mid, X_end, cfg
        )
        scores.append(score)

    return (min(scores), sum(scores) / len(scores))


def _enforce_min_gap_global(
    indices: List[int],
    artist_keys: Optional[np.ndarray] = None,
    min_gap: int = 1,
    *,
    bundle: Optional[ArtifactBundle] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
) -> Tuple[List[int], int]:
    """
    Drop tracks that would violate a global min_gap across concatenated segments.

    Pier-bridge already enforces one-per-artist per segment, but adjacent
    duplicates can appear at segment boundaries. This pass removes any track
    that would repeat a normalized artist within the last `min_gap` slots.

    If artist_identity_cfg is provided and enabled, uses identity-based matching
    (collapsing ensemble variants and splitting collaborations). Each collaboration
    track contributes ALL participant identity keys to the recent window.
    """
    if not indices or min_gap <= 0:
        return indices, 0

    recent: List[str] = []
    output: List[int] = []
    dropped = 0

    use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

    for idx in indices:
        if use_identity:
            # Identity mode: resolve artist string to identity keys (Set[str])
            artist_str = ""
            if bundle is not None:
                try:
                    artist_str = identity_keys_for_index(bundle, int(idx)).artist
                except Exception:
                    artist_str = ""
            if not artist_str and artist_keys is not None:
                try:
                    artist_str = str(artist_keys[int(idx)])
                except Exception:
                    artist_str = ""

            # Resolve to identity keys
            identity_keys_set = resolve_artist_identity_keys(artist_str, artist_identity_cfg)

            # Check if ANY identity key violates min_gap
            violated_key = None
            for identity_key in identity_keys_set:
                if identity_key in recent:
                    violated_key = identity_key
                    break

            if violated_key is not None:
                dropped += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Rejected candidate idx=%d due to identity_min_gap: key=%r in recent window (distance<=%d)",
                        idx, violated_key, min_gap
                    )
                continue

            # Accept track: add to output and update recent window with ALL identity keys
            output.append(idx)
            for identity_key in identity_keys_set:
                recent.append(identity_key)
            # Trim recent window to size min_gap
            while len(recent) > min_gap:
                recent.pop(0)
        else:
            # Legacy mode: single artist key
            key = ""
            if bundle is not None:
                try:
                    key = identity_keys_for_index(bundle, int(idx)).artist_key
                except Exception:
                    key = ""
            if not key and artist_keys is not None:
                try:
                    key = normalize_artist_key(str(artist_keys[int(idx)]))
                except Exception:
                    key = ""
            if not key:
                key = f"unknown_artist:{idx}"

            if key in recent:
                dropped += 1
                continue

            output.append(idx)
            recent.append(key)
            if len(recent) > min_gap:
                recent.pop(0)

    return output, dropped


def build_pier_bridge_playlist(
    *,
    seed_track_ids: List[str],
    total_tracks: int,
    bundle: ArtifactBundle,
    candidate_pool_indices: List[int],
    cfg: Optional[PierBridgeConfig] = None,
    min_genre_similarity: Optional[float] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_method: str = "ensemble",
    internal_connector_indices: Optional[Set[int]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    allowed_track_ids_set: Optional[set[str]] = None,
    infeasible_handling: Optional[InfeasibleHandlingConfig] = None,
    audit_config: Optional[RunAuditConfig] = None,
    audit_events: Optional[list[RunAuditEvent]] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
) -> PierBridgeResult:
    """
    Build playlist using pier + bridge strategy.

    Args:
        seed_track_ids: List of seed track IDs (will become piers)
        total_tracks: Target total playlist length
        bundle: Artifact bundle with sonic features
        candidate_pool_indices: Pre-filtered candidate pool indices
        cfg: Configuration (uses defaults if None)
        min_genre_similarity: Optional genre gate threshold
        X_genre_smoothed: Genre vectors for gating
        genre_method: Genre similarity method

    Returns:
        PierBridgeResult with ordered track IDs and diagnostics
    """
    if cfg is None:
        cfg = PierBridgeConfig()
    if infeasible_handling is None:
        infeasible_handling = InfeasibleHandlingConfig()
    if audit_config is None:
        audit_config = RunAuditConfig()
    audit_enabled = bool(audit_config.enabled) and audit_events is not None
    top_k = int(audit_config.include_top_k) if audit_enabled else 0

    num_seeds = len(seed_track_ids)
    if num_seeds == 0:
        raise ValueError("At least one seed is required")
    if num_seeds > total_tracks:
        raise ValueError(f"Number of seeds ({num_seeds}) exceeds total_tracks ({total_tracks})")

    # Resolve seed indices
    seed_indices: List[int] = []
    for tid in seed_track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            raise ValueError(f"Seed track not found in bundle: {tid}")
        seed_indices.append(idx)

    # Remove duplicates while preserving order
    seed_indices = list(dict.fromkeys(seed_indices))
    num_seeds = len(seed_indices)
    seed_id_set = {str(bundle.track_ids[i]) for i in seed_indices}

    logger.info("Pier+Bridge: %d seeds, target %d tracks", num_seeds, total_tracks)

    # Deduplicate candidate pool by artist+title
    deduped_pool, _ = _dedupe_candidate_pool(candidate_pool_indices, bundle)

    # Exclude seed indices from candidate pool
    seed_set = set(seed_indices)
    universe = [idx for idx in deduped_pool if idx not in seed_set]

    logger.info("Pier+Bridge: universe size after dedupe and seed exclusion: %d", len(universe))

    # Get sonic matrices (raw beat3tower space)
    X_full_raw = bundle.X_sonic
    X_start_raw = bundle.X_sonic_start
    X_mid_raw = bundle.X_sonic_mid
    X_end_raw = bundle.X_sonic_end

    # Similarity space for bridge gating (full vectors) must match DS admission
    from src.similarity.sonic_variant import compute_sonic_variant_matrix, resolve_sonic_variant

    sonic_variant = resolve_sonic_variant(explicit_variant=cfg.sonic_variant, config_variant=None)
    X_full_variant, _ = compute_sonic_variant_matrix(X_full_raw, sonic_variant, l2=False)
    X_full_norm = _l2_normalize_rows(X_full_variant)
    logger.debug("Pier+Bridge sonic sim space: variant=%s dim=%d", sonic_variant, int(X_full_norm.shape[1]))

    # Transition space (optional tower weights + optional mean-centering)
    from src.similarity.sonic_variant import apply_transition_weights

    X_full_tr, _ = apply_transition_weights(X_full_raw, config_weights=cfg.transition_weights)
    X_start_tr = None
    X_mid_tr = None
    X_end_tr = None
    if X_start_raw is not None:
        X_start_tr, _ = apply_transition_weights(X_start_raw, config_weights=cfg.transition_weights)
    if X_mid_raw is not None:
        X_mid_tr, _ = apply_transition_weights(X_mid_raw, config_weights=cfg.transition_weights)
    if X_end_raw is not None:
        X_end_tr, _ = apply_transition_weights(X_end_raw, config_weights=cfg.transition_weights)

    if cfg.center_transitions:
        X_full_tr = X_full_tr - X_full_tr.mean(axis=0, keepdims=True)
        if X_start_tr is not None:
            X_start_tr = X_start_tr - X_start_tr.mean(axis=0, keepdims=True)
        if X_mid_tr is not None:
            X_mid_tr = X_mid_tr - X_mid_tr.mean(axis=0, keepdims=True)
        if X_end_tr is not None:
            X_end_tr = X_end_tr - X_end_tr.mean(axis=0, keepdims=True)

    X_full_tr_norm = _l2_normalize_rows(X_full_tr)
    X_start_tr_norm = _l2_normalize_rows(X_start_tr) if X_start_tr is not None else None
    X_mid_tr_norm = _l2_normalize_rows(X_mid_tr) if X_mid_tr is not None else None
    X_end_tr_norm = _l2_normalize_rows(X_end_tr) if X_end_tr is not None else None

    # Instrument transition saturation (sampled); compare raw vs transformed end→start
    if logger.isEnabledFor(logging.DEBUG) and X_end_raw is not None and X_start_raw is not None:
        rng = np.random.default_rng(0)
        n = int(X_full_raw.shape[0])
        sample_n = int(min(5000, n))
        prev = rng.integers(0, n, size=sample_n)
        cand = rng.integers(0, n, size=sample_n)
        end_raw = X_end_raw[prev]
        start_raw = X_start_raw[cand]
        raw_sims = np.sum(end_raw * start_raw, axis=1) / (
            (np.linalg.norm(end_raw, axis=1) * np.linalg.norm(start_raw, axis=1)) + 1e-12
        )
        end_tr = X_end_tr_norm[prev] if X_end_tr_norm is not None else None
        start_tr = X_start_tr_norm[cand] if X_start_tr_norm is not None else None
        if end_tr is not None and start_tr is not None:
            tr_sims = np.sum(end_tr * start_tr, axis=1)
            if cfg.center_transitions:
                tr_sims = (tr_sims + 1.0) / 2.0
            logger.debug(
                "Transition end→start sample: raw[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] "
                "transformed[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] center_transitions=%s",
                float(np.min(raw_sims)),
                float(np.percentile(raw_sims, 5)),
                float(np.percentile(raw_sims, 50)),
                float(np.percentile(raw_sims, 95)),
                float(np.max(raw_sims)),
                float(np.min(tr_sims)),
                float(np.percentile(tr_sims, 5)),
                float(np.percentile(tr_sims, 50)),
                float(np.percentile(tr_sims, 95)),
                float(np.max(tr_sims)),
                bool(cfg.center_transitions),
            )

    # For seed ordering bridgeability heuristic, prefer transition-normalized mats when present
    X_start_norm = X_start_tr_norm
    X_end_norm = X_end_tr_norm

    # Extract genre matrices from bundle (Phase 2: needed for IDF and vector mode)
    # Prefer parameter if provided, otherwise extract from bundle
    X_genre_raw = getattr(bundle, "X_genre_raw", None)
    if X_genre_smoothed is None:
        X_genre_smoothed = getattr(bundle, "X_genre_smoothed", None)

    # Genre similarity for soft edge penalty / tiebreak (cosine on smoothed genre vectors)
    X_genre_use = X_genre_smoothed if X_genre_smoothed is not None else None
    X_genre_norm = None
    if X_genre_use is not None:
        denom_g = np.linalg.norm(X_genre_use, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre_use / denom_g

    # Compute IDF for genre vector mode (Phase 2)
    genre_idf: Optional[np.ndarray] = None
    X_genre_norm_idf: Optional[np.ndarray] = None
    if bool(cfg.dj_genre_use_idf) and bool(cfg.dj_bridging_enabled):
        if X_genre_raw is not None:
            logger.info("Computing genre IDF (power=%.2f norm=%s)...",
                        cfg.dj_genre_idf_power, cfg.dj_genre_idf_norm)
            genre_idf = _compute_genre_idf(X_genre_raw, cfg)
            logger.info("  IDF computed: min=%.3f median=%.3f max=%.3f",
                        float(np.min(genre_idf)), float(np.median(genre_idf)), float(np.max(genre_idf)))

            # Create IDF-weighted matrix for S3 pooling and beam search
            if X_genre_norm is not None:
                X_genre_norm_idf = _apply_idf_weighting(X_genre_norm, genre_idf)
        else:
            logger.warning("IDF enabled but X_genre_raw unavailable; using base genre weights")

    warnings: list[dict[str, Any]] = []
    if bool(cfg.dj_bridging_enabled):
        if X_genre_norm is None:
            warnings.append({
                "type": "genre_missing",
                "scope": "global",
                "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                "anchors_missing": int(num_seeds),
            })
        else:
            seed_vecs = X_genre_norm[seed_indices]
            norms = np.linalg.norm(seed_vecs, axis=1)
            missing_ids = [
                str(bundle.track_ids[idx])
                for idx, nval in zip(seed_indices, norms)
                if float(nval) <= 1e-8
            ]
            if missing_ids:
                warnings.append({
                    "type": "genre_missing",
                    "scope": "anchors",
                    "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                    "missing_anchor_ids": missing_ids,
                })
        if bool(cfg.dj_anchors_must_include_all) and len(seed_indices) != len(seed_track_ids):
            warnings.append({
                "type": "anchor_deduped",
                "scope": "anchors",
                "message": "Duplicate anchors were removed while must_include_all is set.",
                "requested_count": int(len(seed_track_ids)),
                "resolved_count": int(len(seed_indices)),
            })

    genre_graph: Optional[dict[str, list[tuple[str, float]]]] = None
    if bool(cfg.dj_bridging_enabled):
        route_shape = str(cfg.dj_route_shape or "linear").strip().lower()
        if route_shape == "ladder":
            genre_vocab = getattr(bundle, "genre_vocab", None)
            if genre_vocab is None:
                warnings.append({
                    "type": "genre_ladder_unavailable",
                    "scope": "global",
                    "message": "Genre ladder disabled; missing genre vocab.",
                })
            else:
                repo_root = Path(__file__).resolve().parents[2]
                genre_yaml = repo_root / "data" / "genre_similarity.yaml"
                if bool(cfg.dj_ladder_use_smoothed_waypoint_vectors):
                    _ensure_genre_similarity_overrides_loaded(genre_yaml)
                genre_graph = _load_genre_similarity_graph(
                    genre_yaml,
                    min_similarity=float(cfg.dj_ladder_min_similarity),
                )
                if not genre_graph:
                    warnings.append({
                        "type": "genre_ladder_unavailable",
                        "scope": "global",
                        "message": "Genre ladder disabled; similarity graph unavailable.",
                    })

    # Precompute allowed indices set if caller passed allowed_track_ids_set.
    # (In style-aware runs, the bundle is often already restricted, but this still
    # acts as a hard gate for candidate admission inside pier-bridge.)
    allowed_set_indices: Optional[Set[int]] = None
    if allowed_track_ids_set is not None:
        allowed_set_indices = set()
        for tid in allowed_track_ids_set:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                allowed_set_indices.add(idx)
        # Ensure piers are always allowed
        allowed_set_indices.update(seed_indices)

    # Order seeds by bridgeability (or preserve order if fixed)
    seed_ordering = str(cfg.dj_seed_ordering or "auto").strip().lower()
    if seed_ordering not in {"auto", "fixed"}:
        warnings.append({
            "type": "seed_ordering_invalid",
            "scope": "anchors",
            "message": f"Unknown seed_ordering '{seed_ordering}', defaulting to auto.",
        })
        seed_ordering = "auto"
    if bool(cfg.dj_bridging_enabled) and seed_ordering == "fixed":
        ordered_seeds = list(seed_indices)
    else:
        if bool(cfg.dj_bridging_enabled):
            ordered_seeds = _order_seeds_by_bridgeability(
                seed_indices,
                X_full_norm,
                X_start_norm,
                X_end_norm,
                X_genre_norm,
                weight_sonic=float(cfg.dj_seed_ordering_weight_sonic),
                weight_genre=float(cfg.dj_seed_ordering_weight_genre),
                weight_bridge=float(cfg.dj_seed_ordering_weight_bridge),
            )
        else:
            ordered_seeds = _order_seeds_by_bridgeability(
                seed_indices, X_full_norm, X_start_norm, X_end_norm
            )

    logger.info("Pier+Bridge: seed order = %s",
               [str(bundle.track_ids[i]) for i in ordered_seeds])

    # Handle single seed as both start AND end pier (arc structure)
    # This creates a playlist that starts from seed, explores, and returns to seed-similar sounds
    is_single_seed_arc = (num_seeds == 1)
    if is_single_seed_arc:
        # Duplicate the seed as both start and end pier
        ordered_seeds = [ordered_seeds[0], ordered_seeds[0]]
        num_segments = 1
        total_interior = total_tracks - 1  # Only one seed in final output
        logger.info("Pier+Bridge: single-seed arc mode (seed is both start and end pier)")
    else:
        num_segments = num_seeds - 1
        total_interior = total_tracks - num_seeds

    # Even split with remainder distributed to earlier segments
    base_length = total_interior // num_segments
    remainder = total_interior % num_segments
    segment_lengths = [
        base_length + (1 if i < remainder else 0)
        for i in range(num_segments)
    ]

    logger.info("Pier+Bridge: segment lengths = %s (total_interior=%d)",
               segment_lengths, total_interior)

    # Build segments
    global_used: Set[int] = set(ordered_seeds)  # Seeds are already "used"
    # Track-key dedupe across the full run: prevent "same song twice" even if track_id differs.
    seed_artist_key: Optional[str] = None
    try:
        if seed_indices:
            seed_artist_key = identity_keys_for_index(bundle, int(seed_indices[0])).artist_key
    except Exception:
        seed_artist_key = None

    seed_track_keys: Set[tuple[str, str]] = set()
    for sidx in set(int(i) for i in seed_indices):
        try:
            seed_track_keys.add(identity_keys_for_index(bundle, int(sidx)).track_key)
        except Exception:
            continue
    used_track_keys: Set[tuple[str, str]] = set(seed_track_keys)

    all_segments: List[List[int]] = []
    diagnostics: List[SegmentDiagnostics] = []
    soft_genre_penalty_hits_total = 0
    soft_genre_penalty_edges_scored_total = 0
    segment_bridge_floors_used: list[float] = []
    segment_backoff_attempts_used: list[int] = []

    # Boundary context tracking for cross-segment min_gap enforcement
    # Tracks artist keys from the last min_gap positions of the concatenated result
    MIN_GAP_GLOBAL = 1  # Cross-segment min_gap constraint
    recent_boundary_artists: List[str] = []

    def _bridge_floor_attempts(initial_floor: float) -> list[float]:
        if not infeasible_handling or not infeasible_handling.enabled:
            return [float(initial_floor)]
        steps = list(infeasible_handling.backoff_steps or ())
        if not steps:
            cur = float(initial_floor)
            while cur >= float(infeasible_handling.min_bridge_floor) - 1e-9:
                steps.append(round(cur, 2))
                cur -= 0.01
        attempts: list[float] = [float(initial_floor)]
        for v in steps:
            if not isinstance(v, (int, float)):
                continue
            f = float(v)
            if f < float(initial_floor) and f >= float(infeasible_handling.min_bridge_floor) - 1e-9:
                attempts.append(float(f))
        attempts = list(dict.fromkeys(attempts))
        max_attempts = max(1, int(infeasible_handling.max_attempts_per_segment))
        return attempts[:max_attempts]

    def _run_segment_backoff_attempts(
        *,
        cfg_attempt_base: PierBridgeConfig,
        segment_allow_detours: bool,
        segment_g_targets: Optional[list[np.ndarray]],
        pier_a: int,
        pier_b: int,
        interior_len: int,
        pier_a_id: str,
        pier_b_id: str,
        seg_idx: int,
        recent_boundary_artists: Optional[List[str]],
    ) -> dict[str, Any]:
        cfg = cfg_attempt_base
        segment_path: Optional[List[int]] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts = _bridge_floor_attempts(float(cfg.bridge_floor))
        backoff_used_count = 0
        widened_search_used = False
        last_failure_reason: Optional[str] = None

        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_waypoint_stats: Dict[str, Any] = {}

        for floor_attempt_idx, bridge_floor in enumerate(backoff_attempts):
            backoff_used_count = floor_attempt_idx + 1
            widened = bool(
                infeasible_handling
                and infeasible_handling.enabled
                and infeasible_handling.widen_search_on_backoff
                and floor_attempt_idx > 0
            )
            widened_search_used = widened_search_used or widened
            cfg_attempt = replace(cfg, bridge_floor=float(bridge_floor))

            segment_pool_max = int(cfg.segment_pool_max)
            beam_width = cfg.initial_beam_width
            max_expansion_attempts = cfg.max_expansion_attempts
            if widened:
                extra_pool = int(infeasible_handling.extra_neighbors_m) + int(
                    infeasible_handling.extra_bridge_helpers
                )
                segment_pool_max = min(
                    segment_pool_max + extra_pool, int(cfg.max_segment_pool_max)
                )
                beam_width = min(
                    beam_width + int(infeasible_handling.extra_beam_width),
                    cfg.max_beam_width,
                )
                max_expansion_attempts = max_expansion_attempts + int(
                    infeasible_handling.extra_expansion_attempts
                )

            expansions = 0
            pool_size_initial = 0
            pool_size_final = 0
            soft_genre_penalty_hits_segment = 0
            soft_genre_penalty_edges_scored_segment = 0
            last_failure_reason = None
            expansion_attempts_used = 0
            last_pool_diag: Dict[str, Any] = {}
            last_segment_candidates = []
            last_candidate_artist_keys = {}
            last_arc_stats: Dict[str, Any] = {}
            last_genre_cache_stats: Dict[str, int] = {}
            last_waypoint_stats = {}  # Reset for this backoff attempt
            segment_pool_cache: Optional[Dict[str, Any]] = (
                {} if bool(cfg.dj_pooling_cache_enabled) else None
            )
            last_segment_pool_max = int(segment_pool_max)
            last_beam_width = int(beam_width)

            for attempt in range(max_expansion_attempts):
                pool_diag: Dict[str, Any] = {}
                cand_artist_keys: Dict[int, str] = {}
                arc_stats_segment: Dict[str, Any] = {}
                genre_cache_stats_segment: Dict[str, int] = {}
                waypoint_stats_segment: Dict[str, Any] = {}
                pool_strategy = str(cfg.segment_pool_strategy).strip().lower()
                dj_pooling_strategy = str(cfg.dj_pooling_strategy or "baseline").strip().lower()
                if bool(cfg.dj_bridging_enabled) and dj_pooling_strategy == "dj_union":
                    pool_strategy = "dj_union"

                segment_internal_connectors = internal_connector_indices
                segment_connector_cap = int(internal_connector_max_per_segment)
                if bool(cfg.dj_bridging_enabled) and bool(cfg.dj_connector_bias_enabled) and segment_allow_detours:
                    adventurous = str(cfg.dj_route_shape or "linear").strip().lower() in {"arc", "ladder"}
                    dj_connector_cap = (
                        int(cfg.dj_connector_max_per_segment_adventurous)
                        if adventurous
                        else int(cfg.dj_connector_max_per_segment_linear)
                    )
                    dj_connector_cap = max(0, dj_connector_cap)
                    if dj_connector_cap > 0:
                        available = [int(i) for i in universe if int(i) not in global_used]
                        if allowed_set_indices is not None:
                            allowed = set(int(i) for i in allowed_set_indices)
                            available = [int(i) for i in available if int(i) in allowed]
                        if available:
                            dj_connectors = _select_connector_candidates(
                                available,
                                X_full_norm,
                                pier_a,
                                pier_b,
                                dj_connector_cap,
                            )
                        else:
                            dj_connectors = []
                        pool_diag["dj_connectors_selected"] = int(len(dj_connectors))
                        pool_diag["dj_connectors_injected_count"] = int(len(dj_connectors))
                        if dj_connectors:
                            try:
                                pool_diag["dj_connectors_preview"] = [
                                    str(bundle.track_ids[int(i)])
                                    for i in dj_connectors[:5]
                                ]
                            except Exception:
                                pool_diag["dj_connectors_preview"] = [
                                    str(int(i)) for i in dj_connectors[:5]
                                ]
                            if segment_pool_cache is not None:
                                segment_pool_cache["dj_connectors"] = set(
                                    int(i) for i in dj_connectors
                                )
                        if dj_connectors:
                            if segment_internal_connectors:
                                segment_internal_connectors = set(segment_internal_connectors) | set(dj_connectors)
                            else:
                                segment_internal_connectors = set(dj_connectors)
                            segment_connector_cap = max(segment_connector_cap, dj_connector_cap)

                if pool_strategy == "legacy":
                    neighbors_m = min(
                        int(cfg.initial_neighbors_m) * (2 ** int(attempt)),
                        int(cfg.max_neighbors_m),
                    )
                    bridge_helpers = min(
                        int(cfg.initial_bridge_helpers) * (2 ** int(attempt)),
                        int(cfg.max_bridge_helpers),
                    )
                    pool_diag["pool_strategy"] = "legacy"
                    pool_diag["neighbors_m"] = int(neighbors_m)
                    pool_diag["bridge_helpers"] = int(bridge_helpers)
                    segment_candidates = _build_segment_candidate_pool_legacy(
                        pier_a,
                        pier_b,
                        X_full_norm,
                        universe,
                        global_used,
                        int(neighbors_m),
                        int(bridge_helpers),
                        artist_keys=bundle.artist_keys,
                        bridge_floor=float(bridge_floor),
                        allowed_set=(allowed_set_indices if allowed_set_indices is not None else None),
                        internal_connectors=segment_internal_connectors,
                        internal_connector_cap=segment_connector_cap,
                        internal_connector_priority=internal_connector_priority,
                        diagnostics=pool_diag,
                    )
                    try:
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(
                            bundle, int(pier_a)
                        ).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(
                            bundle, int(pier_b)
                        ).artist_key
                    except Exception:
                        cand_artist_keys = {}
                else:
                    pool_diag["pool_strategy"] = pool_strategy
                    segment_candidates, cand_artist_keys, _cand_title_keys = _build_segment_candidate_pool_scored(
                        pier_a=pier_a,
                        pier_b=pier_b,
                        X_full_norm=X_full_norm,
                        universe_indices=universe,
                        used_track_ids=global_used,
                        bundle=bundle,
                        bridge_floor=float(bridge_floor),
                        segment_pool_max=int(segment_pool_max),
                        allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                        internal_connectors=segment_internal_connectors,
                        internal_connector_cap=segment_connector_cap,
                        internal_connector_priority=internal_connector_priority,
                        seed_artist_key=seed_artist_key,
                        disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                        disallow_seed_artist_in_interiors=bool(cfg.disallow_seed_artist_in_interiors),
                        used_track_keys=used_track_keys,
                        seed_track_keys=seed_track_keys,
                        diagnostics=pool_diag,
                        experiment_bridge_scoring_enabled=bool(
                            cfg.experiment_bridge_scoring_enabled
                        ),
                        experiment_bridge_min_weight=float(
                            cfg.experiment_bridge_min_weight
                        ),
                        experiment_bridge_balance_weight=float(
                            cfg.experiment_bridge_balance_weight
                        ),
                        pool_strategy=str(pool_strategy),
                        interior_length=int(interior_len),
                        progress_arc_enabled=bool(cfg.progress_arc_enabled),
                        progress_arc_shape=str(cfg.progress_arc_shape),
                        X_genre_norm=X_genre_norm,
                        X_genre_norm_idf=X_genre_norm_idf,
                        genre_targets=segment_g_targets,
                        pool_k_local=int(cfg.dj_pooling_k_local),
                        pool_k_toward=int(cfg.dj_pooling_k_toward),
                        pool_k_genre=int(cfg.dj_pooling_k_genre),
                        pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                        pool_step_stride=int(cfg.dj_pooling_step_stride),
                        pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                        pooling_cache=segment_pool_cache,
                        pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                    )
                    try:
                        cand_artist_keys = dict(cand_artist_keys)
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(
                            bundle, int(pier_a)
                        ).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(
                            bundle, int(pier_b)
                        ).artist_key
                    except Exception:
                        cand_artist_keys = {}
                    if (
                        len(segment_candidates) < interior_len
                        and bool(cfg.disallow_seed_artist_in_interiors)
                        and seed_artist_key
                    ):
                        relaxed_candidates, relaxed_artist_keys, _relaxed_title_keys = _build_segment_candidate_pool_scored(
                            pier_a=pier_a,
                            pier_b=pier_b,
                            X_full_norm=X_full_norm,
                            universe_indices=universe,
                            used_track_ids=global_used,
                            bundle=bundle,
                            bridge_floor=float(bridge_floor),
                            segment_pool_max=int(segment_pool_max),
                            allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                            internal_connectors=segment_internal_connectors,
                            internal_connector_cap=segment_connector_cap,
                            internal_connector_priority=internal_connector_priority,
                            seed_artist_key=seed_artist_key,
                            disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                            disallow_seed_artist_in_interiors=False,
                            used_track_keys=used_track_keys,
                            seed_track_keys=seed_track_keys,
                            diagnostics=None,
                            experiment_bridge_scoring_enabled=bool(
                                cfg.experiment_bridge_scoring_enabled
                            ),
                            experiment_bridge_min_weight=float(
                                cfg.experiment_bridge_min_weight
                            ),
                            experiment_bridge_balance_weight=float(
                                cfg.experiment_bridge_balance_weight
                            ),
                            pool_strategy=str(pool_strategy),
                            interior_length=int(interior_len),
                            progress_arc_enabled=bool(cfg.progress_arc_enabled),
                            progress_arc_shape=str(cfg.progress_arc_shape),
                            X_genre_norm=X_genre_norm,
                            X_genre_norm_idf=X_genre_norm_idf,
                            genre_targets=segment_g_targets,
                            pool_k_local=int(cfg.dj_pooling_k_local),
                            pool_k_toward=int(cfg.dj_pooling_k_toward),
                            pool_k_genre=int(cfg.dj_pooling_k_genre),
                            pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                            pool_step_stride=int(cfg.dj_pooling_step_stride),
                            pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                            pooling_cache=segment_pool_cache,
                            pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                        )
                        if len(relaxed_candidates) > len(segment_candidates):
                            segment_candidates = relaxed_candidates
                            cand_artist_keys = dict(relaxed_artist_keys)
                            pool_diag["relaxed_seed_artist_in_interiors"] = True
                            pool_diag["relaxed_seed_artist_pool_size"] = int(
                                len(relaxed_candidates)
                            )
                            warnings.append(
                                {
                                    "type": "relax_seed_artist_in_interiors",
                                    "scope": "segment",
                                    "segment_index": int(seg_idx),
                                    "message": (
                                        "Relaxed seed-artist exclusion in bridge interiors "
                                        "due to insufficient candidates."
                                    ),
                                }
                            )
                    if (
                        bool(cfg.dj_bridging_enabled)
                        and str(cfg.dj_pooling_strategy or "baseline")
                        .strip()
                        .lower()
                        == "dj_union"
                        and bool(cfg.dj_pooling_debug_compare_baseline)
                    ):
                        baseline_candidates, _, _ = _build_segment_candidate_pool_scored(
                            pier_a=pier_a,
                            pier_b=pier_b,
                            X_full_norm=X_full_norm,
                            universe_indices=universe,
                            used_track_ids=global_used,
                            bundle=bundle,
                            bridge_floor=float(bridge_floor),
                            segment_pool_max=int(segment_pool_max),
                            allowed_set=allowed_set_indices if allowed_set_indices is not None else None,
                            internal_connectors=segment_internal_connectors,
                            internal_connector_cap=segment_connector_cap,
                            internal_connector_priority=internal_connector_priority,
                            seed_artist_key=seed_artist_key,
                            disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                            disallow_seed_artist_in_interiors=bool(cfg.disallow_seed_artist_in_interiors),
                            used_track_keys=used_track_keys,
                            seed_track_keys=seed_track_keys,
                            diagnostics=None,
                            experiment_bridge_scoring_enabled=bool(
                                cfg.experiment_bridge_scoring_enabled
                            ),
                            experiment_bridge_min_weight=float(
                                cfg.experiment_bridge_min_weight
                            ),
                            experiment_bridge_balance_weight=float(
                                cfg.experiment_bridge_balance_weight
                            ),
                            pool_strategy=str(pool_strategy),
                            interior_length=int(interior_len),
                            progress_arc_enabled=bool(cfg.progress_arc_enabled),
                            progress_arc_shape=str(cfg.progress_arc_shape),
                            X_genre_norm=X_genre_norm,
                            X_genre_norm_idf=X_genre_norm_idf,
                            genre_targets=segment_g_targets,
                            pool_k_local=int(cfg.dj_pooling_k_local),
                            pool_k_toward=int(cfg.dj_pooling_k_toward),
                            pool_k_genre=int(cfg.dj_pooling_k_genre),
                            pool_k_union_max=int(cfg.dj_pooling_k_union_max),
                            pool_step_stride=int(cfg.dj_pooling_step_stride),
                            pool_cache_enabled=bool(cfg.dj_pooling_cache_enabled),
                            pooling_cache=segment_pool_cache,
                            pool_verbose=bool(cfg.dj_diagnostics_pool_verbose),  # Phase 3 fix
                        )
                        if segment_pool_cache is not None:
                            segment_pool_cache["dj_baseline_pool"] = set(
                                int(i) for i in baseline_candidates
                            )

                if segment_candidates:
                    last_segment_candidates = list(segment_candidates)
                    last_candidate_artist_keys = dict(cand_artist_keys or {})
                last_pool_diag.update(pool_diag)

                # TASK A: Track pool_before_gating (after merge, before gates)
                pool_size_initial = len(segment_candidates) if segment_candidates else 0

                if not segment_candidates or len(segment_candidates) < int(interior_len):
                    last_failure_reason = f"pool_after_gate {len(segment_candidates)} < interior_len {interior_len}"
                    pool_size_final = 0
                else:
                    segment_path, soft_genre_penalty_hits_segment, soft_genre_penalty_edges_scored_segment, beam_failure_reason = _beam_search_segment(
                        pier_a,
                        pier_b,
                        interior_len,
                        segment_candidates,
                        X_full_tr_norm,
                        X_full_norm,
                        X_start_tr_norm,
                        X_mid_tr_norm,
                        X_end_tr_norm,
                        X_genre_norm,
                        cfg_attempt,
                        beam_width,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        genre_idf=genre_idf,
                        genre_vocab=genre_vocab,
                        artist_key_by_idx=(cand_artist_keys if cand_artist_keys else None),
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        arc_stats=arc_stats_segment,
                        genre_cache_stats=genre_cache_stats_segment,
                        g_targets_override=segment_g_targets,
                        waypoint_stats=waypoint_stats_segment,
                    )
                    last_failure_reason = beam_failure_reason
                    if segment_path is not None:
                        # Capture waypoint stats for successful path
                        last_waypoint_stats = dict(waypoint_stats_segment)

                        # TASK A: Extract pool_after_gating count from waypoint_stats
                        pool_size_final = int(waypoint_stats_segment.get("pool_after_gating_count", 0))

                        baseline_pool = None
                        if (
                            segment_pool_cache is not None
                            and "dj_baseline_pool" in segment_pool_cache
                        ):
                            baseline_pool = segment_pool_cache.get(
                                "dj_baseline_pool"
                            )
                        elif pool_strategy != "dj_union":
                            baseline_pool = set(int(i) for i in segment_candidates)
                        sources = None
                        if segment_pool_cache is not None:
                            sources = segment_pool_cache.get("dj_pool_sources")
                        last_pool_diag.update(
                            _compute_chosen_source_counts(
                                segment_path,
                                sources=sources,
                                baseline_pool=baseline_pool,
                            )
                        )
                        if segment_pool_cache is not None and "dj_connectors" in segment_pool_cache:
                            connector_set = segment_pool_cache.get("dj_connectors", set())
                            chosen_connectors = [
                                int(i) for i in segment_path if int(i) in connector_set
                            ]
                            last_pool_diag["dj_connectors_chosen_count"] = int(
                                len(chosen_connectors)
                            )
                            if chosen_connectors:
                                try:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(bundle.track_ids[int(i)])
                                        for i in chosen_connectors[:5]
                                    ]
                                except Exception:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(int(i)) for i in chosen_connectors[:5]
                                    ]

                if segment_path is not None:
                    break

                expansions += 1
                if str(cfg.segment_pool_strategy).strip().lower() != "legacy":
                    segment_pool_max = min(int(segment_pool_max) * 2, int(cfg.max_segment_pool_max))
                beam_width = min(int(beam_width) * 2, int(cfg.max_beam_width))

                if infeasible_handling and infeasible_handling.enabled:
                    if str(cfg.segment_pool_strategy).strip().lower() == "legacy":
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d strategy=legacy beam=%d)",
                            seg_idx,
                            attempt + 1,
                            beam_width,
                        )
                    else:
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d segment_pool_max=%d beam=%d)",
                            seg_idx,
                            attempt + 1,
                            int(segment_pool_max),
                            beam_width,
                        )

            if infeasible_handling and infeasible_handling.enabled:
                logger.info(
                    "Segment %d attempt %d: bridge_floor=%.2f widened=%s pool_after_gate=%d",
                    seg_idx,
                    floor_attempt_idx + 1,
                    float(bridge_floor),
                    widened,
                    int(len(last_segment_candidates)),
                )

            if audit_enabled:
                top_rows, dists = _summarize_candidates_for_audit(
                    candidates=last_segment_candidates,
                    pier_a=pier_a,
                    pier_b=pier_b,
                    X_full_norm=X_full_norm,
                    X_full_tr_norm=X_full_tr_norm,
                    X_start_tr_norm=X_start_tr_norm,
                    X_mid_tr_norm=X_mid_tr_norm,
                    X_end_tr_norm=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_attempt,
                    bundle=bundle,
                    internal_connector_indices=internal_connector_indices,
                    top_k=top_k,
                )
                audit_events.append(
                    RunAuditEvent(
                        kind="segment_attempt",
                        ts_utc=now_utc_iso(),
                        payload={
                            "segment_index": int(seg_idx),
                            "segment_header": f"{pier_a_id} -> {pier_b_id} (interior={interior_len})",
                            "attempt_number": int(floor_attempt_idx + 1),
                            "expansion_attempts": int(expansion_attempts_used),
                            "bridge_floor": float(bridge_floor),
                            "widened": bool(widened),
                            "segment_pool_strategy": str(
                                last_pool_diag.get("pool_strategy", cfg.segment_pool_strategy)
                            ),
                            "segment_pool_max": int(last_segment_pool_max),
                            "beam_width": int(last_beam_width),
                            "pool_counts": dict(last_pool_diag),
                            "pool_size_initial": int(pool_size_initial),
                            "pool_size_final": int(pool_size_final),
                            "distributions": dists,
                            "soft_genre_penalty": {
                                "edges_scored": int(soft_genre_penalty_edges_scored_segment),
                                "hits": int(soft_genre_penalty_hits_segment),
                                "threshold": float(cfg_attempt.genre_penalty_threshold),
                                "strength": float(cfg_attempt.genre_penalty_strength),
                            },
                            "genre_cache": dict(last_genre_cache_stats),
                            "progress_arc": dict(last_arc_stats),
                        },
                    )
                )

            if segment_path is not None:
                chosen_bridge_floor = float(bridge_floor)
                beam_width_used = int(last_beam_width)
                break

            last_segment_pool_cache = segment_pool_cache

        return {
            "segment_path": segment_path,
            "chosen_bridge_floor": float(chosen_bridge_floor),
            "backoff_attempts": [float(x) for x in backoff_attempts],
            "backoff_used_count": int(backoff_used_count),
            "widened_search_used": bool(widened_search_used),
            "expansions": int(expansions),
            "pool_size_initial": int(pool_size_initial),
            "pool_size_final": int(pool_size_final),
            "beam_width_used": int(beam_width_used),
            "soft_genre_penalty_hits_segment": int(soft_genre_penalty_hits_segment),
            "soft_genre_penalty_edges_scored_segment": int(soft_genre_penalty_edges_scored_segment),
            "last_failure_reason": (str(last_failure_reason) if last_failure_reason else None),
            "last_segment_candidates": list(last_segment_candidates),
            "last_candidate_artist_keys": dict(last_candidate_artist_keys),
            "segment_pool_cache": dict(last_segment_pool_cache or {}),
            "last_waypoint_stats": dict(last_waypoint_stats),
            "last_pool_diag": dict(last_pool_diag),
        }

    for seg_idx in range(num_segments):
        pier_a = ordered_seeds[seg_idx]
        pier_b = ordered_seeds[seg_idx + 1]
        interior_len = segment_lengths[seg_idx]

        pier_a_id = str(bundle.track_ids[pier_a])
        pier_b_id = str(bundle.track_ids[pier_b])

        logger.info("Building segment %d: %s -> %s (interior=%d)",
                   seg_idx, pier_a_id, pier_b_id, interior_len)

        segment_g_targets: Optional[list[np.ndarray]] = None
        segment_ladder_diag: dict[str, Any] = {}
        segment_far_stats: Optional[dict[str, Optional[float]]] = None
        segment_is_far = False
        if bool(cfg.dj_bridging_enabled) and X_genre_norm is not None:
            # Phase 3 fix: genre_vocab is optional for vector mode, always try to build targets
            genre_vocab = getattr(bundle, "genre_vocab", None)
            segment_g_targets = _build_genre_targets(
                pier_a=pier_a,
                pier_b=pier_b,
                interior_length=interior_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=genre_vocab,  # Can be None for vector mode
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                ladder_diag=segment_ladder_diag,
                X_genre_raw=X_genre_raw,
                X_genre_smoothed=X_genre_smoothed,
                genre_idf=genre_idf,
            )
            segment_far_stats = _segment_far_stats(
                pier_a=pier_a,
                pier_b=pier_b,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                universe=universe,
                used_track_ids=global_used,
                bridge_floor=float(cfg.bridge_floor),
            )
            if segment_far_stats:
                sonic_sim = segment_far_stats.get("sonic_sim")
                genre_sim = segment_far_stats.get("genre_sim")
                scarcity = segment_far_stats.get("connector_scarcity")
                if sonic_sim is not None and (1.0 - float(sonic_sim)) > float(cfg.dj_far_threshold_sonic):
                    segment_is_far = True
                if genre_sim is not None and (1.0 - float(genre_sim)) > float(cfg.dj_far_threshold_genre):
                    segment_is_far = True
                if scarcity is not None and float(scarcity) < float(cfg.dj_far_threshold_connector_scarcity):
                    segment_is_far = True
        segment_allow_detours = bool(cfg.dj_allow_detours_when_far) and segment_is_far

        relaxation_enabled = (
            bool(cfg.dj_bridging_enabled)
            and bool(cfg.dj_relaxation_enabled)
            and str(cfg.dj_route_shape or "linear").strip().lower() == "ladder"
        )
        relaxation_attempts = (
            _build_dj_relaxation_attempts(cfg)
            if relaxation_enabled
            else [{"label": "baseline", "cfg": cfg, "changes": [], "force_allow_detours": False}]
        )
        segment_relaxation_attempts: list[dict[str, Any]] = []
        relaxation_success_attempt: Optional[int] = None
        cfg_base = cfg
        cfg_used_for_segment = cfg
        segment_allow_detours_base = segment_allow_detours
        segment_path: Optional[List[int]] = None
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_failure_reason: Optional[str] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts: list[float] = [float(cfg.bridge_floor)]
        backoff_used_count = 0
        widened_search_used = False
        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0

        for relax_idx, relax in enumerate(relaxation_attempts):
            cfg = relax["cfg"]
            cfg_used_for_segment = cfg
            attempt_allow_detours = segment_allow_detours_base or bool(relax.get("force_allow_detours"))
            attempt_result = _run_segment_backoff_attempts(
                cfg_attempt_base=cfg,
                segment_allow_detours=attempt_allow_detours,
                segment_g_targets=segment_g_targets,
                pier_a=pier_a,
                pier_b=pier_b,
                interior_len=interior_len,
                pier_a_id=pier_a_id,
                pier_b_id=pier_b_id,
                seg_idx=seg_idx,
                recent_boundary_artists=recent_boundary_artists if seg_idx > 0 else None,
            )
            segment_path = attempt_result["segment_path"]
            chosen_bridge_floor = float(attempt_result["chosen_bridge_floor"])
            backoff_used_count = int(attempt_result["backoff_used_count"])
            backoff_attempts = list(attempt_result.get("backoff_attempts") or [])
            widened_search_used = bool(attempt_result["widened_search_used"])
            expansions = int(attempt_result["expansions"])
            pool_size_initial = int(attempt_result["pool_size_initial"])
            pool_size_final = int(attempt_result["pool_size_final"])
            beam_width_used = int(attempt_result["beam_width_used"])
            soft_genre_penalty_hits_segment = int(attempt_result["soft_genre_penalty_hits_segment"])
            soft_genre_penalty_edges_scored_segment = int(attempt_result["soft_genre_penalty_edges_scored_segment"])
            last_failure_reason = attempt_result["last_failure_reason"]
            last_segment_candidates = list(attempt_result["last_segment_candidates"])
            last_candidate_artist_keys = dict(attempt_result["last_candidate_artist_keys"])
            pool_cache = attempt_result.get("segment_pool_cache")
            last_segment_pool_cache = dict(pool_cache) if pool_cache is not None else None
            last_waypoint_stats = dict(attempt_result.get("last_waypoint_stats", {}))
            last_pool_diag = dict(attempt_result.get("last_pool_diag", {}))

            segment_relaxation_attempts.append({
                "attempt_index": int(relax_idx),
                "label": str(relax.get("label", "")),
                "changes": list(relax.get("changes") or []),
                "failure_reason": (str(last_failure_reason) if segment_path is None else None),
            })
            if segment_path is not None:
                relaxation_success_attempt = int(relax_idx)
                break

        cfg = cfg_base
        segment_allow_detours = segment_allow_detours_base

        micro_pier_diag: dict[str, Any] = {}
        if relaxation_enabled and bool(cfg.dj_relaxation_emit_warnings):
            warnings.append({
                "type": "dj_relaxation_attempts",
                "scope": "segment",
                "segment_index": int(seg_idx),
                "message": "DJ relaxation attempts executed.",
                "attempts": list(segment_relaxation_attempts),
                "success_attempt": relaxation_success_attempt,
            })

        if segment_path is None:
            if _should_attempt_micro_pier(relaxation_enabled=relaxation_enabled, segment_path=segment_path):
                if bool(cfg.dj_micro_piers_enabled):
                    candidates = _micro_pier_candidate_pool(
                        cfg.dj_micro_piers_candidate_source,
                        last_segment_candidates,
                        last_segment_pool_cache,
                    )
                    micro_path = _attempt_micro_pier_split(
                        pier_a=pier_a,
                        pier_b=pier_b,
                        interior_length=interior_len,
                        candidates=candidates,
                        X_full=X_full_tr_norm,
                        X_full_norm=X_full_norm,
                        X_start=X_start_tr_norm,
                        X_mid=X_mid_tr_norm,
                        X_end=X_end_tr_norm,
                        X_genre_norm=X_genre_norm,
                        cfg=cfg_used_for_segment,
                        beam_width=int(beam_width_used),
                        artist_key_by_idx=last_candidate_artist_keys,
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        warnings=warnings,
                        X_genre_vocab=getattr(bundle, "genre_vocab", None),
                        genre_graph=genre_graph,
                        micro_diag=micro_pier_diag,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        genre_idf=genre_idf,
                    )
                    if micro_path is not None and len(micro_path) == interior_len:
                        segment_path = micro_path
                        warnings.append({
                            "type": "micro_pier_fallback",
                            "scope": "segment",
                            "segment_index": int(seg_idx),
                            "message": "Inserted micro-pier due to infeasible bridge; consider lowering genre drift, increasing effort, or adding genre metadata.",
                        })
            elif bool(cfg.dj_bridging_enabled) and bool(cfg.dj_micro_piers_enabled) and segment_allow_detours:
                candidates = _micro_pier_candidate_pool(
                    cfg.dj_micro_piers_candidate_source,
                    last_segment_candidates,
                    last_segment_pool_cache,
                )
                micro_path = _attempt_micro_pier_split(
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_length=interior_len,
                    candidates=candidates,
                    X_full=X_full_tr_norm,
                    X_full_norm=X_full_norm,
                    X_start=X_start_tr_norm,
                    X_mid=X_mid_tr_norm,
                    X_end=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_used_for_segment,
                    beam_width=int(beam_width_used),
                    artist_key_by_idx=last_candidate_artist_keys,
                    seed_artist_key=seed_artist_key,
                    recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
                    durations_ms=bundle.durations_ms,
                    artist_identity_cfg=artist_identity_cfg,
                    bundle=bundle,
                    warnings=warnings,
                    X_genre_vocab=getattr(bundle, "genre_vocab", None),
                    genre_graph=genre_graph,
                    micro_diag=micro_pier_diag,
                    X_genre_norm_idf=X_genre_norm_idf,
                    X_genre_raw=X_genre_raw,
                    X_genre_smoothed=X_genre_smoothed,
                    genre_idf=genre_idf,
                )
                if micro_path is not None and len(micro_path) == interior_len:
                    segment_path = micro_path

            if audit_enabled:
                audit_events.append(
                    RunAuditEvent(
                        kind="segment_failure",
                        ts_utc=now_utc_iso(),
                        payload={
                            "segment_index": int(seg_idx),
                            "failure_reason": str(last_failure_reason or "segment infeasible"),
                            "attempted_bridge_floors": [float(x) for x in backoff_attempts],
                        },
                    )
                )
            if infeasible_handling and infeasible_handling.enabled:
                failure = f"Segment {seg_idx} infeasible under bridge_floor backoff (attempted={backoff_attempts}; last_reason={last_failure_reason})"
            else:
                failure = f"Segment {seg_idx} infeasible under bridge_floor={cfg.bridge_floor}"
            logger.error(failure)
            return PierBridgeResult(
                track_ids=[],
                track_indices=[],
                seed_positions=[],
                segment_diagnostics=[],
                stats={},
                success=False,
                failure_reason=failure,
            )

        soft_genre_penalty_hits_total += int(soft_genre_penalty_hits_segment)
        soft_genre_penalty_edges_scored_total += int(soft_genre_penalty_edges_scored_segment)
        if cfg.genre_penalty_strength > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Segment %d: soft_genre_penalty_hits=%d edges_scored=%d threshold=%.2f strength=%.2f",
                seg_idx,
                int(soft_genre_penalty_hits_segment),
                int(soft_genre_penalty_edges_scored_segment),
                float(cfg.genre_penalty_threshold),
                float(cfg.genre_penalty_strength),
            )

        # Compute edge scores for diagnostics
        full_segment = [pier_a] + segment_path + [pier_b]
        worst_edge, mean_edge = _compute_edge_scores(
            full_segment, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm, X_end_tr_norm, cfg
        )
        micro_pier_used = "micro_pier_index" in micro_pier_diag
        micro_pier_track_id = None
        if micro_pier_used:
            try:
                micro_pier_track_id = str(
                    bundle.track_ids[int(micro_pier_diag["micro_pier_index"])]
                )
            except Exception:
                micro_pier_track_id = str(micro_pier_diag.get("micro_pier_index"))

        # Record diagnostics
        diagnostics.append(SegmentDiagnostics(
            pier_a_id=pier_a_id,
            pier_b_id=pier_b_id,
            target_length=interior_len,
            actual_length=len(segment_path),
            pool_size_initial=pool_size_initial,
            pool_size_final=pool_size_final,
            expansions=expansions,
            beam_width_used=beam_width_used,
            worst_edge_score=worst_edge,
            mean_edge_score=mean_edge,
            success=segment_path is not None and len(segment_path) == interior_len,
            bridge_floor_used=float(chosen_bridge_floor),
            backoff_attempts_used=int(backoff_used_count),
            widened_search=bool(widened_search_used),
            route_shape=str(segment_ladder_diag.get("route_shape", str(cfg.dj_route_shape or "linear"))),
            ladder_waypoint_labels=list(segment_ladder_diag.get("ladder_waypoint_labels", [])),
            ladder_waypoint_count=int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
            ladder_waypoint_vector_mode=str(segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")),
            ladder_waypoint_vector_stats=list(segment_ladder_diag.get("ladder_waypoint_vector_stats", [])),
            relaxation_attempts=list(segment_relaxation_attempts),
            relaxation_success_attempt=relaxation_success_attempt,
            micro_pier_used=bool(micro_pier_used),
            micro_pier_track_id=micro_pier_track_id,
            micro_pier_metric_value=(
                float(micro_pier_diag.get("micro_pier_metric_value"))
                if micro_pier_diag.get("micro_pier_metric_value") is not None
                else None
            ),
            micro_pier_left_success=micro_pier_diag.get("left_success"),
            micro_pier_right_success=micro_pier_diag.get("right_success"),
            waypoint_enabled=bool(last_waypoint_stats.get("waypoint_enabled", False)),
            mean_waypoint_sim=last_waypoint_stats.get("mean_waypoint_sim"),
            p50_waypoint_sim=last_waypoint_stats.get("p50_waypoint_sim"),
            p90_waypoint_sim=last_waypoint_stats.get("p90_waypoint_sim"),
            min_waypoint_sim=last_waypoint_stats.get("min_waypoint_sim"),
            max_waypoint_sim=last_waypoint_stats.get("max_waypoint_sim"),
            waypoint_delta_applied_count=int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
            mean_waypoint_delta=last_waypoint_stats.get("mean_waypoint_delta"),
        ))
        segment_bridge_floors_used.append(float(chosen_bridge_floor))
        segment_backoff_attempts_used.append(int(backoff_used_count))
        logger.info(
            "Segment %d: %s -> %s bridge_floor=%.2f pool_before=%d pool_after=%d",
            seg_idx, pier_a_id, pier_b_id, float(chosen_bridge_floor), pool_size_initial, pool_size_final,
        )

        # Log waypoint influence stats (Phase 2 diagnostics)
        if last_waypoint_stats.get("waypoint_enabled"):
            logger.info(
                "  Waypoint stats: enabled=True mean_sim=%.3f p50=%.3f p90=%.3f delta_applied=%d/%d mean_delta=%.4f",
                float(last_waypoint_stats.get("mean_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p50_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p90_waypoint_sim", 0.0)),
                int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
                interior_len,
                float(last_waypoint_stats.get("mean_waypoint_delta", 0.0)),
            )

            # Log rank impact metrics (TASK B: opt-in diagnostic)
            rank_impact_results = last_waypoint_stats.get("rank_impact_results", [])
            if rank_impact_results:
                sampled_steps_count = len(rank_impact_results)
                winner_changed_count = sum(1 for r in rank_impact_results if r.get("winner_changed"))
                mean_reordered = float(np.mean([r.get("topK_reordered_count", 0) for r in rank_impact_results]))
                mean_topK = float(np.mean([r.get("topK", 10) for r in rank_impact_results]))
                mean_rank_delta = float(np.mean([r.get("mean_abs_rank_delta", 0.0) for r in rank_impact_results]))

                logger.info(
                    "  Waypoint rank impact: sampled_steps=%d winner_changed=%d/%d topK_reordered=%.1f/%.0f mean_rank_delta=%.1f",
                    sampled_steps_count,
                    winner_changed_count,
                    sampled_steps_count,
                    mean_reordered,
                    mean_topK,
                    mean_rank_delta,
                )

                # Phase 2: Coverage bonus impact (compare base+waypoint vs full)
                if bool(cfg.dj_genre_use_coverage):
                    coverage_winner_changed_count = 0
                    coverage_mean_bonus = []
                    for r in rank_impact_results:
                        top10_table = r.get("top10_table", [])
                        if top10_table:
                            # Find winner by base+waypoint score (before coverage)
                            base_waypoint_scores = [(entry["cand_idx"], entry["base_score"] + entry["waypoint_delta"])
                                                     for entry in top10_table]
                            base_waypoint_winner = max(base_waypoint_scores, key=lambda x: x[1])[0]
                            # Find winner by full score (after coverage)
                            full_winner = top10_table[0]["cand_idx"]  # Already sorted by base_rank
                            # Actually need to re-sort by full_score to get true full_winner
                            full_scores = [(entry["cand_idx"], entry["full_score"]) for entry in top10_table]
                            full_winner = max(full_scores, key=lambda x: x[1])[0]

                            if base_waypoint_winner != full_winner:
                                coverage_winner_changed_count += 1

                            # Collect mean coverage bonus for this step
                            coverage_bonuses = [entry["coverage_bonus"] for entry in top10_table]
                            coverage_mean_bonus.append(float(np.mean(coverage_bonuses)))

                    if coverage_mean_bonus:
                        logger.info(
                            "  Coverage bonus impact: winner_changed=%d/%d mean_bonus=%.4f",
                            coverage_winner_changed_count,
                            sampled_steps_count,
                            float(np.mean(coverage_mean_bonus)),
                        )

        # Log chosen edge provenance (dj_union) - TASK A: renamed from "Pool sources"
        if last_pool_diag:
            if "chosen_from_local_count" in last_pool_diag or "dj_pool_strategy" in last_pool_diag:
                # Legacy exclusive counts (priority-based)
                logger.info(
                    "  Chosen edge provenance (exclusive): strategy=%s local=%d toward=%d genre=%d baseline_only=%d",
                    str(last_pool_diag.get("dj_pool_strategy", last_pool_diag.get("pool_strategy", "unknown"))),
                    int(last_pool_diag.get("chosen_from_local_count", 0)),
                    int(last_pool_diag.get("chosen_from_toward_count", 0)),
                    int(last_pool_diag.get("chosen_from_genre_count", 0)),
                    int(last_pool_diag.get("chosen_from_baseline_only_count", 0)),
                )
                # Phase 3: Membership-based counts (all overlaps)
                if "local_only" in last_pool_diag:
                    logger.info(
                        "  Provenance memberships (Phase3): local_only=%d toward_only=%d genre_only=%d " +
                        "local+toward=%d local+genre=%d toward+genre=%d local+toward+genre=%d baseline_only=%d",
                        int(last_pool_diag.get("local_only", 0)),
                        int(last_pool_diag.get("toward_only", 0)),
                        int(last_pool_diag.get("genre_only", 0)),
                        int(last_pool_diag.get("local+toward", 0)),
                        int(last_pool_diag.get("local+genre", 0)),
                        int(last_pool_diag.get("toward+genre", 0)),
                        int(last_pool_diag.get("local+toward+genre", 0)),
                        int(last_pool_diag.get("baseline_only", 0)),
                    )

        # TASK A: Invariant checks (log WARNINGs for inconsistencies)
        if pool_size_final > 0 and pool_size_initial == 0:
            logger.warning(
                "  WARNING: pool_before_gating=0 but pool_after_gating=%d (possible missing instrumentation)",
                pool_size_final
            )

        if last_pool_diag and "chosen_from_local_count" in last_pool_diag:
            chosen_sum = (
                int(last_pool_diag.get("chosen_from_local_count", 0)) +
                int(last_pool_diag.get("chosen_from_toward_count", 0)) +
                int(last_pool_diag.get("chosen_from_genre_count", 0)) +
                int(last_pool_diag.get("chosen_from_baseline_only_count", 0))
            )
            if chosen_sum != interior_len:
                logger.warning(
                    "  WARNING: chosen_from_* sum (%d) != interior_length (%d) (possible provenance tracking gap)",
                    chosen_sum,
                    interior_len
                )

        # Log ladder waypoint labels (route planning)
        if segment_ladder_diag and segment_ladder_diag.get("ladder_waypoint_count", 0) > 0:
            labels = segment_ladder_diag.get("ladder_waypoint_labels", [])
            mode = segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")
            logger.info(
                "  Ladder route: shape=%s mode=%s waypoints=%d labels=%s",
                str(segment_ladder_diag.get("route_shape", "linear")),
                mode,
                int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
                ", ".join(labels[:6]) if labels else "none",
            )
        # DEBUG top candidates for this segment
        if logger.isEnabledFor(logging.DEBUG):
            scores_dbg = []
            sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
            sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])
            for cand in last_segment_candidates[: min(200, len(last_segment_candidates))]:
                sim_a = float(sim_to_a[cand])
                sim_b = float(sim_to_b[cand])
                denom = sim_a + sim_b
                hmean = 0.0 if denom <= 1e-9 else (2 * sim_a * sim_b) / denom
                trans = _compute_transition_score(
                    cand,
                    pier_b,
                    X_full_tr_norm,
                    X_start_tr_norm,
                    X_mid_tr_norm,
                    X_end_tr_norm,
                    cfg,
                )
                final_score = cfg.weight_bridge * hmean + cfg.weight_transition * trans
                scores_dbg.append((final_score, sim_a, sim_b, hmean, trans, cand))
            scores_dbg = sorted(scores_dbg, key=lambda t: t[0], reverse=True)[:10]
            dbg_rows = []
            for final_score, sim_a, sim_b, hmean, trans, cand in scores_dbg:
                keys = identity_keys_for_index(bundle, int(cand))
                artist = (
                    str(bundle.track_artists[cand])
                    if bundle.track_artists is not None
                    else (str(bundle.artist_keys[cand]) if bundle.artist_keys is not None else "")
                )
                title = str(bundle.track_titles[cand]) if bundle.track_titles is not None else ""
                dbg_rows.append({
                    "track_id": str(bundle.track_ids[cand]),
                    "artist": sanitize_for_logging(artist),
                    "title": sanitize_for_logging(title),
                    "artist_key": keys.artist_key,
                    "title_key": keys.title_key,
                    "simA": round(sim_a, 3),
                    "simB": round(sim_b, 3),
                    "hmean": round(hmean, 3),
                    "transition": round(trans, 3),
                    "final": round(final_score, 3),
                    "internal": bool(internal_connector_indices and cand in internal_connector_indices),
                })
            logger.debug("Segment %d top candidates: %s", seg_idx, dbg_rows)

        # Commit segment path to used set
        for idx in segment_path:
            global_used.add(idx)
            try:
                used_track_keys.add(identity_keys_for_index(bundle, int(idx)).track_key)
            except Exception:
                continue

        all_segments.append(full_segment)

        # Update boundary context for next segment (cross-segment min_gap enforcement)
        # Build the concatenated result so far to extract recent artists
        current_concat: List[int] = []
        for concat_seg_idx, concat_seg in enumerate(all_segments):
            if concat_seg_idx == 0:
                current_concat.extend(concat_seg)
            else:
                current_concat.extend(concat_seg[1:])  # Drop duplicate pier

        # Extract artist keys from the last MIN_GAP_GLOBAL positions
        # If artist_identity_cfg is enabled, resolve to identity keys (collapsing ensemble variants)
        recent_boundary_artists = []
        start_pos = max(0, len(current_concat) - MIN_GAP_GLOBAL)
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

        for pos in range(start_pos, len(current_concat)):
            try:
                if use_identity:
                    # Identity mode: resolve artist string to identity keys
                    artist_str = identity_keys_for_index(bundle, int(current_concat[pos])).artist
                    if artist_str:
                        identity_keys_set = resolve_artist_identity_keys(artist_str, artist_identity_cfg)
                        # Add ALL identity keys to boundary tracking
                        for identity_key in identity_keys_set:
                            recent_boundary_artists.append(identity_key)
                else:
                    # Legacy mode: single artist_key
                    artist_key = identity_keys_for_index(bundle, int(current_concat[pos])).artist_key
                    if artist_key:
                        recent_boundary_artists.append(str(artist_key))
            except Exception:
                continue

    # Concatenate segments
    # First segment: keep full [A, ..., B]
    # Subsequent segments: drop first element (the pier) to avoid duplication
    # Single-seed arc: drop last element (the duplicated seed) to avoid repetition
    final_indices: List[int] = []
    seed_positions: List[int] = []

    if is_single_seed_arc:
        # Single-seed arc: segment is [seed, interior..., seed]
        # Output only [seed, interior...] to avoid duplicate seed at end
        segment = all_segments[0] if all_segments else [ordered_seeds[0]]
        final_indices = segment[:-1]  # Drop the trailing duplicate seed
        seed_positions = [0]  # Seed is at position 0
        logger.info("Pier+Bridge: single-seed arc output: %d tracks (seed at start, arc returns to seed-similar)", len(final_indices))
    else:
        for seg_idx, segment in enumerate(all_segments):
            if seg_idx == 0:
                final_indices.extend(segment)
                seed_positions.append(0)  # First pier
                seed_positions.append(len(final_indices) - 1)  # Second pier
            else:
                # Drop first element (the pier, already included)
                final_indices.extend(segment[1:])
                seed_positions.append(len(final_indices) - 1)  # New pier

    # Convert to track IDs
    # Cross-segment min_gap is enforced DURING generation (boundary-aware beam search),
    # not as a post-order filter. This ensures exact length guarantees.
    final_track_ids = [str(bundle.track_ids[i]) for i in final_indices]

    # Strict length validation: pier-bridge must return EXACTLY the requested number of tracks
    if len(final_track_ids) != total_tracks:
        failure_msg = (
            f"Pier-bridge length mismatch: generated {len(final_track_ids)} tracks "
            f"but expected exactly {total_tracks}. This indicates a bug in segment generation."
        )
        logger.error(failure_msg)
        return PierBridgeResult(
            track_ids=[],
            track_indices=[],
            seed_positions=[],
            segment_diagnostics=diagnostics,
            stats={"error": "length_mismatch", "expected": total_tracks, "actual": len(final_track_ids)},
            success=False,
            failure_reason=failure_msg,
        )

    # Compute per-edge transition scores for reporting (matches builder scoring)
    edge_scores: list[dict[str, Any]] = []
    transition_vals: list[float] = []
    for i in range(1, len(final_indices)):
        prev_idx = final_indices[i - 1]
        cur_idx = final_indices[i]
        t_val = _compute_transition_score(
            prev_idx,
            cur_idx,
            X_full_tr_norm,
            X_start_tr_norm,
            X_mid_tr_norm,
            X_end_tr_norm,
            cfg,
        )
        s_val = float(np.dot(X_full_norm[prev_idx], X_full_norm[cur_idx]))
        g_val = None
        if X_genre_norm is not None:
            g_val = float(np.dot(X_genre_norm[prev_idx], X_genre_norm[cur_idx]))
        transition_vals.append(float(t_val))
        edge_scores.append(
            {
                "prev_id": str(bundle.track_ids[prev_idx]),
                "cur_id": str(bundle.track_ids[cur_idx]),
                "prev_idx": int(prev_idx),
                "cur_idx": int(cur_idx),
                "T": float(t_val),
                "S": float(s_val),
                "G": (float(g_val) if g_val is not None else None),
            }
        )

    # Recompute seed positions from final track IDs for diagnostic accuracy
    seed_positions = [idx for idx, tid in enumerate(final_track_ids) if tid in seed_id_set]
    if len(seed_positions) != (1 if is_single_seed_arc else len(seed_id_set)):
        logger.warning(
            "Pier+Bridge: seed count mismatch in final result (expected %d, found %d)",
            (1 if is_single_seed_arc else len(seed_id_set)),
            len(seed_positions),
        )

    # Compute overall stats
    actual_num_seeds = 1 if is_single_seed_arc else len(seed_indices)
    stats = {
        "num_seeds": actual_num_seeds,
        "single_seed_arc": is_single_seed_arc,
        "target_tracks": total_tracks,
        "actual_tracks": len(final_indices),
        "universe_size": len(universe),
        "segments_built": len(all_segments),
        "segments_successful": sum(1 for d in diagnostics if d.success),
        "total_expansions": sum(d.expansions for d in diagnostics),
        "edge_scores": edge_scores,
        "min_transition": float(np.min(transition_vals)) if transition_vals else None,
        "mean_transition": float(np.mean(transition_vals)) if transition_vals else None,
        "transition_centered": bool(cfg.center_transitions),
        "soft_genre_penalty_hits": int(soft_genre_penalty_hits_total),
        "soft_genre_penalty_edges_scored": int(soft_genre_penalty_edges_scored_total),
        "segment_bridge_floors_used": [float(x) for x in segment_bridge_floors_used],
        "segment_backoff_attempts_used": [int(x) for x in segment_backoff_attempts_used],
        "warnings": warnings,
        "config": {
            "transition_floor": cfg.transition_floor,
            "transition_weights": cfg.transition_weights,
            "initial_neighbors_m": cfg.initial_neighbors_m,
            "initial_beam_width": cfg.initial_beam_width,
            "eta_destination_pull": cfg.eta_destination_pull,
            "genre_tiebreak_weight": float(cfg.genre_tiebreak_weight),
            "genre_penalty_threshold": float(cfg.genre_penalty_threshold),
            "genre_penalty_strength": float(cfg.genre_penalty_strength),
            "genre_tie_break_band": (
                float(cfg.genre_tie_break_band) if cfg.genre_tie_break_band is not None else None
            ),
            "bridge_floor": float(cfg.bridge_floor),
            "infeasible_handling_enabled": bool(infeasible_handling and infeasible_handling.enabled),
            "experiment_bridge_scoring": {
                "enabled": bool(cfg.experiment_bridge_scoring_enabled),
                "min_weight": float(cfg.experiment_bridge_min_weight),
                "balance_weight": float(cfg.experiment_bridge_balance_weight),
            },
            "dj_bridging": {
                "enabled": bool(cfg.dj_bridging_enabled),
                "seed_ordering": str(cfg.dj_seed_ordering),
                "anchors_must_include_all": bool(cfg.dj_anchors_must_include_all),
                "route_shape": str(cfg.dj_route_shape),
                "waypoint_weight": float(cfg.dj_waypoint_weight),
                "waypoint_floor": float(cfg.dj_waypoint_floor),
                "waypoint_penalty": float(cfg.dj_waypoint_penalty),
                "waypoint_tie_break_band": (
                    float(cfg.dj_waypoint_tie_break_band) if cfg.dj_waypoint_tie_break_band is not None else None
                ),
                "waypoint_cap": float(cfg.dj_waypoint_cap),
                "seed_ordering_weights": {
                    "sonic": float(cfg.dj_seed_ordering_weight_sonic),
                    "genre": float(cfg.dj_seed_ordering_weight_genre),
                    "bridge": float(cfg.dj_seed_ordering_weight_bridge),
                },
                "pooling_strategy": str(cfg.dj_pooling_strategy),
                "pooling_k_local": int(cfg.dj_pooling_k_local),
                "pooling_k_toward": int(cfg.dj_pooling_k_toward),
                "pooling_k_genre": int(cfg.dj_pooling_k_genre),
                "pooling_k_union_max": int(cfg.dj_pooling_k_union_max),
                "pooling_step_stride": int(cfg.dj_pooling_step_stride),
                "pooling_cache_enabled": bool(cfg.dj_pooling_cache_enabled),
                "pooling_debug_compare_baseline": bool(
                    cfg.dj_pooling_debug_compare_baseline
                ),
                "allow_detours_when_far": bool(cfg.dj_allow_detours_when_far),
                "far_thresholds": {
                    "sonic": float(cfg.dj_far_threshold_sonic),
                    "genre": float(cfg.dj_far_threshold_genre),
                    "connector_scarcity": float(cfg.dj_far_threshold_connector_scarcity),
                },
                "connector_bias": {
                    "enabled": bool(cfg.dj_connector_bias_enabled),
                    "max_per_segment_linear": int(cfg.dj_connector_max_per_segment_linear),
                    "max_per_segment_adventurous": int(cfg.dj_connector_max_per_segment_adventurous),
                },
                "ladder": {
                    "top_labels": int(cfg.dj_ladder_top_labels),
                    "min_label_weight": float(cfg.dj_ladder_min_label_weight),
                    "min_similarity": float(cfg.dj_ladder_min_similarity),
                    "max_steps": int(cfg.dj_ladder_max_steps),
                    "use_smoothed_waypoint_vectors": bool(
                        cfg.dj_ladder_use_smoothed_waypoint_vectors
                    ),
                    "smooth_top_k": int(cfg.dj_ladder_smooth_top_k),
                    "smooth_min_sim": float(cfg.dj_ladder_smooth_min_sim),
                },
                "waypoint_fallback_k": int(cfg.dj_waypoint_fallback_k),
                "micro_piers": {
                    "enabled": bool(cfg.dj_micro_piers_enabled),
                    "max": int(cfg.dj_micro_piers_max),
                    "topk": int(cfg.dj_micro_piers_topk),
                    "candidate_source": str(cfg.dj_micro_piers_candidate_source),
                    "selection_metric": str(cfg.dj_micro_piers_selection_metric),
                },
                "relaxation": {
                    "enabled": bool(cfg.dj_relaxation_enabled),
                    "max_attempts": int(cfg.dj_relaxation_max_attempts),
                    "emit_warnings": bool(cfg.dj_relaxation_emit_warnings),
                    "allow_floor_relaxation": bool(cfg.dj_relaxation_allow_floor_relaxation),
                },
            },
            "progress_arc": {
                "enabled": bool(cfg.progress_arc_enabled),
                "weight": float(cfg.progress_arc_weight),
                "shape": str(cfg.progress_arc_shape),
                "tolerance": float(cfg.progress_arc_tolerance),
                "loss": str(cfg.progress_arc_loss),
                "huber_delta": float(cfg.progress_arc_huber_delta),
                "max_step": (float(cfg.progress_arc_max_step) if cfg.progress_arc_max_step is not None else None),
                "max_step_mode": str(cfg.progress_arc_max_step_mode),
                "max_step_penalty": float(cfg.progress_arc_max_step_penalty),
                "autoscale": {
                    "enabled": bool(cfg.progress_arc_autoscale_enabled),
                    "min_distance": float(cfg.progress_arc_autoscale_min_distance),
                    "distance_scale": float(cfg.progress_arc_autoscale_distance_scale),
                    "per_step_scale": bool(cfg.progress_arc_autoscale_per_step_scale),
                },
            },
        },
    }

    logger.info("Pier+Bridge complete: %d tracks, %d segments, %d successful",
               len(final_indices), len(all_segments),
               sum(1 for d in diagnostics if d.success))

    return PierBridgeResult(
        track_ids=final_track_ids,
        track_indices=final_indices,
        seed_positions=seed_positions,
        segment_diagnostics=diagnostics,
        stats=stats,
    )


def generate_pier_bridge_playlist(
    *,
    artifact_path: str,
    seed_track_ids: List[str],
    total_tracks: int,
    mode: str = "dynamic",
    random_seed: int = 0,
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
    transition_floor: Optional[float] = None,
) -> PierBridgeResult:
    """
    High-level entry point for pier+bridge playlist generation.

    Loads artifacts, builds candidate pool, and runs pier+bridge construction.
    """
    from src.features.artifacts import load_artifact_bundle
    from src.playlist.config import default_ds_config
    from src.playlist.candidate_pool import build_candidate_pool
    from src.similarity.hybrid import build_hybrid_embedding
    from src.similarity.sonic_variant import compute_sonic_variant_matrix, resolve_sonic_variant

    bundle = load_artifact_bundle(artifact_path)

    # Validate seeds
    valid_seed_ids = []
    for tid in seed_track_ids:
        if str(tid) in bundle.track_id_to_index:
            valid_seed_ids.append(str(tid))
        else:
            logger.warning("Seed track not found, skipping: %s", tid)

    if not valid_seed_ids:
        raise ValueError("No valid seed tracks found in artifact bundle")

    seed_idx = bundle.track_id_to_index[valid_seed_ids[0]]

    # Build config
    cfg = default_ds_config(mode, playlist_len=total_tracks)

    # Build embedding
    resolved_variant = resolve_sonic_variant()
    X_sonic_for_embed, _ = compute_sonic_variant_matrix(bundle.X_sonic, resolved_variant, l2=False)

    embedding_model = build_hybrid_embedding(
        X_sonic_for_embed,
        bundle.X_genre_smoothed,
        n_components_sonic=32,
        n_components_genre=32,
        w_sonic=0.6,
        w_genre=0.4,
        random_seed=random_seed,
    )

    # Build candidate pool (for genre gating)
    pool = build_candidate_pool(
        seed_idx=seed_idx,
        seed_indices=[seed_idx],
        embedding=embedding_model.embedding,
        artist_keys=bundle.artist_keys,
        track_ids=bundle.track_ids,
        track_titles=bundle.track_titles,
        track_artists=bundle.track_artists,
        durations_ms=bundle.durations_ms,
        cfg=cfg.candidate,
        random_seed=random_seed,
        X_sonic=X_sonic_for_embed,
        X_genre_raw=bundle.X_genre_raw if min_genre_similarity else None,
        X_genre_smoothed=bundle.X_genre_smoothed if min_genre_similarity else None,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method,
        mode=mode,
    )

    # Build pier config
    pier_cfg = PierBridgeConfig()
    if transition_floor is not None:
        pier_cfg = PierBridgeConfig(transition_floor=transition_floor)
    else:
        pier_cfg = PierBridgeConfig(transition_floor=cfg.construct.transition_floor)

    return build_pier_bridge_playlist(
        seed_track_ids=valid_seed_ids,
        total_tracks=total_tracks,
        bundle=bundle,
        candidate_pool_indices=list(pool.pool_indices),
        cfg=pier_cfg,
        min_genre_similarity=min_genre_similarity,
        X_genre_smoothed=bundle.X_genre_smoothed,
        genre_method=genre_method,
    )
