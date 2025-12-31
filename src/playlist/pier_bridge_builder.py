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
- One track per artist per segment (provides implicit min_gap without explicit constraints)
- Single seed mode: seed acts as both start AND end pier, creating an arc structure
- Seed artist is allowed in bridges with same constraints as other artists
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.features.artifacts import ArtifactBundle, get_sonic_matrix
from src.title_dedupe import normalize_title_for_dedupe, normalize_artist_key   
from src.string_utils import sanitize_for_logging
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.config import resolve_pier_bridge_tuning as _resolve_pier_bridge_tuning_cfg
from src.playlist.run_audit import InfeasibleHandlingConfig, RunAuditConfig, RunAuditEvent, now_utc_iso

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


@dataclass
class SegmentDiagnostics:
    """Diagnostics for a single segment."""
    pier_a_id: str
    pier_b_id: str
    target_length: int
    actual_length: int
    pool_size_initial: int
    pool_size_final: int
    expansions: int
    beam_width_used: int
    worst_edge_score: float
    mean_edge_score: float
    success: bool
    bridge_floor_used: float = 0.0
    backoff_attempts_used: int = 1
    widened_search: bool = False


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


def _order_seeds_by_bridgeability(
    seed_indices: List[int],
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
) -> List[int]:
    """
    Order seed indices to maximize total bridgeability.
    For <=6 seeds, evaluates all permutations.
    For >6 seeds, uses greedy nearest-neighbor heuristic.
    """
    n = len(seed_indices)
    if n <= 1:
        return seed_indices

    if n <= 6:
        # Exhaustive search for small seed counts
        best_order = None
        best_score = -float('inf')

        for perm in itertools.permutations(seed_indices):
            total_score = 0.0
            for i in range(len(perm) - 1):
                total_score += _compute_bridgeability_score(
                    perm[i], perm[i + 1],
                    X_full_norm, X_start_norm, X_end_norm
                )
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
                score = _compute_bridgeability_score(
                    current, candidate,
                    X_full_norm, X_start_norm, X_end_norm
                )
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
) -> tuple[List[int], Dict[int, str], Dict[int, str]]:
    """
    Segment-local candidate pool builder ("segment_scored").

    Builds a segment candidate pool by scoring candidates jointly vs BOTH endpoints (pier A and pier B),
    applying structural exclusions (used ids, allowed-set clamp, artist policies, track_key collisions),
    gating by bridge_floor, then taking top-K by harmonic_mean(simA, simB).

    Returns:
      - candidates: list[int] indices for beam search
      - artist_key_by_idx: mapping for candidates (robust identity key)
      - title_key_by_idx: mapping for candidates (normalized title key)
    """
    segment_pool_max = int(max(0, segment_pool_max))
    if segment_pool_max <= 0:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "pool_strategy": "segment_scored",
                    "base_universe": int(len(universe_indices)),
                    "excluded_used_track_ids": 0,
                    "excluded_allowed_set": 0,
                    "excluded_seed_artist_policy": 0,
                    "excluded_pier_artist_policy": 0,
                    "excluded_track_key_collision": 0,
                    "excluded_track_key_collision_with_piers": 0,
                    "eligible_after_structural": 0,
                    "below_bridge_floor": 0,
                    "pass_bridge_floor": 0,
                    "collapsed_by_artist_key": 0,
                    "selected_external": 0,
                    "internal_connectors_candidates": 0,
                    "internal_connectors_pass_gate": 0,
                    "internal_connectors_selected": 0,
                    "final": 0,
                    "segment_pool_max": int(segment_pool_max),
                }
            )
        return [], {}, {}

    used_track_keys = used_track_keys or set()
    seed_track_keys = seed_track_keys or set()

    # Endpoint artist keys (robust identity), for optional policies
    pier_a_artist_key = identity_keys_for_index(bundle, pier_a).artist_key
    pier_b_artist_key = identity_keys_for_index(bundle, pier_b).artist_key

    excluded_used = 0
    excluded_allowed = 0
    excluded_seed_artist = 0
    excluded_pier_artist = 0
    excluded_track_key = 0
    excluded_track_key_with_piers = 0

    artist_key_by_idx: Dict[int, str] = {}
    title_key_by_idx: Dict[int, str] = {}

    # Structural filtering (segment-local): used ids, allowed-set clamp, policies, track_key collisions
    structural: List[int] = []
    for idx in universe_indices:
        i = int(idx)
        if i in used_track_ids:
            excluded_used += 1
            continue
        if allowed_set is not None and i not in allowed_set:
            excluded_allowed += 1
            continue

        keys = identity_keys_for_index(bundle, i)
        ak = keys.artist_key
        tk = keys.title_key
        artist_key_by_idx[i] = ak
        title_key_by_idx[i] = tk

        if disallow_seed_artist_in_interiors and seed_artist_key and ak == seed_artist_key:
            excluded_seed_artist += 1
            continue
        if disallow_pier_artists_in_interiors and ak in {pier_a_artist_key, pier_b_artist_key}:
            excluded_pier_artist += 1
            continue

        if keys.track_key in used_track_keys:
            excluded_track_key += 1
            if keys.track_key in seed_track_keys:
                excluded_track_key_with_piers += 1
            continue

        structural.append(i)

    if not structural:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "pool_strategy": "segment_scored",
                    "base_universe": int(len(universe_indices)),
                    "excluded_used_track_ids": int(excluded_used),
                    "excluded_allowed_set": int(excluded_allowed),
                    "excluded_seed_artist_policy": int(excluded_seed_artist),
                    "excluded_pier_artist_policy": int(excluded_pier_artist),
                    "excluded_track_key_collision": int(excluded_track_key),
                    "excluded_track_key_collision_with_piers": int(excluded_track_key_with_piers),
                    "eligible_after_structural": 0,
                    "below_bridge_floor": 0,
                    "pass_bridge_floor": 0,
                    "collapsed_by_artist_key": 0,
                    "selected_external": 0,
                    "internal_connectors_candidates": 0,
                    "internal_connectors_pass_gate": 0,
                    "internal_connectors_selected": 0,
                    "final": 0,
                    "segment_pool_max": int(segment_pool_max),
                }
            )
        return [], {}, {}

    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    below_bridge_floor = 0
    passing: List[int] = []
    bridge_sim: Dict[int, float] = {}
    for i in structural:
        sim_a = float(sim_to_a[i])
        sim_b = float(sim_to_b[i])
        if min(sim_a, sim_b) < float(bridge_floor):
            below_bridge_floor += 1
            continue
        denom = sim_a + sim_b
        hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
        bridge_sim[i] = float(hmean)
        passing.append(i)

    # Rank by bridge_sim (two-sided), then apply 1-per-artist_key within the segment.
    passing_sorted = sorted(passing, key=lambda i: (-float(bridge_sim.get(i, 0.0)), int(i)))

    # Internal connectors (optional; still gated + policy-checked)
    internal_candidates = 0
    internal_pass_gate = 0
    internal_selected: List[int] = []
    internal_ranked: List[tuple[float, int]] = []
    if internal_connectors:
        for idx in internal_connectors:
            i = int(idx)
            if i in used_track_ids:
                continue
            if allowed_set is not None and i not in allowed_set:
                continue
            keys = identity_keys_for_index(bundle, i)
            ak = keys.artist_key
            tk = keys.title_key
            artist_key_by_idx[i] = ak
            title_key_by_idx[i] = tk

            if disallow_seed_artist_in_interiors and seed_artist_key and ak == seed_artist_key:
                continue
            if disallow_pier_artists_in_interiors and ak in {pier_a_artist_key, pier_b_artist_key}:
                continue
            if keys.track_key in used_track_keys:
                continue

            internal_candidates += 1
            sim_a = float(sim_to_a[i])
            sim_b = float(sim_to_b[i])
            if min(sim_a, sim_b) < float(bridge_floor):
                continue
            internal_pass_gate += 1
            denom = sim_a + sim_b
            hmean = 0.0 if denom <= 1e-9 else (2.0 * sim_a * sim_b) / denom
            internal_ranked.append((float(hmean), i))
        internal_ranked.sort(key=lambda t: (-t[0], t[1]))

    selected_external: List[int] = []
    collapsed_by_artist = 0

    if internal_connector_priority:
        # Select internal connectors first, then fill from external candidates.
        used_artists: Set[str] = set()
        cap = int(internal_connector_cap) if int(internal_connector_cap) > 0 else len(internal_ranked)
        for _score, i in internal_ranked:
            ak = artist_key_by_idx.get(i) or identity_keys_for_index(bundle, i).artist_key
            if ak in used_artists:
                continue
            internal_selected.append(i)
            used_artists.add(ak)
            if len(internal_selected) >= cap:
                break

        for i in passing_sorted:
            ak = artist_key_by_idx.get(i) or identity_keys_for_index(bundle, i).artist_key
            if ak in used_artists:
                collapsed_by_artist += 1
                continue
            used_artists.add(ak)
            selected_external.append(i)
            if len(selected_external) >= int(segment_pool_max):
                break
        combined = list(dict.fromkeys(internal_selected + selected_external))
    else:
        # Select external first, then add internal connectors up to cap.
        used_artists = set()
        for i in passing_sorted:
            ak = artist_key_by_idx.get(i) or identity_keys_for_index(bundle, i).artist_key
            if ak in used_artists:
                collapsed_by_artist += 1
                continue
            used_artists.add(ak)
            selected_external.append(i)
            if len(selected_external) >= int(segment_pool_max):
                break

        cap = int(internal_connector_cap) if int(internal_connector_cap) > 0 else len(internal_ranked)
        for _score, i in internal_ranked:
            ak = artist_key_by_idx.get(i) or identity_keys_for_index(bundle, i).artist_key
            if ak in used_artists:
                continue
            internal_selected.append(i)
            used_artists.add(ak)
            if len(internal_selected) >= cap:
                break
        combined = list(dict.fromkeys(selected_external + internal_selected))

    if diagnostics is not None:
        diagnostics.update(
            {
                "pool_strategy": "segment_scored",
                "base_universe": int(len(universe_indices)),
                "excluded_used_track_ids": int(excluded_used),
                "excluded_allowed_set": int(excluded_allowed),
                "excluded_seed_artist_policy": int(excluded_seed_artist),
                "excluded_pier_artist_policy": int(excluded_pier_artist),
                "excluded_track_key_collision": int(excluded_track_key),
                "excluded_track_key_collision_with_piers": int(excluded_track_key_with_piers),
                "eligible_after_structural": int(len(structural)),
                "below_bridge_floor": int(below_bridge_floor),
                "pass_bridge_floor": int(len(passing)),
                "collapsed_by_artist_key": int(collapsed_by_artist),
                "selected_external": int(len(selected_external)),
                "internal_connectors_candidates": int(internal_candidates),
                "internal_connectors_pass_gate": int(internal_pass_gate),
                "internal_connectors_selected": int(len(internal_selected)),
                "final": int(len(combined)),
                "segment_pool_max": int(segment_pool_max),
            }
        )

    # Only return mappings for indices in the final candidate list (beam search scope)
    artist_key_final = {i: artist_key_by_idx.get(i, "") for i in combined}
    title_key_final = {i: title_key_by_idx.get(i, "") for i in combined}
    return combined, artist_key_final, title_key_final


@dataclass
class BeamState:
    """State for beam search."""
    path: List[int]
    score: float
    used: Set[int]
    used_artists: Set[str] = field(default_factory=set)
    last_progress: float = 0.0


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
    artist_key_by_idx: Optional[Dict[int, str]] = None,
    seed_artist_key: Optional[str] = None,
) -> Tuple[Optional[List[int]], int, int, Optional[str]]:
    """
    Constrained beam search to find path from pier_a to pier_b.

    Returns interior track indices (not including piers) or None if no path found.
    """

    genre_penalty_hits = 0
    edges_scored = 0
    penalty_strength = float(cfg.genre_penalty_strength)
    if not math.isfinite(penalty_strength):
        penalty_strength = 0.0
    penalty_strength = float(max(0.0, min(1.0, penalty_strength)))
    penalty_threshold = float(cfg.genre_penalty_threshold)

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

    progress_by_idx: Dict[int, float] = {}
    if progress_active:
        vec_a_full = X_full_norm[pier_a]
        vec_b_full = X_full_norm[pier_b]
        d = vec_b_full - vec_a_full
        denom = float(np.dot(d, d))
        if (not math.isfinite(denom)) or denom <= 1e-12:
            progress_active = False
        else:
            progress_by_idx[pier_a] = 0.0
            progress_by_idx[pier_b] = 1.0
            for cand in candidates:
                i = int(cand)
                t_raw = float(np.dot((X_full_norm[i] - vec_a_full), d) / denom)
                t = 0.0 if not math.isfinite(t_raw) else float(max(0.0, min(1.0, t_raw)))
                progress_by_idx[i] = t

    vec_b_full = X_full_norm[pier_b]
    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    # Initialize beam with pier_a
    used_artists_init: Set[str] = set()
    if artist_key_by_idx is not None:
        if cfg.disallow_pier_artists_in_interiors:
            ak_a = str(artist_key_by_idx.get(int(pier_a), "") or "")
            ak_b = str(artist_key_by_idx.get(int(pier_b), "") or "")
            if ak_a:
                used_artists_init.add(ak_a)
            if ak_b:
                used_artists_init.add(ak_b)
        if cfg.disallow_seed_artist_in_interiors and seed_artist_key:
            used_artists_init.add(str(seed_artist_key))

    initial_state = BeamState(
        path=[pier_a],
        score=0.0,
        used={pier_a, pier_b},
        used_artists=used_artists_init,
        last_progress=0.0,
    )
    beam = [initial_state]

    for step in range(interior_length):
        next_beam: List[BeamState] = []
        target_t = float(step + 1) / float(interior_length + 1)

        for state in beam:
            current = state.path[-1]

            for cand in candidates:
                if cand in state.used:
                    continue
                if artist_key_by_idx is not None:
                    cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                    if cand_artist and cand_artist in state.used_artists:
                        continue
                if min(sim_to_a[cand], sim_to_b[cand]) < cfg.bridge_floor:      
                    continue
                if progress_active:
                    cand_t = float(progress_by_idx.get(int(cand), 0.0))
                    if cand_t < float(state.last_progress) - progress_eps:
                        continue

                # Compute transition score
                trans_score = _compute_transition_score(
                    current, cand, X_full, X_start, X_mid, X_end, cfg
                )

                # Hard floors: transition + bridge-local
                if trans_score < cfg.transition_floor:
                    continue

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
                edges_scored += 1
                if X_genre_norm is not None:
                    genre_sim = float(np.dot(X_genre_norm[current], X_genre_norm[cand]))
                    if math.isfinite(genre_sim):
                        if cfg.genre_tiebreak_weight:
                            combined_score += cfg.genre_tiebreak_weight * genre_sim
                        if (
                            penalty_strength > 0
                            and genre_sim < penalty_threshold
                        ):
                            combined_score *= (1.0 - penalty_strength)
                            genre_penalty_hits += 1

                new_score = state.score + combined_score + dest_pull
                new_path = state.path + [cand]
                new_used = state.used | {cand}
                new_used_artists = state.used_artists
                if artist_key_by_idx is not None:
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
            return None, genre_penalty_hits, edges_scored, f"no valid continuations at step={step}"

        # Keep top beam_width states
        next_beam.sort(key=lambda s: s.score, reverse=True)
        beam = next_beam[:beam_width]

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
            genre_sim = float(np.dot(X_genre_norm[last], X_genre_norm[pier_b]))
            if math.isfinite(genre_sim):
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
        return None, genre_penalty_hits, edges_scored, "no valid final connection to destination"

    # Return interior tracks (exclude pier_a which is path[0])
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
) -> Tuple[List[int], int]:
    """
    Drop tracks that would violate a global min_gap across concatenated segments.

    Pier-bridge already enforces one-per-artist per segment, but adjacent
    duplicates can appear at segment boundaries. This pass removes any track
    that would repeat a normalized artist within the last `min_gap` slots.
    """
    if not indices or min_gap <= 0:
        return indices, 0

    recent: List[str] = []
    output: List[int] = []
    dropped = 0

    for idx in indices:
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

    # Genre similarity for soft edge penalty / tiebreak (cosine on smoothed genre vectors)
    X_genre_use = X_genre_smoothed if X_genre_smoothed is not None else getattr(bundle, "X_genre_smoothed", None)
    X_genre_norm = None
    if X_genre_use is not None:
        denom_g = np.linalg.norm(X_genre_use, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre_use / denom_g

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

    # Order seeds by bridgeability
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

    for seg_idx in range(num_segments):
        pier_a = ordered_seeds[seg_idx]
        pier_b = ordered_seeds[seg_idx + 1]
        interior_len = segment_lengths[seg_idx]

        pier_a_id = str(bundle.track_ids[pier_a])
        pier_b_id = str(bundle.track_ids[pier_b])

        logger.info("Building segment %d: %s -> %s (interior=%d)",
                   seg_idx, pier_a_id, pier_b_id, interior_len)

        # Optional bridge_floor backoff on infeasible segments (default OFF)
        segment_path: Optional[List[int]] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts = _bridge_floor_attempts(float(cfg.bridge_floor))
        backoff_used_count = 0
        widened_search_used = False
        last_failure_reason: Optional[str] = None

        # Defaults for diagnostics (filled on success/last attempt)
        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0

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
                extra_pool = int(infeasible_handling.extra_neighbors_m) + int(infeasible_handling.extra_bridge_helpers)
                segment_pool_max = min(segment_pool_max + extra_pool, int(cfg.max_segment_pool_max))
                beam_width = min(beam_width + int(infeasible_handling.extra_beam_width), cfg.max_beam_width)
                max_expansion_attempts = max_expansion_attempts + int(infeasible_handling.extra_expansion_attempts)

            expansions = 0
            pool_size_initial = 0
            pool_size_final = 0
            soft_genre_penalty_hits_segment = 0
            soft_genre_penalty_edges_scored_segment = 0
            last_failure_reason = None
            expansion_attempts_used = 0
            last_pool_diag: Dict[str, Any] = {}
            last_segment_candidates: List[int] = []
            last_candidate_artist_keys: Dict[int, str] = {}
            last_segment_pool_max = int(segment_pool_max)
            last_beam_width = int(beam_width)

            for attempt in range(max_expansion_attempts):
                pool_diag: Dict[str, Any] = {}
                cand_artist_keys: Dict[int, str] = {}
                if str(cfg.segment_pool_strategy).strip().lower() == "legacy":
                    # Legacy/compat strategy (combined neighbor pooling).
                    neighbors_m = min(int(cfg.initial_neighbors_m) * (2 ** int(attempt)), int(cfg.max_neighbors_m))
                    bridge_helpers = min(int(cfg.initial_bridge_helpers) * (2 ** int(attempt)), int(cfg.max_bridge_helpers))
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
                        internal_connectors=internal_connector_indices,
                        internal_connector_cap=internal_connector_max_per_segment,
                        internal_connector_priority=internal_connector_priority,
                        diagnostics=pool_diag,
                    )
                    try:
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(bundle, int(pier_a)).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(bundle, int(pier_b)).artist_key
                        for c in segment_candidates:
                            cand_artist_keys[int(c)] = identity_keys_for_index(bundle, int(c)).artist_key
                    except Exception:
                        cand_artist_keys = {}
                else:
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
                        internal_connectors=internal_connector_indices,
                        internal_connector_cap=internal_connector_max_per_segment,
                        internal_connector_priority=internal_connector_priority,
                        seed_artist_key=seed_artist_key,
                        disallow_pier_artists_in_interiors=bool(cfg.disallow_pier_artists_in_interiors),
                        disallow_seed_artist_in_interiors=bool(cfg.disallow_seed_artist_in_interiors),
                        used_track_keys=used_track_keys,
                        seed_track_keys=seed_track_keys,
                        diagnostics=pool_diag,
                    )
                    try:
                        cand_artist_keys = dict(cand_artist_keys)
                        cand_artist_keys[int(pier_a)] = identity_keys_for_index(bundle, int(pier_a)).artist_key
                        cand_artist_keys[int(pier_b)] = identity_keys_for_index(bundle, int(pier_b)).artist_key
                    except Exception:
                        cand_artist_keys = {}
                expansion_attempts_used = attempt + 1
                if attempt == 0:
                    pool_size_initial = len(segment_candidates)
                pool_size_final = len(segment_candidates)
                last_pool_diag = dict(pool_diag)
                last_segment_candidates = list(segment_candidates)        
                last_candidate_artist_keys = dict(cand_artist_keys)
                last_segment_pool_max = int(segment_pool_max)
                last_beam_width = int(beam_width)

                if len(segment_candidates) < interior_len:
                    last_failure_reason = f"pool_after_gate {len(segment_candidates)} < interior_len {interior_len}"
                    segment_path = None
                    soft_genre_penalty_hits_segment = 0
                    soft_genre_penalty_edges_scored_segment = 0
                    beam_failure_reason = None
                else:
                    if cand_artist_keys:
                        try:
                            cand_artist_keys[int(pier_a)] = identity_keys_for_index(
                                bundle, int(pier_a)
                            ).artist_key
                            cand_artist_keys[int(pier_b)] = identity_keys_for_index(
                                bundle, int(pier_b)
                            ).artist_key
                        except Exception:
                            pass
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
                        artist_key_by_idx=(cand_artist_keys if cand_artist_keys else None),
                        seed_artist_key=seed_artist_key,
                    )
                    last_failure_reason = beam_failure_reason

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
                            "segment_pool_strategy": str(cfg.segment_pool_strategy),
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
                            "top_candidates": top_rows,
                            "reason": ("success" if segment_path is not None else last_failure_reason),
                        },
                    )
                )

            if segment_path is not None:
                chosen_bridge_floor = float(bridge_floor)
                beam_width_used = int(last_beam_width)
                if audit_enabled:
                    audit_events.append(
                        RunAuditEvent(
                            kind="segment_success",
                            ts_utc=now_utc_iso(),
                            payload={
                                "segment_index": int(seg_idx),
                                "bridge_floor_used": float(chosen_bridge_floor),
                                "backoff_attempts_used": int(backoff_used_count),
                                "widened_search": bool(widened_search_used),
                            },
                        )
                    )
                break

        if segment_path is None:
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
        ))
        segment_bridge_floors_used.append(float(chosen_bridge_floor))
        segment_backoff_attempts_used.append(int(backoff_used_count))
        logger.info(
            "Segment %d: %s -> %s bridge_floor=%.2f pool_before=%d pool_after=%d",
            seg_idx, pier_a_id, pier_b_id, float(chosen_bridge_floor), pool_size_initial, pool_size_final,
        )
        # DEBUG top candidates for this segment
        if logger.isEnabledFor(logging.DEBUG):
            scores_dbg = []
            sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
            sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])
            for cand in segment_candidates[: min(200, len(segment_candidates))]:
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

    # Convert to track IDs (after enforcing cross-segment min_gap to avoid back-to-back repeats)
    final_indices, dropped = _enforce_min_gap_global(
        final_indices, bundle.artist_keys, min_gap=1, bundle=bundle
    )
    if dropped:
        logger.debug(
            "Pier+Bridge: dropped %d tracks to enforce cross-segment min_gap", dropped
        )

    final_track_ids = [str(bundle.track_ids[i]) for i in final_indices]

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

    # Recompute seed positions after any min-gap pruning to keep diagnostics consistent
    seed_positions = [idx for idx, tid in enumerate(final_track_ids) if tid in seed_id_set]
    if len(seed_positions) != (1 if is_single_seed_arc else len(seed_id_set)):
        logger.debug(
            "Pier+Bridge: seed count mismatch after pruning (expected %d, found %d)",
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
        "config": {
            "transition_floor": cfg.transition_floor,
            "transition_weights": cfg.transition_weights,
            "initial_neighbors_m": cfg.initial_neighbors_m,
            "initial_beam_width": cfg.initial_beam_width,
            "eta_destination_pull": cfg.eta_destination_pull,
            "genre_tiebreak_weight": float(cfg.genre_tiebreak_weight),
            "genre_penalty_threshold": float(cfg.genre_penalty_threshold),
            "genre_penalty_strength": float(cfg.genre_penalty_strength),
            "bridge_floor": float(cfg.bridge_floor),
            "infeasible_handling_enabled": bool(infeasible_handling and infeasible_handling.enabled),
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
