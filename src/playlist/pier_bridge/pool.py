"""
Tier-3.1 PR-7: segment candidate pool builders + bridge-score kernel.

Extracted verbatim from pier_bridge_builder.py:
  _compute_bridge_score           — scoring kernel (harmonic mean or experiment blend)
  _build_segment_candidate_pool_legacy  — legacy pool (debug/compat; still reachable)
  _build_segment_candidate_pool_scored  — production pool (thin wrapper around SegmentCandidatePoolBuilder)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.segment_pool_builder import (
    SegmentCandidatePoolBuilder,
    SegmentPoolConfig,
)
from src.title_dedupe import normalize_artist_key

logger = logging.getLogger(__name__)


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
    genre_pool_transition_blend: float = 0.0,  # Task D: Blend weight for genre pool
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
        genre_pool_transition_blend=float(genre_pool_transition_blend),  # Task D
    )
    result = SegmentCandidatePoolBuilder().build(pool_cfg)
    return result.candidates, result.artist_key_by_idx, result.title_key_by_idx
