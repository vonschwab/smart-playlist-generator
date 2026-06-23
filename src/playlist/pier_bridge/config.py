"""Pier-bridge configuration dataclasses + helpers (Tier-3.1 PR-3).

Extracted from pier_bridge_builder.py. This module owns:
  * PierBridgeConfig — the master config dataclass (~70 fields)
  * PierBridgeResult — the result dataclass
  * resolve_pier_bridge_tuning — dict-returning back-compat wrapper around the
    canonical dataclass-returning resolver in src.playlist.config
  * _compute_genre_idf / _compute_transition_score(_raw_and_transformed) —
    back-compat wrappers that unpack a PierBridgeConfig into the primitive
    args expected by pier_bridge.genre / pier_bridge.vec
  * SegmentDiagnostics alias — re-exports the type from pier_bridge_diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.playlist.config import resolve_pier_bridge_tuning as _resolve_pier_bridge_tuning_cfg
from src.playlist.pier_bridge_diagnostics import (
    SegmentDiagnostics as _SegmentDiagnosticsExtracted,
)
from src.playlist.genre_idf import compute_genre_idf as _compute_genre_idf_shared
from src.playlist.pier_bridge.vec import (
    _compute_transition_score as _compute_transition_score_impl,
    _compute_transition_score_raw_and_transformed as _compute_transition_score_raw_and_transformed_impl,
)


# Backward compatibility: SegmentDiagnostics now imported from extracted module
# Kept here as alias for existing code
SegmentDiagnostics = _SegmentDiagnosticsExtracted


@dataclass
class PierBridgeConfig:
    """Configuration for pier + bridge playlist builder."""
    # NOTE: Defaults represent the recommended "dynamic" mode behavior. Narrow
    # mode defaults are resolved by the DS pipeline config layer.
    transition_floor: float = 0.35
    bridge_floor: float = 0.03  # min(simA, simB) for bridge candidates
    pace_bridge_floor: float = 0.0  # rhythm-axis moving-target floor; 0 disables
    bpm_bridge_max_log_distance: float = float("inf")  # inf = disabled
    bpm_stability_min: float = 0.5
    # BPM is meaningless on beatless audio (drone/ambient) — librosa still emits a
    # confident garbage tempo and tempo_stability is fooled (reads ~0.96 even for
    # drone). onset_rate is the reliable beat-presence signal. When > 0, the BPM
    # bridge band is bypassed for any track whose onset_rate is below this threshold:
    # a beatless PIER disables the band for the segment (its BPM can't set a target),
    # a beatless CANDIDATE skips it (its BPM can't be judged). 0.0 = off (legacy:
    # trust all BPMs). The onset band is unaffected — it is the trustworthy signal.
    bpm_trust_min_onset_rate: float = 0.0
    onset_bridge_max_log_distance: float = float("inf")  # inf = disabled
    # Pace bridge bands as SOFT penalties instead of hard gates. When strength > 0,
    # an out-of-band candidate is demoted by strength * (log_distance - max_log_distance)
    # rather than rejected — so an onset/BPM-outlier pier (e.g. a near-silent ambient
    # track) can't strand a segment and detonate the relaxation cascade. 0.0 = legacy
    # hard gate (reject), preserving backward-compatible behavior.
    bpm_bridge_soft_penalty_strength: float = 0.0
    onset_bridge_soft_penalty_strength: float = 0.0
    rhythm_soft_penalty_threshold: float = 0.0  # below this rhythm cosine, demote
    rhythm_soft_penalty_strength: float = 0.0   # multiplicative penalty (0 = off)
    # Energy (arousal) steering: soft penalty terms (never hard gates).
    # All default to 0.0 (disabled/no-op); presets enable per-mode.
    energy_step_cap: float = 0.0  # max z-std jump between adjacent tracks (soft cap)
    energy_step_strength: float = 0.0  # strength of step penalty (0 = disabled)
    energy_arc_band: float = 0.0  # z-std target band for segment arc (soft floor/ceiling)
    energy_arc_strength: float = 0.0  # strength of arc penalty (0 = disabled)
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
    # Layered genre graph transition scoring (opt-in; default OFF).
    # Uses sidecar-derived leaf/family/bridge/facet matrices when present on
    # the artifact bundle. This is separate from legacy flat genre steering.
    layered_transition_scoring_enabled: bool = False
    layered_transition_weight: float = 0.0
    layered_transition_mode: str = "dynamic"
    # Genre edge safeguards & steering (opt-in; code default OFF).
    # When enabled, the beam scores genre on the dense embedding, rejects edges
    # below genre_arc_floor (absolute fallback), and adds weight_genre * genre_sim
    # as a third term (bridge/transition/genre weights are pre-renormalized to sum to 1).
    genre_steering_enabled: bool = False
    # Steering space: "dense" (legacy PMI-SVD embedding) | "taxonomy" (SP3a graph
    # arc routing + hub-damped similarity). Opt-in; default preserves behavior.
    genre_steering_source: str = "dense"
    # Per-segment pool genre blend: blends genre harmonic-mean with sonic bridge
    # score during segment pool re-ranking. 0.0 = pure sonic (default, no change).
    segment_pool_genre_weight: float = 0.0
    weight_genre: float = 0.0
    genre_arc_floor: float = 0.0
    genre_arc_floor_percentile: float = 0.0
    genre_admission_percentile: float = 0.0
    # Pairwise genre-edge soft penalty: when the two ADJACENT tracks' tag-level
    # taxonomy similarity falls below genre_pair_floor, the edge's score is demoted
    # by subtracting genre_pair_penalty (NOT rejected — a hard gate detonates the
    # infeasibility/expansion machinery on broad-genre segments). The arc vote scores
    # candidates vs the smoothed waypoint target, never vs the actual neighbor; this
    # penalty steers away from bad edges while keeping the beam feasible. Genreless
    # endpoints are exempt. floor 0.0 = off (default); penalty only applies above it.
    genre_pair_floor: float = 0.0
    genre_pair_penalty: float = 0.5
    # Local sonic edge penalty (does not gate by default): demote candidates
    # whose immediate predecessor/successor sonic cosine falls below threshold.
    # Use local_sonic_edge_floor only for explicit hard-gate experiments.
    local_sonic_edge_penalty_enabled: bool = False
    local_sonic_edge_penalty_threshold: float = 0.10
    local_sonic_edge_penalty_strength: float = 0.0
    local_sonic_edge_penalty_mode: str = "legacy"
    """'legacy' (default) preserves the existing strength*(threshold-edge_cos) math.
    'scaled' uses scale*(threshold-edge_cos), producing penalties of 0.05-0.30
    that can actually influence beam selection. Tune local_sonic_edge_penalty_scale."""
    local_sonic_edge_penalty_scale: float = 1.0
    """Multiplier used in 'scaled' mode. Typical values 1.0-3.0. Ignored in 'legacy' mode."""
    local_sonic_edge_floor: Optional[float] = None
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
    # If True (legacy default), collapse the segment pool to one track per artist
    # before beam search. The beam already enforces per-segment artist diversity
    # via its own used_artists set, so setting this False gives the beam many more
    # tracks per artist at varied projection positions — useful for long
    # narrow-style segments where the "best per artist" tracks cluster around the
    # middle of the projection and starve the high-progress region.
    collapse_segment_pool_by_artist: bool = True
    # Progress model (A→B) to avoid "teleporting" / bouncing.
    progress_enabled: bool = True
    progress_monotonic_epsilon: float = 0.05
    progress_penalty_weight: float = 0.15
    # Interior artist policies (configured/wired by pipeline for legacy --artist runs).
    disallow_pier_artists_in_interiors: bool = False
    disallow_seed_artist_in_interiors: bool = False
    max_non_seed_tracks_per_artist: Optional[int] = None
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
    taxonomy_waypoint_min_library_mass: int = 0
    dj_ladder_use_smoothed_waypoint_vectors: bool = False
    dj_ladder_smooth_top_k: int = 10
    dj_ladder_smooth_min_sim: float = 0.20
    dj_waypoint_fallback_k: int = 25
    # Genre vector mode + IDF + Coverage (Phase 2)
    # Default flipped from legacy "onehot" to recommended "vector" (Tier-3.4).
    # The legacy "onehot" code path is preserved for anyone who explicitly sets it.
    dj_ladder_target_mode: str = "vector"  # "onehot" (legacy) | "vector"
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
    dj_genre_pool_transition_blend: float = 0.0  # Task D: Blend weight for genre pool (0.0-1.0)
    # Phase 3: Waypoint delta mode + squashing
    # Defaults flipped from legacy to Phase 3 recommended values (Tier-3.4).
    # Legacy values ("absolute", "none", "same") are still accepted if explicitly set.
    dj_waypoint_delta_mode: str = "centered"  # "absolute" (legacy) | "centered" (Phase 3)
    dj_waypoint_centered_baseline: str = "median"  # "median" | "mean" (for centered mode)
    dj_waypoint_squash: str = "tanh"  # "none" (hard cap, legacy) | "tanh" (smooth squashing)
    dj_waypoint_squash_alpha: float = 4.0  # Alpha for tanh squashing
    # Phase 3: Coverage enhancements
    dj_coverage_presence_source: str = "raw"  # "same" (legacy) | "raw" (Phase 3 recommended)
    dj_coverage_mode: str = "binary"  # "binary" (0/1 count) | "weighted" (mean weights)
    # Diagnostic-only: per-edge audit table for final emitted playlist
    emit_selected_edge_audit: bool = False
    """Diagnostic-only: when True, log a per-edge audit table for the final
    emitted playlist showing T, T_centered_cos, S, G, bridge_score,
    trans_score_in_beam, progress_t/jump, local_sonic_raw_cos,
    local_sonic_penalty_applied, genre_penalty_applied, below_transition_floor.
    No behavior change."""
    # Soft title-artifact penalty (opt-in; default OFF).
    title_artifact_penalty_enabled: bool = False
    """When True, candidates whose title matches artifact flags
    (demo/live/medley/remix/instrumental/remaster/version/take/edit/outtake/alternate)
    are demoted by the sum of their flag weights. Hard exclusion list
    (title_exclusion_words in candidate_pool) is not affected."""
    title_artifact_penalty_weights: Optional[Dict[str, float]] = None
    """Per-flag penalty magnitudes. None or empty = no penalty.
    Recommended starting values (tune after diagnostics):
    demo:0.10, live:0.05, medley:0.20, remix:0.10, instrumental:0.08,
    version:0.05, take:0.10, outtake:0.15, alternate:0.10"""
    min_edge_objective: str = "total_score"
    """Beam selection objective at the end of each segment:
    'total_score' (default) — pick highest cumulative score (current behavior)
    'min_edge'              — lexicographic (highest min-edge, ties by total)
    Optimizes for 'no broken moments' per Layer 1 principle 5."""
    # Last-mile edge repair fallback (opt-in; default OFF).
    edge_repair_enabled: bool = False
    edge_repair_centered_cos_floor: float = -0.5
    edge_repair_margin: float = 0.05
    edge_repair_variety_guard_enabled: bool = False
    edge_repair_variety_guard_threshold: float = 0.85
    # Total-generation wall-clock budget (seconds). When a shared deadline is
    # threaded in from core.py, this is the default budget used to compute it.
    # 60s leaves a comfortable margin under the 90s hard ceiling for pre-build
    # overhead (pool build, Last.fm recency); the killer cell measured 87.8s
    # at 70s which was only ~2s under the ceiling. Expose as
    # playlists.pier_bridge.generation_budget_s in config.yaml.
    generation_budget_s: float = 60.0


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
        "genre_steering_enabled": bool(tuning.genre_steering_enabled),
        "genre_steering_source": str(getattr(tuning, "genre_steering_source", "dense")),
        "segment_pool_genre_weight": float(getattr(tuning, "segment_pool_genre_weight", 0.0)),
        "weight_genre": float(tuning.weight_genre),
        "genre_arc_floor": float(tuning.genre_arc_floor),
        "genre_arc_floor_percentile": float(tuning.genre_arc_floor_percentile),
        "genre_admission_percentile": float(tuning.genre_admission_percentile),
        "genre_pair_floor": float(getattr(tuning, "genre_pair_floor", 0.0)),
        "genre_pair_penalty": float(getattr(tuning, "genre_pair_penalty", 0.5)),
        "dj_route_shape": str(tuning.dj_route_shape),
        "initial_beam_width": int(getattr(tuning, "initial_beam_width", 20)),
        "max_beam_width": int(getattr(tuning, "max_beam_width", 100)),
        "initial_neighbors_m": int(getattr(tuning, "initial_neighbors_m", 100)),
        "max_neighbors_m": int(getattr(tuning, "max_neighbors_m", 400)),
        "initial_bridge_helpers": int(getattr(tuning, "initial_bridge_helpers", 50)),
        "max_bridge_helpers": int(getattr(tuning, "max_bridge_helpers", 200)),
    }


def _compute_genre_idf(X_genre_raw: np.ndarray, cfg: PierBridgeConfig) -> np.ndarray:
    """Backward-compat wrapper — unpacks PierBridgeConfig to primitives."""
    return _compute_genre_idf_shared(
        X_genre_raw=X_genre_raw,
        power=float(cfg.dj_genre_idf_power),
        norm=str(cfg.dj_genre_idf_norm),
    )


def _compute_transition_score(
    idx_a: int,
    idx_b: int,
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
) -> float:
    """Backward-compat wrapper — unpacks PierBridgeConfig to primitives."""
    return _compute_transition_score_impl(
        idx_a, idx_b, X_full, X_start, X_mid, X_end,
        center_transitions=bool(cfg.center_transitions),
        weight_end_start=float(cfg.weight_end_start),
        weight_mid_mid=float(cfg.weight_mid_mid),
        weight_full_full=float(cfg.weight_full_full),
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
    """Backward-compat wrapper — unpacks PierBridgeConfig to primitives."""
    return _compute_transition_score_raw_and_transformed_impl(
        idx_a, idx_b, X_full, X_start, X_mid, X_end,
        center_transitions=bool(cfg.center_transitions),
        weight_end_start=float(cfg.weight_end_start),
        weight_mid_mid=float(cfg.weight_mid_mid),
        weight_full_full=float(cfg.weight_full_full),
    )
