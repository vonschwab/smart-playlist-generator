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
    # Instrumental lean (soft, continuous on voice_prob; never a hard gate).
    # enabled is per-request (policy override); penalty_weight is static (config.yaml).
    instrumental_enabled: bool = False
    instrumental_penalty_weight: float = 0.0
    # Energy (arousal) steering: soft penalty terms (never hard gates).
    # All default to 0.0 (disabled/no-op); presets enable per-mode.
    energy_step_cap: float = 0.0  # max z-std jump between adjacent tracks (soft cap)
    energy_step_strength: float = 0.0  # strength of step penalty (0 = disabled)
    energy_arc_band: float = 0.0  # z-std target band for segment arc (soft floor/ceiling)
    energy_arc_strength: float = 0.0  # strength of arc penalty (0 = disabled)
    center_transitions: bool = False  # if True, mean-center transition mats and rescale sims to [0,1]
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
    # Calibrated-sigmoid transition rescale params (used when center_transitions=True).
    # Single source of truth: vec._calibrate_transition_cos; fixed constants from
    # the library cosine band (scripts/research/calibrate_transition_sigmoid.py).
    transition_calib_center: float = 0.594
    transition_calib_scale: float = 0.092
    transition_calib_gain: float = 1.0
    # Bridge scoring weights
    weight_bridge: float = 0.6
    weight_transition: float = 0.4
    genre_tiebreak_weight: float = 0.05
    # Soft genre penalty (does not gate candidates): if edge_genre < threshold,
    # multiply the edge score by (1 - strength).
    genre_penalty_threshold: float = 0.20
    genre_penalty_strength: float = 0.10
    # Oops, All Bangers: graded popularity demotion of bridge candidates in the beam.
    # combined_score *= (1 - popularity_penalty_strength * (1 - popularity)); NaN -> max.
    # 0.0 = off / today's behavior. Resolved from popularity_mode (off/on/oops).
    popularity_penalty_strength: float = 0.0
    # Oops, All Bangers admission gate: resolved per popularity_mode (off->None,
    # on->50, oops->10). None = gate disabled. Consumed by core.generate_playlist_ds.
    popularity_rank_cutoff: Optional[int] = None
    # SP2 seed-character anti-collapse scoring (off by default -> byte-identical).
    # "anti_center" penalizes combined_score by how much closer a candidate sits to the
    # local pool center than to its own piers (the within-bridge sag fix; validated as
    # the winner over the retired "hubness" variant via the collapse harness).
    seed_character_mode: str = "off"      # off | anti_center
    seed_character_strength: float = 0.0  # 0 = inert
    # SP3 mini-piers (off by default -> byte-identical). Insert high-character
    # waypoints as extra piers in long bridges so the beam can't sag past them.
    mini_pier_enabled: bool = False
    mini_pier_max_interior: int = 5        # split any segment whose interior exceeds K
    mini_pier_smoothness_margin: float = 0.12
    # Even anchor spacing (live default 2026-07-06): when subdividing at all,
    # equalize the waypoint count across every seed-gap so the seed anchors stay
    # evenly spaced. Without it, W waypoints over M gaps (W not a multiple of M)
    # leave the trailing gap unsplit and the last anchors bunch (4 piers / 30
    # tracks -> gaps 12/12/5; balanced -> 10/10/9). false = rollback (may bunch).
    mini_pier_balance_gaps: bool = True
    # Tail-DP segment endgame (spec 2026-07-02; live default ON). After each
    # segment's beam+var-bridge finalizes segment_path, re-opens the last
    # min(2, interior) slots and exactly maximizes the window min-edge over the
    # segment's own candidate pool (never-worse; falls back to the original tail
    # on any internal error). false = byte-identical to today.
    tail_dp_enabled: bool = True
    tail_dp_epsilon: float = 0.02
    tail_dp_floor: float = 0.30  # weak-landing trigger: re-optimize a segment only
    # when its landing-window min-edge is below this. 0 = always-on. Spec 2026-07-02.
    # RELATIVE trigger (Phase 2 Task 2, spec 2026-07-18): the effective floor
    # becomes max(tail_dp_floor, segment_mean_T - tail_dp_relative_epsilon), so a
    # landing that clears the absolute floor but sits well below its OWN
    # segment's achievable level still re-opens the search. Evidence: Parquet
    # Courts segment 4's worst edge (0.394) cleared tail_dp_floor=0.3 while a
    # ~0.7-0.8-class connector sat unused in the same admitted pool; healthy
    # segments' internal T spread runs ~0.10-0.15, so 0.25 targets "meaningfully
    # below the segment's own level" without over-firing on normal variance. See
    # docs/corridor_baseline/phase2_mechanism_probes.md. 0.0 = legacy
    # absolute-only behavior (rollback).
    # Full legacy disable = relative_epsilon: 0.0 -- tail_dp_floor alone no
    # longer suffices (a non-zero relative_epsilon still keeps an effective
    # floor at segment_mean_T - relative_epsilon even with tail_dp_floor=0).
    tail_dp_relative_epsilon: float = 0.25
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
    # Steering space: "taxonomy" (SP3a graph arc routing + hub-damped similarity —
    # the canonical default; uses in-artifact X_genre_raw, rebuild-robust) | "dense"
    # (legacy PMI-SVD sidecar, opt-in only; build_pier_bridge_playlist raises if its
    # X_genre_dense is unavailable rather than silently steering on nothing).
    genre_steering_source: str = "taxonomy"
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
    # ── Roam corridors (Phase-1, opt-in; default off = identical to legacy) ──
    # Per-dimension soft corridor around an on-manifold kNN-graph reference path
    # between piers. width 0 = no roam allowed (hug the geodesic); larger = wider.
    roam_corridors_enabled: bool = False
    roam_knn_k: int = 25                    # kNN graph degree (corridor width primitive)
    roam_mutual_proximity: bool = True      # hubness-correct the sonic kNN distances
    roam_width_sonic: float = 0.0
    roam_width_genre: float = 0.0
    roam_width_energy: float = 0.0
    roam_penalty_slope: float = 1.0         # soft-penalty steepness beyond the width
    # Min-bottleneck guard: BINARY (lexicographic worst-edge-first), not a blend
    # weight. True => protect the single weakest edge before total score.
    worst_edge_minimax_enabled: bool = False
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
    # Break-glass weak-edge trigger: repair edges with T below this (0 = legacy
    # anti-alignment-only). Aligned with variable_bridge_min_edge. Spec 2026-07-01.
    edge_repair_t_floor: float = 0.30
    # RELATIVE trigger (Phase 2 Task 2, spec 2026-07-18): the effective floor
    # becomes max(edge_repair_t_floor, playlist_mean_T - edge_repair_relative_epsilon),
    # so an edge that clears the absolute floor but sits well below the whole
    # playlist's own achievable level is still considered repair-worthy.
    # Evidence: SADE/home's weakest edge (0.454, the segment's FIRST edge --
    # structurally unreachable by tail-DP) cleared edge_repair_t_floor=0.3 while
    # a fully-admitted 0.697-class connector went unused. Same 0.25 default and
    # rollback semantics as tail_dp_relative_epsilon (0.0 = legacy absolute-only).
    # See docs/corridor_baseline/phase2_mechanism_probes.md.
    # Full legacy disable = relative_epsilon: 0.0 -- edge_repair_t_floor alone
    # no longer suffices (a non-zero relative_epsilon still keeps an effective
    # floor at playlist_mean_T - relative_epsilon even with edge_repair_t_floor=0).
    edge_repair_relative_epsilon: float = 0.25
    edge_repair_margin: float = 0.05
    edge_repair_variety_guard_enabled: bool = False
    edge_repair_variety_guard_threshold: float = 0.85
    # Remove-only last resort (repair-by-deletion): runs AFTER break-glass edge
    # repair. Deletes an interior track only when doing so strictly lifts a
    # still-broken edge (never-worse); never removes a pier/seed. ACTIVATED
    # 2026-07-02 (live default per "activate fixes" discipline); rollback via
    # edge_delete_enabled: false. Shared floor = 0.30, aligned with
    # variable_bridge_min_edge / tail_dp_floor / edge_repair_t_floor. See
    # docs/superpowers/plans/2026-07-02-weak-edge-cascade-reorder.md.
    edge_delete_enabled: bool = True
    edge_delete_floor: float = 0.30
    edge_delete_max_deletions: int = 4
    # Total-generation wall-clock budget (seconds). When a shared deadline is
    # threaded in from core.py, this is the default budget used to compute it.
    # 60s leaves a comfortable margin under the 90s hard ceiling for pre-build
    # overhead (pool build, Last.fm recency); the killer cell measured 87.8s
    # at 70s which was only ~2s under the ceiling. Expose as
    # playlists.pier_bridge.generation_budget_s in config.yaml.
    generation_budget_s: float = 60.0
    # --- Variable bridge length (ACTIVATED 2026-06-28 via config.yaml after the
    # worst-edge gate + audition passed). Each segment may flex its interior length
    # to land more smoothly on the next pier; lengths reallocate within a soft total
    # band. The LIVE default is ON (config.yaml / config.example.yaml set true); this
    # in-code default stays False as the rollback fallback + to keep test baselines on
    # the even-split path. False => even split. ---
    variable_bridge_length: bool = False
    variable_bridge_flex: int = 2          # k: +/- interior tracks a segment may flex
    variable_bridge_min_edge: float = 0.30  # only flex a segment whose nominal worst edge is below this
    variable_bridge_epsilon: float = 0.02   # prefer nominal length unless a flex beats it by > eps
    variable_bridge_max_flex_segments: int = 3  # max segments that may actually flex (deterministic cap)

    # ── Corridor segment pooling (Phase 1) ──
    # Corridor pooling (a library-wide eligible universe --
    # pier_bridge.eligible_universe.build_eligible_universe -- narrowed per
    # segment by a self-calibrating min-sim corridor --
    # pier_bridge.corridor.build_corridor) is THE segment-pool strategy since
    # Phase 1 Task 8. The "legacy"/"corridor" dev flag (`pooling: str`) and
    # the legacy KNN-union / segment-scored pool builders it selected between
    # were deleted in that task -- see
    # docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md and
    # .superpowers/sdd/p1-task-8-report.md.
    corridor_width_percentile: Optional[float] = None  # TUNING ESCAPE HATCH ONLY.
    # None (default) = "use the sonic_mode -> width mapping below"; an explicit
    # numeric value here WINS OVER the mode mapping unconditionally (same
    # override discipline as every other per-mode knob in this codebase). Set
    # this only for one-off tuning/debugging -- normal runs should let
    # sonic_mode drive the width via corridor_width_percentile_<mode> below.
    #
    # HISTORY: through the width re-pin (.superpowers/sdd/p1-width-repin-report.md),
    # this field WAS the sole width knob (flat default 0.95, no mode axis) --
    # Task 8's restrict_bundle fix (13256f1) had widened the corridor universe
    # from Artist mode's old amputated universe (a few thousand tracks) to the
    # full ~43k-track library, and 0.95 was the best single value found by
    # probing 4 open-mode artists against min_T recovery. That probe's own
    # concerns section flagged the real problem: a single global percentile
    # cannot serve both strict and dynamic sonic modes over the same much-
    # bigger universe (SADE/home cratered to 0.374 at 0.95 vs legacy 0.706,
    # while open cells were fine) -- recommending a per-sonic_mode width as
    # the real fix. This task (spec section 4, pulled forward from Phase 2 by
    # Dylan's 2026-07-18 decision) is that fix: see
    # corridor_width_percentile_strict/narrow/dynamic/discover below and
    # ``pier_bridge.corridor.resolve_corridor_width_percentile`` (the pure
    # resolver; "off" sonic_mode hardcodes to 0.0 = whole universe, no field
    # needed). ``corridor_width_percentile_dynamic``'s default carries
    # forward the 0.95 pin (dynamic is the mode the old probe actually
    # calibrated against -- policy.py's "open" detent is sonic_mode=dynamic).
    #
    # CALIBRATION (.superpowers/sdd/p1-permode-width-report.md): strict probed
    # {0.985, 0.99, 0.995} on the 4 home cells (BET/SADE/Aaliyah/Swirlies);
    # 0.985 gave the smallest mean |min_T delta| vs legacy (0.218 vs 0.230 at
    # 0.99, 0.263 at 0.995) while keeping Swirlies/home below_floor=0 at every
    # tested width -- picked per the brief's own min_T-recovery-primary rule.
    # dynamic probed {0.95, 0.97} on the 4 open cells (SADE/Aaliyah/AlexG/
    # Strokes); 0.95 gave the smaller mean |min_T delta| (0.114 vs 0.135 at
    # 0.97) AND matches the pre-existing flat pin (continuity with history) --
    # kept at 0.95.
    #
    # narrow/discover CALIBRATION (Phase 2 Task 4, .superpowers/sdd/
    # p2-task-4-report.md): the midpoint(strict,dynamic)=0.9675 and
    # dynamic-0.02=0.93 interpolations above were PROVISIONAL until this task
    # -- superseded by a direct probe (3 artists: SADE/Swirlies/Alex G, via the
    # GUI's close/wander detents so genre_mode moves with sonic_mode, matching
    # real usage) bracketing each interpolation with 3 candidate widths.
    # narrow {0.96, 0.9675, 0.975}: 0.975 won on BOTH mean|min_T| (0.6164 vs
    # 0.9675's 0.6144, 0.96's 0.6102) AND worst-case min_T across the 3 artists
    # (0.6005 vs 0.9675's 0.5939) -- below_floor=0 at every width. Pinned
    # 0.975 (changed from the 0.9675 interpolation).
    # discover {0.92, 0.93, 0.94}: 0.94 won decisively on both mean (0.6290 vs
    # 0.6044 tied at 0.92/0.93) and worst-case (0.6228 vs 0.5490, Alex G/wander
    # specifically) -- below_floor=0 at every width. Pinned 0.94 (changed from
    # the 0.93 interpolation). Two of three probe artists (Swirlies, Alex G)
    # hit segment_pool_max=800 post-cap at every tested discover width (the
    # SAME cap-saturation pattern the Phase 1 per-mode-width report's dial
    # audit found for open/wander) -- the win traces to which candidates enter
    # the pre-cap top-800 ranking at the tighter 0.94 threshold, not to size
    # differentiation; SADE (not cap-bound, sizes 566->495->424 across
    # 0.92->0.93->0.94) stayed flat across the whole bracket, a legitimate
    # beam-convergence saturation (its winning sequence's candidates all
    # survive well within the tightest tested width). Full probe data + mini-
    # corpus regression check: docs/corridor_baseline/phase2_task4_width_calibration.md.
    corridor_width_percentile_strict: float = 0.985
    corridor_width_percentile_narrow: float = 0.975
    corridor_width_percentile_dynamic: float = 0.95
    corridor_width_percentile_discover: float = 0.94
    # NOTE: sonic_mode "off" is NOT a field here -- resolve_corridor_width_percentile
    # hardcodes it to 0.0 (whole eligible universe, no sonic narrowing at all),
    # matching spec section 4 ("`off` = universe") exactly the way genre_mode
    # "off" disables the relevance mask rather than resolving to a number.
    corridor_widen_step: float = 0.05        # unused until Task 4's widening ladder
    corridor_widen_attempts: int = 2         # unused until Task 4's widening ladder
    # Task 6 remediation, iteration 2: empirical continue-gate — widening
    # continues only while it demonstrably helps; replaces the falsified
    # support-threshold predictor, see p1-task6-remediation-report.md.
    #
    # Iteration 1 tried a PREDICTIVE gate (skip widening when anchor-support
    # coverage was already healthy, on the theory that a weak edge with a
    # healthy pool must be beam-path-internal). Falsified by real evidence:
    # Alex G/home's segment 1 had support ~0.8 (comfortably "healthy") yet
    # still gained +0.42 T from one widen attempt (0.189 -> 0.611) -- a wider
    # pool can unlock better interior-to-interior beam combinations no
    # anchor-only metric can predict. Retired; corridor_widen_support_threshold
    # never shipped past dev.
    #
    # Iteration 2 replaces prediction with observation: the ladder always
    # tries widen attempt 1 unconditionally (no gate on the first attempt --
    # the trigger firing is signal enough). After that, it widens FURTHER
    # only if the attempt just run improved the best-seen min_edge_T by more
    # than this epsilon versus the best-seen value from strictly before that
    # attempt; a non-improving (or worsening) attempt means further widening
    # is empirically not paying for itself, so the ladder stops and hands the
    # segment to the repair stack. Hard infeasibility (no path found) always
    # widens to the full attempt budget regardless of this knob -- see
    # src.playlist.pier_bridge.corridor.corridor_widen_decision.
    corridor_widen_improvement_epsilon: float = 0.02

    # ── C1 duration soft-penalty + title-hygiene ON-case (Task 7 fix) ──
    # Corridor-pooling-ONLY seam: build_eligible_universe's corridor call site
    # (pier_bridge_builder.py) reads these to wire the duration penalty's ON
    # case and title hard-exclusion. core.py's generate_playlist_ds populates
    # them from cfg.candidate (the SAME CandidatePoolConfig fields the legacy
    # pool reads: candidate_pool.duration_penalty_enabled/_weight/
    # _cutoff_multiplier/title_hard_exclude_flags in config.yaml) via
    # `replace(pb_cfg, ...)`, mirroring how pace_bridge_floor/bpm_stability_min
    # are already threaded there. Legacy (candidate_pool.py) never reads these
    # PierBridgeConfig fields -- it reads cfg.candidate directly -- so this
    # addition backs the corridor-pooling seam (the sole pooling path).
    # Defaults keep the dead-knob-trap regression (Task 3,
    # test_corridor_universe_duration_reference_is_none_dead_knob_trap) green
    # for any PierBridgeConfig built without this threading (e.g. direct
    # unit-test construction): duration_penalty_enabled=False ->
    # duration_reference_ms=None, duration_penalty_weight=0.0 at the call site.
    duration_penalty_enabled: bool = False
    duration_penalty_weight: float = 0.0
    duration_cutoff_multiplier: float = 2.5
    # tuple, NOT frozenset: core.py's generate_playlist_ds (ds_pipeline_runner.py:211)
    # unconditionally `json.dumps(pb_cfg.__dict__)`-logs the effective pier-bridge
    # config on every generation (enable_logging: true is the config.yaml default,
    # legacy AND corridor). A frozenset field there raises
    # "TypeError: Object of type frozenset is not JSON serializable" on EVERY
    # generation, not just corridor -- caught via the corridor_baseline sweep
    # harness during this task. tuple is JSON-safe (serializes as an array) and
    # still immutable/hashable; the corridor call site (pier_bridge_builder.py)
    # wraps it in frozenset(...) only at the point build_eligible_universe needs it.
    title_hard_exclude_flags: tuple[str, ...] = ()


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
        "genre_steering_source": str(getattr(tuning, "genre_steering_source", "taxonomy")),
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
        calib_center=float(cfg.transition_calib_center),
        calib_scale=float(cfg.transition_calib_scale),
        calib_gain=float(cfg.transition_calib_gain),
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
        calib_center=float(cfg.transition_calib_center),
        calib_scale=float(cfg.transition_calib_scale),
        calib_gain=float(cfg.transition_calib_gain),
    )


def roam_kwargs_from_dict(roam_raw: Optional[dict]) -> Dict[str, Any]:
    """Parse a user/config roam override dict into PierBridgeConfig kwargs.

    Shared by the seeds/DS override path (apply_pier_bridge_overrides) and the
    artist path (which builds PierBridgeConfig explicitly, so it must apply roam
    itself). Absent/empty => {}, so ``replace(cfg, **roam_kwargs_from_dict(None))``
    is a safe no-op. Keys: enabled, knn_k, mutual_proximity,
    width_sonic/genre/energy, penalty_slope, worst_edge_minimax.
    """
    if not isinstance(roam_raw, dict):
        return {}
    out: Dict[str, Any] = {}
    if isinstance(roam_raw.get("enabled"), bool):
        out["roam_corridors_enabled"] = bool(roam_raw["enabled"])
    k = roam_raw.get("knn_k")
    if isinstance(k, int) and not isinstance(k, bool):
        out["roam_knn_k"] = int(k)
    if isinstance(roam_raw.get("mutual_proximity"), bool):
        out["roam_mutual_proximity"] = bool(roam_raw["mutual_proximity"])
    for _dim in ("sonic", "genre", "energy"):
        v = roam_raw.get(f"width_{_dim}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"roam_width_{_dim}"] = float(v)
    ps = roam_raw.get("penalty_slope")
    if isinstance(ps, (int, float)) and not isinstance(ps, bool):
        out["roam_penalty_slope"] = float(ps)
    if isinstance(roam_raw.get("worst_edge_minimax"), bool):
        out["worst_edge_minimax_enabled"] = bool(roam_raw["worst_edge_minimax"])
    return out
