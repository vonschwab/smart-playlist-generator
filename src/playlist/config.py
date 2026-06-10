from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import math

Mode = Literal["strict", "narrow", "dynamic", "discover"]
AlphaSchedule = Literal["constant", "arc"]
RepairObjective = Literal["gap_penalty", "below_floor_first"]

_logger = logging.getLogger(__name__)


def resolve_cohesion_mode(playlists_cfg: Optional[dict]) -> "Mode":
    """
    Read playlists.cohesion_mode with validation.

    Sole reader of the cohesion_mode key. Warns (and ignores) if the legacy
    ds_pipeline.mode key is present so stale configs surface immediately.
    """
    if not isinstance(playlists_cfg, dict):
        return "dynamic"

    ds_pipeline = playlists_cfg.get("ds_pipeline")
    if isinstance(ds_pipeline, dict) and "mode" in ds_pipeline:
        _logger.warning(
            "playlists.ds_pipeline.mode is no longer used; remove from config. "
            "Use playlists.cohesion_mode instead."
        )

    raw = str(playlists_cfg.get("cohesion_mode", "dynamic")).strip().lower()
    if raw not in {"strict", "narrow", "dynamic", "discover"}:
        _logger.warning("Invalid cohesion_mode %r; falling back to 'dynamic'", raw)
        return "dynamic"
    return raw  # type: ignore[return-value]


@dataclass(frozen=True)
class CandidatePoolConfig:
    similarity_floor: float
    min_sonic_similarity: Optional[float]
    max_pool_size: int
    target_artists: int
    candidates_per_artist: int
    seed_artist_bonus: int
    max_artist_fraction_final: float
    duration_penalty_enabled: bool = True
    duration_penalty_weight: float = 0.6
    duration_cutoff_multiplier: float = 2.5
    title_exclusion_enabled: bool = True
    title_exclusion_words: tuple[str, ...] = ("interlude", "skit")
    broad_filters: tuple[str, ...] = ()
    genre_compatibility_enabled: bool = False
    genre_compatibility_penalty_strength: float = 0.0
    genre_compatibility_compatible_threshold: float = 0.35
    genre_compatibility_conflict_threshold: float = 0.15
    title_hard_exclude_flags: frozenset[str] = frozenset({"interlude", "skit", "acapella"})
    genre_idf_enabled: bool = True
    pace_admission_floor: float = 0.0
    pace_bridge_floor: float = 0.0
    bpm_admission_max_log_distance: float = float("inf")  # inf = disabled
    bpm_stability_min: float = 0.5  # tracks below this skip BPM gate


@dataclass(frozen=True)
class ConstructionConfig:
    local_top_m: int
    alpha_schedule: AlphaSchedule
    alpha: float
    alpha_start: float
    alpha_mid: float
    alpha_end: float
    arc_midpoint: float
    beta: float
    gamma: float
    hard_floor: bool
    transition_gamma: float
    transition_floor: float
    min_gap: int
    max_artist_fraction_final: float
    center_transitions: bool


@dataclass(frozen=True)
class RepairConfig:
    enabled: bool
    objective: RepairObjective
    max_iters: int
    max_edges: int
    allow_substitute_next: bool
    allow_substitute_prev: bool


@dataclass(frozen=True)
class DSPipelineConfig:
    mode: Mode
    candidate: CandidatePoolConfig
    construct: ConstructionConfig
    repair: RepairConfig


@dataclass(frozen=True)
class PierBridgeTuning:
    """Resolved pier-bridge tuning knobs (per cohesion_mode)."""

    transition_floor: float
    bridge_floor: float
    weight_bridge: float
    weight_transition: float
    genre_tiebreak_weight: float
    genre_penalty_threshold: float
    genre_penalty_strength: float
    genre_steering_enabled: bool = False
    genre_steering_source: str = "dense"
    weight_genre: float = 0.0
    genre_arc_floor: float = 0.0
    genre_arc_floor_percentile: float = 0.0
    genre_admission_percentile: float = 0.0
    segment_pool_genre_weight: float = 0.0
    dj_route_shape: str = "linear"


def _resolve_mode_number_with_source(
    cfg: Any,
    key: str,
    mode: str,
    default: float,
    *,
    source_prefix: str,
) -> tuple[float, str]:
    """
    Resolve a numeric config value that may be:
      - per-mode scalar via `<key>_<mode>`
      - scalar via `<key>`
      - mapping via `<key>: {dynamic: ..., narrow: ..., default: ...}`
    """
    if not isinstance(cfg, dict):
        return float(default), "default"

    mode = str(mode).strip().lower()

    mode_specific = cfg.get(f"{key}_{mode}")
    if isinstance(mode_specific, (int, float)):
        return float(mode_specific), f"{source_prefix}.{key}_{mode}"

    raw = cfg.get(key)
    if isinstance(raw, (int, float)):
        return float(raw), f"{source_prefix}.{key}"

    if isinstance(raw, dict):
        raw_mode = raw.get(mode)
        if isinstance(raw_mode, (int, float)):
            return float(raw_mode), f"{source_prefix}.{key}.{mode}"

        raw_default = raw.get("default")
        if isinstance(raw_default, (int, float)):
            return float(raw_default), f"{source_prefix}.{key}.default"

    return float(default), "default"


def _resolve_transition_floor_with_source(
    *,
    mode: Mode,
    similarity_floor: float,
    constraints: Any,
) -> tuple[float, str]:
    """
    Resolve transition floor.

    Priority:
      1) constraints.transition_floor_<mode>
      2) constraints.transition_floor (scalar OR mapping with {dynamic,narrow})
      3) Built-in defaults: dynamic=0.35, narrow=0.45, else similarity_floor
    """
    mode_s = str(mode).strip().lower()
    default = float({"dynamic": 0.35, "narrow": 0.45}.get(mode_s, similarity_floor))

    if not isinstance(constraints, dict):
        return default, "default"

    mode_specific = constraints.get(f"transition_floor_{mode_s}")
    if isinstance(mode_specific, (int, float)):
        return float(mode_specific), f"constraints.transition_floor_{mode_s}"

    raw = constraints.get("transition_floor")
    if isinstance(raw, (int, float)):
        return float(raw), "constraints.transition_floor"

    if isinstance(raw, dict):
        raw_mode = raw.get(mode_s)
        if isinstance(raw_mode, (int, float)):
            return float(raw_mode), f"constraints.transition_floor.{mode_s}"

        raw_default = raw.get("default")
        if isinstance(raw_default, (int, float)):
            return float(raw_default), "constraints.transition_floor.default"

    return default, "default"


def resolve_pier_bridge_tuning(
    *,
    mode: Mode,
    similarity_floor: float,
    overrides: Optional[dict] = None,
) -> tuple[PierBridgeTuning, dict[str, str]]:
    """
    Resolve pier-bridge tuning for the given cohesion_mode.

    Sources:
      - overrides['constraints'] for transition_floor overrides
      - overrides['pier_bridge'] for bridge floor / weights / soft genre penalty knobs
    """
    mode_s = str(mode).strip().lower()
    if overrides is None:
        overrides = {}

    constraints = overrides.get("constraints", {})
    pier_raw = overrides.get("pier_bridge", {}) or {}
    if not isinstance(pier_raw, dict):
        pier_raw = {}

    sources: dict[str, str] = {}

    transition_floor, src = _resolve_transition_floor_with_source(
        mode=mode,
        similarity_floor=float(similarity_floor),
        constraints=constraints,
    )
    sources["transition_floor"] = src

    # Bridge floor defaults (Phase 3A relaxation)
    if mode_s == "strict":
        default_bridge_floor = 0.10
    elif mode_s == "narrow":
        default_bridge_floor = 0.05  # Relaxed from 0.08
    elif mode_s == "dynamic":
        default_bridge_floor = 0.02  # Relaxed from 0.03
    else:
        default_bridge_floor = 0.0

    # Weight defaults
    if mode_s == "strict":
        default_weight_bridge = 0.7
        default_weight_transition = 0.3
    elif mode_s == "narrow":
        default_weight_bridge = 0.7
        default_weight_transition = 0.3
    elif mode_s == "dynamic":
        default_weight_bridge = 0.6
        default_weight_transition = 0.4
    else:
        default_weight_bridge = 0.0
        default_weight_transition = 1.0

    # Optional nested weights block:
    # weights: {bridge, transition} OR {dynamic: {bridge, transition}, narrow: {...}}
    weight_bridge = default_weight_bridge
    weight_transition = default_weight_transition
    weights_raw = pier_raw.get("weights")
    if isinstance(weights_raw, dict):
        by_mode = weights_raw.get(mode_s)
        if isinstance(by_mode, dict):
            if isinstance(by_mode.get("bridge"), (int, float)):
                weight_bridge = float(by_mode.get("bridge", default_weight_bridge))
                sources["weight_bridge"] = f"pier_bridge.weights.{mode_s}.bridge"
            if isinstance(by_mode.get("transition"), (int, float)):
                weight_transition = float(by_mode.get("transition", default_weight_transition))
                sources["weight_transition"] = f"pier_bridge.weights.{mode_s}.transition"
        else:
            if isinstance(weights_raw.get("bridge"), (int, float)):
                weight_bridge = float(weights_raw.get("bridge", default_weight_bridge))
                sources["weight_bridge"] = "pier_bridge.weights.bridge"
            if isinstance(weights_raw.get("transition"), (int, float)):
                weight_transition = float(weights_raw.get("transition", default_weight_transition))
                sources["weight_transition"] = "pier_bridge.weights.transition"

    if "weight_bridge" not in sources:
        weight_bridge, src = _resolve_mode_number_with_source(
            pier_raw, "weight_bridge", mode_s, default_weight_bridge, source_prefix="pier_bridge"
        )
        sources["weight_bridge"] = src

    if "weight_transition" not in sources:
        weight_transition, src = _resolve_mode_number_with_source(
            pier_raw, "weight_transition", mode_s, default_weight_transition, source_prefix="pier_bridge"
        )
        sources["weight_transition"] = src

    bridge_floor, src = _resolve_mode_number_with_source(
        pier_raw, "bridge_floor", mode_s, default_bridge_floor, source_prefix="pier_bridge"
    )
    sources["bridge_floor"] = src

    genre_tiebreak_weight, src = _resolve_mode_number_with_source(
        pier_raw, "genre_tiebreak_weight", mode_s, 0.05, source_prefix="pier_bridge"
    )
    sources["genre_tiebreak_weight"] = src

    genre_penalty_threshold, src = _resolve_mode_number_with_source(
        pier_raw, "soft_genre_penalty_threshold", mode_s, 0.20, source_prefix="pier_bridge"
    )
    sources["genre_penalty_threshold"] = src

    genre_penalty_strength, src = _resolve_mode_number_with_source(
        pier_raw, "soft_genre_penalty_strength", mode_s, 0.10, source_prefix="pier_bridge"
    )
    sources["genre_penalty_strength"] = src

    if not math.isfinite(float(genre_penalty_strength)):
        genre_penalty_strength = 0.0
    genre_penalty_strength = float(max(0.0, min(1.0, float(genre_penalty_strength))))

    genre_steering_enabled = bool(pier_raw.get("genre_steering_enabled", False))
    genre_steering_source = str(pier_raw.get("genre_steering_source", "dense")).strip().lower()
    if genre_steering_source not in {"dense", "taxonomy"}:
        genre_steering_source = "dense"
    sources["genre_steering_source"] = (
        "pier_bridge.genre_steering_source"
        if "genre_steering_source" in pier_raw
        else "default"
    )
    weight_genre, src = _resolve_mode_number_with_source(
        pier_raw, "weight_genre", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["weight_genre"] = src
    genre_arc_floor, src = _resolve_mode_number_with_source(
        pier_raw, "genre_arc_floor", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["genre_arc_floor"] = src
    genre_arc_floor_percentile, src = _resolve_mode_number_with_source(
        pier_raw, "genre_arc_floor_percentile", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["genre_arc_floor_percentile"] = src
    genre_admission_percentile, src = _resolve_mode_number_with_source(
        pier_raw, "genre_admission_percentile", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["genre_admission_percentile"] = src
    segment_pool_genre_weight_raw = pier_raw.get("segment_pool_genre_weight", 0.0)
    segment_pool_genre_weight = float(segment_pool_genre_weight_raw) if isinstance(segment_pool_genre_weight_raw, (int, float)) else 0.0
    segment_pool_genre_weight = max(0.0, min(1.0, segment_pool_genre_weight))
    sources["segment_pool_genre_weight"] = "pier_bridge.segment_pool_genre_weight" if "segment_pool_genre_weight" in pier_raw else "default"
    dj_route_shape_raw = pier_raw.get("dj_route_shape", "linear")
    dj_route_shape = str(dj_route_shape_raw).strip().lower() if dj_route_shape_raw else "linear"
    sources["dj_route_shape"] = "pier_bridge.dj_route_shape" if "dj_route_shape" in pier_raw else "default"

    # When steering is active, genre is a co-equal edge weight: renormalize the
    # (bridge, transition, genre) triple to sum to 1 so the score stays in range.
    if genre_steering_enabled and float(weight_genre) > 0.0:
        _wsum = float(weight_bridge) + float(weight_transition) + float(weight_genre)
        if _wsum > 0:
            weight_bridge = float(weight_bridge) / _wsum
            weight_transition = float(weight_transition) / _wsum
            weight_genre = float(weight_genre) / _wsum

    tuning = PierBridgeTuning(
        transition_floor=float(transition_floor),
        bridge_floor=float(bridge_floor),
        weight_bridge=float(weight_bridge),
        weight_transition=float(weight_transition),
        genre_tiebreak_weight=float(genre_tiebreak_weight),
        genre_penalty_threshold=float(genre_penalty_threshold),
        genre_penalty_strength=float(genre_penalty_strength),
        genre_steering_enabled=bool(genre_steering_enabled),
        genre_steering_source=str(genre_steering_source),
        weight_genre=float(weight_genre),
        genre_arc_floor=float(genre_arc_floor),
        genre_arc_floor_percentile=float(genre_arc_floor_percentile),
        genre_admission_percentile=float(genre_admission_percentile),
        segment_pool_genre_weight=float(segment_pool_genre_weight),
        dj_route_shape=str(dj_route_shape),
    )
    return tuning, sources


def get_min_sonic_similarity(candidate_pool_cfg: dict, mode: Mode) -> Optional[float]:
    """
    Resolve the sonic similarity floor for the given mode from config.

    Single writer for this setting is apply_mode_presets() (driven by sonic_mode).
    Returns None when nothing is set — apply_mode_presets writes a value in
    the normal config-loading path.

    Priority:
    1) min_sonic_similarity_<mode> (per-mode override)
    2) min_sonic_similarity (base override applied to all modes)
    3) None (no per-mode default; apply_mode_presets is responsible)
    """
    mode = mode.lower()  # type: ignore[assignment]
    mode_key = f"min_sonic_similarity_{mode}"
    if mode_key in candidate_pool_cfg and candidate_pool_cfg.get(mode_key) is None:
        return None
    if "min_sonic_similarity" in candidate_pool_cfg and candidate_pool_cfg.get("min_sonic_similarity") is None:
        return None
    mode_specific = candidate_pool_cfg.get(mode_key)
    base = candidate_pool_cfg.get("min_sonic_similarity")
    resolved = mode_specific if mode_specific is not None else base
    return float(resolved) if resolved is not None else None


def default_ds_config(
    mode: Mode,
    *,
    playlist_len: int,
    overrides: Optional[dict] = None,
) -> DSPipelineConfig:
    """
    Return mode defaults matching experiments:
    - max_artist_fraction defaults to 0.125 for all modes; per-mode overrides are
      applied by policy.py (the caller), not here
    - min_gap: strict(3), narrow(3), dynamic(6), discover(9)
    - floors roughly: strict ~0.40, narrow ~0.35, dynamic ~0.28, discover ~0.22
    - hard_floor True only for dynamic
    - pick sensible defaults for top_m and alpha/beta/gamma aligned with current experiments
    Compute max_per_artist_final = ceil(L * max_artist_fraction_final) later in code.

    Args:
        mode: One of "strict", "narrow", "dynamic", "discover"
        playlist_len: Target playlist length
        overrides: Optional dict from config.yaml with keys like 'scoring', 'constraints', etc.
    """
    mode = mode.lower()  # type: ignore[assignment]
    if mode not in {"strict", "narrow", "dynamic", "discover"}:
        raise ValueError(f"Unsupported mode {mode}")

    # Parse overrides from config.yaml
    if overrides is None:
        overrides = {}
    scoring = overrides.get("scoring", {})
    constraints = overrides.get("constraints", {})
    candidate_pool = overrides.get("candidate_pool", {})
    repair = overrides.get("repair", {})
    pace_mode_name = candidate_pool.get("pace_mode") or overrides.get("pace_mode") or "dynamic"
    from src.playlist.mode_presets import resolve_pace_mode
    pace_settings = resolve_pace_mode(str(pace_mode_name))

    # Mode defaults - can be overridden by config.yaml values
    max_artist_fraction_final = candidate_pool.get("max_artist_fraction", 0.125)
    min_gap = constraints.get(
        "min_gap",
        {"strict": 3, "narrow": 3, "dynamic": 6, "discover": 9}[mode],
    )
    similarity_floor = candidate_pool.get(
        "similarity_floor",
        # Shift each mode narrower: higher floors for tighter cohesion.
        {"strict": 0.40, "narrow": 0.35, "dynamic": 0.28, "discover": 0.22}[mode],
    )
    min_sonic_similarity = get_min_sonic_similarity(candidate_pool, mode)
    broad_filters_cfg_raw = candidate_pool.get("broad_filters", None)
    if broad_filters_cfg_raw is None:
        broad_filters_cfg: list[str] = []
    elif isinstance(broad_filters_cfg_raw, str):
        broad_filters_cfg = [broad_filters_cfg_raw]
    elif isinstance(broad_filters_cfg_raw, (list, tuple)):
        broad_filters_cfg = [str(b) for b in broad_filters_cfg_raw]
    else:
        try:
            broad_filters_cfg = [str(b) for b in list(broad_filters_cfg_raw)]
        except Exception:
            broad_filters_cfg = []
    pier_tuning, _ = resolve_pier_bridge_tuning(
        mode=mode,
        similarity_floor=float(similarity_floor),
        overrides=overrides,
    )

    # Candidate pool knobs
    target_artists = {
        "strict": max(int((playlist_len + 1) // 3), 10),  # Ultra-cohesive: fewer artists
        "narrow": max(int((playlist_len + 1) // 2), 12),
        "dynamic": max(int(round(0.75 * playlist_len)), 16),
        "discover": min(playlist_len, 24),
    }[mode]

    seed_artist_bonus = 2
    # max_per_artist_final computed downstream, but candidate_per_artist mirrors experiments
    def _candidate_per_artist(max_per_artist_final: int) -> int:
        if mode == "strict":
            return max(3, min(2 * max_per_artist_final, 10))  # Ultra-cohesive
        if mode == "narrow":
            return max(3, min(2 * max_per_artist_final, 8))
        if mode == "dynamic":
            return max(3, min(2 * max_per_artist_final, 6))
        return max(2, min(2 * max_per_artist_final, 4))

    # Temporary max_per_artist estimate using playlist_len
    from math import ceil

    max_per_artist_final_est = ceil(playlist_len * max_artist_fraction_final)
    candidate_cfg = CandidatePoolConfig(
        similarity_floor=similarity_floor,
        min_sonic_similarity=min_sonic_similarity,
        max_pool_size=candidate_pool.get(
            "max_pool_size",
            {"strict": 600, "narrow": 800, "dynamic": 1200, "discover": 2000}[mode],
        ),
        target_artists=target_artists,
        candidates_per_artist=_candidate_per_artist(max_per_artist_final_est),
        seed_artist_bonus=seed_artist_bonus,
        max_artist_fraction_final=max_artist_fraction_final,
        duration_penalty_enabled=candidate_pool.get("duration_penalty_enabled", True),
        duration_penalty_weight=float(
            candidate_pool.get("duration_penalty_weight", 0.6)
        ),
        duration_cutoff_multiplier=float(
            candidate_pool.get("duration_cutoff_multiplier", 2.5)
        ),
        title_exclusion_enabled=bool(candidate_pool.get("title_exclusion_enabled", True)),
        title_exclusion_words=tuple(
            str(word).strip().lower()
            for word in candidate_pool.get("title_exclusion_words", ["interlude", "skit"])
            if str(word).strip()
        ),
        broad_filters=tuple(str(b).lower() for b in broad_filters_cfg),
        genre_compatibility_enabled=bool(candidate_pool.get("genre_compatibility_enabled", False)),
        genre_compatibility_penalty_strength=float(
            candidate_pool.get("genre_compatibility_penalty_strength", 0.0)
        ),
        genre_compatibility_compatible_threshold=float(
            candidate_pool.get("genre_compatibility_compatible_threshold", 0.35)
        ),
        genre_compatibility_conflict_threshold=float(
            candidate_pool.get("genre_compatibility_conflict_threshold", 0.15)
        ),
        title_hard_exclude_flags=frozenset(
            str(f).strip().lower()
            for f in (candidate_pool.get("title_hard_exclude_flags", ["interlude", "skit", "acapella"]) or [])
            if str(f).strip()
        ),
        genre_idf_enabled=bool(
            candidate_pool.get(
                "genre_idf_enabled",
                # discover turns IDF off — exploration mode shouldn't reward narrow tag matches
                mode != "discover",
            )
        ),
        pace_admission_floor=float(
            candidate_pool.get("pace_admission_floor", pace_settings["admission_floor"])
        ),
        pace_bridge_floor=float(
            candidate_pool.get("pace_bridge_floor", pace_settings["bridge_floor"])
        ),
        bpm_admission_max_log_distance=float(
            candidate_pool.get(
                "bpm_admission_max_log_distance",
                pace_settings["bpm_admission_max_log_distance"],
            )
        ),
        bpm_stability_min=float(candidate_pool.get("bpm_stability_min", 0.5)),
    )

    # Construction config with config.yaml overrides
    mode_alpha_schedule = "constant" if mode in ("strict", "narrow") else "arc"
    construct_cfg = ConstructionConfig(
        local_top_m=25 if mode not in ("strict", "narrow") else 15,
        alpha_schedule=scoring.get("alpha_schedule", mode_alpha_schedule),
        alpha=scoring.get(
            "alpha",
            # Lean more on seed similarity for all modes (narrowest gets highest).
            {"strict": 0.75, "narrow": 0.70, "dynamic": 0.62, "discover": 0.50}[mode],
        ),
        alpha_start=scoring.get(
            "alpha_start",
            {"strict": 0.75, "narrow": 0.70, "dynamic": 0.65, "discover": 0.55}[mode],
        ),
        alpha_mid=scoring.get(
            "alpha_mid",
            {"strict": 0.75, "narrow": 0.70, "dynamic": 0.50, "discover": 0.35}[mode],
        ),
        alpha_end=scoring.get(
            "alpha_end",
            {"strict": 0.75, "narrow": 0.70, "dynamic": 0.62, "discover": 0.50}[mode],
        ),
        arc_midpoint=scoring.get("arc_midpoint", 0.55),
        beta=scoring.get(
            "beta",
            {"strict": 0.50, "narrow": 0.55, "dynamic": 0.58, "discover": 0.62}[mode],
        ),
        gamma=scoring.get(
            "gamma",
            # Dial back diversity bonus to favor tighter cohesion.
            {"strict": 0.005, "narrow": 0.01, "dynamic": 0.025, "discover": 0.06}[mode],
        ),
        hard_floor=constraints.get(
            "hard_floor",
            True if mode == "dynamic" else False,
        ),
        transition_gamma=1.0,
        transition_floor=float(pier_tuning.transition_floor),
        min_gap=min_gap,
        max_artist_fraction_final=max_artist_fraction_final,
        center_transitions=constraints.get("center_transitions", False),
    )

    # Repair config with config.yaml overrides
    repair_cfg = RepairConfig(
        enabled=repair.get("enabled", True),
        objective=repair.get("objective", "gap_penalty"),
        max_iters=repair.get("max_iters", 5 if mode != "discover" else 8),
        max_edges=repair.get("max_edges", 5 if mode != "discover" else 8),
        allow_substitute_next=True,
        allow_substitute_prev=True,
    )

    # Log resolved threshold values for diagnostic purposes (Phase 2A/2B/3A implementation)
    _logger.info(
        "DS Pipeline Resolved Thresholds: mode=%s | "
        "sonic_floor=%s | genre_floor=N/A (see genre_similarity) | "
        "bridge_floor=%.3f | transition_floor=%.3f | "
        "similarity_floor=%.3f | min_gap=%d",
        mode,
        f"{min_sonic_similarity:.3f}" if min_sonic_similarity is not None else "None",
        float(pier_tuning.bridge_floor),
        float(pier_tuning.transition_floor),
        float(similarity_floor),
        min_gap,
    )

    return DSPipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        candidate=candidate_cfg,
        construct=construct_cfg,
        repair=repair_cfg,
    )
