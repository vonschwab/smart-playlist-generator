from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import math

Mode = Literal["narrow", "dynamic", "discover", "sonic_only"]
AlphaSchedule = Literal["constant", "arc"]
RepairObjective = Literal["gap_penalty", "below_floor_first"]


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
    broad_filters: tuple[str, ...] = ()


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
    """Resolved pier-bridge tuning knobs (per ds_mode)."""

    transition_floor: float
    bridge_floor: float
    weight_bridge: float
    weight_transition: float
    genre_tiebreak_weight: float
    genre_penalty_threshold: float
    genre_penalty_strength: float


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
    Resolve pier-bridge tuning for the given ds_mode.

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

    default_bridge_floor = 0.03 if mode_s == "dynamic" else 0.08 if mode_s == "narrow" else 0.0
    default_weight_bridge = 0.6 if mode_s == "dynamic" else 0.7 if mode_s == "narrow" else 0.0
    default_weight_transition = 0.4 if mode_s == "dynamic" else 0.3 if mode_s == "narrow" else 1.0

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

    tuning = PierBridgeTuning(
        transition_floor=float(transition_floor),
        bridge_floor=float(bridge_floor),
        weight_bridge=float(weight_bridge),
        weight_transition=float(weight_transition),
        genre_tiebreak_weight=float(genre_tiebreak_weight),
        genre_penalty_threshold=float(genre_penalty_threshold),
        genre_penalty_strength=float(genre_penalty_strength),
    )
    return tuning, sources


def get_min_sonic_similarity(candidate_pool_cfg: dict, mode: Mode) -> Optional[float]:
    """
    Resolve the sonic similarity floor for the given mode with sensible defaults.

    Priority:
    1) min_sonic_similarity_<mode>
    2) min_sonic_similarity (applies to all modes)
    3) Built-in defaults: narrow=0.10, dynamic=0.00, discover/sonic_only=None
    """
    mode = mode.lower()  # type: ignore[assignment]
    mode_key = f"min_sonic_similarity_{mode}"
    if mode_key in candidate_pool_cfg and candidate_pool_cfg.get(mode_key) is None:
        return None
    if "min_sonic_similarity" in candidate_pool_cfg and candidate_pool_cfg.get("min_sonic_similarity") is None:
        return None
    mode_specific = candidate_pool_cfg.get(mode_key)
    base = candidate_pool_cfg.get("min_sonic_similarity")

    default = {
        "narrow": 0.10,
        "dynamic": 0.00,
        "discover": None,
        "sonic_only": None,
    }.get(mode, None)

    resolved = mode_specific if mode_specific is not None else base
    resolved = default if resolved is None else resolved
    return float(resolved) if resolved is not None else None


def default_ds_config(
    mode: Mode,
    *,
    playlist_len: int,
    overrides: Optional[dict] = None,
) -> DSPipelineConfig:
    """
    Return mode defaults matching experiments:
    - artist caps and min_gap: narrow(20%,3), dynamic(12.5%,6), discover(5%,9)
    - floors roughly: narrow ~0.35, dynamic ~0.30, discover ~0.25 (as in plan)
    - hard_floor True only for dynamic
    - pick sensible defaults for top_m and alpha/beta/gamma aligned with current experiments
    Compute max_per_artist_final = ceil(L * max_artist_fraction_final) later in code.

    Args:
        mode: One of "narrow", "dynamic", "discover", "sonic_only"
        playlist_len: Target playlist length
        overrides: Optional dict from config.yaml with keys like 'scoring', 'constraints', etc.
    """
    mode = mode.lower()  # type: ignore[assignment]
    if mode not in {"narrow", "dynamic", "discover", "sonic_only"}:
        raise ValueError(f"Unsupported mode {mode}")

    # Parse overrides from config.yaml
    if overrides is None:
        overrides = {}
    scoring = overrides.get("scoring", {})
    constraints = overrides.get("constraints", {})
    candidate_pool = overrides.get("candidate_pool", {})
    repair = overrides.get("repair", {})

    # Mode defaults - can be overridden by config.yaml values
    max_artist_fraction_final = candidate_pool.get(
        "max_artist_fraction",
        {"narrow": 0.20, "dynamic": 0.125, "discover": 0.05, "sonic_only": 0.125}[mode],
    )
    min_gap = constraints.get(
        "min_gap",
        {"narrow": 3, "dynamic": 6, "discover": 9, "sonic_only": 6}[mode],
    )
    similarity_floor = candidate_pool.get(
        "similarity_floor",
        # Shift each mode narrower: higher floors for tighter cohesion.
        {"narrow": 0.35, "dynamic": 0.28, "discover": 0.22, "sonic_only": 0.0}[mode],
    )
    min_sonic_similarity = get_min_sonic_similarity(candidate_pool, mode)
    broad_filters_cfg_raw = candidate_pool.get("broad_filters", None)
    if broad_filters_cfg_raw is None:
        broad_filters_cfg: list[str] = ["rock", "indie", "alternative", "pop"] if mode == "narrow" else []
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
        "narrow": max(int((playlist_len + 1) // 2), 12),
        "dynamic": max(int(round(0.75 * playlist_len)), 16),
        "discover": min(playlist_len, 24),
        "sonic_only": max(int(round(0.75 * playlist_len)), 16),
    }[mode]

    seed_artist_bonus = 2
    # max_per_artist_final computed downstream, but candidate_per_artist mirrors experiments
    def _candidate_per_artist(max_per_artist_final: int) -> int:
        if mode == "narrow":
            return max(3, min(2 * max_per_artist_final, 8))
        if mode == "dynamic":
            return max(3, min(2 * max_per_artist_final, 6))
        if mode == "sonic_only":
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
            {"narrow": 800, "dynamic": 1200, "discover": 2000, "sonic_only": 1200}[mode],
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
        broad_filters=tuple(str(b).lower() for b in broad_filters_cfg),
    )

    # Construction config with config.yaml overrides
    mode_alpha_schedule = "constant" if mode == "narrow" else "arc"
    construct_cfg = ConstructionConfig(
        local_top_m=25 if mode != "narrow" else 15,
        alpha_schedule=scoring.get("alpha_schedule", mode_alpha_schedule),
        alpha=scoring.get(
            "alpha",
            # Lean more on seed similarity for all modes (narrowest gets highest).
            {"narrow": 0.70, "dynamic": 0.62, "discover": 0.50, "sonic_only": 0.55}[mode],
        ),
        alpha_start=scoring.get(
            "alpha_start",
            {"narrow": 0.70, "dynamic": 0.65, "discover": 0.55, "sonic_only": 0.65}[mode],
        ),
        alpha_mid=scoring.get(
            "alpha_mid",
            {"narrow": 0.70, "dynamic": 0.50, "discover": 0.35, "sonic_only": 0.45}[mode],
        ),
        alpha_end=scoring.get(
            "alpha_end",
            {"narrow": 0.70, "dynamic": 0.62, "discover": 0.50, "sonic_only": 0.60}[mode],
        ),
        arc_midpoint=scoring.get("arc_midpoint", 0.55),
        beta=scoring.get(
            "beta",
            {"narrow": 0.55, "dynamic": 0.58, "discover": 0.62, "sonic_only": 0.45}[mode],
        ),
        gamma=scoring.get(
            "gamma",
            # Dial back diversity bonus to favor tighter cohesion.
            {"narrow": 0.01, "dynamic": 0.025, "discover": 0.06, "sonic_only": 0.04}[mode],
        ),
        hard_floor=constraints.get(
            "hard_floor",
            True if mode in {"dynamic", "sonic_only"} else False,
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

    return DSPipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        candidate=candidate_cfg,
        construct=construct_cfg,
        repair=repair_cfg,
    )
