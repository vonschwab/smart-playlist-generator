from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Mode = Literal["narrow", "dynamic", "discover", "sonic_only"]
AlphaSchedule = Literal["constant", "arc"]
RepairObjective = Literal["gap_penalty", "below_floor_first"]


@dataclass(frozen=True)
class CandidatePoolConfig:
    similarity_floor: float
    max_pool_size: int
    target_artists: int
    candidates_per_artist: int
    seed_artist_bonus: int
    max_artist_fraction_final: float


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
        {"narrow": 0.30, "dynamic": 0.20, "discover": 0.10, "sonic_only": 0.0}[mode],
    )
    transition_floor = constraints.get("transition_floor", similarity_floor)

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
        max_pool_size=candidate_pool.get(
            "max_pool_size",
            {"narrow": 800, "dynamic": 1200, "discover": 2000, "sonic_only": 1200}[mode],
        ),
        target_artists=target_artists,
        candidates_per_artist=_candidate_per_artist(max_per_artist_final_est),
        seed_artist_bonus=seed_artist_bonus,
        max_artist_fraction_final=max_artist_fraction_final,
    )

    # Construction config with config.yaml overrides
    mode_alpha_schedule = "constant" if mode == "narrow" else "arc"
    construct_cfg = ConstructionConfig(
        local_top_m=25 if mode != "narrow" else 15,
        alpha_schedule=scoring.get("alpha_schedule", mode_alpha_schedule),
        alpha=scoring.get(
            "alpha",
            {"narrow": 0.65, "dynamic": 0.55, "discover": 0.40, "sonic_only": 0.55}[mode],
        ),
        alpha_start=scoring.get(
            "alpha_start",
            {"narrow": 0.65, "dynamic": 0.65, "discover": 0.55, "sonic_only": 0.65}[mode],
        ),
        alpha_mid=scoring.get(
            "alpha_mid",
            {"narrow": 0.65, "dynamic": 0.45, "discover": 0.30, "sonic_only": 0.45}[mode],
        ),
        alpha_end=scoring.get(
            "alpha_end",
            {"narrow": 0.65, "dynamic": 0.60, "discover": 0.45, "sonic_only": 0.60}[mode],
        ),
        arc_midpoint=scoring.get("arc_midpoint", 0.55),
        beta=scoring.get(
            "beta",
            {"narrow": 0.50, "dynamic": 0.55, "discover": 0.60, "sonic_only": 0.45}[mode],
        ),
        gamma=scoring.get(
            "gamma",
            {"narrow": 0.02, "dynamic": 0.04, "discover": 0.10, "sonic_only": 0.04}[mode],
        ),
        hard_floor=constraints.get(
            "hard_floor",
            True if mode in {"dynamic", "sonic_only"} else False,
        ),
        transition_gamma=1.0,
        transition_floor=transition_floor,
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
