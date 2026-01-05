"""
Scoring Constraints for Pier-Bridge Playlists
==============================================

Extracted from pier_bridge_builder.py (Phase 3.1).

This module defines constraints and parameters used during
pier-bridge scoring and construction.

Dataclasses extracted from pier_bridge_builder.py config.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionWeights:
    """Weights for multi-segment transition scoring."""

    weight_end_start: float
    """Weight for end-to-start segment similarity."""

    weight_mid_mid: float
    """Weight for mid-to-mid segment similarity."""

    weight_full_full: float
    """Weight for full-track similarity."""

    def __post_init__(self):
        """Validate weights sum to reasonable range."""
        total = self.weight_end_start + self.weight_mid_mid + self.weight_full_full
        if not (0.9 <= total <= 1.1):
            raise ValueError(f"Transition weights should sum to ~1.0, got {total:.3f}")


@dataclass(frozen=True)
class ScoringConstraints:
    """Constraints applied during scoring and candidate selection."""

    min_gap: int = 6
    """Minimum gap between same-artist tracks in playlist."""

    bridge_floor: float = 0.03
    """Minimum similarity score for bridge candidates."""

    transition_floor: float = 0.20
    """Minimum transition score for edges."""

    genre_penalty_threshold: float = 0.20
    """Genre similarity below this triggers penalty."""

    genre_penalty_strength: float = 0.10
    """Strength of genre penalty (subtracted from score)."""

    duration_penalty_weight: float = 0.30
    """Weight for duration mismatch penalty."""

    progress_penalty_weight: float = 0.15
    """Weight for progress-based diversity penalty."""

    center_transitions: bool = True
    """If True, rescale transition scores from [-1,1] to [0,1]."""

    def __post_init__(self):
        """Validate constraint values."""
        if not (0.0 <= self.bridge_floor <= 1.0):
            raise ValueError(f"bridge_floor must be in [0,1], got {self.bridge_floor}")
        if not (0.0 <= self.transition_floor <= 1.0):
            raise ValueError(f"transition_floor must be in [0,1], got {self.transition_floor}")
        if self.min_gap < 0:
            raise ValueError(f"min_gap must be >= 0, got {self.min_gap}")


@dataclass(frozen=True)
class SeedOrderingConfig:
    """Configuration for ordering seed tracks by bridgeability."""

    max_exhaustive_search: int = 6
    """Maximum seeds for exhaustive permutation search."""

    use_greedy_for_large: bool = True
    """Use greedy nearest-neighbor for >max_exhaustive_search seeds."""

    def __post_init__(self):
        """Validate config."""
        if self.max_exhaustive_search < 1:
            raise ValueError(f"max_exhaustive_search must be >= 1, got {self.max_exhaustive_search}")
