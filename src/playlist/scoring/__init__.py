"""
Scoring Module for Pier-Bridge Playlists
=========================================

Extracted from pier_bridge_builder.py during Phase 3.1 refactoring.

This module provides scoring functions for:
- Transition quality (track A → track B)
- Bridge quality (how well two seeds can be connected)
- Scoring constraints and configuration

Public API:
-----------
Transition Scoring:
    compute_transition_score()
    compute_transition_score_raw_and_transformed()

Bridge Scoring:
    compute_bridgeability_score()

Constraints:
    TransitionWeights
    ScoringConstraints
    SeedOrderingConfig
"""

from .transition_scoring import (
    compute_transition_score,
    compute_transition_score_raw_and_transformed,
)
from .bridge_scoring import (
    compute_bridgeability_score,
)
from .constraints import (
    TransitionWeights,
    ScoringConstraints,
    SeedOrderingConfig,
)

__all__ = [
    # Transition scoring
    "compute_transition_score",
    "compute_transition_score_raw_and_transformed",
    # Bridge scoring
    "compute_bridgeability_score",
    # Constraints
    "TransitionWeights",
    "ScoringConstraints",
    "SeedOrderingConfig",
]
