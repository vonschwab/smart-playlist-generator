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
Bridge Scoring:
    compute_bridgeability_score()

Constraints:
    TransitionWeights
    ScoringConstraints
    SeedOrderingConfig

(Transition scoring lives in src/playlist/transition_metrics.py
(score_transition_edge); the legacy transition_scoring duplicate was removed.)
"""

from .bridge_scoring import (
    compute_bridgeability_score,
)
from .constraints import (
    TransitionWeights,
    ScoringConstraints,
    SeedOrderingConfig,
)

__all__ = [
    # Bridge scoring
    "compute_bridgeability_score",
    # Constraints
    "TransitionWeights",
    "ScoringConstraints",
    "SeedOrderingConfig",
]
