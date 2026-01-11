"""
UI State Model - Single source of truth for generation UI state.

This dataclass captures all user-facing controls for playlist generation,
independent of the underlying config format. The PolicyLayer derives
runtime configuration from this state.

Created: Phase 1 of GUI "Just Works" implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class UIStateModel:
    """
    Single source of truth for "Generate" UI state.

    All fields have sensible defaults matching the target UX.
    The PolicyLayer reads this to derive runtime config overrides.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Top-level mode
    # ─────────────────────────────────────────────────────────────────────────
    mode: Literal["artist", "history", "seeds"] = "artist"

    # ─────────────────────────────────────────────────────────────────────────
    # Cohesion (replaces separate genre_mode + sonic_mode)
    # Monotonic progression: tight → balanced → wide → discover
    # ─────────────────────────────────────────────────────────────────────────
    cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"

    # ─────────────────────────────────────────────────────────────────────────
    # Track count
    # ─────────────────────────────────────────────────────────────────────────
    track_count: int = 30

    # ─────────────────────────────────────────────────────────────────────────
    # Recency filter (hard exclude; never relaxed by policy)
    # Semantics: exclude tracks played >= recency_plays_threshold times
    #            in the last recency_days days
    # ─────────────────────────────────────────────────────────────────────────
    recency_enabled: bool = True
    recency_days: int = 14
    recency_plays_threshold: int = 1

    # ─────────────────────────────────────────────────────────────────────────
    # Artist spacing
    # ─────────────────────────────────────────────────────────────────────────
    artist_spacing: Literal["normal", "strong"] = "normal"

    # ─────────────────────────────────────────────────────────────────────────
    # Artist(s) mode specific
    # ─────────────────────────────────────────────────────────────────────────
    artist_queries: List[str] = field(default_factory=list)
    """
    Artist name chips. For now, only the first is used by the backend.
    Multi-artist journeys are a future feature.
    """

    artist_presence: Literal["low", "medium", "high", "max"] = "medium"
    """
    Target percentage of playlist from the seed artist(s):
    - low: ~10%
    - medium: ~25%
    - high: ~40%
    - max: ~60%
    """

    artist_variety: Literal["focused", "balanced", "sprawling"] = "balanced"
    """
    Controls internal artist seed dispersion (clustering variance).
    - focused: tight style clustering
    - balanced: moderate diversity
    - sprawling: wide stylistic spread
    """

    # ─────────────────────────────────────────────────────────────────────────
    # History mode specific
    # ─────────────────────────────────────────────────────────────────────────
    history_window_days: int = 30
    """Time window for history-based generation (7/14/30/90)."""

    # ─────────────────────────────────────────────────────────────────────────
    # Seed(s) mode specific
    # ─────────────────────────────────────────────────────────────────────────
    seed_track_ids: List[str] = field(default_factory=list)
    """
    Stable track IDs for seeds. These should be database IDs, not display strings.
    The Seeds UI must resolve track names to IDs before populating this field.
    """

    seed_auto_order: bool = True
    """
    When True, seeds are reordered for optimal bridging flow.
    When False, seeds are used in the order provided by the user.

    Implementation note (verified from pier_bridge_builder.py):
    - seed_auto_order=True maps to dj_seed_ordering="auto" (optimizes order)
    - seed_auto_order=False maps to dj_seed_ordering="fixed" (preserves order)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def primary_artist(self) -> Optional[str]:
        """
        Return the first artist chip, or None if empty.

        Used for single-artist mode until multi-artist journeys are implemented.
        """
        if self.artist_queries:
            return self.artist_queries[0]
        return None

    def seed_count(self) -> int:
        """Return the number of seed tracks."""
        return len(self.seed_track_ids)
