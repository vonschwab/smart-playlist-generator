"""
Diagnostics Collection for Pier-Bridge Playlists
================================================

Extracted from pier_bridge_builder.py (Phase 3.3).

This module provides diagnostic collection for pier-bridge playlist
construction, tracking segment-level metrics and overall statistics.

Classes extracted from pier_bridge_builder.py:
- SegmentDiagnostics (dataclass)
- PierBridgeDiagnosticsCollector (new)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SegmentDiagnostics:
    """Diagnostics for a single bridge segment.

    Tracks metrics about segment construction including pool sizes,
    beam search parameters, and edge quality scores.
    """

    pier_a_id: str
    """Track ID of first pier (segment start)."""

    pier_b_id: str
    """Track ID of second pier (segment end)."""

    target_length: int
    """Target number of interior tracks for this segment."""

    actual_length: int
    """Actual number of interior tracks found."""

    pool_size_initial: int
    """Initial candidate pool size before gating."""

    pool_size_final: int
    """Final candidate pool size after gating."""

    expansions: int
    """Number of search expansions performed."""

    beam_width_used: int
    """Beam width used for final successful search."""

    worst_edge_score: float
    """Worst (minimum) transition score among edges."""

    mean_edge_score: float
    """Mean transition score across all edges."""

    success: bool
    """Whether segment was successfully constructed."""

    bridge_floor_used: float = 0.0
    """Bridge floor threshold used for final successful attempt."""

    backoff_attempts_used: int = 1
    """Number of bridge floor backoff attempts used."""

    widened_search: bool = False
    """Whether search was widened during construction."""


class PierBridgeDiagnosticsCollector:
    """Collects diagnostics during pier-bridge construction.

    Provides optional diagnostic collection via dependency injection.
    When enabled, tracks segment-level metrics and logs progress.
    """

    def __init__(self, enabled: bool = False):
        """Initialize diagnostics collector.

        Args:
            enabled: If True, collect diagnostics. If False, no-op.
        """
        self.enabled = enabled
        self.segments: List[SegmentDiagnostics] = []
        self.logger = logging.getLogger(__name__)

    def record_segment(self, diag: SegmentDiagnostics) -> None:
        """Record diagnostics for a completed segment.

        Args:
            diag: Segment diagnostics to record
        """
        if not self.enabled:
            return

        self.segments.append(diag)

        # Log segment completion
        self.logger.info(
            "Segment %d: %s -> %s | pool=%d/%d expansions=%d beam=%d mean_edge=%.3f",
            len(self.segments) - 1,
            diag.pier_a_id,
            diag.pier_b_id,
            diag.pool_size_final,
            diag.pool_size_initial,
            diag.expansions,
            diag.beam_width_used,
            diag.mean_edge_score,
        )

        if not diag.success:
            self.logger.warning(
                "Segment %d FAILED: target=%d actual=%d",
                len(self.segments) - 1,
                diag.target_length,
                diag.actual_length,
            )

    def get_segment_count(self) -> int:
        """Get number of segments recorded.

        Returns:
            Number of segments
        """
        return len(self.segments)

    def get_segment(self, index: int) -> Optional[SegmentDiagnostics]:
        """Get diagnostics for a specific segment.

        Args:
            index: Segment index (0-based)

        Returns:
            SegmentDiagnostics if found, None otherwise
        """
        if not self.enabled or index < 0 or index >= len(self.segments):
            return None
        return self.segments[index]

    def get_all_segments(self) -> List[SegmentDiagnostics]:
        """Get diagnostics for all segments.

        Returns:
            List of all segment diagnostics
        """
        if not self.enabled:
            return []
        return self.segments.copy()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all segments.

        Returns:
            Dictionary of summary statistics
        """
        if not self.enabled or not self.segments:
            return {}

        total_segments = len(self.segments)
        successful_segments = sum(1 for s in self.segments if s.success)
        failed_segments = total_segments - successful_segments

        # Pool size statistics
        initial_pool_sizes = [s.pool_size_initial for s in self.segments]
        final_pool_sizes = [s.pool_size_final for s in self.segments]

        # Edge score statistics
        worst_edge_scores = [s.worst_edge_score for s in self.segments]
        mean_edge_scores = [s.mean_edge_score for s in self.segments]

        # Search complexity statistics
        expansions = [s.expansions for s in self.segments]
        beam_widths = [s.beam_width_used for s in self.segments]
        backoff_attempts = [s.backoff_attempts_used for s in self.segments]

        return {
            "total_segments": total_segments,
            "successful_segments": successful_segments,
            "failed_segments": failed_segments,
            "success_rate": successful_segments / total_segments if total_segments > 0 else 0.0,
            "pool_size": {
                "initial_mean": sum(initial_pool_sizes) / len(initial_pool_sizes) if initial_pool_sizes else 0.0,
                "initial_min": min(initial_pool_sizes) if initial_pool_sizes else 0,
                "initial_max": max(initial_pool_sizes) if initial_pool_sizes else 0,
                "final_mean": sum(final_pool_sizes) / len(final_pool_sizes) if final_pool_sizes else 0.0,
                "final_min": min(final_pool_sizes) if final_pool_sizes else 0,
                "final_max": max(final_pool_sizes) if final_pool_sizes else 0,
            },
            "edge_quality": {
                "worst_edge_mean": sum(worst_edge_scores) / len(worst_edge_scores) if worst_edge_scores else 0.0,
                "worst_edge_min": min(worst_edge_scores) if worst_edge_scores else 0.0,
                "worst_edge_max": max(worst_edge_scores) if worst_edge_scores else 0.0,
                "mean_edge_mean": sum(mean_edge_scores) / len(mean_edge_scores) if mean_edge_scores else 0.0,
                "mean_edge_min": min(mean_edge_scores) if mean_edge_scores else 0.0,
                "mean_edge_max": max(mean_edge_scores) if mean_edge_scores else 0.0,
            },
            "search_complexity": {
                "expansions_mean": sum(expansions) / len(expansions) if expansions else 0.0,
                "expansions_max": max(expansions) if expansions else 0,
                "beam_width_mean": sum(beam_widths) / len(beam_widths) if beam_widths else 0.0,
                "beam_width_max": max(beam_widths) if beam_widths else 0,
                "backoff_attempts_mean": sum(backoff_attempts) / len(backoff_attempts) if backoff_attempts else 0.0,
                "backoff_attempts_max": max(backoff_attempts) if backoff_attempts else 0,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export diagnostics to dictionary for audit reports.

        Returns:
            Dictionary with segments and summary statistics
        """
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "segments": [asdict(s) for s in self.segments],
            "summary": self.get_summary_stats(),
        }

    def log_final_summary(self) -> None:
        """Log final summary of diagnostics.

        Logs summary statistics after all segments are complete.
        """
        if not self.enabled or not self.segments:
            return

        stats = self.get_summary_stats()

        self.logger.info(
            "=== Pier-Bridge Diagnostics Summary ===\n"
            "Segments: %d total, %d successful (%.1f%%)\n"
            "Pool sizes: initial=%.1f final=%.1f\n"
            "Edge quality: worst=%.3f mean=%.3f\n"
            "Search complexity: expansions=%.1f beam=%.1f backoff=%.1f",
            stats["total_segments"],
            stats["successful_segments"],
            stats["success_rate"] * 100,
            stats["pool_size"]["initial_mean"],
            stats["pool_size"]["final_mean"],
            stats["edge_quality"]["worst_edge_mean"],
            stats["edge_quality"]["mean_edge_mean"],
            stats["search_complexity"]["expansions_mean"],
            stats["search_complexity"]["beam_width_mean"],
            stats["search_complexity"]["backoff_attempts_mean"],
        )

    def clear(self) -> None:
        """Clear all collected diagnostics.

        Useful for reusing the collector across multiple runs.
        """
        self.segments.clear()
