"""Unit tests for pier-bridge diagnostics module.

Tests extracted diagnostics collection logic from pier_bridge_builder.py (Phase 3.3).

Coverage:
- SegmentDiagnostics dataclass
- Diagnostics collection (enabled/disabled)
- Summary statistics computation
- Logging behavior
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.pier_bridge_diagnostics import (
    SegmentDiagnostics,
    PierBridgeDiagnosticsCollector,
)


# =============================================================================
# SegmentDiagnostics Tests
# =============================================================================

class TestSegmentDiagnostics:
    """Test SegmentDiagnostics dataclass."""

    def test_create_segment_diagnostics(self):
        """Test creating segment diagnostics."""
        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
            bridge_floor_used=0.03,
            backoff_attempts_used=1,
            widened_search=False,
        )

        assert diag.pier_a_id == "track_001"
        assert diag.pier_b_id == "track_002"
        assert diag.target_length == 5
        assert diag.actual_length == 5
        assert diag.success is True

    def test_segment_diagnostics_defaults(self):
        """Test default values for optional fields."""
        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        assert diag.bridge_floor_used == 0.0
        assert diag.backoff_attempts_used == 1
        assert diag.widened_search is False


# =============================================================================
# Diagnostics Collector Tests
# =============================================================================

class TestDiagnosticsCollector:
    """Test PierBridgeDiagnosticsCollector."""

    def test_collector_disabled_by_default(self):
        """Test that collector is disabled by default."""
        collector = PierBridgeDiagnosticsCollector()

        assert collector.enabled is False
        assert collector.get_segment_count() == 0

    def test_collector_enabled(self):
        """Test that collector can be enabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        assert collector.enabled is True
        assert collector.get_segment_count() == 0

    def test_record_segment_when_disabled(self):
        """Test that recording segment does nothing when disabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=False)

        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        collector.record_segment(diag)

        assert collector.get_segment_count() == 0

    def test_record_segment_when_enabled(self):
        """Test recording segments when enabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag1 = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        diag2 = SegmentDiagnostics(
            pier_a_id="track_002",
            pier_b_id="track_003",
            target_length=5,
            actual_length=5,
            pool_size_initial=80,
            pool_size_final=40,
            expansions=0,
            beam_width_used=16,
            worst_edge_score=0.30,
            mean_edge_score=0.50,
            success=True,
        )

        collector.record_segment(diag1)
        collector.record_segment(diag2)

        assert collector.get_segment_count() == 2

    def test_get_segment(self):
        """Test retrieving specific segment."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag1 = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        diag2 = SegmentDiagnostics(
            pier_a_id="track_002",
            pier_b_id="track_003",
            target_length=5,
            actual_length=5,
            pool_size_initial=80,
            pool_size_final=40,
            expansions=0,
            beam_width_used=16,
            worst_edge_score=0.30,
            mean_edge_score=0.50,
            success=True,
        )

        collector.record_segment(diag1)
        collector.record_segment(diag2)

        retrieved = collector.get_segment(0)
        assert retrieved is not None
        assert retrieved.pier_a_id == "track_001"

        retrieved = collector.get_segment(1)
        assert retrieved is not None
        assert retrieved.pier_a_id == "track_002"

    def test_get_segment_out_of_range(self):
        """Test retrieving segment with invalid index."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        collector.record_segment(diag)

        # Out of range indices
        assert collector.get_segment(-1) is None
        assert collector.get_segment(1) is None
        assert collector.get_segment(999) is None

    def test_get_all_segments(self):
        """Test retrieving all segments."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag1 = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        diag2 = SegmentDiagnostics(
            pier_a_id="track_002",
            pier_b_id="track_003",
            target_length=5,
            actual_length=5,
            pool_size_initial=80,
            pool_size_final=40,
            expansions=0,
            beam_width_used=16,
            worst_edge_score=0.30,
            mean_edge_score=0.50,
            success=True,
        )

        collector.record_segment(diag1)
        collector.record_segment(diag2)

        all_segments = collector.get_all_segments()

        assert len(all_segments) == 2
        assert all_segments[0].pier_a_id == "track_001"
        assert all_segments[1].pier_a_id == "track_002"

        # Should be a copy
        all_segments.clear()
        assert collector.get_segment_count() == 2


# =============================================================================
# Summary Statistics Tests
# =============================================================================

class TestSummaryStatistics:
    """Test summary statistics computation."""

    def test_summary_stats_empty(self):
        """Test summary stats with no segments."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        stats = collector.get_summary_stats()

        assert stats == {}

    def test_summary_stats_disabled(self):
        """Test summary stats when collector disabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=False)

        stats = collector.get_summary_stats()

        assert stats == {}

    def test_summary_stats_single_segment(self):
        """Test summary stats with single segment."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        collector.record_segment(diag)

        stats = collector.get_summary_stats()

        assert stats["total_segments"] == 1
        assert stats["successful_segments"] == 1
        assert stats["failed_segments"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["pool_size"]["initial_mean"] == 100.0
        assert stats["pool_size"]["final_mean"] == 50.0
        assert stats["edge_quality"]["worst_edge_mean"] == 0.25
        assert stats["edge_quality"]["mean_edge_mean"] == 0.45

    def test_summary_stats_multiple_segments(self):
        """Test summary stats with multiple segments."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag1 = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        diag2 = SegmentDiagnostics(
            pier_a_id="track_002",
            pier_b_id="track_003",
            target_length=5,
            actual_length=5,
            pool_size_initial=80,
            pool_size_final=40,
            expansions=2,
            beam_width_used=64,
            worst_edge_score=0.30,
            mean_edge_score=0.50,
            success=True,
        )

        collector.record_segment(diag1)
        collector.record_segment(diag2)

        stats = collector.get_summary_stats()

        assert stats["total_segments"] == 2
        assert stats["successful_segments"] == 2
        assert stats["pool_size"]["initial_mean"] == 90.0  # (100 + 80) / 2
        assert stats["pool_size"]["final_mean"] == 45.0   # (50 + 40) / 2
        assert stats["edge_quality"]["worst_edge_mean"] == 0.275  # (0.25 + 0.30) / 2
        assert stats["search_complexity"]["expansions_mean"] == 1.5  # (1 + 2) / 2

    def test_summary_stats_with_failures(self):
        """Test summary stats with some failed segments."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag1 = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        diag2 = SegmentDiagnostics(
            pier_a_id="track_002",
            pier_b_id="track_003",
            target_length=5,
            actual_length=3,  # Failed to reach target
            pool_size_initial=80,
            pool_size_final=40,
            expansions=3,
            beam_width_used=64,
            worst_edge_score=0.15,
            mean_edge_score=0.30,
            success=False,  # Failed
        )

        collector.record_segment(diag1)
        collector.record_segment(diag2)

        stats = collector.get_summary_stats()

        assert stats["total_segments"] == 2
        assert stats["successful_segments"] == 1
        assert stats["failed_segments"] == 1
        assert stats["success_rate"] == 0.5


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Test diagnostics export functionality."""

    def test_to_dict_when_disabled(self):
        """Test to_dict when collector is disabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=False)

        result = collector.to_dict()

        assert result == {"enabled": False}

    def test_to_dict_when_enabled(self):
        """Test to_dict when collector is enabled."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        collector.record_segment(diag)

        result = collector.to_dict()

        assert result["enabled"] is True
        assert "segments" in result
        assert len(result["segments"]) == 1
        assert "summary" in result
        assert result["summary"]["total_segments"] == 1


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtilities:
    """Test utility methods."""

    def test_clear(self):
        """Test clearing collected diagnostics."""
        collector = PierBridgeDiagnosticsCollector(enabled=True)

        diag = SegmentDiagnostics(
            pier_a_id="track_001",
            pier_b_id="track_002",
            target_length=5,
            actual_length=5,
            pool_size_initial=100,
            pool_size_final=50,
            expansions=1,
            beam_width_used=32,
            worst_edge_score=0.25,
            mean_edge_score=0.45,
            success=True,
        )

        collector.record_segment(diag)
        assert collector.get_segment_count() == 1

        collector.clear()
        assert collector.get_segment_count() == 0
