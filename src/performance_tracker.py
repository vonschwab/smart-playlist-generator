"""
Performance Tracker - Tracks stage-level timing and throughput metrics.

Used to monitor performance of long-running operations like library scanning
and analysis. Provides structured metrics for logging, UI display, and diagnostics.
"""
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class StageStatus(Enum):
    """Status of a stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageMetrics:
    """Metrics for a single stage."""
    name: str
    status: StageStatus = StageStatus.PENDING
    items_processed: int = 0
    total_items: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    throughput: float = 0.0  # items per second
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return int(self.elapsed_seconds * 1000)

    @property
    def duration_str(self) -> str:
        """Duration as human-readable string."""
        if self.elapsed_seconds < 1:
            return f"{self.duration_ms}ms"
        elif self.elapsed_seconds < 60:
            return f"{self.elapsed_seconds:.1f}s"
        else:
            minutes = int(self.elapsed_seconds // 60)
            seconds = self.elapsed_seconds % 60
            return f"{minutes}m {seconds:.0f}s"


class PerformanceTracker:
    """
    Tracks stage-level timing and throughput metrics.

    Usage:
        tracker = PerformanceTracker("run_123")
        tracker.start_stage("scan")
        for i, item in enumerate(items):
            process_item(item)
            tracker.record_item("scan")  # or tracker.record_item("scan", 10) for batch
        tracker.end_stage("scan", "completed")

        # Get metrics
        metrics = tracker.get_stage_metrics("scan")
        print(f"Scanned {metrics.items_processed} items in {metrics.duration_str}")
        print(f"Throughput: {metrics.throughput:.1f} items/sec")

        # Export all metrics
        data = tracker.to_json()
    """

    def __init__(self, run_id: str):
        """
        Initialize the performance tracker.

        Args:
            run_id: Identifier for this run (for logging/diagnostics)
        """
        self.run_id = run_id
        self.stages: Dict[str, StageMetrics] = {}
        self._logger = logging.getLogger(__name__)
        self._stage_start_item_count: Dict[str, int] = {}

    def start_stage(self, stage_name: str, total_items: int = 0) -> None:
        """
        Mark the start of a stage.

        Args:
            stage_name: Name of the stage (e.g., 'scan', 'genres', 'sonic')
            total_items: Optional total item count for progress tracking
        """
        current_time = time.perf_counter()
        metrics = StageMetrics(
            name=stage_name,
            status=StageStatus.RUNNING,
            total_items=total_items,
            start_time=current_time
        )
        self.stages[stage_name] = metrics
        self._stage_start_item_count[stage_name] = 0
        self._logger.debug(f"Stage '{stage_name}' started")

    def record_item(self, stage_name: str, items: int = 1) -> None:
        """
        Record item processing for a stage.

        Args:
            stage_name: Name of the stage
            items: Number of items processed (default 1)
        """
        if stage_name not in self.stages:
            self._logger.warning(f"Stage '{stage_name}' not started")
            return

        metrics = self.stages[stage_name]
        metrics.items_processed += items

    def end_stage(
        self,
        stage_name: str,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> StageMetrics:
        """
        Mark the end of a stage and calculate metrics.

        Args:
            stage_name: Name of the stage
            status: Final status ('completed', 'failed', 'cancelled')
            error_message: Optional error message if failed

        Returns:
            StageMetrics for this stage
        """
        if stage_name not in self.stages:
            self._logger.warning(f"Stage '{stage_name}' not started")
            return StageMetrics(name=stage_name)

        metrics = self.stages[stage_name]
        current_time = time.perf_counter()

        # Calculate metrics
        if metrics.start_time:
            metrics.elapsed_seconds = current_time - metrics.start_time
        else:
            metrics.elapsed_seconds = 0.0

        if metrics.elapsed_seconds > 0 and metrics.items_processed > 0:
            metrics.throughput = metrics.items_processed / metrics.elapsed_seconds
        else:
            metrics.throughput = 0.0

        # Set status
        try:
            metrics.status = StageStatus(status)
        except ValueError:
            metrics.status = StageStatus.COMPLETED

        metrics.error_message = error_message
        metrics.end_time = current_time

        # Log summary
        if metrics.elapsed_seconds < 1:
            duration_str = f"{int(metrics.elapsed_seconds * 1000)}ms"
        elif metrics.elapsed_seconds < 60:
            duration_str = f"{metrics.elapsed_seconds:.1f}s"
        else:
            minutes = int(metrics.elapsed_seconds // 60)
            seconds = metrics.elapsed_seconds % 60
            duration_str = f"{minutes}m {seconds:.0f}s"

        if metrics.items_processed > 0:
            self._logger.info(
                f"Stage '{stage_name}' {status}: "
                f"{metrics.items_processed} items in {duration_str} "
                f"({metrics.throughput:.1f} items/sec)"
            )
        else:
            self._logger.info(
                f"Stage '{stage_name}' {status}: {duration_str}"
            )

        if error_message:
            self._logger.error(f"  Error: {error_message}")

        return metrics

    def get_stage_metrics(self, stage_name: str) -> Optional[StageMetrics]:
        """Get metrics for a specific stage."""
        return self.stages.get(stage_name)

    def get_all_metrics(self) -> Dict[str, StageMetrics]:
        """Get all stage metrics."""
        return dict(self.stages)

    def to_json(self) -> Dict[str, Any]:
        """
        Export all metrics as JSON-serializable dictionary.

        Returns:
            Dict with run_id and stages data
        """
        return {
            "run_id": self.run_id,
            "stages": {
                name: metrics.to_dict()
                for name, metrics in self.stages.items()
            }
        }

    def get_summary(self) -> str:
        """
        Get a human-readable summary of all stages.

        Returns:
            Formatted summary string
        """
        if not self.stages:
            return "No stages tracked"

        lines = [f"Performance Summary (run_id={self.run_id})"]
        lines.append("─" * 70)

        total_elapsed = 0.0
        total_items = 0

        for stage_name in sorted(self.stages.keys()):
            metrics = self.stages[stage_name]
            if metrics.status == StageStatus.COMPLETED:
                total_elapsed += metrics.elapsed_seconds
                total_items += metrics.items_processed

            status_str = metrics.status.value.upper()
            lines.append(
                f"{stage_name:20} {status_str:12} "
                f"{metrics.items_processed:8} items  "
                f"{metrics.duration_str:12}  "
                f"{metrics.throughput:8.1f} items/sec"
            )

            if metrics.error_message:
                lines.append(f"  Error: {metrics.error_message}")

        lines.append("─" * 70)
        if total_elapsed > 0:
            overall_throughput = total_items / total_elapsed if total_items > 0 else 0
            lines.append(
                f"{'TOTAL':20} {total_items:8} items  "
                f"{self._format_duration(total_elapsed):12}  "
                f"{overall_throughput:8.1f} items/sec"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds as human-readable string."""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
