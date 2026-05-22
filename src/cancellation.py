"""
Cancellation infrastructure for long-running operations.

Provides thread-safe cancellation tokens that can be passed to worker operations
to enable graceful shutdown and checkpoint emission.
"""

import threading
from typing import Optional


class CancellationToken:
    """
    Thread-safe cancellation signal for long-running operations.

    Usage:
        token = CancellationToken()

        # In worker thread:
        for item in items:
            if token.is_cancelled():
                raise CancelledException("Operation cancelled by user")
            # ... process item

        # In main thread:
        token.cancel()
    """

    def __init__(self):
        """Initialize an uncancelled token."""
        self._cancelled = False
        self._lock = threading.Lock()
        self._cancel_message: Optional[str] = None

    def cancel(self, message: str = "Operation cancelled"):
        """
        Signal cancellation to the operation.

        Args:
            message: Optional cancellation message for diagnostics
        """
        with self._lock:
            self._cancelled = True
            self._cancel_message = message

    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        Returns:
            True if cancel() has been called, False otherwise
        """
        with self._lock:
            return self._cancelled

    def get_message(self) -> Optional[str]:
        """
        Get the cancellation message if set.

        Returns:
            Cancellation message or None if not cancelled
        """
        with self._lock:
            return self._cancel_message

    def reset(self):
        """Reset the token to uncancelled state (for reuse)."""
        with self._lock:
            self._cancelled = False
            self._cancel_message = None


class CancelledException(Exception):
    """
    Exception raised when an operation is cancelled via CancellationToken.

    This exception should be caught at appropriate boundaries to:
    1. Commit partial progress to database
    2. Emit checkpoint for resumption
    3. Clean up resources
    4. Return gracefully to caller

    Example:
        try:
            scanner.run(cancellation_token=token)
        except CancelledException as e:
            logger.info(f"Scan cancelled: {e}")
            # Checkpoint already emitted by scanner
            return {"status": "cancelled", "progress": scan_progress}
    """

    def __init__(self, message: str = "Operation cancelled", progress: Optional[dict] = None):
        """
        Initialize cancellation exception.

        Args:
            message: Human-readable cancellation message
            progress: Optional progress data (items completed, current state, etc.)
        """
        super().__init__(message)
        self.progress = progress or {}
