"""
Cancellation infrastructure for long-running operations.

Provides thread-safe cancellation tokens that can be passed to worker operations
to enable graceful shutdown and checkpoint emission.
"""

import threading
from typing import Callable, Optional


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


# ─────────────────────────────────────────────────────────────────────────────
# Process-global cancellation hook for long-running CORE operations
# ─────────────────────────────────────────────────────────────────────────────
#
# The worker's cancellation protocol is cooperative: a `cancel` command sets a
# flag and the running command must poll it at safe checkpoints and unwind. The
# playlist-generation core (`build_pier_bridge_playlist` -> beam search) lives
# several layers below the worker and cannot reach that flag without threading a
# callback through many load-bearing signatures. Instead the worker registers a
# process-global predicate here, and the core polls it via `raise_if_cancelled`.
#
# This is global state, but it mirrors the worker's existing single-flight model
# (the web bridge enforces one tracked command at a time) and the existing
# `set_cancellation_token` global. Callers MUST clear the hook in a `finally`.


class OperationCancelled(BaseException):
    """Cooperative-cancellation signal raised by `raise_if_cancelled`.

    Derives from ``BaseException`` (NOT ``Exception``) — like
    ``asyncio.CancelledError`` and ``KeyboardInterrupt`` — so the generation
    core's broad ``except Exception`` handlers do not swallow it and convert a
    user cancellation into a "segment failed" that keeps the cascade grinding.
    It must unwind all the way to the worker command handler, which catches it
    explicitly and reports the run as cancelled.
    """


_hook_lock = threading.Lock()
_cancel_predicate: Optional[Callable[[], bool]] = None


def set_cancellation_hook(predicate: Optional[Callable[[], bool]]) -> None:
    """Register (or clear, with ``None``) the process-global cancellation
    predicate polled by core operations via :func:`raise_if_cancelled`.

    ``predicate`` returns ``True`` when the active operation should abort.
    """
    global _cancel_predicate
    with _hook_lock:
        _cancel_predicate = predicate


def raise_if_cancelled() -> None:
    """Poll the registered cancellation hook; raise :class:`OperationCancelled`
    if it reports cancellation. No-op when no hook is registered (CLI, tests).
    """
    with _hook_lock:
        predicate = _cancel_predicate
    if predicate is not None and predicate():
        raise OperationCancelled("Operation cancelled by user")
