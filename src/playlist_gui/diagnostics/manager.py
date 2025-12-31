"""
Diagnostics manager with TTL caching, debounce, and busy-aware worker doctor integration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QTimer, QRunnable, QThreadPool, Signal

from .checks import run_checks, CheckResult


class _DiagnosticsSignals(QObject):
    results_ready = Signal(object, object)  # results, include_worker


class _DiagnosticsTask(QRunnable):
    """Runs diagnostics in a worker thread to avoid blocking the UI."""

    def __init__(
        self,
        base_config_path: str,
        config_model_provider: Callable[[], object],
        runner: Callable[[str, object], List[CheckResult]],
        include_worker: bool,
    ):
        super().__init__()
        self.signals = _DiagnosticsSignals()
        self._base_config_path = base_config_path
        self._config_model_provider = config_model_provider
        self._runner = runner
        self._include_worker = include_worker

    def run(self) -> None:  # pragma: no cover - exercised via manager
        results = self._runner(self._base_config_path, self._config_model_provider())
        self.signals.results_ready.emit(results, self._include_worker)


class DiagnosticsManager(QObject):
    """
    Coordinates diagnostics execution with TTL caching, debounce, and worker-aware throttling.
    """

    diagnostics_updated = Signal(object, object)  # results, datetime

    def __init__(
        self,
        base_config_provider: Callable[[], str],
        config_model_provider: Callable[[], object],
        worker_client: object = None,
        ttl_seconds: int = 60,
        debounce_ms: int = 350,
        run_in_thread: bool = True,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__()
        self._base_config_provider = base_config_provider
        self._config_model_provider = config_model_provider
        self._worker_client = worker_client
        self._ttl = timedelta(seconds=ttl_seconds)
        self._debounce_ms = debounce_ms
        self._run_in_thread = run_in_thread
        self._pool = QThreadPool.globalInstance()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._execute_pending)

        self._pending_force = False
        self._pending_include_worker = True

        self._last_results: List[CheckResult] = []
        self._last_checked: Optional[datetime] = None
        self._last_include_worker = False

        self._worker_busy = False
        self._doctor_pending = False
        self._deferred_worker = False
        self._pending_local_results: List[CheckResult] = []

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    def run_checks(self, force: bool = False, include_worker: bool = True) -> None:
        """
        Trigger diagnostics. If within TTL and not forced, cached results are re-emitted.
        Requests are debounced to avoid spamming the worker.
        """
        if self._doctor_pending and not force:
            if self._last_results and self._last_checked:
                self.diagnostics_updated.emit(self._last_results, self._last_checked)
            return
        now = self._now_fn()
        if (
            not force
            and self._last_checked
            and not self._doctor_pending
            and now - self._last_checked < self._ttl
            and (not include_worker or self._last_include_worker)
        ):
            self.diagnostics_updated.emit(self._last_results, self._last_checked)
            return

        self._pending_force = force
        self._pending_include_worker = include_worker

        if self._debounce_ms <= 0:
            self._execute_pending()
        else:
            self._debounce_timer.start(self._debounce_ms)

    def handle_busy_changed(self, busy: bool) -> None:
        """Track worker busy state and trigger deferred doctor runs when idle."""
        self._worker_busy = busy
        if not busy and self._deferred_worker:
            self._deferred_worker = False
            self.run_checks(force=True, include_worker=True)

    def handle_worker_doctor(self, checks: List[dict]) -> None:
        """Integrate worker doctor results."""
        doctor_results = [
            CheckResult(name=c.get("name", ""), ok=bool(c.get("ok", False)), detail=c.get("detail", ""))
            for c in checks
        ]
        combined = list(self._pending_local_results) if self._pending_local_results else list(self._last_results)

        # Drop any placeholder worker entries
        combined = [r for r in combined if r.name != "worker"]
        combined.extend(doctor_results)

        self._doctor_pending = False
        self._pending_local_results = []
        self._update_results(combined, include_worker=True)

    def last_results(self) -> List[CheckResult]:
        return list(self._last_results)

    def last_checked(self) -> Optional[datetime]:
        return self._last_checked

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────
    def _execute_pending(self) -> None:
        include_worker = self._pending_include_worker
        force = self._pending_force
        self._pending_force = False
        self._pending_include_worker = True

        base_path = self._base_config_provider() or "config.yaml"

        if not self._run_in_thread:
            results = run_checks(base_path, self._config_model_provider())
            self._on_checks_done(results, include_worker)
            return

        task = _DiagnosticsTask(
            base_config_path=base_path,
            config_model_provider=self._config_model_provider,
            runner=run_checks,
            include_worker=include_worker,
        )
        task.signals.results_ready.connect(self._on_checks_done)
        self._pool.start(task)

    def _on_checks_done(self, results: List[CheckResult], include_worker: bool) -> None:
        combined = list(results)
        self._pending_local_results = combined

        if include_worker and self._worker_client:
            base_path = self._base_config_provider()
            if not getattr(self._worker_client, "is_running", lambda: False)():
                combined.append(CheckResult("worker", False, "Worker not running"))
                self._update_results(combined, include_worker=False)
                return
            if getattr(self._worker_client, "is_busy", lambda: False)():
                combined.append(CheckResult("worker", True, "Skipped (worker busy)"))
                self._deferred_worker = True
                self._update_results(combined, include_worker=False)
                return

            doctor = getattr(self._worker_client, "doctor", None)
            if callable(doctor):
                req_id = doctor(config_path=base_path, overrides={})
                if not req_id:
                    combined.append(CheckResult("worker", False, "Doctor command failed"))
                    self._update_results(combined, include_worker=False)
                    return
                self._doctor_pending = True
                combined.append(CheckResult("worker", True, "Doctor running"))
                self._update_results(combined, include_worker=False)
                return

        self._update_results(combined, include_worker=include_worker)

    def _update_results(self, results: List[CheckResult], include_worker: bool) -> None:
        self._last_results = results
        self._last_include_worker = include_worker or self._last_include_worker
        self._last_checked = self._now_fn()
        if not self._doctor_pending:
            self._pending_local_results = []
        self.diagnostics_updated.emit(self._last_results, self._last_checked)
