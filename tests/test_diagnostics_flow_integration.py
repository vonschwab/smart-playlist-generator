from datetime import datetime, timedelta, timezone

from src.playlist_gui.diagnostics.manager import DiagnosticsManager
from src.playlist_gui.diagnostics.checks import CheckResult


class FakeWorker:
    def __init__(self):
        self.busy = False
        self.doctor_calls = 0
        self.last_config = None

    def is_running(self):
        return True

    def is_busy(self):
        return self.busy

    def doctor(self, config_path=None, overrides=None):
        self.doctor_calls += 1
        self.last_config = config_path
        return f"req-{self.doctor_calls}"


def test_diagnostics_flow_debounce_and_busy(monkeypatch):
    call_count = 0

    def fake_run(base, model):
        nonlocal call_count
        call_count += 1
        return [CheckResult("config", True, "ok")]

    monkeypatch.setattr("src.playlist_gui.diagnostics.manager.run_checks", fake_run)

    now = [datetime(2024, 1, 1, tzinfo=timezone.utc)]

    def now_fn():
        return now[0]

    worker = FakeWorker()
    manager = DiagnosticsManager(
        base_config_provider=lambda: "config.yaml",
        config_model_provider=lambda: None,
        worker_client=worker,
        ttl_seconds=60,
        debounce_ms=0,
        run_in_thread=False,
        now_fn=now_fn,
    )

    results = []
    times = []
    manager.diagnostics_updated.connect(lambda res, ts: (results.append(res), times.append(ts)))

    # First run triggers worker doctor
    manager.run_checks(force=True, include_worker=True)
    assert worker.doctor_calls == 1
    assert call_count == 1

    # Within TTL, should reuse cache (no new worker call)
    manager.run_checks()
    assert worker.doctor_calls == 1
    assert call_count == 1

    # Simulate time passing beyond TTL
    now[0] = now[0] + timedelta(seconds=120)
    worker.busy = True
    manager.run_checks(force=True, include_worker=True)
    assert worker.doctor_calls == 1  # skipped while busy
    assert any(r.name == "worker" for r in manager.last_results())

    # When busy clears, deferred doctor runs once
    worker.busy = False
    manager.handle_busy_changed(False)
    assert worker.doctor_calls == 2
