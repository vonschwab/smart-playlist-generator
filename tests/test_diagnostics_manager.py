from src.playlist_gui.diagnostics.manager import DiagnosticsManager
from src.playlist_gui.diagnostics.checks import CheckResult


def test_diagnostics_ttl_cache(monkeypatch):
    calls = []

    def fake_run(base, model):
        calls.append(base)
        return [CheckResult("config", True, "ok")]

    monkeypatch.setattr("src.playlist_gui.diagnostics.manager.run_checks", fake_run)

    now = [0]

    def now_fn():
        now[0] += 1
        from datetime import datetime, timezone

        return datetime.fromtimestamp(now[0], tz=timezone.utc)

    manager = DiagnosticsManager(
        base_config_provider=lambda: "config.yaml",
        config_model_provider=lambda: None,
        worker_client=None,
        ttl_seconds=60,
        debounce_ms=0,
        run_in_thread=False,
        now_fn=now_fn,
    )

    manager.run_checks(force=True)
    manager.run_checks()

    assert len(calls) == 1
    assert manager.last_results()[0].ok


def test_diagnostics_skips_worker_when_busy(monkeypatch):
    calls = []

    def fake_run(base, model):
        calls.append(base)
        return [CheckResult("config", True, "ok")]

    monkeypatch.setattr("src.playlist_gui.diagnostics.manager.run_checks", fake_run)

    class StubWorker:
        def __init__(self):
            self.busy = True
            self.doctor_calls = 0

        def is_running(self):
            return True

        def is_busy(self):
            return self.busy

        def doctor(self, config_path=None, overrides=None):
            self.doctor_calls += 1
            return "req"

    worker = StubWorker()
    manager = DiagnosticsManager(
        base_config_provider=lambda: "config.yaml",
        config_model_provider=lambda: None,
        worker_client=worker,
        ttl_seconds=60,
        debounce_ms=0,
        run_in_thread=False,
    )

    manager.run_checks(force=True, include_worker=True)
    assert worker.doctor_calls == 0  # busy

    worker.busy = False
    manager.handle_busy_changed(False)

    assert worker.doctor_calls == 1  # deferred when idle
    assert len(calls) == 2  # run twice (initial + deferred)
