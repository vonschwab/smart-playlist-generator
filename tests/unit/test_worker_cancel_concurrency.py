"""The worker must hear a cancel while a tracked command is still running.

Regression guard for the single-threaded stdin loop bug: a long-running tracked
command blocked the reader, so the cancel line sat unread until the command had
already finished. The reader now stays free to dispatch untracked commands
(cancel) inline while the tracked command runs on a worker thread.
"""
from __future__ import annotations

import json
import sys
import threading

import src.playlist_gui.worker as worker


def test_cancel_is_delivered_while_tracked_command_runs(monkeypatch):
    started = threading.Event()
    observed_cancel = threading.Event()
    captured: list[dict] = []

    monkeypatch.setattr(worker, "emit_event", lambda event: captured.append(dict(event)))

    def slow_handler(cmd_data):
        # Signal that the request is registered and running, then poll the
        # cancellation flag (a safe checkpoint) until the cancel lands.
        started.set()
        for _ in range(500):
            worker.check_cancelled()  # raises CancellationError once cancelled
            threading.Event().wait(0.01)
        raise AssertionError("cancel was never observed by the running command")

    monkeypatch.setitem(worker.TRACKED_COMMAND_HANDLERS, "slow_noop", slow_handler)

    run_line = json.dumps({"cmd": "slow_noop", "request_id": "r1", "job_id": "j1"})
    cancel_line = json.dumps({"cmd": "cancel", "request_id": "r1"})

    def fake_stdin():
        # Start the tracked command, wait until it is actually running, then
        # deliver the cancel — mirroring a user clicking cancel mid-run.
        yield run_line + "\n"
        assert started.wait(timeout=5), "tracked command never started"
        observed_cancel.set()
        yield cancel_line + "\n"
        # End of input: main() falls through and joins the worker thread.

    monkeypatch.setattr(sys, "stdin", fake_stdin())
    monkeypatch.setattr(worker, "setup_worker_logging", lambda: None)

    worker.main()

    assert observed_cancel.is_set()
    done = [e for e in captured if e.get("type") == "done" and e.get("cmd") == "slow_noop"]
    assert done, f"no done event for slow_noop; events={[e.get('type') for e in captured]}"
    assert done[-1].get("cancelled") is True
    assert done[-1].get("ok") is False
