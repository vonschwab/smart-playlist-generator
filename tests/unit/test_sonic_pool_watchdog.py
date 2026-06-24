"""Watchdog for the sonic-analysis process pool.

Regression guard for the Analyze Library hang root-caused 2026-06-23: pool
children deadlocked re-importing the heavy ``__main__`` during Windows spawn,
and the run hung forever. ``run_pool_with_watchdog`` must (a) run normally when
workers behave, and (b) when the pool makes no progress, kill the stuck workers
and fall back to serial in-process execution so the run always completes.

These tests use REAL process pools (no mocks). The "hang" worker stalls only
when running inside a spawned child (``parent_process() is not None``) and
completes normally in-process, faithfully modelling the spawn-only deadlock.
All sleeps are bounded so a failed kill cannot hang the suite.
"""
import multiprocessing
import time

from scripts.update_sonic import run_pool_with_watchdog


def _ok_worker(x):
    return x * 2


def _hang_in_pool_ok_serial(x):
    # Mirrors the real bug: wedges in a spawned pool child, but works in-process.
    if multiprocessing.parent_process() is not None:
        time.sleep(30)  # bounded so a missed kill can't hang the suite
    return x * 2


def test_normal_path_runs_in_pool_without_fallback():
    items = [1, 2, 3, 4]
    results = {}

    info = run_pool_with_watchdog(
        items,
        _ok_worker,
        workers=2,
        on_result=lambda item, result: results.__setitem__(item, result),
        no_progress_timeout=5.0,
    )

    assert results == {1: 2, 2: 4, 3: 6, 4: 8}
    assert info["serial_fallback"] is False
    assert info["stalls"] == 0


def test_stalled_pool_falls_back_to_serial_after_one_stall():
    items = [1, 2, 3]
    results = {}

    info = run_pool_with_watchdog(
        items,
        _hang_in_pool_ok_serial,
        workers=2,
        on_result=lambda item, result: results.__setitem__(item, result),
        no_progress_timeout=1.0,
    )

    # Every item is processed despite the pool hanging, and recovery is fast:
    # exactly ONE stall, then straight to serial (no slow per-worker retries).
    assert results == {1: 2, 2: 4, 3: 6}
    assert info["serial_fallback"] is True
    assert info["stalls"] == 1
