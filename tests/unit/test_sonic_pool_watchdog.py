"""Thread pool for sonic analysis.

Regression guard for the Analyze Library hang root-caused 2026-06-23: a *process*
pool's children deadlocked re-importing the heavy ``__main__`` during Windows
spawn (numpy C-extension loader-lock race), hanging the run forever. The fix
(2026-06-24) runs the workers in a THREAD pool instead: threads share the
already-imported modules, so that import deadlock cannot occur, while the
librosa/numba DSP releases the GIL so analysis still parallelizes (measured
~2.4x on 6 workers).

``run_pool_with_watchdog`` must therefore (a) process every item, (b) run the
items concurrently (not serially), and (c) isolate a worker that raises so one
bad track does not sink the run. A no-progress heartbeat may still log a
diagnostic warning, but there is no kill/serial-fallback recovery -- threads
have nothing to deadlock on.

These tests use a REAL thread pool (no mocks). All sleeps are short and bounded.
"""
import time

from scripts.update_sonic import run_pool_with_watchdog


def _ok_worker(x):
    return x * 2


def _sleep_worker(x):
    time.sleep(0.4)  # releases the GIL -> concurrent threads overlap this wait
    return x * 2


def _raise_on_two(x):
    if x == 2:
        raise ValueError("boom")
    return x * 2


def test_thread_pool_processes_all_items():
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
    assert info["completed"] == 4
    assert info["stalls"] == 0
    assert info["serial_fallback"] is False


def test_thread_pool_runs_concurrently():
    # 4 items x 0.4s each = 1.6s if serial. With 4 worker threads they overlap,
    # so wall time must be well under the serial sum -- proves real concurrency.
    items = [1, 2, 3, 4]
    results = {}

    start = time.monotonic()
    info = run_pool_with_watchdog(
        items,
        _sleep_worker,
        workers=4,
        on_result=lambda item, result: results.__setitem__(item, result),
        no_progress_timeout=5.0,
    )
    elapsed = time.monotonic() - start

    assert results == {1: 2, 2: 4, 3: 6, 4: 8}
    assert info["completed"] == 4
    assert elapsed < 1.0, f"expected concurrent (<1.0s), got {elapsed:.2f}s"


def test_worker_exception_is_isolated():
    # A worker that raises must not sink the run: that item yields None, the
    # rest succeed, and every item is still delivered to on_result.
    items = [1, 2, 3]
    results = {}

    info = run_pool_with_watchdog(
        items,
        _raise_on_two,
        workers=2,
        on_result=lambda item, result: results.__setitem__(item, result),
        no_progress_timeout=5.0,
    )

    assert results == {1: 2, 2: None, 3: 6}
    assert info["completed"] == 3
    assert info["serial_fallback"] is False
