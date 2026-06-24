"""Stream a child process's output line-by-line with cooperative cancellation.

Used by analyze_library.stage_sonic to run the sonic pool in a separate
(import-light) process while forwarding its progress and honoring cancellation.
Tested with a fake Popen so no real process is spawned.
"""
import pytest

from src.playlist.subprocess_stream import run_streaming_subprocess


class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.killed = False
        self._rc = 0

    def kill(self):
        self.killed = True

    def wait(self):
        return self._rc


def test_forwards_each_stdout_line_and_returns_returncode():
    proc = _FakeProc(["one\n", "two\n", "three\n"])
    collected = []

    rc = run_streaming_subprocess(
        ["anything"],
        on_line=collected.append,
        popen=lambda *a, **k: proc,
    )

    assert collected == ["one", "two", "three"]
    assert rc == 0
    assert proc.killed is False


def test_cancellation_kills_process_and_propagates():
    proc = _FakeProc(["one\n", "two\n", "three\n"])
    state = {"checks": 0}

    class _Cancelled(Exception):
        pass

    def cancel_check():
        state["checks"] += 1
        if state["checks"] == 2:
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        run_streaming_subprocess(
            ["anything"],
            on_line=lambda line: None,
            cancellation_check=cancel_check,
            popen=lambda *a, **k: proc,
        )

    assert proc.killed is True
