"""Run a child process while streaming its output and honoring cancellation.

Kept dependency-light (stdlib only) so it can be reused and unit-tested without
pulling the heavy analysis stack.
"""
import subprocess
from typing import Callable, Optional, Sequence


def run_streaming_subprocess(
    argv: Sequence[str],
    *,
    on_line: Callable[[str], None],
    cancellation_check: Optional[Callable[[], None]] = None,
    popen: Callable[..., "subprocess.Popen"] = subprocess.Popen,
) -> int:
    """Run ``argv``, delivering each stdout line (stderr merged in) to ``on_line``.

    Before reading each line, ``cancellation_check`` (if given) is called; if it
    raises, the child is killed and the exception propagates. Returns the child's
    exit code once its output stream is exhausted.
    """
    proc = popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        stream = proc.stdout
        if stream is not None:
            for line in stream:
                if cancellation_check is not None:
                    cancellation_check()
                on_line(line.rstrip("\n"))
    except BaseException:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    return proc.wait()
