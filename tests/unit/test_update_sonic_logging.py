"""Regression: the sonic thread-pool worker must not mutate the shared root logger.

`analyze_track_worker` historically ran in a subprocess and set the root logger to
WARNING to quiet per-track spam. The sonic pool was migrated to a ThreadPoolExecutor
(scripts/update_sonic.py), so the worker now shares this process's root logger --
mutating it globally silenced every later analyze stage's INFO (and the GUI log).
Root-caused 2026-07-07.
"""
import logging

from scripts.update_sonic import analyze_track_worker


def test_sonic_worker_does_not_mutate_root_log_level():
    root = logging.getLogger()
    prev = root.level
    root.setLevel(logging.DEBUG)
    try:
        # A nonexistent file makes the worker return None early -- but on its way
        # there it must NOT touch the shared root logger (thread pool == shared
        # process logging; a global level change silences later stages).
        result = analyze_track_worker(
            ("tid", "/nonexistent/does-not-exist.flac", "Artist", "Title", False, True)
        )
        assert result is None
        assert root.level == logging.DEBUG, (
            "analyze_track_worker mutated the shared root logger level (now "
            f"{logging.getLevelName(root.level)}); a thread-pool worker must not -- "
            "it silences every later stage's INFO logging."
        )
    finally:
        root.setLevel(prev)
