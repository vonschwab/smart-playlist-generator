"""Tests for per-playlist DEBUG log file helpers (logging_utils.py).

Covers: playlist_log_dir, make_playlist_log_path, playlist_log_file,
cleanup_old_playlist_logs, cleanup_old_playlist_logs_async.

See docs/superpowers/specs/2026-07-02-per-playlist-logging-design.md.
"""
import logging
import os
import tempfile
import time
from pathlib import Path

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.logging_utils import (
    playlist_log_dir,
    make_playlist_log_path,
    playlist_log_file,
    cleanup_old_playlist_logs,
    cleanup_old_playlist_logs_async,
    _PLAYLIST_HANDLER_TAG,
)


class TestPlaylistLogDir:
    """playlist_log_dir() is ROOT-anchored, independent of cwd."""

    def test_root_anchored(self):
        d = playlist_log_dir()
        assert d == ROOT_DIR / "logs" / "playlists"


class TestMakePlaylistLogPath:
    """Naming shape, sanitization, and uniqueness."""

    def test_naming_shape(self, tmp_path):
        path = make_playlist_log_path("Radiohead", "abcdef1234", dir=tmp_path)
        assert path.parent == tmp_path
        assert path.suffix == ".log"
        # <YYYY-MM-DD_HHMMSS>_<artist>_<shortid>.log
        parts = path.stem.split("_")
        assert len(parts) >= 4  # date, time, artist(+), shortid
        assert "Radiohead" in path.name
        # shortid truncated to first 6 chars of request_id
        assert path.stem.endswith("_abcdef")

    def test_artist_sanitization_strips_unsafe_chars(self, tmp_path):
        path = make_playlist_log_path("AC/DC: Live!", "req-123456", dir=tmp_path)
        assert "/" not in path.name
        assert ":" not in path.name
        assert "!" not in path.name
        assert " " not in path.name

    def test_artist_length_is_capped(self, tmp_path):
        long_artist = "A" * 100
        path = make_playlist_log_path(long_artist, "req123456", dir=tmp_path)
        # Full 100-char run must not survive; capped variant (<=~40) should.
        assert ("A" * 100) not in path.name
        assert ("A" * 41) not in path.name

    def test_fallback_shortid_when_no_request_id(self, tmp_path):
        path = make_playlist_log_path("NoId", None, dir=tmp_path)
        assert path.name.endswith(".log")
        assert path.exists() is False  # path construction doesn't create the file

    def test_two_calls_without_request_id_are_unique(self, tmp_path):
        p1 = make_playlist_log_path("Same Artist", None, dir=tmp_path)
        p2 = make_playlist_log_path("Same Artist", None, dir=tmp_path)
        assert p1 != p2

    def test_different_request_ids_produce_different_shortids(self, tmp_path):
        # Different request_ids (differing within the first 6 chars) -> different path.
        p1 = make_playlist_log_path("Artist", "aaa111", dir=tmp_path)
        p2 = make_playlist_log_path("Artist", "bbb222", dir=tmp_path)
        assert p1 != p2

    def test_default_dir_is_playlist_log_dir(self):
        path = make_playlist_log_path("Artist", "req000001")
        assert path.parent == playlist_log_dir()


class TestPlaylistLogFile:
    """The playlist_log_file context manager."""

    def test_enabled_attaches_debug_handler_and_writes_debug_record(self, tmp_path):
        root = logging.getLogger()
        before_handlers = set(root.handlers)
        test_logger = logging.getLogger("test_playlist_log_file_enabled")
        test_logger.setLevel(logging.DEBUG)

        with playlist_log_file("Test Artist", "req000001", enabled=True, dir=tmp_path) as path:
            assert path is not None
            assert path.exists()

            tagged = [h for h in root.handlers if getattr(h, _PLAYLIST_HANDLER_TAG, False)]
            assert len(tagged) == 1
            assert tagged[0].level == logging.DEBUG

            test_logger.debug("hello from playlist log test")
            for h in root.handlers:
                h.flush()

            content = path.read_text(encoding="utf-8")
            assert "hello from playlist log test" in content

        # Handler removed from root after the block exits.
        after_handlers = set(root.handlers)
        assert after_handlers == before_handlers
        assert not any(getattr(h, _PLAYLIST_HANDLER_TAG, False) for h in root.handlers)

        # Handler is actually closed (file not locked) -- deletable on Windows.
        path.unlink()

    def test_disabled_attaches_nothing_and_yields_none(self, tmp_path):
        root = logging.getLogger()
        before = list(root.handlers)

        with playlist_log_file("Test Artist", "req000002", enabled=False, dir=tmp_path) as path:
            assert path is None
            assert list(root.handlers) == before

        assert list(root.handlers) == before
        assert list(tmp_path.iterdir()) == []

    def test_bad_dir_does_not_raise_and_yields_none(self):
        # Point "dir" at an existing plain file, so mkdir(dir) must fail.
        fd, bad_dir_name = tempfile.mkstemp()
        os.close(fd)
        bad_dir = Path(bad_dir_name)
        try:
            with playlist_log_file("Test Artist", "req000003", enabled=True, dir=bad_dir) as path:
                assert path is None
        finally:
            try:
                bad_dir.unlink()
            except OSError:
                pass

    def test_exception_in_body_still_tears_down_handler(self, tmp_path):
        root = logging.getLogger()
        before_handlers = set(root.handlers)

        try:
            with playlist_log_file("Test Artist", "req000004", enabled=True, dir=tmp_path):
                raise ValueError("boom")
        except ValueError:
            pass

        assert set(root.handlers) == before_handlers


class TestCleanupOldPlaylistLogs:
    """cleanup_old_playlist_logs / cleanup_old_playlist_logs_async."""

    def test_deletes_only_files_older_than_retention(self, tmp_path):
        old_file = tmp_path / "old.log"
        new_file = tmp_path / "new.log"
        old_file.write_text("old")
        new_file.write_text("new")

        now = time.time()
        old_time = now - 31 * 86400
        os.utime(old_file, (old_time, old_time))
        os.utime(new_file, (now, now))

        count = cleanup_old_playlist_logs(dir=tmp_path, retention_days=30)

        assert count == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_missing_dir_returns_zero_and_does_not_raise(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert cleanup_old_playlist_logs(dir=missing) == 0

    def test_ignores_non_log_files(self, tmp_path):
        old_txt = tmp_path / "old.txt"
        old_txt.write_text("old")
        old_time = time.time() - 31 * 86400
        os.utime(old_txt, (old_time, old_time))

        count = cleanup_old_playlist_logs(dir=tmp_path, retention_days=30)
        assert count == 0
        assert old_txt.exists()

    def test_async_runs_in_background_thread_and_does_not_raise(self, tmp_path):
        old_file = tmp_path / "old.log"
        old_file.write_text("old")
        old_time = time.time() - 31 * 86400
        os.utime(old_file, (old_time, old_time))

        cleanup_old_playlist_logs_async(dir=tmp_path, retention_days=30)

        # Give the daemon thread a moment to run, then verify it did the work.
        deadline = time.time() + 2.0
        while old_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not old_file.exists()
