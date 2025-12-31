import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import scripts.fetch_mbids_musicbrainz as fetch_mbids
import src.logging_utils as logging_utils


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_progress_logger_default_emits_summaries_only(monkeypatch):
    handler = ListHandler()
    logger = logging.getLogger("progress_default")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    t = {"value": 0.0}

    def fake_perf_counter():
        t["value"] += 0.1
        return t["value"]

    monkeypatch.setattr(logging_utils.time, "perf_counter", fake_perf_counter)

    prog = logging_utils.ProgressLogger(
        logger,
        total=100,
        label="test",
        interval_s=100.0,  # Avoid time-based emission
        every_n=50,
    )
    for _ in range(100):
        prog.update()
    prog.finish()

    # Expect a couple of summaries (at 50, 100, finish) but not one per item
    assert len(handler.records) <= 4
    assert any(rec.levelno == logging.INFO for rec in handler.records)


def test_progress_logger_verbose_emits_each(monkeypatch):
    handler = ListHandler()
    logger = logging.getLogger("progress_verbose")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    t = {"value": 0.0}

    def fake_perf_counter():
        val = t["value"]
        t["value"] += 1.0
        return val

    monkeypatch.setattr(logging_utils.time, "perf_counter", fake_perf_counter)

    prog = logging_utils.ProgressLogger(
        logger,
        total=10,
        label="test",
        verbose_each=True,
        interval_s=1.0,
        every_n=5,
    )
    for i in range(10):
        prog.update(detail=f"/tmp/file{i}.flac")
    prog.finish()

    debug_records = [r for r in handler.records if r.levelno == logging.DEBUG]
    info_records = [r for r in handler.records if r.levelno == logging.INFO]
    assert len(debug_records) == 10
    assert info_records  # final summary present


def test_human_time_formatting():
    assert logging_utils._human_time(12) == "12s"
    assert logging_utils._human_time(194) == "3m14s"
    assert logging_utils._human_time(3720) == "1h02m"
    assert logging_utils._human_time(190000) == "2d4h"


def test_scan_library_help_includes_progress_flags():
    result = subprocess.run(
        [sys.executable, "scripts/scan_library.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.lower()
    assert "--progress-interval" in output
    assert "--verbose" in output


def test_update_sonic_help_includes_progress_flags():
    result = subprocess.run(
        [sys.executable, "scripts/update_sonic.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.lower()
    assert "--progress-every" in output
    assert "--verbose" in output


def test_fetch_mbids_verbose_sets_progress_verbose(monkeypatch, tmp_path):
    called = {}

    class FakeProgress:
        def __init__(self, logger, total, label, unit, interval_s, every_n, verbose_each):
            called["verbose_each"] = verbose_each

        def update(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    monkeypatch.setattr(fetch_mbids, "ProgressLogger", FakeProgress)
    monkeypatch.setattr(fetch_mbids, "get_session", lambda timeout, max_retries: object())
    monkeypatch.setattr(fetch_mbids, "search_recording_relaxed", lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch_mbids.time, "sleep", lambda *args, **kwargs: None)

    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tracks (
            track_id TEXT,
            title TEXT,
            artist TEXT,
            duration_ms INTEGER,
            musicbrainz_id TEXT,
            mbid_status TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO tracks (track_id, title, artist, duration_ms, musicbrainz_id, mbid_status) VALUES (?, ?, ?, ?, ?, ?)",
        ("t1", "Song One", "Artist", 180000, None, None),
    )
    conn.commit()
    conn.close()

    args = fetch_mbids.parse_args(
        ["--db", str(db_path), "--limit", "1", "--verbose", "--progress-interval", "0.01", "--progress-every", "1"]
    )
    args.log_file = str(tmp_path / "mbid.log")
    monkeypatch.setattr(fetch_mbids, "parse_args", lambda: args)

    fetch_mbids.main()
    assert called.get("verbose_each") is True
