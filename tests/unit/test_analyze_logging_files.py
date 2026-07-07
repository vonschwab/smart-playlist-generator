import os
import time
from pathlib import Path

from src.logging_utils import (
    make_analyze_log_path,
    cleanup_old_analyze_logs,
)


def test_make_analyze_log_path_shape(tmp_path):
    p = make_analyze_log_path("abcdef123456", dir=tmp_path)
    assert p.parent == Path(tmp_path)
    assert p.suffix == ".log"
    assert p.name.endswith("_abcdef.log")   # run_id truncated to 6


def test_cleanup_deletes_old_keeps_recent(tmp_path):
    old = tmp_path / "2020-01-01_000000_aaaaaa.log"
    new = tmp_path / "2026-07-06_000000_bbbbbb.log"
    old.write_text("x", encoding="utf-8")
    new.write_text("y", encoding="utf-8")
    old_mtime = time.time() - (40 * 86400)      # 40 days old
    os.utime(old, (old_mtime, old_mtime))

    deleted = cleanup_old_analyze_logs(dir=tmp_path, retention_days=30)
    assert deleted == 1
    assert not old.exists()
    assert new.exists()
