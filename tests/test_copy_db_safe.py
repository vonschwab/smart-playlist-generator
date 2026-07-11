"""Tests for tools/copy_db_safe.py — the blessed atomic DB-copy CLI."""
import sqlite3
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "copy_db_safe.py"


def _make_db(p: Path, rows: int = 2) -> None:
    con = sqlite3.connect(str(p))
    try:
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        con.execute("CREATE INDEX ix ON t(v)")
        con.executemany("INSERT INTO t (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
        con.commit()
    finally:
        con.close()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, timeout=60)


def test_copy_ok_and_verified(tmp_path):
    src, dest = tmp_path / "s.db", tmp_path / "d.db"
    _make_db(src, rows=7)
    r = _run(str(dest), "--src", str(src))
    assert r.returncode == 0, r.stderr
    assert dest.exists()
    con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 7
    finally:
        con.close()


def test_refuses_to_overwrite(tmp_path):
    src, dest = tmp_path / "s.db", tmp_path / "d.db"
    _make_db(src)
    dest.write_text("existing")
    r = _run(str(dest), "--src", str(src))
    assert r.returncode == 2
    assert "already exists" in r.stderr


def test_missing_source(tmp_path):
    r = _run(str(tmp_path / "d.db"), "--src", str(tmp_path / "nope.db"))
    assert r.returncode == 2
    assert "not found" in r.stderr
