"""Tests for src/db_backup.py — atomic SQLite backups + integrity classification.

Guards the prevention put in place after the 2026-07 metadata.db idx_tracks_file_path
corruption (a torn shutil.copy of an open WAL DB propagated a desynced index into every
backup). backup_database() must produce a consistent copy and flag a not-clean source.
"""
import sqlite3

from src.db_backup import _classify_integrity, backup_database, check_integrity


def _make_db(path, rows: int = 5) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        con.execute("CREATE INDEX idx_v ON t(v)")
        con.executemany("INSERT INTO t (v) VALUES (?)", [(f"row{i}",) for i in range(rows)])
        con.commit()
    finally:
        con.close()


def test_classify_ok():
    ok, detail = _classify_integrity(["ok"])
    assert ok is True and detail == "ok"


def test_classify_benign_orphan_pages():
    # 'never used' orphan pages are wasted space, not data loss -> still clean
    rows = ["*** in database main ***", "Page 5961: never used", "Page 7223: never used"]
    ok, _ = _classify_integrity(rows)
    assert ok is True


def test_classify_index_corruption_is_not_clean():
    rows = [
        "*** in database main ***",
        "row 25469 missing from index idx_tracks_file_path",
        "wrong # of entries in index idx_tracks_file_path",
    ]
    ok, detail = _classify_integrity(rows)
    assert ok is False
    assert "index" in detail


def test_classify_table_corruption_is_not_clean():
    ok, _ = _classify_integrity(["*** in database main ***", "row 5 missing from index sqlite_autoindex_tracks_1"])
    assert ok is False


def test_backup_roundtrip_preserves_data_and_verifies_clean(tmp_path):
    src, dest = tmp_path / "src.db", tmp_path / "dest.db"
    _make_db(src, rows=10)
    res = backup_database(src, dest)
    assert dest.exists()
    assert res.integrity_ok is True
    assert res.integrity_detail == "ok"
    con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 10
    finally:
        con.close()


def test_backup_is_consistent_while_source_has_open_transaction(tmp_path):
    # An uncommitted writer must not leak into the snapshot (online-backup consistency).
    src, dest = tmp_path / "src.db", tmp_path / "dest.db"
    _make_db(src, rows=5)
    writer = sqlite3.connect(str(src))
    try:
        writer.execute("BEGIN")
        writer.execute("INSERT INTO t (v) VALUES ('uncommitted')")  # not committed
        res = backup_database(src, dest)
        assert res.integrity_ok is True
        con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            # snapshot sees the 5 committed rows, never the uncommitted 6th
            assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 5
        finally:
            con.close()
    finally:
        writer.rollback()
        writer.close()


def test_check_integrity_clean_db(tmp_path):
    src = tmp_path / "s.db"
    _make_db(src)
    ok, detail = check_integrity(src)
    assert ok is True and detail == "ok"


def test_backup_verify_false_skips_check(tmp_path):
    src, dest = tmp_path / "s.db", tmp_path / "d.db"
    _make_db(src)
    res = backup_database(src, dest, verify=False)
    assert dest.exists()
    assert res.integrity_ok is None  # not verified
