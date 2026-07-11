"""Safe, atomic SQLite backups + integrity verification.

Why this module exists (2026-07 incident): the pre-write ``metadata.db`` backup was a raw
``shutil.copy2`` of an *open* database. Copying an open SQLite file can capture a torn WAL
state, producing a snapshot whose table and indexes disagree. That is the root cause of the
``idx_tracks_file_path`` corruption that was found propagated into *every* retained ``.bak``
(the torn copy became the working DB, and each subsequent file-copy carried it forward),
undetected for over a month because backups were never verified.

Prevention, two prongs:

* **Atomic copy** — :func:`backup_database` uses SQLite's online-backup API
  (``sqlite3.Connection.backup``), which reads a transactionally consistent snapshot even
  while another connection is writing. It cannot produce a torn copy.
* **Detection** — it then runs ``PRAGMA integrity_check`` on the copy. Because the snapshot
  is consistent, a *not-clean* result means the **source** is corrupt. Since a backup is
  taken before every write, this surfaces corruption within one write cycle instead of
  silently propagating it. A not-clean result is logged loudly (never silently swallowed).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_PathLike = Union[str, Path]


@dataclass(frozen=True)
class BackupResult:
    """Outcome of :func:`backup_database`.

    ``integrity_ok`` is ``True``/``False`` when the copy was verified, or ``None`` when
    verification was skipped. ``integrity_detail`` is ``"ok"``, ``"not-verified"``, or the
    joined ``integrity_check`` messages.
    """

    dest: Path
    integrity_ok: Optional[bool]
    integrity_detail: str


def _classify_integrity(rows: list[str]) -> tuple[bool, str]:
    """Classify ``PRAGMA integrity_check`` output as clean-enough or not.

    Clean: ``["ok"]``, or output whose only complaints are benign ``"Page N: never used"``
    orphan pages (wasted free space, not data loss). Not clean: any index-vs-table or other
    structural error. Returns ``(clean, detail)``.
    """
    if rows == ["ok"]:
        return True, "ok"
    serious = [
        r for r in rows
        if "never used" not in r and r not in ("ok", "*** in database main ***")
    ]
    detail = "; ".join(rows[:10])
    return (len(serious) == 0), detail


def check_integrity(db_path: _PathLike) -> tuple[bool, str]:
    """Run ``PRAGMA integrity_check`` read-only on ``db_path``.

    Returns ``(clean, detail)`` per :func:`_classify_integrity`. Read-only: opens the DB
    with ``mode=ro`` and never writes.
    """
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        rows = [r[0] for r in con.execute("PRAGMA integrity_check").fetchall()]
    finally:
        con.close()
    return _classify_integrity(rows)


def backup_database(src: _PathLike, dest: _PathLike, *, verify: bool = True) -> BackupResult:
    """Atomically back up SQLite database ``src`` to ``dest`` via the online-backup API.

    Unlike a file copy, this reads a transactionally consistent snapshot and cannot capture
    a torn WAL state. The source is opened read-only; only ``dest`` is written. When
    ``verify`` is true, the copy is integrity-checked and a not-clean result (which implies a
    corrupt *source*) is logged as a warning — never swallowed.
    """
    src, dest = Path(src), Path(dest)
    scon = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    dcon = sqlite3.connect(str(dest))
    try:
        with dcon:
            scon.backup(dcon)
    finally:
        dcon.close()
        scon.close()

    if not verify:
        return BackupResult(dest=dest, integrity_ok=None, integrity_detail="not-verified")

    ok, detail = check_integrity(dest)
    if not ok:
        logger.warning(
            "DB backup of %s is NOT CLEAN — the source database is likely corrupt (%s). "
            "This backup is not a trustworthy rollback point; run REINDEX/repair on the source.",
            src.name, detail,
        )
    return BackupResult(dest=dest, integrity_ok=ok, integrity_detail=detail)
