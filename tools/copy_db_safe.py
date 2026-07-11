#!/usr/bin/env python3
"""Atomically copy a SQLite database — the safe alternative to `cp`/`Copy-Item`.

Copying an *open* SQLite DB with a plain file copy can capture a torn WAL state, producing
a snapshot whose table and indexes disagree. That is the root cause of the 2026-07
metadata.db idx_tracks_file_path corruption (see the incident memory / docs). This tool uses
SQLite's online-backup API (a transactionally consistent snapshot even under concurrent
writes) and verifies the copy with integrity_check.

Usage:
    python tools/copy_db_safe.py <dest>                 # copies data/metadata.db -> <dest>
    python tools/copy_db_safe.py <dest> --src <db>      # copy a specific DB
    python tools/copy_db_safe.py <dest> --no-verify     # skip the integrity check

Exit codes: 0 = copied (and, unless --no-verify, verified clean); 1 = copy made but the
SOURCE failed integrity_check (it needs REINDEX/repair — the copy is faithful but corrupt);
2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db_backup import backup_database  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Atomic, verified SQLite DB copy.")
    ap.add_argument("dest", help="destination path for the copy")
    ap.add_argument("--src", default=str(ROOT / "data" / "metadata.db"),
                    help="source DB (default: data/metadata.db)")
    ap.add_argument("--no-verify", action="store_true", help="skip integrity_check on the copy")
    args = ap.parse_args(argv)

    src = Path(args.src)
    dest = Path(args.dest)
    if not src.exists():
        print(f"error: source DB not found: {src}", file=sys.stderr)
        return 2
    if dest.exists():
        print(f"error: destination already exists (refusing to overwrite): {dest}", file=sys.stderr)
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        res = backup_database(src, dest, verify=not args.no_verify)
    except Exception as e:  # IO / sqlite error
        print(f"error: copy failed: {e}", file=sys.stderr)
        return 2

    size_mb = dest.stat().st_size / (1 << 20)
    print(f"copied {src.name} -> {dest}  ({size_mb:.1f} MB)")
    if res.integrity_ok is None:
        print("integrity: not verified (--no-verify)")
        return 0
    if res.integrity_ok:
        print("integrity: ok — copy is a consistent, trustworthy snapshot")
        return 0
    print(f"WARNING: source integrity_check FAILED ({res.integrity_detail}). "
          "The copy is faithful but the SOURCE is corrupt — run REINDEX/repair.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
