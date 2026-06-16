#!/usr/bin/env python3
"""
Consolidate stranded enrichment into the canonical DB.

Enrichment from earlier scaled/themed runs landed in per-run ``ai_genre_*_test.db``
files instead of the canonical ``data/ai_genre_enrichment.db`` the artifact builds
from. This copies any enriched_genre_signatures (+ their enriched_genres rows) whose
release_key is NOT already in the canonical DB. Additive only — never overwrites or
deletes canonical rows. Backs the canonical DB up before writing.

Dry-run by default; --apply to write.

    python scripts/consolidate_enrichment_dbs.py            # report
    python scripts/consolidate_enrichment_dbs.py --apply    # merge (with .bak)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sqlite3
from datetime import datetime

CANON = "data/ai_genre_enrichment.db"
# Obvious throwaway fixtures — skipped by default (override with --include-fixtures).
SKIP_SUBSTR = ("skill_smoke", "workflow_test", "refinement")


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default=CANON)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-fixtures", action="store_true",
                    help="also merge skill_smoke/workflow fixture DBs")
    args = ap.parse_args()

    canon = sqlite3.connect(args.canon)
    canon.row_factory = sqlite3.Row
    canon_sig = {r["release_key"] for r in canon.execute(
        "SELECT release_key FROM enriched_genre_signatures")}
    print(f"canonical: {len(canon_sig)} signature release_keys")

    # newest first so a release present in multiple runs takes the latest
    others = [p for p in sorted(glob.glob("data/ai_genre_*.db"), key=os.path.getmtime, reverse=True)
              if os.path.basename(p) != os.path.basename(args.canon)]
    if not args.include_fixtures:
        others = [p for p in others if not any(s in os.path.basename(p) for s in SKIP_SUBSTR)]

    eg_cols = [r[1] for r in canon.execute("PRAGMA table_info(enriched_genres)")]
    eg_insert_cols = [c for c in eg_cols if c != "enriched_genre_id"]
    sig_cols = [r[1] for r in canon.execute("PRAGMA table_info(enriched_genre_signatures)")]

    merged_keys: set[str] = set()
    plan: list[tuple[str, str]] = []  # (release_key, source_db)
    sig_rows: dict[str, sqlite3.Row] = {}
    eg_rows: dict[str, list[sqlite3.Row]] = {}

    for p in others:
        src = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        tables = _tables(src)
        if "enriched_genre_signatures" not in tables:
            src.close()
            continue
        if "enriched_genres" not in tables:
            print(f"warning: skipping {os.path.basename(p)}: missing required table enriched_genres")
            src.close()
            continue
        for r in src.execute("SELECT * FROM enriched_genre_signatures"):
            rk = r["release_key"]
            if rk in canon_sig or rk in merged_keys:
                continue
            merged_keys.add(rk)
            plan.append((rk, os.path.basename(p)))
            sig_rows[rk] = r
            eg_rows[rk] = list(src.execute(
                "SELECT * FROM enriched_genres WHERE release_key = ?", (rk,)))
        src.close()

    print(f"\nWould merge {len(plan)} stranded signature release_keys:")
    by_db: dict[str, int] = {}
    for rk, db in plan:
        by_db[db] = by_db.get(db, 0) + 1
    for db, n in sorted(by_db.items()):
        print(f"  {db:46} +{n}")
    print("  examples:", [rk for rk, _ in plan[:6]])

    if not args.apply:
        canon.close()
        print("\nDRY RUN — no writes. Re-run with --apply.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{args.canon}.bak_{ts}"
    shutil.copy2(args.canon, bak)
    print(f"\nbacked up canonical -> {bak}")

    n_sig = n_eg = 0
    for rk, _ in plan:
        s = sig_rows[rk]
        canon.execute(
            f"INSERT INTO enriched_genre_signatures ({','.join(sig_cols)}) "
            f"VALUES ({','.join('?' for _ in sig_cols)})",
            [s[c] if c in s.keys() else None for c in sig_cols])
        n_sig += 1
        for g in eg_rows[rk]:
            vals = []
            for c in eg_insert_cols:
                # null out source refs that point at rows not present in canonical
                vals.append(
                    None
                    if c in ("source_tag_id", "source_page_id") or c not in g.keys()
                    else g[c]
                )
            canon.execute(
                f"INSERT INTO enriched_genres ({','.join(eg_insert_cols)}) "
                f"VALUES ({','.join('?' for _ in eg_insert_cols)})", vals)
            n_eg += 1
    canon.commit()
    new_total = canon.execute("SELECT COUNT(*) FROM enriched_genre_signatures").fetchone()[0]
    canon.close()
    print(f"merged {n_sig} signatures + {n_eg} enriched_genres rows. "
          f"canonical signatures now: {new_total}")
    print("NEXT: refresh artifact + rebuild sidecar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
