#!/usr/bin/env python3
"""Validate a published copy of metadata.db. Read-only checks; no live writes."""
import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def validate(meta_db: str) -> int:
    conn = sqlite3.connect(meta_db)
    conn.row_factory = sqlite3.Row
    problems = []

    # every album resolves to exactly one base source (graph xor legacy), or none
    bad = conn.execute(
        "SELECT album_id, COUNT(DISTINCT source) c FROM release_effective_genres "
        "WHERE source != 'user' GROUP BY album_id HAVING c > 1"
    ).fetchall()
    if bad:
        problems.append(f"{len(bad)} albums have >1 base source")

    n_albums = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
    n_resolved = conn.execute(
        "SELECT COUNT(DISTINCT album_id) FROM release_effective_genres").fetchone()[0]
    print(f"albums={n_albums} resolved_albums={n_resolved}")
    by_source = conn.execute(
        "SELECT source, COUNT(DISTINCT album_id) FROM release_effective_genres GROUP BY source"
    ).fetchall()
    print("by source:", [(r[0], r[1]) for r in by_source])

    # spot-checks (only assert if the album is present)
    spot = {
        "a tribe called quest::midnight marauders japanese": {"east_coast_hip_hop", "jazz_rap", "boom_bap"},
        "antonio carlos jobim::wave": {"bossa_nova", "latin_jazz", "mpb"},
    }
    for rk, expected in spot.items():
        got = {r[0] for r in conn.execute(
            "SELECT genre_id FROM release_effective_genres WHERE release_key = ?", (rk,))}
        missing = expected - got
        if got and missing:
            problems.append(f"{rk}: missing {missing}")
        elif got:
            print(f"OK spot-check {rk}: {sorted(expected & got)}")

    conn.close()
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("VALIDATION OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-db", required=True, help="Path to the COPY to validate")
    return validate(p.parse_args(argv).metadata_db)


if __name__ == "__main__":
    sys.exit(main())
