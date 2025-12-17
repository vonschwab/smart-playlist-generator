#!/usr/bin/env python3
"""
Build a group file of track_ids from metadata.db to feed into cross_groups diagnostics.

Examples (PowerShell-friendly):
  python scripts/build_track_group.py --artist "Minor Threat" --title-contains "Minor Threat" --out diagnostics/groups/minor_threat.json
  python scripts/build_track_group.py --artist "Green-House" --album "Music For Living Spaces" --out diagnostics/groups/green_house.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build track group JSON from metadata.db")
    parser.add_argument("--db", default="data/metadata.db", help="Path to metadata.db")
    parser.add_argument("--artist", required=True, help="Artist exact or partial match (uses norm_artist LIKE)")
    parser.add_argument("--album", help="Album exact or partial match (matches lower(album) LIKE)")
    parser.add_argument("--title-contains", dest="title_contains", help="Substring match on norm_title")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch (default 1000)")
    parser.add_argument("--out", required=True, help="Output JSON path under diagnostics/groups/")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"metadata db not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    norm_artist = normalize(args.artist)
    norm_album = normalize(args.album) if args.album else None
    norm_title = normalize(args.title_contains) if args.title_contains else None

    clauses = []
    params: List[str] = []
    clauses.append("(norm_artist LIKE ? OR lower(artist) LIKE ?)")
    params.extend([f"%{norm_artist}%", f"%{args.artist.lower()}%"])
    if norm_album:
        clauses.append("lower(album) LIKE ?")
        params.append(f"%{args.album.lower()}%")
    if norm_title:
        clauses.append("(norm_title LIKE ? OR lower(title) LIKE ?)")
        params.extend([f"%{norm_title}%", f"%{args.title_contains.lower()}%"])
    where = " AND ".join(clauses) if clauses else "1=1"

    query = f"""
        SELECT track_id, artist, album, title
        FROM tracks
        WHERE {where}
        LIMIT ?
    """
    params.append(int(args.limit))
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise SystemExit("No tracks matched the query.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"track_id": r[0], "artist": r[1], "album": r[2], "title": r[3]}
        for r in rows
    ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} rows to {out_path}")


if __name__ == "__main__":
    main()
