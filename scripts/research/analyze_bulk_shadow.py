#!/usr/bin/env python
"""Operational analysis of a bulk shadow run (no gold needed — library albums).

Reports the signals that inform the conservative trust policy: token burn at scale,
floor-firing rate, escalation rate, tightness (genres/release), de-bloat vs the current
authority, taxonomy-gap rate, and a random spot-check sample.

Usage:
  python scripts/research/analyze_bulk_shadow.py [checkpoint.db]
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_album_adjudicator import open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_scoring import distribution  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import canonicalize_proposed  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

DEFAULT_DB = _ROOT / "data" / "genre_adjudication_shadow.db"


def main(argv: list[str]) -> int:
    db = argv[0] if argv else str(DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT album_id, response_json, dropped_file_tags_json, total_tokens, status FROM adjudications"
    ).fetchall()
    conn.close()
    complete = [r for r in rows if r[4] == "complete"]
    failed = [r for r in rows if r[4] != "complete"]
    n = len(complete)
    print(f"checkpoint: {db}")
    print(f"rows={len(rows)}  complete={n}  failed={len(failed)}")
    if not n:
        return 0

    canon = load_graph_adapter().canonicalize_tag
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}

    ngenres, debloat, toks = [], [], []
    escal = floor_fired = gap_rel = 0
    samples = []
    for album_id, resp_json, dropped_json, total_tokens, _ in complete:
        resp = json.loads(resp_json)
        dropped = json.loads(dropped_json) if dropped_json else []
        c = canonicalize_proposed([g["term"] for g in resp["genres"]], canon)
        ngenres.append(len(c["canonical"]))
        if resp.get("escalate"):
            escal += 1
        if dropped:
            floor_fired += 1
        if c["gaps"]:
            gap_rel += 1
        if total_tokens:
            toks.append(float(total_tokens))
        obs = [id2name[g] for (g,) in meta.execute(
            "SELECT genre_id FROM release_effective_genres WHERE album_id=? AND assignment_layer='observed_leaf'",
            (album_id,)) if g in id2name]
        debloat.append(float(len(obs) - len(c["canonical"])))
        samples.append((album_id, c["canonical"], c["gaps"], resp.get("escalate"), dropped, obs))

    print(f"\nescalation rate:        {escal}/{n} ({100*escal/n:.0f}%)")
    print(f"floor fired (file drop): {floor_fired}/{n} ({100*floor_fired/n:.0f}%)")
    print(f"releases with gaps:     {gap_rel}/{n} ({100*gap_rel/n:.0f}%)")
    print(f"tokens/release:         mean={sum(toks)/len(toks):.0f}  total={sum(toks):.0f}" if toks else "tokens: n/a")
    print(f"genres/release:         {distribution(ngenres)}")
    print(f"de-bloat (authority-proposed): {distribution(debloat)}")

    print("\n--- spot check (12 random) ---")
    random.seed(0)
    for album_id, canonical, gaps, esc, dropped, obs in random.sample(samples, min(12, len(samples))):
        row = meta.execute("SELECT artist, title FROM albums WHERE album_id=?", (album_id,)).fetchone()
        at = f"{row[0]} — {row[1]}" if row else album_id
        print(f"  {at[:54]}")
        print(f"     authority={obs}")
        print(f"     proposed ={canonical}"
              + (f"  GAPS={gaps}" if gaps else "")
              + ("  [ESCALATE]" if esc else "")
              + (f"  DROPPED_FILE={dropped}" if dropped else ""))
    meta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
