#!/usr/bin/env python
"""Auto lane: materialize every non-escalated adjudication into the sidecar, and
write a non-blocking diff report for albums that introduced a new genre. Escalated
albums are left for review_escalated.py. Resumable: materialize is idempotent
(replace-by-release-key), so re-running is safe.

Usage:
  python scripts/research/apply_adjudication.py --dry-run   # counts + report only
  python scripts/research/apply_adjudication.py             # write sidecar + report
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adjudicator_bulk import effective_prompt_version  # noqa: E402
from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_materializer import materialize_adjudication  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import canonicalize_proposed  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

REPORT = _ROOT / "docs" / "genre_adjudication" / "phase4_added_genres_report.md"


def best_results(rows, *, thorough_pv):
    """Return best response per album_id — thorough version wins over standard."""
    best: dict = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


def split_lanes(best):
    """Split best results into auto and escalated dicts."""
    auto = {a: r for a, r in best.items() if not r.get("escalate")}
    escalated = {a: r for a, r in best.items() if r.get("escalate")}
    return auto, escalated


def invented_genres(proposed_canonical: list[str], prior_leaf: list[str]) -> list[str]:
    """Return genres in proposed that are not in the prior observed-leaf set."""
    prior = set(prior_leaf)
    return [g for g in proposed_canonical if g not in prior]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    std_pv = effective_prompt_version(thorough=False)
    tho_pv = effective_prompt_version(thorough=True)

    with sqlite3.connect(f"file:{args.shadow_db}?mode=ro", uri=True) as conn:
        raw = [
            (a, pv, json.loads(rj))
            for a, pv, rj in conn.execute(
                "SELECT album_id, prompt_version, response_json FROM adjudications WHERE status='complete'"
            )
        ]

    best = best_results(raw, thorough_pv=tho_pv)
    auto, escalated = split_lanes(best)

    taxonomy = load_default_layered_taxonomy()
    adapter = load_graph_adapter()
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {
        r[0]: r[1]
        for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")
    }
    store = None if args.dry_run else SidecarStore(str(resolve_db("ai_genre_enrichment.db")))

    added_rows, materialized = [], 0
    for album_id, resp in auto.items():
        ev = build_evidence(meta, album_id, id2name)
        canon = canonicalize_proposed(
            [g["term"] for g in resp["genres"]], adapter.canonicalize_tag
        )["canonical"]
        prior = ev["current_observed_leaf"]
        new = invented_genres(canon, prior)
        if new:
            added_rows.append((ev["artist"], ev["album"], prior, canon, new))
        if not args.dry_run:
            materialize_adjudication(
                store,
                album_id=album_id,
                artist=ev["artist"],
                album=ev["album"],
                response=resp,
                taxonomy=taxonomy,
                prompt_version=std_pv,
                model="haiku",  # provenance only; not load-bearing
            )
            materialized += 1

    meta.close()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Phase-4 added-genre diff report ({len(added_rows)} albums)\n",
        "Non-blocking. Auto-published albums where the adjudicator added >=1 genre",
        "not in the prior authority. Skim for invented-genre errors.\n",
    ]
    for artist, album, prior, proposed, new in sorted(added_rows):
        lines.append(
            f"- **{artist} — {album}**  NEW={new}\n"
            f"    prior={prior}\n    proposed={proposed}"
        )
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"auto={len(auto)} escalated={len(escalated)} materialized={materialized} "
        f"added_genre_albums={len(added_rows)}{' (dry-run)' if args.dry_run else ''}"
    )
    print(f"diff report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
