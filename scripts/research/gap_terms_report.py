#!/usr/bin/env python
"""Aggregate taxonomy-gap terms across the bulk adjudication shadow DB.

A term is a GAP if classify_layered_term (the materializer's own check) returns
term_kind 'review' or canonical_id None — i.e. it would be SKIPPED (never invented)
when applying. This is the precise set that taxonomy growth must cover before the
Pass-2 results can land. Read-only.

Usage:
  python scripts/research/gap_terms_report.py            # all best-results
  python scripts/research/gap_terms_report.py --min 3    # only terms on >=3 albums
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adjudicator_bulk import effective_prompt_version  # noqa: E402

from src.ai_genre_enrichment.layered_assignment import classify_layered_term  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402


def best_results(rows, *, thorough_pv):
    best: dict = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--min", type=int, default=1, help="min album frequency to list")
    args = ap.parse_args()

    tho_pv = effective_prompt_version(thorough=True)
    with sqlite3.connect(f"file:{Path(args.shadow_db).as_posix()}?mode=ro", uri=True) as conn:
        rows = [
            (a, pv, json.loads(rj))
            for a, pv, rj in conn.execute(
                "SELECT album_id, prompt_version, response_json FROM adjudications "
                "WHERE status='complete'"
            )
        ]
    best = best_results(rows, thorough_pv=tho_pv)

    taxonomy = load_default_layered_taxonomy()
    freq: Counter = Counter()
    cooc: dict[str, Counter] = defaultdict(Counter)
    escal_only: dict[str, int] = defaultdict(int)
    releases_with_gap = 0
    total = len(best)

    for resp in best.values():
        proposed = [g["term"] for g in resp.get("genres", [])]
        resolved, gaps = [], []
        for term in proposed:
            cls = classify_layered_term(taxonomy, term, context_terms=proposed)
            if cls.term_kind == "review" or cls.canonical_id is None:
                gaps.append(term)
            else:
                resolved.append(term)
        if gaps:
            releases_with_gap += 1
        is_esc = bool(resp.get("escalate"))
        for term in gaps:
            freq[term] += 1
            if is_esc:
                escal_only[term] += 1
            for r in resolved:
                cooc[term][r] += 1

    print(f"taxonomy={taxonomy.version}")
    print(f"best-results albums: {total}")
    print(f"releases with >=1 gap term: {releases_with_gap} ({100*releases_with_gap/total:.0f}%)")
    print(f"unique gap terms: {len(freq)} (total occurrences {sum(freq.values())})\n")
    print(f"{'freq':>5} {'esc':>4}  term  ->  top co-occurring resolved genres")
    for term, n in freq.most_common():
        if n < args.min:
            continue
        top = ", ".join(g for g, _ in cooc[term].most_common(4))
        print(f"{n:>5} {escal_only[term]:>4}  {term!r}  ->  {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
