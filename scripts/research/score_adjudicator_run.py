#!/usr/bin/env python
"""Phase-2 scorer: score one or more adjudicator shadow runs against gold.

For each release, compares two arms to gold (metrics.md protocol):
  A = Claude's proposed set     B = current published authority (observed_leaf)
on canonical-equivalent match keys, and reports per-bucket DISTRIBUTIONS
(min/p10/p50/p90 — never just means; the worst release defines trust).

Usage:
  python scripts/research/score_adjudicator_run.py docs/genre_adjudication/shadow/run_haiku_*.json [run_sonnet_*.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.ai_genre_enrichment.adjudication_scoring import (  # noqa: E402
    distribution,
    match_keys,
    preservation,
    set_metrics,
)
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

BUCKETS = ["failure", "control", "sparse", "ALL"]


def score_run(path: Path, canon) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # per (bucket, arm) -> dict of metric -> list[float]
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    escalations = defaultdict(int)
    counts = defaultdict(int)
    n_failed = 0
    for rec in data["results"]:
        prop = rec.get("proposed") or {}
        if "genres" not in prop:  # failed/dry call
            n_failed += 1
            continue
        gk = match_keys(rec["gold_genres"], canon)
        mk = match_keys(rec.get("must_preserve", []), canon)
        pk = match_keys([g["term"] for g in prop["genres"]], canon)
        bk = match_keys(rec.get("current_observed_leaf", []), canon)
        arms = {"claude": (pk, prop.get("escalate", False)), "authority": (bk, None)}
        for bucket in (rec["bucket"], "ALL"):
            counts[bucket] += 1
            if arms["claude"][1]:
                escalations[bucket] += 1
            for arm, (keys, _esc) in arms.items():
                m = set_metrics(keys, gk)
                a = acc[(bucket, arm)]
                a["precision"].append(m["precision"])
                a["recall"].append(m["recall"])
                a["f1"].append(m["f1"])
                a["noise_rate"].append(m["noise_rate"])
                a["preservation"].append(preservation(keys, mk))
                a["n_proposed"].append(float(m["n_proposed"]))
    return {"acc": acc, "escalations": escalations, "counts": counts, "n_failed": n_failed,
            "model": data.get("model"), "elapsed_s": data.get("elapsed_s")}


def fmt(d: dict[str, float]) -> str:
    return f"p50={d['p50']:.2f} p10={d['p10']:.2f} min={d['min']:.2f}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    canon = load_graph_adapter().canonicalize_tag
    for raw in argv:
        path = Path(raw)
        r = score_run(path, canon)
        print("=" * 78)
        print(f"RUN: {path.name}   model={r['model']}   elapsed={r['elapsed_s']}s   failed={r['n_failed']}")
        print("=" * 78)
        for bucket in BUCKETS:
            n = r["counts"].get(bucket, 0)
            if not n:
                continue
            esc = r["escalations"].get(bucket, 0)
            print(f"\n[{bucket}]  N={n}  escalations={esc} ({100*esc/n:.0f}%)")
            for metric in ("noise_rate", "precision", "recall", "preservation"):
                cl = distribution(r["acc"][(bucket, "claude")][metric])
                au = distribution(r["acc"][(bucket, "authority")][metric])
                tag = "  (lower=better)" if metric == "noise_rate" else ""
                print(f"   {metric:13s}{tag}")
                print(f"       claude    {fmt(cl)}")
                print(f"       authority {fmt(au)}")
            npc = distribution(r["acc"][(bucket, "claude")]["n_proposed"])["p50"]
            npa = distribution(r["acc"][(bucket, "authority")]["n_proposed"])["p50"]
            print(f"   genres/release (p50): claude={npc:.0f}  authority={npa:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
