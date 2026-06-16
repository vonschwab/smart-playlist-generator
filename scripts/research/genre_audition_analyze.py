"""Aggregate genre-audition captures and write findings.

Reads all *_capture.yaml in the data directory, expands each rated entry by
provenance (an entry proposed by both graph and co-occurrence counts for
both), and writes findings.md: verdict distribution and mean score per
provenance, graph-vs-cooccurrence and graph-vs-decoy contrasts, sim-vs-verdict
correlation, and callout lists for bad graph edges and missed co-occurrence
neighbors.

Usage:
    python scripts/genre_audition_analyze.py [--data-dir docs/run_audits/genre_audition]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICT_ORDER = ["same", "related", "loose", "unrelated"]
VERDICT_SCORE = {"same": 3, "related": 2, "loose": 1, "unrelated": 0}
PROVENANCES = ["graph", "cooccurrence", "decoy"]


def load_captures(data_dir: Path) -> List[dict]:
    """Return all entries from every *_capture.yaml, tagged with their seed."""
    entries = []
    for p in sorted(data_dir.glob("*_capture.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        seed = p.stem.replace("_capture", "")
        for e in data.get("entries", []):
            e = dict(e)
            e.setdefault("seed", seed)
            entries.append(e)
    return entries


def aggregate_by_provenance(entries: List[dict]) -> Dict[str, Dict[str, int]]:
    """{provenance: {verdict: count}} over rated entries, expanded by provenance."""
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        verdict = e.get("verdict", "")
        if not verdict:
            continue
        for prov in (e.get("spaces") or {}):
            result[prov][verdict] += 1
    return {k: dict(v) for k, v in result.items()}


def mean_score_by_provenance(entries: List[dict]) -> Dict[str, float]:
    """{provenance: mean verdict score} over rated entries, expanded by provenance."""
    totals: Dict[str, list] = defaultdict(list)
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for prov in (e.get("spaces") or {}):
            totals[prov].append(score)
    return {k: round(float(np.mean(v)), 3) for k, v in totals.items() if v}


def sim_verdict_rows(entries: List[dict]) -> List[dict]:
    """[{provenance, sim, score}] for graph/cooccurrence entries with a sim."""
    rows = []
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for prov, meta in (e.get("spaces") or {}).items():
            if meta and "sim" in meta:
                rows.append({"provenance": prov, "sim": meta["sim"], "score": score})
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/genre_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    entries = load_captures(data_dir)
    if not entries:
        print(f"No capture entries in {data_dir}. Complete the audition first.")
        return

    rated = [e for e in entries if e.get("verdict")]
    if not rated:
        print(f"No rated entries yet in {data_dir}. Rate some candidates first.")
        return
    by_prov = aggregate_by_provenance(entries)
    means = mean_score_by_provenance(entries)
    corr_rows = sim_verdict_rows(entries)

    lines = [
        "# Genre Similarity Audition — Findings",
        "",
        f"Total entries: {len(entries)}  |  Rated: {len(rated)}",
        "",
        "## Verdict Distribution by Provenance",
        "",
        "| Provenance | same | related | loose | unrelated | total | mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for prov in PROVENANCES:
        counts = by_prov.get(prov, {})
        total = sum(counts.values())
        if total == 0:
            continue
        row = [prov] + [str(counts.get(v, 0)) for v in VERDICT_ORDER]
        row += [str(total), f"{means.get(prov, float('nan')):.3f}"]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Headline Contrasts", ""]
    g, c, d = means.get("graph"), means.get("cooccurrence"), means.get("decoy")
    if g is not None and c is not None:
        verdict = "graph WINS" if g > c else ("co-occurrence WINS" if c > g else "TIE")
        lines.append(f"- **Graph vs co-occurrence (Q1):** graph={g:.3f} vs cooc={c:.3f} → **{verdict}** (Δ={g-c:+.3f})")
    if g is not None and d is not None:
        lines.append(f"- **Graph vs decoy (Q2):** graph={g:.3f} vs decoy={d:.3f} → gap={g-d:+.3f} "
                     f"({'discriminative' if g - d > 0.5 else 'WEAK — investigate'})")

    lines += ["", "## Similarity ↔ Verdict Correlation (Pearson r)", ""]
    groups: Dict[str, list] = defaultdict(list)
    for r in corr_rows:
        groups[r["provenance"]].append((r["sim"], r["score"]))
    for prov in ("graph", "cooccurrence"):
        pairs = groups.get(prov, [])
        if len(pairs) < 3:
            lines.append(f"- **{prov}**: too few rated pairs ({len(pairs)})")
            continue
        sims = np.array([p[0] for p in pairs])
        scores = np.array([p[1] for p in pairs])
        if sims.std() == 0 or scores.std() == 0:
            lines.append(f"- **{prov}**: r undefined (no variance), {len(pairs)} pairs")
            continue
        r_val = float(np.corrcoef(sims, scores)[0, 1])
        lines.append(f"- **{prov}**: r={r_val:.3f} ({len(pairs)} pairs, "
                     f"sim [{sims.min():.3f}, {sims.max():.3f}])")

    lines += ["", "## Graph Neighbors Rated `unrelated` (candidate bad edges → SP3a)", ""]
    bad = [e for e in rated if e.get("verdict") == "unrelated" and "graph" in (e.get("spaces") or {})]
    if bad:
        for e in bad:
            lines.append(f"- **{e.get('seed','')}** → `{e['name']}`"
                         + (f" — {e['notes']}" if e.get("notes") else ""))
    else:
        lines.append("*(none)*")

    lines += ["", "## Co-occurrence-only Neighbors Rated `same`/`related` (candidate gaps)", ""]
    gaps = [
        e for e in rated
        if e.get("verdict") in ("same", "related")
        and "cooccurrence" in (e.get("spaces") or {})
        and "graph" not in (e.get("spaces") or {})
    ]
    if gaps:
        for e in gaps:
            lines.append(f"- **{e.get('seed','')}** → `{e['name']}` ({e['verdict']})"
                         + (f" — {e['notes']}" if e.get("notes") else ""))
    else:
        lines.append("*(none)*")

    lines += ["", "## Notable Notes", ""]
    for e in sorted(rated, key=lambda x: VERDICT_SCORE.get(x.get("verdict", ""), 0)):
        if e.get("notes"):
            lines.append(f"- **{e.get('seed','')}** → {e['name']} | {e.get('verdict','')} | {e['notes']}")

    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("Means by provenance:", means)


if __name__ == "__main__":
    main()
