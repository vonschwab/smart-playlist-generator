"""Aggregate sonic audition capture files and write findings.

Reads all *_capture.yaml files in the data directory, computes per-space
verdict distributions and cosine-vs-verdict correlation, reports on
negative-S pairs, and writes findings.md.

Usage:
    python scripts/sonic_audition_analyze.py [--data-dir docs/run_audits/sonic_audition]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERDICT_ORDER = ["match", "close", "off", "wrong"]
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}
SPACES = ["full_track", "production_transition", "rhythm", "timbre", "harmony"]


def load_captures(data_dir: Path) -> List[dict]:
    """Return all entries from every *_capture.yaml in data_dir."""
    entries = []
    for p in sorted(data_dir.glob("*_capture.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        slug = p.stem.replace("_capture", "")
        for e in data.get("entries", []):
            e.setdefault("seed", slug)
            entries.append(e)
    return entries


def aggregate_by_space(entries: List[dict]) -> Dict[str, Dict[str, int]]:
    """Return {space: {verdict: count}} for entries with a verdict and space data."""
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        verdict = e.get("verdict", "")
        if not verdict:
            continue
        for space in (e.get("spaces") or {}):
            result[space][verdict] += 1
    return {k: dict(v) for k, v in result.items()}


def cosine_verdict_correlation(entries: List[dict]) -> List[dict]:
    """Return [{space, cosine, score}, ...] for all rated entries with cosine data."""
    rows = []
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict not in VERDICT_SCORE:
            continue
        score = VERDICT_SCORE[verdict]
        for space, meta in (e.get("spaces") or {}).items():
            if meta and "cosine" in meta:
                rows.append({"space": space, "cosine": meta["cosine"], "score": score})
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="docs/run_audits/sonic_audition")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    entries = load_captures(data_dir)
    if not entries:
        print(f"No capture entries found in {data_dir}. Complete the audition first.")
        return

    rated = [e for e in entries if e.get("verdict")]
    by_space = aggregate_by_space(entries)
    corr_rows = cosine_verdict_correlation(entries)

    lines = [
        "# Sonic Audition — Phase 2 Findings",
        "",
        f"Total entries: {len(entries)}  |  Rated: {len(rated)}",
        "",
        "## Verdict Distribution by Space",
        "",
        "| Space | match | close | off | wrong | total |",
        "|---|---|---|---|---|---|",
    ]
    for space in SPACES:
        counts = by_space.get(space, {})
        total = sum(counts.values())
        if total == 0:
            continue
        row = [space] + [str(counts.get(v, 0)) for v in VERDICT_ORDER] + [str(total)]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Cosine ↔ Verdict Correlation (Pearson r)", ""]
    space_groups: Dict[str, list] = defaultdict(list)
    for r in corr_rows:
        space_groups[r["space"]].append((r["cosine"], r["score"]))

    for space in SPACES:
        pairs = space_groups.get(space, [])
        if len(pairs) < 3:
            continue
        cosines = np.array([p[0] for p in pairs])
        scores = np.array([p[1] for p in pairs])
        r_val = float(np.corrcoef(cosines, scores)[0, 1])
        lines.append(
            f"- **{space}**: r={r_val:.3f} "
            f"({len(pairs)} rated, cosine [{cosines.min():.3f}, {cosines.max():.3f}])"
        )

    lines += ["", "## Negative-S Transition Pairs", ""]
    neg = [e for e in entries if e.get("seed") == "negative_s"]
    if neg:
        for e in neg:
            lines.append(
                f"- `{e['track_id']}`: verdict=**{e.get('verdict','—')}**"
                + (f" — {e['notes']}" if e.get("notes") else "")
            )
    else:
        lines.append("*(no negative-S entries yet)*")

    lines += ["", "## Notable Notes", ""]
    for e in sorted(rated, key=lambda x: VERDICT_SCORE.get(x.get("verdict", ""), 0)):
        if e.get("notes"):
            lines.append(
                f"- **{e.get('seed','')}** | {e.get('artist','')} — "
                f"{e.get('title','')} | {e.get('verdict','')} | {e['notes']}"
            )

    out = data_dir / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"\nVerdicts by space:")
    for space, counts in sorted(by_space.items()):
        print(f"  {space}: {counts}")


if __name__ == "__main__":
    main()
