"""Determinism gate: same cell twice must be byte-identical. MUST pass before
the knob sweep has any meaning. Usage:
    python scripts/corridor_baseline/determinism_check.py --artist "Bill Evans Trio" --detent open
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.corridor_baseline.runner import DETENTS, run_cell  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="Bill Evans Trio")
    ap.add_argument("--detent", default="open", choices=sorted(DETENTS))
    a = ap.parse_args()
    r1 = run_cell(a.artist, a.detent, log_tag="det1")
    r2 = run_cell(a.artist, a.detent, log_tag="det2")
    for r in (r1, r2):
        if r["err"]:
            print(f"GENERATION ERROR: {r['err']}\nlog: {r['log_path']}")
            return 2
    same_tracks = r1["track_ids"] == r2["track_ids"]
    same_eff = json.dumps(r1["effective"], sort_keys=True) == json.dumps(r2["effective"], sort_keys=True)
    print(f"tracks identical: {same_tracks}  effective identical: {same_eff}")
    if not (same_tracks and same_eff):
        print("NONDETERMINISTIC — STOP. Diff the two logs:")
        print(r1["log_path"], r2["log_path"], sep="\n")
        return 1
    print("DETERMINISM GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
