#!/usr/bin/env python3
"""
Lightweight helper to build a pairs file for diagnose_sonic_signal.py.

Examples:
  python scripts/make_pairs_file.py --out pairs.json --pair prev_id,cur_id,same_album --pair foo,bar
  python scripts/make_pairs_file.py --out pairs.csv --pair a,b,label1 --validate-artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle


def parse_pair(raw: str) -> Tuple[str, str, str]:
    """
    Accepts prev_id,cur_id[,label] with either comma or colon separators.
    """
    raw = raw.strip()
    sep = "," if "," in raw else ":"
    parts = [p.strip() for p in raw.split(sep) if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid pair string (need prev_id{sep}cur_id): {raw}")
    prev_id, cur_id = parts[0], parts[1]
    label = parts[2] if len(parts) >= 3 else ""
    return prev_id, cur_id, label


def write_json(out_path: Path, pairs: List[Tuple[str, str, str]]) -> None:
    payload = {"pairs": [{"prev_id": a, "cur_id": b, "label": lbl} for a, b, lbl in pairs]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(out_path: Path, pairs: List[Tuple[str, str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prev_id", "cur_id", "label"])
        writer.writeheader()
        for a, b, lbl in pairs:
            writer.writerow({"prev_id": a, "cur_id": b, "label": lbl})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JSON/CSV pairs file for diagnostics.")
    parser.add_argument("--out", required=True, help="Output path (.json or .csv)")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Pair string: prev_id,cur_id[,label] (comma or colon separators). Can be repeated.",
    )
    parser.add_argument(
        "--validate-artifact",
        type=str,
        help="Optional artifact path to validate that track_ids exist.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        help="Force output format (otherwise inferred from --out extension).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    fmt = args.format or out_path.suffix.lower().lstrip(".")
    if fmt not in {"json", "csv"}:
        fmt = "json"

    if not args.pair:
        raise SystemExit("Provide at least one --pair prev_id,cur_id[,label].")

    pairs: List[Tuple[str, str, str]] = []
    for raw in args.pair:
        prev_id, cur_id, label = parse_pair(raw)
        pairs.append((prev_id, cur_id, label))

    missing = []
    if args.validate_artifact:
        try:
            bundle = load_artifact_bundle(args.validate_artifact)
            for prev_id, cur_id, _ in pairs:
                if prev_id not in bundle.track_id_to_index:
                    missing.append(prev_id)
                if cur_id not in bundle.track_id_to_index:
                    missing.append(cur_id)
        except Exception as exc:
            print(f"Warning: could not validate artifact ({exc})")

    if fmt == "json":
        write_json(out_path, pairs)
    else:
        write_csv(out_path, pairs)

    print(f"Wrote {len(pairs)} pairs to {out_path} ({fmt.upper()})")
    if missing:
        uniq = sorted(set(missing))
        print(f"Note: {len(uniq)} ids not found in artifact during validation: {uniq}")


if __name__ == "__main__":
    main()
