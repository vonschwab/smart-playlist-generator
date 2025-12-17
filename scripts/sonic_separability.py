#!/usr/bin/env python3
"""
Compute separability of sonic similarity for raw vs z variants.

Examples:
  python scripts/sonic_separability.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --group-a diagnostics/groups/minor_threat.json --group-b diagnostics/groups/green_house.json --n 50000 --seed 1
  python scripts/sonic_separability.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --artist "Aaliyah" --n 20000 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional sklearn for AUC
try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore

# ensure root
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_variant import compute_sonic_variant_norm


def _parse_group_file(path: Path) -> List[str]:
    ext = path.suffix.lower()
    ids: List[str] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "group" in data:
                data = data["group"]
            elif "pairs" in data:
                tmp: List[str] = []
                for row in data["pairs"]:
                    if isinstance(row, dict):
                        if row.get("prev_id"):
                            tmp.append(str(row["prev_id"]))
                        if row.get("cur_id"):
                            tmp.append(str(row["cur_id"]))
                data = tmp
        if not isinstance(data, list):
            raise ValueError("group file must be a list or contain 'group'/'pairs'")
        for row in data:
            if isinstance(row, dict) and row.get("track_id"):
                ids.append(str(row["track_id"]))
            elif isinstance(row, str):
                ids.append(str(row))
    else:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("track_id") or row.get("id")
                if tid:
                    ids.append(str(tid))
    return list(dict.fromkeys(ids))


def _sample_pairs(indices: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if indices.size < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    max_pairs = indices.size * (indices.size - 1)
    if n > 0 and n < max_pairs:
        ia = rng.choice(indices, size=n)
        ib = rng.choice(indices, size=n)
        same = ia == ib
        if same.any():
            ib[same] = indices[(np.searchsorted(indices, ib[same]) + 1) % indices.size]
        return ia.astype(int), ib.astype(int)
    ia_list = []
    ib_list = []
    for i in indices:
        for j in indices:
            if i == j:
                continue
            ia_list.append(int(i))
            ib_list.append(int(j))
    return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)


def _cosine_for_variant(X: np.ndarray, ia: np.ndarray, ib: np.ndarray, variant: str) -> np.ndarray:
    normed, _ = compute_sonic_variant_norm(X, variant)
    return np.sum(normed[ia] * normed[ib], axis=1)


def _auc(within: np.ndarray, across: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(within), np.zeros_like(across)])
    scores = np.concatenate([within, across])
    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if scores.size < 2:
        return float("nan")
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            pass
    # manual AUC via rank
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = ranks[y_true == 1].sum()
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    da = float(np.mean(a))
    db = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return (da - db) / pooled


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "spread": float(np.percentile(arr, 90) - np.percentile(arr, 10)),
        "mean": float(np.mean(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sonic separability for raw vs z variants.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--group-a", type=Path)
    parser.add_argument("--group-b", type=Path)
    parser.add_argument("--artist", type=str, help="Same-artist sampling instead of groups")
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact)
    rng = np.random.default_rng(args.seed)

    variants = ["raw", "z"]
    results: Dict[str, Any] = {"artifact": str(args.artifact), "seed": args.seed, "n": args.n, "variants": {}}

    def sample_groups() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if args.group_a and args.group_b:
            ids_a = _parse_group_file(args.group_a)
            ids_b = _parse_group_file(args.group_b)
            idx_a = [bundle.track_id_to_index.get(str(t)) for t in ids_a]
            idx_b = [bundle.track_id_to_index.get(str(t)) for t in ids_b]
            idx_a = np.array([i for i in idx_a if i is not None], dtype=int)
            idx_b = np.array([i for i in idx_b if i is not None], dtype=int)
            ia_a, ib_a = _sample_pairs(idx_a, args.n, rng)
            ia_b, ib_b = _sample_pairs(idx_b, args.n, rng)
            if idx_a.size == 0 or idx_b.size == 0:
                ia_ac = ib_ac = np.array([], dtype=int)
            else:
                ia_ac = rng.choice(idx_a, size=min(args.n, max(1, idx_a.size * idx_b.size)))
                ib_ac = rng.choice(idx_b, size=ia_ac.size)
            return ia_a, ib_a, ia_b, ib_b, ia_ac, ib_ac
        artist_keys = getattr(bundle, "artist_keys", None)
        if artist_keys is None or not args.artist:
            raise SystemExit("Provide group-a/group-b or --artist")
        keys = np.array([str(a).lower() for a in artist_keys])
        mask = keys == args.artist.lower()
        idx = np.nonzero(mask)[0]
        ia_w, ib_w = _sample_pairs(idx, args.n, rng)
        others = np.nonzero(~mask)[0]
        ia_ac = rng.choice(idx, size=min(args.n, idx.size)) if idx.size else np.array([], dtype=int)
        ib_ac = rng.choice(others, size=ia_ac.size) if ia_ac.size else np.array([], dtype=int)
        return ia_w, ib_w, np.array([], dtype=int), np.array([], dtype=int), ia_ac, ib_ac

    ia_a, ib_a, ia_b, ib_b, ia_ac, ib_ac = sample_groups()

    X = bundle.X_sonic
    for variant in variants:
        entry: Dict[str, Any] = {}
        within_a = _cosine_for_variant(X, ia_a, ib_a, variant) if ia_a.size else np.array([])
        within_b = _cosine_for_variant(X, ia_b, ib_b, variant) if ia_b.size else np.array([])
        across = _cosine_for_variant(X, ia_ac, ib_ac, variant) if ia_ac.size else np.array([])
        if within_a.size and across.size:
            entry["auc_a"] = _auc(within_a, across)
            entry["d_a"] = _cohen_d(within_a, across)
        if within_b.size and across.size:
            entry["auc_b"] = _auc(within_b, across)
            entry["d_b"] = _cohen_d(within_b, across)
        if within_a.size:
            entry["within_a_stats"] = _stats(within_a)
        if within_b.size:
            entry["within_b_stats"] = _stats(within_b)
        if across.size:
            entry["across_stats"] = _stats(across)
        results["variants"][variant] = entry

    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    ts = np.datetime64("now").astype(str).replace("-", "").replace(":", "").split(".")[0]
    out_json = diagnostics_dir / f"sonic_separability_{ts}.json"
    out_log = diagnostics_dir / "sonic_separability_runs.log"
    out_md = diagnostics_dir / "SONIC_SEPARABILITY_REPORT.md"

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with out_log.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {results}\n")

    lines = ["# Sonic Separability Report", f"- Artifact: {args.artifact}", f"- Seed: {args.seed}", ""]
    for var, entry in results["variants"].items():
        lines.append(f"## Variant: {var}")
        for key in ["auc_a", "auc_b", "d_a", "d_b"]:
            if key in entry:
                lines.append(f"- {key}: {entry[key]:.4f}")
        for name in ["within_a_stats", "within_b_stats", "across_stats"]:
            if name in entry:
                s = entry[name]
                lines.append(
                    f"- {name}: p10={s.get('p10'):.4f} p50={s.get('p50'):.4f} p90={s.get('p90'):.4f} spread={s.get('spread'):.4f}"
                )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
