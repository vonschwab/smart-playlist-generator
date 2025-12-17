#!/usr/bin/env python3
"""
Diagnose which sonic feature blocks dominate S_sonic_raw via ablations.

Examples:
  python scripts/diagnose_sonic_feature_ablation.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --mode random --n 50000 --seed 1
  python scripts/diagnose_sonic_feature_ablation.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --mode same_artist --artist "Aaliyah" --n 20000 --seed 1
  python scripts/diagnose_sonic_feature_ablation.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --mode cross_groups --group-a diagnostics/groups/minor_threat.json --group-b diagnostics/groups/green_house.json --n 50000 --seed 1 --blocks diagnostics/sonic_blocks.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.similarity.sonic_schema import dim_labels


def l2norm_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def summary_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = arr.reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "p01": percentile(arr, 1),
        "p10": percentile(arr, 10),
        "p50": percentile(arr, 50),
        "p90": percentile(arr, 90),
        "p99": percentile(arr, 99),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "var": float(np.var(arr)),
        "std": float(np.std(arr)),
        "spread": float(percentile(arr, 90) - percentile(arr, 10)),
    }


def saturated_flag(stats: Dict[str, float], thresh: float = 0.005) -> bool:
    return stats["spread"] < thresh


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def cosine_pairs(norm_a: np.ndarray, norm_b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    return np.sum(norm_a[ia] * norm_b[ib], axis=1)


def _parse_pairs_file(path: Path) -> List[Dict[str, str]]:
    ext = path.suffix.lower()
    rows: List[Dict[str, str]] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if not isinstance(data, list):
            raise ValueError("JSON pairs file must be a list or contain 'pairs'.")
        for row in data:
            if isinstance(row, dict):
                rows.append(
                    {
                        "prev_id": str(row.get("prev_id") or row.get("prev") or row.get("from") or "").strip(),
                        "cur_id": str(row.get("cur_id") or row.get("cur") or row.get("to") or "").strip(),
                        "label": str(row.get("label") or ""),
                    }
                )
    else:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "prev_id": str(row.get("prev_id") or row.get("prev") or row.get("from") or "").strip(),
                        "cur_id": str(row.get("cur_id") or row.get("cur") or row.get("to") or "").strip(),
                        "label": str(row.get("label") or ""),
                    }
                )
    return [r for r in rows if r["prev_id"] and r["cur_id"]]


def _resolve_pairs(bundle, rows: Iterable[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]], List[str]]:
    ia: List[int] = []
    ib: List[int] = []
    kept_meta: List[Dict[str, str]] = []
    missing: List[str] = []
    for row in rows:
        prev_id = row.get("prev_id", "")
        cur_id = row.get("cur_id", "")
        label = row.get("label", "")
        prev_idx = bundle.track_id_to_index.get(prev_id)
        cur_idx = bundle.track_id_to_index.get(cur_id)
        if prev_idx is None:
            missing.append(prev_id)
        if cur_idx is None:
            missing.append(cur_id)
        if prev_idx is None or cur_idx is None:
            continue
        ia.append(int(prev_idx))
        ib.append(int(cur_idx))
        kept_meta.append({"prev_id": prev_id, "cur_id": cur_id, "label": label})
    return np.array(ia, dtype=int), np.array(ib, dtype=int), kept_meta, missing


def _parse_group_file(path: Path) -> List[str]:
    ext = path.suffix.lower()
    ids: List[str] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "group" in data:
            data = data["group"]
        if isinstance(data, dict) and "pairs" in data:
            for row in data["pairs"]:
                if isinstance(row, dict):
                    if row.get("prev_id"):
                        ids.append(str(row["prev_id"]))
                    if row.get("cur_id"):
                        ids.append(str(row["cur_id"]))
            return list(dict.fromkeys(ids))
        if not isinstance(data, list):
            raise ValueError("Group JSON must be a list or contain 'group'.")
        for row in data:
            if isinstance(row, dict) and row.get("track_id"):
                ids.append(str(row["track_id"]))
            elif isinstance(row, str):
                ids.append(row)
    else:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("track_id") or row.get("id")
                if tid:
                    ids.append(str(tid))
    return list(dict.fromkeys(ids))


def _sample_random(n: int, N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ia = rng.integers(0, N, size=n, dtype=int)
    ib = rng.integers(0, N, size=n, dtype=int)
    same = ia == ib
    if same.any():
        ib[same] = (ib[same] + 1) % N
    return ia, ib


def _sample_same_artist(n: int, artist_keys: Sequence[str], artist_filter: Optional[str], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for idx, key in enumerate(artist_keys):
        k = str(key).strip().lower()
        groups.setdefault(k, []).append(idx)
    if artist_filter:
        target = artist_filter.strip().lower()
        groups = {k: v for k, v in groups.items() if k == target}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return np.array([], dtype=int), np.array([], dtype=int)
    artists = list(groups.keys())
    ia: List[int] = []
    ib: List[int] = []
    for _ in range(n):
        artist = artists[rng.integers(0, len(artists))]
        idxs = groups[artist]
        a, b = rng.choice(idxs, size=2, replace=False)
        ia.append(int(a))
        ib.append(int(b))
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def _sample_cross_groups(idx_a: np.ndarray, idx_b: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if idx_a.size == 0 or idx_b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    max_pairs = idx_a.size * idx_b.size
    if n > 0 and n < max_pairs:
        ia_local = rng.choice(idx_a, size=n)
        ib_local = rng.choice(idx_b, size=n)
        return ia_local.astype(int), ib_local.astype(int)
    ia_list = []
    ib_list = []
    for a in idx_a:
        for b in idx_b:
            ia_list.append(int(a))
            ib_list.append(int(b))
    return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)


def _load_blocks(blocks_path: Optional[Path], dim: int) -> Dict[str, Tuple[int, int]]:
    if not blocks_path:
        return {"full": (0, dim - 1)}
    data = json.loads(blocks_path.read_text())
    if isinstance(data, dict) and "blocks" in data:
        data = data["blocks"]
    blocks: Dict[str, Tuple[int, int]] = {}
    for name, val in data.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            start, end = int(val[0]), int(val[1])
            start = max(0, start)
            end = min(dim - 1, end)
            if start <= end:
                blocks[str(name)] = (start, end)
    if "full" not in blocks:
        blocks["full"] = (0, dim - 1)
    return blocks


def _ablate_block(norm_mat: np.ndarray, block: Tuple[int, int]) -> np.ndarray:
    start, end = block
    ablated = norm_mat.copy()
    ablated[:, start : end + 1] = 0.0
    denom = np.linalg.norm(ablated, axis=1, keepdims=True) + 1e-12
    return ablated / denom


def _norm_stats(x: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(x, axis=1)
    return {
        "mean": float(norms.mean()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "std": float(norms.std()),
    }


def _variance_topk(x: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    var = np.var(x, axis=0)
    idx = np.argsort(var)[::-1][:k]
    return [(int(i), float(var[i])) for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sonic feature block influence via ablation.")
    parser.add_argument("--artifact", required=True, help="Path to artifact npz")
    parser.add_argument("--n", type=int, default=50000, help="Number of pairs to sample (default 50000)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    parser.add_argument("--mode", choices=["random", "same_artist", "cross_groups"], default="random", help="Sampling mode")
    parser.add_argument("--artist", help="Artist key/name for same_artist mode")
    parser.add_argument("--pairs", type=Path, help="Explicit pairs file (overrides mode sampling)")
    parser.add_argument("--group-a", type=Path, help="Group A file for cross_groups")
    parser.add_argument("--group-b", type=Path, help="Group B file for cross_groups")
    parser.add_argument("--blocks", type=Path, help="JSON mapping of block name -> [start, end] (inclusive)")
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact)
    labels = dim_labels(bundle)
    rng = np.random.default_rng(args.seed)

    X_sonic = getattr(bundle, "X_sonic", None)
    if X_sonic is None:
        raise SystemExit("X_sonic missing in artifact; cannot run ablations.")

    N = int(X_sonic.shape[0])
    print(f"Loaded artifact: {Path(args.artifact).resolve()}")
    print(f"Tracks: {N} | Dim: {X_sonic.shape[1]}")

    ia: np.ndarray
    ib: np.ndarray
    notes: List[str] = []
    pairs_meta: List[Dict[str, str]] = []

    if args.pairs:
        rows = _parse_pairs_file(args.pairs)
        ia, ib, pairs_meta, missing = _resolve_pairs(bundle, rows)
        if missing:
            notes.append(f"Missing {len(set(missing))} ids from pairs file.")
    elif args.mode == "random":
        ia, ib = _sample_random(args.n, N, rng)
    elif args.mode == "same_artist":
        artist_keys = getattr(bundle, "artist_keys", None)
        if artist_keys is None:
            ia = ib = np.array([], dtype=int)
            notes.append("artist_keys missing; same_artist unavailable.")
        else:
            ia, ib = _sample_same_artist(args.n, artist_keys, args.artist, rng)
            if ia.size == 0:
                notes.append("No pairs for requested artist.")
    elif args.mode == "cross_groups":
        if not args.group_a or not args.group_b:
            raise SystemExit("cross_groups requires --group-a and --group-b")
        ids_a = _parse_group_file(args.group_a)
        ids_b = _parse_group_file(args.group_b)
        def resolve(ids: List[str]) -> np.ndarray:
            idxs = []
            missing_local = []
            for tid in ids:
                idx = bundle.track_id_to_index.get(str(tid))
                if idx is None:
                    missing_local.append(tid)
                else:
                    idxs.append(int(idx))
            if missing_local:
                notes.append(f"Missing {len(set(missing_local))} ids in group.")
            return np.array(sorted(set(idxs)), dtype=int)
        idx_a = resolve(ids_a)
        idx_b = resolve(ids_b)
        if idx_a.size == 0 or idx_b.size == 0:
            raise SystemExit("cross_groups resolved empty group.")
        ia, ib = _sample_cross_groups(idx_a, idx_b, args.n, rng)
    else:
        ia = ib = np.array([], dtype=int)

    if ia.size == 0:
        print("No pairs to evaluate.")
        return

    sonic_norm = l2norm_rows(X_sonic)
    baseline = cosine_pairs(sonic_norm, sonic_norm, ia, ib)
    blocks = _load_blocks(args.blocks, X_sonic.shape[1])
    baseline_stats = summary_stats(baseline)
    baseline_stats["saturated"] = saturated_flag(baseline_stats)

    top_dims = _variance_topk(X_sonic, k=min(10, X_sonic.shape[1]))
    results: Dict[str, Any] = {
        "baseline": {
            "stats": baseline_stats,
            "pair_count": int(ia.size),
        },
        "blocks": {},
        "norms": _norm_stats(X_sonic),
        "top_variance_dims": [
            {"index": idx, "label": labels[idx], "variance": var} for idx, var in top_dims
        ],
        "notes": notes,
    }

    for name, (start, end) in blocks.items():
        if name == "full":
            continue
        ablated_norm = _ablate_block(sonic_norm, (start, end))
        vals = cosine_pairs(ablated_norm, ablated_norm, ia, ib)
        stats = summary_stats(vals)
        stats["saturated"] = saturated_flag(stats)
        delta_p50 = stats["p50"] - baseline_stats["p50"]
        delta_spread = stats["spread"] - baseline_stats["spread"]
        corr = pearson_corr(baseline, vals)
        results["blocks"][name] = {
            "range": [start, end],
            "stats": stats,
            "corr_with_baseline": float(corr) if np.isfinite(corr) else None,
            "delta_p50_vs_baseline": float(delta_p50),
            "delta_spread_vs_baseline": float(delta_spread),
        }

    # prepare report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_json = diagnostics_dir / f"sonic_feature_ablation_{ts}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_path = diagnostics_dir / "sonic_feature_ablation_runs.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} | mode={args.mode} | pairs={len(ia)} | blocks={list(blocks.keys())} ===\n")
        f.write(
            f"baseline p50={baseline_stats['p50']:.4f} spread={baseline_stats['spread']:.4f} saturated={baseline_stats['saturated']}\n"
        )
        for name, info in results["blocks"].items():
            stats = info["stats"]
            f.write(
                f"{name:15s} p50={stats['p50']:.4f} spread={stats['spread']:.4f} "
                f"corr={info['corr_with_baseline'] if info['corr_with_baseline'] is not None else 'nan'} "
                f"dp50={info['delta_p50_vs_baseline']:.4f} dspread={info['delta_spread_vs_baseline']:.4f}\n"
            )
        if notes:
            f.write(f"Notes: {notes}\n")

    print(f"Wrote JSON report to {out_json}")
    print(f"Appended summary to {log_path}")


if __name__ == "__main__":
    main()
