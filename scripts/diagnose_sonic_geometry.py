#!/usr/bin/env python3
"""
Probe the geometry of S_sonic and related signals without changing playlist behavior.

Examples:
  python scripts/diagnose_sonic_geometry.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --mode random --n 100000 --seed 1
  python scripts/diagnose_sonic_geometry.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --mode same_artist --artist "Aaliyah" --n 50000
  python scripts/diagnose_sonic_geometry.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --pairs diagnostics/my_pairs.json
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

# Optional import guarded for environments without sklearn; PCA metric is skipped if unavailable.
try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - optional dependency
    PCA = None  # type: ignore

# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle


def l2norm_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def center_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    return x - mu


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (x - mu) / std_safe


def cosine_pairs(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    a_norm = l2norm_rows(a)
    b_norm = l2norm_rows(b)
    return np.sum(a_norm[ia] * b_norm[ib], axis=1)


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


def _parse_group_file(path: Path) -> List[Dict[str, str]]:
    ext = path.suffix.lower()
    rows: List[Dict[str, str]] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "pairs" not in data and "group" in data:
            data = data["group"]
        if isinstance(data, dict) and "pairs" in data:
            # allow reuse of pairs schema: take both prev_id/cur_id entries
            tmp: List[Dict[str, str]] = []
            for row in data["pairs"]:
                if isinstance(row, dict):
                    if "prev_id" in row:
                        tmp.append({"track_id": row["prev_id"]})
                    if "cur_id" in row:
                        tmp.append({"track_id": row["cur_id"]})
            data = tmp
        if not isinstance(data, list):
            raise ValueError("Group file JSON must be a list or object with 'group'.")
        for row in data:
            if isinstance(row, dict) and row.get("track_id"):
                rows.append({"track_id": str(row["track_id"]), "label": str(row.get("label", ""))})
            elif isinstance(row, str):
                rows.append({"track_id": row, "label": ""})
    else:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("track_id") or row.get("id") or row.get("track")
                if tid:
                    rows.append({"track_id": str(tid), "label": str(row.get("label", ""))})
    # de-dupe while preserving order
    seen = set()
    unique_rows: List[Dict[str, str]] = []
    for row in rows:
        tid = row["track_id"]
        if tid in seen:
            continue
        seen.add(tid)
        unique_rows.append(row)
    return unique_rows


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


def _sample_cross_genre(
    bundle,
    tag_a: str,
    tag_b: str,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return pairs from tag_a (prev) to tag_b (cur); returns missing tags list if unavailable."""
    missing: List[str] = []
    vocab = [str(g).lower() for g in bundle.genre_vocab.tolist()]
    try:
        idx_a = vocab.index(tag_a.lower())
    except ValueError:
        missing.append(tag_a)
        return np.array([], dtype=int), np.array([], dtype=int), missing
    try:
        idx_b = vocab.index(tag_b.lower())
    except ValueError:
        missing.append(tag_b)
        return np.array([], dtype=int), np.array([], dtype=int), missing

    Xa = bundle.X_genre_raw[:, idx_a] if bundle.X_genre_raw is not None else None
    Xb = bundle.X_genre_raw[:, idx_b] if bundle.X_genre_raw is not None else None
    if Xa is None or Xb is None:
        missing.append("genre_raw")
        return np.array([], dtype=int), np.array([], dtype=int), missing

    prev_candidates = np.nonzero(Xa > 0)[0]
    cur_candidates = np.nonzero(Xb > 0)[0]
    if prev_candidates.size == 0 or cur_candidates.size == 0:
        missing.append("candidates")
        return np.array([], dtype=int), np.array([], dtype=int), missing

    ia = rng.choice(prev_candidates, size=n, replace=True)
    ib = rng.choice(cur_candidates, size=n, replace=True)
    return ia.astype(int), ib.astype(int), missing


def _compute_metrics(
    bundle,
    ia: np.ndarray,
    ib: np.ndarray,
    *,
    include_pca: bool,
) -> Dict[str, np.ndarray]:
    metrics: Dict[str, np.ndarray] = {}
    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    X_start = getattr(bundle, "X_sonic_start", None)
    X_end = getattr(bundle, "X_sonic_end", None)

    if X_sonic is not None:
        metrics["S_sonic_raw"] = cosine_pairs(X_sonic, X_sonic, ia, ib)
        metrics["S_sonic_centered"] = cosine_pairs(center_rows(X_sonic), center_rows(X_sonic), ia, ib)
        metrics["S_sonic_z"] = cosine_pairs(zscore_rows(X_sonic), zscore_rows(X_sonic), ia, ib)
        if include_pca and PCA is not None:
            z = zscore_rows(X_sonic)
            n_comp = min(64, z.shape[1], z.shape[0])
            try:
                pca = PCA(n_components=n_comp, random_state=0)
                Zp = pca.fit_transform(z)
                metrics["S_sonic_pca"] = cosine_pairs(Zp, Zp, ia, ib)
            except Exception:
                pass

    if X_start is not None and X_end is not None:
        metrics["T_raw"] = cosine_pairs(X_end, X_start, ia, ib)
        metrics["T_centered"] = cosine_pairs(center_rows(X_end), center_rows(X_start), ia, ib)
        metrics["T_centered_rescaled"] = np.clip((metrics["T_centered"] + 1.0) / 2.0, 0.0, 1.0)

    if X_genre is not None:
        metrics["G_genre"] = cosine_pairs(X_genre, X_genre, ia, ib)

    return metrics


def _percentile_position(baseline: np.ndarray, value: float) -> float:
    """Return percentile of value within baseline (0-100)."""
    baseline = baseline.reshape(-1)
    if baseline.size == 0:
        return float("nan")
    rank = (baseline <= value).sum()
    return float(100.0 * rank / baseline.size)


def _correlations(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    keys = list(metrics.keys())
    corr: Dict[str, float] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            corr[f"corr({a},{b})"] = pearson_corr(metrics[a], metrics[b])
    # Explicitly highlight the requested ones when present
    if "S_sonic_raw" in metrics and "T_raw" in metrics:
        corr["corr(S_sonic_raw,T_raw)"] = pearson_corr(metrics["S_sonic_raw"], metrics["T_raw"])
    if "S_sonic_raw" in metrics and "G_genre" in metrics:
        corr["corr(S_sonic_raw,G_genre)"] = pearson_corr(metrics["S_sonic_raw"], metrics["G_genre"])
    if "S_sonic_raw" in metrics and "T_centered_rescaled" in metrics:
        corr["corr(S_sonic_raw,T_centered_rescaled)"] = pearson_corr(metrics["S_sonic_raw"], metrics["T_centered_rescaled"])
    if "T_centered_rescaled" in metrics and "G_genre" in metrics:
        corr["corr(T_centered_rescaled,G_genre)"] = pearson_corr(metrics["T_centered_rescaled"], metrics["G_genre"])
    return corr


def _summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, arr in metrics.items():
        stats = summary_stats(arr)
        stats["saturated"] = saturated_flag(stats)
        out[name] = stats
    return out


def _print_summary(mode: str, ia: np.ndarray, metrics: Dict[str, np.ndarray], corr: Dict[str, float]) -> None:
    print(f"\n=== Mode: {mode} | pairs={len(ia)} ===")
    for name, stats in _summarize_metrics(metrics).items():
        print(
            f"{name:20s} mean={stats['mean']:.4f} p01={stats['p01']:.4f} p10={stats['p10']:.4f} "
            f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} p99={stats['p99']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} var={stats['var']:.6f} std={stats['std']:.4f} "
            f"spread={stats['spread']:.4f} saturated={stats['saturated']}"
        )
    for k, v in corr.items():
        print(f"{k:35s}: {v if np.isfinite(v) else 'nan'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose sonic geometry and saturation.")
    parser.add_argument("--artifact", required=True, help="Path to artifact NPZ")
    parser.add_argument(
        "--mode",
        choices=["random", "same_artist", "cross_genre_extremes", "cross_groups"],
        default="random",
        help="Sampling mode",
    )
    parser.add_argument("--artist", help="Artist key/name for same_artist mode")
    parser.add_argument("--n", type=int, default=100000, help="Number of pairs to sample (default 100000)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    parser.add_argument("--pairs", type=Path, help="JSON/CSV pairs file (overrides mode sampling when provided)")
    parser.add_argument("--tag-a", dest="tag_a", help="Genre tag A for cross_genre_extremes (prev)")
    parser.add_argument("--tag-b", dest="tag_b", help="Genre tag B for cross_genre_extremes (cur)")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA-based similarity even if sklearn is available")
    parser.add_argument("--group-a", type=Path, help="Group A track_id list (json/csv) for cross_groups mode")
    parser.add_argument("--group-b", type=Path, help="Group B track_id list (json/csv) for cross_groups mode")
    parser.add_argument("--baseline-n", type=int, default=50000, help="Random baseline pairs for percentile placement (cross_groups)")
    parser.add_argument("--dump-sonic-blocks", action="store_true", help="Best-effort print sonic block guidance (if any).")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    bundle = load_artifact_bundle(artifact_path)
    rng = np.random.default_rng(args.seed)
    include_pca = not args.no_pca

    if args.dump_sonic_blocks:
        names = getattr(bundle, "sonic_feature_names", None)
        diagnostics_dir = Path("diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if names is None:
            print("Sonic block inference not available in artifact; provide a --blocks JSON (per-dim fallback).")
        else:
            names_list = [str(n) for n in names]
            blocks = {}
            for idx, name in enumerate(names_list):
                prefix = name.split("_")[0] if "_" in name else name
                blocks.setdefault(prefix, []).append(idx)
            suggested = {}
            for prefix, idxs in blocks.items():
                suggested[prefix] = [min(idxs), max(idxs)]
            out_blocks = diagnostics_dir / "sonic_blocks_suggested.json"
            out_blocks.write_text(json.dumps({"blocks": suggested}, indent=2), encoding="utf-8")
            print(f"Inferred blocks by prefix; wrote suggestion to {out_blocks}")
        return

    N = int(bundle.track_ids.shape[0])
    print(f"Loaded artifact: {artifact_path.resolve()}")
    print(f"Tracks: {N}")

    ia: np.ndarray
    ib: np.ndarray
    mode_label = args.mode
    notes: List[str] = []
    pairs_meta: List[Dict[str, str]] = []

    if args.pairs:
        rows = _parse_pairs_file(args.pairs)
        ia, ib, pairs_meta, missing = _resolve_pairs(bundle, rows)
        mode_label = f"manual_pairs ({args.pairs.name})"
        if missing:
            notes.append(f"Missing {len(set(missing))} ids from pairs file.")
    elif args.mode == "random":
        ia, ib = _sample_random(args.n, N, rng)
    elif args.mode == "same_artist":
        artist_keys = getattr(bundle, "artist_keys", None)
        fallback_artists = getattr(bundle, "track_artists", None)
        if artist_keys is None and fallback_artists is not None:
            artist_keys = fallback_artists
        if artist_keys is None:
            ia = ib = np.array([], dtype=int)
            notes.append("artist_keys missing; same_artist mode unavailable.")
        else:
            ia, ib = _sample_same_artist(args.n, artist_keys, args.artist, rng)
            if ia.size == 0:
                notes.append("No same-artist pairs available for requested artist.")
            if args.artist:
                mode_label = f"same_artist ({args.artist})"
    elif args.mode == "cross_genre_extremes":
        if not args.tag_a or not args.tag_b:
            ia = ib = np.array([], dtype=int)
            notes.append("Provide --tag-a and --tag-b for cross_genre_extremes.")
        else:
            ia, ib, missing_tags = _sample_cross_genre(bundle, args.tag_a, args.tag_b, args.n, rng)
            if missing_tags:
                ia = ib = np.array([], dtype=int)
                notes.append(f"cross_genre_extremes skipped: missing {missing_tags}.")
            else:
                mode_label = f"cross_genre_extremes ({args.tag_a}->{args.tag_b})"
    elif args.mode == "cross_groups":
        if not args.group_a or not args.group_b:
            print("cross_groups mode requires --group-a and --group-b")
            return
        group_a_rows = _parse_group_file(args.group_a)
        group_b_rows = _parse_group_file(args.group_b)
        if not group_a_rows or not group_b_rows:
            print("cross_groups: one of the groups is empty.")
            return
        def resolve_group(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
            idxs = []
            missing_ids: List[str] = []
            for row in rows:
                tid = row.get("track_id")
                if tid is None:
                    continue
                idx = bundle.track_id_to_index.get(str(tid))
                if idx is None:
                    missing_ids.append(str(tid))
                else:
                    idxs.append(int(idx))
            return np.array(sorted(set(idxs)), dtype=int), missing_ids
        idx_a, miss_a = resolve_group(group_a_rows)
        idx_b, miss_b = resolve_group(group_b_rows)
        if miss_a or miss_b:
            missing_all = miss_a + miss_b
            print(f"cross_groups: missing {len(missing_all)} ids in artifact. Example: {missing_all[:5]}")
        if idx_a.size == 0 or idx_b.size == 0:
            print("cross_groups: resolved empty group after missing ids; aborting.")
            return
        def sample_pairs_from_indices(indices: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
            if indices.size < 2:
                return np.array([], dtype=int), np.array([], dtype=int)
            max_pairs = indices.size * (indices.size - 1)
            if n > 0 and n < max_pairs:
                ia_local = rng.choice(indices, size=n)
                ib_local = rng.choice(indices, size=n)
                same = ia_local == ib_local
                if same.any():
                    ib_local[same] = indices[(np.searchsorted(indices, ib_local[same]) + 1) % indices.size]
                return ia_local.astype(int), ib_local.astype(int)
            else:
                # full cartesian without self
                ia_list = []
                ib_list = []
                for i in indices:
                    for j in indices:
                        if i == j:
                            continue
                        ia_list.append(i)
                        ib_list.append(j)
                return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)

        ia_within_a, ib_within_a = sample_pairs_from_indices(idx_a, args.n)
        ia_within_b, ib_within_b = sample_pairs_from_indices(idx_b, args.n)
        # across
        def sample_across(a: np.ndarray, b: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
            if a.size == 0 or b.size == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            max_pairs = a.size * b.size
            if n > 0 and n < max_pairs:
                ia_local = rng.choice(a, size=n)
                ib_local = rng.choice(b, size=n)
                return ia_local.astype(int), ib_local.astype(int)
            else:
                ia_list = []
                ib_list = []
                for i in a:
                    for j in b:
                        ia_list.append(i)
                        ib_list.append(j)
                return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)
        ia_across, ib_across = sample_across(idx_a, idx_b, args.n)

        # Baseline random for percentile placement
        ia_base, ib_base = _sample_random(args.baseline_n, N, rng)
        baseline_metrics = _compute_metrics(bundle, ia_base, ib_base, include_pca=include_pca)

        def compute_with_name(label: str, ia_arr: np.ndarray, ib_arr: np.ndarray) -> Tuple[str, Dict[str, np.ndarray], Dict[str, float]]:
            if ia_arr.size == 0:
                print(f"cross_groups: {label} has no pairs.")
                return label, {}, {}
            m = _compute_metrics(bundle, ia_arr, ib_arr, include_pca=include_pca)
            c = _correlations(m)
            return label, m, c

        results = []
        results.append(compute_with_name("within_a", ia_within_a, ib_within_a))
        results.append(compute_with_name("within_b", ia_within_b, ib_within_b))
        results.append(compute_with_name("across", ia_across, ib_across))

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        diagnostics_dir = Path("diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        out_json = diagnostics_dir / f"sonic_geometry_{ts}.json"

        report_modes: Dict[str, Any] = {}
        for label, mets, corrs in results:
            if not mets:
                continue
            _print_summary(f"{mode_label}:{label}", ia_within_a if label=='within_a' else ia_within_b if label=='within_b' else ia_across, mets, corrs)
            stats = _summarize_metrics(mets)
            # baseline percentiles for p50
            baseline_pct: Dict[str, float] = {}
            for metric_name, metric_stats in stats.items():
                if metric_name in baseline_metrics:
                    base = baseline_metrics[metric_name]
                    baseline_pct[metric_name] = _percentile_position(base, metric_stats["p50"])
            report_modes[label] = {
                "metrics": stats,
                "correlations": {k: (float(v) if np.isfinite(v) else None) for k, v in corrs.items()},
                "baseline_percentile_p50": baseline_pct,
                "pair_count": int(next(iter(mets.values())).shape[0]) if mets else 0,
            }

        baseline_stats = {k: _summarize_metrics({k: v})[k] for k, v in baseline_metrics.items()}

        report = {
            "artifact": str(artifact_path),
            "timestamp_utc": ts,
            "mode": mode_label,
            "pair_count": {
                "within_a": int(ia_within_a.size),
                "within_b": int(ia_within_b.size),
                "across": int(ia_across.size),
                "baseline": int(ia_base.size),
            },
            "modes": report_modes,
            "baseline_metrics": baseline_stats,
            "notes": notes,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log_path = diagnostics_dir / "sonic_geometry_runs.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== {ts} | mode={mode_label} | within_a={ia_within_a.size} within_b={ia_within_b.size} across={ia_across.size} ===\n")
            for lbl, mets, _ in results:
                if not mets:
                    continue
                stats = _summarize_metrics(mets)
                f.write(f"[{lbl}]\n")
                for name, st in stats.items():
                    f.write(f"{name:20s} p10={st['p10']:.4f} p50={st['p50']:.4f} p90={st['p90']:.4f} spread={st['spread']:.4f} saturated={st['saturated']}\n")
        print(f"\nWrote JSON report to {out_json}")
        print(f"Appended summary to {log_path}")
        return
    else:
        ia = ib = np.array([], dtype=int)

    if ia.size == 0:
        print(f"No pairs available for mode {mode_label}. Notes: {notes}")
        return

    metrics = _compute_metrics(bundle, ia, ib, include_pca=include_pca)
    if not metrics:
        print("No metrics computed (missing matrices).")
        return
    corr = _correlations(metrics)

    _print_summary(mode_label, ia, metrics, corr)

    # Prepare JSON report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_json = diagnostics_dir / f"sonic_geometry_{ts}.json"
    report = {
        "artifact": str(artifact_path),
        "timestamp_utc": ts,
        "mode": mode_label,
        "pair_count": int(len(ia)),
        "metrics": _summarize_metrics(metrics),
        "correlations": {k: (float(v) if np.isfinite(v) else None) for k, v in corr.items()},
        "notes": notes,
    }
    if pairs_meta:
        report["pairs"] = pairs_meta
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_path = diagnostics_dir / "sonic_geometry_runs.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} | mode={mode_label} | pairs={len(ia)} ===\n")
        for name, stats in _summarize_metrics(metrics).items():
            f.write(
                f"{name:20s} mean={stats['mean']:.4f} p10={stats['p10']:.4f} p90={stats['p90']:.4f} "
                f"spread={stats['spread']:.4f} saturated={stats['saturated']}\n"
            )
        for k, v in corr.items():
            f.write(f"{k}: {v}\n")
        if notes:
            f.write(f"Notes: {notes}\n")

    print(f"\nWrote JSON report to {out_json}")
    print(f"Appended summary to {log_path}")


if __name__ == "__main__":
    main()
