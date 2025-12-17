#!/usr/bin/env python3
"""
Diagnose the DS sonic signal (S_sonic) versus transition and genre signals.

Examples:
  python scripts/diagnose_sonic_signal.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --n 50000 --seed 1
  python scripts/diagnose_sonic_signal.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --pairs-file my_pairs.json
  python scripts/diagnose_sonic_signal.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --artist "Aaliyah" --k 200 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.similarity.hybrid import build_hybrid_embedding, transition_similarity_end_to_start


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


def saturated_flag(stats: Dict[str, float], spread_thresh: float = 0.01, hi_thresh: float = 0.999) -> bool:
    return stats["spread"] < spread_thresh and stats["p99"] > hi_thresh


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    try:
        corr = np.corrcoef(a[mask], b[mask])[0, 1]
    except Exception:
        return float("nan")
    if not np.isfinite(corr):
        return float("nan")
    return float(corr)


def resolve_optional_arrays(npz_path: Path) -> Dict[str, Optional[np.ndarray]]:
    """Load optional album/track number metadata if present in the NPZ."""
    opts = {"album_ids": None, "album_names": None, "track_numbers": None}
    try:
        data = np.load(npz_path, allow_pickle=True)
        for key in opts:
            if key in data:
                opts[key] = data[key]
    except Exception:
        return opts
    return opts


def sample_random_pairs(n: int, N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ia = rng.integers(0, N, size=n, dtype=int)
    ib = rng.integers(0, N, size=n, dtype=int)
    same = ia == ib
    if same.any():
        ib[same] = (ib[same] + 1) % N
    return ia, ib


def _build_group_index(keys: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[str(key)].append(idx)
    return groups


def sample_same_artist(
    n: int,
    artist_keys: Sequence[str],
    rng: np.random.Generator,
    target_artist: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = _build_group_index(artist_keys)
    if target_artist is not None:
        tgt = str(target_artist).lower().strip()
        groups = {k: v for k, v in groups.items() if k == tgt}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return np.array([], dtype=int), np.array([], dtype=int)

    artists = list(groups.keys())
    ia_list: List[int] = []
    ib_list: List[int] = []
    for _ in range(n):
        artist = artists[rng.integers(0, len(artists))]
        idxs = groups[artist]
        a, b = rng.choice(idxs, size=2, replace=False)
        ia_list.append(int(a))
        ib_list.append(int(b))
    return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)


def sample_same_album(
    n: int,
    album_ids: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if album_ids is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    groups = _build_group_index([str(a) for a in album_ids])
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return np.array([], dtype=int), np.array([], dtype=int)
    albums = list(groups.keys())
    ia_list: List[int] = []
    ib_list: List[int] = []
    for _ in range(n):
        album = albums[rng.integers(0, len(albums))]
        idxs = groups[album]
        a, b = rng.choice(idxs, size=2, replace=False)
        ia_list.append(int(a))
        ib_list.append(int(b))
    return np.array(ia_list, dtype=int), np.array(ib_list, dtype=int)


def sample_adjacent_in_album(
    album_ids: Optional[np.ndarray],
    track_numbers: Optional[np.ndarray],
    rng: np.random.Generator,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if album_ids is None or track_numbers is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    pairs: List[Tuple[int, int]] = []
    by_album: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for idx, (aid, tnum) in enumerate(zip(album_ids, track_numbers)):
        try:
            tnum_val = float(tnum)
        except Exception:
            continue
        by_album[str(aid)].append((idx, tnum_val))
    for album, items in by_album.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[1])
        for i in range(len(items) - 1):
            pairs.append((items[i][0], items[i + 1][0]))
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    ia, ib = zip(*pairs)
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def _coerce_to_indices(
    bundle,
    items: Iterable[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    ia: List[int] = []
    ib: List[int] = []
    meta: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in items:
        prev_id = str(row.get("prev_id") or row.get("prev") or "").strip()
        cur_id = str(row.get("cur_id") or row.get("cur") or "").strip()
        label = row.get("label") or ""
        if not prev_id or not cur_id:
            continue
        prev_idx = bundle.track_id_to_index.get(prev_id)
        cur_idx = bundle.track_id_to_index.get(cur_id)
        if prev_idx is None or cur_idx is None:
            if prev_idx is None:
                missing.append(prev_id)
            if cur_idx is None:
                missing.append(cur_id)
            continue
        ia.append(int(prev_idx))
        ib.append(int(cur_idx))
        meta.append({"prev_id": prev_id, "cur_id": cur_id, "label": label})
    return np.array(ia, dtype=int), np.array(ib, dtype=int), meta


def _load_pairs_file(path: Path) -> List[Dict[str, str]]:
    ext = path.suffix.lower()
    pairs: List[Dict[str, str]] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        if not isinstance(data, list):
            raise ValueError("JSON pairs file must be a list or an object with a 'pairs' key.")
        for row in data:
            if not isinstance(row, dict):
                continue
            pairs.append(
                {
                    "prev_id": row.get("prev_id") or row.get("prev") or row.get("from"),
                    "cur_id": row.get("cur_id") or row.get("cur") or row.get("to"),
                    "label": row.get("label") or "",
                }
            )
    else:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(
                    {
                        "prev_id": row.get("prev_id") or row.get("prev") or row.get("from"),
                        "cur_id": row.get("cur_id") or row.get("cur") or row.get("to"),
                        "label": row.get("label") or "",
                    }
                )
    return [p for p in pairs if p.get("prev_id") and p.get("cur_id")]


def compute_metrics(
    bundle,
    prev_idx: np.ndarray,
    cur_idx: np.ndarray,
    *,
    X_start: np.ndarray,
    X_end: np.ndarray,
    X_sonic: np.ndarray,
    X_genre: Optional[np.ndarray],
    center_transitions: bool,
    hybrid_model: Optional[Any],
) -> Dict[str, np.ndarray]:
    metrics: Dict[str, np.ndarray] = {}
    sonic_norm = l2norm_rows(X_sonic)
    metrics["S_sonic"] = np.sum(sonic_norm[prev_idx] * sonic_norm[cur_idx], axis=1)

    start_used = X_start
    end_used = X_end
    if center_transitions:
        end_center = X_end - X_end.mean(axis=0, keepdims=True)
        start_center = X_start - X_start.mean(axis=0, keepdims=True)
        end_norm_center = l2norm_rows(end_center)
        start_norm_center = l2norm_rows(start_center)
        t_center_cos = np.sum(end_norm_center[prev_idx] * start_norm_center[cur_idx], axis=1)
        metrics["T_centered_cos"] = t_center_cos
        metrics["T_centered_rescaled"] = np.clip((t_center_cos + 1.0) / 2.0, 0.0, 1.0)
    # Raw transition cosine
    end_norm = l2norm_rows(end_used)
    start_norm = l2norm_rows(start_used)
    metrics["T_raw"] = np.sum(end_norm[prev_idx] * start_norm[cur_idx], axis=1)

    if X_genre is not None:
        genre_norm = l2norm_rows(X_genre)
        metrics["G_genre"] = np.sum(genre_norm[prev_idx] * genre_norm[cur_idx], axis=1)

    if hybrid_model is not None:
        emb = hybrid_model.embedding
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        metrics["S_hybrid"] = np.sum(emb_norm[prev_idx] * emb_norm[cur_idx], axis=1)
    return metrics


def correlations(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    corr: Dict[str, float] = {}
    def c(a: str, b: str) -> float:
        return pearson_corr(metrics[a], metrics[b]) if a in metrics and b in metrics else float("nan")
    corr["corr_S_sonic__G_genre"] = c("S_sonic", "G_genre")
    corr["corr_S_sonic__T_raw"] = c("S_sonic", "T_raw")
    corr["corr_S_sonic__T_centered_rescaled"] = c("S_sonic", "T_centered_rescaled")
    corr["corr_T_centered_rescaled__G_genre"] = c("T_centered_rescaled", "G_genre")
    return corr


def print_mode(name: str, count: int, metrics: Dict[str, np.ndarray], corr: Dict[str, float]) -> Dict[str, Any]:
    print(f"\n=== Mode: {name} | pairs={count} ===")
    out: Dict[str, Any] = {"count": count, "metrics": {}, "correlations": corr}
    for key in ["S_sonic", "S_hybrid", "T_raw", "T_centered_cos", "T_centered_rescaled", "G_genre"]:
        if key not in metrics:
            continue
        vals = metrics[key]
        stats = summary_stats(vals)
        stats["saturated"] = saturated_flag(stats)
        print(
            f"{key:20s} mean={stats['mean']:.4f} p01={stats['p01']:.4f} p10={stats['p10']:.4f} "
            f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} p99={stats['p99']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f} var={stats['var']:.6f} std={stats['std']:.4f} "
            f"spread={stats['spread']:.4f} saturated={stats['saturated']}"
        )
        out["metrics"][key] = stats
    for k, v in corr.items():
        print(f"{k:35s}: {v if np.isfinite(v) else 'nan'}")
    out["correlations"] = {k: (float(v) if np.isfinite(v) else None) for k, v in corr.items()}
    return out


def build_pairs_from_manual(bundle, pairs_file: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rows = _load_pairs_file(pairs_file)
    ia, ib, meta = _coerce_to_indices(bundle, rows)
    missing = []
    for row in rows:
        if row.get("prev_id") not in bundle.track_id_to_index:
            missing.append(row.get("prev_id"))
        if row.get("cur_id") not in bundle.track_id_to_index:
            missing.append(row.get("cur_id"))
    if missing:
        sample_known = list(bundle.track_id_to_index.keys())[:5]
        print(f"Warning: {len(set(missing))} ids from pairs file missing in artifact. Sample known track_ids: {sample_known}")
    return ia, ib, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose S_sonic saturation vs transitions/genre.")
    parser.add_argument("--artifact", required=True, help="Path to artifact npz")
    parser.add_argument("--n", type=int, default=50000, help="Pairs to sample for random/same_artist (default 50000)")
    parser.add_argument("--k", type=int, default=5000, help="Pairs to sample for artist-specific mode (default 5000)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    parser.add_argument("--pairs-file", type=Path, help="JSON/CSV listing prev_id,cur_id[,label]")
    parser.add_argument("--artist", type=str, help="Focus on a specific artist key for same-artist sampling")
    parser.add_argument("--center-transitions", action="store_true", help="Apply DS mean-centering on start/end before cosine and rescale.")
    parser.add_argument("--out", type=Path, help="Optional path to write JSON report")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    bundle = load_artifact_bundle(artifact_path)
    rng = np.random.default_rng(args.seed)

    N = int(bundle.track_ids.shape[0])
    print(f"Loaded artifact: {artifact_path.resolve()}")
    print(f"Tracks: {N}")

    # Optional metadata (album ids / track numbers)
    opt = resolve_optional_arrays(artifact_path)
    album_ids = opt.get("album_ids")
    album_names = opt.get("album_names")
    track_numbers = opt.get("track_numbers")
    if album_ids is None and album_names is not None:
        album_ids = album_names
    has_album = album_ids is not None
    has_track_numbers = track_numbers is not None

    # Matrices
    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    X_start = getattr(bundle, "X_sonic_start", None)
    X_end = getattr(bundle, "X_sonic_end", None)
    if X_sonic is None:
        raise SystemExit("Artifact missing X_sonic; cannot compute S_sonic.")
    if X_start is None or X_end is None:
        print("Warning: X_sonic_start/end missing; transitions will fall back to X_sonic.")
        X_start = X_sonic
        X_end = X_sonic

    hybrid_model = None
    if X_genre is not None:
        try:
            hybrid_model = build_hybrid_embedding(
                X_sonic,
                X_genre,
                n_components_sonic=32,
                n_components_genre=32,
                w_sonic=1.0,
                w_genre=1.0,
                random_seed=args.seed,
            )
            print("Hybrid embedding built (StandardScaler+PCA on X_sonic and X_genre_smoothed, row-L2 at scoring).")
        except Exception as exc:
            print(f"Hybrid embedding skipped due to error: {exc}")
    else:
        print("Hybrid embedding skipped: X_genre_smoothed missing.")

    report: Dict[str, Any] = {
        "artifact": str(artifact_path),
        "n_tracks": N,
        "uses": {
            "S_sonic_matrix": "X_sonic (aggregate)",
            "transitions_start": "X_sonic_start" if getattr(bundle, "X_sonic_start", None) is not None else "X_sonic (fallback)",
            "transitions_end": "X_sonic_end" if getattr(bundle, "X_sonic_end", None) is not None else "X_sonic (fallback)",
            "center_transitions": bool(args.center_transitions),
        },
        "modes": {},
    }

    # Mode: random
    ia, ib = sample_random_pairs(args.n, N, rng)
    metrics = compute_metrics(
        bundle,
        ia,
        ib,
        X_start=X_start,
        X_end=X_end,
        X_sonic=X_sonic,
        X_genre=X_genre,
        center_transitions=args.center_transitions,
        hybrid_model=hybrid_model,
    )
    corr = correlations(metrics)
    report["modes"]["random"] = print_mode("random", len(ia), metrics, corr)

    # Mode: same_artist (global or specific)
    ia, ib = sample_same_artist(args.n, bundle.artist_keys, rng, None if args.artist is None else args.artist)
    if ia.size > 0:
        metrics_sa = compute_metrics(
            bundle,
            ia,
            ib,
            X_start=X_start,
            X_end=X_end,
            X_sonic=X_sonic,
            X_genre=X_genre,
            center_transitions=args.center_transitions,
            hybrid_model=hybrid_model,
        )
        corr_sa = correlations(metrics_sa)
        label = f"same_artist ({args.artist})" if args.artist else "same_artist"
        report["modes"][label] = print_mode(label, len(ia), metrics_sa, corr_sa)
    else:
        msg = "specific artist" if args.artist else "artist-level"
        print(f"\n=== Mode: same_artist skipped (no {msg} pairs available) ===")

    # Mode: same_album
    if has_album:
        ia, ib = sample_same_album(args.n, album_ids, rng)
        if ia.size > 0:
            metrics_alb = compute_metrics(
                bundle,
                ia,
                ib,
                X_start=X_start,
                X_end=X_end,
                X_sonic=X_sonic,
                X_genre=X_genre,
                center_transitions=args.center_transitions,
                hybrid_model=hybrid_model,
            )
            corr_alb = correlations(metrics_alb)
            report["modes"]["same_album"] = print_mode("same_album", len(ia), metrics_alb, corr_alb)
        else:
            print("\n=== Mode: same_album skipped (no album groups with >=2 tracks) ===")
    else:
        print("\n=== Mode: same_album skipped (album_ids/album_names missing in artifact) ===")

    # Mode: adjacent_in_album
    if has_album and has_track_numbers:
        ia, ib = sample_adjacent_in_album(album_ids, track_numbers, rng, max_pairs=args.n)
        if ia.size > 0:
            metrics_adj = compute_metrics(
                bundle,
                ia,
                ib,
                X_start=X_start,
                X_end=X_end,
                X_sonic=X_sonic,
                X_genre=X_genre,
                center_transitions=args.center_transitions,
                hybrid_model=hybrid_model,
            )
            corr_adj = correlations(metrics_adj)
            report["modes"]["adjacent_in_album"] = print_mode("adjacent_in_album", len(ia), metrics_adj, corr_adj)
        else:
            print("\n=== Mode: adjacent_in_album skipped (no adjacent track numbers found) ===")
    elif has_album:
        print("\n=== Mode: adjacent_in_album skipped (track_numbers missing) ===")
    else:
        # already messaged missing album
        pass

    # Mode: manual pairs via file
    if args.pairs_file:
        ia, ib, meta = build_pairs_from_manual(bundle, args.pairs_file)
        if ia.size == 0:
            print(f"\n=== Mode: far_apart_manual skipped (no valid pairs resolved from {args.pairs_file}) ===")
        else:
            metrics_manual = compute_metrics(
                bundle,
                ia,
                ib,
                X_start=X_start,
                X_end=X_end,
                X_sonic=X_sonic,
                X_genre=X_genre,
                center_transitions=args.center_transitions,
                hybrid_model=hybrid_model,
            )
            corr_manual = correlations(metrics_manual)
            report["modes"]["far_apart_manual"] = print_mode(
                f"far_apart_manual ({args.pairs_file.name})", len(ia), metrics_manual, corr_manual
            )
            print("\nPair-by-pair scores:")
            titles = getattr(bundle, "track_titles", None)
            for i, (a, b) in enumerate(zip(ia, ib)):
                meta_row = meta[i] if i < len(meta) else {}
                row_out = {
                    "prev_id": meta_row.get("prev_id"),
                    "cur_id": meta_row.get("cur_id"),
                    "label": meta_row.get("label") or "",
                    "S_sonic": float(metrics_manual["S_sonic"][i]),
                    "T_raw": float(metrics_manual["T_raw"][i]),
                    "T_centered_rescaled": float(metrics_manual.get("T_centered_rescaled", [float("nan")]*len(ia))[i]),
                    "G_genre": float(metrics_manual.get("G_genre", [float("nan")]*len(ia))[i]),
                }
                if "S_hybrid" in metrics_manual:
                    row_out["S_hybrid"] = float(metrics_manual["S_hybrid"][i])
                title_prev = titles[a] if titles is not None and a < len(titles) else ""
                title_cur = titles[b] if titles is not None and b < len(titles) else ""
                label_txt = f" [{row_out['label']}]" if row_out["label"] else ""
                print(
                    f"{row_out['prev_id']} -> {row_out['cur_id']}{label_txt} | "
                    f"S_sonic={row_out['S_sonic']:.4f} "
                    f"T_raw={row_out['T_raw']:.4f} "
                    f"T_centered_rescaled={row_out['T_centered_rescaled']:.4f} "
                    f"G_genre={row_out.get('G_genre', float('nan')):.4f} "
                    f"S_hybrid={row_out.get('S_hybrid', float('nan')):.4f} "
                    f"prev_title='{title_prev}' cur_title='{title_cur}'"
                )
            report["modes"]["far_apart_manual"]["pairs"] = meta

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {args.out}")


if __name__ == "__main__":
    main()
