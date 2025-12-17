#!/usr/bin/env python3
"""
Batch A/B evaluation of DS playlists under different sonic variants (raw/z), using the
existing SONIC_SIM_VARIANT gating. Defaults remain unchanged; this script only sets the
env per run.

Examples:
  python scripts/batch_eval_sonic_variant.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --n-seeds 25 --seed 1
  python scripts/batch_eval_sonic_variant.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --variants raw,z --seeds-file diagnostics/seeds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ensure project root
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.playlist.pipeline import generate_playlist_ds
from src.similarity.hybrid import build_hybrid_embedding, transition_similarity_end_to_start
from src.similarity.sonic_variant import compute_sonic_variant_norm


def _l2(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def percentile(arr: Sequence[float], q: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.array(arr, dtype=float), q))


def correlations(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b) or len(a) < 2:
        return float("nan")
    return float(np.corrcoef(np.array(a, dtype=float), np.array(b, dtype=float))[0, 1])


def summarize_edges(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["T", "H", "S", "G", "T_centered_rescaled", "T_used", "T_raw_uncentered"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [float(e.get(k, float("nan"))) for e in edges if k in e]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        out[f"{k}_p10"] = percentile(vals, 10)
        out[f"{k}_p50"] = percentile(vals, 50)
        out[f"{k}_p90"] = percentile(vals, 90)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_min"] = float(np.min(vals))
        out[f"{k}_spread"] = out[f"{k}_p90"] - out[f"{k}_p10"]
        out[f"{k}_count"] = len(vals)
    s_vals = [float(e.get("S", float("nan"))) for e in edges if "S" in e]
    t_vals = [float(e.get("T", float("nan"))) for e in edges if "T" in e]
    tc_vals = [float(e.get("T_centered_rescaled", float("nan"))) for e in edges if "T_centered_rescaled" in e]
    if s_vals and t_vals:
        s_arr = np.array(s_vals, dtype=float)
        t_arr = np.array(t_vals, dtype=float)
        mask = np.isfinite(s_arr) & np.isfinite(t_arr)
        if mask.sum() >= 2:
            out["corr_S_T"] = float(np.corrcoef(s_arr[mask], t_arr[mask])[0, 1])
    if s_vals and tc_vals:
        s_arr = np.array(s_vals, dtype=float)
        tc_arr = np.array(tc_vals, dtype=float)
        mask = np.isfinite(s_arr) & np.isfinite(tc_arr)
        if mask.sum() >= 2:
            out["corr_S_Tc"] = float(np.corrcoef(s_arr[mask], tc_arr[mask])[0, 1])
    if t_vals:
        pairs = []
        for e in edges:
            if "T" in e:
                pairs.append((float(e["T"]), e.get("prev_id"), e.get("cur_id")))
        pairs.sort(key=lambda x: x[0])
        out["weakest_edges"] = pairs[:3]
    return out
def _load_metadata(db_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not db_path or not db_path.exists():
        return {}
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT track_id, file_path, artist, title FROM tracks").fetchall()
        return {
            str(r["track_id"]): {"file_path": r["file_path"], "artist": r["artist"], "title": r["title"]}
            for r in rows
            if r["track_id"] is not None
        }
    except Exception:
        return {}
    finally:
        conn.close()


def load_seeds(artifact_path: Path, n_seeds: int, seed: int, seeds_file: Optional[Path]) -> List[str]:
    if seeds_file:
        ext = seeds_file.suffix.lower()
        seeds: List[str] = []
        if ext == ".json":
            data = json.loads(seeds_file.read_text())
            if isinstance(data, dict) and "seeds" in data:
                data = data["seeds"]
            if not isinstance(data, list):
                raise ValueError("seeds-file JSON must be a list or contain 'seeds'")
            seeds = [str(x) for x in data]
        else:
            with seeds_file.open() as f:
                for line in f:
                    seeds.append(line.strip())
        return seeds
    bundle = load_artifact_bundle(artifact_path)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(bundle.track_ids, size=min(n_seeds, bundle.track_ids.shape[0]), replace=False)
    return [str(x) for x in idxs.tolist()]


@contextmanager
def temp_env(var: str, val: str):
    old = os.environ.get(var)
    os.environ[var] = val
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def run_variant(
    variant: str,
    seeds: List[str],
    artifact: Path,
    mode: str,
    length: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    edges_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    tracklists: Dict[str, List[str]] = {}
    with temp_env("SONIC_SIM_VARIANT", variant):
        for seed_id in seeds:
            try:
                t0 = time.perf_counter()
                res = generate_playlist_ds(
                    artifact_path=artifact,
                    seed_track_id=seed_id,
                    num_tracks=length,
                    mode=mode,
                    random_seed=0,
                    sonic_variant=variant,
                )
                runtime = time.perf_counter() - t0
            except Exception as exc:
                rows.append({"seed": seed_id, "error": str(exc)})
                continue
            stats = res.stats
            edge_scores = stats.get("edge_scores", [])
            if not edge_scores:
                # recompute edges post-hoc to ensure we have metrics
                edge_scores = recompute_edges(artifact, res.track_ids, variant)
            edges_by_key[(variant, seed_id)] = edge_scores
            tracklists[seed_id] = res.track_ids
            edge_summary = summarize_edges(edge_scores)
            row = {
                "seed": seed_id,
                "variant": variant,
                "evaluation_mode": "rerun_ds",
                "playlist_len": stats.get("playlist_length"),
                "distinct_artists": stats.get("distinct_artists"),
                "max_artist_share": stats.get("max_artist_share"),
                "below_floor": stats.get("below_floor_count"),
                "min_transition": stats.get("min_transition"),
                "mean_transition": stats.get("mean_transition"),
                "runtime_sec": runtime,
            }
            if (row["min_transition"] is None or np.isnan(row["min_transition"])) and "T_min" in edge_summary:
                row["min_transition"] = edge_summary["T_min"]
            if (row["mean_transition"] is None or (isinstance(row["mean_transition"], float) and np.isnan(row["mean_transition"]))) and "T_mean" in edge_summary:
                row["mean_transition"] = edge_summary["T_mean"]
            row.update(edge_summary)
            # store track_ids for churn
            row["track_ids"] = ";".join(res.track_ids)
            rows.append(row)
    return rows, edges_by_key, tracklists


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def recompute_edges(artifact: Path, track_ids: List[str], variant: str) -> List[Dict[str, Any]]:
    bundle = load_artifact_bundle(artifact)
    id_to_idx = bundle.track_id_to_index
    order = [id_to_idx.get(str(tid)) for tid in track_ids]
    if any(i is None for i in order) or len(order) < 2:
        return []
    order = [int(i) for i in order]
    X_sonic = bundle.X_sonic
    X_end = bundle.X_sonic_end if getattr(bundle, "X_sonic_end", None) is not None else X_sonic
    X_start = bundle.X_sonic_start if getattr(bundle, "X_sonic_start", None) is not None else X_sonic
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    # Sonic variant
    sonic_norm, _ = compute_sonic_variant_norm(X_sonic, variant)
    # Hybrid
    H_norm = None
    try:
        model = build_hybrid_embedding(X_sonic, X_genre)
        emb = model.embedding
        H_norm = _l2(emb)
    except Exception:
        pass
    # Genre
    G_norm = _l2(X_genre) if X_genre is not None else None
    # Transition
    T_norm_end = _l2(X_end) if X_end is not None else None
    T_norm_start = _l2(X_start) if X_start is not None else None
    # centered transitions
    T_center_end = _l2(X_end - X_end.mean(axis=0, keepdims=True)) if X_end is not None else None
    T_center_start = _l2(X_start - X_start.mean(axis=0, keepdims=True)) if X_start is not None else None
    edges: List[Dict[str, Any]] = []
    for i in range(1, len(order)):
        prev = order[i - 1]
        cur = order[i]
        rec: Dict[str, Any] = {
            "prev_id": str(bundle.track_ids[prev]),
            "cur_id": str(bundle.track_ids[cur]),
        }
        rec["S"] = float(sonic_norm[prev] @ sonic_norm[cur])
        if G_norm is not None:
            rec["G"] = float(G_norm[prev] @ G_norm[cur])
        if H_norm is not None:
            rec["H"] = float(H_norm[prev] @ H_norm[cur])
        if T_norm_end is not None and T_norm_start is not None:
            rec["T"] = float(T_norm_end[prev] @ T_norm_start[cur])
        if T_center_end is not None and T_center_start is not None:
            tc = float(T_center_end[prev] @ T_center_start[cur])
            rec["T_centered_rescaled"] = float(np.clip((tc + 1) / 2, 0, 1))
        edges.append(rec)
    return edges
def _edges_with_labels(edges: List[Dict[str, Any]], bundle, meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    labeled = []
    for e in edges:
        prev_id = str(e.get("prev_id"))
        cur_id = str(e.get("cur_id"))
        prev_meta = meta.get(prev_id, {})
        cur_meta = meta.get(cur_id, {})
        labeled.append(
            {
                **e,
                "prev_artist": prev_meta.get("artist") or "",
                "prev_title": prev_meta.get("title") or "",
                "cur_artist": cur_meta.get("artist") or "",
                "cur_title": cur_meta.get("title") or "",
            }
        )
    return labeled


def compare_rows(rows_by_variant: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seeds = set()
    for rows in rows_by_variant.values():
        for r in rows:
            seeds.add(r.get("seed"))
    comparison: List[Dict[str, Any]] = []
    metrics = [
        "playlist_len",
        "distinct_artists",
        "max_artist_share",
        "min_transition",
        "mean_transition",
        "T_p50",
        "T_p90",
        "S_p50",
        "S_p90",
        "S_spread",
        "corr_S_T",
    ]
    for seed in seeds:
        row = {"seed": seed}
        base = rows_by_variant.get("raw", [])
        zrows = rows_by_variant.get("z", [])
        base_row = next((r for r in base if r.get("seed") == seed), None)
        z_row = next((r for r in zrows if r.get("seed") == seed), None)
        if base_row:
            for k, v in base_row.items():
                row[f"raw_{k}"] = v
        if z_row:
            for k, v in z_row.items():
                row[f"z_{k}"] = v
        # churn vs raw
        for var_name, var_rows in rows_by_variant.items():
            if var_name == "raw":
                continue
            vrow = next((r for r in var_rows if r.get("seed") == seed), None)
            if vrow and base_row and "track_ids" in base_row and "track_ids" in vrow:
                raw_set = set(str(x) for x in base_row["track_ids"].split(";"))
                var_set = set(str(x) for x in vrow["track_ids"].split(";"))
                inter = len(raw_set & var_set)
                union = len(raw_set | var_set) or 1
                row[f"jaccard_raw_{var_name}"] = inter / union
        for m in metrics:
            if f"raw_{m}" in row and f"z_{m}" in row:
                try:
                    row[f"delta_{m}"] = float(row[f"z_{m}"]) - float(row[f"raw_{m}"])
                except Exception:
                    pass
        comparison.append(row)
    return comparison
def _run_fixed_order(
    variants: List[str],
    seeds: List[str],
    artifact: Path,
    mode: str,
    length: int,
    base_variant: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, List[str]]]:
    rows_by_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    edges_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    tracklists: Dict[str, List[str]] = {}
    with temp_env("SONIC_SIM_VARIANT", base_variant):
        for seed_id in seeds:
            t0 = time.perf_counter()
            res = generate_playlist_ds(
                artifact_path=artifact,
                seed_track_id=seed_id,
                num_tracks=length,
                mode=mode,
                random_seed=0,
                sonic_variant=base_variant,
            )
            base_runtime = time.perf_counter() - t0
            base_track_ids = res.track_ids
            tracklists[seed_id] = base_track_ids
            base_stats = res.stats
            for variant in variants:
                edge_scores = recompute_edges(artifact, base_track_ids, variant)
                edges_by_key[(variant, seed_id)] = edge_scores
                edge_summary = summarize_edges(edge_scores)
                row = {
                    "seed": seed_id,
                    "variant": variant,
                    "evaluation_mode": "fixed_order",
                    "base_variant": base_variant,
                    "playlist_len": base_stats.get("playlist_length"),
                    "distinct_artists": base_stats.get("distinct_artists"),
                    "max_artist_share": base_stats.get("max_artist_share"),
                    "below_floor": base_stats.get("below_floor_count"),
                    "runtime_sec": base_runtime if variant == base_variant else 0.0,
                    "order_source": base_variant,
                }
                if row["playlist_len"] is None and "T_count" in edge_summary:
                    row["playlist_len"] = edge_summary["T_count"] + 1
                row.update(edge_summary)
                row["track_ids"] = ";".join(base_track_ids)
                rows_by_variant.setdefault(variant, []).append(row)
    return rows_by_variant, edges_by_key, tracklists


def summarize_markdown(rows_by_variant: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    def avg(vals: Iterable[float]) -> float:
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    lines = ["# Sonic Variant A/B Report", ""]
    for variant, rows in rows_by_variant.items():
        lines.append(f"## Variant: {variant}")
        metrics = {
            "S_spread": [r.get("S_spread") for r in rows],
            "T_spread": [r.get("T_spread") for r in rows],
            "H_spread": [r.get("H_spread") for r in rows],
            "G_spread": [r.get("G_spread") for r in rows],
            "corr_S_T": [r.get("corr_S_T") for r in rows],
            "corr_S_Tc": [r.get("corr_S_Tc") for r in rows],
            "min_transition": [r.get("min_transition") for r in rows],
            "mean_transition": [r.get("mean_transition") for r in rows],
            "runtime_sec": [r.get("runtime_sec") for r in rows],
        }
        for k, v in metrics.items():
            lines.append(f"- Avg {k}: {avg(v):.4f}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_m3u(track_ids: List[str], meta: Dict[str, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for tid in track_ids:
        m = meta.get(str(tid), {})
        artist = m.get("artist") or ""
        title = m.get("title") or ""
        line_info = f"#EXTINF:-1,{artist} - {title}".strip(", ")
        lines.append(line_info)
        lines.append(m.get("file_path") or str(tid))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_diff_report(
    seed: str,
    variants: List[str],
    tracklists: Dict[str, Dict[str, List[str]]],
    edges_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    meta: Dict[str, Dict[str, Any]],
    export_dir: Path,
) -> None:
    raw_list = tracklists.get("raw", {}).get(seed)
    if raw_list is None:
        return
    raw_set = set(map(str, raw_list))
    lines = [f"# Seed {seed} diff vs raw"]
    for variant in variants:
        if variant == "raw":
            continue
        v_list = tracklists.get(variant, {}).get(seed)
        if v_list is None:
            continue
        v_set = set(map(str, v_list))
        inter = len(raw_set & v_set)
        union = len(raw_set | v_set) or 1
        jaccard = inter / union
        only_v = [tid for tid in v_list if str(tid) not in raw_set]
        lines.append(f"\n## Variant {variant}")
        lines.append(f"- Jaccard vs raw: {jaccard:.3f}")
        lines.append(f"- Only-in-{variant} ({len(only_v)}): {', '.join(map(str, only_v)) or 'none'}")
        edges = edges_by_key.get((variant, seed), [])
        if edges:
            t_sorted = sorted(edges, key=lambda e: e.get("T", float("inf")))[:5]
            s_sorted = sorted(edges, key=lambda e: e.get("S", float("inf")))[:5]
            def fmt_edge(e, key: str) -> str:
                pid, cid = str(e.get("prev_id")), str(e.get("cur_id"))
                pm = meta.get(pid, {})
                cm = meta.get(cid, {})
                return f"{key}={e.get(key):.3f} | {pm.get('artist','')} - {pm.get('title','')} -> {cm.get('artist','')} - {cm.get('title','')}"
            lines.append("- Bottom 5 by T:")
            for e in t_sorted:
                if "T" in e and np.isfinite(e.get("T", float("nan"))):
                    lines.append(f"  - {fmt_edge(e, 'T')}")
            lines.append("- Bottom 5 by S:")
            for e in s_sorted:
                if "S" in e and np.isfinite(e.get("S", float("nan"))):
                    lines.append(f"  - {fmt_edge(e, 'S')}")
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / f"{seed}__diff.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch A/B eval for sonic variants (raw/z) on DS pipeline.")
    parser.add_argument("--artifact", default="experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz")
    parser.add_argument("--pipeline", default="ds")
    parser.add_argument("--mode", default="dynamic")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variants", default="raw,z,z_clip,whiten_pca", help="Comma-separated variants (raw,centered,z,z_clip,whiten_pca)")
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--eval-mode", choices=["rerun_ds", "fixed_order"], default="rerun_ds", help="rerun_ds runs DS per variant; fixed_order runs DS once (base variant) and recomputes stats per variant.")
    parser.add_argument("--base-variant", default="raw", help="Base variant for fixed_order mode.")
    parser.add_argument("--export-m3u-dir", type=Path, help="Optional directory to write variant playlists and diffs for listening checks.")
    parser.add_argument("--db", type=Path, default=Path("data/metadata.db"), help="Optional metadata.db path for file paths and labels.")
    args = parser.parse_args()

    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    artifact_path = Path(args.artifact)
    seeds = load_seeds(artifact_path, args.n_seeds, args.seed, args.seeds_file)

    rows_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    edges_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    tracklists_by_variant: Dict[str, Dict[str, List[str]]] = {}
    if args.eval_mode == "fixed_order":
        print(f"Fixed-order mode: running base variant {args.base_variant} once per seed, then recomputing stats for {variants}")
        rows_by_variant, edges_by_key, base_tracklists = _run_fixed_order(
            variants, seeds, artifact_path, args.mode, args.length, base_variant=args.base_variant
        )
        for var in variants:
            out_csv = diagnostics_dir / f"ab_sonic_variant_{var}.csv"
            write_csv(out_csv, rows_by_variant.get(var, []))
            print(f"Wrote {out_csv}")
        # tracklists are identical across variants; store per variant
        tracklists_by_variant = {var: base_tracklists for var in variants}
    else:
        for var in variants:
            print(f"Running variant {var} on {len(seeds)} seeds (rerun_ds)...")
            rows, edges_map, tracklists = run_variant(var, seeds, artifact_path, args.mode, args.length)
            rows_by_variant[var] = rows
            edges_by_key.update(edges_map)
            tracklists_by_variant[var] = tracklists
            out_csv = diagnostics_dir / f"ab_sonic_variant_{var}.csv"
            write_csv(out_csv, rows)
            print(f"Wrote {out_csv}")

    compare = compare_rows(rows_by_variant)
    write_csv(diagnostics_dir / "ab_sonic_variant_compare.csv", compare)
    summarize_markdown(rows_by_variant, diagnostics_dir / "AB_SONIC_VARIANT_REPORT.md")
    print("Completed A/B eval; see diagnostics/ for CSV/MD outputs.")

    # Optional export for listening checks
    if args.export_m3u_dir:
        bundle = load_artifact_bundle(artifact_path)
        meta = _load_metadata(args.db)
        # enrich with artifact labels
        for tid, artist, title in zip(bundle.track_ids, getattr(bundle, "track_artists", []) or [], getattr(bundle, "track_titles", []) or []):
            meta.setdefault(str(tid), {}).update({"artist": str(artist), "title": str(title)})
        for variant, seeds_map in tracklists_by_variant.items():
            for seed_id, tids in seeds_map.items():
                m3u_path = args.export_m3u_dir / f"{seed_id}__{variant}.m3u8"
                _write_m3u(tids, meta, m3u_path)
        for seed_id in seeds:
            _write_diff_report(
                seed_id, variants, tracklists_by_variant, edges_by_key, meta, args.export_m3u_dir
            )
        print(f"Wrote M3U playlists and diffs under {args.export_m3u_dir}")


if __name__ == "__main__":
    main()
