#!/usr/bin/env python3
"""
Infer the sonic feature schema used inside DS artifacts by correlating artifact dims
with librosa-derived measurements.

Examples:
  python scripts/infer_sonic_feature_schema.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --n 300 --seed 1
  python scripts/infer_sonic_feature_schema.py --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz --group diagnostics/groups/minor_threat.json --n 200 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.librosa_analyzer import LibrosaAnalyzer
from src.similarity.sonic_schema import dim_label, dim_labels


def _parse_group_file(path: Path) -> List[str]:
    ext = path.suffix.lower()
    rows: List[str] = []
    if ext == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "tracks" in data:
                data = data["tracks"]
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    tid = entry.get("track_id")
                else:
                    tid = entry
                if tid:
                    rows.append(str(tid))
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "track_id" in reader.fieldnames:
                for row in reader:
                    tid = row.get("track_id")
                    if tid:
                        rows.append(str(tid))
            else:
                f.seek(0)
                for line in f:
                    tid = line.strip()
                    if tid:
                        rows.append(tid)
    return rows


def _load_track_files(db_path: str, track_ids: Iterable[str]) -> Dict[str, str]:
    ids = list(dict.fromkeys(str(x) for x in track_ids))
    if not ids:
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT track_id, file_path FROM tracks WHERE track_id IN ({placeholders}) AND file_path IS NOT NULL"
        cursor = conn.execute(query, ids)
        return {str(row["track_id"]): str(row["file_path"]) for row in cursor if row["file_path"]}
    finally:
        conn.close()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float("nan")
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    return float(corr)


def _pad_or_trim(values: List[float], length: int) -> List[float]:
    if len(values) >= length:
        return list(values[:length])
    return values + [0.0] * (length - len(values))


def _build_candidate_arrays(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if not samples:
        return [], {}
    mfcc_len = len(samples[0]["mfcc"])
    chroma_len = len(samples[0]["chroma"])
    mfcc_mat = np.array([_pad_or_trim(s["mfcc"], mfcc_len) for s in samples], dtype=float)
    chroma_mat = np.array([_pad_or_trim(s["chroma"], chroma_len) for s in samples], dtype=float)
    tempo_arr = np.array([float(s["tempo"]) for s in samples], dtype=float)
    centroid_arr = np.array([float(s["centroid"]) for s in samples], dtype=float)
    rms_arr = np.array([float(s["rms"]) for s in samples], dtype=float)
    zcr_arr = np.array([float(s["zcr"]) for s in samples], dtype=float)

    candidates: List[Dict[str, Any]] = []
    for idx in range(mfcc_mat.shape[1]):
        candidates.append(
            {"label": f"mfcc_{idx+1}", "values": mfcc_mat[:, idx], "units": "mfcc_coeff"}
        )
    for idx in range(chroma_mat.shape[1]):
        candidates.append(
            {"label": f"chroma_{idx+1}", "values": chroma_mat[:, idx], "units": "chroma_bin"}
        )
    candidates.append({"label": "bpm", "values": tempo_arr, "units": "bpm"})
    candidates.append({"label": "spectral_centroid", "values": centroid_arr, "units": "hz"})
    candidates.append({"label": "rms_energy", "values": rms_arr, "units": "rms"})
    candidates.append({"label": "zero_crossing_rate", "values": zcr_arr, "units": "ratio"})
    arrays = {"mfcc": mfcc_mat, "chroma": chroma_mat}
    return candidates, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer sonic feature schema by correlating dims vs librosa features.")
    parser.add_argument("--artifact", required=True, help="Path to artifact NPZ")
    parser.add_argument("--db", default="data/metadata.db", help="Path to metadata.db")
    parser.add_argument("--group", type=Path, help="Optional group file (json/csv) with track_ids")
    parser.add_argument("--n", type=int, default=100, help="Max tracks to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="Max wall-clock seconds for feature extraction")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    bundle = load_artifact_bundle(artifact_path)
    labels = dim_labels(bundle)
    rng = np.random.default_rng(args.seed)

    if args.group:
        requested = _parse_group_file(args.group)
        candidates = [tid for tid in requested if str(tid) in bundle.track_id_to_index]
        if not candidates:
            print("No matching track_ids found in the artifact for the provided group.")
            return
    else:
        all_ids = [str(tid) for tid in bundle.track_ids]
        count = min(args.n * 2, len(all_ids))
        candidates = rng.choice(all_ids, size=count, replace=False).tolist()

    track_files = _load_track_files(args.db, candidates)
    analyzer = LibrosaAnalyzer()

    samples: List[Dict[str, Any]] = []
    notes: List[str] = []
    start_time = time.perf_counter()
    processed = 0
    for tid in candidates:
        if len(samples) >= args.n:
            break
        file_path = track_files.get(tid)
        if not file_path:
            notes.append(f"{tid} missing file_path")
            continue
        elapsed = time.perf_counter() - start_time
        if elapsed > args.max_seconds:
            notes.append("Reached --max-seconds budget")
            break
        feats = analyzer.extract_similarity_features(file_path)
        processed += 1
        if not feats:
            notes.append(f"{tid} failed librosa extraction")
            continue
        avg = feats.get("average") or feats
        mfcc = avg.get("mfcc_mean") or avg.get("mfcc")
        chroma = avg.get("chroma_mean")
        if not mfcc or not chroma:
            notes.append(f"{tid} missing mfcc/chroma")
            continue
        samples.append(
            {
                "track_id": tid,
                "idx": bundle.track_id_to_index[str(tid)],
                "mfcc": list(map(float, mfcc)),
                "chroma": list(map(float, chroma)),
                "tempo": avg.get("bpm", 0.0),
                "centroid": avg.get("spectral_centroid", 0.0),
                "rms": avg.get("rms_energy", 0.0),
                "zcr": avg.get("zero_crossing_rate", 0.0),
            }
        )

    if not samples:
        print("No samples processed; adjust --n/--group or ensure library files are accessible.")
        return

    candidates_list, _ = _build_candidate_arrays(samples)
    order = [sample["idx"] for sample in samples]
    X = bundle.X_sonic[np.array(order, dtype=int)]
    dim_results: List[Dict[str, Any]] = []
    for dim_idx in range(X.shape[1]):
        dim_vals = X[:, dim_idx]
        best: Optional[Dict[str, Any]] = None
        runner_up: Optional[Dict[str, Any]] = None
        for cand in candidates_list:
            corr = _safe_corr(dim_vals, cand["values"])
            if not np.isfinite(corr):
                continue
            entry = {"label": cand["label"], "corr": corr, "units": cand["units"]}
            if best is None or abs(corr) > abs(best["corr"]):
                runner_up = best
                best = entry
            elif runner_up is None or abs(corr) > abs(runner_up["corr"]):
                runner_up = entry
        label = dim_label(bundle, dim_idx)
        dim_notes: List[str] = []
        best_label = best["label"] if best else "none"
        best_corr = float(best["corr"]) if best else float("nan")
        best_units = best["units"] if best else ""
        if not best or not np.isfinite(best_corr):
            dim_notes.append("no valid correlation")
        elif abs(best_corr) < 0.25:
            dim_notes.append("weak match")
        dim_results.append(
            {
                "dim_index": dim_idx,
                "dim_label": label,
                "best_label": best_label,
                "best_corr": best_corr,
                "best_units": best_units,
                "runner_up": runner_up,
                "notes": dim_notes,
            }
        )

    blocks: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for entry in dim_results:
        prefix = entry["best_label"].split("_")[0]
        if not prefix:
            prefix = "unknown"
        block = blocks.setdefault(prefix, {"start": entry["dim_index"], "end": entry["dim_index"], "dims": []})
        block["start"] = min(block["start"], entry["dim_index"])
        block["end"] = max(block["end"], entry["dim_index"])
        block["dims"].append(entry["dim_index"])

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics_dir / f"sonic_feature_schema_{ts}.json"
    report = {
        "artifact": str(artifact_path),
        "timestamp_utc": ts,
        "samples": [sample["track_id"] for sample in samples],
        "dim_labels": labels,
        "dimensions": dim_results,
        "blocks": [
            {"block": name, "range": [info["start"], info["end"]], "count": len(info["dims"])}
            for name, info in blocks.items()
        ],
        "notes": notes,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_path = diagnostics_dir / "sonic_feature_schema_runs.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{ts} samples={len(samples)} dim={X.shape[1]} blocks={list(blocks.keys())}\n")

    md_path = diagnostics_dir / "SONIC_FEATURE_SCHEMA_REPORT.md"
    lines: List[str] = [
        "# Sonic Feature Schema Report",
        "",
        f"- Artifact: {artifact_path}",
        f"- Samples: {len(samples)}",
        "",
        "## Block inference",
    ]
    for block in report["blocks"]:
        lines.append(f"- {block['block']}: dims {block['range'][0]}-{block['range'][1]} ({block['count']} dims)")
    lines.append("")
    target_dims = {25: "tempo/bpm", 26: "spectral centroid"}
    for idx, desc in target_dims.items():
        if idx < len(dim_results):
            entry = dim_results[idx]
            lines.append(
                f"- {entry['dim_label']} (dim {entry['dim_index']}): best match {entry['best_label']} "
                f"(corr={entry['best_corr']:.3f})"
            )
    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote schema report to {report_path}")
    print(f"Updated summary at {md_path}")


if __name__ == "__main__":
    main()
