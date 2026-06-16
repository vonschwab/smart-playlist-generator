#!/usr/bin/env python3
"""
Genre Arc Calibration Harness
==============================
Sweeps (genre_admission_percentile, genre_arc_floor_percentile, weight_genre)
per cohesion mode across five reference seeds, records feasibility + arc-quality
metrics, and emits a markdown shortlist you audition by ear.

Read-only. Idempotent.

Usage:
    python scripts/calibrate_genre_arc.py                 # full sweep, all modes
    python scripts/calibrate_genre_arc.py --mode narrow   # one mode
    python scripts/calibrate_genre_arc.py --quick         # tiny grid, fast
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds

ARTIFACT = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"

REFERENCE = {
    "charli_xcx":  {"tid": "4637b6d6b70e473818f58a474c6b0df4", "label": "Charli XCX"},
    "real_estate": {"tid": "0c138cb426a104b0d560c46f390f4226", "label": "Real Estate"},
    "bill_evans":  {"tid": "ab3f750afa7a912ba3cb790bdaf4a559", "label": "Bill Evans"},
    "beach_house": {"tid": "08c709d12e59bdb2bf64addf4215881d", "label": "Beach House"},
    "minor_threat":{"tid": "9969ea5d00040d1cc43ef84ac0bfe296", "label": "Minor Threat"},
}

MODES = ["strict", "narrow", "dynamic", "discover"]

GRID_FULL = {
    "P_admit":  [0.85, 0.90, 0.93],
    "P_arc":    [0.75, 0.80, 0.85],
    "w_genre":  [0.12, 0.18, 0.25],
}
GRID_QUICK = {
    "P_admit":  [0.88, 0.92],
    "P_arc":    [0.78, 0.85],
    "w_genre":  [0.15, 0.22],
}


def _overrides(mode: str, cfg: dict) -> dict:
    return {
        "pier_bridge": {
            "genre_steering_enabled": True,
            "dj_route_shape": "ladder",
            f"weight_genre_{mode}": cfg["w_genre"],
            f"genre_admission_percentile_{mode}": cfg["P_admit"],
            f"genre_arc_floor_percentile_{mode}": cfg["P_arc"],
            "infeasible_handling": {
                "enabled": True,
                "genre_arc_relaxation_enabled": True,
                "min_genre_arc_percentile": 0.40,
            },
        }
    }


def _arc_monotonicity(D: np.ndarray, ti: dict, track_ids, last_tid: str | None) -> float:
    if last_tid not in ti:
        return 0.0
    b = D[ti[last_tid]]
    sims = []
    for t in track_ids:
        idx = ti.get(str(t))
        if idx is None:
            continue
        n = np.linalg.norm(D[idx])
        if n < 1e-9:
            continue
        sims.append(float(D[idx] @ b))
    if len(sims) < 2:
        return 1.0
    return sum(1 for i in range(len(sims) - 1) if sims[i + 1] >= sims[i] - 0.05) / (len(sims) - 1)


def _mean_arc_sim(D: np.ndarray, ti: dict, track_ids, g_targets: list[np.ndarray] | None) -> float:
    if g_targets is None or len(g_targets) == 0:
        return 0.0
    sims = []
    interior = [str(t) for t in track_ids[1:-1]]
    for i, t in enumerate(interior):
        idx = ti.get(t)
        if idx is None:
            continue
        j = min(i, len(g_targets) - 1)
        n = np.linalg.norm(D[idx])
        if n < 1e-9:
            continue
        sims.append(float(D[idx] @ g_targets[j]))
    return float(np.mean(sims)) if sims else 0.0


def run_cell(bundle, mode: str, seed: str, tid: str, cfg: dict, length: int = 28) -> dict:
    ov = _overrides(mode, cfg)
    try:
        res = generate_playlist_ds(
            artifact_path=str(ARTIFACT), seed_track_id=tid,
            mode=mode, length=length, random_seed=42, overrides=ov,
        )
    except Exception as exc:
        return {"ok": False, "err": str(exc)[:100]}
    D = bundle.X_genre_dense
    ti = bundle.track_id_to_index
    last = str(res.track_ids[-1]) if res.track_ids else None
    mono = _arc_monotonicity(D, ti, res.track_ids[1:-1], last)
    distinct = len({str(t) for t in res.track_ids})
    min_t = res.metrics.get("min_transition", 0.0) or 0.0
    return {
        "ok": True,
        "n": len(res.track_ids),
        "mono": round(mono, 3),
        "distinct": distinct,
        "min_T": round(float(min_t), 3),
    }


def shortlist(results_by_cfg: list[dict], top_n: int = 3) -> list[dict]:
    feasible = [r for r in results_by_cfg if r["n_feasible"] == r["n_seeds"]]
    if not feasible:
        feasible = sorted(results_by_cfg, key=lambda r: -r["n_feasible"])[:top_n]
    return sorted(feasible, key=lambda r: (-r["mean_mono"], -r["mean_min_T"]))[:top_n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Genre arc floor/weight calibration sweep")
    ap.add_argument("--mode", choices=MODES, default=None, help="single mode (default: all)")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast check")
    ap.add_argument("--length", type=int, default=28)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "run_audits")
    args = ap.parse_args()

    if not ARTIFACT.exists():
        print(f"ERROR: artifact not found: {ARTIFACT}")
        return 1

    load_artifact_bundle.cache_clear()
    print("Loading artifact …")
    bundle = load_artifact_bundle(str(ARTIFACT))

    grid = GRID_QUICK if args.quick else GRID_FULL
    modes = [args.mode] if args.mode else MODES
    seeds = list(REFERENCE.items())
    cfgs = [{"P_admit": pa, "P_arc": pc, "w_genre": wg}
            for pa in grid["P_admit"] for pc in grid["P_arc"] for wg in grid["w_genre"]]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"genre_arc_calibration_{ts}.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Genre Arc Calibration — {ts}\n"]
    lines.append(f"Grid: P_admit={grid['P_admit']} P_arc={grid['P_arc']} w_genre={grid['w_genre']}\n")

    for mode in modes:
        print(f"\n{'='*60}\nMODE: {mode}\n{'='*60}")
        cfg_results: list[dict] = []
        for cfg in cfgs:
            row: dict = {"cfg": cfg, "n_seeds": len(seeds), "n_feasible": 0,
                         "mono_vals": [], "min_t_vals": []}
            for name, info in seeds:
                r = run_cell(bundle, mode, name, info["tid"], cfg, length=args.length)
                if r["ok"]:
                    row["n_feasible"] += 1
                    row["mono_vals"].append(r["mono"])
                    row["min_t_vals"].append(r["min_T"])
                    print(f"  {name:12} cfg=({cfg['P_admit']},{cfg['P_arc']},{cfg['w_genre']:.2f}) "
                          f"OK n={r['n']} mono={r['mono']:.2f} minT={r['min_T']:.2f}")
                else:
                    print(f"  {name:12} cfg=({cfg['P_admit']},{cfg['P_arc']},{cfg['w_genre']:.2f}) "
                          f"INFEASIBLE {r['err'][:50]}")
            row["mean_mono"] = float(np.mean(row["mono_vals"])) if row["mono_vals"] else 0.0
            row["mean_min_T"] = float(np.mean(row["min_t_vals"])) if row["min_t_vals"] else 0.0
            cfg_results.append(row)

        sl = shortlist(cfg_results)
        lines.append(f"\n## {mode}\n")
        lines.append("| P_admit | P_arc | w_genre | feasible | mean_mono | mean_minT |\n")
        lines.append("|---------|-------|---------|----------|-----------|-----------|\n")
        for r in sl:
            c = r["cfg"]
            lines.append(f"| {c['P_admit']} | {c['P_arc']} | {c['w_genre']:.2f} | "
                         f"{r['n_feasible']}/{r['n_seeds']} | {r['mean_mono']:.3f} | {r['mean_min_T']:.3f} |\n")
        print(f"\n  SHORTLIST ({mode}):")
        for r in sl:
            c = r["cfg"]
            print(f"    P_admit={c['P_admit']} P_arc={c['P_arc']} w_genre={c['w_genre']:.2f} "
                  f"feasible={r['n_feasible']}/{r['n_seeds']} mono={r['mean_mono']:.3f}")

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
