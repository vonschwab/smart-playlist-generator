"""Phase-1 sonic metrics: cosine spread + per-tower contribution + generation smoke.

Run on the CURRENT artifact (baseline) BEFORE editing runtime code, then again on
the rebuilt artifact (after). Writes JSON under docs/run_audits/sonic_phase1/.

    python scripts/sonic_phase1_metrics.py --artifact <path> --label baseline
    python scripts/sonic_phase1_metrics.py --artifact <path> --label after --generate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 86-dim tower layout
R, T, H = slice(0, 9), slice(9, 66), slice(66, 86)

SEEDS = [
    "Green-House", "Boards of Canada", "Autechre", "Charli XCX",
    "Bill Evans", "Jean-Yves Thibaudet", "William Tyler", "Elliott Smith",
    "Duster", "Real Estate", "Slowdive", "Sonic Youth", "Minor Threat",
    "James Brown", "J Dilla", "Beyoncé", "Grouper",
]


def _l2(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def cosine_spread_to_seed(X: np.ndarray, seed_idx: int) -> Dict[str, float]:
    Xn = _l2(X.astype(np.float64))
    mask = np.ones(len(Xn), dtype=bool)
    mask[seed_idx] = False
    sims = np.sort(Xn[mask] @ Xn[seed_idx])[::-1]
    return {
        "max": float(sims[0]),
        "p99": float(np.percentile(sims, 99)),
        "p90": float(np.percentile(sims, 90)),
        "median": float(np.percentile(sims, 50)),
    }


def per_tower_contribution(X: np.ndarray) -> Dict[str, float]:
    """Mean fraction of squared row norm carried by each tower block."""
    Xs = X.astype(np.float64)
    e_r = (Xs[:, R] ** 2).sum(axis=1)
    e_t = (Xs[:, T] ** 2).sum(axis=1)
    e_h = (Xs[:, H] ** 2).sum(axis=1)
    tot = np.maximum(e_r + e_t + e_h, 1e-12)
    return {
        "rhythm": float(np.mean(e_r / tot)),
        "timbre": float(np.mean(e_t / tot)),
        "harmony": float(np.mean(e_h / tot)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="data/artifacts/beat3tower_32k/data_matrices_step1.npz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--generate", action="store_true",
                    help="also run a generation smoke for the 5 core seeds (uses current code)")
    args = ap.parse_args()

    from src.features.artifacts import load_artifact_bundle
    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(args.artifact)
    X = b.X_sonic
    artists = np.array([str(a) for a in b.track_artists])

    def medoid(name: str):
        idx = np.where(np.char.lower(artists) == name.lower())[0]
        if len(idx) == 0:
            return None
        sub = _l2(X[idx].astype(np.float64))
        c = sub.mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return int(idx[int(np.argmax(sub @ c))])

    report: Dict[str, object] = {
        "label": args.label,
        "artifact": args.artifact,
        "sonic_variant": str(b.sonic_variant),
        "sonic_pre_scaled": bool(b.sonic_pre_scaled),
        "per_tower_contribution": per_tower_contribution(X),
        "seeds": {},
    }
    for nm in SEEDS:
        s = medoid(nm)
        if s is None:
            report["seeds"][nm] = {"missing": True}  # type: ignore[index]
            continue
        report["seeds"][nm] = {  # type: ignore[index]
            "track_id": str(b.track_ids[s]),
            "title": str(b.track_titles[s]) if b.track_titles is not None else None,
            "cosine_spread": cosine_spread_to_seed(X, s),
        }

    if args.generate:
        from tests.support.gui_fidelity import generate_like_gui
        gen: Dict[str, object] = {}
        for nm in ["Charli XCX", "Real Estate", "Bill Evans", "Beach House", "Minor Threat"]:
            s = medoid(nm)
            if s is None:
                gen[nm] = {"missing": True}
                continue
            try:
                res = generate_like_gui(
                    seeds=[str(b.track_ids[s])],
                    cohesion_mode="narrow", genre_mode="narrow",
                    sonic_mode="narrow", pace_mode="narrow",
                    artifact_path=args.artifact, length=20,
                )
                tids = [str(t) for t in res.track_ids]
                arts = [str(b.track_artists[b.track_id_to_index[t]]) for t in tids
                        if t in b.track_id_to_index]
                gen[nm] = {"length": len(tids), "distinct_artists": len(set(arts))}
            except Exception as exc:  # record, don't crash the metrics run
                gen[nm] = {"error": repr(exc)}
        report["generation_smoke"] = gen

    out_dir = ROOT / "docs" / "run_audits" / "sonic_phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(report["per_tower_contribution"], indent=2))


if __name__ == "__main__":
    main()
