"""Harmony-weight sweep: the 0.30 harmony weight was calibrated when the harmony
tower was noisy. With key-invariant (2DFTM) harmony, the optimal blend weight is
probably higher. Sweep w_harmony (rhythm fixed 0.20, timbre fixed 0.50) and score
the blend against audition verdicts, for both current and 2DFTM harmony.

If the 2DFTM blend peaks well above the +0.361 it gets at w=0.30, the rebuild
buys more than the Gate-1 delta suggested.

Read-only; reuses caches + artifact.

Usage:
    python scripts/sonic_harmony_weight_sweep.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
AUDITION_DIR = ROOT / "docs/run_audits/sonic_audition"
HARM_CACHE = ROOT / "docs/run_audits/sonic_phase2/richer_harmony_cache.npz"
OUT = ROOT / "docs/run_audits/sonic_phase2/harmony_weight_sweep.json"
W_RHYTHM, W_TIMBRE = 0.20, 0.50
SWEEP = [0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


def load_ground_truth():
    import yaml

    seeds, verdicts = {}, {}
    for cap in glob.glob(str(AUDITION_DIR / "*_capture.yaml")):
        slug = Path(cap).stem.replace("_capture", "")
        man = AUDITION_DIR / f"{slug}_manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        if m.get("type") == "transition_pairs":
            continue
        seeds[slug] = m["seed"]["track_id"]
        data = yaml.safe_load(Path(cap).read_text(encoding="utf-8")) or {}
        for e in data.get("entries", []):
            if e.get("verdict") in VERDICT_SCORE:
                verdicts[(slug, e["track_id"])] = VERDICT_SCORE[e["verdict"]]
    return seeds, verdicts


def _l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _z(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def score(vectors, seeds, verdicts):
    from scipy.stats import spearmanr

    rhos = []
    for slug, seed_tid in seeds.items():
        if seed_tid not in vectors:
            continue
        sv = vectors[seed_tid]
        cos, sc = [], []
        for (s, tid), v in verdicts.items():
            if s != slug or tid == seed_tid or tid not in vectors:
                continue
            cos.append(float(vectors[tid] @ sv))
            sc.append(v)
        if len(cos) >= 5:
            r = spearmanr(cos, sc).correlation
            rhos.append(r if r == r else 0.0)
    return float(np.mean(rhos)) if rhos else 0.0


def main():
    from src.features.artifacts import load_artifact_bundle

    seeds, verdicts = load_ground_truth()
    hz = np.load(HARM_CACHE, allow_pickle=True)
    twodftm = {str(t): hz["twodftm"][i] for i, t in enumerate(hz["track_ids"])}

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    row = {t: i for i, t in enumerate(bundle.track_ids)}
    tids = [t for t in twodftm if t in row]
    idx = np.array([row[t] for t in tids])

    rhythm = _l2(npz["X_sonic_rhythm"][idx].astype(np.float64))
    timbre = _l2(npz["X_sonic_timbre"][idx].astype(np.float64))
    harm_cur = _l2(npz["X_sonic_harmony"][idx].astype(np.float64))
    harm_2df = _l2(_z(np.stack([twodftm[t] for t in tids]).astype(np.float64)))

    sr_, st_ = np.sqrt(W_RHYTHM), np.sqrt(W_TIMBRE)

    def blend_score(harm, wh):
        B = _l2(np.hstack([sr_ * rhythm, st_ * timbre, np.sqrt(wh) * harm]))
        return score({t: B[i] for i, t in enumerate(tids)}, seeds, verdicts)

    results = {"current": {}, "twodftm": {}}
    print(f"Harmony-weight sweep over {len(tids)} tracks "
          f"(rhythm=0.20, timbre=0.50 fixed)\n")
    print(f"{'w_harmony':>9} {'current':>9} {'2dftm':>9}")
    print("-" * 30)
    for wh in SWEEP:
        rc = blend_score(harm_cur, wh)
        r2 = blend_score(harm_2df, wh)
        results["current"][f"{wh:.2f}"] = rc
        results["twodftm"][f"{wh:.2f}"] = r2
        mark = "  <- production" if abs(wh - 0.30) < 1e-9 else ""
        print(f"{wh:>9.2f} {rc:>+9.3f} {r2:>+9.3f}{mark}")

    best_w = max(results["twodftm"], key=results["twodftm"].get)
    print(f"\n2DFTM blend peaks at w_harmony={best_w} "
          f"(rho={results['twodftm'][best_w]:+.3f})")
    print(f"vs current harmony at w=0.30 (rho={results['current']['0.30']:+.3f})")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
