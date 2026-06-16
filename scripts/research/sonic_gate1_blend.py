"""Gate 1: does key-invariant (2DFTM) harmony improve the BLEND, not just the
isolated harmony tower?

The probe showed 2DFTM fixes harmony in isolation (-0.144 -> +0.210). But the
production blend (rhythm 0.20 / timbre 0.50 / harmony 0.30) already worked
because timbre masks harmony. This gate reconstructs the tower-weighted blend
for the rated tracks two ways — current 20-dim harmony vs 2DFTM harmony — and
scores each against the audition verdicts. If the 2DFTM blend beats the current
blend, the 40k re-extraction is justified; if it's a wash, timbre was already
covering harmony and the rebuild isn't worth it.

Each tower contributes a unit vector scaled by sqrt(weight); swapping the
harmony block is the only change. 2DFTM is z-scored over the pool before L2 so
its raw, wildly-scaled FFT dims form a well-conditioned unit block (mirroring
what whitening already does for the shipped 20-dim tower).

Read-only. Uses the probe cache + the shipped artifact. No extraction.

Usage:
    python scripts/sonic_gate1_blend.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
AUDITION_DIR = ROOT / "docs/run_audits/sonic_audition"
CACHE = ROOT / "docs/run_audits/sonic_phase2/richer_harmony_cache.npz"
OUT = ROOT / "docs/run_audits/sonic_phase2/gate1_blend.json"

# Production tower weights (CLAUDE.md principle #17). Blend = concat(sqrt(w)*l2(tower)).
W_RHYTHM, W_TIMBRE, W_HARMONY = 0.20, 0.50, 0.30
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


def load_ground_truth() -> tuple[dict, dict]:
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


def _l2_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def score_blend(vectors: dict, seeds: dict, verdicts: dict) -> dict:
    """Spearman(seed->track cosine, verdict) per seed; vectors are pre-L2'd rows."""
    from scipy.stats import spearmanr

    per_seed = {}
    for slug, seed_tid in seeds.items():
        if seed_tid not in vectors:
            continue
        sv = vectors[seed_tid]
        cos, score = [], []
        for (s, tid), v in verdicts.items():
            if s != slug or tid == seed_tid or tid not in vectors:
                continue
            cos.append(float(vectors[tid] @ sv))
            score.append(v)
        if len(cos) >= 5:
            rho = spearmanr(cos, score).correlation
            per_seed[slug] = float(rho) if rho == rho else 0.0
    return {
        "per_seed": per_seed,
        "mean_rho": float(np.mean(list(per_seed.values()))) if per_seed else 0.0,
    }


def main() -> None:
    from src.features.artifacts import load_artifact_bundle

    seeds, verdicts = load_ground_truth()

    z = np.load(CACHE, allow_pickle=True)
    cache_tids = [str(t) for t in z["track_ids"]]
    twodftm = {t: z["twodftm"][i] for i, t in enumerate(cache_tids)}

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    row = {t: i for i, t in enumerate(bundle.track_ids)}

    # Tracks present in both the cache and the artifact.
    tids = [t for t in cache_tids if t in row]
    idx = np.array([row[t] for t in tids])

    rhythm = _l2_rows(npz["X_sonic_rhythm"][idx].astype(np.float64))
    timbre = _l2_rows(npz["X_sonic_timbre"][idx].astype(np.float64))
    harm_cur = _l2_rows(npz["X_sonic_harmony"][idx].astype(np.float64))
    harm_2df = _l2_rows(_zscore_rows(np.stack([twodftm[t] for t in tids]).astype(np.float64)))
    full_shipped = _l2_rows(npz["X_sonic"][idx].astype(np.float64))

    sr, st, sh = np.sqrt(W_RHYTHM), np.sqrt(W_TIMBRE), np.sqrt(W_HARMONY)
    blend_cur = _l2_rows(np.hstack([sr * rhythm, st * timbre, sh * harm_cur]))
    blend_2df = _l2_rows(np.hstack([sr * rhythm, st * timbre, sh * harm_2df]))

    def as_dict(M):
        return {t: M[i] for i, t in enumerate(tids)}

    reps = {
        "harmony_only_current": as_dict(harm_cur),
        "harmony_only_2dftm": as_dict(harm_2df),
        "blend_shipped_Xsonic": as_dict(full_shipped),
        "blend_current": as_dict(blend_cur),
        "blend_2dftm": as_dict(blend_2df),
    }
    results = {name: score_blend(v, seeds, verdicts) for name, v in reps.items()}

    print(f"Gate 1 — blend test over {len(tids)} tracks, {len(seeds)} seeds\n")
    order = [
        "harmony_only_current", "harmony_only_2dftm",
        "blend_shipped_Xsonic", "blend_current", "blend_2dftm",
    ]
    print(f"{'representation':24} {'mean_rho':>9}   per-seed")
    print("-" * 84)
    for name in order:
        r = results[name]
        ps = "  ".join(f"{k[:8]}={v:+.2f}" for k, v in r["per_seed"].items())
        print(f"{name:24} {r['mean_rho']:>+9.3f}   {ps}")

    delta = results["blend_2dftm"]["mean_rho"] - results["blend_current"]["mean_rho"]
    print(f"\nBlend delta (2dftm - current): {delta:+.3f}")
    print("GATE 1 PASS — rebuild justified." if delta > 0.02
          else "GATE 1 INCONCLUSIVE/FAIL — timbre already covers harmony.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
