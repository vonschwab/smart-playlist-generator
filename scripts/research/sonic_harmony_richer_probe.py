"""Richer-harmony probe against audition ground truth (Phase 2).

Tests whether research-grounded harmony representations separate the auditioned
verdicts (match/close/off/wrong) better than the shipped 20-dim harmony tower.

Representations (all read-only extraction from audio; metadata.db untouched):
  - current     : shipped X_sonic_harmony (20-dim tower)            [baseline]
  - cens         : librosa chroma_cens mean+std on harmonic signal   (24)
  - chroma_cov  : 12x12 chroma correlation upper-tri (chord co-occur)(66)
  - twodftm     : 2D-FFT magnitude of beat-chroma  (KEY-INVARIANT)   (12*K)
  - tonnetz     : tonnetz mean+std                                   (12)
  - richer_all  : concat(cens, chroma_cov, twodftm, tonnetz)

Metric: per seed, Spearman(cosine_to_seed, verdict_score) over its rated
neighbors; reported per representation, averaged across the 5 rated seeds.
Higher = the representation ranks perceptually-similar tracks closer.

Extraction is cached to a sidecar npz so re-runs are instant.

Usage:
    python scripts/sonic_harmony_richer_probe.py
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
OUT_DIR = ROOT / "docs/run_audits/sonic_phase2"
CACHE = OUT_DIR / "richer_harmony_cache.npz"
SR = 22050
HOP = 512
TWODFTM_K = 8          # time-freq bins kept from the 2D-FFT magnitude
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_ground_truth() -> tuple[dict, dict, dict]:
    """Return (seeds: slug->tid, paths: tid->path, verdicts: (slug,tid)->score)."""
    import yaml

    seeds, paths, verdicts = {}, {}, {}
    for cap in glob.glob(str(AUDITION_DIR / "*_capture.yaml")):
        slug = Path(cap).stem.replace("_capture", "")
        man = AUDITION_DIR / f"{slug}_manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        if m.get("type") == "transition_pairs":
            continue
        seeds[slug] = m["seed"]["track_id"]
        paths[m["seed"]["track_id"]] = m["seed"].get("file_path")
        for n in m.get("neighbors", []):
            paths[n["track_id"]] = n.get("file_path")
        data = yaml.safe_load(Path(cap).read_text(encoding="utf-8")) or {}
        for e in data.get("entries", []):
            v = e.get("verdict", "")
            if v in VERDICT_SCORE:
                verdicts[(slug, e["track_id"])] = VERDICT_SCORE[v]
    return seeds, paths, verdicts


# --------------------------------------------------------------------------- #
# Richer harmony extraction
# --------------------------------------------------------------------------- #
def _upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def extract_richer(path: str) -> dict | None:
    """Extract richer harmony blocks from one audio file. Read-only."""
    import librosa

    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if y.size < SR:  # < 1s
        return None

    # Clean: isolate harmonic component (strip percussion/transients).
    y_h = librosa.effects.harmonic(y, margin=8)

    # CENS — dynamics/timbre-invariant chroma for matching.
    cens = librosa.feature.chroma_cens(y=y_h, sr=SR, hop_length=HOP)  # (12, T)
    cens_feat = np.concatenate([cens.mean(axis=1), cens.std(axis=1)])  # (24,)

    # chroma_cqt correlation — chord/pitch-class co-occurrence structure.
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)  # (12, T)
    cc = np.corrcoef(chroma)  # (12,12)
    cc = np.nan_to_num(cc, nan=0.0)
    chroma_cov = _upper_tri(cc)  # (66,)

    # 2D-FFT magnitude of chroma — transposition (key) invariant fingerprint.
    # Resample time axis to a fixed length so the descriptor is length-stable.
    T_fixed = 256
    if chroma.shape[1] >= 2:
        idx = np.linspace(0, chroma.shape[1] - 1, T_fixed)
        chroma_rs = np.stack([np.interp(idx, np.arange(chroma.shape[1]), row) for row in chroma])
    else:
        chroma_rs = np.repeat(chroma, T_fixed, axis=1)[:, :T_fixed]
    F = np.abs(np.fft.fft2(chroma_rs))  # (12, 256); |.| discards transposition phase
    twodftm = F[:, :TWODFTM_K].flatten()  # (12*K,)

    # Tonnetz — tonal centroid (full stats).
    ton = librosa.feature.tonnetz(y=y_h, sr=SR)  # (6, T)
    tonnetz_feat = np.concatenate([ton.mean(axis=1), ton.std(axis=1)])  # (12,)

    return {
        "cens": cens_feat.astype(np.float32),
        "chroma_cov": chroma_cov.astype(np.float32),
        "twodftm": twodftm.astype(np.float32),
        "tonnetz": tonnetz_feat.astype(np.float32),
    }


def build_cache(paths: dict) -> dict:
    """Extract (with caching) richer features for all tracks that have a file."""
    blocks = ["cens", "chroma_cov", "twodftm", "tonnetz"]
    cache: dict[str, dict] = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        tids = list(z["track_ids"])
        for i, tid in enumerate(tids):
            cache[str(tid)] = {b: z[b][i] for b in blocks}
        print(f"Loaded {len(cache)} cached extractions.")

    todo = [
        (tid, p)
        for tid, p in paths.items()
        if p and Path(p).exists() and tid not in cache
    ]
    print(f"Extracting richer harmony for {len(todo)} new tracks...")
    for k, (tid, p) in enumerate(todo, 1):
        feats = extract_richer(p)
        if feats is not None:
            cache[tid] = feats
        if k % 20 == 0:
            print(f"  {k}/{len(todo)}")
            _save_cache(cache, blocks)
    _save_cache(cache, blocks)
    return cache


def _save_cache(cache: dict, blocks: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tids = list(cache.keys())
    arrays = {b: np.stack([cache[t][b] for t in tids]) for b in blocks}
    np.savez(CACHE, track_ids=np.array(tids), **arrays)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _l2(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def build_representations(cache: dict, bundle) -> dict[str, dict]:
    """track_id -> vector, per representation. 'current' from the shipped tower."""
    tid_to_row = {t: i for i, t in enumerate(bundle.track_ids)}
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    harm = npz["X_sonic_harmony"]

    reps: dict[str, dict] = {
        "current": {}, "cens": {}, "chroma_cov": {},
        "twodftm": {}, "tonnetz": {}, "richer_all": {},
    }
    for tid, feats in cache.items():
        reps["cens"][tid] = feats["cens"]
        reps["chroma_cov"][tid] = feats["chroma_cov"]
        reps["twodftm"][tid] = feats["twodftm"]
        reps["tonnetz"][tid] = feats["tonnetz"]
        reps["richer_all"][tid] = np.concatenate(
            [feats["cens"], feats["chroma_cov"], feats["twodftm"], feats["tonnetz"]]
        )
        if tid in tid_to_row:
            reps["current"][tid] = harm[tid_to_row[tid]].astype(np.float32)
    return reps


def zscore_block(reps_block: dict) -> dict:
    """Z-score each dim across the pool so cosine treats dims comparably."""
    tids = list(reps_block.keys())
    X = np.stack([reps_block[t] for t in tids]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xz = (X - mu) / sd
    return {t: Xz[i] for i, t in enumerate(tids)}


def evaluate(reps: dict, seeds: dict, verdicts: dict) -> dict:
    from scipy.stats import spearmanr

    results: dict[str, dict] = {}
    for rep_name, block in reps.items():
        zblock = zscore_block(block)
        per_seed = {}
        for slug, seed_tid in seeds.items():
            if seed_tid not in zblock:
                continue
            sv = _l2(zblock[seed_tid])
            cos, score = [], []
            for (s, tid), vscore in verdicts.items():
                if s != slug or tid == seed_tid or tid not in zblock:
                    continue
                cos.append(float(_l2(zblock[tid]) @ sv))
                score.append(vscore)
            if len(cos) >= 5:
                rho = spearmanr(cos, score).correlation
                per_seed[slug] = float(rho) if rho == rho else 0.0
        if per_seed:
            results[rep_name] = {
                "per_seed": per_seed,
                "mean_rho": float(np.mean(list(per_seed.values()))),
                "n_seeds": len(per_seed),
            }
    return results


def main() -> None:
    from src.features.artifacts import load_artifact_bundle

    seeds, paths, verdicts = load_ground_truth()
    print(f"{len(seeds)} seeds, {len(verdicts)} verdicts, {len(paths)} tracks with paths.\n")

    cache = build_cache(paths)
    print(f"\n{len(cache)} tracks have richer features.\n")

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    reps = build_representations(cache, bundle)
    results = evaluate(reps, seeds, verdicts)

    order = ["current", "cens", "chroma_cov", "twodftm", "tonnetz", "richer_all"]
    print(f"{'representation':14} {'mean_rho':>9}   per-seed Spearman(cosine, verdict)")
    print("-" * 78)
    for name in order:
        if name not in results:
            continue
        r = results[name]
        ps = "  ".join(f"{k[:8]}={v:+.2f}" for k, v in r["per_seed"].items())
        flag = "  <- baseline" if name == "current" else ""
        print(f"{name:14} {r['mean_rho']:>+9.3f}   {ps}{flag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "richer_harmony_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
