"""Gate 2: does richer, character-based rhythm help — especially the seeds the
harmony fix couldn't move (Duster: slow-ambient vs slowcore)?

Same methodology as the harmony probe. Current rhythm tower is absolute-tempo
coded (BPM, tempo lags); the research says the perceptual quantities are PULSE
CLARITY (is there a beat at all — librosa.beat.plp) and TEMPO-INVARIANT rhythmic
pattern (subdivision/swing — librosa.feature.tempogram_ratio). We extract those
for the rated tracks, score them against the audition verdicts in isolation,
then run the blend test for the combined fix (rich rhythm + 2DFTM harmony).

Extraction is read-only and cached. Reuses the harmony probe's 2DFTM cache for
the combined-blend test.

Usage:
    python scripts/sonic_gate2_rhythm.py
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
PHASE2 = ROOT / "docs/run_audits/sonic_phase2"
HARM_CACHE = PHASE2 / "richer_harmony_cache.npz"
RHY_CACHE = PHASE2 / "richer_rhythm_cache.npz"
OUT = PHASE2 / "gate2_rhythm.json"
SR = 22050
HOP = 512
W_RHYTHM, W_TIMBRE, W_HARMONY = 0.20, 0.50, 0.30
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_ground_truth() -> tuple[dict, dict, dict]:
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
            if e.get("verdict") in VERDICT_SCORE:
                verdicts[(slug, e["track_id"])] = VERDICT_SCORE[e["verdict"]]
    return seeds, paths, verdicts


# --------------------------------------------------------------------------- #
# Richer rhythm extraction
# --------------------------------------------------------------------------- #
def extract_rhythm(path: str) -> dict | None:
    import librosa

    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if y.size < SR:
        return None

    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    if onset_env.size < 8 or not np.any(onset_env > 0):
        return None

    # Pulse clarity (PLP): is there a clear beat? Stats of the pulse curve +
    # its periodicity strength (peak of normalized autocorrelation).
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    pac = librosa.autocorrelate(pulse)
    pac0 = pac[0] if pac[0] > 1e-9 else 1.0
    periodicity = float(np.max(pac[4:]) / pac0) if pac.size > 4 else 0.0
    pulse_feat = np.array([
        float(pulse.mean()), float(pulse.std()), float(pulse.max()),
        float(np.percentile(pulse, 90) - np.percentile(pulse, 10)),
        periodicity,
    ], dtype=np.float32)

    # Tempo-invariant rhythmic pattern: energy at metrical ratios of the tempo.
    tr = librosa.feature.tempogram_ratio(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    tempo_ratio = np.concatenate([tr.mean(axis=1), tr.std(axis=1)]).astype(np.float32)

    return {"pulse": pulse_feat, "tempo_ratio": tempo_ratio}


def build_cache(paths: dict) -> dict:
    blocks = ["pulse", "tempo_ratio"]
    cache: dict[str, dict] = {}
    if RHY_CACHE.exists():
        z = np.load(RHY_CACHE, allow_pickle=True)
        for i, tid in enumerate(z["track_ids"]):
            cache[str(tid)] = {b: z[b][i] for b in blocks}
        print(f"Loaded {len(cache)} cached rhythm extractions.")

    todo = [(t, p) for t, p in paths.items()
            if p and Path(p).exists() and t not in cache]
    print(f"Extracting richer rhythm for {len(todo)} tracks...")
    for k, (tid, p) in enumerate(todo, 1):
        feats = extract_rhythm(p)
        if feats is not None:
            cache[tid] = feats
        if k % 20 == 0:
            print(f"  {k}/{len(todo)}")
            _save(cache, blocks)
    _save(cache, blocks)
    return cache


def _save(cache: dict, blocks: list[str]) -> None:
    PHASE2.mkdir(parents=True, exist_ok=True)
    tids = list(cache.keys())
    arrays = {b: np.stack([cache[t][b] for t in tids]) for b in blocks}
    np.savez(RHY_CACHE, track_ids=np.array(tids), **arrays)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _l2_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def score(vectors: dict, seeds: dict, verdicts: dict) -> dict:
    from scipy.stats import spearmanr

    per_seed = {}
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
            rho = spearmanr(cos, sc).correlation
            per_seed[slug] = float(rho) if rho == rho else 0.0
    return {"per_seed": per_seed,
            "mean_rho": float(np.mean(list(per_seed.values()))) if per_seed else 0.0}


def main() -> None:
    from src.features.artifacts import load_artifact_bundle

    seeds, paths, verdicts = load_ground_truth()
    print(f"{len(seeds)} seeds, {len(verdicts)} verdicts.\n")

    rcache = build_cache(paths)
    print(f"\n{len(rcache)} tracks have richer rhythm.\n")

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    row = {t: i for i, t in enumerate(bundle.track_ids)}

    hz = np.load(HARM_CACHE, allow_pickle=True)
    twodftm = {str(t): hz["twodftm"][i] for i, t in enumerate(hz["track_ids"])}

    # tracks present everywhere
    tids = [t for t in rcache if t in row and t in twodftm]
    idx = np.array([row[t] for t in tids])

    rhythm_cur = _l2_rows(npz["X_sonic_rhythm"][idx].astype(np.float64))
    timbre = _l2_rows(npz["X_sonic_timbre"][idx].astype(np.float64))
    harm_cur = _l2_rows(npz["X_sonic_harmony"][idx].astype(np.float64))
    harm_2df = _l2_rows(_zscore_rows(np.stack([twodftm[t] for t in tids]).astype(np.float64)))

    pulse = np.stack([rcache[t]["pulse"] for t in tids]).astype(np.float64)
    tratio = np.stack([rcache[t]["tempo_ratio"] for t in tids]).astype(np.float64)
    rhythm_rich = _l2_rows(_zscore_rows(np.hstack([pulse, tratio])))
    pulse_only = _l2_rows(_zscore_rows(pulse))
    tratio_only = _l2_rows(_zscore_rows(tratio))

    def d(M):
        return {t: M[i] for i, t in enumerate(tids)}

    # ---- isolated rhythm probe ----
    iso = {
        "rhythm_current": d(rhythm_cur),
        "pulse_only": d(pulse_only),
        "tempo_ratio_only": d(tratio_only),
        "rhythm_rich": d(rhythm_rich),
    }
    iso_res = {k: score(v, seeds, verdicts) for k, v in iso.items()}

    # ---- blend test ----
    sr_, st_, sh_ = np.sqrt(W_RHYTHM), np.sqrt(W_TIMBRE), np.sqrt(W_HARMONY)

    def blend(rhy, harm):
        return _l2_rows(np.hstack([sr_ * rhy, st_ * timbre, sh_ * harm]))

    blends = {
        "blend_current": d(blend(rhythm_cur, harm_cur)),
        "blend_2dftm_harmony": d(blend(rhythm_cur, harm_2df)),
        "blend_rich_rhythm": d(blend(rhythm_rich, harm_cur)),
        "blend_both": d(blend(rhythm_rich, harm_2df)),
    }
    blend_res = {k: score(v, seeds, verdicts) for k, v in blends.items()}

    def show(title, res, order):
        print(f"\n{title}")
        print(f"{'representation':22} {'mean_rho':>9}   per-seed")
        print("-" * 84)
        for name in order:
            r = res[name]
            ps = "  ".join(f"{k[:8]}={v:+.2f}" for k, v in r["per_seed"].items())
            print(f"{name:22} {r['mean_rho']:>+9.3f}   {ps}")

    print(f"\n=== Gate 2 over {len(tids)} tracks ===")
    show("ISOLATED rhythm (expect low — slow != genre):", iso_res,
         ["rhythm_current", "pulse_only", "tempo_ratio_only", "rhythm_rich"])
    show("BLEND test:", blend_res,
         ["blend_current", "blend_2dftm_harmony", "blend_rich_rhythm", "blend_both"])

    base = blend_res["blend_current"]["mean_rho"]
    print(f"\nblend_current        {base:+.3f}  (baseline)")
    print(f"blend_2dftm_harmony  {blend_res['blend_2dftm_harmony']['mean_rho']:+.3f}  "
          f"({blend_res['blend_2dftm_harmony']['mean_rho']-base:+.3f})")
    print(f"blend_rich_rhythm    {blend_res['blend_rich_rhythm']['mean_rho']:+.3f}  "
          f"({blend_res['blend_rich_rhythm']['mean_rho']-base:+.3f})")
    print(f"blend_both           {blend_res['blend_both']['mean_rho']:+.3f}  "
          f"({blend_res['blend_both']['mean_rho']-base:+.3f})")
    dd = blend_res["blend_current"]["per_seed"].get("duster", 0)
    db = blend_res["blend_both"]["per_seed"].get("duster", 0)
    print(f"\nDuster (the rhythm-shaped residual): {dd:+.2f} -> {db:+.2f}")

    OUT.write_text(json.dumps({"isolated": iso_res, "blend": blend_res}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
