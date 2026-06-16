"""Beat-synchronous 2DFTM vs frame-level 2DFTM (the version the probe validated).

The production extractor already computes beat frames; the Bertin-Mahieux & Ellis
method is beat-synchronous chroma -> 2D-FFT magnitude. Our probe used frame-level
chroma resampled to a fixed length instead. This isolates the single variable:
chroma columns = BEATS (librosa.util.sync) vs resampled FRAMES. Beats are tracked
on the full mix; chroma is taken from the harmonic component (the correct split).

If beat-sync >= frame-level on the audition verdicts, the production rebuild should
use beat-sync (more faithful + tempo-normalized in musical units). If it's worse on
beatless material, we learn that too (fallback counts reported).

Read-only; extraction cached. Reuses the frame-level 2DFTM cache for comparison.

Usage:
    python scripts/sonic_beatsync_2dftm.py
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
FRAME_CACHE = PHASE2 / "richer_harmony_cache.npz"
BEAT_CACHE = PHASE2 / "richer_harmony_beatsync_cache.npz"
OUT = PHASE2 / "beatsync_2dftm.json"
SR = 22050
HOP = 512
TWODFTM_K = 8
T_FIXED = 256
W_RHYTHM, W_TIMBRE, W_HARMONY = 0.20, 0.50, 0.30
VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


def load_ground_truth():
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


def _twodftm(chroma_cols: np.ndarray) -> np.ndarray:
    if chroma_cols.shape[1] >= 2:
        idx = np.linspace(0, chroma_cols.shape[1] - 1, T_FIXED)
        rs = np.stack([np.interp(idx, np.arange(chroma_cols.shape[1]), r) for r in chroma_cols])
    else:
        rs = np.repeat(chroma_cols, T_FIXED, axis=1)[:, :T_FIXED]
    return np.abs(np.fft.fft2(rs))[:, :TWODFTM_K].flatten().astype(np.float32)


def extract_beatsync(path: str) -> dict | None:
    import librosa

    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if y.size < SR:
        return None
    y_h = librosa.effects.harmonic(y, margin=8)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)  # (12, T)
    _, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)     # beats on full mix
    if beats is not None and len(beats) >= 4:
        cs = librosa.util.sync(chroma, beats, aggregate=np.median)     # (12, n_beats)
        mode = "beats"
    else:
        cs = chroma
        mode = "frame_fallback"
    return {"beat_2dftm": _twodftm(cs), "mode": mode}


def build_cache(paths: dict) -> tuple[dict, dict]:
    cache, modes = {}, {}
    if BEAT_CACHE.exists():
        z = np.load(BEAT_CACHE, allow_pickle=True)
        for i, tid in enumerate(z["track_ids"]):
            cache[str(tid)] = z["beat_2dftm"][i]
            modes[str(tid)] = str(z["mode"][i])
        print(f"Loaded {len(cache)} cached beat-sync extractions.")
    todo = [(t, p) for t, p in paths.items()
            if p and Path(p).exists() and t not in cache]
    print(f"Extracting beat-sync 2DFTM for {len(todo)} tracks...")
    for k, (tid, p) in enumerate(todo, 1):
        r = extract_beatsync(p)
        if r is not None:
            cache[tid], modes[tid] = r["beat_2dftm"], r["mode"]
        if k % 20 == 0:
            print(f"  {k}/{len(todo)}")
            _save(cache, modes)
    _save(cache, modes)
    return cache, modes


def _save(cache, modes):
    PHASE2.mkdir(parents=True, exist_ok=True)
    tids = list(cache.keys())
    np.savez(BEAT_CACHE, track_ids=np.array(tids),
             beat_2dftm=np.stack([cache[t] for t in tids]),
             mode=np.array([modes[t] for t in tids]))


def _l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _z(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def score(vectors, seeds, verdicts):
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
            r = spearmanr(cos, sc).correlation
            per_seed[slug] = float(r) if r == r else 0.0
    return {"per_seed": per_seed,
            "mean_rho": float(np.mean(list(per_seed.values()))) if per_seed else 0.0}


def main():
    from src.features.artifacts import load_artifact_bundle

    seeds, paths, verdicts = load_ground_truth()
    print(f"{len(seeds)} seeds, {len(verdicts)} verdicts.\n")

    beat, modes = build_cache(paths)
    nb = sum(1 for m in modes.values() if m == "beats")
    print(f"\n{len(beat)} beat-sync extractions ({nb} beat-tracked, "
          f"{len(beat)-nb} frame-fallback).\n")

    fz = np.load(FRAME_CACHE, allow_pickle=True)
    frame = {str(t): fz["twodftm"][i] for i, t in enumerate(fz["track_ids"])}

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    npz = np.load(bundle.artifact_path, allow_pickle=True)
    row = {t: i for i, t in enumerate(bundle.track_ids)}

    tids = [t for t in beat if t in frame and t in row]
    idx = np.array([row[t] for t in tids])
    rhythm = _l2(npz["X_sonic_rhythm"][idx].astype(np.float64))
    timbre = _l2(npz["X_sonic_timbre"][idx].astype(np.float64))
    harm_cur = _l2(npz["X_sonic_harmony"][idx].astype(np.float64))
    h_frame = _l2(_z(np.stack([frame[t] for t in tids]).astype(np.float64)))
    h_beat = _l2(_z(np.stack([beat[t] for t in tids]).astype(np.float64)))

    def d(M):
        return {t: M[i] for i, t in enumerate(tids)}

    sr_, st_, sh_ = np.sqrt(W_RHYTHM), np.sqrt(W_TIMBRE), np.sqrt(W_HARMONY)

    def blend(h):
        return d(_l2(np.hstack([sr_ * rhythm, st_ * timbre, sh_ * h])))

    iso = {
        "harmony_current": score(d(harm_cur), seeds, verdicts),
        "harmony_frame_2dftm": score(d(h_frame), seeds, verdicts),
        "harmony_beat_2dftm": score(d(h_beat), seeds, verdicts),
    }
    bl = {
        "blend_current": score(blend(harm_cur), seeds, verdicts),
        "blend_frame_2dftm": score(blend(h_frame), seeds, verdicts),
        "blend_beat_2dftm": score(blend(h_beat), seeds, verdicts),
    }

    def show(title, res, order):
        print(f"\n{title}")
        print(f"{'representation':22} {'mean_rho':>9}   per-seed")
        print("-" * 84)
        for name in order:
            r = res[name]
            ps = "  ".join(f"{k[:8]}={v:+.2f}" for k, v in r["per_seed"].items())
            print(f"{name:22} {r['mean_rho']:>+9.3f}   {ps}")

    print(f"\n=== Beat-sync vs frame-level 2DFTM over {len(tids)} tracks ===")
    show("ISOLATED harmony:", iso,
         ["harmony_current", "harmony_frame_2dftm", "harmony_beat_2dftm"])
    show("BLEND:", bl, ["blend_current", "blend_frame_2dftm", "blend_beat_2dftm"])
    print(f"\nframe blend {bl['blend_frame_2dftm']['mean_rho']:+.3f}  vs  "
          f"beat blend {bl['blend_beat_2dftm']['mean_rho']:+.3f}  "
          f"(delta {bl['blend_beat_2dftm']['mean_rho']-bl['blend_frame_2dftm']['mean_rho']:+.3f})")

    OUT.write_text(json.dumps({"isolated": iso, "blend": bl,
                               "n_beat_tracked": nb, "n_total": len(beat)}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
