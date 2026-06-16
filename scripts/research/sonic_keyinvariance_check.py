"""Sanity check: is 2DFTM actually key-invariant on OUR code?

Mechanism claim: transposing a track should leave the 2DFTM harmony vector
nearly unchanged (transposition -> circular shift on the pitch axis -> phase ->
discarded by |FFT|), while the current chroma-based features should move a lot.

We pitch-shift each seed by N semitones and measure cosine(shifted, original)
for each representation. If the claim holds: 2DFTM ~1.0 across shifts;
chroma_median (current-style) and CENS drop. Read-only; no caching needed.

Usage:
    python scripts/sonic_keyinvariance_check.py
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

AUDITION_DIR = ROOT / "docs/run_audits/sonic_audition"
SR = 22050
HOP = 512
TWODFTM_K = 8
SHIFTS = [0, 1, 2, 4, 7]  # semitones


def seed_paths(n: int = 4) -> list[tuple[str, str]]:
    out = []
    for man in sorted(glob.glob(str(AUDITION_DIR / "*_manifest.json"))):
        m = json.loads(Path(man).read_text())
        if m.get("type") == "transition_pairs":
            continue
        s = m["seed"]
        p = s.get("file_path")
        if p and Path(p).exists():
            out.append((s.get("artist", "?"), p))
        if len(out) >= n:
            break
    return out


def reps(y_h, librosa) -> dict:
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)
    chroma_median = np.median(chroma, axis=1)  # current-tower-style (key-sensitive)
    cens = librosa.feature.chroma_cens(y=y_h, sr=SR, hop_length=HOP).mean(axis=1)

    T = 256
    idx = np.linspace(0, chroma.shape[1] - 1, T)
    crs = np.stack([np.interp(idx, np.arange(chroma.shape[1]), r) for r in chroma])
    twodftm = np.abs(np.fft.fft2(crs))[:, :TWODFTM_K].flatten()
    return {"chroma_median": chroma_median, "cens": cens, "twodftm": twodftm}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    import librosa

    tracks = seed_paths(4)
    print(f"Key-invariance check on {len(tracks)} seeds, shifts {SHIFTS} semitones.")
    print("Cosine(shifted vs original) — higher = more transposition-invariant.\n")

    agg = {r: {s: [] for s in SHIFTS} for r in ["chroma_median", "cens", "twodftm"]}
    for artist, path in tracks:
        y, _ = librosa.load(path, sr=SR, mono=True)
        y = y[: SR * 60]  # first 60s is plenty and keeps pitch_shift fast
        base = {s: None for s in SHIFTS}
        ref = reps(librosa.effects.harmonic(y, margin=8), librosa)
        print(f"{artist}")
        print(f"  {'shift':>5}  {'chroma_median':>13}  {'cens':>8}  {'twodftm':>8}")
        for s in SHIFTS:
            ys = y if s == 0 else librosa.effects.pitch_shift(y, sr=SR, n_steps=s)
            r = reps(librosa.effects.harmonic(ys, margin=8), librosa)
            cm = cos(r["chroma_median"], ref["chroma_median"])
            ce = cos(r["cens"], ref["cens"])
            td = cos(r["twodftm"], ref["twodftm"])
            for name, v in [("chroma_median", cm), ("cens", ce), ("twodftm", td)]:
                agg[name][s].append(v)
            print(f"  {s:>5}  {cm:>13.3f}  {ce:>8.3f}  {td:>8.3f}")
        print()

    print("MEAN across seeds:")
    print(f"  {'shift':>5}  {'chroma_median':>13}  {'cens':>8}  {'twodftm':>8}")
    for s in SHIFTS:
        print(f"  {s:>5}  {np.mean(agg['chroma_median'][s]):>13.3f}  "
              f"{np.mean(agg['cens'][s]):>8.3f}  {np.mean(agg['twodftm'][s]):>8.3f}")


if __name__ == "__main__":
    main()
