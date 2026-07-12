"""Re-score saved collapse runs in any sonic variant's geometry (mert | muq).

The collapse harness saves each playlist's interior track_ids. This recomputes the
cluster collapse_load from those SAME track_ids in each variant's space — so we can
ask whether collapse seen in one space (e.g. MuQ-generated playlists measured by the
validated MERT-CI) is real or a metric-space artifact, by judging the identical
playlists with both rulers. No regeneration => no nondeterminism in the comparison.

Self-check: the MERT column must reproduce the harness's reported MERT collapse_load.

  python scripts/research/collapse_rescore.py docs/run_audits/collapse/run_muq_gen_r1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.research.collapse_metric import centroid, cluster_ci, unit_rows  # noqa: E402

ROOT = Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
ARTIFACT = ROOT / "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
VARIANTS = ["mert", "muq"]


def load_art():
    a = np.load(ARTIFACT, allow_pickle=True)
    ids = [str(t) for t in a["track_ids"]]
    id2idx = {t: i for i, t in enumerate(ids)}
    artists = np.array([str(x).strip().lower() for x in a["track_artists"]])
    M = {v: np.asarray(a[f"X_sonic_{v}"], dtype=np.float32) for v in VARIANTS}
    return id2idx, artists, M


def _catalog_idx(seed, artists):
    sl = seed.strip().lower()
    return [i for i, a in enumerate(artists) if a == sl]


def global_blur(M):
    """Per-variant generic-average direction + anisotropy.

    blur = unit mean of all unit track vectors (the dominant 'average' direction the
    library leans toward). anisotropy = ||raw mean|| in [0,1]: ~0 means isotropic (no
    meaningful average direction — the global-centroid sag is degenerate there);
    larger means a real dense bulk the average points at.
    """
    out = {}
    for v, X in M.items():
        raw = unit_rows(X.astype(np.float64)).mean(axis=0)
        out[v] = {"center": raw / max(float(np.linalg.norm(raw)), 1e-12),
                  "anisotropy": float(np.linalg.norm(raw))}
    return out


def _raw_sag(V, blur_seed, seed_char):
    """Mean raw sag over rows V: cos(t, blur_seed) - cos(t, seed_character)."""
    return float(np.mean((V @ blur_seed) - (V @ seed_char))) if len(V) else 0.0


def sag_loads(run, variant, id2idx, artists, M, k_local=200):
    """CALIBRATED within-bridge sag per cluster.

    For each seed: seed_char = catalog centroid; blur_seed = centroid of its k nearest
    LIBRARY tracks (excl. own catalog) = the dense generic region it lives next to.
    Raw sag(t) = cos(t, blur_seed) - cos(t, seed_char). Two anchors fix the scale:
      floor   = mean raw sag over the seed's OWN tracks (they ARE the character ~ 0%)
      ceiling = mean raw sag over the blur neighbors      (the local generic ~ 100%)
    A playlist's interiors normalize to (raw - floor)/(ceiling - floor) = the fraction
    of the way from on-character to local-generic. Per cluster = mean over playlists.
    """
    An = unit_rows(M[variant].astype(np.float64))  # all rows unit -> An @ v = cosine
    out = {}
    for cname, cdata in run["clusters"].items():
        seeds = cdata["seeds"]
        sc_, bs_, floor_, ceil_ = {}, {}, {}, {}
        for s in seeds:
            cidx = sorted(set(_catalog_idx(s, artists)))
            sc = centroid(An[cidx]); sc_[s] = sc
            order = np.argsort(-(An @ sc))
            nn = [int(i) for i in order if int(i) not in set(cidx)][:k_local]
            bs = centroid(An[nn]); bs_[s] = bs
            floor_[s] = _raw_sag(An[cidx], bs, sc)   # on-character anchor (≈ negative)
            ceil_[s] = _raw_sag(An[nn], bs, sc)      # local-generic anchor (≈ positive)
        norms = []
        for rep in cdata["repeats"]:
            for s in seeds:
                idx = [id2idx[t] for t in rep["playlists"][s]["interior_ids"] if t in id2idx]
                if not idx:
                    continue
                raw = _raw_sag(An[idx], bs_[s], sc_[s])
                span = ceil_[s] - floor_[s]
                if abs(span) > 1e-9:
                    norms.append((raw - floor_[s]) / span)
        out[cname] = {
            "norm": float(np.mean(norms)) if norms else 0.0,
            "floor": float(np.mean([floor_[s] for s in seeds])),
            "ceil": float(np.mean([ceil_[s] for s in seeds])),
        }
    return out


def collapse_loads(run, variant, id2idx, artists, M):
    """Per-cluster {collapse_load mean, per-repeat loads, worst-pair} in `variant` space."""
    A = M[variant].astype(np.float64)
    out = {}
    for cname, cdata in run["clusters"].items():
        seeds = cdata["seeds"]
        seed_cent = {s: centroid(unit_rows(A[_catalog_idx(s, artists)])) for s in seeds}
        loads, worst = [], []
        for rep in cdata["repeats"]:
            interiors = {}
            for s in seeds:
                iid = rep["playlists"][s]["interior_ids"]
                idx = [id2idx[t] for t in iid if t in id2idx]
                interiors[s] = unit_rows(A[idx]) if idx else np.zeros((0, 1))
            ci = cluster_ci(interiors, seed_cent)
            cis = [float(p["ci"]) for p in ci["pairs"]]
            loads.append(float(np.mean([max(0.0, c) for c in cis])) if cis else 0.0)
            worst.append(max(cis) if cis else 0.0)
        out[cname] = {"load": float(np.mean(loads)), "worst": float(np.mean(worst)),
                      "per_repeat": [round(x, 4) for x in loads]}
    return out


def main():
    run_path = Path(sys.argv[1])
    full = run_path if run_path.is_absolute() else (ROOT / run_path)
    run = json.loads(full.read_text(encoding="utf-8"))
    id2idx, artists, M = load_art()
    blur = global_blur(M)
    res = {v: collapse_loads(run, v, id2idx, artists, M) for v in VARIANTS}
    sag = {v: sag_loads(run, v, id2idx, artists, M) for v in VARIANTS}
    clusters = list(run["clusters"].keys())

    print(f"\nRESCORE {run_path.name}  (tag={run.get('config_label')}, repeats={run.get('repeats')})")
    print(f"  global-centroid anisotropy: mert={blur['mert']['anisotropy']:.3f}  "
          f"muq={blur['muq']['anisotropy']:.3f}   (~0 => global blur direction is degenerate)")

    print("\n  FACE 1 — cross-seed convergence (CI = S_play - S_seed):")
    print(f"  {'cluster':16s} | {'MERT-CI load':>12s}  worst | {'MuQ-CI load':>11s}  worst")
    for c in clusters:
        m, q = res["mert"][c], res["muq"][c]
        print(f"  {c:16s} | {m['load']:>12.4f} {m['worst']:>+6.3f} | {q['load']:>11.4f} {q['worst']:>+6.3f}")

    print("\n  FACE 2 — within-bridge sag, CALIBRATED (0% = on seed character, 100% = local generic blur):")
    print(f"  {'cluster':16s} | {'MERT sag%':>9s} | {'MuQ sag%':>9s}  (anchors floor->ceil, MuQ)")
    for c in clusters:
        m, q = sag["mert"][c], sag["muq"][c]
        print(f"  {c:16s} | {m['norm']*100:>8.1f}% | {q['norm']*100:>8.1f}%  "
              f"({q['floor']:+.3f} -> {q['ceil']:+.3f})")
    print("\n  (CI MERT column should match the harness's reported collapse_load — self-check)")


if __name__ == "__main__":
    main()
