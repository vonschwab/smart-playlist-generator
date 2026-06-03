"""Structural diagnostic for the per-tower sonic spaces.

Answers: is the harmony tower's high-cosine behavior a HUB (dominant shared
direction / mean offset — cheaply fixable by centering or nulling top dirs)
or a LOW-DIMENSIONALITY artifact (too few dims, cosines inherently inflated —
needs more features, expensive)?

For each tower (rhythm/timbre/harmony) and the full 86-dim space, measured over
all tracks on L2-normalized rows (matching how the audition computes cosine):

  - hub_strength = ||mean(unit rows)||   (0 = no hub, 1 = total collapse)
  - anisotropy   = top-1 / top-3 fraction of variance + participation ratio
                   (effective dimensionality)
  - random-pair cosine mean/median/p90/p99, BEFORE and AFTER mean-centering
  - isotropic baseline 1/sqrt(d): expected cosine std if dims were isotropic

Read-only. No writes to metadata.db or audio. Output JSON + console summary.

Usage:
    python scripts/sonic_tower_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
OUT_DIR = ROOT / "docs/run_audits/sonic_phase2"
N_PAIRS = 400_000
SEED = 0


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def _pair_cosines(U: np.ndarray, rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    """Cosine of random distinct row pairs of an already-L2-normalized matrix."""
    n = U.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.einsum("ij,ij->i", U[i], U[j])


def _dist(c: np.ndarray) -> dict:
    return {
        "mean": float(c.mean()),
        "median": float(np.median(c)),
        "p90": float(np.percentile(c, 90)),
        "p99": float(np.percentile(c, 99)),
    }


def analyze_tower(name: str, X: np.ndarray, rng: np.random.Generator) -> dict:
    d = X.shape[1]
    U = _l2(X.astype(np.float64))

    # Hub: norm of the mean unit vector. Random unit vectors in d dims -> ~1/sqrt(N).
    mean_vec = U.mean(axis=0)
    hub_strength = float(np.linalg.norm(mean_vec))

    # Anisotropy: SVD of centered matrix -> variance per principal direction.
    Uc = U - mean_vec
    # economy SVD; singular values^2 ∝ variance
    sv = np.linalg.svd(Uc, full_matrices=False, compute_uv=False)
    var = sv**2
    var_frac = var / var.sum()
    participation_ratio = float((var.sum() ** 2) / (var**2).sum())  # effective dim

    # Cosine distributions: raw (uncentered) vs centered-and-renormalized.
    cos_raw = _pair_cosines(U, rng, N_PAIRS)
    Uc_n = _l2(Uc)
    cos_centered = _pair_cosines(Uc_n, rng, N_PAIRS)

    return {
        "dims": int(d),
        "isotropic_cos_std_baseline": float(1.0 / np.sqrt(d)),
        "hub_strength": hub_strength,
        "top1_var_frac": float(var_frac[0]),
        "top3_var_frac": float(var_frac[:3].sum()),
        "participation_ratio": participation_ratio,
        "cosine_raw": _dist(cos_raw),
        "cosine_centered": _dist(cos_centered),
    }


def main() -> None:
    from src.features.artifacts import load_artifact_bundle

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(ARTIFACT)
    npz = np.load(bundle.artifact_path, allow_pickle=True)

    towers = {
        "rhythm": npz["X_sonic_rhythm"],
        "timbre": npz["X_sonic_timbre"],
        "harmony": npz["X_sonic_harmony"],
        "full_86": npz["X_sonic"],
    }
    rng = np.random.default_rng(SEED)
    n_tracks = towers["full_86"].shape[0]
    print(f"Diagnostic over {n_tracks} tracks, {N_PAIRS} random pairs per cosine dist.\n")

    results = {}
    for name, X in towers.items():
        results[name] = analyze_tower(name, X, rng)

    # Console summary
    hdr = f"{'tower':9} {'d':>3} {'hub':>6} {'top1':>6} {'top3':>6} {'eff_d':>6} {'cos_raw_mean':>12} {'cos_raw_p99':>11} {'cos_cen_mean':>12} {'cos_cen_p99':>11} {'iso_std':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(
            f"{name:9} {r['dims']:>3} {r['hub_strength']:>6.3f} "
            f"{r['top1_var_frac']:>6.3f} {r['top3_var_frac']:>6.3f} "
            f"{r['participation_ratio']:>6.1f} "
            f"{r['cosine_raw']['mean']:>12.3f} {r['cosine_raw']['p99']:>11.3f} "
            f"{r['cosine_centered']['mean']:>12.3f} {r['cosine_centered']['p99']:>11.3f} "
            f"{r['isotropic_cos_std_baseline']:>8.3f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tower_diagnostic.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
