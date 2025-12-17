"""
Artifact similarity diagnostics.

Usage (PowerShell):
  python scripts/diagnose_artifact_similarity.py --artifact path\\to\\artifact.npz --n 20000 --seed 1
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path

# Ensure repository root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle
from src.similarity.hybrid import build_hybrid_embedding, transition_similarity_end_to_start


def l2norm_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def summary(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: empty")
        return
    arr_flat = arr.reshape(-1)
    print(
        f"{name:20s} mean={np.mean(arr_flat):.4f} p01={np.percentile(arr_flat,1):.4f} "
        f"p10={np.percentile(arr_flat,10):.4f} p50={np.percentile(arr_flat,50):.4f} "
        f"p90={np.percentile(arr_flat,90):.4f} p99={np.percentile(arr_flat,99):.4f} "
        f"min={arr_flat.min():.4f} max={arr_flat.max():.4f}"
    )


def cosine_pairs(a_norm: np.ndarray, b_norm: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    return np.sum(a_norm[idx_a] * b_norm[idx_b], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose artifact similarity distributions.")
    parser.add_argument("--artifact", required=True, help="Path to artifact npz")
    parser.add_argument("--n", type=int, default=20000, help="Number of random pairs to sample (default 20000)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--center-transitions",
        action="store_true",
        help="Also compute centered end->start cosine and rescaled [0,1] variant",
    )
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact)
    rng = np.random.default_rng(args.seed)
    N = int(bundle.track_ids.shape[0])
    pair_count = min(args.n, max(N * 2, args.n))

    def sample_pairs(count: int) -> tuple[np.ndarray, np.ndarray]:
        ia = rng.integers(0, N, size=count, dtype=int)
        ib = rng.integers(0, N, size=count, dtype=int)
        # avoid identical pairs
        mask = ia == ib
        if mask.any():
            ib[mask] = (ib[mask] + 1) % N
        return ia, ib

    print(f"Loaded artifact: {args.artifact}")
    print(f"Tracks: {N}")

    X_end = getattr(bundle, "X_sonic_end", None)
    X_start = getattr(bundle, "X_sonic_start", None)
    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)

    if X_end is None or X_start is None:
        print("Warning: X_sonic_end/start missing; T_raw/T_centered will fall back to X_sonic.")
        X_end = X_sonic
        X_start = X_sonic

    # Row norm stats (pre-normalization)
    for name, mat in (
        ("X_end", X_end),
        ("X_start", X_start),
        ("X_sonic", X_sonic),
        ("X_genre_smoothed", X_genre),
    ):
        if mat is None:
            print(f"{name}: missing")
            continue
        norms = np.linalg.norm(mat, axis=1)
        summary(f"{name} row-norm", norms)

    # Per-dimension std for start/end
    for name, mat in (("X_end", X_end), ("X_start", X_start)):
        if mat is None:
            continue
        stds = np.std(mat, axis=0)
        print(
            f"{name} per-dim std: min={stds.min():.6f} median={np.median(stds):.6f} max={stds.max():.6f}"
        )

    # Mean-direction alignment
    def mean_dir_cos(name: str, mat: np.ndarray) -> None:
        if mat is None:
            return
        normed = l2norm_rows(mat)
        mean_vec = normed.mean(axis=0)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-12
        cos_vals = normed @ mean_vec
        summary(f"{name} cos to mean", cos_vals)

    mean_dir_cos("X_end", X_end)
    mean_dir_cos("X_start", X_start)

    # Prepare normalized matrices
    end_norm = l2norm_rows(X_end) if X_end is not None else None
    start_norm = l2norm_rows(X_start) if X_start is not None else None
    sonic_norm = l2norm_rows(X_sonic) if X_sonic is not None else None
    genre_norm = l2norm_rows(X_genre) if X_genre is not None else None

    # Hybrid embedding (matches DS pipeline defaults)
    emb_norm = None
    try:
        if X_sonic is not None and X_genre is not None:
            emb_model = build_hybrid_embedding(
                X_sonic,
                X_genre,
                n_components_sonic=32,
                n_components_genre=32,
                w_sonic=1.0,
                w_genre=1.0,
                random_seed=args.seed,
            )
            emb_norm = l2norm_rows(emb_model.embedding)
            summary("Hybrid row-norm", np.linalg.norm(emb_model.embedding, axis=1))
        else:
            print("Hybrid embedding skipped (missing X_sonic or X_genre_smoothed).")
    except Exception as exc:
        print(f"Hybrid embedding failed: {exc}")

    ia, ib = sample_pairs(pair_count)

    if end_norm is not None and start_norm is not None:
        t_raw = cosine_pairs(end_norm, start_norm, ia, ib)
        summary("T_raw end->start", t_raw)
        if args.center_transitions:
            # Center before L2 to probe mean-direction dominance
            end_center = l2norm_rows(X_end - X_end.mean(axis=0, keepdims=True))
            start_center = l2norm_rows(X_start - X_start.mean(axis=0, keepdims=True))
            t_centered = cosine_pairs(end_center, start_center, ia, ib)
            summary("T_centered", t_centered)
            t_centered_rescaled = np.clip((t_centered + 1.0) / 2.0, 0.0, 1.0)
            summary("T_centered_rescaled", t_centered_rescaled)
    if emb_norm is not None:
        s_hybrid = cosine_pairs(emb_norm, emb_norm, ia, ib)
        summary("S_hybrid", s_hybrid)
    if sonic_norm is not None:
        s_sonic = cosine_pairs(sonic_norm, sonic_norm, ia, ib)
        summary("S_sonic", s_sonic)
    if genre_norm is not None:
        g_vals = cosine_pairs(genre_norm, genre_norm, ia, ib)
        summary("G_genre", g_vals)


if __name__ == "__main__":
    main()
