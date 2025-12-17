#!/usr/bin/env python3
"""
Debug edge scores directly from DS artifact.
- Resolves indices by track_id
- Computes sonic/genre cosine and transition end->start using the same helpers as DS
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.artifacts import load_artifact_bundle, get_sonic_matrix
from src.similarity.hybrid import build_hybrid_embedding, transition_similarity_end_to_start


def cosine_row(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Debug DS artifact edge scores for a specific pair.")
    parser.add_argument("--artifact", default="experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz")
    parser.add_argument("--prev", required=True, help="Prev track_id")
    parser.add_argument("--cur", required=True, help="Cur track_id")
    parser.add_argument("--transition-gamma", type=float, default=1.0, help="Gamma used in constructor (1.0 = seg only)")
    parser.add_argument("--random-pairs", type=int, default=10, help="Number of random pairs to sample for ranges")
    parser.add_argument("--center-transitions", action="store_true", help="Center end/start before cosine and rescale to [0,1]")
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact)

    if args.prev not in bundle.track_id_to_index or args.cur not in bundle.track_id_to_index:
        missing = [tid for tid in [args.prev, args.cur] if tid not in bundle.track_id_to_index]
        raise SystemExit(f"Missing track_ids in artifact: {missing}")

    idx_prev = bundle.track_id_to_index[args.prev]
    idx_cur = bundle.track_id_to_index[args.cur]

    # Hybrid embedding (same as pipeline: smoothed genres)
    model = build_hybrid_embedding(bundle.X_sonic, bundle.X_genre_smoothed)
    emb = model.embedding
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    S_ds = float(emb_norm[idx_prev] @ emb_norm[idx_cur])

    # Genre cosine (smoothed)
    g_prev = bundle.X_genre_smoothed[idx_prev]
    g_cur = bundle.X_genre_smoothed[idx_cur]
    G_ds = cosine_row(g_prev, g_cur)

    # Transition raw (end -> start) and gamma-weighted used
    X_start_raw = get_sonic_matrix(bundle, "start")
    X_end_raw = get_sonic_matrix(bundle, "end")
    X_start = X_start_raw
    X_end = X_end_raw
    rescale = False
    if args.center_transitions and X_start is not None and X_end is not None:
        mu_end = X_end.mean(axis=0, keepdims=True)
        mu_start = X_start.mean(axis=0, keepdims=True)
        X_end = X_end - mu_end
        X_start = X_start - mu_start
        rescale = True

    if X_start_raw is None or X_end_raw is None:
        T_raw_uncentered = float(emb_norm[idx_cur] @ emb_norm[idx_prev])
    else:
        T_raw_uncentered = float(
            transition_similarity_end_to_start(
                X_end_raw, X_start_raw, idx_prev, np.array([idx_cur], dtype=int)
            )[0]
        )
    T_center_cos = None
    T_center_rescaled = None
    if rescale and X_start is not None and X_end is not None:
        T_center_cos = float(
            transition_similarity_end_to_start(
                X_end, X_start, idx_prev, np.array([idx_cur], dtype=int)
            )[0]
        )
        T_center_rescaled = float(np.clip((T_center_cos + 1.0) / 2.0, 0.0, 1.0))
        T_for_gamma = T_center_rescaled
    else:
        T_for_gamma = T_raw_uncentered

    T_ds_used = args.transition_gamma * T_for_gamma + (1 - args.transition_gamma) * float(
        emb_norm[idx_cur] @ emb_norm[idx_prev]
    )
    if rescale and T_ds_used == T_raw_uncentered:
        print("WARNING: center_transitions enabled but T_used equals uncentered raw; centering may be bypassed.")

    print(f"artifact: {Path(args.artifact).resolve()}")
    print(f"idx_prev={idx_prev}, idx_cur={idx_cur}")
    print(f"S_ds (hybrid cosine): {S_ds:.6f}")
    print(f"G_ds (genre smoothed cosine): {G_ds:.6f}")
    print(f"T_raw_uncentered (end->start cosine): {T_raw_uncentered:.6f}")
    if args.center_transitions and T_center_cos is not None:
        print(f"T_centered_cos (pre-rescale): {T_center_cos:.6f}")
        print(f"T_centered_rescaled: {T_center_rescaled:.6f}")
        print(f"T_used (centered,rescaled,gamma={args.transition_gamma}): {T_ds_used:.6f}")
    else:
        print(f"T_used (uncentered,gamma={args.transition_gamma}): {T_ds_used:.6f}")

    # Random sanity pairs
    rng = np.random.default_rng(0)
    n = emb_norm.shape[0]
    print("\nSample pairs (S_ds, T_used):")
    for _ in range(args.random_pairs):
        i, j = rng.integers(0, n, size=2)
        s_val = float(emb_norm[i] @ emb_norm[j])
        if X_start is not None and X_end is not None:
            t_val = float(
                transition_similarity_end_to_start(
                    X_end, X_start, i, np.array([j], dtype=int)
                )[0]
            )
            if rescale:
                t_val = float(np.clip((t_val + 1.0) / 2.0, 0.0, 1.0))
        else:
            t_val = float(emb_norm[j] @ emb_norm[i])
        t_used_pair = args.transition_gamma * t_val + (1 - args.transition_gamma) * float(emb_norm[j] @ emb_norm[i])
        print(f"{i}->{j}: S={s_val:.3f}  T_used={t_used_pair:.3f}")


if __name__ == "__main__":
    main()
