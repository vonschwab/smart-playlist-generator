"""
Wrap co-occurrence Jaccard similarity into a final genre similarity matrix.

Loads genre_cooc_matrix.npz (genres, sim_cooc) and writes a final similarity
matrix with optional min_sim thresholding.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build final genre similarity matrix from co-occurrence sim.")
    parser.add_argument(
        "--cooc-path",
        default="experiments/genre_similarity_lab/artifacts/genre_cooc_matrix.npz",
        help="Input co-occurrence npz (default: experiments/genre_similarity_lab/artifacts/genre_cooc_matrix.npz)",
    )
    parser.add_argument(
        "--output",
        default="experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz",
        help="Output similarity npz (default: experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz)",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.0,
        help="Zero out off-diagonal similarities below this threshold (default: 0.0).",
    )
    return parser.parse_args()


def build_similarity(sim_cooc: np.ndarray, min_sim: float) -> np.ndarray:
    """Construct final similarity matrix from co-occurrence similarity."""
    S = sim_cooc.astype(np.float32).copy()
    # Force diagonal to 1.0
    np.fill_diagonal(S, 1.0)
    # Symmetrize
    S = 0.5 * (S + S.T)
    if min_sim > 0:
        # Zero out off-diagonals below threshold
        mask = (S < min_sim)
        # Keep diagonal intact
        np.fill_diagonal(mask, False)
        S[mask] = 0.0
    return S


def top_neighbors(genre: str, genres: List[str], S: np.ndarray, k: int = 5) -> List[str]:
    """Return top-k neighbor strings for a given genre (excluding itself)."""
    if genre not in genres:
        return []
    idx = genres.index(genre)
    sims = S[idx].copy()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:k]
    return [f"{genres[j]}:{sims[j]:.2f}" for j in top_idx if sims[j] > 0]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cooc_path = Path(args.cooc_path)
    data = np.load(cooc_path, allow_pickle=True)
    genres = data["genres"].tolist()
    sim_cooc = data["sim_cooc"]
    N = len(genres)
    logging.info("Loaded co-occurrence sim: %s (genres=%d)", cooc_path, N)

    S_final = build_similarity(sim_cooc, args.min_sim)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, genres=np.array(genres), S=S_final)
    logging.info("Saved final similarity matrix: %s", out_path)

    # Diagnostics
    off_diag = S_final[np.triu_indices(N, k=1)]
    nonzero = off_diag[off_diag > 0]
    if nonzero.size:
        logging.info(
            "Off-diagonal similarity stats: min=%.4f max=%.4f mean=%.4f (nonzero count=%d)",
            nonzero.min(),
            nonzero.max(),
            nonzero.mean(),
            nonzero.size,
        )
    else:
        logging.info("No nonzero off-diagonal similarities.")

    anchors = ["jazz", "hard bop", "cool jazz", "rock", "post-punk", "shoegaze", "dream pop", "hip hop"]
    for a in anchors:
        neigh = top_neighbors(a, genres, S_final, k=5)
        if neigh:
            logging.info("Anchor %s -> %s", a, "; ".join(neigh))


if __name__ == "__main__":
    main()
