"""
Step 2: PCA embeddings + KNN neighbor explorer.

Loads Step 1 artifacts (data_matrices_step1.npz), fits PCA on sonic/genre
matrices, builds mode-specific embeddings, and reports nearest neighbors
for a seed track via cosine similarity.
"""

import argparse
import logging
import sys
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DEFAULT_ARTIFACT = "experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"

logger = logging.getLogger(__name__)


def fit_pca_embeddings(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """
    Standardize X, fit PCA(n_components=n_components, random_state=0),
    return (embeddings, scaler, pca).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    embeddings = pca.fit_transform(X_scaled)
    return embeddings, scaler, pca


def get_embedding_matrix(
    mode: str,
    E_sonic: np.ndarray,
    E_genre: np.ndarray,
    w_sonic: float,
    w_genre: float,
) -> np.ndarray:
    """
    - 'sonic'  -> return E_sonic
    - 'genre'  -> return E_genre
    - 'hybrid' -> return np.concatenate([w_sonic * E_sonic, w_genre * E_genre], axis=1)
    """
    if mode == "sonic":
        return E_sonic
    if mode == "genre":
        return E_genre
    if mode == "hybrid":
        return np.concatenate([w_sonic * E_sonic, w_genre * E_genre], axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def find_neighbors(
    seed_track_id: str,
    embedding_matrix: np.ndarray,
    track_ids: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """
    - Find index of seed_track_id in track_ids.
    - Compute cosine similarity between that row and all rows.
    - Exclude the seed itself.
    - Return a list of (index, similarity) for the top-k neighbors, sorted by similarity descending.
    """
    matches = np.where(track_ids == seed_track_id)[0]
    if len(matches) == 0:
        raise ValueError(f"Seed track_id not found: {seed_track_id}")
    seed_idx = int(matches[0])

    sims = cosine_similarity(embedding_matrix[seed_idx : seed_idx + 1], embedding_matrix)[0]
    sims[seed_idx] = -1  # exclude self

    k = min(k, len(track_ids) - 1)
    neighbor_indices = np.argpartition(-sims, k)[:k]
    sorted_indices = neighbor_indices[np.argsort(-sims[neighbor_indices])]
    return [(int(idx), float(sims[idx])) for idx in sorted_indices]


def _nonzero_genres(row: np.ndarray, genre_vocab: np.ndarray, limit: int = 5) -> List[str]:
    """Return up to `limit` genre names where the genre vector is non-zero."""
    nz = np.nonzero(row)[0]
    genres = [str(genre_vocab[i]) for i in nz[:limit]]
    return genres


def main():
    parser = argparse.ArgumentParser(description="PCA embeddings + KNN neighbor explorer (Step 2).")
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT,
        help=f"Path to Step 1 artifact (default: {DEFAULT_ARTIFACT})",
    )
    parser.add_argument("--seed-track-id", required=True, help="Seed track_id to query neighbors for.")
    parser.add_argument("--k", type=int, default=25, help="Number of neighbors to return (default: 25).")
    parser.add_argument(
        "--mode",
        choices=["sonic", "genre", "hybrid"],
        default="hybrid",
        help="Embedding mode (default: hybrid).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of PCA components for each space (default: 32).",
    )
    parser.add_argument("--w-sonic", type=float, default=0.6, help="Sonic weight for hybrid mode (default: 0.6).")
    parser.add_argument("--w-genre", type=float, default=0.4, help="Genre weight for hybrid mode (default: 0.4).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    artifact_path = args.artifact_path
    try:
        data = np.load(artifact_path, allow_pickle=True)
    except FileNotFoundError:
        print(
            f"Artifact not found at {artifact_path}. "
            "Run data_matrix_lab.py with --save-artifacts to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_sonic = data["X_sonic"]
    # Backward compatible loading of raw/smoothed genre matrices
    X_genre_raw = data["X_genre_raw"] if "X_genre_raw" in data else data["X_genre"]
    X_genre_smoothed = data["X_genre_smoothed"] if "X_genre_smoothed" in data else data["X_genre"]
    X_genre = X_genre_smoothed  # used for PCA/embedding
    track_ids = data["track_ids"]
    artist_names = data["artist_names"]
    track_titles = data["track_titles"]
    genre_vocab = data["genre_vocab"]

    n_tracks = X_sonic.shape[0]
    print("=== Loaded Artifacts ===")
    print(f"Tracks: {n_tracks}")
    print(f"X_sonic shape: {X_sonic.shape}")
    print(f"X_genre shape: {X_genre.shape}")
    print(f"Genre vocab size: {len(genre_vocab)}")
    print()

    # Fit PCA embeddings (clamp n_components to valid range)
    n_comp_sonic = min(args.n_components, X_sonic.shape[1], X_sonic.shape[0])
    n_comp_genre = min(args.n_components, X_genre.shape[1], X_genre.shape[0])
    if n_comp_sonic < args.n_components or n_comp_genre < args.n_components:
        print(
            f"Adjusted n_components: sonic={n_comp_sonic}, genre={n_comp_genre} "
            f"(requested {args.n_components})"
        )
    E_sonic, _, pca_sonic = fit_pca_embeddings(X_sonic, n_comp_sonic)
    E_genre, _, pca_genre = fit_pca_embeddings(X_genre, n_comp_genre)

    print("=== PCA Diagnostics ===")
    print(f"Sonic explained variance (sum): {pca_sonic.explained_variance_ratio_.sum():.4f}")
    print(f"Genre explained variance (sum): {pca_genre.explained_variance_ratio_.sum():.4f}")
    print(f"Sonic first components variance: {pca_sonic.explained_variance_ratio_[:5]}")
    print(f"Genre first components variance: {pca_genre.explained_variance_ratio_[:5]}")
    print()

    embedding_matrix = get_embedding_matrix(args.mode, E_sonic, E_genre, args.w_sonic, args.w_genre)

    try:
        neighbors = find_neighbors(args.seed_track_id, embedding_matrix, track_ids, args.k)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    seed_idx = int(np.where(track_ids == args.seed_track_id)[0][0])
    seed_label = f"{track_ids[seed_idx]} | {artist_names[seed_idx]}  {track_titles[seed_idx]}"

    print(f"Seed track:\n  {seed_label}\n")
    print(f"Nearest neighbors (mode={args.mode}, k={args.k}):\n")
    header = f"{'rank':>4} | {'sim':>6} | {'track_id':<32} | {'artist  title':<50} | genres"
    print(header)
    print("-" * len(header))

    for rank, (idx, sim) in enumerate(neighbors, start=1):
        track_id = track_ids[idx]
        artist = artist_names[idx]
        title = track_titles[idx]
        genres = _nonzero_genres(X_genre_raw[idx], genre_vocab, limit=58)
        print(
            f"{rank:>4} | {sim:>6.3f} | {track_id:<32} | "
            f"{(str(artist) + '  ' + str(title))[:50]:<50} | {genres}"
        )


if __name__ == "__main__":
    main()
