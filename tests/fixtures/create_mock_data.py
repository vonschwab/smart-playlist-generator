"""Create mock data fixtures for testing.

Generates:
- mock_genre_vectors.npz: Synthetic genre vectors for testing
- mock_sonic_matrices.npz: Synthetic sonic feature matrices for testing
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
RNG = np.random.default_rng(42)

# Configuration
N_TRACKS = 100  # Number of tracks
N_GENRES = 50   # Genre vocabulary size
SONIC_DIM = 32  # Sonic feature dimensionality


def create_mock_genre_vectors():
    """Create mock genre vectors."""
    print("Creating mock genre vectors...")

    # Create sparse genre vectors (most tracks have 2-5 genres)
    X_genre_raw = RNG.random(size=(N_TRACKS, N_GENRES))
    X_genre_raw[X_genre_raw < 0.9] = 0.0  # Make sparse (10% fill)

    # Create smoothed version (slightly less sparse)
    X_genre_smoothed = X_genre_raw + RNG.normal(scale=0.05, size=X_genre_raw.shape)
    X_genre_smoothed = np.clip(X_genre_smoothed, 0, 1)

    # Genre vocabulary
    genre_vocab = np.array([f"genre_{i:02d}" for i in range(N_GENRES)])

    # Track IDs
    track_ids = np.array([f"track_{i:04d}" for i in range(N_TRACKS)])

    # Save
    output_path = Path(__file__).parent / "mock_genre_vectors.npz"
    np.savez(
        output_path,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
        track_ids=track_ids,
    )
    print(f"Saved to: {output_path}")
    print(f"  Shape: {X_genre_raw.shape}")
    print(f"  Sparsity: {(X_genre_raw == 0).sum() / X_genre_raw.size:.1%}")


def create_mock_sonic_matrices():
    """Create mock sonic feature matrices."""
    print("Creating mock sonic matrices...")

    # Full-track sonic features (N x D)
    X_sonic = RNG.normal(size=(N_TRACKS, SONIC_DIM))

    # Segment-based features (start, mid, end)
    # Add slight variations to full track
    X_sonic_start = X_sonic + RNG.normal(scale=0.1, size=X_sonic.shape)
    X_sonic_mid = X_sonic + RNG.normal(scale=0.05, size=X_sonic.shape)
    X_sonic_end = X_sonic + RNG.normal(scale=0.1, size=X_sonic.shape)

    # Create transition matrices (end-to-start similarities)
    # These should be precomputed cosine similarities
    from scipy.spatial.distance import cdist
    X_full_sim = 1 - cdist(X_sonic, X_sonic, metric='cosine')
    X_transition_sim = 1 - cdist(X_sonic_end, X_sonic_start, metric='cosine')

    # Track metadata
    track_ids = np.array([f"track_{i:04d}" for i in range(N_TRACKS)])
    artist_keys = np.array([f"artist_{i % 20:02d}" for i in range(N_TRACKS)])
    track_artists = np.array([f"Artist {i % 20}" for i in range(N_TRACKS)])
    track_titles = np.array([f"Song {i}" for i in range(N_TRACKS)])

    # Save
    output_path = Path(__file__).parent / "mock_sonic_matrices.npz"
    np.savez(
        output_path,
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_mid=X_sonic_mid,
        X_sonic_end=X_sonic_end,
        X_full_sim=X_full_sim,
        X_transition_sim=X_transition_sim,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
    )
    print(f"Saved to: {output_path}")
    print(f"  Shape: {X_sonic.shape}")
    print(f"  Full sim matrix: {X_full_sim.shape}")
    print(f"  Transition sim matrix: {X_transition_sim.shape}")


if __name__ == "__main__":
    create_mock_genre_vectors()
    print()
    create_mock_sonic_matrices()
    print()
    print("Mock data fixtures created successfully!")
