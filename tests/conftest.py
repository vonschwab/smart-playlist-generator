"""Test configuration and fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.artifacts import load_artifact_bundle


def _build_artifact(tmp_path, seed: int = 0, include_segments: bool = True):
    """Build a synthetic artifact for testing."""
    rng = np.random.default_rng(seed)
    N = 50
    G = 18
    track_ids = np.array([f"t{i}" for i in range(N)])
    artist_keys = np.array([f"a{i % 10}" for i in range(N)])
    track_artists = np.array([f"Artist {i % 10}" for i in range(N)])
    track_titles = np.array([f"Song {i}" for i in range(N)])
    X_sonic = rng.normal(size=(N, 12))
    X_sonic_start = X_sonic + rng.normal(scale=0.05, size=X_sonic.shape) if include_segments else None
    X_sonic_mid = X_sonic + rng.normal(scale=0.05, size=X_sonic.shape) if include_segments else None
    X_sonic_end = X_sonic + rng.normal(scale=0.05, size=X_sonic.shape) if include_segments else None
    X_genre_raw = rng.random(size=(N, G))
    X_genre_raw[X_genre_raw < 0.7] = 0.0  # sparse-ish
    X_genre_smoothed = X_genre_raw + rng.normal(scale=0.02, size=X_genre_raw.shape)
    genre_vocab = np.array([f"genre_{i}" for i in range(G)])

    path = tmp_path / "artifact.npz"
    np.savez(
        path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_mid=X_sonic_mid,
        X_sonic_end=X_sonic_end,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )
    return path


@pytest.fixture()
def synthetic_artifact(tmp_path):
    """Create a synthetic artifact for testing."""
    path = _build_artifact(tmp_path, seed=123)
    bundle = load_artifact_bundle(path)
    return path, bundle


@pytest.fixture()
def qtbot():
    """
    Minimal shim so tests that accept `qtbot` can run without pytest-qt.
    The fixture isn't used in assertions, so a dummy object is sufficient.
    """
    class _DummyQtBot:
        def __getattr__(self, name):
            # Allow arbitrary attribute access without failing the test
            return lambda *args, **kwargs: None

    return _DummyQtBot()
