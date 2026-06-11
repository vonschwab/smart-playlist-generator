"""Unit tests for scripts/calibrate_mert_transform.py.

All tests use synthetic data (tiny fake sidecar + artifact in tmp_path).
The real artifact and sidecar are never touched.

Tests follow TDD — they were written before the implementation to prove
each failure mode before the production code makes them pass.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# --- helpers ------------------------------------------------------------------


def _make_sidecar(
    path: Path,
    *,
    n: int = 30,
    dim: int = 16,
    rng_seed: int = 42,
    track_id_prefix: str = "track",
) -> list[str]:
    """Write a tiny fake sidecar npz; return the track_ids list."""
    rng = np.random.default_rng(rng_seed)
    tids = [f"{track_id_prefix}_{i:04d}" for i in range(n)]
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    np.savez(
        path,
        track_ids=np.array(tids, dtype=object),
        emb_start=emb,
        emb_mid=emb + rng.standard_normal((n, dim)).astype(np.float32) * 0.05,
        emb_end=emb + rng.standard_normal((n, dim)).astype(np.float32) * 0.05,
        model_name=np.array("m-a-p/MERT-v1-95M"),
        model_revision=np.array("dummy-revision"),
    )
    return tids


def _make_artifact(
    path: Path,
    tids: list[str],
    *,
    n_genres: int = 10,
    sonic_dim: int = 8,
    rng_seed: int = 7,
    artist_per_track: dict[str, str] | None = None,
) -> None:
    """Write a tiny fake artifact npz aligned to *tids*."""
    n = len(tids)
    rng = np.random.default_rng(rng_seed)

    # Build genre vectors: sparse, values in [0, 1]
    Xg = rng.random((n, n_genres)).astype(np.float32)
    Xg[Xg < 0.6] = 0.0

    Xs = rng.standard_normal((n, sonic_dim)).astype(np.float32)

    if artist_per_track is None:
        # default: one artist per 3 tracks
        artists = [f"Artist{i // 3}" for i in range(n)]
    else:
        artists = [artist_per_track.get(t, "Unknown") for t in tids]

    np.savez(
        path,
        track_ids=np.array(tids, dtype=object),
        artist_keys=np.array(artists, dtype=object),
        track_artists=np.array(artists, dtype=object),
        track_titles=np.array([f"Song {i}" for i in range(n)], dtype=object),
        X_sonic=Xs,
        X_sonic_tower_weighted=Xs,
        X_genre_raw=Xg,
        X_genre_smoothed=Xg,
        genre_vocab=np.array([f"genre_{i}" for i in range(n_genres)], dtype=object),
    )


# --- import the module under test (lazy so we can test import errors) ----------


def _import() -> types.ModuleType:
    mod_name = "scripts.calibrate_mert_transform"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).resolve().parents[2] / "scripts" / "calibrate_mert_transform.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# =============================================================================
# Test 1 — subset always includes seeds; no dups; total ≤ subset_size
# =============================================================================


def test_build_subset_includes_seeds(tmp_path: Path) -> None:
    """All requested seeds appear in the subset; no duplicates; total ≤ limit."""
    mod = _import()

    n_total = 60
    tids = _make_sidecar(tmp_path / "sc.npz", n=n_total, dim=16)
    _make_artifact(tmp_path / "art.npz", tids)

    # pick 5 seeds from the available track ids
    seeds = tids[::12][:5]  # indices 0, 12, 24, 36, 48

    result = mod.build_subset(
        sidecar_path=tmp_path / "sc.npz",
        artifact_path=tmp_path / "art.npz",
        db_path=None,  # no DB needed for synthetic test
        seed_ids=seeds,
        error_artist_ids=[],
        subset_size=40,
        rng=np.random.default_rng(0),
    )

    # all seeds must be present
    result_set = set(result)
    for s in seeds:
        assert s in result_set, f"seed {s!r} missing from subset"

    # no duplicates
    assert len(result) == len(result_set), "duplicate ids in subset"

    # total within limit
    assert len(result) <= 40


# =============================================================================
# Test 2 — stratified: no single artist dominates
# =============================================================================


def test_build_subset_stratified_by_artist(tmp_path: Path) -> None:
    """After subset selection no single artist should exceed max_per_artist."""
    mod = _import()

    # 60 tracks where Artist0 holds 30 and Artist1 holds 30
    tids = [f"t_{i:04d}" for i in range(60)]
    artist_map = {t: "DominantArtist" if i < 30 else f"Artist{i % 5}" for i, t in enumerate(tids)}

    _make_sidecar(tmp_path / "sc.npz", n=60, dim=16, track_id_prefix="t")
    # Override track_ids to match our custom list
    rng = np.random.default_rng(99)
    emb = rng.standard_normal((60, 16)).astype(np.float32)
    np.savez(
        tmp_path / "sc.npz",
        track_ids=np.array(tids, dtype=object),
        emb_start=emb,
        emb_mid=emb,
        emb_end=emb,
        model_name=np.array("m-a-p/MERT-v1-95M"),
        model_revision=np.array("dummy"),
    )
    _make_artifact(tmp_path / "art.npz", tids, artist_per_track=artist_map)

    MAX_PER_ARTIST = 10
    result = mod.build_subset(
        sidecar_path=tmp_path / "sc.npz",
        artifact_path=tmp_path / "art.npz",
        db_path=None,
        seed_ids=[],
        error_artist_ids=[],
        subset_size=50,
        rng=np.random.default_rng(0),
        max_per_artist=MAX_PER_ARTIST,
    )

    # Count dominant artist in result
    from collections import Counter

    artist_counts = Counter(artist_map.get(t, "Unknown") for t in result)
    assert artist_counts["DominantArtist"] <= MAX_PER_ARTIST, (
        f"DominantArtist has {artist_counts['DominantArtist']} tracks, "
        f"exceeds max_per_artist={MAX_PER_ARTIST}"
    )


# =============================================================================
# Test 3 — center_l2: vectors are unit-norm; mean ≈ 0 before normalization
# =============================================================================


def test_fit_center_l2(tmp_path: Path) -> None:
    """center_l2: output is L2-normalized; centered mean ≈ 0."""
    mod = _import()

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    # shift so raw mean is far from 0
    X += 5.0

    params, Xt = mod.fit_transform("center_l2", X)

    # L2-normalized: all norms ≈ 1
    norms = np.linalg.norm(Xt, axis=1)
    np.testing.assert_allclose(norms, np.ones(40), atol=1e-5)

    # mean before normalization should be near 0 (centering worked)
    mean_param = params["center_l2_mean"]
    centered = X - mean_param
    np.testing.assert_allclose(centered.mean(axis=0), np.zeros(8), atol=1e-4)


# =============================================================================
# Test 4 — PCA dimension reduction
# =============================================================================


def test_fit_pca_reduces_dim(tmp_path: Path) -> None:
    """center_pca128 → 128-d; whiten_pca256 → 256-d (or min(n-1, k) in low-n case)."""
    mod = _import()

    rng = np.random.default_rng(2)

    # Use enough samples > max PCA dims
    # For the synthetic case we use smaller dims to keep it fast
    X_small = rng.standard_normal((100, 64)).astype(np.float32)

    # center_pca128 with k=32 (can't exceed input dim=64)
    _params128, Xt128 = mod.fit_transform("center_pca128", X_small, pca_k=32)
    assert Xt128.shape == (100, 32), f"expected (100, 32), got {Xt128.shape}"

    # whiten_pca256 with k=16
    _params256, Xt256 = mod.fit_transform("whiten_pca256", X_small, pca_k=16)
    assert Xt256.shape == (100, 16), f"expected (100, 16), got {Xt256.shape}"

    # Both must be L2-normalized
    np.testing.assert_allclose(
        np.linalg.norm(Xt128, axis=1), np.ones(100), atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(Xt256, axis=1), np.ones(100), atol=1e-5
    )


# =============================================================================
# Test 5 — coherence metric rewards correct embedding over shuffled
# =============================================================================


def test_coherence_metric_rewards_same_genre_neighbors(tmp_path: Path) -> None:
    """Genre-coherent embedding should score higher coherence than shuffled embedding."""
    mod = _import()

    rng = np.random.default_rng(3)
    n = 40
    n_genres = 8

    # Build genre matrix: 4 genre clusters of 10 tracks each
    Xg = np.zeros((n, n_genres), dtype=np.float32)
    for cluster in range(4):
        lo = cluster * 10
        hi = lo + 10
        Xg[lo:hi, cluster * 2 : cluster * 2 + 2] = 1.0  # strong same-genre signal

    # "Good" embedding: tracks in the same cluster are nearest neighbors
    # Build as cluster centroid + small noise
    X_good = np.zeros((n, 8), dtype=np.float32)
    for cluster in range(4):
        lo = cluster * 10
        hi = lo + 10
        centroid = np.zeros(8, dtype=np.float32)
        centroid[cluster] = 5.0
        X_good[lo:hi] = centroid + rng.standard_normal((10, 8)).astype(np.float32) * 0.1
    X_good /= np.linalg.norm(X_good, axis=1, keepdims=True) + 1e-9

    # "Shuffled" embedding: random — no cluster structure
    X_shuffled = rng.standard_normal((n, 8)).astype(np.float32)
    X_shuffled /= np.linalg.norm(X_shuffled, axis=1, keepdims=True) + 1e-9

    # seeds = one per cluster
    seed_indices = [0, 10, 20, 30]

    coherence_good = mod.compute_coherence(
        seed_indices=seed_indices,
        X_emb=X_good,
        X_genre_raw=Xg,
        k_neighbors=6,
    )
    coherence_shuffled = mod.compute_coherence(
        seed_indices=seed_indices,
        X_emb=X_shuffled,
        X_genre_raw=Xg,
        k_neighbors=6,
    )

    assert coherence_good > coherence_shuffled, (
        f"Good embedding coherence {coherence_good:.4f} should exceed "
        f"shuffled {coherence_shuffled:.4f}"
    )


# =============================================================================
# Test 6 — output NPZ contains all required transform parameter keys
# =============================================================================


def test_output_npz_has_all_transform_keys(tmp_path: Path) -> None:
    """After fitting all four transforms, the output NPZ has the expected keys."""
    mod = _import()

    rng = np.random.default_rng(4)
    n, dim = 50, 16
    tids = [f"u_{i:04d}" for i in range(n)]

    emb = rng.standard_normal((n, dim)).astype(np.float32)
    np.savez(
        tmp_path / "sc.npz",
        track_ids=np.array(tids, dtype=object),
        emb_start=emb,
        emb_mid=emb,
        emb_end=emb,
        model_name=np.array("m-a-p/MERT-v1-95M"),
        model_revision=np.array("dummy"),
    )
    _make_artifact(tmp_path / "art.npz", tids)

    out_path = tmp_path / "calib.npz"
    mod.fit_and_save_transforms(
        X=emb,
        out_path=out_path,
        pca_k_center=8,   # smaller than dim to keep test fast
        pca_k_whiten=8,
    )

    with np.load(out_path, allow_pickle=True) as z:
        keys = set(z.keys())

    # center_l2 requires mean
    assert "center_l2_mean" in keys, f"missing center_l2_mean in {keys}"

    # whiten_l2 requires mean + std
    assert "whiten_l2_mean" in keys
    assert "whiten_l2_std" in keys

    # center_pca128 requires mean + components
    assert "center_pca128_mean" in keys
    assert "center_pca128_components" in keys

    # whiten_pca256 requires mean + std + components
    assert "whiten_pca256_mean" in keys
    assert "whiten_pca256_std" in keys
    assert "whiten_pca256_components" in keys
