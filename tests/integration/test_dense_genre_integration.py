"""
Dense genre embedding integration tests (Phase 3+4).

Tests are organized by scope:
  - Fast synthetic tests (no real artifact needed): sidecar loading, pool routing
  - Slow real-data tests (require live artifact + sidecar): pool expansion,
    genre coherence, and playlist generation

Run the fast subset only:
    pytest tests/integration/test_dense_genre_integration.py -m "not slow"

Run everything:
    pytest tests/integration/test_dense_genre_integration.py -m "integration"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.artifacts import ArtifactBundle, load_artifact_bundle
from src.genre.artifact_identity import (
    DENSE_SIDECAR_SCHEMA_VERSION,
    dense_sidecar_mismatch_reason_from_paths,
    genre_artifact_identity,
)
from src.genre.pmi_svd import train_pmi_svd
from src.playlist.candidate_pool import build_candidate_pool, CandidatePoolConfig

ARTIFACT_PATH = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
SIDECAR_PATH = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1_genre_emb_dim64.npz"

# Reference cases — track IDs and expected genre groups
REFERENCE_CASES = {
    "charli_xcx": {
        "track_id": "065933d8e2e0db664ec57af1511b662b",
        "niche_genres": {"bubblegum bass", "pc music", "hyperpop", "deconstructed club"},
        "description": "dense-niche enriched — known failure case",
    },
    "acetone": {
        "track_id": "6d5696a29b57a765f945ed1be2d6dfee",
        "niche_genres": {"slowcore", "dreampop", "neo-psychedelia"},
        "description": "narrow-niche raw — baseline OK case",
    },
    "pharoah_sanders": {
        "track_id": "a3bb6db554f1d2bae1a8b998ebf53925",
        "niche_genres": {"spiritual jazz", "free jazz", "avant-garde jazz", "soul jazz"},
        "description": "rich-mixed jazz",
    },
    "beach_boys": {
        "track_id": "fc302980e8359e5bab53e7a2f45fc61b",
        "niche_genres": {"sunshine pop", "surf rock", "surf"},
        "description": "sparse pop",
    },
}

# Minimum pool sizes that confirm the dense method is working
DENSE_POOL_MINIMUMS = {
    "charli_xcx": 200,  # was 31 with sparse; 625 with dense
    "acetone": 500,     # was 154 with sparse; hits cap with dense
    "pharoah_sanders": 300,
    "beach_boys": 200,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _requires_live_artifact(func):
    """Decorator: skip if artifact or sidecar missing."""
    return pytest.mark.skipif(
        not ARTIFACT_PATH.exists() or not SIDECAR_PATH.exists(),
        reason="Live artifact or genre embedding sidecar not found. "
               "Run: python scripts/build_genre_embedding.py --skip-prior",
    )(func)


def _make_pool_cfg(**kwargs) -> CandidatePoolConfig:
    defaults = dict(
        similarity_floor=0.0,
        min_sonic_similarity=None,
        max_pool_size=2000,
        target_artists=50,
        candidates_per_artist=5,
        seed_artist_bonus=3,
        max_artist_fraction_final=0.3,
    )
    defaults.update(kwargs)
    return CandidatePoolConfig(**defaults)


@pytest.fixture(scope="module")
def live_bundle():
    if not ARTIFACT_PATH.exists():
        pytest.skip("Live artifact not found")
    load_artifact_bundle.cache_clear()
    return load_artifact_bundle(ARTIFACT_PATH)


@pytest.fixture(scope="module")
def mini_bundle_with_sidecar(tmp_path_factory):
    """
    Synthetic 120-track artifact + matching dense sidecar for fast pipeline tests.
    Three genre clusters: rock (0-39), electronic (40-79), jazz (80-119).
    """
    rng = np.random.default_rng(0)
    N, V, dim = 120, 12, 8
    tmpdir = tmp_path_factory.mktemp("dense_genre")

    # Genre vocab: 4 per cluster
    vocab = ["rock", "indie-rock", "alternative", "post-punk",
             "techno", "house", "trance", "electronic",
             "jazz", "bebop", "swing", "soul-jazz"]

    X_raw = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        cluster = i // 40  # 0 = rock, 1 = electronic, 2 = jazz
        base = cluster * 4
        X_raw[i, base:base + 4] = rng.uniform(0.5, 1.0, size=4)

    # Dense sidecar: PMI-SVD from X_raw
    genre_emb = train_pmi_svd(X_raw, dim=dim, smoothing=1.0, random_state=42)
    projected = X_raw @ genre_emb
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    X_dense = projected / np.maximum(norms, 1e-12)
    X_dense = X_dense.astype(np.float32)

    track_ids = np.array([f"track_{i:04d}" for i in range(N)], dtype=object)
    artist_keys = np.array([f"artist_{i % 30:02d}" for i in range(N)], dtype=object)
    X_sonic = rng.normal(size=(N, 16)).astype(np.float32)

    artifact_path = tmpdir / "mini.npz"
    sidecar_path = tmpdir / "mini_genre_emb_dim8.npz"

    np.savez(artifact_path,
             track_ids=track_ids, artist_keys=artist_keys,
             track_artists=np.array([f"Artist {i % 30}" for i in range(N)], dtype=object),
             track_titles=np.array([f"Track {i}" for i in range(N)], dtype=object),
             X_sonic=X_sonic, X_genre_raw=X_raw, X_genre_smoothed=X_raw,
             genre_vocab=np.array(vocab, dtype=object))

    # Sidecar uses dim=8 but load_artifact_bundle looks for dim=64 by default.
    # Patch: write a dim=64 sidecar with zero-padded embeddings for this test.
    X_dense_64 = np.pad(X_dense, ((0, 0), (0, 56))).astype(np.float32)
    genre_emb_64 = np.pad(genre_emb, ((0, 0), (0, 56))).astype(np.float32)
    np.savez(tmpdir / "mini_genre_emb_dim64.npz",
             X_genre_dense=X_dense_64, genre_emb=genre_emb_64,
             genre_vocab=np.array(vocab, dtype=object), track_ids=track_ids,
             emb_config={
                 "dim": 64,
                 "schema_version": DENSE_SIDECAR_SCHEMA_VERSION,
                 "sparse_genre_identity": genre_artifact_identity(
                     track_ids, np.array(vocab, dtype=object), X_raw,
                 ),
             })

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact_path)
    return bundle, X_dense  # Return true dim-8 dense for reference


def _write_mini_artifact(tmp_path):
    track_ids = np.array(["t1"], dtype=object)
    vocab = np.array(["rock"], dtype=object)
    X_raw = np.array([[1.0]], dtype=np.float32)
    artifact = tmp_path / "mini.npz"
    np.savez(
        artifact,
        track_ids=track_ids,
        artist_keys=np.array(["a"], dtype=object),
        track_artists=np.array(["A"], dtype=object),
        track_titles=np.array(["T"], dtype=object),
        X_sonic=np.array([[1.0]], dtype=np.float32),
        X_genre_raw=X_raw,
        X_genre_smoothed=X_raw,
        genre_vocab=vocab,
    )
    return artifact, track_ids, vocab, X_raw


def _write_mini_sidecar(tmp_path, *, track_ids, vocab, emb_config):
    np.savez(
        tmp_path / "mini_genre_emb_dim64.npz",
        X_genre_dense=np.zeros((1, 64), dtype=np.float32),
        genre_emb=np.zeros((1, 64), dtype=np.float32),
        genre_vocab=vocab,
        track_ids=track_ids,
        emb_config=emb_config,
    )


# ---------------------------------------------------------------------------
# 1. Sidecar integrity tests (fast, uses live artifact if present)
# ---------------------------------------------------------------------------

def test_loader_rejects_dense_sidecar_when_vocab_differs(tmp_path, caplog):
    from src.genre.artifact_identity import (
        DENSE_SIDECAR_SCHEMA_VERSION,
        genre_artifact_identity,
    )

    artifact, track_ids, vocab, X_raw = _write_mini_artifact(tmp_path)
    _write_mini_sidecar(
        tmp_path,
        track_ids=track_ids,
        vocab=np.array(["jazz"], dtype=object),
        emb_config={
            "schema_version": DENSE_SIDECAR_SCHEMA_VERSION,
            "sparse_genre_identity": genre_artifact_identity(track_ids, vocab, X_raw),
        },
    )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "vocabulary mismatch" in caplog.text


def test_loader_rejects_dense_sidecar_when_track_ids_differ(tmp_path, caplog):
    artifact, _, vocab, _ = _write_mini_artifact(tmp_path)
    _write_mini_sidecar(
        tmp_path,
        track_ids=np.array(["other"], dtype=object),
        vocab=vocab,
        emb_config={},
    )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "track_ids mismatch" in caplog.text


def test_loader_rejects_dense_sidecar_when_sparse_identity_differs(tmp_path, caplog):
    from src.genre.artifact_identity import DENSE_SIDECAR_SCHEMA_VERSION

    artifact, track_ids, vocab, _ = _write_mini_artifact(tmp_path)
    _write_mini_sidecar(
        tmp_path,
        track_ids=track_ids,
        vocab=vocab,
        emb_config={
            "schema_version": DENSE_SIDECAR_SCHEMA_VERSION,
            "sparse_genre_identity": "wrong",
        },
    )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "sparse genre identity mismatch" in caplog.text


def test_loader_rejects_legacy_dense_sidecar_without_identity(tmp_path, caplog):
    artifact, track_ids, vocab, _ = _write_mini_artifact(tmp_path)
    _write_mini_sidecar(tmp_path, track_ids=track_ids, vocab=vocab, emb_config={"dim": 64})

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "schema version mismatch" in caplog.text
    assert "build_genre_embedding.py" in caplog.text


def test_path_validator_rejects_sidecar_missing_expected_key(tmp_path):
    artifact, track_ids, vocab, X_raw = _write_mini_artifact(tmp_path)
    sidecar = tmp_path / "mini_genre_emb_dim64.npz"
    np.savez(
        sidecar,
        X_genre_dense=np.zeros((1, 64), dtype=np.float32),
        genre_vocab=vocab,
        track_ids=track_ids,
        emb_config={
            "schema_version": DENSE_SIDECAR_SCHEMA_VERSION,
            "sparse_genre_identity": genre_artifact_identity(track_ids, vocab, X_raw),
        },
    )

    reason = dense_sidecar_mismatch_reason_from_paths(
        artifact_path=artifact,
        sidecar_path=sidecar,
    )
    assert reason == "sidecar missing required keys: genre_emb"


def test_builder_persists_dense_sidecar_identity_metadata(tmp_path, monkeypatch):
    from scripts import build_genre_embedding

    artifact, track_ids, vocab, X_raw = _write_mini_artifact(tmp_path)
    monkeypatch.setattr(
        build_genre_embedding,
        "train_pmi_svd",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        build_genre_embedding,
        "project_tracks",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=np.float32),
    )

    sidecar = build_genre_embedding.build_genre_embedding_sidecar(artifact, dim=1)
    with np.load(sidecar, allow_pickle=True) as data:
        config = data["emb_config"].item()

    assert config["schema_version"] == DENSE_SIDECAR_SCHEMA_VERSION
    assert config["sparse_genre_identity"] == genre_artifact_identity(track_ids, vocab, X_raw)


@pytest.mark.integration
@_requires_live_artifact
def test_sidecar_loaded_into_bundle(live_bundle):
    """Bundle has X_genre_dense when sidecar exists."""
    assert live_bundle.X_genre_dense is not None, "X_genre_dense should be loaded from sidecar"
    assert live_bundle.genre_emb is not None


@pytest.mark.integration
@_requires_live_artifact
def test_sidecar_shape(live_bundle):
    N = live_bundle.track_ids.shape[0]
    V = live_bundle.genre_vocab.shape[0]
    assert live_bundle.X_genre_dense.shape == (N, 64), \
        f"Expected ({N}, 64), got {live_bundle.X_genre_dense.shape}"
    assert live_bundle.genre_emb.shape == (V, 64), \
        f"Expected ({V}, 64), got {live_bundle.genre_emb.shape}"


@pytest.mark.integration
@_requires_live_artifact
def test_sidecar_rows_l2_normalized_for_tracks_with_genres(live_bundle):
    """Rows with at least one genre should have L2 norm ≈ 1."""
    X_dense = live_bundle.X_genre_dense
    X_raw = live_bundle.X_genre_raw
    has_genre = (X_raw > 0).any(axis=1)
    norms = np.linalg.norm(X_dense[has_genre], axis=1)
    # All norms should be in [0.99, 1.01]
    bad = np.count_nonzero(np.abs(norms - 1.0) > 0.01)
    assert bad == 0, f"{bad} / {has_genre.sum()} genre-bearing tracks have abnormal L2 norm"


@pytest.mark.integration
@_requires_live_artifact
def test_sidecar_zero_rows_for_tracks_without_genres(live_bundle):
    """Tracks with no genres should have zero dense vector."""
    X_dense = live_bundle.X_genre_dense
    X_raw = live_bundle.X_genre_raw
    no_genre = ~(X_raw > 0).any(axis=1)
    if no_genre.sum() == 0:
        pytest.skip("All tracks have at least one genre — nothing to check")
    norms = np.linalg.norm(X_dense[no_genre], axis=1)
    assert norms.max() < 0.01, "Tracks without genres should have near-zero dense vectors"


@pytest.mark.integration
@_requires_live_artifact
def test_genre_emb_rows_l2_normalized(live_bundle):
    """Every row in genre_emb (V, 64) should be L2-normalized."""
    norms = np.linalg.norm(live_bundle.genre_emb, axis=1)
    bad = np.count_nonzero(np.abs(norms - 1.0) > 0.01)
    assert bad == 0, f"{bad} genre embedding rows have abnormal L2 norm"


# ---------------------------------------------------------------------------
# 2. Dense routing correctness (synthetic, fast)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_dense_path_used_when_x_genre_dense_present(mini_bundle_with_sidecar, caplog):
    """When X_genre_dense is in the bundle, pool builder logs 'method=dense'."""
    bundle, _ = mini_bundle_with_sidecar
    cfg = _make_pool_cfg(max_pool_size=60, candidates_per_artist=3)

    with caplog.at_level(logging.INFO, logger="src.playlist.candidate_pool"):
        build_candidate_pool(
            seed_idx=0,
            embedding=bundle.X_sonic,
            artist_keys=bundle.artist_keys,
            cfg=cfg, random_seed=42,
            X_genre_raw=bundle.X_genre_raw,
            X_genre_smoothed=bundle.X_genre_smoothed,
            X_genre_dense=bundle.X_genre_dense,
            min_genre_similarity=0.30,
        )

    assert any("method=dense" in r.message for r in caplog.records), \
        "Expected 'method=dense (PMI-SVD)' in log when X_genre_dense is present"


@pytest.mark.integration
def test_sparse_path_used_when_x_genre_dense_absent(mini_bundle_with_sidecar, caplog):
    """When X_genre_dense=None, pool builder falls back to sparse method."""
    bundle, _ = mini_bundle_with_sidecar
    cfg = _make_pool_cfg(max_pool_size=60, candidates_per_artist=3)

    with caplog.at_level(logging.INFO, logger="src.playlist.candidate_pool"):
        build_candidate_pool(
            seed_idx=0,
            embedding=bundle.X_sonic,
            artist_keys=bundle.artist_keys,
            cfg=cfg, random_seed=42,
            X_genre_raw=bundle.X_genre_raw,
            X_genre_smoothed=bundle.X_genre_smoothed,
            X_genre_dense=None,
            min_genre_similarity=0.30,
        )

    assert not any("method=dense" in r.message for r in caplog.records), \
        "Expected sparse path when X_genre_dense=None"
    assert any("method=" in r.message for r in caplog.records), \
        "Expected some genre method logged"


@pytest.mark.integration
def test_same_cluster_tracks_have_high_dense_similarity(mini_bundle_with_sidecar):
    """Within-cluster tracks should have high cosine similarity in dense space."""
    bundle, X_dense = mini_bundle_with_sidecar
    # Rock cluster: tracks 0-39. Electronic: 40-79.
    rock_rock = float(np.dot(X_dense[0], X_dense[10]))
    rock_electronic = float(np.dot(X_dense[0], X_dense[50]))
    assert rock_rock > rock_electronic, \
        f"Within-cluster ({rock_rock:.3f}) should exceed cross-cluster ({rock_electronic:.3f})"


@pytest.mark.integration
def test_dense_gate_expands_pool_for_tight_cluster(mini_bundle_with_sidecar):
    """Dense method gives more candidates than sparse for same min_threshold."""
    bundle, _ = mini_bundle_with_sidecar
    cfg = _make_pool_cfg(max_pool_size=200, candidates_per_artist=5)

    # Seed = first electronic track (idx 40); tight threshold
    result_dense = build_candidate_pool(
        seed_idx=40, embedding=bundle.X_sonic, artist_keys=bundle.artist_keys,
        cfg=cfg, random_seed=42,
        X_genre_raw=bundle.X_genre_raw, X_genre_smoothed=bundle.X_genre_smoothed,
        X_genre_dense=bundle.X_genre_dense, min_genre_similarity=0.30,
    )
    result_sparse = build_candidate_pool(
        seed_idx=40, embedding=bundle.X_sonic, artist_keys=bundle.artist_keys,
        cfg=cfg, random_seed=42,
        X_genre_raw=bundle.X_genre_raw, X_genre_smoothed=bundle.X_genre_smoothed,
        X_genre_dense=None, min_genre_similarity=0.30,
    )
    assert len(result_dense.pool_indices) >= len(result_sparse.pool_indices), \
        f"Dense ({len(result_dense.pool_indices)}) should be ≥ sparse ({len(result_sparse.pool_indices)})"


# ---------------------------------------------------------------------------
# 3. Real-data pool expansion tests (slow)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
@pytest.mark.parametrize("name", list(REFERENCE_CASES.keys()))
def test_dense_pool_meets_minimum(live_bundle, name):
    """Dense pool size for each reference case exceeds minimum threshold."""
    case = REFERENCE_CASES[name]
    seed_idx = live_bundle.track_id_to_index[case["track_id"]]
    cfg = _make_pool_cfg()

    result = build_candidate_pool(
        seed_idx=seed_idx,
        embedding=live_bundle.X_sonic,
        artist_keys=live_bundle.artist_keys,
        track_ids=live_bundle.track_ids,
        track_titles=live_bundle.track_titles,
        track_artists=live_bundle.track_artists,
        durations_ms=live_bundle.durations_ms,
        cfg=cfg, random_seed=42,
        X_genre_raw=live_bundle.X_genre_raw,
        X_genre_smoothed=live_bundle.X_genre_smoothed,
        X_genre_dense=live_bundle.X_genre_dense,
        min_genre_similarity=0.30,  # live config value
    )

    minimum = DENSE_POOL_MINIMUMS[name]
    assert len(result.pool_indices) >= minimum, (
        f"{name}: dense pool {len(result.pool_indices)} < minimum {minimum}. "
        f"Dense path may not be active or sidecar is stale."
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
@pytest.mark.parametrize("name", list(REFERENCE_CASES.keys()))
def test_dense_pool_exceeds_sparse_pool(live_bundle, name):
    """Dense pool must be strictly larger than sparse pool for every reference case."""
    case = REFERENCE_CASES[name]
    seed_idx = live_bundle.track_id_to_index[case["track_id"]]
    cfg = _make_pool_cfg()
    common = dict(
        seed_idx=seed_idx, embedding=live_bundle.X_sonic, artist_keys=live_bundle.artist_keys,
        track_ids=live_bundle.track_ids, track_titles=live_bundle.track_titles,
        track_artists=live_bundle.track_artists, durations_ms=live_bundle.durations_ms,
        cfg=cfg, random_seed=42, X_genre_raw=live_bundle.X_genre_raw,
        X_genre_smoothed=live_bundle.X_genre_smoothed, min_genre_similarity=0.30,
    )
    dense = build_candidate_pool(**common, X_genre_dense=live_bundle.X_genre_dense)
    sparse = build_candidate_pool(**common, X_genre_dense=None)

    assert len(dense.pool_indices) > len(sparse.pool_indices), (
        f"{name}: dense pool {len(dense.pool_indices)} not > sparse pool {len(sparse.pool_indices)}"
    )


# ---------------------------------------------------------------------------
# 4. Genre coherence tests (slow, real data)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
def test_charli_xcx_dense_sim_distribution(live_bundle):
    """
    In dense space, Charli XCX p50 similarity should be above the p50 from
    the sparse baseline (which was 0.0 — most tracks had zero sparse similarity).
    """
    seed_tid = REFERENCE_CASES["charli_xcx"]["track_id"]
    seed_idx = live_bundle.track_id_to_index[seed_tid]

    X_dense = live_bundle.X_genre_dense
    seed_vec = X_dense[seed_idx]
    sims = (X_dense @ seed_vec).copy()
    sims[seed_idx] = 0.0

    p50_dense = float(np.percentile(sims, 50))
    # Sparse baseline p50 was 0.0 (more than half the corpus has zero genre overlap)
    assert p50_dense > 0.0, (
        f"Dense p50 similarity for Charli XCX is {p50_dense:.4f}; "
        "should be > 0 (sparse baseline was 0.0)"
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
def test_dense_sim_niche_artists_higher_than_sparse(live_bundle):
    """
    Charli XCX dense p90 similarity should be higher than the sparse p90 baseline (0.19).
    This confirms the embedding compresses niche genre distances properly.
    """
    seed_tid = REFERENCE_CASES["charli_xcx"]["track_id"]
    seed_idx = live_bundle.track_id_to_index[seed_tid]

    X_dense = live_bundle.X_genre_dense
    seed_vec = X_dense[seed_idx]
    sims = (X_dense @ seed_vec).copy()
    sims[seed_idx] = 0.0

    p90_dense = float(np.percentile(sims, 90))
    p90_sparse_baseline = 0.19  # from baseline_metrics.json
    assert p90_dense > p90_sparse_baseline, (
        f"Dense p90 {p90_dense:.3f} should exceed sparse baseline {p90_sparse_baseline}"
    )


# ---------------------------------------------------------------------------
# 5. Full pipeline smoke test (slow, real data)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
@pytest.mark.parametrize("name,mode,length", [
    # acetone and pharoah work in dynamic mode (sonic-coherent pool)
    ("acetone", "dynamic", 20),
    ("pharoah_sanders", "dynamic", 20),
    # charli_xcx and beach_boys fail in dynamic mode due to a pre-existing
    # transition_floor=0.35 issue (sparse-pool sonic chains too tight).
    # They succeed in discover mode where the genre gate is off.
    # This documents the known limitation, NOT a regression from dense changes.
    ("charli_xcx", "discover", 20),
    ("beach_boys", "discover", 20),
])
def test_full_pipeline_generates_playlist(name, mode, length):
    """End-to-end playlist generation completes without error and has correct length."""
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    case = REFERENCE_CASES[name]
    load_artifact_bundle.cache_clear()

    result = generate_playlist_ds(
        artifact_path=str(ARTIFACT_PATH),
        seed_track_id=case["track_id"],
        mode=mode,
        length=length,
        random_seed=42,
    )

    assert len(result.track_ids) == length, \
        f"{name} [{mode}]: expected {length} tracks, got {len(result.track_ids)}"
    assert result.track_ids[0] == case["track_id"], \
        f"{name} [{mode}]: first track should be seed"
    assert len(set(result.track_ids)) == length, \
        f"{name} [{mode}]: playlist has duplicate tracks"


@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
def test_charli_xcx_playlist_genre_coherence():
    """
    Charli XCX playlist (discover mode — avoids pre-existing dynamic transition_floor issue):
    playlist tracks should have meaningful dense genre similarity with seed.
    """
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    case = REFERENCE_CASES["charli_xcx"]
    load_artifact_bundle.cache_clear()

    result = generate_playlist_ds(
        artifact_path=str(ARTIFACT_PATH),
        seed_track_id=case["track_id"],
        mode="discover",  # discover: no genre gate, so we test genre coherence post-hoc
        length=20,
        random_seed=42,
    )

    bundle = load_artifact_bundle(ARTIFACT_PATH)
    seed_idx = bundle.track_id_to_index[case["track_id"]]
    X_dense = bundle.X_genre_dense
    seed_vec = X_dense[seed_idx]

    playlist_dense_sims = []
    for tid in result.track_ids[1:]:  # Skip seed itself
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is not None:
            sim = float(np.dot(X_dense[idx], seed_vec))
            playlist_dense_sims.append(sim)

    assert len(playlist_dense_sims) > 0
    median_sim = float(np.median(playlist_dense_sims))
    # Even without the genre gate, dense sim should be above zero
    # (sonic-similar tracks tend to be genre-similar in dense space too)
    assert median_sim > 0.05, (
        f"Charli XCX playlist (discover) median dense genre sim = {median_sim:.3f}; "
        "expected > 0.05."
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
def test_dynamic_mode_failure_is_not_caused_by_dense_embedding():
    """
    Document that Charli XCX dynamic-mode failure is pre-existing when infeasible_handling
    is disabled (the default): it should fail regardless of whether X_genre_dense is present.
    This is a regression guard — the failure is expected with no infeasible_handling config.
    """
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    case = REFERENCE_CASES["charli_xcx"]

    # With dense sidecar loaded (our new behavior), no infeasible_handling overrides
    load_artifact_bundle.cache_clear()
    with pytest.raises(Exception, match="infeasible"):
        generate_playlist_ds(
            artifact_path=str(ARTIFACT_PATH),
            seed_track_id=case["track_id"],
            mode="dynamic",
            length=20,
            random_seed=42,
        )


@pytest.mark.integration
@pytest.mark.slow
@_requires_live_artifact
def test_dynamic_mode_succeeds_with_infeasible_handling_relaxation():
    """
    Charli XCX dynamic mode succeeds when infeasible_handling is enabled with
    transition_floor relaxation.  Without relaxation the beam hard-gates ~95% of
    edges at transition_floor=0.35; with relaxation we step down to min_transition_floor
    (0.20) and find a valid path.
    """
    from src.playlist.ds_pipeline_runner import generate_playlist_ds

    case = REFERENCE_CASES["charli_xcx"]

    overrides = {
        "pier_bridge": {
            "infeasible_handling": {
                "enabled": True,
                "min_bridge_floor": 0.0,
                "widen_search_on_backoff": True,
                "extra_neighbors_m": 400,
                "extra_bridge_helpers": 200,
                "extra_beam_width": 100,
                "extra_expansion_attempts": 2,
                "transition_floor_relaxation_enabled": True,
                "min_transition_floor": 0.20,
            }
        }
    }

    load_artifact_bundle.cache_clear()
    result = generate_playlist_ds(
        artifact_path=str(ARTIFACT_PATH),
        seed_track_id=case["track_id"],
        mode="dynamic",
        length=20,
        random_seed=42,
        overrides=overrides,
    )

    assert result is not None
    assert len(result.track_ids) >= 15, (
        f"Expected ≥15 tracks in relaxed dynamic playlist, got {len(result.track_ids)}"
    )
