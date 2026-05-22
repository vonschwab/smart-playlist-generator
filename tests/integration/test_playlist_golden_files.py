"""Golden file tests for deterministic synthetic DS playlist generation."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.playlist.ds_pipeline_runner import generate_playlist_ds

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden_playlists"


def _load_golden(filename: str) -> Dict[str, Any] | None:
    path = GOLDEN_DIR / filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_golden(filename: str, data: Dict[str, Any]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _ds_overrides(artifact_path: Path) -> Dict[str, Any]:
    return {
        "artifact_path": str(artifact_path),
        "tower_weights": {"rhythm": 0.20, "timbre": 0.50, "harmony": 0.30},
        "transition_weights": {"rhythm": 0.40, "timbre": 0.35, "harmony": 0.25},
        "embedding": {
            "sonic_components": 16,
            "genre_components": 16,
            "sonic_weight": 0.60,
            "genre_weight": 0.40,
        },
        "candidate_pool": {
            "similarity_floor": 0.00,
            "min_sonic_similarity_dynamic": 0.00,
            "min_sonic_similarity_narrow": 0.00,
            "max_pool_size": 80,
            "max_artist_fraction": 0.20,
            "broad_filters": [],
        },
        "scoring": {
            "alpha": 0.55,
            "beta": 0.55,
            "gamma": 0.04,
            "alpha_schedule": "constant",
        },
        "constraints": {
            "min_gap": 1,
            "hard_floor": False,
            "transition_floor": 0.00,
            "center_transitions": True,
        },
        "pier_bridge": {"bridge_floor": 0.00},
        "repair": {"enabled": True, "max_iters": 3, "max_edges": 3},
    }


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("golden_artifacts")
    rng = np.random.default_rng(123)

    track_count = 100
    sonic_dim = 32
    genre_dim = 20

    track_ids = np.array([f"golden_track_{i:04d}" for i in range(track_count)])
    artist_keys = np.array([f"golden_artist_{i % 20:02d}" for i in range(track_count)])
    track_artists = np.array([f"Golden Artist {i % 20}" for i in range(track_count)])
    track_titles = np.array([f"Golden Song {i}" for i in range(track_count)])
    durations_ms = rng.integers(120000, 300000, size=track_count)

    x_sonic = rng.normal(size=(track_count, sonic_dim))
    x_genre_raw = rng.random(size=(track_count, genre_dim))
    x_genre_raw[x_genre_raw < 0.8] = 0.0
    x_genre_smoothed = np.clip(x_genre_raw + rng.normal(scale=0.05, size=x_genre_raw.shape), 0, 1)
    genre_vocab = np.array([f"golden_genre_{i:02d}" for i in range(genre_dim)])

    artifact_path = tmpdir / "synthetic_golden_artifact.npz"
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        durations_ms=durations_ms,
        X_sonic=x_sonic,
        X_sonic_start=x_sonic + rng.normal(scale=0.1, size=x_sonic.shape),
        X_sonic_mid=x_sonic + rng.normal(scale=0.05, size=x_sonic.shape),
        X_sonic_end=x_sonic + rng.normal(scale=0.1, size=x_sonic.shape),
        X_genre_raw=x_genre_raw,
        X_genre_smoothed=x_genre_smoothed,
        genre_vocab=genre_vocab,
    )
    return artifact_path


def _extract_playlist_data(result, mode: str) -> Dict[str, Any]:
    track_ids = list(result.track_ids)
    unique_artists = result.metrics.get("distinct_artists")
    artist_diversity = None
    if unique_artists is not None and track_ids:
        artist_diversity = unique_artists / len(track_ids)
    return {
        "track_ids": track_ids,
        "track_count": len(track_ids),
        "metrics": {
            "mean_transition": result.metrics.get("mean_transition"),
            "min_transition": result.metrics.get("min_transition"),
            "artist_diversity": artist_diversity,
            "unique_artists": unique_artists,
        },
        "mode": mode,
        "seed_info": {"seed_track_id": result.requested.get("seed_track_id")},
    }


def _compare_playlists(result: Dict[str, Any], golden: Dict[str, Any], tolerance: float = 0.01) -> tuple[bool, str]:
    if result["track_ids"] != golden["track_ids"]:
        return False, f"Track IDs differ:\nGot: {result['track_ids']}\nExpected: {golden['track_ids']}"
    if result["track_count"] != golden["track_count"]:
        return False, f"Track count differs: {result['track_count']} vs {golden['track_count']}"
    for metric_name, result_value in result["metrics"].items():
        golden_value = golden["metrics"].get(metric_name)
        if result_value is None or golden_value is None:
            continue
        if abs(result_value - golden_value) > tolerance:
            return False, f"Metric '{metric_name}' differs: {result_value:.4f} vs {golden_value:.4f}"
    return True, ""


@pytest.mark.integration
@pytest.mark.golden
@pytest.mark.parametrize(
    ("mode", "seed_track_id", "filename"),
    [
        ("narrow", "golden_track_0000", "synthetic_narrow_seed_0000.json"),
        ("dynamic", "golden_track_0001", "synthetic_dynamic_seed_0001.json"),
        ("dynamic", "golden_track_0010", "synthetic_genre_style_seed_0010.json"),
        ("discover", "golden_track_0020", "synthetic_discover_seed_0020.json"),
    ],
)
def test_synthetic_playlist_golden(mode, seed_track_id, filename, synthetic_artifact, request):
    result = generate_playlist_ds(
        artifact_path=str(synthetic_artifact),
        seed_track_id=seed_track_id,
        mode=mode,
        length=10,
        random_seed=42,
        overrides=_ds_overrides(synthetic_artifact),
    )
    data = _extract_playlist_data(result, mode)

    if request.config.getoption("--generate-golden", default=False):
        _save_golden(filename, data)

    golden = _load_golden(filename)
    assert golden is not None, f"Golden file not found: {filename}. Run with --generate-golden to create it."

    matches, error = _compare_playlists(data, golden)
    assert matches, f"Playlist differs from golden file:\n{error}"
