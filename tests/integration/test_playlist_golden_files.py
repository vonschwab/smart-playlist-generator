"""Golden file tests for playlist generation.

These tests lock the current behavior of the playlist generator by saving
reference outputs ("golden files") and verifying that future runs produce
identical results. This ensures refactoring doesn't introduce regressions.

Usage:
    # First run: generates golden files
    pytest tests/integration/test_playlist_golden_files.py --generate-golden

    # Subsequent runs: verify against golden files
    pytest tests/integration/test_playlist_golden_files.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import Config
from src.playlist_generator import PlaylistGenerator

# Path to golden file directory
GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden_playlists"


def _load_golden(filename: str) -> Dict[str, Any]:
    """Load a golden reference file."""
    path = GOLDEN_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_golden(filename: str, data: Dict[str, Any]):
    """Save data as a golden reference file."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _extract_playlist_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparable data from playlist result.

    We extract:
    - Track IDs (exact match required)
    - Metrics (within tolerance)
    - Configuration used
    """
    return {
        "track_ids": result.get("track_ids", []),
        "track_count": len(result.get("track_ids", [])),
        "playlist_name": result.get("name", ""),
        "metrics": {
            "mean_transition": result.get("metrics", {}).get("mean_transition_score"),
            "min_transition": result.get("metrics", {}).get("min_transition_score"),
            "artist_diversity": result.get("metrics", {}).get("artist_diversity"),
            "unique_artists": result.get("metrics", {}).get("unique_artists"),
            "total_duration_minutes": result.get("metrics", {}).get("total_duration_minutes"),
        },
        "mode": result.get("mode", "unknown"),
        "seed_info": result.get("seed_info", {}),
    }


def _compare_playlists(result: Dict[str, Any], golden: Dict[str, Any], tolerance: float = 0.01) -> tuple[bool, str]:
    """Compare playlist result against golden file.

    Returns:
        (matches, error_message) - matches is True if identical within tolerance
    """
    # Exact track ID match required
    if result["track_ids"] != golden["track_ids"]:
        return False, f"Track IDs differ:\nGot: {result['track_ids'][:5]}...\nExpected: {golden['track_ids'][:5]}..."

    # Track count must match
    if result["track_count"] != golden["track_count"]:
        return False, f"Track count differs: {result['track_count']} vs {golden['track_count']}"

    # Metrics within tolerance
    for metric_name, result_value in result["metrics"].items():
        golden_value = golden["metrics"].get(metric_name)
        if result_value is None or golden_value is None:
            continue

        if abs(result_value - golden_value) > tolerance:
            return False, f"Metric '{metric_name}' differs: {result_value:.4f} vs {golden_value:.4f}"

    return True, ""


@pytest.fixture(scope="module")
def generator():
    """Create playlist generator instance with test config."""
    # Use test config if it exists, otherwise use main config
    config_path = Path("config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found - skipping golden file tests")

    gen = PlaylistGenerator(config_path=str(config_path))

    # Verify artifact path exists
    artifact_path = gen.config.get_ds_artifact_path()
    if not Path(artifact_path).exists():
        pytest.skip(f"Artifact file not found: {artifact_path} - run build_beat3tower_artifacts.py first")

    return gen


@pytest.mark.integration
@pytest.mark.golden
class TestGoldenFiles:
    """Golden file tests for playlist generation."""

    def test_narrow_mode_bill_evans(self, generator, request):
        """Test narrow mode with Bill Evans generates consistent results."""
        # Use fixed seed for reproducibility
        result = generator.create_playlist_for_artist(
            artist="Bill Evans",
            num_tracks=30,
            mode="narrow",
            random_seed=42,
        )

        # Extract comparable data
        data = _extract_playlist_data(result)

        # Check against golden file
        golden_file = "narrow_mode_bill_evans.json"
        golden = _load_golden(golden_file)

        if request.config.getoption("--generate-golden", default=False):
            _save_golden(golden_file, data)
            pytest.skip(f"Generated golden file: {golden_file}")

        if golden is None:
            pytest.fail(f"Golden file not found: {golden_file}. Run with --generate-golden to create it.")

        matches, error = _compare_playlists(data, golden)
        assert matches, f"Playlist differs from golden file:\n{error}"

    def test_dynamic_mode_radiohead(self, generator, request):
        """Test dynamic mode with Radiohead generates consistent results."""
        result = generator.create_playlist_for_artist(
            artist="Radiohead",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
        )

        data = _extract_playlist_data(result)
        golden_file = "dynamic_mode_radiohead.json"
        golden = _load_golden(golden_file)

        if request.config.getoption("--generate-golden", default=False):
            _save_golden(golden_file, data)
            pytest.skip(f"Generated golden file: {golden_file}")

        if golden is None:
            pytest.fail(f"Golden file not found: {golden_file}. Run with --generate-golden to create it.")

        matches, error = _compare_playlists(data, golden)
        assert matches, f"Playlist differs from golden file:\n{error}"

    def test_genre_mode_ambient(self, generator, request):
        """Test genre-based playlist generation consistency."""
        result = generator.create_playlist_for_genre(
            genre="ambient",
            num_tracks=30,
            mode="dynamic",
            random_seed=42,
        )

        data = _extract_playlist_data(result)
        golden_file = "genre_mode_ambient.json"
        golden = _load_golden(golden_file)

        if request.config.getoption("--generate-golden", default=False):
            _save_golden(golden_file, data)
            pytest.skip(f"Generated golden file: {golden_file}")

        if golden is None:
            pytest.fail(f"Golden file not found: {golden_file}. Run with --generate-golden to create it.")

        matches, error = _compare_playlists(data, golden)
        assert matches, f"Playlist differs from golden file:\n{error}"

    def test_discover_mode_jazz(self, generator, request):
        """Test discover mode generates consistent results."""
        result = generator.create_playlist_for_genre(
            genre="jazz",
            num_tracks=30,
            mode="discover",
            random_seed=42,
        )

        data = _extract_playlist_data(result)
        golden_file = "discover_mode_jazz.json"
        golden = _load_golden(golden_file)

        if request.config.getoption("--generate-golden", default=False):
            _save_golden(golden_file, data)
            pytest.skip(f"Generated golden file: {golden_file}")

        if golden is None:
            pytest.fail(f"Golden file not found: {golden_file}. Run with --generate-golden to create it.")

        matches, error = _compare_playlists(data, golden)
        assert matches, f"Playlist differs from golden file:\n{error}"


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--generate-golden",
        action="store_true",
        default=False,
        help="Generate golden reference files instead of comparing against them",
    )
