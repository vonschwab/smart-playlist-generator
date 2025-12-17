import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.diagnose_sonic_geometry import saturated_flag, summary_stats


def _make_artifact(tmp_path: Path) -> Path:
    """Create a tiny synthetic artifact NPZ for diagnostics."""
    track_ids = np.array(["t1", "t2", "t3", "t4"])
    artist_keys = np.array(["a", "a", "b", "b"])
    # Sonic matrix with modest variance so raw vs z differ
    X_sonic = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.8, 0.0],
            [0.2, 0.2, 0.0],
        ],
        dtype=float,
    )
    # Provide start/end identical to keep transitions available
    X_sonic_start = X_sonic.copy()
    X_sonic_end = X_sonic.copy()
    # Simple genre one-hot
    genre_vocab = np.array(["ambient", "punk", "rock"])
    X_genre_raw = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    X_genre_smoothed = X_genre_raw.copy()
    art_path = tmp_path / "artifact.npz"
    np.savez(
        art_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=np.array(["A", "A", "B", "B"]),
        track_titles=np.array(["ta", "tb", "tc", "td"]),
        X_sonic=X_sonic,
        X_sonic_start=X_sonic_start,
        X_sonic_end=X_sonic_end,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )
    return art_path


def test_saturation_flag_on_constant_matrix():
    arr = np.ones((10,))
    stats = summary_stats(arr)
    assert saturated_flag(stats, thresh=0.005) is True


def test_script_runs_and_writes_json(tmp_path):
    art_path = _make_artifact(tmp_path)
    cmd = [
        sys.executable,
        "scripts/diagnose_sonic_geometry.py",
        "--artifact",
        str(art_path),
        "--mode",
        "random",
        "--n",
        "20",
        "--seed",
        "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    # JSON report should exist in diagnostics
    diag_dir = Path("diagnostics")
    reports = sorted(diag_dir.glob("sonic_geometry_*.json"))
    assert reports, "No diagnostics JSON written"
    latest = reports[-1]
    data = json.loads(latest.read_text())
    assert data["pair_count"] == 20
    assert "S_sonic_raw" in data["metrics"]
    assert "S_sonic_z" in data["metrics"]
    # z-score cosine should differ from raw cosine mean for this synthetic matrix
    raw_mean = data["metrics"]["S_sonic_raw"]["mean"]
    z_mean = data["metrics"]["S_sonic_z"]["mean"]
    assert raw_mean != z_mean


def test_cross_groups_mode_writes_three_distributions(tmp_path):
    art_path = _make_artifact(tmp_path)
    group_a = tmp_path / "group_a.json"
    group_b = tmp_path / "group_b.json"
    group_a.write_text(json.dumps([{"track_id": "t1"}, {"track_id": "t2"}]), encoding="utf-8")
    group_b.write_text(json.dumps([{"track_id": "t3"}, {"track_id": "t4"}]), encoding="utf-8")
    before = set(Path("diagnostics").glob("sonic_geometry_*.json")) if Path("diagnostics").exists() else set()
    cmd = [
        sys.executable,
        "scripts/diagnose_sonic_geometry.py",
        "--artifact",
        str(art_path),
        "--mode",
        "cross_groups",
        "--group-a",
        str(group_a),
        "--group-b",
        str(group_b),
        "--n",
        "10",
        "--baseline-n",
        "10",
        "--seed",
        "1",
        "--no-pca",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    after = set(Path("diagnostics").glob("sonic_geometry_*.json"))
    new_reports = sorted(after - before) if after else []
    if new_reports:
        latest = new_reports[-1]
    else:
        latest = sorted(after)[-1]
    data = json.loads(latest.read_text())
    assert "modes" in data
    for key in ["within_a", "within_b", "across"]:
        assert key in data["modes"], f"{key} missing in cross_groups report"
