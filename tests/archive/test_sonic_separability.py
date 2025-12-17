import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_artifact(tmp_path: Path) -> Path:
    track_ids = np.array([f"t{i}" for i in range(6)])
    artist_keys = np.array(["a", "a", "b", "b", "c", "c"])
    # sonic with two clusters
    X_sonic = np.array(
        [
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0, 1, 0],
            [0, 0.9, 0.1],
            [0.2, 0.2, 1.0],
            [0.1, 0.1, 0.9],
        ],
        dtype=float,
    )
    X_genre_raw = np.zeros((6, 2))
    X_genre_smoothed = X_genre_raw.copy()
    genre_vocab = np.array(["g1", "g2"])
    art = tmp_path / "art.npz"
    np.savez(
        art,
        track_ids=track_ids,
        artist_keys=artist_keys,
        X_sonic=X_sonic,
        X_sonic_start=X_sonic,
        X_sonic_end=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )
    return art


def _make_group(path: Path, ids):
    path.write_text(json.dumps([{"track_id": tid} for tid in ids]), encoding="utf-8")


def test_separability_script_creates_outputs(tmp_path):
    art = _make_artifact(tmp_path)
    ga = tmp_path / "ga.json"
    gb = tmp_path / "gb.json"
    _make_group(ga, ["t0", "t1"])
    _make_group(gb, ["t2", "t3"])
    cmd = [
        sys.executable,
        "scripts/sonic_separability.py",
        "--artifact",
        str(art),
        "--group-a",
        str(ga),
        "--group-b",
        str(gb),
        "--n",
        "50",
        "--seed",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    jsons = sorted(Path("diagnostics").glob("sonic_separability_*.json"))
    assert jsons, "No separability JSON produced"
    data = json.loads(jsons[-1].read_text())
    assert "variants" in data
    for var in ["raw", "z"]:
        assert var in data["variants"]
