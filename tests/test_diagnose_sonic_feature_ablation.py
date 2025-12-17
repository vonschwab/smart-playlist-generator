import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_artifact(tmp_path: Path) -> Path:
    track_ids = np.array(["t1", "t2", "t3", "t4"])
    artist_keys = np.array(["a", "a", "b", "b"])
    # Two blocks: block0 informative, block1 mostly zeros
    block0 = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    block1 = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ]
    )
    X_sonic = np.hstack([block0, block1])
    X_sonic_start = X_sonic.copy()
    X_sonic_end = X_sonic.copy()
    genre_vocab = np.array(["g1"])
    X_genre_raw = np.zeros((4, 1), dtype=float)
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


def test_feature_ablation_json_contains_blocks(tmp_path):
    art_path = _make_artifact(tmp_path)
    blocks_path = tmp_path / "blocks.json"
    blocks_path.write_text(json.dumps({"blocks": {"mfcc": [0, 1], "rest": [2, 3]}}), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/diagnose_sonic_feature_ablation.py",
        "--artifact",
        str(art_path),
        "--n",
        "20",
        "--seed",
        "0",
        "--blocks",
        str(blocks_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out_files = sorted(Path("diagnostics").glob("sonic_feature_ablation_*.json"))
    assert out_files, "No ablation report written"
    data = json.loads(out_files[-1].read_text())
    assert "baseline" in data
    assert "blocks" in data
    assert "mfcc" in data["blocks"]
    base_p50 = data["baseline"]["stats"]["p50"]
    mfcc_p50 = data["blocks"]["mfcc"]["stats"]["p50"]
    # Ablating informative block should change p50 vs baseline
    assert mfcc_p50 != base_p50
