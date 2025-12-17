import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

import scripts.batch_eval_sonic_variant as bev


def _make_artifact(tmp_path: Path) -> Path:
    track_ids = np.array([f"t{i}" for i in range(5)])
    artist_keys = np.array([f"a{i%2}" for i in range(5)])
    X_sonic = np.random.rand(5, 3)
    X_genre_raw = np.random.rand(5, 2)
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


def test_load_seeds_from_artifact(tmp_path):
    art = _make_artifact(tmp_path)
    seeds = bev.load_seeds(art, n_seeds=3, seed=0, seeds_file=None)
    assert len(seeds) == 3


def test_write_csv_and_md(tmp_path, monkeypatch):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    # stub variant runner results
    rows_by_variant = {
        "raw": [
            {"seed": "t1", "variant": "raw", "S_spread": 0.01, "corr_S_T": 0.9, "corr_S_Tc": 0.2, "min_transition": 0.5}
        ],
        "z": [
            {"seed": "t1", "variant": "z", "S_spread": 0.2, "corr_S_T": 0.4, "corr_S_Tc": 0.1, "min_transition": 0.4}
        ],
    }
    bev.write_csv(diagnostics / "raw.csv", rows_by_variant["raw"])
    bev.write_csv(diagnostics / "z.csv", rows_by_variant["z"])
    bev.summarize_markdown(rows_by_variant, diagnostics / "AB_SONIC_VARIANT_REPORT.md")
    assert (diagnostics / "raw.csv").exists()
    assert (diagnostics / "z.csv").exists()
    md = (diagnostics / "AB_SONIC_VARIANT_REPORT.md").read_text()
    assert "Variant: raw" in md
    assert "Variant: z" in md


def test_recompute_edges_produces_similarity(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    # simple artifact with two tracks
    track_ids = ["t1", "t2"]
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    art = tmp_path / "art.npz"
    np.savez(
        art,
        track_ids=np.array(track_ids),
        artist_keys=np.array(["a", "b"]),
        X_sonic=X,
        X_sonic_start=X,
        X_sonic_end=X,
        X_genre_raw=np.zeros((2, 1)),
        X_genre_smoothed=np.zeros((2, 1)),
        genre_vocab=np.array(["g"]),
    )
    edges = bev.recompute_edges(art, track_ids, "raw")
    assert edges and "S" in edges[0]


def test_fixed_order_and_export(tmp_path):
    # Build artifact and metadata db
    art = _make_artifact(tmp_path)
    db_path = tmp_path / "meta.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY, file_path TEXT, artist TEXT, title TEXT)")
    for tid in np.load(art)["track_ids"]:
        conn.execute(
            "INSERT INTO tracks(track_id, file_path, artist, title) VALUES (?,?,?,?)",
            (str(tid), f"/music/{tid}.flac", f"A{tid}", f"T{tid}"),
        )
    conn.commit()
    conn.close()

    diag_dir = tmp_path / "diagnostics"
    cmd = [
        sys.executable,
        str(Path("scripts/batch_eval_sonic_variant.py")),
        "--artifact",
        str(art),
        "--n-seeds",
        "2",
        "--seed",
        "0",
        "--length",
        "3",
        "--variants",
        "raw,z",
        "--eval-mode",
        "fixed_order",
        "--export-m3u-dir",
        str(tmp_path / "m3u"),
        "--db",
        str(db_path),
    ]
    res = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    # Check outputs in diagnostics (under repo root) or tmp? script writes relative to cwd (repo)
    assert (Path("diagnostics") / "ab_sonic_variant_raw.csv").exists()
    assert (Path("diagnostics") / "AB_SONIC_VARIANT_REPORT.md").exists()
    # Export files
    m3us = list((tmp_path / "m3u").glob("*.m3u8"))
    assert m3us, "No M3U files written"
    diffs = list((tmp_path / "m3u").glob("*__diff.md"))
    assert diffs, "No diff reports written"
