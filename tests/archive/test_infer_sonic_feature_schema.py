import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

import scripts.infer_sonic_feature_schema as schema_script


def _write_artifact(tmp_path: Path) -> Path:
    track_ids = np.array([f"t{i}" for i in range(3)])
    artist_keys = np.array([f"a{i}" for i in range(3)])
    X_sonic = np.array(
        [
            [1.0, 0.0, 0.1, 0.2, 60.0, 2000.0],
            [2.0, 1.0, 0.2, 0.3, 65.0, 1900.0],
            [3.0, 2.0, 0.3, 0.4, 70.0, 2100.0],
        ],
        dtype=float,
    )
    genre_vocab = np.array(["g1"])
    X_genre = np.zeros((3, 1), dtype=float)
    path = tmp_path / "artifact.npz"
    np.savez(
        path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=np.array(["Artist 0", "Artist 1", "Artist 2"]),
        track_titles=np.array(["Track 0", "Track 1", "Track 2"]),
        X_sonic=X_sonic,
        X_sonic_start=X_sonic,
        X_sonic_end=X_sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=genre_vocab,
        sonic_feature_names=np.array(["mfcc_01", "mfcc_02", "chroma_01", "chroma_02", "tempo", "spectral_centroid"]),
        sonic_feature_units=np.array(["mfcc", "mfcc", "chroma", "chroma", "bpm", "hz"]),
    )
    return path


def _write_db(tmp_path: Path, track_ids: np.ndarray) -> Path:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, file_path TEXT)")
    for tid in track_ids:
        cur.execute("INSERT INTO tracks(track_id, file_path) VALUES (?, ?)", (str(tid), str(tid)))
    conn.commit()
    conn.close()
    return db_path


class DummyAnalyzer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_similarity_features(self, path: str):
        idx = int(str(path).lstrip("t"))
        centroid_values = [2000.0, 1900.0, 2100.0]
        return {
            "average": {
                "mfcc_mean": [1.0, 1.0],
                "chroma_mean": [0.2, 0.2],
                "bpm": 60.0 + idx * 5.0,
                "spectral_centroid": centroid_values[idx],
                "rms_energy": 0.1 + idx * 0.01,
                "zero_crossing_rate": 0.01 + idx * 0.001,
            }
        }


def test_infer_schema_identifies_tempo_and_centroid(tmp_path: Path, monkeypatch):
    artifact = _write_artifact(tmp_path)
    db_path = _write_db(tmp_path, np.array(["t0", "t1", "t2"]))
    group = tmp_path / "group.json"
    group.write_text(json.dumps({"tracks": [{"track_id": "t0"}, {"track_id": "t1"}, {"track_id": "t2"}]}), encoding="utf-8")
    monkeypatch.setattr(schema_script, "LibrosaAnalyzer", DummyAnalyzer)
    initial_reports = list(Path("diagnostics").glob("sonic_feature_schema_*.json"))
    monkeypatch.setattr(sys, "argv", ["infer_schema", "--artifact", str(artifact), "--group", str(group), "--db", str(db_path), "--n", "3", "--seed", "0"])
    schema_script.main()
    reports = sorted(Path("diagnostics").glob("sonic_feature_schema_*.json"))
    new_reports = [p for p in reports if p not in initial_reports]
    assert new_reports, "Schema report not created"
    report = json.loads(new_reports[-1].read_text())
    dims = report["dimensions"]
    assert dims[4]["best_label"] == "bpm"
    assert dims[5]["best_label"] == "spectral_centroid"


def test_dim_label_fallback():
    from src.similarity.sonic_schema import dim_label

    class FakeBundle:
        X_sonic = np.zeros((3, 4))

    assert dim_label(FakeBundle, 2) == "dim_02"
