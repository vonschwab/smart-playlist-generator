import json
import sqlite3

import numpy as np
import pytest

from src.playlist.bpm_loader import load_bpm_arrays


def _make_db(tmp_path, rows):
    """Build a minimal tracks DB with sonic_features blobs."""
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            sonic_features TEXT
        )
    """)
    for tid, bpm_info in rows.items():
        blob = json.dumps({"full": {"bpm_info": bpm_info}}) if bpm_info else None
        conn.execute(
            "INSERT INTO tracks (track_id, sonic_features) VALUES (?, ?)",
            (tid, blob),
        )
    conn.commit()
    conn.close()
    return db_path


def test_load_bpm_arrays_basic(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t2": {"primary_bpm": 70.0, "half_tempo_likely": True, "double_tempo_likely": False, "tempo_stability": 0.8},
    })
    track_ids = np.array(["t1", "t2"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    np.testing.assert_allclose(result["primary_bpm"], [120.0, 70.0])
    # t2 has half_tempo_likely → perceptual = 70 * 2 = 140
    np.testing.assert_allclose(result["perceptual_bpm"], [120.0, 140.0])
    np.testing.assert_allclose(result["tempo_stability"], [0.9, 0.8])


def test_load_bpm_arrays_missing_track_returns_nan(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
    })
    track_ids = np.array(["t1", "tX"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    assert result["perceptual_bpm"][0] == 120.0
    assert np.isnan(result["perceptual_bpm"][1])
    assert np.isnan(result["tempo_stability"][1])


def test_load_bpm_arrays_null_sonic_features_returns_nan(tmp_path):
    db_path = _make_db(tmp_path, {"t1": None})
    track_ids = np.array(["t1"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    assert np.isnan(result["perceptual_bpm"][0])


def test_load_bpm_arrays_preserves_order(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 100.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t2": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t3": {"primary_bpm": 140.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
    })
    track_ids = np.array(["t3", "t1", "t2"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    np.testing.assert_allclose(result["perceptual_bpm"], [140.0, 100.0, 120.0])


def test_load_bpm_arrays_double_tempo_halves(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 160.0, "half_tempo_likely": False, "double_tempo_likely": True, "tempo_stability": 0.9},
    })
    track_ids = np.array(["t1"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    np.testing.assert_allclose(result["perceptual_bpm"], [80.0])


def test_load_bpm_arrays_batches_large_track_id_sets(tmp_path, monkeypatch):
    from src.playlist import bpm_loader

    db_path = _make_db(
        tmp_path,
        {
            f"t{i}": {
                "primary_bpm": 100.0 + i,
                "half_tempo_likely": False,
                "double_tempo_likely": False,
                "tempo_stability": 0.9,
            }
            for i in range(5)
        },
    )
    monkeypatch.setattr(bpm_loader, "SQL_VARIABLE_BATCH_SIZE", 2)

    result = bpm_loader.load_bpm_arrays(
        np.array([f"t{i}" for i in range(5)], dtype=object),
        db_path=str(db_path),
    )

    np.testing.assert_allclose(result["perceptual_bpm"], [100.0, 101.0, 102.0, 103.0, 104.0])
