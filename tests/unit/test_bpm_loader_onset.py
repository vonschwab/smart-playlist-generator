"""load_bpm_arrays must return an onset_rate array aligned to track_ids."""
import sqlite3
import json
import numpy as np
import pytest

from src.playlist.bpm_loader import load_bpm_arrays


@pytest.fixture
def tiny_db(tmp_path):
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, sonic_features TEXT)")
    feat = {"full": {"bpm_info": {"primary_bpm": 120.0, "tempo_stability": 0.9,
                                  "half_tempo_likely": False, "double_tempo_likely": False},
                     "rhythm": {"onset_rate": 2.5}}}
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("t1", json.dumps(feat)))
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("t2", None))  # missing features
    conn.commit()
    conn.close()
    return str(db)


def test_onset_rate_loaded_and_aligned(tiny_db):
    arrs = load_bpm_arrays(np.array(["t1", "t2"]), db_path=tiny_db)
    assert "onset_rate" in arrs
    assert arrs["onset_rate"][0] == pytest.approx(2.5)
    assert np.isnan(arrs["onset_rate"][1])  # NaN for missing
