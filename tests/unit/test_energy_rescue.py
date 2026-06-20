import numpy as np
from src.playlist.energy_rescue import select_energy_rescue

def test_spans_arousal_range():
    arousal = np.array([0.0, -2.0, 2.0, 1.0, -1.0, 0.5])
    src = [0, 1, 2, 3, 4, 5]
    picked = select_energy_rescue(arousal, src, k_energy=3)
    vals = sorted(arousal[i] for i in picked)
    assert len(picked) == 3
    assert vals[0] == -2.0 and vals[-1] == 2.0   # endpoints of the range are represented

def test_returns_all_when_source_small():
    arousal = np.array([0.0, 1.0, -1.0])
    assert sorted(select_energy_rescue(arousal, [0, 1, 2], k_energy=5)) == [0, 1, 2]

def test_zero_k_and_empty_source():
    arousal = np.array([0.0, 1.0])
    assert select_energy_rescue(arousal, [0, 1], k_energy=0) == []
    assert select_energy_rescue(arousal, [], k_energy=3) == []

def test_skips_nan_arousal():
    arousal = np.array([0.0, np.nan, 2.0])
    picked = select_energy_rescue(arousal, [0, 1, 2], k_energy=3)
    assert 1 not in picked and set(picked) == {0, 2}
