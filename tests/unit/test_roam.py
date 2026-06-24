"""Roam corridors: per-segment deviation assembly (sonic geodesic + energy band)."""
import numpy as np

from src.playlist.pier_bridge.roam import segment_sonic_detour, energy_band_deviation


def test_sonic_detour_is_global_indexed_and_zero_on_path():
    # Cosine distance => directions matter. Lay nodes on an arc 0..60deg (the
    # geodesic A->B), with an off-arc outlier (200deg) and one node (100deg) that
    # is NOT part of this segment.
    ang = np.deg2rad([0.0, 20.0, 40.0, 60.0, 200.0, 100.0])  # nodes 0..5
    X = np.c_[np.cos(ang), np.sin(ang)]                       # already unit
    det = segment_sonic_detour(0, 3, [1, 2, 4], X, k=2, mutual_proximity=False)
    assert det.shape[0] == 6
    assert det[0] == 0.0 and det[3] == 0.0          # piers are on their own geodesic
    assert np.isinf(det[5])                          # 100deg node not in this segment
    assert np.isfinite(det[1])                       # 20deg on-arc candidate reachable
    assert det[4] >= det[1]                          # 200deg outlier detours at least as far


def test_sonic_detour_no_interior_candidates():
    Xn = np.eye(3)
    det = segment_sonic_detour(0, 1, [], Xn, k=2, mutual_proximity=False)
    assert det[0] == 0.0 and det[1] == 0.0 and np.isinf(det[2])


def test_energy_band_deviation_uses_seed_range():
    energy = np.array([0.0, 0.5, 1.0, 2.0, -1.0])
    # seeds 1 and 2 define band [0.5, 1.0]
    dev = energy_band_deviation(energy, seed_indices=[1, 2])
    assert dev[1] == 0.0 and dev[2] == 0.0      # in band
    assert dev[3] == 1.0                          # 2.0 is 1.0 above hi
    assert dev[4] == 1.5                          # -1.0 is 1.5 below lo
    assert dev[0] == 0.5                          # 0.0 is 0.5 below lo


def test_energy_band_deviation_none_when_no_energy():
    assert energy_band_deviation(None, [0, 1]) is None
