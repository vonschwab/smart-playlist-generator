import numpy as np
from src.playlist.pier_bridge.corridors import geodesic_detour, corridor_penalty, band_deviation


def test_detour_zero_on_geodesic():
    # a=0,b=2; node 1 lies on the shortest path (d_a=1,d_b=1, geodesic=2 -> detour 0).
    d_a = np.array([0.0, 1.0, 2.0, 5.0])
    d_b = np.array([2.0, 1.0, 0.0, 4.0])
    det = geodesic_detour(d_a, d_b, pier_b=2)
    assert det[1] == 0.0           # on the path
    assert det[3] > 0.0            # off the path (5+4-2)


def test_detour_unreachable_is_inf():
    d_a = np.array([0.0, np.inf, 2.0])
    d_b = np.array([2.0, 1.0, 0.0])
    det = geodesic_detour(d_a, d_b, pier_b=2)
    assert np.isinf(det[1])


def test_corridor_penalty_free_inside_then_smooth():
    dev = np.array([0.0, 0.5, 1.0, 3.0])
    p = corridor_penalty(dev, width=1.0, slope=2.0)
    assert p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0   # within width => free
    assert p[3] > 0.0 and np.isfinite(p[3])              # beyond => smooth, finite


def test_corridor_penalty_unreachable_is_finite():
    dev = np.array([np.inf])
    p = corridor_penalty(dev, width=1.0, slope=1.0)
    assert np.isfinite(p[0]) and p[0] > 0.0


def test_band_deviation():
    vals = np.array([-1.0, 0.0, 0.5, 2.0])
    dev = band_deviation(vals, lo=0.0, hi=1.0)
    assert list(dev) == [1.0, 0.0, 0.0, 1.0]
