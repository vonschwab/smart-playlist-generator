"""Unit tests for the calibrated-sigmoid transition rescale (v6 sonic).

Replaces the legacy (x+1)/2 rescale, which compressed the real cosine band
[~0.14, 0.50] into [~0.57, 0.75] (good-vs-bad gap collapsed to ~8%). The
calibrated logistic spreads that band across (0,1) and restores discrimination.
"""
import math

from src.playlist.transition_metrics import _calibrate_transition_cos, is_broken_transition

# Provisional params (band midpoint center, gain/scale = 16 → p1→~0.05, p99→~0.95).
P = dict(center=0.32, scale=0.0625, gain=1.0)


def test_band_is_spread_across_unit_interval():
    lo = _calibrate_transition_cos(0.138, **P)   # p1 of the real centered band
    hi = _calibrate_transition_cos(0.501, **P)   # p99 of the real centered band
    assert lo < 0.12 and hi > 0.88               # band maps to ~[0.05, 0.95]
    assert 0.0 < lo < hi < 1.0                    # strictly inside (0,1): no clip ties


def test_restores_good_vs_bad_gap():
    bad = _calibrate_transition_cos(0.151, **P)   # the Yuji (bad) edge cosine
    good = _calibrate_transition_cos(0.260, **P)  # the Beach House (good) edge cosine
    rel_gap = (good - bad) / bad
    assert rel_gap > 0.40                          # legacy (x+1)/2 gives ~0.08


def test_monotonic_and_finite():
    xs = [-0.2, 0.0, 0.14, 0.27, 0.50, 0.71]
    ys = [_calibrate_transition_cos(x, **P) for x in xs]
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))
    assert all(math.isfinite(y) for y in ys)


def test_nan_passthrough():
    assert math.isnan(_calibrate_transition_cos(float("nan"), **P))


# --- transition hard-gate removal (roam-only) -----------------------------
# Roam shapes via soft corridor penalty + worst-edge minimax, never elimination.
# is_broken_transition must NOT reject on a low T anymore; only the -0.5
# catastrophic anti-alignment safety remains.

def test_low_T_is_not_broken():
    # A weak (but not anti-aligned) edge: under the old gate this was rejected
    # (T < transition_floor); under roam it must pass — the score, not a gate,
    # demotes it.
    edge = {"T": 0.05, "T_centered_cos": 0.30}
    assert is_broken_transition(edge, transition_floor=0.20, centered_cos_floor=-0.5) is False


def test_anti_alignment_still_gates():
    edge = {"T": 0.40, "T_centered_cos": -0.6}
    assert is_broken_transition(edge, transition_floor=0.20, centered_cos_floor=-0.5) is True


def test_within_anti_alignment_safety_passes():
    edge = {"T": 0.05, "T_centered_cos": -0.3}
    assert is_broken_transition(edge, transition_floor=0.20, centered_cos_floor=-0.5) is False
