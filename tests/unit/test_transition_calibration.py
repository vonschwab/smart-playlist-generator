"""Unit tests for the calibrated-sigmoid transition rescale (v6 sonic).

Replaces the legacy (x+1)/2 rescale, which compressed the real cosine band
[~0.14, 0.50] into [~0.57, 0.75] (good-vs-bad gap collapsed to ~8%). The
calibrated logistic spreads that band across (0,1) and restores discrimination.
"""
import math

import pytest

from src.playlist.transition_metrics import (
    _calibrate_transition_cos,
    is_broken_transition,
    resolve_transition_calib,
)

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


# --- variant-aware transition calibration ---------------------------------
# The realistic centered end->start cosine band differs by sonic embedding, so
# the sigmoid center/scale must track the ACTIVE variant or the rescale
# saturates. MERT band p1/p50/p99 ~ 0.12/0.26/0.51; MuQ (contrastive, hot)
# ~ 0.32/0.55/0.87. Derived on the full 41k via
# scripts/research/calibrate_transition_sigmoid.py.

def test_none_variant_maps_to_muq_default():
    # Post-SP-B: muq is the only registered variant; legacy/no-variant
    # artifacts get the muq band (pre-variant artifacts no longer exist).
    assert resolve_transition_calib(None) == (0.594, 0.092, 1.0)


def test_mert_variant_now_raises():
    # The mert band was removed with the MERT path (SP-B).
    with pytest.raises(ValueError, match="No transition calibration"):
        resolve_transition_calib("mert")


def test_muq_band_unchanged():
    assert resolve_transition_calib("muq") == (0.594, 0.092, 1.0)


def test_resolve_muq_uses_the_hot_band():
    center, scale, gain = resolve_transition_calib("muq")
    assert 0.55 < center < 0.65        # MuQ band midpoint, well above MERT's 0.32
    assert 0.08 < scale < 0.11
    assert gain == 1.0


def test_resolve_is_case_insensitive():
    assert resolve_transition_calib("MuQ") == resolve_transition_calib("muq")


def test_muq_band_avoids_saturation_that_mert_calib_would_cause():
    # WHY this exists: a MuQ median edge (cos ~0.55) rescaled through MERT's
    # center 0.32 saturates to ~0.98 (no discrimination). With the MuQ-aware
    # calib it lands mid-band, preserving the gradient.
    muq_c, muq_s, muq_g = resolve_transition_calib("muq")
    mid_under_muq = _calibrate_transition_cos(0.55, center=muq_c, scale=muq_s, gain=muq_g)
    mid_under_mert = _calibrate_transition_cos(0.55, center=0.32, scale=0.0625, gain=1.0)
    assert 0.30 < mid_under_muq < 0.70     # discriminating
    assert mid_under_mert > 0.95           # saturated — the bug we're preventing


def test_resolve_override_wins_for_tuning():
    assert resolve_transition_calib("muq", override=(0.40, 0.05)) == (0.40, 0.05, 1.0)
    assert resolve_transition_calib("muq", override=(0.40, 0.05, 0.8)) == (0.40, 0.05, 0.8)


def test_resolve_unknown_variant_with_no_override_raises():
    # A configured sonic space the calibration can't act on is a startup error,
    # not a silent fall-through to MERT's band.
    with pytest.raises(ValueError):
        resolve_transition_calib("some_future_embedding")
