"""The opt-in audit path and the live beam scorer must compute the SAME T.

After the calibrated-sigmoid rescale was single-sourced into
vec._calibrate_transition_cos, both score_transition_edge (live beam) and the
audit's _compute_transition_score_raw_and_transformed (via the PierBridgeConfig
wrapper) route through that one function. This guards the latent divergence the
old duplicate (x+1)/2 copies would have caused post-fix.
"""
import numpy as np

from src.playlist.transition_metrics import TransitionMetricContext, score_transition_edge
from src.playlist.pier_bridge.config import (
    PierBridgeConfig,
    _compute_transition_score_raw_and_transformed,
)
from src.playlist.pier_bridge.vec import _l2_normalize_rows

CALIB = dict(calib_center=0.32, calib_scale=0.0625, calib_gain=1.0)


def _ctx(Xn):
    return TransitionMetricContext(
        X_full=Xn, X_start=Xn, X_mid=Xn, X_end=Xn, X_sonic_norm=Xn,
        center_transitions=True,
        weight_end_start=0.7, weight_mid_mid=0.15, weight_full_full=0.15,
        **CALIB,
    )


def _cfg():
    return PierBridgeConfig(
        center_transitions=True,
        weight_end_start=0.7, weight_mid_mid=0.15, weight_full_full=0.15,
        transition_calib_center=0.32, transition_calib_scale=0.0625, transition_calib_gain=1.0,
    )


def test_audit_transformed_equals_beam_T():
    rng = np.random.default_rng(0)
    Xn = _l2_normalize_rows(rng.standard_normal((8, 12)).astype(np.float32))
    ctx = _ctx(Xn)
    cfg = _cfg()
    for a, b in [(0, 1), (2, 4), (3, 5), (6, 7)]:
        _raw, transformed = _compute_transition_score_raw_and_transformed(
            a, b, ctx.X_full, ctx.X_start, ctx.X_mid, ctx.X_end, cfg
        )
        beam_t = score_transition_edge(ctx, a, b)["T"]
        assert abs(float(transformed) - float(beam_t)) < 1e-9, (a, b, transformed, beam_t)
