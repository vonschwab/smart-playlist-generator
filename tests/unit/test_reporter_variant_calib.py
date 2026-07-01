"""Regression: the edge-report path must score T in the ACTIVE sonic variant's
calibration band, resolved from ``bundle.sonic_variant`` — NOT the hardcoded
MERT 0.32 default that ``build_transition_metric_context`` falls back to.

The bug (2026-07-01): ``compute_edge_scores_from_artifact`` passed the variant
name for the sonic-space transform (which ``resolve_sonic_variant`` normalizes,
losing mert/muq -> tower_pca) but never resolved the calib. Under MuQ that ran
the rescale sigmoid at MERT's center 0.32, saturating every reported edge to
~1.0 — the GUI Diagnostics panel showed mean≈0.99 / min≈0.86 with the true
worst edge hidden. The beam already resolved per-variant calib
(pier_bridge_builder.py:490); this guards the report path that builds its own
context. Fix: reporter.py resolves ``resolve_transition_calib(bundle.sonic_variant)``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.playlist import reporter
from src.playlist.transition_metrics import (
    TransitionMetricContext,
    resolve_transition_calib,
)
from src.playlist.pier_bridge.vec import _l2_normalize_rows


def _fake_bundle(variant: str) -> SimpleNamespace:
    X = _l2_normalize_rows(np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32))
    return SimpleNamespace(
        track_id_to_index={"a": 0, "b": 1},
        track_ids=["a", "b"],
        X_sonic=X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_smoothed=None,
        genre_vocab=[],
        sonic_variant=variant,
    )


def _spy_build(captured: dict):
    """Capture the calib kwargs the reporter passes, return a valid tiny context."""
    def _build(**kwargs):
        captured["calib"] = (
            kwargs.get("calib_center"),
            kwargs.get("calib_scale"),
            kwargs.get("calib_gain"),
        )
        X = _l2_normalize_rows(np.random.default_rng(1).standard_normal((2, 8)).astype(np.float32))
        return TransitionMetricContext(
            X_full=X, X_start=None, X_mid=None, X_end=None, X_sonic_norm=X,
            center_transitions=True,
            calib_center=kwargs["calib_center"],
            calib_scale=kwargs["calib_scale"],
            calib_gain=kwargs["calib_gain"],
        )
    return _build


@pytest.mark.parametrize("variant", ["muq"])
def test_reporter_resolves_calib_from_bundle_variant(monkeypatch, variant):
    captured: dict = {}
    monkeypatch.setattr(reporter, "load_artifact_bundle", lambda path: _fake_bundle(variant))
    monkeypatch.setattr(reporter, "build_transition_metric_context", _spy_build(captured))

    reporter.compute_edge_scores_from_artifact(
        tracks=[{"rating_key": "a"}, {"rating_key": "b"}],
        artifact_path="unused.npz",
        config_sonic_variant=variant,
        sonic_variant=variant,       # the (normalized) transform variant — must NOT drive calib
        center_transitions=True,
        transition_floor=0.2,
    )

    assert captured["calib"] == resolve_transition_calib(variant)


def test_muq_does_not_get_the_mert_default(monkeypatch):
    """The exact saturation bug this guards against: MuQ must land on its own
    hot band. Post-SP-B there's no MERT band left to silently fall back to —
    querying "mert" raises instead of resolving a stale default.
    """
    captured: dict = {}
    monkeypatch.setattr(reporter, "load_artifact_bundle", lambda path: _fake_bundle("muq"))
    monkeypatch.setattr(reporter, "build_transition_metric_context", _spy_build(captured))

    reporter.compute_edge_scores_from_artifact(
        tracks=[{"rating_key": "a"}, {"rating_key": "b"}],
        artifact_path="unused.npz",
        config_sonic_variant="muq",
        sonic_variant="muq",
        center_transitions=True,
    )

    center = captured["calib"][0]
    assert center == pytest.approx(0.594, abs=1e-6)

    with pytest.raises(ValueError, match="No transition calibration"):
        resolve_transition_calib("mert")
