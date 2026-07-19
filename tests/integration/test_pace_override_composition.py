"""Integration test for the pace-override two-call-site composition fix
(corridor-phase2 review follow-up #2, branch commit 0abac05 "thread overrides
into resolve_pace_mode (pace-plumb dead outlet)").

``playlists.ds_pipeline.candidate_pool.pace_admission_floor`` is resolved onto
``cfg.candidate.pace_admission_floor`` at TWO call sites in
src/playlist/pipeline/core.py::generate_playlist_ds:

  1. ``default_ds_config`` (src/playlist/config.py:615-616) reads the override
     directly off the ``candidate_pool`` overrides dict, falling back to the
     pace-mode preset's ``admission_floor`` (always 0.0 for every preset --
     see mode_presets.py PACE_MODE_PRESETS) when absent.
  2. A later ``replace(cfg.candidate, pace_admission_floor=...)`` (core.py:483-
     494) re-derives the value from ``pace_settings["admission_floor"]``, where
     ``pace_settings = resolve_pace_mode(pace_mode, overrides=_resolve_pace_
     overrides(overrides, pb_overrides) or None)`` -- i.e. a SECOND, independent
     read of the same override key via ``_resolve_pace_overrides``.

Before the pace-plumb fix, call site 2 unconditionally clobbered call site 1's
resolved value with the bare preset (0.0), because ``resolve_pace_mode`` was
never passed an ``overrides=`` at all. This test proves the two sites now
compose to the SAME override value (no clobber, no double-apply) by running a
real generation through the production config chain and reading the value off
the FINAL ``cfg.candidate`` snapshot the pipeline returns
(``DsRunResult.requested["candidate"]["pace_admission_floor"]``, itself
``_params_from_config(cfg)`` captured after both call sites have run --
core.py:1498).

Every pace-mode preset's ``admission_floor`` is 0.0 (mode_presets.py), so any
nonzero override value can only be present in the resolved config if BOTH call
sites honored it -- a coincidental match with the preset default is impossible.
"""
from __future__ import annotations

import pytest

from tests.support.gui_fidelity import generate_like_gui
from tests.integration.test_gui_fidelity_regressions import ART, SEEDS

_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

_PACE_ADMISSION_FLOOR_OVERRIDE = 0.4321


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_pace_admission_floor_override_survives_both_call_sites():
    """A candidate_pool.pace_admission_floor override must reach the final
    resolved config unchanged -- not clobbered back to the pace-mode preset's
    0.0 default by the second (bridge-side) replace() in generate_playlist_ds.
    """
    res = generate_like_gui(
        seeds=SEEDS,
        cohesion_mode="narrow",
        genre_mode="narrow",
        sonic_mode="narrow",
        pace_mode="narrow",
        artist_spacing="strong",
        length=12,
        random_seed=0,
        config_overrides={
            "playlists": {
                "ds_pipeline": {
                    "candidate_pool": {
                        "pace_admission_floor": _PACE_ADMISSION_FLOOR_OVERRIDE,
                    }
                }
            }
        },
    )

    resolved = res.requested["candidate"]["pace_admission_floor"]
    assert resolved == pytest.approx(_PACE_ADMISSION_FLOOR_OVERRIDE), (
        "pace_admission_floor override did not survive the two-call-site "
        f"composition (default_ds_config + core.py's pace_settings replace()); "
        f"resolved={resolved!r}, expected={_PACE_ADMISSION_FLOOR_OVERRIDE!r} "
        "(every pace-mode preset defaults this to 0.0, so this can only match "
        "by the override actually flowing through both sites)"
    )
