"""On-feature integration test for variable bridge length (Task 3).

Drives a real multi-pier playlist through the production GUI-fidelity harness
(``tests/support/gui_fidelity``) with ``pier_bridge.variable_bridge_length=True``,
exactly as the GUI worker would resolve config. This is NOT a hand-built
single-seed config — it walks the same policy -> overrides -> config.yaml chain
the worker uses (see the playlist-testing skill).

The exact-N length invariant has been retired (Dylan's decision, Task 3):
both the builder assembly check and ``post_validation.run_post_order_validation``
now emit soft warnings instead of raising, so a band-length result is returned
normally. The xfail that previously guarded the on-feature assertion has been
removed. The test is still marked ``@integration`` + ``@slow`` + ``_requires_artifact``
and will SKIP in CI where the live artifact is absent; the full corpus run
happens in Task 4.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from tests.support.gui_fidelity import (
    gui_ui_state,
    resolve_gui_genre_params,
    resolve_gui_overrides,
)

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_requires_artifact = pytest.mark.skipif(
    not ART.exists(), reason="live artifact required"
)

# Real bundle track_ids (same multi-pier seed set the gui_fidelity regression
# catalog uses) — these resolve against the live artifact bundle.
SEEDS = [
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler - Howling at the Second Moon
    "b8f8aa0e86f977f9fcb26f615e130ac9",  # Hayden Pedigo - Nearer, Nearer
    "42473b911cef5674e56b8e2ce87df7cb",  # Steve Hiett - Are These My Memories?
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia - Hold On Magnolia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan - Let's Move to the Country
]

_CONFIG = "config.yaml"


def _generate(*, variable_bridge: bool, length: int = 30):
    """Generate a multi-pier playlist exactly as the GUI worker resolves config,
    optionally enabling variable bridge length via the pier_bridge override
    channel (Task 2 wiring -> apply_pier_bridge_overrides)."""
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
    )
    ds_overrides = resolve_gui_overrides(ui, config_path=_CONFIG)
    genre_params = resolve_gui_genre_params(ui, config_path=_CONFIG)
    if variable_bridge:
        pb = dict(ds_overrides.get("pier_bridge", {}) or {})
        pb["variable_bridge_length"] = True
        ds_overrides = {**ds_overrides, "pier_bridge": pb}
    bundle = load_artifact_bundle(str(ART))
    ti = bundle.track_id_to_index
    seeds = [s for s in SEEDS if s in ti]
    if len(seeds) < 4:
        pytest.skip("multi-pier seeds not present in this artifact build")
    return generate_playlist_ds(
        artifact_path=str(ART),
        seed_track_id=seeds[0],
        anchor_seed_ids=seeds,
        mode=ui.cohesion_mode,
        pace_mode=ui.pace_mode,
        length=length,
        random_seed=0,
        overrides=ds_overrides,
        artist_style_enabled=False,
        artist_playlist=False,
        **genre_params,
    )


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_variable_bridge_holds_total_in_band_and_helps_or_holds_worst_edge():
    load_artifact_bundle.cache_clear()
    base = _generate(variable_bridge=False)
    flex = _generate(variable_bridge=True)
    # Total in band [N-5, N+5] (variable_bridge_band default = 5).
    assert 25 <= len(flex.track_ids) <= 35
    base_mt = base.playlist_stats.get("min_transition")
    flex_mt = flex.playlist_stats.get("min_transition")
    if base_mt is not None and flex_mt is not None:
        # Worst edge must not regress.
        assert float(flex_mt) >= float(base_mt) - 1e-6


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_variable_bridge_off_default_generates_exact_length():
    """Sanity: with the feature OFF (default), the multi-pier path still returns
    exactly the requested length — the byte-identical off-path, end to end."""
    load_artifact_bundle.cache_clear()
    res = _generate(variable_bridge=False)
    assert len(res.track_ids) == 30
