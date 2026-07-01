# tests/integration/test_mini_pier_integration.py
from pathlib import Path
import pytest
from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from tests.support.gui_fidelity import gui_ui_state, resolve_gui_overrides, resolve_gui_genre_params

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_req = pytest.mark.skipif(not ART.exists(), reason="live artifact required")
SEEDS = ["f28fd5cebac845cf64fee59d5ac3b3aa", "b8f8aa0e86f977f9fcb26f615e130ac9",
         "42473b911cef5674e56b8e2ce87df7cb", "49f8bba75408d4e0e0e000d1dc708add"]


def _gen(mini_pier: bool):
    ui = gui_ui_state(cohesion_mode="dynamic", genre_mode="dynamic",
                      sonic_mode="dynamic", pace_mode="dynamic")
    ov = resolve_gui_overrides(ui)
    if mini_pier:
        pb = dict(ov.get("pier_bridge", {}) or {})
        pb["mini_pier_enabled"] = True
        pb["mini_pier_max_interior"] = 4
        ov = {**ov, "pier_bridge": pb}
    gp = resolve_gui_genre_params(ui)
    b = load_artifact_bundle(str(ART))
    seeds = [s for s in SEEDS if s in b.track_id_to_index]
    if len(seeds) < 3:
        pytest.skip("seeds absent from this artifact build")
    return generate_playlist_ds(artifact_path=str(ART), seed_track_id=seeds[0],
        anchor_seed_ids=seeds, mode="dynamic", pace_mode="dynamic", length=30,
        random_seed=0, overrides=ov, artist_style_enabled=False, artist_playlist=False, **gp)


@pytest.mark.integration
@pytest.mark.slow
@_req
def test_mini_pier_off_matches_baseline_length():
    load_artifact_bundle.cache_clear()
    assert len(_gen(mini_pier=False).track_ids) == 30


@pytest.mark.integration
@pytest.mark.slow
@_req
def test_mini_pier_on_generates_and_changes_ordering():
    load_artifact_bundle.cache_clear()
    off = _gen(mini_pier=False).track_ids
    load_artifact_bundle.cache_clear()
    on = _gen(mini_pier=True).track_ids
    assert len(on) == 30                 # length preserved (waypoints take interior slots)
    assert list(on) != list(off)         # the flag actually changed the playlist
