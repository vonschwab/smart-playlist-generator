"""Resolution of the tower split used to slice X_sonic into perceptual axes.

The artifact's own ``tower_dims`` is authoritative — it records exactly how the
blend is laid out (rhythm | timbre | harmony). Inferring the split from the total
blend width is a lossy fallback that goes wrong whenever the towers aren't in the
default proportion (e.g. the 162-dim 2DFTM harmony rebuild: true (9,57,96) but
width-inference yields (40,81,41)).
"""
from types import SimpleNamespace

import numpy as np

from src.playlist_gui.worker import _resolve_tower_pca_dims


def _bundle(blend_dim, tower_dims):
    return SimpleNamespace(
        X_sonic=np.zeros((2, blend_dim), dtype=np.float32),
        tower_dims=tower_dims,
    )


def test_prefers_artifact_tower_dims_over_inference():
    # 162-dim 2DFTM rebuild: width-inference would give (40,81,41); the artifact
    # knows the true split is (9,57,96).
    bundle = _bundle(162, (9, 57, 96))
    assert _resolve_tower_pca_dims(bundle, ds_cfg={}) == (9, 57, 96)


def test_ignores_artifact_tower_dims_that_do_not_match_blend_width():
    # A stale/mismatched tower_dims must not be trusted — fall through to inference.
    bundle = _bundle(86, (9, 57, 96))  # sums to 162, not 86
    dims = _resolve_tower_pca_dims(bundle, ds_cfg={})
    assert sum(dims) == 86


def test_config_three_list_used_when_no_artifact_dims():
    bundle = _bundle(6, None)
    dims = _resolve_tower_pca_dims(bundle, ds_cfg={"tower_pca_dims": [2, 2, 2]})
    assert dims == (2, 2, 2)


def test_config_dict_is_not_treated_as_a_split():
    # The real config stores tower_pca_dims as a dict (PCA targets), which is a
    # different concept and must not be used to slice the blend.
    bundle = _bundle(162, (9, 57, 96))
    dims = _resolve_tower_pca_dims(
        bundle, ds_cfg={"tower_pca_dims": {"rhythm": 8, "timbre": 16, "harmony": 8}}
    )
    assert dims == (9, 57, 96)  # artifact wins; dict ignored


def test_falls_back_to_inference_without_artifact_or_config():
    bundle = _bundle(86, None)
    dims = _resolve_tower_pca_dims(bundle, ds_cfg={})
    assert sum(dims) == 86
