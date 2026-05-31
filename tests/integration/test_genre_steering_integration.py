import numpy as np
import pytest
from pathlib import Path

from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_requires = pytest.mark.skipif(not ART.exists(), reason="live artifact required")
SMITHS = "de11fcb727aae7853a1b6c1e0d89ab25"      # This Charming Man
CHARLI = "5dda14ae880acbcc911e32710c50d5a5"      # a Charli XCX track


def _mean_edge_genre(bundle, track_ids):
    D = bundle.X_genre_dense
    ti = bundle.track_id_to_index
    sims = []
    for a, b in zip(track_ids, track_ids[1:]):
        ia, ib = ti.get(str(a)), ti.get(str(b))
        if ia is None or ib is None:
            continue
        na, nb = np.linalg.norm(D[ia]), np.linalg.norm(D[ib])
        if na < 1e-9 or nb < 1e-9:
            continue
        sims.append(float(D[ia] @ D[ib]))
    return float(np.mean(sims)) if sims else 0.0


@pytest.mark.integration
@pytest.mark.slow
@_requires
def test_smiths_edge_genre_coherence_improves_with_steering():
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))

    base = generate_playlist_ds(artifact_path=str(ART), seed_track_id=SMITHS,
                                mode="narrow", length=30, random_seed=42,
                                overrides={"pier_bridge": {"genre_steering_enabled": False}})
    steered = generate_playlist_ds(artifact_path=str(ART), seed_track_id=SMITHS,
                                   mode="narrow", length=30, random_seed=42,
                                   overrides={"pier_bridge": {
                                       "genre_steering_enabled": True,
                                       "weight_genre_narrow": 0.20,
                                       "genre_edge_floor_narrow": 0.40,
                                   }})
    g_base = _mean_edge_genre(bundle, base.track_ids)
    g_steer = _mean_edge_genre(bundle, steered.track_ids)
    assert g_steer > g_base, f"steering should raise mean edge genre sim: base={g_base:.3f} steered={g_steer:.3f}"


@pytest.mark.integration
@pytest.mark.slow
@_requires
def test_charli_narrow_still_feasible_with_steering_and_relaxation():
    load_artifact_bundle.cache_clear()
    res = generate_playlist_ds(artifact_path=str(ART), seed_track_id=CHARLI,
                               mode="narrow", length=40, random_seed=42,
                               overrides={"pier_bridge": {
                                   "genre_steering_enabled": True,
                                   "weight_genre_narrow": 0.20,
                                   "genre_edge_floor_narrow": 0.40,
                                   "infeasible_handling": {
                                       "enabled": True,
                                       "genre_floor_relaxation_enabled": True,
                                       "min_genre_edge_floor": 0.0,
                                   }}})
    assert res is not None and len(res.track_ids) >= 30
