import numpy as np

from src.playlist.pipeline import generate_playlist_ds
from src.similarity import sonic_variant as sv


def test_resolve_variant_prefers_env(monkeypatch):
    monkeypatch.setenv("SONIC_SIM_VARIANT", "z")
    # explicit beats env
    assert sv.resolve_sonic_variant(explicit_variant="centered") == "centered"
    monkeypatch.delenv("SONIC_SIM_VARIANT", raising=False)
    assert sv.resolve_sonic_variant(config_variant="unknown") == "raw"


def test_pipeline_uses_variant_and_records(monkeypatch, synthetic_artifact):
    path, bundle = synthetic_artifact
    called = {}

    def fake_compute(mat: np.ndarray, variant: str, l2: bool = False):
        called["variant"] = variant
        called["l2"] = l2
        return mat * (2 if variant == "z" else 1), {"variant": variant, "dim": mat.shape[1]}

    monkeypatch.setattr("src.playlist.pipeline.compute_sonic_variant_matrix", fake_compute)
    overrides = {
        "candidate": {"similarity_floor": -1.0, "max_pool_size": 2000, "target_artists": 10, "candidates_per_artist": 6},
        "construct": {"transition_floor": -1.0, "hard_floor": False, "min_gap": 1},
    }
    result = generate_playlist_ds(
        artifact_path=path,
        seed_track_id=str(bundle.track_ids[0]),
        num_tracks=10,
        mode="dynamic",
        random_seed=42,
        overrides=overrides,
        sonic_variant="z",
    )
    assert called["variant"] == "z"
    assert called["l2"] is False
    assert result.params_effective.get("sonic_variant", {}).get("variant") == "z"
