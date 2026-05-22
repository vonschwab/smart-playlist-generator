from types import SimpleNamespace

from src.playlist import ds_pipeline_runner as runner
from src.playlist.ds_pipeline_builder import DSPipelineBuilder


def test_ds_runner_forwards_pace_mode_to_core(monkeypatch):
    captured = {}

    def fake_core_generate_playlist_ds(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            track_ids=["a"],
            stats={"playlist": {}},
            params_requested={},
            params_effective={"playlist": {}},
        )

    monkeypatch.setattr(runner, "core_generate_playlist_ds", fake_core_generate_playlist_ds)

    runner.generate_playlist_ds(
        artifact_path="artifact.npz",
        seed_track_id="a",
        mode="dynamic",
        pace_mode="narrow",
        length=1,
        random_seed=0,
    )

    assert captured["pace_mode"] == "narrow"


def test_ds_pipeline_builder_stores_pace_mode():
    request = (
        DSPipelineBuilder()
        .with_artifacts("data/artifacts/test.npz")
        .with_seed("track_001")
        .with_pace_mode("strict")
        .build()
    )

    assert request.pace_mode == "strict"
