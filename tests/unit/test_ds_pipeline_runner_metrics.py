from types import SimpleNamespace

from src.playlist import ds_pipeline_runner as runner


def test_ds_runner_metrics_use_recomputed_final_edges(monkeypatch):
    stale_playlist_stats = {
        "below_floor_count": 0,
        "min_transition": 0.9,
        "mean_transition": 0.95,
        "artist_counts": None,
        "distinct_artists": None,
        "strategy": "pier_bridge",
        "repair_applied": False,
        "num_segments": 1,
        "transition_floor": 0.2,
        "transition_gamma": 1.0,
        "transition_centered": True,
        "edge_scores": [{"T": 0.9}, {"T": 1.0}],
    }

    def fake_core_generate_playlist_ds(**kwargs):
        return SimpleNamespace(
            track_ids=["a", "b", "c"],
            stats={"playlist": dict(stale_playlist_stats)},
            params_requested={},
            params_effective={"playlist": {}},
        )

    final_edges = [
        {"prev_id": "a", "cur_id": "b", "T": 0.1},
        {"prev_id": "b", "cur_id": "c", "T": 0.3},
    ]

    monkeypatch.setattr(runner, "core_generate_playlist_ds", fake_core_generate_playlist_ds)
    monkeypatch.setattr(
        runner.reporter,
        "compute_edge_scores_from_artifact",
        lambda **kwargs: final_edges,
    )

    result = runner.generate_playlist_ds(
        artifact_path="artifact.npz",
        seed_track_id="a",
        mode="dynamic",
        length=3,
        random_seed=0,
    )

    assert result.metrics["below_floor"] == 1
    assert result.metrics["min_transition"] == 0.1
    assert result.metrics["mean_transition"] == 0.2
    assert result.metrics["edge_metric_source"] == "final_emitted_playlist"
    assert result.playlist_stats["playlist"]["edge_scores"] == final_edges
