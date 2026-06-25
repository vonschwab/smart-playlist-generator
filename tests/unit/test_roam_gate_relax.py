"""Roam broad-pool: when roam is on, core.py relaxes the dense PMI-SVD genre gate
(genre_admission_percentile) before building the candidate pool — sonic leads, the
graph keeps it honest. Spies on build_candidate_pool to capture the gate it receives.
"""
import pytest

import src.playlist.pipeline.core as core_mod
from tests.unit.test_pipeline_smoke_golden import _build_smoke_fixture


class _PoolSentinel(Exception):
    pass


def _captured_gate(tmp_path, monkeypatch, overrides):
    captured = {}

    def _spy(*args, **kwargs):
        captured["gap"] = kwargs.get("genre_admission_percentile")
        raise _PoolSentinel()

    monkeypatch.setattr(core_mod, "build_candidate_pool", _spy)
    artifact = _build_smoke_fixture(tmp_path)
    with pytest.raises(_PoolSentinel):
        core_mod.generate_playlist_ds(
            artifact_path=str(artifact),
            seed_track_id="t0",
            num_tracks=8,
            mode="dynamic",
            random_seed=0,
            overrides=overrides,
            dry_run=False,
        )
    return captured.get("gap")


def test_roam_off_keeps_dense_genre_gate(tmp_path, monkeypatch):
    gap = _captured_gate(tmp_path, monkeypatch, {
        "pier_bridge": {"genre_admission_percentile_dynamic": 0.85},
    })
    assert gap == 0.85


def test_roam_on_relaxes_dense_genre_gate(tmp_path, monkeypatch):
    gap = _captured_gate(tmp_path, monkeypatch, {
        "pier_bridge": {"genre_admission_percentile_dynamic": 0.85, "roam": {"enabled": True}},
    })
    assert gap == 0.0


def test_roam_genre_gate_percentile_is_tunable(tmp_path, monkeypatch):
    # roam.genre_gate_percentile picks the relaxed value (e.g. a light gate at 0.5).
    gap = _captured_gate(tmp_path, monkeypatch, {
        "pier_bridge": {
            "genre_admission_percentile_dynamic": 0.85,
            "roam": {"enabled": True, "genre_gate_percentile": 0.5},
        },
    })
    assert gap == 0.5
