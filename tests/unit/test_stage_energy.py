import types
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_library as al


def _ctx(tmp_path, force=False, energy_workers=None):
    args = types.SimpleNamespace(force=force, energy_workers=energy_workers)
    return {
        "args": args,
        "config_path": str(tmp_path / "config.yaml"),
        "out_dir": tmp_path,
        "cancellation_check": None,
    }


def _make_artifact(tmp_path, ids=("a", "b")):
    np.savez(tmp_path / "data_matrices_step1.npz",
             track_ids=np.array(list(ids), dtype=object))


def test_stage_energy_skips_when_no_artifact(tmp_path):
    res = al.stage_energy(_ctx(tmp_path))
    assert res["skipped"] is True
    assert res.get("reason") == "no_artifact"


def test_stage_energy_skips_when_nothing_pending(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (0, 2))
    called = {"preflight": False}
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: called.__setitem__("preflight", True))
    res = al.stage_energy(_ctx(tmp_path, force=False))
    assert res["skipped"] is True
    assert called["preflight"] is False  # never touches WSL when up-to-date


def test_stage_energy_runs_and_returns_counts(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: None)
    monkeypatch.setattr(
        al, "_energy_run",
        lambda cfg, *, force, cancellation_check: {
            "ok": 2, "missing": 0, "error": 0, "total": 2, "sidecar": "/x.npz"},
    )
    res = al.stage_energy(_ctx(tmp_path, force=False))
    assert res["skipped"] is False
    assert res["ok"] == 2 and res["pending"] == 2


def test_stage_energy_preflight_failure_propagates(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))

    def boom(cfg):
        raise RuntimeError("WSL not available")

    monkeypatch.setattr(al, "_energy_preflight", boom)
    with pytest.raises(RuntimeError, match="WSL"):
        al.stage_energy(_ctx(tmp_path))


def test_energy_registered_and_ordered():
    assert "energy" in al.STAGE_FUNCS
    assert "energy" in al.STAGE_ORDER_DEFAULT
    assert al.STAGE_ORDER_DEFAULT.index("energy") == al.STAGE_ORDER_DEFAULT.index("artifacts") + 1
