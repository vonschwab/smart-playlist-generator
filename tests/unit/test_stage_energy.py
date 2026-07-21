import sqlite3
import sys
import types

import numpy as np
import pytest

import scripts.analyze_library as al

# The energy stage only runs on Windows+WSL (SP-1); on Linux/macOS it always
# skips before touching _energy_pending/_energy_preflight/_energy_run. These
# tests assert the Windows RUN behavior of that stage. The cross-platform
# SKIP behavior is covered separately by test_analyze_stage_policy.py.
_windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="energy stage runs only on Windows+WSL; cross-platform skip behavior covered by test_analyze_stage_policy",
)


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


@_windows_only
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


@_windows_only
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


@_windows_only
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


def test_checkpoint_metadata_for_wsl_flushes_wal(tmp_path):
    """After checkpoint, an immutable (main-file-only) read sees the committed data —
    proving the WSL extractor's snapshot read will be complete."""
    db = tmp_path / "metadata.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (id INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()  # committed into the WAL, not yet folded into the main file
    con.close()

    al._checkpoint_metadata_for_wsl(str(db))  # must not raise

    ro = sqlite3.connect(f"file:{db}?immutable=1", uri=True)  # main file only, ignores WAL
    assert ro.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    ro.close()


def test_checkpoint_metadata_for_wsl_best_effort(tmp_path):
    al._checkpoint_metadata_for_wsl(None)                        # no-op, no raise
    al._checkpoint_metadata_for_wsl(str(tmp_path / "nope.db"))   # no raise


@_windows_only
def test_stage_energy_checkpoints_before_extract(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    db = tmp_path / "metadata.db"
    sqlite3.connect(db).close()
    order = []
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: None)
    monkeypatch.setattr(al, "_checkpoint_metadata_for_wsl", lambda p: order.append(("checkpoint", p)))
    monkeypatch.setattr(
        al, "_energy_run",
        lambda cfg, *, force, cancellation_check: order.append(("run",)) or {
            "ok": 2, "missing": 0, "error": 0, "total": 2, "sidecar": "/x.npz"},
    )
    ctx = _ctx(tmp_path)
    ctx["db_path"] = str(db)
    al.stage_energy(ctx)
    assert order == [("checkpoint", str(db)), ("run",)]  # checkpoint BEFORE the WSL extract
