"""Unit tests for doctor's satellite data-path check (stub-landmine gate)."""

import importlib.util
import pathlib
import sys

_DOCTOR = pathlib.Path(__file__).resolve().parents[1] / "tools" / "doctor.py"
_spec = importlib.util.spec_from_file_location("doctor", _DOCTOR)
assert _spec is not None and _spec.loader is not None
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)


def _make_satellite(tmp_path, db_bytes=2 * 1024 * 1024, art_bytes=11 * 1024 * 1024,
                    db_abs=True, art_abs=True):
    """A fake satellite clone + a fake canonical data dir; returns (sat_root, cfg_text)."""
    sat = tmp_path / "sat"
    (sat / ".git").mkdir(parents=True)
    (sat / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = C:/canonical\n', encoding="utf-8"
    )
    canon = tmp_path / "canonical" / "data"
    canon.mkdir(parents=True)
    db = canon / "metadata.db"
    db.write_bytes(b"\0" * db_bytes)
    art = canon / "artifact.npz"
    art.write_bytes(b"\0" * art_bytes)
    db_val = db.as_posix() if db_abs else "data/metadata.db"
    art_val = art.as_posix() if art_abs else "data/artifacts/x.npz"
    (sat / "config.yaml").write_text(
        f"library:\n  database_path: {db_val}\n"
        f"playlists:\n  ds_pipeline:\n    artifact_path: {art_val}\n",
        encoding="utf-8",
    )
    return sat


def _run_check(sat_root):
    checks = doctor.DoctorChecks()
    ok = checks.check_satellite_data_paths(root=sat_root)
    return ok, checks


def test_valid_satellite_passes(tmp_path):
    sat = _make_satellite(tmp_path)
    ok, checks = _run_check(sat)
    assert ok is True and checks.failed == 0


def test_relative_db_path_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_abs=False)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_stub_sized_db_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_bytes=0)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_small_artifact_fails(tmp_path):
    sat = _make_satellite(tmp_path, art_bytes=1024)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_db_inside_satellite_fails(tmp_path):
    sat = _make_satellite(tmp_path)
    inside = sat / "data"
    inside.mkdir()
    (inside / "metadata.db").write_bytes(b"\0" * (2 * 1024 * 1024))
    art = (tmp_path / "canonical" / "data" / "artifact.npz").as_posix()
    (sat / "config.yaml").write_text(
        f"library:\n  database_path: {(inside / 'metadata.db').as_posix()}\n"
        f"playlists:\n  ds_pipeline:\n    artifact_path: {art}\n",
        encoding="utf-8",
    )
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_canonical_workspace_passes_trivially(tmp_path):
    canon = tmp_path / "repo"
    (canon / ".git").mkdir(parents=True)
    (canon / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/x/y.git\n', encoding="utf-8"
    )
    ok, checks = _run_check(canon)
    assert ok is True and checks.failed == 0
