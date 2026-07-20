"""Unit tests for the satellite data-path check (stub-landmine gate).

Migrated (mixarc-sp2 Task 2) from loading tools/doctor.py's now-deleted
DoctorChecks class to calling src.setup.checks.check_satellite_data_paths
directly -- doctor.py is a thin printer now; this logic's single source of
truth lives in src/setup/checks.py (extracted there in Task 1). Scenarios
and assertions are unchanged; only the call surface moved.
"""

from src.mixarc.paths import MixarcHome
from src.setup.checks import check_satellite_data_paths


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
    # source="repo" (anything but "platformdirs") so the check doesn't
    # short-circuit to its no-op pass; the function reads root/"config.yaml"
    # directly, same as the old root=sat_root parameter.
    home = MixarcHome(config_path=sat_root / "config.yaml", anchor_dir=sat_root, source="repo")
    return check_satellite_data_paths(home)


def test_valid_satellite_passes(tmp_path):
    sat = _make_satellite(tmp_path)
    r = _run_check(sat)
    assert r.status != "fail"


def test_relative_db_path_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_abs=False)
    r = _run_check(sat)
    assert r.status == "fail"


def test_stub_sized_db_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_bytes=0)
    r = _run_check(sat)
    assert r.status == "fail"


def test_small_artifact_fails(tmp_path):
    sat = _make_satellite(tmp_path, art_bytes=1024)
    r = _run_check(sat)
    assert r.status == "fail"


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
    r = _run_check(sat)
    assert r.status == "fail"


def test_canonical_workspace_passes_trivially(tmp_path):
    canon = tmp_path / "repo"
    (canon / ".git").mkdir(parents=True)
    (canon / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/x/y.git\n', encoding="utf-8"
    )
    r = _run_check(canon)
    assert r.status != "fail"
