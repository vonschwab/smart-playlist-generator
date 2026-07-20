"""CheckResult library: pure check functions return structured results."""
import sqlite3

from src.mixarc.paths import MixarcHome
from src.setup.result import CheckResult
from src.setup.checks import (
    check_python_version, check_config_file, check_database, run_all_checks,
    check_satellite_data_paths,
)


def _home(tmp_path, source="platformdirs"):
    return MixarcHome(config_path=tmp_path / "config.yaml", anchor_dir=tmp_path, source=source)


def test_python_version_passes_on_current_interpreter():
    r = check_python_version()
    assert isinstance(r, CheckResult)
    assert r.id == "python_version"
    assert r.status == "pass"  # test runs on 3.11+


def test_config_file_fails_when_absent(tmp_path):
    r = check_config_file(_home(tmp_path))
    assert r.id == "config_file" and r.status == "fail"
    assert r.fix_hint is not None


def test_config_file_passes_when_present(tmp_path):
    (tmp_path / "config.yaml").write_text("library: {}\n", encoding="utf-8")
    assert check_config_file(_home(tmp_path)).status == "pass"


def test_database_warns_when_absent(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "library:\n  database_path: data/metadata.db\n", encoding="utf-8")
    r = check_database(_home(tmp_path))
    assert r.id == "database" and r.status == "warn"


def test_database_passes_with_tracks(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "library:\n  database_path: data/metadata.db\n", encoding="utf-8")
    dbdir = tmp_path / "data"
    dbdir.mkdir()
    conn = sqlite3.connect(dbdir / "metadata.db")
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO tracks VALUES ('t1')")
    conn.commit()
    conn.close()
    assert check_database(_home(tmp_path)).status == "pass"


def test_satellite_check_noops_for_platformdirs(tmp_path):
    r = check_satellite_data_paths(_home(tmp_path, source="platformdirs"))
    assert r.status == "pass"


def test_run_all_checks_returns_ordered_list(tmp_path):
    results = run_all_checks(_home(tmp_path))
    assert all(isinstance(r, CheckResult) for r in results)
    ids = [r.id for r in results]
    assert ids[0] == "python_version"
    assert "config_file" in ids and "database" in ids


def test_database_degrades_gracefully_on_malformed_config(tmp_path):
    # A YAML syntax error (unclosed flow sequence) must not crash the check
    # -- doctor.py tolerates a bad config.yaml (except Exception: pass) and
    # this extraction must match, never raise out of check_database.
    (tmp_path / "config.yaml").write_text("library: [unclosed\n", encoding="utf-8")
    r = check_database(_home(tmp_path))
    assert isinstance(r, CheckResult)
    assert r.id == "database"
    assert r.status in ("pass", "warn", "fail")


def test_run_all_checks_does_not_raise_on_malformed_config(tmp_path):
    (tmp_path / "config.yaml").write_text("library: [unclosed\n", encoding="utf-8")
    results = run_all_checks(_home(tmp_path))
    assert isinstance(results, list)
    assert all(isinstance(r, CheckResult) for r in results)
