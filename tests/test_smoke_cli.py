"""Smoke tests for CLI entrypoints.

These tests verify that CLI scripts can run with --help without errors.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_script_help(script_path: Path) -> subprocess.CompletedProcess:
    """Run a script with --help and capture output."""
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )
    return result


class TestMainAppCLI:
    """Test main_app.py CLI."""

    def test_main_app_help(self):
        """main_app.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "main_app.py")
        assert result.returncode == 0, f"Error: {result.stderr}"
        assert "artist" in result.stdout.lower()
        assert "tracks" in result.stdout.lower()


class TestScriptsCLI:
    """Test production scripts CLI."""

    def test_scan_library_help(self):
        """scan_library.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "scripts" / "scan_library.py")
        assert result.returncode == 0, f"Error: {result.stderr}"
        assert "--quick" in result.stdout or "quick" in result.stdout.lower()

    def test_update_sonic_help(self):
        """update_sonic.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "scripts" / "update_sonic.py")
        assert result.returncode == 0, f"Error: {result.stderr}"
        assert "--beat3tower" in result.stdout or "beat3tower" in result.stdout.lower()

    def test_update_genres_help(self):
        """update_genres_v3_normalized.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "scripts" / "update_genres_v3_normalized.py")
        assert result.returncode == 0, f"Error: {result.stderr}"
        assert "--artists" in result.stdout or "artists" in result.stdout.lower()

    def test_build_artifacts_help(self):
        """build_beat3tower_artifacts.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "scripts" / "build_beat3tower_artifacts.py")
        assert result.returncode == 0, f"Error: {result.stderr}"
        assert "--db-path" in result.stdout or "--output" in result.stdout

    def test_analyze_library_help(self):
        """analyze_library.py --help should succeed."""
        result = run_script_help(ROOT_DIR / "scripts" / "analyze_library.py")
        assert result.returncode == 0, f"Error: {result.stderr}"


class TestDoctorCLI:
    """Test doctor script CLI."""

    def test_doctor_help(self):
        """doctor.py --help should succeed."""
        doctor_path = ROOT_DIR / "tools" / "doctor.py"
        if not doctor_path.exists():
            pytest.skip("doctor.py not yet created")
        result = run_script_help(doctor_path)
        assert result.returncode == 0, f"Error: {result.stderr}"
