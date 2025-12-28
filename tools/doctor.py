#!/usr/bin/env python3
"""
Playlist Generator Doctor
=========================

Validates environment, dependencies, database, and artifacts.
Run this first before using the playlist generator.

Usage:
    python tools/doctor.py
    python tools/doctor.py --verbose
    python tools/doctor.py --fix  # Attempt to fix common issues

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import argparse
import importlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.GREEN = cls.YELLOW = cls.RED = cls.RESET = cls.BOLD = ''


def check_pass(msg: str) -> None:
    """Print a passing check."""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def check_warn(msg: str) -> None:
    """Print a warning."""
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")


def check_fail(msg: str, fix: str = None) -> None:
    """Print a failing check with optional fix instruction."""
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")
    if fix:
        print(f"    {Colors.BOLD}Fix:{Colors.RESET} {fix}")


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}[{title}]{Colors.RESET}")


class DoctorChecks:
    """Collection of environment validation checks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def check_python_version(self) -> bool:
        """Check Python version is 3.8+."""
        version = sys.version_info
        if version >= (3, 8):
            check_pass(f"Python {version.major}.{version.minor}.{version.micro}")
            self.passed += 1
            return True
        else:
            check_fail(
                f"Python {version.major}.{version.minor} (need 3.8+)",
                "Install Python 3.8 or newer"
            )
            self.failed += 1
            return False

    def check_dependency(self, module: str, pip_name: str = None) -> bool:
        """Check if a Python module is importable."""
        pip_name = pip_name or module
        try:
            importlib.import_module(module)
            if self.verbose:
                check_pass(f"{module}")
            self.passed += 1
            return True
        except ImportError as e:
            check_fail(f"{module} not found", f"pip install {pip_name}")
            self.failed += 1
            return False

    def check_core_dependencies(self) -> int:
        """Check all core dependencies."""
        deps = [
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("yaml", "pyyaml"),
            ("librosa", "librosa"),
            ("mutagen", "mutagen"),
            ("sklearn", "scikit-learn"),
            ("tqdm", "tqdm"),
            ("requests", "requests"),
        ]
        failed = 0
        for module, pip_name in deps:
            if not self.check_dependency(module, pip_name):
                failed += 1
        return failed

    def check_config_file(self) -> bool:
        """Check config.yaml exists."""
        config_path = ROOT_DIR / "config.yaml"
        if config_path.exists():
            check_pass("config.yaml found")
            self.passed += 1
            return True
        else:
            example_path = ROOT_DIR / "config.example.yaml"
            if example_path.exists():
                check_fail(
                    "config.yaml not found",
                    f"cp config.example.yaml config.yaml && edit config.yaml"
                )
            else:
                check_fail("config.yaml not found", "Create config.yaml from template")
            self.failed += 1
            return False

    def check_database(self) -> bool:
        """Check database exists and has valid schema."""
        # Try to load config for DB path
        db_path = ROOT_DIR / "data" / "metadata.db"

        try:
            from src.config_loader import Config
            config_path = ROOT_DIR / "config.yaml"
            if config_path.exists():
                config = Config(str(config_path))
                db_path_str = config.get('library', 'database_path', default='data/metadata.db')
                if db_path_str:
                    db_path = ROOT_DIR / db_path_str
        except Exception:
            pass  # Use default path

        if not db_path.exists():
            check_warn(f"Database not found: {db_path}")
            check_warn("Run: python scripts/scan_library.py")
            self.warned += 1
            return True  # Not fatal, just needs setup

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check for tracks table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            )
            if not cursor.fetchone():
                check_fail("Database missing 'tracks' table", "Run: python scripts/scan_library.py")
                self.failed += 1
                conn.close()
                return False

            # Count tracks
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]

            # Check for sonic features
            cursor.execute("SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL")
            sonic_count = cursor.fetchone()[0]

            conn.close()

            if count == 0:
                check_warn(f"Database has no tracks")
                check_warn("Run: python scripts/scan_library.py")
                self.warned += 1
            else:
                check_pass(f"Database: {count:,} tracks ({sonic_count:,} with sonic features)")
                self.passed += 1

            return True

        except sqlite3.Error as e:
            check_fail(f"Database error: {e}")
            self.failed += 1
            return False

    def check_artifacts(self) -> bool:
        """Check DS pipeline artifacts exist."""
        artifacts_dir = ROOT_DIR / "data" / "artifacts"

        if not artifacts_dir.exists():
            check_warn("Artifacts directory not found")
            check_warn("Run: python scripts/build_beat3tower_artifacts.py ...")
            self.warned += 1
            return True  # Not fatal

        # Look for any .npz files
        npz_files = list(artifacts_dir.rglob("*.npz"))
        if not npz_files:
            check_warn("No artifact files found (.npz)")
            check_warn("Run: python scripts/build_beat3tower_artifacts.py ...")
            self.warned += 1
            return True

        # Check the main artifact
        main_artifact = artifacts_dir / "beat3tower_32k" / "data_matrices_step1.npz"
        if main_artifact.exists():
            size_mb = main_artifact.stat().st_size / (1024 * 1024)
            check_pass(f"Main artifact: {main_artifact.name} ({size_mb:.1f} MB)")
            self.passed += 1
        else:
            check_warn(f"Main artifact not at expected path: {main_artifact}")
            check_warn(f"Found {len(npz_files)} artifact(s) in {artifacts_dir}")
            self.warned += 1

        return True

    def check_genre_similarity(self) -> bool:
        """Check genre similarity file exists."""
        genre_sim_path = ROOT_DIR / "data" / "genre_similarity.yaml"

        if genre_sim_path.exists():
            check_pass(f"Genre similarity matrix: {genre_sim_path.name}")
            self.passed += 1
            return True
        else:
            check_warn("Genre similarity file not found")
            self.warned += 1
            return True  # Not fatal

    def check_playlist_module_imports(self) -> bool:
        """Check core playlist module imports work."""
        try:
            from src.config_loader import Config
            from src.local_library_client import LocalLibraryClient
            from src.playlist.pipeline import DSPipelineResult
            from src.playlist.constructor import construct_playlist
            check_pass("Core modules importable")
            self.passed += 1
            return True
        except ImportError as e:
            check_fail(f"Import error: {e}")
            self.failed += 1
            return False


def main():
    """Run all doctor checks."""
    parser = argparse.ArgumentParser(
        description="Validate Playlist Generator environment"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    print(f"\n{Colors.BOLD}Playlist Generator Doctor{Colors.RESET}")
    print("=" * 40)

    doctor = DoctorChecks(verbose=args.verbose)

    # Run checks
    section("Python Environment")
    doctor.check_python_version()

    section("Dependencies")
    doctor.check_core_dependencies()

    section("Module Imports")
    doctor.check_playlist_module_imports()

    section("Configuration")
    doctor.check_config_file()

    section("Database")
    doctor.check_database()

    section("Artifacts")
    doctor.check_artifacts()
    doctor.check_genre_similarity()

    # Summary
    print("\n" + "=" * 40)
    total = doctor.passed + doctor.warned + doctor.failed

    if doctor.failed == 0 and doctor.warned == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All {total} checks passed!{Colors.RESET}")
        print("\nYour environment is ready. Run:")
        print("  python main_app.py --artist \"Your Artist\" --dry-run")
        sys.exit(0)
    elif doctor.failed == 0:
        print(f"{Colors.GREEN}✓ {doctor.passed} passed{Colors.RESET}, "
              f"{Colors.YELLOW}⚠ {doctor.warned} warnings{Colors.RESET}")
        print("\nEnvironment is functional but some setup may be needed.")
        print("See warnings above for setup instructions.")
        sys.exit(0)
    else:
        print(f"{Colors.GREEN}✓ {doctor.passed} passed{Colors.RESET}, "
              f"{Colors.YELLOW}⚠ {doctor.warned} warnings{Colors.RESET}, "
              f"{Colors.RED}✗ {doctor.failed} failed{Colors.RESET}")
        print("\nPlease fix the failed checks above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
