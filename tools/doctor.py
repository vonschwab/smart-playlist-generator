#!/usr/bin/env python3
"""
MixArc Doctor
=============

Validates environment, dependencies, database, and artifacts.
Run this first before using the playlist generator.

Usage:
    python tools/doctor.py
    python tools/doctor.py --verbose
    python tools/doctor.py --fix  # Attempt to fix common issues

Exit codes:
    0 - All checks passed
    1 - One or more checks failed

This is a thin printer over src/setup/checks.py -- the single source of
truth for what each check does now lives there (returning CheckResult),
not here. This module only resolves a MixarcHome, runs the checks, and
renders the results under the same section headers / ✓/⚠/✗ glyphs doctor
has always used.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.mixarc.paths import resolve_home  # noqa: E402
from src.setup.checks import run_all_checks  # noqa: E402


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


# Section headers in doctor's historical order (old doctor.py L371-389).
# Every CheckResult id produced by run_all_checks() is mapped to exactly one
# of these so the on-screen grouping is unchanged even though the checks
# themselves now live in src/setup/checks.py.
_SECTION_ORDER = [
    "Python Environment",
    "Dependencies",
    "Module Imports",
    "Configuration",
    "Database",
    "Artifacts",
]

_ID_TO_SECTION = {
    "python_version": "Python Environment",
    "module_imports": "Module Imports",
    "config_file": "Configuration",
    "satellite_paths": "Configuration",
    "database": "Database",
    "artifacts": "Artifacts",
}


def _section_for(check_id: str) -> str:
    if check_id.startswith("dep_"):
        return "Dependencies"
    return _ID_TO_SECTION.get(check_id, "Other")


def main():
    """Run all doctor checks."""
    parser = argparse.ArgumentParser(
        description="Validate MixArc environment"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    print(f"\n{Colors.BOLD}MixArc Doctor{Colors.RESET}")
    print("=" * 40)

    home = resolve_home(None)
    results = run_all_checks(home)

    buckets: dict[str, list] = {name: [] for name in _SECTION_ORDER}
    for r in results:
        buckets.setdefault(_section_for(r.id), []).append(r)

    for name in _SECTION_ORDER:
        section(name)
        for r in buckets[name]:
            if r.status == "pass":
                # Dependency passes were only ever printed in --verbose mode
                # (old check_dependency() L90-102); everything else always
                # prints its pass line, matching prior behavior exactly.
                if r.id.startswith("dep_") and not args.verbose:
                    continue
                check_pass(r.summary)
            elif r.status == "warn":
                check_warn(r.summary)
            else:
                check_fail(r.summary, r.fix_hint)

    # Summary
    print("\n" + "=" * 40)
    passed = sum(1 for r in results if r.status == "pass")
    warned = sum(1 for r in results if r.status == "warn")
    failed = sum(1 for r in results if r.status == "fail")
    total = passed + warned + failed

    if failed == 0 and warned == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All {total} checks passed!{Colors.RESET}")
        print("\nYour environment is ready. Run:")
        print("  python main_app.py --artist \"Your Artist\" --dry-run")
        sys.exit(0)
    elif failed == 0:
        print(f"{Colors.GREEN}✓ {passed} passed{Colors.RESET}, "
              f"{Colors.YELLOW}⚠ {warned} warnings{Colors.RESET}")
        print("\nEnvironment is functional but some setup may be needed.")
        print("See warnings above for setup instructions.")
        sys.exit(0)
    else:
        print(f"{Colors.GREEN}✓ {passed} passed{Colors.RESET}, "
              f"{Colors.YELLOW}⚠ {warned} warnings{Colors.RESET}, "
              f"{Colors.RED}✗ {failed} failed{Colors.RESET}")
        print("\nPlease fix the failed checks above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
