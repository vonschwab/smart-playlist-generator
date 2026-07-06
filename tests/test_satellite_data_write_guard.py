"""Unit tests: data-writing pipeline commands are denied in satellites only."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

_HOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "satellite_data_write_guard.py"
)
_spec = importlib.util.spec_from_file_location("satellite_data_write_guard", _HOOK)
assert _spec is not None and _spec.loader is not None, f"hook not found at {_HOOK}"
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


DENIED_IN_SATELLITE = [
    "python scripts/analyze_library.py",
    "python scripts/analyze_library.py --stages publish",
    "python scripts/fold_2dftm_into_artifact.py",
    "python -m src.analyze.muq_runner --extract",
    "python scripts/scan_library.py",
    "python src/analyze/muq_runner.py",
    "cd /d C:\\x && python scripts/analyze_library.py",  # compound: python segment triggers
]

ALWAYS_ALLOWED = [
    "python -m pytest -q",
    "python main_app.py --artist X --tracks 30",
    "python tools/serve_web.py --port 8771",
    "python tools/doctor.py",
    "git status",
]

# Commands that merely MENTION a data-writer filename — test runs and
# read-only git/cat — must NEVER be denied, in either workspace. These are
# the false positives the anchored, execution-aware detection eliminates.
MENTIONS_ALLOWED = [
    "pytest tests/unit/test_muq_runner.py",
    "python -m pytest tests/unit/test_muq_runner.py",
    "python -m pytest tests/unit/test_fold_2dftm.py",
    "git show HEAD -- scripts/analyze_library.py",
    "git log -- scripts/scan_library.py",
    "cat scripts/fold_muq_into_artifact.py",
]


def test_data_writers_denied_in_satellite():
    for cmd in DENIED_IN_SATELLITE:
        assert hook.command_denied(cmd, satellite=True) is not None, cmd


def test_data_writers_allowed_in_canonical():
    for cmd in DENIED_IN_SATELLITE:
        assert hook.command_denied(cmd, satellite=False) is None, cmd


def test_other_commands_allowed_everywhere():
    for cmd in ALWAYS_ALLOWED:
        assert hook.command_denied(cmd, satellite=True) is None, cmd
        assert hook.command_denied(cmd, satellite=False) is None, cmd


def test_mentions_not_denied_in_satellite():
    # A data-writer filename mentioned by a test run or read-only tool is not
    # an execution of that writer — never deny it, even in a satellite.
    for cmd in MENTIONS_ALLOWED:
        assert hook.command_denied(cmd, satellite=True) is None, cmd


def test_mentions_allowed_in_canonical():
    for cmd in MENTIONS_ALLOWED:
        assert hook.command_denied(cmd, satellite=False) is None, cmd


def test_e2e_deny_in_satellite_env(tmp_path):
    # Simulate a satellite: CLAUDE_PROJECT_DIR with a local-path origin.
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text(
        '[remote "origin"]\n\turl = C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3\n',
        encoding="utf-8",
    )
    payload = {"tool_name": "Bash", "tool_input": {"command": "python scripts/analyze_library.py"}}
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    parsed = json.loads(proc.stdout.strip())
    assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_e2e_silent_in_canonical_env(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/vonschwab/playlist-generator.git\n',
        encoding="utf-8",
    )
    payload = {"tool_name": "Bash", "tool_input": {"command": "python scripts/analyze_library.py"}}
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    assert proc.stdout.strip() == ""
