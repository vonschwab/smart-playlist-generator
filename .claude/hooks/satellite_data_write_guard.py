"""PreToolUse hook: data-writing pipeline stages run ONLY in the canonical checkout.

`scripts/analyze_library.py` hardcodes ROOT_DIR-relative data paths (the WAL
corruption vector, memory feedback_worktree_sqlite_wal_aliasing), so scan/
enrich/adjudicate/publish/artifacts/folds and MuQ extraction must never run
from a satellite clone. In a satellite this hook denies those invocations; in
the canonical checkout it is silent. Detection: workspace_identity (origin URL).

Contract mirrors the repo's other PreToolUse hooks: stdin JSON in, deny JSON
out, silent to allow. FAIL OPEN — errors never block.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workspace_identity import is_satellite  # noqa: E402

# Data-writer identity, anchored to a path basename / module stem so that
# `test_muq_runner.py` does NOT read as `muq_runner.py`. A command only trips
# the guard when python actually RUNS one of these — as a `.py` script target
# or a `-m` module — never when a test/read-only tool merely NAMES the file.
_WRITER_STEM = r"(?:analyze_library|scan_library|muq_runner|fold_[a-z0-9_]*)"
_WRITER_SCRIPT = re.compile(rf"^{_WRITER_STEM}\.py$", re.IGNORECASE)
_WRITER_MODULE_STEM = re.compile(rf"^{_WRITER_STEM}$", re.IGNORECASE)
_PY_LAUNCHER = re.compile(r"^(?:python|python3|py)(?:\.exe)?$", re.IGNORECASE)
_PYTEST_LAUNCHER = re.compile(r"^pytest(?:\.exe)?$", re.IGNORECASE)
# Segment separators: sequence/pipe operators between commands.
_SEGMENT_SEP = re.compile(r"&&|\|\||;|\|")


def _basename(token):
    """Last path component of a token (splitting on both / and \\)."""
    return re.split(r"[\\/]", token)[-1]


def _segment_runs_writer(segment):
    """True iff this shell segment invokes python to RUN a data-writer."""
    tokens = segment.split()
    if not tokens:
        return False
    # A bare `pytest ...` (or `pytest.exe`) run is a test, never a data write.
    if _PYTEST_LAUNCHER.match(_basename(tokens[0])):
        return False
    # The segment must invoke a python launcher.
    launcher_idx = next(
        (i for i, tok in enumerate(tokens) if _PY_LAUNCHER.match(_basename(tok))),
        None,
    )
    if launcher_idx is None:
        return False
    args = tokens[launcher_idx + 1:]
    # `-m <module>`: pytest is a test run; a writer module stem is a data write.
    if "-m" in args:
        mi = args.index("-m")
        module = args[mi + 1] if mi + 1 < len(args) else ""
        if module.split(".")[0].lower() == "pytest":
            return False
        if module and _WRITER_MODULE_STEM.match(module.split(".")[-1]):
            return True
    # Positional `.py` script target whose basename is a data-writer.
    for tok in args:
        if tok.startswith("-"):
            continue
        if _WRITER_SCRIPT.match(_basename(tok)):
            return True
    return False


_REASON = (
    "Blocked by the satellite data-write guard: data-writing pipeline stages "
    "(scan/analyze/adjudicate/publish/artifacts/folds/MuQ) run ONLY in the "
    "canonical checkout (C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3) — "
    "analyze_library.py resolves data paths relative to its own tree, so a "
    "satellite run writes junk into the satellite's stub data/ (or worse). "
    "Run this command from a session in the canonical checkout instead."
)


def command_denied(command, satellite):
    """Return the deny message when a data-writing command runs in a satellite.

    Judges each `;`/`&&`/`||`/`|`-separated segment independently, so a
    compound like `cd x && python scripts/analyze_library.py` still trips on
    its python segment, while a read-only or test segment that merely names a
    data-writer file is allowed.
    """
    if not satellite:
        return None
    for segment in _SEGMENT_SEP.split(command or ""):
        if _segment_runs_writer(segment):
            return _REASON
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    try:
        if (data.get("tool_name") or "") not in ("Bash", "PowerShell"):
            return
        command = (data.get("tool_input") or {}).get("command") or ""
        reason = command_denied(command, is_satellite())
        if reason:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": reason,
                        }
                    }
                )
            )
    except Exception:
        return  # fail-open


if __name__ == "__main__":
    main()
