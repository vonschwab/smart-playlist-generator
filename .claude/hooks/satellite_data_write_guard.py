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

_DATA_WRITERS = re.compile(
    r"analyze_library\.py|scan_library\.py|fold_[a-z0-9_]*\.py|muq_runner",
    re.IGNORECASE,
)

_REASON = (
    "Blocked by the satellite data-write guard: data-writing pipeline stages "
    "(scan/analyze/adjudicate/publish/artifacts/folds/MuQ) run ONLY in the "
    "canonical checkout (C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3) — "
    "analyze_library.py resolves data paths relative to its own tree, so a "
    "satellite run writes junk into the satellite's stub data/ (or worse). "
    "Run this command from a session in the canonical checkout instead."
)


def command_denied(command, satellite):
    """Return the deny message when a data-writing command runs in a satellite."""
    if not satellite:
        return None
    if _DATA_WRITERS.search(command or ""):
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
