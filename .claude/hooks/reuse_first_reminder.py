"""PreToolUse hook: nudge reuse-before-adding when new code is about to land.

Fires at most once per session per category (general code-add vs dependency
manifest), injecting an advisory `additionalContext` that points at the
`reuse-first` skill. Advisory only -- never blocks, never edits anything.
Mirrors stale_state_reminder.py.
"""

import json
import os
import re
import sys
import tempfile

THRESHOLD = 300  # net-added characters that count as "adding code" (tunable)

_SOURCE_EXT = re.compile(r"\.(py|ts|tsx|js|jsx|mjs)$")
_SOURCE_DIRS = ("src/", "web/src/", "scripts/", "tools/")
_DECL = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|const|interface)\s"
)

GENERAL_MSG = (
    "Hook reminder: you're about to add new code. Walk the reuse-first skill's "
    "ladder before writing more: (1) grep this repo for an existing helper -- "
    "artist_utils/string_utils/logging_utils/playlist.utils, identity -> "
    "normalization.py, genre reads -> authority.py, runtime/mode -> "
    "policy.py::derive_runtime_config, energy -> energy_loader.py; (2) stdlib / "
    "native; (3) an already-installed dependency. In a god-class hotspot, extract "
    "a helper instead of growing it. Do NOT minimize away diagnostics, "
    "validation, or config knobs. One reminder per session."
)

DEPENDENCY_MSG = (
    "Hook reminder: you're editing a dependency manifest. Adding a dependency is "
    "deliberate -- first confirm the stdlib, an already-installed dependency, and "
    "existing repo code can't cover this (reuse-first skill, rungs 3-4). One "
    "reminder per session."
)


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/")


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path


def _is_source(path: str) -> bool:
    if _is_test_path(path):
        return False
    if not _SOURCE_EXT.search(path):
        return False
    return any(d in path for d in _SOURCE_DIRS)


def _is_dep_manifest(path: str) -> bool:
    return path.endswith("pyproject.toml") or path.endswith("web/package.json")


def _added_text(tool_input: dict) -> tuple[int, str, str]:
    """Return (net_added_chars, new_text, old_text) for Write or Edit input."""
    if "content" in tool_input:  # Write
        new = tool_input.get("content") or ""
        return len(new), new, ""
    old = tool_input.get("old_string") or ""
    new = tool_input.get("new_string") or ""
    return len(new) - len(old), new, old


def _adds_declaration(new: str, old: str) -> bool:
    return len(_DECL.findall(new)) > len(_DECL.findall(old))


def build_message(data: dict) -> tuple[str, str] | None:
    """Pure decision logic: return (category, message) or None. No side effects."""
    tool_input = data.get("tool_input") or {}
    path = _norm(tool_input.get("file_path") or "")
    if not path:
        return None

    if _is_dep_manifest(path):
        return ("dependency", DEPENDENCY_MSG)

    if _is_source(path):
        net, new, old = _added_text(tool_input)
        if net >= THRESHOLD or _adds_declaration(new, old):
            return ("general", GENERAL_MSG)

    return None


def _already_fired(session_id: str, category: str) -> bool:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id) or "nosession"
    marker = os.path.join(
        tempfile.gettempdir(), f"claude_reuse_first_{safe}_{category}"
    )
    if os.path.exists(marker):
        return True
    try:
        with open(marker, "w", encoding="utf-8"):
            pass
    except OSError:
        pass
    return False


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    result = build_message(data)
    if result is None:
        return
    category, message = result
    if _already_fired(data.get("session_id") or "nosession", category):
        return
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": message,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
