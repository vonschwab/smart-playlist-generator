"""PreToolUse hook: deny pytest commands piped through tail/head/Select-Object.

Piping pytest output has hung sessions (CLAUDE.md "Session discipline").
Reads the hook input JSON on stdin; emits a permissionDecision deny when a
pytest invocation is piped into a tail-like consumer. Silent otherwise.
"""

import json
import re
import sys

# pytest ... | tail / head / Select-Object / select -First|-Last
# The filler (?:[^;&]|&(?!&))* lets the match cross intermediate pipes
# (pytest | grep x | tail) and redirects (2>&1), but not command
# separators (pytest; echo done | tail / pytest && echo | tail).
_FILLER = r"(?:[^;&]|&(?!&))*"
_BLOCKED = re.compile(
    rf"pytest{_FILLER}\|\s*(tail|head|select-object)\b"
    rf"|pytest{_FILLER}\|\s*select\s+-(first|last)\b",
    re.IGNORECASE,
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    command = (data.get("tool_input") or {}).get("command") or ""
    if _BLOCKED.search(command):
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "Piping pytest through tail/head/Select-Object has hung "
                            "sessions (CLAUDE.md Session discipline). Run pytest "
                            "directly with -q and bound it with the tool's timeout "
                            "parameter instead of a pipe."
                        ),
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
