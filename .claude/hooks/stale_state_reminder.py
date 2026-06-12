"""PostToolUse hook: remind once per session when an edit makes served state stale.

- web/src edits  -> the served GUI runs web/dist; rebuild before judging behavior.
- src/ edits     -> a running serve_web.py worker still has the old code; restart.

Advisory only (injects context for the model); never blocks, never restarts
anything itself. Once-per-session-per-category via marker files in %TEMP%.
"""

import json
import os
import re
import sys
import tempfile


def _already_fired(session_id: str, category: str) -> bool:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id) or "nosession"
    marker = os.path.join(
        tempfile.gettempdir(), f"claude_stale_reminder_{safe}_{category}"
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
    file_path = ((data.get("tool_input") or {}).get("file_path") or "").replace(
        "\\", "/"
    )
    session_id = data.get("session_id") or "nosession"

    if "web/src/" in file_path:
        category = "dist"
        message = (
            "Hook reminder: web/src was edited, so web/dist is now stale. The served "
            "GUI runs dist — run `npm --prefix web run build` before judging GUI "
            "behavior (web-gui skill, Core Rule 1). One reminder per session."
        )
    elif "/src/" in file_path or file_path.startswith("src/"):
        category = "worker"
        message = (
            "Hook reminder: backend code under src/ was edited. A running "
            "serve_web.py worker still has the old code — restart it before judging "
            "runtime behavior (web-gui skill, Core Rule 2). One reminder per session."
        )
    else:
        return

    if _already_fired(session_id, category):
        return
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": message,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
