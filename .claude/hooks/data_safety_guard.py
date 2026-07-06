"""PreToolUse hook: deny destructive ops against irreplaceable data.

Guards the three targets CLAUDE.md marks as never-writable / irreplaceable:
  - data/metadata.db          the SQLite track DB (days to re-analyze)
  - data/archive/mert_2026/   archived MERT shards/sidecar (~55h CPU)
  - the audio library root     config.yaml `music_directory` (read-only forever)

Denies deletes, moves/renames, output redirects, PowerShell content-writes,
sqlite3 write statements, and python open(...,'w'/'a'/...) against those paths,
plus any Edit/Write tool call whose file_path lands on them. READS (SELECT,
cat/type, Get-Content, ffprobe) pass untouched, and metadata.db*.bak backups
pass so the backup discipline still works.

Contract mirrors the repo's other PreToolUse hooks: read the tool-call JSON on
stdin; emit a permissionDecision "deny" to block, stay silent to allow. FAIL
OPEN — any error returns without blocking (this is a speed bump for accidents,
not a security sandbox; it must never brick a session).
"""

import json
import os
import re
import sys

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

# --- protected path sub-patterns (matched against normalized lowercased text) ---
# metadata.db but NOT metadata.db.bak* (the `.`/word lookahead lets backups pass);
# `metadata.db-wal` / `-shm` still match, since they are live DB state.
DB_PAT = r"metadata\.db(?![.\w])"
ARCHIVE_PAT = r"data/archive/mert_2026"


def _audio_root():
    """The `music_directory` from config.yaml, normalized to lowercase forward-slash.

    Read at runtime so a config change is honored and nothing is hardcoded.
    Returns None if config.yaml is missing or still the example placeholder.
    """
    cfg = os.path.join(PROJECT_DIR, "config.yaml")
    try:
        with open(cfg, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"\s*music_directory:\s*(.+?)\s*$", line)
                if m:
                    val = m.group(1).strip().strip('"').strip("'")
                    if val and val != "/path/to/music":
                        return val.replace("\\", "/").rstrip("/").lower()
    except OSError:
        pass
    return None


def _protected():
    """List of {name, pattern, detail} — the audio entry is runtime-resolved."""
    out = [
        {
            "name": "data/metadata.db",
            "pattern": DB_PAT,
            "detail": (
                "the SQLite track DB — days to re-analyze and irreplaceable. Back "
                "it up to metadata.db.bak.<timestamp> and get explicit user sign-off "
                "before any write."
            ),
        },
        {
            "name": "the MERT archive (data/archive/mert_2026/)",
            "pattern": ARCHIVE_PAT,
            "detail": "archived embeddings, ~55h CPU to regenerate — never delete or overwrite.",
        },
    ]
    audio = _audio_root()
    if audio:
        out.append(
            {
                "name": "the audio library",
                "pattern": re.escape(audio) + r"(?:/|\b)",
                "detail": "music files are permanently read-only — never written, moved, renamed, or deleted.",
            }
        )
    return out


def _norm(text):
    return (text or "").replace("\\", "/").lower()


# Verb patterns take the protected path sub-pattern `P` via .format(P=...).
# Filler `[^|&;<>\n]*?` keeps a match inside one command segment (won't cross
# ; & | or a redirect) so `rm tmp; sqlite3 metadata.db "SELECT"` is not flagged.
_VERB_TEMPLATES = (
    r"\b(?:rm|del|erase|unlink|remove-item)\b[^|&;<>\n]*?{P}",  # delete
    r"\b(?:mv|move|rename|move-item|rename-item)\b[^|&;<>\n]*?{P}",  # move/rename
    r">>?\s*[\"']?[^\s\"'|&;<>]*{P}",  # output redirect onto the path
    r"\b(?:set-content|add-content|clear-content|out-file|new-item)\b[^|&;\n]*?{P}",  # PS writes
    r"open\s*\(\s*[a-z]*[\"'][^\"']*{P}[^\"']*[\"']\s*,\s*[\"'][^\"']*[wax+][^\"']*[\"']",  # py open write-mode
)
# sqlite3 <db> ... <write-verb>  (either order within one segment)
_SQL_WRITE = (
    r"sqlite3\b[^|&;\n]*(?:"
    r"{P}[^|&;\n]*\b(?:update|insert|delete|drop|alter|replace|vacuum|reindex|attach|create\s+(?:table|index|trigger|view))\b"
    r"|\b(?:update|insert|delete|drop|alter|replace|vacuum|reindex|attach)\b[^|&;\n]*{P}"
    r")"
)


def command_denied(command):
    """Return an entry dict if the (normalized) command is a destructive op, else None."""
    for entry in _protected():
        p = entry["pattern"]
        patterns = [t.format(P=p) for t in _VERB_TEMPLATES]
        patterns.append(_SQL_WRITE.format(P=p))
        for pat in patterns:
            if re.search(pat, command):
                return entry
    return None


def path_denied(file_path):
    """Return an entry dict if an Edit/Write target lands on a protected path."""
    p = _norm(file_path)
    if not p:
        return None
    for entry in _protected():
        if re.search(entry["pattern"], p):
            return entry
    return None


def _deny(entry):
    reason = (
        f"Blocked by the data-safety guard: this looks like a destructive operation "
        f"against {entry['name']} — {entry['detail']} "
        f"Reads (SELECT / cat / Get-Content / ffprobe) are NOT blocked. If this write "
        f"is intended and the user has approved it, back up first and run it outside "
        f"the guarded pattern, or temporarily disable the data_safety_guard hook."
    )
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


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    try:
        tool = data.get("tool_name") or ""
        tool_input = data.get("tool_input") or {}
        entry = None
        if tool in ("Bash", "PowerShell"):
            entry = command_denied(_norm(tool_input.get("command")))
        elif tool in ("Edit", "Write", "NotebookEdit"):
            entry = path_denied(tool_input.get("file_path"))
        if entry:
            _deny(entry)
    except Exception:
        return  # fail-open: never brick a session on a guard bug


if __name__ == "__main__":
    main()
