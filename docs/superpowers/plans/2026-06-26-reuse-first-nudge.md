# Reuse-first nudge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PreToolUse advisory hook that, once per session, nudges the agent to reuse existing repo code / stdlib / installed deps before adding new code or a dependency, backed by a tuned `reuse-first` skill.

**Architecture:** A pure-logic Python hook (`reuse_first_reminder.py`, mirroring the existing `stale_state_reminder.py`) decides fire/no-fire from the PreToolUse tool-call JSON and emits `additionalContext`; it is registered in `.claude/settings.json` under an `Edit|Write` matcher and points at a new `reuse-first` skill that holds the reuse ladder tuned to this repo's utilities and god-class hotspots.

**Tech Stack:** Python 3.11+ (stdlib only — `json`, `os`, `re`, `sys`, `tempfile`), pytest, Claude Code hooks + skills.

## Global Constraints

- Python 3.11+ (uses `tuple[str, str] | None` syntax). Stdlib only — no new dependencies.
- Hook is **advisory only**: never blocks, never edits/restarts anything; emits at most once per session **per category** (`general`, `dependency`) via temp-dir marker files.
- Mirror the existing hook idiom in `.claude/hooks/stale_state_reminder.py`: read tool-call JSON from stdin, normalize paths (`\` → `/`), emit `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": <msg>}}`, fail silently on bad input.
- Net-add threshold = 300 characters (tunable constant `THRESHOLD`).
- Scope source detection to `.py/.ts/.tsx/.js/.jsx/.mjs` under `src/`, `web/src/`, `scripts/`, `tools/`; exclude any path under `tests/`.
- The skill must NOT encourage minimizing away diagnostics, quality metrics, audit scaffolding, validation, graceful fallbacks, or config knobs (Design Principles 20–21, 23, 25).
- pytest: run bounded, never piped through `tail`/`head` (CLAUDE.md session discipline).

---

### Task 1: Reuse-first hook (pure logic + unit test)

**Files:**
- Create: `.claude/hooks/reuse_first_reminder.py`
- Test: `tests/test_reuse_first_reminder.py`

**Interfaces:**
- Consumes: PreToolUse JSON on stdin — `{"session_id": str, "tool_name": "Edit"|"Write", "tool_input": {"file_path": str, "content"?: str, "old_string"?: str, "new_string"?: str}}`.
- Produces: `build_message(data: dict) -> tuple[str, str] | None` returning `(category, message)` where `category ∈ {"general", "dependency"}`, or `None` for no-fire. `main()` handles stdin, once-per-session marker, and stdout.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reuse_first_reminder.py`:

```python
"""Unit tests for the reuse-first PreToolUse advisory hook (build_message logic)."""

import importlib.util
import pathlib

_HOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "reuse_first_reminder.py"
)
_spec = importlib.util.spec_from_file_location("reuse_first_reminder", _HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _write(path, content):
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


def _edit(path, old, new):
    return {
        "tool_name": "Edit",
        "tool_input": {"file_path": path, "old_string": old, "new_string": new},
    }


def test_write_new_source_file_fires_general():
    data = _write("C:/repo/src/playlist/new_feature.py", "x = 1\n" * 100)
    result = hook.build_message(data)
    assert result is not None and result[0] == "general"


def test_trivial_edit_is_silent():
    data = _edit("C:/repo/src/playlist/pipeline.py", "x = 1", "x = 2")
    assert hook.build_message(data) is None


def test_edit_adding_a_def_fires_even_if_small():
    data = _edit(
        "C:/repo/src/playlist/pipeline.py",
        "y = 1",
        "y = 1\ndef helper():\n    return 1\n",
    )
    result = hook.build_message(data)
    assert result is not None and result[0] == "general"


def test_pyproject_edit_fires_dependency():
    data = _edit("C:/repo/pyproject.toml", "deps = []", 'deps = ["requests"]')
    result = hook.build_message(data)
    assert result is not None and result[0] == "dependency"


def test_web_package_json_fires_dependency():
    data = _edit("C:/repo/web/package.json", "{}", '{"x": 1}')
    assert hook.build_message(data)[0] == "dependency"


def test_markdown_file_is_silent():
    data = _write("C:/repo/docs/notes.md", "lots of text\n" * 100)
    assert hook.build_message(data) is None


def test_test_file_is_silent():
    data = _write("C:/repo/tests/test_thing.py", "def t():\n    assert True\n" * 50)
    assert hook.build_message(data) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reuse_first_reminder.py -q`
Expected: collection/execution error — `FileNotFoundError` / module load fails because `.claude/hooks/reuse_first_reminder.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `.claude/hooks/reuse_first_reminder.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reuse_first_reminder.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add .claude/hooks/reuse_first_reminder.py tests/test_reuse_first_reminder.py
git commit -m "feat(hooks): reuse-first PreToolUse advisory hook + tests"
```

---

### Task 2: The `reuse-first` skill

**Files:**
- Create: `.claude/skills/reuse-first/SKILL.md`

**Interfaces:**
- Consumes: nothing (static skill doc). The hook's `GENERAL_MSG` references this skill by name.
- Produces: a skill named `reuse-first` discoverable by the Skill tool; the named modules in the ladder must match the hook message.

- [ ] **Step 1: Create the skill file**

Create `.claude/skills/reuse-first/SKILL.md`:

```markdown
---
name: reuse-first
description: Use BEFORE writing new code, adding a function/module, or adding a dependency in this repo. Walks a reuse ladder — existing repo code, then stdlib/native, then an already-installed dependency — tuned to this project's utilities and god-class hotspots, before any new code is written.
---

# Reuse first

The best change reuses what already exists. Before writing new code, walk this
ladder and stop at the first rung that holds. This is *not* an excuse to skip
understanding the problem, and it never overrides this repo's instrumentation
discipline (see "Never minimize away" below).

## Rung 0 — Understand first

Read the code the change touches and trace the real flow before picking a rung.
A small diff you don't understand is guessing, not reuse. (Pairs with CLAUDE.md:
read the generation logs / grep before answering.)

## Rung 1 — Is it already in this repo?

`grep` before writing. High-value reuse targets that get reinvented:

- generic helpers: `src/artist_utils.py`, `src/string_utils.py`,
  `src/logging_utils.py`, `src/playlist/utils.py`
- identity / normalization: `src/ai_genre_enrichment/normalization.py`
- genre reads: `src/genre/authority.py` (and the `genre-data-authority` skill —
  every genre consumer must read through authority.py)
- runtime / mode config: `src/playlist_gui/policy.py::derive_runtime_config`
  (never hand-roll mode strings — standing gotcha)
- energy: `src/playlist/energy_loader.py`

## Rung 2 — Hotspot rule

If the change lands in a god-class — `src/playlist/pier_bridge_builder.py`,
`src/playlist_generator.py`, `src/playlist/pipeline.py`,
`src/playlist_gui/worker.py` — extract a helper. Do not grow the monolith
(CLAUDE.md Hotspots).

## Rung 3 — Stdlib / native?

`itertools`, `functools`, `dataclasses`, `pathlib`, `statistics`,
`collections`. Prefer numpy/scipy vectorization over hand-rolled loops — both
are already dependencies.

## Rung 4 — Already-installed dependency?

Check `pyproject.toml` / `web/package.json` before reaching for a new
dependency. Adding a dependency is a deliberate act — confirm rungs 1–3 can't
cover it first, then flag the addition explicitly.

## Rung 5 — Only then write new code

Write the minimum that works, following the patterns already in the file.

## Never minimize away

Reuse-first is about *not duplicating*, not about stripping a feature. Keep, in
full, regardless of rung:

- diagnostic logging, quality metrics, opt-in audit reports (Design Principles 20–21)
- input validation, error handling, graceful fallbacks (Principle 25)
- config knobs / tunability (Principle 23)
- anything the user explicitly asked to keep

## Output

After the change, note in 1–2 lines what existing code/stdlib/dep you reused, or
why new code was unavoidable.
```

- [ ] **Step 2: Verify the skill is well-formed**

Run: `python -c "import pathlib, re; t = pathlib.Path('.claude/skills/reuse-first/SKILL.md').read_text(encoding='utf-8'); m = re.match(r'^---\r?\n.*?name:\s*reuse-first.*?description:.*?\r?\n---', t, re.S); print('OK' if m else 'BAD FRONTMATTER'); assert m"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/reuse-first/SKILL.md
git commit -m "feat(skills): reuse-first ladder skill tuned to this repo"
```

---

### Task 3: Wire the hook into settings.json + verify end-to-end

**Files:**
- Modify: `.claude/settings.json` (add an `Edit|Write` matcher block under `hooks.PreToolUse`)

**Interfaces:**
- Consumes: the hook from Task 1 at `.claude/hooks/reuse_first_reminder.py`.
- Produces: a registered PreToolUse hook that runs on `Edit|Write`.

- [ ] **Step 1: Add the matcher block**

Edit `.claude/settings.json` so the `PreToolUse` array contains the new block alongside the existing `Bash|PowerShell` one. Final file:

```json
{
  "worktree": {
    "baseRef": "head",
    "symlinkDirectories": ["data", "web/node_modules"]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "python",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/pytest_pipe_guard.py"],
            "timeout": 15,
            "statusMessage": "Checking pytest pipe discipline"
          }
        ]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/reuse_first_reminder.py"],
            "timeout": 15,
            "statusMessage": "Checking reuse-first discipline"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/stale_state_reminder.py"],
            "timeout": 15,
            "statusMessage": "Checking for stale dist/worker state"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Verify settings.json is valid JSON**

Run: `python -c "import json; json.load(open('.claude/settings.json', encoding='utf-8')); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Verify the configured command emits the advisory end-to-end**

Run (general path — simulates a large source Write):
```bash
python -c "import json; print(json.dumps({'session_id': 'verify-sess', 'tool_name': 'Write', 'tool_input': {'file_path': 'src/playlist/zzz_demo.py', 'content': 'x = 1\n' * 100}}))" | python .claude/hooks/reuse_first_reminder.py
```
Expected: a JSON line containing `"hookEventName": "PreToolUse"` and the reuse-first reminder text.

Run (dependency path):
```bash
python -c "import json; print(json.dumps({'session_id': 'verify-sess2', 'tool_name': 'Edit', 'tool_input': {'file_path': 'pyproject.toml', 'old_string': 'a', 'new_string': 'a b'}}))" | python .claude/hooks/reuse_first_reminder.py
```
Expected: a JSON line containing the dependency reminder text.

Run (silent path — trivial edit):
```bash
python -c "import json; print(json.dumps({'session_id': 'verify-sess3', 'tool_name': 'Edit', 'tool_input': {'file_path': 'src/playlist/pipeline.py', 'old_string': 'x = 1', 'new_string': 'x = 2'}}))" | python .claude/hooks/reuse_first_reminder.py
```
Expected: no output.

> Note: the registered hook only takes effect in a **new** Claude Code session (hooks load at session start). The standalone runs above prove the wiring is correct now.

- [ ] **Step 4: Run the full hook test once more to confirm nothing regressed**

Run: `python -m pytest tests/test_reuse_first_reminder.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add .claude/settings.json
git commit -m "chore(hooks): register reuse-first hook on PreToolUse Edit|Write"
```

---

## Self-review

- **Spec coverage:** skill (Task 2) ✓; hook with general + dependency + net-add/declaration triggers and `tests/` exclusion (Task 1) ✓; settings wiring (Task 3) ✓; fire/no-fire unit test (Task 1) ✓; once-per-session-per-category markers (Task 1 `_already_fired`) ✓; known-limitation note carried in spec ✓.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `build_message(data) -> tuple[str, str] | None` and category strings `"general"`/`"dependency"` are used identically in the hook, the tests, and the Task 1 interface block.
