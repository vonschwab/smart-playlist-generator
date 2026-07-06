# Standing Satellite Clones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved satellite-clones design (`docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md`): standing full clones with absolute-path data access, canonical-only data writes (hook-enforced), workspace-aware git guard, scripted bootstrap, and a live acceptance gate.

**Architecture:** A shared `workspace_identity` helper (reads `.git/config` origin URL; local path ⇒ satellite) drives three consumers: a new `satellite_data_write_guard` hook, satellite-mode downgrades in the existing git guard, and a new doctor check. `tools/create_satellite.py` bootstraps a satellite (clone → config rewrite → local-file copies → npm → memory pointer → doctor). One app-code change: `mcp_sqlite_readonly.py` honors the configured DB path.

**Tech Stack:** Python 3.11 (stdlib + PyYAML), pytest, PowerShell-driven acceptance. No new dependencies.

## Global Constraints

- All implementation happens in the **canonical checkout on `master`** (`C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3`). Other sessions may have in-flight work: stage explicit paths only, and commit with `git commit --only -m "..." -- <paths>` — the live `git_shared_checkout_guard` hook DENIES bare `git commit` and `git add -A`, so the `--only -- <paths>` form is mandatory, not stylistic.
- Untracked files need `git add <path>` before `git commit --only -- <path>` will accept them.
- Pytest: run directly (`python -m pytest <file> -q`), never piped through `tail`/`head`; bound with the tool's timeout parameter.
- Hooks must **fail open**: any internal error returns silently (never brick a session). Every hook mirrors the existing stdin-JSON contract (see `.claude/hooks/pytest_pipe_guard.py`).
- Hook unit tests load the hook by file path via `importlib` (see `tests/test_git_shared_checkout_guard.py:14-21` for the exact pattern).
- Satellite names/paths fixed by the spec: `C:\Users\Dylan\Desktop\PG3_SAT1` (port 8771), `PG3_SAT2` (port 8772); canonical port 8770.
- Size floors (spec §2, pinned): resolved `metadata.db` ≥ 1 MB; resolved artifact ≥ 10 MB (stubs are 0 bytes; real files are ~100s of MB).
- Do NOT make `analyze_library.py` config-driven (spec: single-writer topology is the safety property).

---

### Task 1: `workspace_identity` shared helper

**Files:**
- Create: `.claude/hooks/workspace_identity.py`
- Test: `tests/test_workspace_identity.py`

**Interfaces:**
- Produces: `origin_url(project_dir: str|Path) -> str|None` and `is_satellite(project_dir: str|Path|None = None) -> bool`. `is_satellite` defaults `project_dir` to `CLAUDE_PROJECT_DIR` env or cwd. Local-path origin (drive letter, UNC, `/`-rooted, or `file://`) ⇒ `True`; `http(s)://`, `git@`, `ssh://`, `git://`, or no origin ⇒ `False` (unknown states resolve toward canonical = strictest guard behavior). Tasks 2, 3, 5 import this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workspace_identity.py`:

```python
"""Unit tests for the shared workspace-detection helper (canonical vs satellite)."""

import importlib.util
import pathlib

_MOD = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "workspace_identity.py"
)
_spec = importlib.util.spec_from_file_location("workspace_identity", _MOD)
assert _spec is not None and _spec.loader is not None, f"helper not found at {_MOD}"
wsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsi)


def _repo_with_origin(tmp_path, url):
    git = tmp_path / ".git"
    git.mkdir()
    body = "[core]\n\trepositoryformatversion = 0\n"
    if url is not None:
        body += f'[remote "origin"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    (git / "config").write_text(body, encoding="utf-8")
    return tmp_path


def test_github_https_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, "https://github.com/vonschwab/playlist-generator.git")
    assert wsi.is_satellite(repo) is False


def test_github_ssh_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, "git@github.com:vonschwab/playlist-generator.git")
    assert wsi.is_satellite(repo) is False


def test_windows_path_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_backslash_path_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_file_url_origin_is_satellite(tmp_path):
    repo = _repo_with_origin(tmp_path, "file:///C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
    assert wsi.is_satellite(repo) is True


def test_no_origin_is_canonical(tmp_path):
    repo = _repo_with_origin(tmp_path, None)
    assert wsi.is_satellite(repo) is False


def test_missing_git_dir_is_canonical(tmp_path):
    assert wsi.is_satellite(tmp_path) is False


def test_origin_url_extraction(tmp_path):
    repo = _repo_with_origin(tmp_path, "https://github.com/x/y.git")
    assert wsi.origin_url(repo) == "https://github.com/x/y.git"


def test_this_repo_is_canonical():
    # The canonical checkout's origin is GitHub — detection must say canonical.
    assert wsi.is_satellite(pathlib.Path(__file__).resolve().parents[1]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workspace_identity.py -q` (timeout 60s)
Expected: FAIL at module load — `helper not found at ...workspace_identity.py`

- [ ] **Step 3: Write the implementation**

Create `.claude/hooks/workspace_identity.py`:

```python
"""Workspace detection shared by guard hooks and doctor: canonical vs satellite.

A SATELLITE is a standing full clone of the canonical checkout whose git
`origin` is a LOCAL path (see docs/superpowers/specs/
2026-07-06-simultaneous-sessions-design.md). The canonical checkout's origin
is GitHub. Detection reads `.git/config` directly (no subprocess — this runs
inside PreToolUse hooks on every tool call).

Unknown states (no .git, no origin, unreadable config) report CANONICAL:
that resolves toward the strictest guard behavior, never toward allowing a
satellite-only action in the wrong place.
"""

import os
import re
from pathlib import Path

_ORIGIN_SECTION = re.compile(r'^\s*\[remote\s+"origin"\]\s*$')
_ANY_SECTION = re.compile(r"^\s*\[")
_URL_LINE = re.compile(r"^\s*url\s*=\s*(.+?)\s*$")
_NONLOCAL_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")
_LOCAL_PATH = re.compile(r"^[a-zA-Z]:[\\/]|^\\\\|^/")  # drive, UNC, posix root


def origin_url(project_dir):
    """The `[remote "origin"] url` from .git/config, or None."""
    cfg = Path(project_dir) / ".git" / "config"
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_origin = False
    for line in text.splitlines():
        if _ANY_SECTION.match(line):
            in_origin = bool(_ORIGIN_SECTION.match(line))
            continue
        if in_origin:
            m = _URL_LINE.match(line)
            if m:
                return m.group(1)
    return None


def is_satellite(project_dir=None):
    """True iff this checkout's origin is a local filesystem path."""
    root = project_dir or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    url = origin_url(root)
    if not url:
        return False
    lowered = url.lower()
    if lowered.startswith(_NONLOCAL_PREFIXES):
        return False
    if lowered.startswith("file://"):
        return True
    return bool(_LOCAL_PATH.match(url))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workspace_identity.py -q` (timeout 60s)
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add .claude/hooks/workspace_identity.py tests/test_workspace_identity.py
git commit --only -m "feat(hooks): workspace_identity helper - canonical vs satellite detection via origin URL" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- .claude/hooks/workspace_identity.py tests/test_workspace_identity.py
```

---

### Task 2: `satellite_data_write_guard` hook + registration

**Files:**
- Create: `.claude/hooks/satellite_data_write_guard.py`
- Modify: `.claude/settings.json` (add one PreToolUse entry)
- Test: `tests/test_satellite_data_write_guard.py`

**Interfaces:**
- Consumes: `workspace_identity.is_satellite()` (Task 1).
- Produces: `command_denied(command: str, satellite: bool) -> str|None` (message when denied) — used by tests; `main()` reads the hook stdin JSON, checks `tool_name in ("Bash","PowerShell")`, and emits a PreToolUse `permissionDecision: deny` when denied.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_satellite_data_write_guard.py`:

```python
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
]

ALWAYS_ALLOWED = [
    "python -m pytest -q",
    "python main_app.py --artist X --tracks 30",
    "python tools/serve_web.py --port 8771",
    "python tools/doctor.py",
    "git status",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_satellite_data_write_guard.py -q` (timeout 60s)
Expected: FAIL at module load — hook not found

- [ ] **Step 3: Write the hook**

Create `.claude/hooks/satellite_data_write_guard.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_satellite_data_write_guard.py -q` (timeout 60s)
Expected: 5 passed

- [ ] **Step 5: Register the hook**

In `.claude/settings.json`, the `PreToolUse` array currently starts with the `data_safety_guard` entry (matcher `"Bash|PowerShell|Edit|Write|NotebookEdit"`). Insert this entry immediately AFTER the data_safety_guard entry (and before the pytest_pipe_guard entry):

```json
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "python",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/satellite_data_write_guard.py"],
            "timeout": 15,
            "statusMessage": "Guarding canonical-only data writes"
          }
        ]
      },
```

Validate: `python -c "import json; json.load(open('.claude/settings.json')); print('valid')"`
Expected: `valid`

- [ ] **Step 6: Commit**

```bash
git add .claude/hooks/satellite_data_write_guard.py tests/test_satellite_data_write_guard.py .claude/settings.json
git commit --only -m "feat(hooks): satellite data-write guard - pipeline data writes are canonical-only" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- .claude/hooks/satellite_data_write_guard.py tests/test_satellite_data_write_guard.py .claude/settings.json
```

---

### Task 3: Git guard satellite mode

**Files:**
- Modify: `.claude/hooks/git_shared_checkout_guard.py`
- Test: extend `tests/test_git_shared_checkout_guard.py`

**Interfaces:**
- Consumes: `workspace_identity.is_satellite()` (Task 1).
- Produces: `analyze(command: str, satellite: bool = False)` and `analyze_segment(tokens, satellite=False)` — existing return contract `('deny'|'warn', msg) | None` unchanged. Existing callers/tests that omit `satellite` keep current strict behavior.

**Behavior matrix (spec §5):**

| Rule | Canonical | Satellite |
|---|---|---|
| `git add -A/-u/.` etc. | deny | **warn** |
| `git commit -a` / bare commit | deny | **warn** |
| `git checkout .` / `git restore .` | deny | **warn** |
| `git reset --hard` | deny | deny |
| `git clean -f` | deny | deny |
| `git switch` / `checkout -b` | warn | **silent** |

Satellite warns are emitted at most once per session per category (marker-file pattern copied from `stale_state_reminder.py`), with satellite-appropriate wording (the tree is private; the discipline is habit-preservation, not protection).

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_git_shared_checkout_guard.py`:

```python
# ------------------------------ satellite mode ------------------------------
# In a satellite clone the index is private: sweepers downgrade to warnings,
# destroyers stay denied, branch-switch warnings disappear (spec 2026-07-06 §5).

def _sat_verdict(command):
    result = hook.analyze(command, satellite=True)
    return result[0] if result else None


def test_satellite_add_all_warns_not_denies():
    assert _sat_verdict("git add -A") == "warn"


def test_satellite_bare_commit_warns_not_denies():
    assert _sat_verdict('git commit -m "x"') == "warn"


def test_satellite_commit_a_warns_not_denies():
    assert _sat_verdict('git commit -am "x"') == "warn"


def test_satellite_checkout_dot_warns_not_denies():
    assert _sat_verdict("git checkout .") == "warn"


def test_satellite_reset_hard_still_denied():
    assert _sat_verdict("git reset --hard") == "deny"


def test_satellite_clean_f_still_denied():
    assert _sat_verdict("git clean -fd") == "deny"


def test_satellite_switch_is_silent():
    assert _sat_verdict("git switch feature-x") is None


def test_satellite_checkout_new_branch_is_silent():
    assert _sat_verdict("git checkout -b feature-x") is None


def test_satellite_safe_forms_still_silent():
    assert _sat_verdict("git add src/foo.py") is None
    assert _sat_verdict("git commit --only -- src/foo.py -m 'x'") is None


def test_default_analyze_unchanged_strict():
    # No-satellite callers keep canonical strictness (regression guard).
    assert _verdict("git add -A") == "deny"
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_git_shared_checkout_guard.py -q` (timeout 60s)
Expected: existing tests pass; the new satellite tests FAIL (`analyze() got an unexpected keyword argument 'satellite'`)

- [ ] **Step 3: Implement satellite mode**

In `.claude/hooks/git_shared_checkout_guard.py`:

(a) After the existing imports, add:

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workspace_identity import is_satellite  # noqa: E402
```

(b) Add the satellite warning text next to the other message constants:

```python
_SAT_WARN = (
    "Hook note (satellite clone): this tree is private so nothing else is at "
    "risk, but broad staging/committing is a habit that corrupts work in the "
    "SHARED canonical checkout. Prefer `git add <paths>` + `git commit --only "
    "-- <paths>` everywhere. One reminder per session."
)
```

(c) Change `analyze_segment(tokens)` to `analyze_segment(tokens, satellite=False)` and adjust returns:
- `add` broad-selector branch: `return ("warn", _SAT_WARN) if satellite else ("deny", ...existing message...)`
- `commit -a/--all` branch: same pattern.
- bare-commit branch: same pattern (keep the `has_pathspec`/`conclusion`/merge-in-progress conditions unchanged).
- `checkout`/`restore` with `"."`: same pattern.
- `reset --hard` and `clean -f`: unchanged (deny in both modes).
- `checkout -b/-B` warn and `switch` warn: `return None if satellite else ("warn", ...existing message...)`.

(d) Change `analyze(command)` to `analyze(command, satellite=False)` and pass it through: `analyze_segment(_tokens(seg), satellite)`.

(e) In `main()`: compute `satellite = is_satellite()` before analyzing; call `analyze(command, satellite)`. For a `warn` result **when satellite is True**, gate once-per-session with the marker-file pattern (copy `_already_fired` verbatim from `.claude/hooks/stale_state_reminder.py`, marker prefix `claude_git_guard_sat_`, category `"discipline"`, session id from `data.get("session_id")`); canonical warns stay ungated (current behavior).

- [ ] **Step 4: Run the full guard test file**

Run: `python -m pytest tests/test_git_shared_checkout_guard.py -q` (timeout 60s)
Expected: all pass (existing + 10 new)

- [ ] **Step 5: Commit**

```bash
git add .claude/hooks/git_shared_checkout_guard.py tests/test_git_shared_checkout_guard.py
git commit --only -m "feat(hooks): git guard satellite mode - private-tree downgrades, destroyers stay denied" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- .claude/hooks/git_shared_checkout_guard.py tests/test_git_shared_checkout_guard.py
```

---

### Task 4: `mcp_sqlite_readonly.py` honors configured DB path

**Files:**
- Modify: `tools/mcp_sqlite_readonly.py:34-51`
- Test: `tests/test_mcp_sqlite_db_map.py`

**Interfaces:**
- Produces: `_config_db_path(root: Path) -> Path|None` (resolved `library.database_path` from `<root>/config.yaml`, absolute-safe) and `_load_db_map(root: Path = _REPO_ROOT) -> dict[str, Path]`. Default map: `metadata` from config (fallback `<root>/data/metadata.db`), `enrichment` = sibling `ai_genre_enrichment.db` in the SAME directory as the resolved metadata DB (so a satellite pointing at canonical data gets canonical enrichment too). `SQLITE_RO_DBS` env override unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_sqlite_db_map.py`:

```python
"""Unit tests: the read-only SQLite MCP resolves the DB from config.yaml.

Guards against the satellite stub trap: a clone's tracked data/metadata.db is
a 0-byte placeholder; the MCP must follow config's absolute path instead.
"""

import importlib.util
import pathlib

_MOD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "mcp_sqlite_readonly.py"
_spec = importlib.util.spec_from_file_location("mcp_sqlite_readonly", _MOD)
assert _spec is not None and _spec.loader is not None
mcp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcp_mod)


def test_config_absolute_db_path_wins(tmp_path):
    canonical_db = tmp_path / "elsewhere" / "metadata.db"
    canonical_db.parent.mkdir()
    canonical_db.write_bytes(b"x")
    (tmp_path / "config.yaml").write_text(
        f"library:\n  database_path: {canonical_db.as_posix()}\n", encoding="utf-8"
    )
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == canonical_db.resolve()
    assert dbs["enrichment"] == (canonical_db.parent / "ai_genre_enrichment.db").resolve()


def test_relative_config_path_resolves_against_root(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "library:\n  database_path: data/metadata.db\n", encoding="utf-8"
    )
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == (tmp_path / "data" / "metadata.db").resolve()


def test_missing_config_falls_back_to_default(tmp_path):
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == (tmp_path / "data" / "metadata.db").resolve()
    assert dbs["enrichment"] == (tmp_path / "data" / "ai_genre_enrichment.db").resolve()


def test_env_override_still_wins(tmp_path, monkeypatch):
    override = tmp_path / "other.db"
    monkeypatch.setenv("SQLITE_RO_DBS", f'{{"metadata": "{override.as_posix()}"}}')
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == override.resolve()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_sqlite_db_map.py -q` (timeout 60s)
Expected: FAIL — `_load_db_map() got an unexpected keyword argument 'root'` (and no `_config_db_path`)

- [ ] **Step 3: Implement**

In `tools/mcp_sqlite_readonly.py`, replace the `_load_db_map` block (lines 38-51) with:

```python
def _config_db_path(root: Path) -> Path | None:
    """library.database_path from <root>/config.yaml, resolved; None if unset."""
    cfg = root / "config.yaml"
    try:
        import yaml

        with open(cfg, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw = str(((data.get("library") or {}).get("database_path")) or "").strip()
        if not raw:
            return None
        p = Path(raw)
        return (p if p.is_absolute() else root / p).resolve()
    except Exception:
        return None  # unreadable config -> fall back to default path


def _load_db_map(root: Path = _REPO_ROOT) -> dict[str, Path]:
    """Name -> absolute path for every queryable DB. cwd-independent.

    Honors config.yaml `library.database_path` (satellites point it at the
    canonical checkout by absolute path — the tracked data/metadata.db in a
    clone is a 0-byte stub). `enrichment` lives beside the resolved metadata
    DB so both follow the same data directory.
    """
    raw = os.environ.get("SQLITE_RO_DBS")
    if raw:
        mapping = {name: Path(p) for name, p in json.loads(raw).items()}
    else:
        meta = _config_db_path(root) or (root / "data" / "metadata.db")
        mapping = {
            "metadata": meta,
            "enrichment": meta.parent / "ai_genre_enrichment.db",
        }
    return {name: p.resolve() for name, p in mapping.items()}
```

(The module-level `_DBS = _load_db_map()` call at line 51 stays as-is — it now uses the default `root=_REPO_ROOT`.)

- [ ] **Step 4: Run tests + confirm the live MCP still resolves canonical paths**

Run: `python -m pytest tests/test_mcp_sqlite_db_map.py -q` (timeout 60s)
Expected: 4 passed

Run: `python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('m', 'tools/mcp_sqlite_readonly.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m._DBS['metadata'])"`
Expected: `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db` (canonical config is relative `data/metadata.db` → resolves against repo root; unchanged behavior)

- [ ] **Step 5: Commit**

```bash
git add tools/mcp_sqlite_readonly.py tests/test_mcp_sqlite_db_map.py
git commit --only -m "fix(mcp): sqlite-ro resolves DB from config.yaml - kills the satellite stub-query trap" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- tools/mcp_sqlite_readonly.py tests/test_mcp_sqlite_db_map.py
```

---

### Task 5: Doctor satellite check

**Files:**
- Modify: `tools/doctor.py` (new method on `DoctorChecks` + one call in `main()`)
- Test: `tests/test_doctor_satellite.py`

**Interfaces:**
- Consumes: `workspace_identity` (loaded by file path via importlib — `tools/` is not on the hooks' path).
- Produces: `DoctorChecks.check_satellite_data_paths(root: Path = ROOT_DIR) -> bool`. In canonical: single pass line. In a satellite: FAILS unless (1) config.yaml exists, (2) `library.database_path` is written as an absolute path, resolves outside `root`, exists, and is ≥ 1 MB, (3) `playlists.ds_pipeline.artifact_path` is written as an absolute path, resolves outside `root`, exists, and is ≥ 10 MB. Task 6's bootstrap runs doctor and requires exit 0.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doctor_satellite.py`:

```python
"""Unit tests for doctor's satellite data-path check (stub-landmine gate)."""

import importlib.util
import pathlib
import sys

_DOCTOR = pathlib.Path(__file__).resolve().parents[1] / "tools" / "doctor.py"
_spec = importlib.util.spec_from_file_location("doctor", _DOCTOR)
assert _spec is not None and _spec.loader is not None
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)


def _make_satellite(tmp_path, db_bytes=2 * 1024 * 1024, art_bytes=11 * 1024 * 1024,
                    db_abs=True, art_abs=True):
    """A fake satellite clone + a fake canonical data dir; returns (sat_root, cfg_text)."""
    sat = tmp_path / "sat"
    (sat / ".git").mkdir(parents=True)
    (sat / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = C:/canonical\n', encoding="utf-8"
    )
    canon = tmp_path / "canonical" / "data"
    canon.mkdir(parents=True)
    db = canon / "metadata.db"
    db.write_bytes(b"\0" * db_bytes)
    art = canon / "artifact.npz"
    art.write_bytes(b"\0" * art_bytes)
    db_val = db.as_posix() if db_abs else "data/metadata.db"
    art_val = art.as_posix() if art_abs else "data/artifacts/x.npz"
    (sat / "config.yaml").write_text(
        f"library:\n  database_path: {db_val}\n"
        f"playlists:\n  ds_pipeline:\n    artifact_path: {art_val}\n",
        encoding="utf-8",
    )
    return sat


def _run_check(sat_root):
    checks = doctor.DoctorChecks()
    ok = checks.check_satellite_data_paths(root=sat_root)
    return ok, checks


def test_valid_satellite_passes(tmp_path):
    sat = _make_satellite(tmp_path)
    ok, checks = _run_check(sat)
    assert ok is True and checks.failed == 0


def test_relative_db_path_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_abs=False)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_stub_sized_db_fails(tmp_path):
    sat = _make_satellite(tmp_path, db_bytes=0)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_small_artifact_fails(tmp_path):
    sat = _make_satellite(tmp_path, art_bytes=1024)
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_db_inside_satellite_fails(tmp_path):
    sat = _make_satellite(tmp_path)
    inside = sat / "data"
    inside.mkdir()
    (inside / "metadata.db").write_bytes(b"\0" * (2 * 1024 * 1024))
    art = (tmp_path / "canonical" / "data" / "artifact.npz").as_posix()
    (sat / "config.yaml").write_text(
        f"library:\n  database_path: {(inside / 'metadata.db').as_posix()}\n"
        f"playlists:\n  ds_pipeline:\n    artifact_path: {art}\n",
        encoding="utf-8",
    )
    ok, checks = _run_check(sat)
    assert ok is False and checks.failed >= 1


def test_canonical_workspace_passes_trivially(tmp_path):
    canon = tmp_path / "repo"
    (canon / ".git").mkdir(parents=True)
    (canon / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/x/y.git\n', encoding="utf-8"
    )
    ok, checks = _run_check(canon)
    assert ok is True and checks.failed == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_doctor_satellite.py -q` (timeout 60s)
Expected: FAIL — `DoctorChecks` has no attribute `check_satellite_data_paths`

- [ ] **Step 3: Implement the check**

In `tools/doctor.py`, add to `DoctorChecks` (after `check_artifacts`, ~line 232):

```python
    def check_satellite_data_paths(self, root: Path = ROOT_DIR) -> bool:
        """Satellite clones must reach REAL data via absolute canonical paths.

        A fresh clone's data/metadata.db is a tracked 0-byte stub; running with
        it silently degrades generation (BPM gates disable inside a try/except).
        Spec: docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md §2.
        """
        import importlib.util as _ilu

        # Load the helper from THIS repo (doctor's own location), not from `root`:
        # in tests `root` is a bare fake satellite dir that has no .claude/hooks.
        wsi_path = ROOT_DIR / ".claude" / "hooks" / "workspace_identity.py"
        if not wsi_path.exists():
            # canonical fallback for odd layouts: nothing to enforce
            check_pass("Workspace: canonical (no workspace_identity helper)")
            self.passed += 1
            return True
        spec = _ilu.spec_from_file_location("workspace_identity", wsi_path)
        wsi = _ilu.module_from_spec(spec)
        spec.loader.exec_module(wsi)

        if not wsi.is_satellite(root):
            check_pass("Workspace: canonical checkout (satellite checks n/a)")
            self.passed += 1
            return True

        ok = True
        cfg_path = root / "config.yaml"
        if not cfg_path.exists():
            check_fail("Satellite has no config.yaml",
                       "python tools/create_satellite.py rewrites one from canonical")
            self.failed += 1
            return False
        import yaml
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        floors = {"database_path": 1 * 1024 * 1024, "artifact_path": 10 * 1024 * 1024}
        raw_db = str(((cfg.get("library") or {}).get("database_path")) or "")
        raw_art = str((((cfg.get("playlists") or {}).get("ds_pipeline") or {})
                       .get("artifact_path")) or "")
        for label, raw, floor in (
            ("database_path", raw_db, floors["database_path"]),
            ("artifact_path", raw_art, floors["artifact_path"]),
        ):
            p = Path(raw) if raw else None
            if p is None or not p.is_absolute():
                check_fail(f"Satellite {label} must be an ABSOLUTE canonical path (got {raw!r})",
                           "Point it at the canonical checkout's data/ (create_satellite.py does this)")
                self.failed += 1
                ok = False
                continue
            resolved = p.resolve()
            if resolved.is_relative_to(root.resolve()):
                check_fail(f"Satellite {label} resolves INSIDE this clone ({resolved}) — that's the stub")
                self.failed += 1
                ok = False
                continue
            if not resolved.exists():
                check_fail(f"Satellite {label} target missing: {resolved}")
                self.failed += 1
                ok = False
                continue
            size = resolved.stat().st_size
            if size < floor:
                check_fail(f"Satellite {label} target suspiciously small ({size} bytes < {floor}) — stub?")
                self.failed += 1
                ok = False
                continue
            check_pass(f"Satellite {label}: {resolved} ({size / (1024*1024):.0f} MB)")
            self.passed += 1
        return ok
```

Wire it into `main()`: after the `doctor.check_config_file()` call (~line 291), add:

```python
    doctor.check_satellite_data_paths()
```

- [ ] **Step 4: Run tests + the live doctor**

Run: `python -m pytest tests/test_doctor_satellite.py -q` (timeout 60s)
Expected: 6 passed

Run: `python tools/doctor.py` (timeout 120s)
Expected: exit 0, output includes `Workspace: canonical checkout (satellite checks n/a)`; no regressions in other checks

- [ ] **Step 5: Commit**

```bash
git add tools/doctor.py tests/test_doctor_satellite.py
git commit --only -m "feat(doctor): satellite data-path check - stub landmine gate (absolute, outside-clone, size floors)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- tools/doctor.py tests/test_doctor_satellite.py
```

---

### Task 6: `tools/create_satellite.py` bootstrap

**Files:**
- Create: `tools/create_satellite.py`
- Test: `tests/test_create_satellite.py`

**Interfaces:**
- Consumes: doctor (Task 5) as the final gate; canonical root derived from the script's own location.
- Produces: CLI `python tools/create_satellite.py --name PG3_SAT1 --port 8771 [--dest-root <dir>]` plus pure functions used by tests: `rewrite_config_text(text: str, canonical_root: Path) -> str`, `memory_project_key(path_str: str) -> str`, `memory_pointer_text(canonical_root: Path) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_create_satellite.py`:

```python
"""Unit tests for the satellite bootstrap's pure functions (config rewrite, memory key)."""

import importlib.util
import pathlib

_MOD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "create_satellite.py"
_spec = importlib.util.spec_from_file_location("create_satellite", _MOD)
assert _spec is not None and _spec.loader is not None
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

CANON = pathlib.Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
DB_LINE = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"
ART_LINE = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"


def test_db_path_rewritten_and_comments_preserved():
    text = (
        "library:\n"
        "  music_directory: E:\\MUSIC  # my library\n"
        "  database_path: data/metadata.db\n"
        "# a comment that must survive\n"
    )
    out = cs.rewrite_config_text(text, CANON)
    assert f"  database_path: {DB_LINE}\n" in out
    assert "# a comment that must survive" in out
    assert "music_directory: E:\\MUSIC  # my library" in out


def test_artifact_path_replaced_when_present():
    text = (
        "library:\n  database_path: data/metadata.db\n"
        "playlists:\n  ds_pipeline:\n    artifact_path: data/artifacts/beat3tower_32k/data_matrices_step1.npz\n"
    )
    out = cs.rewrite_config_text(text, CANON)
    assert f"    artifact_path: {ART_LINE}\n" in out


def test_artifact_key_inserted_under_existing_ds_pipeline():
    text = "playlists:\n  ds_pipeline:\n    enabled: true\nlibrary:\n  database_path: data/metadata.db\n"
    out = cs.rewrite_config_text(text, CANON)
    assert f"    artifact_path: {ART_LINE}\n" in out
    # inserted directly after the ds_pipeline: line
    assert out.index("ds_pipeline:") < out.index("artifact_path:") < out.index("enabled: true")


def test_playlists_block_appended_when_absent():
    text = "library:\n  database_path: data/metadata.db\n"
    out = cs.rewrite_config_text(text, CANON)
    assert "playlists:\n  ds_pipeline:\n    artifact_path: " + ART_LINE in out


def test_memory_project_key_matches_harness_munging():
    # Known ground truth: this project's own memory dir name.
    assert (
        cs.memory_project_key("C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3")
        == "C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3"
    )
    assert (
        cs.memory_project_key("C:\\Users\\Dylan\\Desktop\\PG3_SAT1")
        == "C--Users-Dylan-Desktop-PG3-SAT1"
    )


def test_memory_pointer_names_canonical_index():
    text = cs.memory_pointer_text(CANON)
    assert "C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3" in text
    assert "MEMORY.md" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_create_satellite.py -q` (timeout 60s)
Expected: FAIL at module load — `tools/create_satellite.py` not found

- [ ] **Step 3: Write the bootstrap script**

Create `tools/create_satellite.py`:

```python
#!/usr/bin/env python3
"""Bootstrap a standing satellite clone for simultaneous Claude sessions.

Spec: docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md (§7).

    python tools/create_satellite.py --name PG3_SAT1 --port 8771

Steps: git clone -> config.yaml copy with absolute canonical data paths ->
copy untracked local config (.claude/settings.local.json, .mcp.json) ->
npm install + build -> auto-memory pointer -> doctor gate (fails loudly).
Satellites NEVER get links/junctions of any kind, and data-writing pipeline
stages stay canonical-only (satellite_data_write_guard enforces this).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

CANONICAL_ROOT = Path(__file__).resolve().parent.parent


def rewrite_config_text(text: str, canonical_root: Path) -> str:
    """Point database_path + ds_pipeline.artifact_path at canonical, by absolute path.

    Targeted line edits only — the rest of the file (comments included) is
    preserved byte-for-byte. Handles: artifact_path present; ds_pipeline
    present without artifact_path; playlists absent entirely.
    """
    db_abs = (canonical_root / "data" / "metadata.db").as_posix()
    art_abs = (
        canonical_root / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
    ).as_posix()

    out = re.sub(
        r"(?m)^(\s*)database_path:.*$",
        rf"\g<1>database_path: {db_abs}",
        text,
        count=1,
    )

    if re.search(r"(?m)^\s*artifact_path:", out):
        out = re.sub(
            r"(?m)^(\s*)artifact_path:.*$",
            rf"\g<1>artifact_path: {art_abs}",
            out,
            count=1,
        )
    elif (m := re.search(r"(?m)^(\s*)ds_pipeline:\s*$", out)):
        indent = m.group(1) + "  "
        insert_at = m.end()
        out = out[:insert_at] + f"\n{indent}artifact_path: {art_abs}" + out[insert_at:]
    else:
        block = f"playlists:\n  ds_pipeline:\n    artifact_path: {art_abs}\n"
        out = out.rstrip("\n") + "\n" + block
    return out


def memory_project_key(path_str: str) -> str:
    """The harness's project-dir munging: [:\\/_] -> '-' (verified against this repo's key)."""
    return re.sub(r"[:\\/_]", "-", path_str)


def memory_pointer_text(canonical_root: Path) -> str:
    canon_key = memory_project_key(str(canonical_root))
    canon_memory = Path.home() / ".claude" / "projects" / canon_key / "memory"
    return (
        "<!-- Satellite clone pointer: this workspace shares the canonical project's memory. -->\n"
        f"- [CANONICAL MEMORY — read this first](file://{(canon_memory / 'MEMORY.md').as_posix()}) — "
        "this satellite has no memory of its own. Read the canonical MEMORY.md index at "
        f"`{canon_memory / 'MEMORY.md'}` at session start, and write any new memories into "
        f"`{canon_memory}` (absolute path), not this directory.\n"
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        sys.exit(f"FAILED ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="satellite dir name, e.g. PG3_SAT1")
    ap.add_argument("--port", type=int, required=True, help="GUI port, e.g. 8771")
    ap.add_argument("--dest-root", default=str(CANONICAL_ROOT.parent),
                    help="parent dir for the clone (default: canonical's parent)")
    args = ap.parse_args()

    sat = Path(args.dest_root) / args.name
    if sat.exists():
        sys.exit(f"Refusing: {sat} already exists (satellites are standing; delete manually first).")

    print(f"[1/6] Cloning canonical -> {sat}")
    _run(["git", "clone", str(CANONICAL_ROOT), str(sat)])

    print("[2/6] Writing satellite config.yaml (absolute canonical data paths)")
    canon_cfg = CANONICAL_ROOT / "config.yaml"
    if not canon_cfg.exists():
        sys.exit("Canonical config.yaml missing — cannot derive satellite config.")
    (sat / "config.yaml").write_text(
        rewrite_config_text(canon_cfg.read_text(encoding="utf-8"), CANONICAL_ROOT),
        encoding="utf-8",
    )

    print("[3/6] Copying untracked local config (settings.local.json, .mcp.json)")
    for rel in (Path(".claude") / "settings.local.json", Path(".mcp.json")):
        src = CANONICAL_ROOT / rel
        if src.exists():
            dest = sat / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    print("[4/6] npm install + build (one-time; a few minutes)")
    npm = shutil.which("npm") or "npm"
    _run([npm, "--prefix", str(sat / "web"), "install"])
    _run([npm, "--prefix", str(sat / "web"), "run", "build"])

    print("[5/6] Auto-memory pointer")
    mem_dir = Path.home() / ".claude" / "projects" / memory_project_key(str(sat)) / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    pointer = mem_dir / "MEMORY.md"
    if not pointer.exists():
        pointer.write_text(memory_pointer_text(CANONICAL_ROOT), encoding="utf-8")

    print("[6/6] Doctor gate (satellite data-path checks)")
    _run([sys.executable, "tools/doctor.py"], cwd=sat)

    print(
        f"\nSatellite ready: {sat}\n"
        f"  Launch Claude Code with cwd = {sat}  (never switch into it mid-session)\n"
        f"  GUI: python tools/serve_web.py --port {args.port}\n"
        f"  Work on feature branches; land via: git push origin <branch>, then merge in canonical.\n"
        f"  Data writes (analyze/publish/folds) stay in canonical — the guard will remind you."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_create_satellite.py -q` (timeout 60s)
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tools/create_satellite.py tests/test_create_satellite.py
git commit --only -m "feat(tools): create_satellite.py - one-shot satellite clone bootstrap with doctor gate" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- tools/create_satellite.py tests/test_create_satellite.py
```

---

### Task 7: CLAUDE.md three-workspace doctrine

**Files:**
- Modify: `CLAUDE.md` ("Session discipline", the three shared-checkout bullets committed in `5999580`)

**Interfaces:** none (docs).

- [ ] **Step 1: Apply the edit**

Find these exact three bullets:

```markdown
- **Simultaneous sessions share ONE working checkout — worktrees are retired.** Worktrees caused more problems than they solved here (data/ symlink corruption, SQLite WAL aliasing, junction-removal footguns, mid-session-entry deadlocks), so every session now works in the shared main checkout on `master`. One working tree means one shared index and one HEAD, so the discipline below is how sessions coexist without clobbering each other. (A cleaner simultaneous-development workflow is still an open design question — 2026-07-06.)
- **Stage and commit explicit paths ONLY — now hook-enforced.** Never `git add -A`/`-u`/`.` and never a bare `git commit` (both sweep other sessions' in-flight work from the shared index). Use `git add <paths>` then `git commit --only -- <paths>`, and verify with `git diff --cached --name-only` first. The `git_shared_checkout_guard` PreToolUse hook denies the sweeping/destroying forms (`add -A`, `commit -a`, bare commit, `reset --hard`, `clean -f`, `checkout .`) — including for subagents — but the guard is a backstop, not a substitute for the discipline. Treat unexpected modified files or commits that appear under you as another session's work: leave them out, leave them alone, and re-derive groupings from the live diff, never a remembered snapshot.
- **Keep your uncommitted pile small.** Commit your own paths to `master` frequently rather than letting a large diff accumulate in the shared tree — the bigger your uncommitted footprint, the more surface another session can trip over.
```

Replace with:

```markdown
- **Simultaneous sessions: three workspaces — canonical + standing satellite clones (worktrees stay retired).** The canonical checkout (this directory) sits on `master` and owns ALL data writes. Satellites `C:\Users\Dylan\Desktop\PG3_SAT1` (GUI port 8771) and `PG3_SAT2` (8772) are full clones with their own branches and full dev powers — real generations included — reaching canonical `data/` via absolute paths in their gitignored `config.yaml`. One active session per workspace; ALWAYS launch the session with cwd already in its workspace (mid-session switching breaks hook/subagent anchoring). Bootstrap a new satellite with `python tools/create_satellite.py --name PG3_SAT3 --port 8773`. Spec: `docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md`.
- **Satellite rules.** Work on feature branches off `origin/master` (`git fetch origin` first — staleness is the main satellite risk); land by `git push origin <branch>` then merging to `master` in canonical. Data-writing pipeline stages (`analyze_library.py`, folds, MuQ extraction) are canonical-only — `satellite_data_write_guard` denies them in satellites. GUI genre publishing (Genre Review "Publish decided") is canonical-only by policy (port 8770). If a satellite behaves oddly, run `python tools/doctor.py` there first — it validates the absolute data paths (the clone's own `data/metadata.db` is a 0-byte stub; config must point away from it).
- **Canonical (shared) checkout: stage and commit explicit paths ONLY — hook-enforced.** Never `git add -A`/`-u`/`.` and never a bare `git commit` (both sweep other sessions' in-flight work from the shared index). Use `git add <paths>` then `git commit --only -- <paths>`, and verify with `git diff --cached --name-only` first. The `git_shared_checkout_guard` hook denies the sweeping/destroying forms here (in satellites it downgrades to a once-per-session reminder — their trees are private). Treat unexpected modified files or commits that appear under you as another session's work: leave them out, leave them alone, and re-derive groupings from the live diff, never a remembered snapshot. Keep your uncommitted pile small — commit your own paths frequently.
```

- [ ] **Step 2: Verify**

Run: `git diff CLAUDE.md`
Expected: one hunk replacing exactly those three bullets; surrounding bullets untouched.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit --only -m "docs(claude-md): three-workspace doctrine - canonical + satellite clones" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- CLAUDE.md
```

---

### Task 8: Live acceptance (spec §9)

**Files:** none created in-repo (SAT1 is created on disk at `C:\Users\Dylan\Desktop\PG3_SAT1`); memory updates at the end.

**Interfaces:** consumes everything above. This is the gate — the feature is NOT done until these pass (unit-green ≠ works).

- [ ] **Step 1: Full test suite (regression gate before going live)**

Run: `python -m pytest -q -m "not slow"` (timeout 600000)
Expected: no new failures vs the pre-plan baseline (quote real counts; diagnose pre-existing failures as pre-existing before touching anything).

- [ ] **Step 2: Bootstrap SAT1**

Run: `python tools/create_satellite.py --name PG3_SAT1 --port 8771` (timeout 600000 — npm install is slow)
Expected: all 6 steps print; doctor gate passes with `Satellite database_path: ...PLAYLIST_GENERATOR_V3\data\metadata.db (NNN MB)` lines; final "Satellite ready" block.

- [ ] **Step 3: Hook satellite-mode e2e (env-pinned, no session needed)**

Hooks read `CLAUDE_PROJECT_DIR`; pin it at SAT1 and pipe tool-call JSON:

```powershell
$env:CLAUDE_PROJECT_DIR = "C:\Users\Dylan\Desktop\PG3_SAT1"
'{"tool_name":"Bash","tool_input":{"command":"python scripts/analyze_library.py"}}' | python .claude/hooks/satellite_data_write_guard.py
'{"tool_name":"Bash","tool_input":{"command":"git add -A"},"session_id":"acceptance"}' | python .claude/hooks/git_shared_checkout_guard.py
$env:CLAUDE_PROJECT_DIR = "C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3"
'{"tool_name":"Bash","tool_input":{"command":"python scripts/analyze_library.py"}}' | python .claude/hooks/satellite_data_write_guard.py
'{"tool_name":"Bash","tool_input":{"command":"git add -A"}}' | python .claude/hooks/git_shared_checkout_guard.py
Remove-Item Env:CLAUDE_PROJECT_DIR
```

Expected, in order: (1) deny JSON (data write in satellite), (2) `additionalContext` warn JSON — NOT deny (sweeper in satellite), (3) empty (data write allowed in canonical), (4) deny JSON (sweeper in canonical).

- [ ] **Step 4: Real generation from SAT1 (the stub-poisoning test)**

Read the `playlist-testing` skill first (mandatory before generation runs). Then run a real multi-pier generation from SAT1's root at INFO — use the golden command form from `docs/GOLDEN_COMMANDS.md`, executed with SAT1 as cwd (e.g. `Push-Location C:\Users\Dylan\Desktop\PG3_SAT1; python main_app.py --artist "<pick a golden artist>" --tracks 12; Pop-Location`).
Expected: generation completes; the log contains NO `BPM load failed` / `BPM gates disabled` stub tell; BPM/pace gate tallies appear; transition stats sane. If the stub tell appears, STOP — the satellite config is mis-wired; run doctor in SAT1 and fix before proceeding.

- [ ] **Step 5: Branch round-trip SAT1 → canonical**

```powershell
Push-Location C:\Users\Dylan\Desktop\PG3_SAT1
git fetch origin
git switch -c sat1-acceptance-canary origin/master
git commit --allow-empty --only -m "test: satellite acceptance canary (empty)"
git push origin sat1-acceptance-canary
Pop-Location
git merge --ff-only sat1-acceptance-canary 2>$null; if (-not $?) { git merge sat1-acceptance-canary -m "merge: satellite acceptance canary" }
git branch -d sat1-acceptance-canary
```

Expected: push accepted by canonical; merge succeeds in canonical; branch deleted. (The empty commit leaves no file churn on master.)

- [ ] **Step 6: Dual GUI**

Start SAT1's GUI in the background (`Push-Location C:\Users\Dylan\Desktop\PG3_SAT1; python tools/serve_web.py --port 8771`), then `Invoke-WebRequest http://localhost:8771/ -UseBasicParsing` → expect HTTP 200 while canonical's 8770 (if running) also answers. Stop the SAT1 server afterward.

- [ ] **Step 7: Update memory + hand off the one human step**

Bootstrap the second standing satellite now that the pattern is proven: `python tools/create_satellite.py --name PG3_SAT2 --port 8772` (expect the same doctor-gated success). Update memory `project_config_enforcement_mechanisms` (new hook + git-guard satellite mode) and write a new `project_satellite_clones_workflow` memory (topology, bootstrap command, landing flow, acceptance results) with a MEMORY.md index line. Then hand off to Dylan: **launch a real Claude Code session with cwd `C:\Users\Dylan\Desktop\PG3_SAT1`** and confirm (a) the memory pointer loads and directs to canonical memory, (b) hooks fire (try `git add -A` — expect the once-per-session satellite reminder, not a deny), (c) normal editing works. That session-level check is the only part that cannot be automated from here.

- [ ] **Step 8: Final commit (memory files live outside the repo — commit only if repo docs changed in this task)**

If Task 8 changed no repo files, there is nothing to commit; state that explicitly in the report.

---

## Execution order & dependencies

1 → (2, 3, 5 depend on 1) → 4 anytime → 6 depends on 5 → 7 anytime after 3 → 8 last, after all.

## Out of scope (from the spec)

- No worktrees, junctions, or symlinks anywhere.
- No config-driving of `analyze_library.py` (single-writer topology is deliberate).
- No GitHub/push-gate changes; no per-clone venvs; no cross-session task-claiming automation (YAGNI until collision data says otherwise).
