# Config-driven DB-path resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every metadata-DB open in the CLI-generation and GUI-worker paths resolve `library.database_path` to an absolute path via one helper, so satellite generations (cwd ≠ data root) read the real DB instead of a relative stub.

**Architecture:** Add `resolve_database_path(config)` to `config_loader.py` (type-tolerant: Config object or dict; always returns an absolute path resolved against the repo root, never cwd). Replace hardcoded relative literals and the `config.get('library', {}).get('database_path', ...)` anti-pattern at the in-scope sites with resolver calls, and thread the resolved absolute path into the DS-pipeline `overrides` dict so deep pipeline code reads an absolute value. A static guard test bans the relative literal from reappearing; a live SAT1 generation is the completeness gate.

**Tech Stack:** Python 3.11+ (stdlib `pathlib`), pytest. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-07-db-path-resolution-design.md`.
- All work in the **canonical checkout on `master`** (`C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3`). Concurrent sessions edit these hotspots — stage explicit paths, commit `git commit --only -m "..." -- <paths>` (bare `git commit` / `git add -A` are hook-DENIED), re-check `git status` before each commit, and leave other sessions' modified files alone.
- The resolver ALWAYS returns an absolute path; it must NEVER return a bare relative path.
- Relative values resolve against the **repo root** (`config_loader.py`'s own location: `Path(__file__).resolve().parent.parent`), NOT the process cwd. This is the property that fixes satellites.
- **Canonical behavior must be unchanged:** canonical's `database_path` is the relative `data/metadata.db`; the resolver resolves it against the repo root = the same real DB. Verify this explicitly.
- Do NOT change the static default arguments of shared client constructors (`local_library_client.py:28`, `similarity_calculator.py:36`) — make in-scope callers pass the resolved path instead.
- Out of scope: `discovery.py`, `source_extraction.py` (enrichment; declined), `analyze_library.py`.
- Pytest: run directly (`python -m pytest <file> -q`), never piped through tail/head; bound with the tool timeout.
- Line numbers below are from 2026-07-07; concurrent edits shift them — locate by the quoted code, not the number.

---

### Task 1: `resolve_database_path` helper + unit tests

**Files:**
- Modify: `src/config_loader.py` (add a module-level function after the `Config` class)
- Test: `tests/test_resolve_database_path.py`

**Interfaces:**
- Produces: `resolve_database_path(config: "Config | dict | None") -> str` — returns an absolute path string. Tasks 2 and 3 import it: `from src.config_loader import resolve_database_path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resolve_database_path.py`:

```python
"""Unit tests for resolve_database_path: config -> absolute DB path (repo-root, not cwd)."""

import os
from pathlib import Path

from src.config_loader import Config, resolve_database_path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg_object(tmp_path, db_value):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"library:\n  music_directory: E:\\\\MUSIC\n  database_path: {db_value}\n",
        encoding="utf-8",
    )
    return Config(str(cfg_file))


def test_absolute_path_returned_as_is(tmp_path):
    abs_db = (tmp_path / "elsewhere" / "metadata.db")
    result = resolve_database_path(_cfg_object(tmp_path, abs_db.as_posix()))
    assert Path(result) == abs_db.resolve()


def test_relative_resolves_against_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd != repo root
    result = resolve_database_path(_cfg_object(tmp_path, "data/metadata.db"))
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()
    assert os.path.isabs(result)


def test_dict_input_absolute(tmp_path):
    abs_db = (tmp_path / "d" / "metadata.db")
    result = resolve_database_path({"library": {"database_path": abs_db.as_posix()}})
    assert Path(result) == abs_db.resolve()


def test_dict_input_relative_resolves_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = resolve_database_path({"library": {"database_path": "data/metadata.db"}})
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_missing_library_falls_back_to_default_absolute(tmp_path):
    assert Path(resolve_database_path({})) == (_REPO_ROOT / "data" / "metadata.db").resolve()
    assert Path(resolve_database_path({"library": {}})) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_none_input_falls_back_to_default_absolute(tmp_path):
    assert Path(resolve_database_path(None)) == (_REPO_ROOT / "data" / "metadata.db").resolve()


def test_canonical_relative_config_resolves_to_repo_db(tmp_path, monkeypatch):
    # Regression: the real canonical pattern (relative 'data/metadata.db') must
    # resolve to the repo's real DB path regardless of cwd.
    monkeypatch.chdir(tmp_path)
    result = resolve_database_path({"library": {"database_path": "data/metadata.db"}})
    assert Path(result) == (_REPO_ROOT / "data" / "metadata.db").resolve()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_resolve_database_path.py -q` (timeout 60s)
Expected: FAIL — `ImportError: cannot import name 'resolve_database_path'`

- [ ] **Step 3: Implement the resolver**

In `src/config_loader.py`, add at the top (after `from typing import Any`):

```python
from pathlib import Path
```

Then add at module level, AFTER the `Config` class definition ends:

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_REL = "data/metadata.db"


def resolve_database_path(config: "Config | dict | None") -> str:
    """Absolute metadata.db path from config — the single source of truth.

    Accepts a Config object or a plain dict (the two shapes at call sites).
    Reads library.database_path; an absolute value is used as-is, a relative
    value is resolved against the REPO ROOT (not the process cwd), and a
    missing/blank value falls back to <repo-root>/data/metadata.db. Always
    returns an absolute path string — never a bare relative path. This makes
    the DB location independent of where the process was launched, which is
    what lets a satellite clone (cwd != data root) read the real canonical DB.
    """
    if isinstance(config, Config):
        raw = config.get("library", "database_path", default=None)
    elif isinstance(config, dict):
        raw = (config.get("library") or {}).get("database_path")
    else:
        raw = None
    raw = (str(raw).strip() if raw else "") or _DEFAULT_DB_REL
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p.resolve())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_resolve_database_path.py -q` (timeout 60s)
Expected: 7 passed

- [ ] **Step 5: Lint + commit**

Run: `ruff check src/config_loader.py tests/test_resolve_database_path.py` → expect clean (no output).

```bash
git add src/config_loader.py tests/test_resolve_database_path.py
git commit --only -m "feat(config): resolve_database_path - absolute DB path from config (repo-root, not cwd)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/config_loader.py tests/test_resolve_database_path.py
```

---

### Task 2: CLI generation path — resolver calls + thread into overrides

**Files:**
- Modify: `main_app.py` (the `LocalLibraryClient(db_path="data/metadata.db")` line)
- Modify: `src/playlist_generator.py` (the 4 `database_path` reads + overrides threading)
- Modify: `src/lastfm_client.py` (the 2 `sqlite3.connect("data/metadata.db")` literals)

**Interfaces:**
- Consumes: `resolve_database_path` (Task 1).

- [ ] **Step 1: Fix `main_app.py` (the confirmed blocker)**

Add the import near the other `from src.` imports at the top of `main_app.py`:

```python
from src.config_loader import resolve_database_path
```

Find:

```python
        self.library = LocalLibraryClient(db_path="data/metadata.db")
```

Replace with:

```python
        self.library = LocalLibraryClient(db_path=resolve_database_path(self.config))
```

(`self.config` is the Config object the CLIApp holds. If the surrounding `__init__` does not have `self.config` in scope at this line, use the config it was constructed with — grep upward for where config is stored on `self`.)

- [ ] **Step 2: Fix the 4 reads in `src/playlist_generator.py`**

Add `resolve_database_path` to the existing `from src.config_loader import ...` line (or add an import).

Replace each of these (they resolve to the same call):

- `db_path=config.get("library", "database_path", default="data/metadata.db")` (in the `LocalLibraryClient(` construction) → `db_path=resolve_database_path(config)`
- `db_path = self.config.get('library', 'database_path', default='data/metadata.db')` (in `_ensure_metadata_client`; this line was commit `2ae6d06`) → `db_path = resolve_database_path(self.config)`
- the two `metadata_db_path=self.config.get("library", "database_path", default="data/metadata.db")` (popularity paths) → `metadata_db_path=resolve_database_path(self.config)`

- [ ] **Step 3: Thread the resolved path into DS-pipeline overrides**

Grep `src/playlist_generator.py` for every `build_ds_overrides(` call (currently one at `overrides = build_ds_overrides(ds_cfg)`). Immediately AFTER each such assignment, insert:

```python
            overrides.setdefault("library", {})["database_path"] = resolve_database_path(self.config)
```

Match the surrounding indentation. This makes the deep pipeline reads (`pipeline/core.py:401,581`, `(overrides or {}).get("library", {}).get("database_path") or "data/metadata.db"`) receive an absolute path. (Deep pipeline code is NOT edited — it reads from overrides unchanged.)

- [ ] **Step 4: Fix `src/lastfm_client.py` literals**

Find the two `conn = sqlite3.connect("data/metadata.db")` (lines ~108, ~141). If the surrounding class holds config or a db_path, use `resolve_database_path(self.config)` / the instance's resolved db_path. If it holds neither, add a `db_path` parameter to the method/constructor threaded from its caller, defaulting to `resolve_database_path(None)` (absolute repo-root default). Do NOT leave the bare `"data/metadata.db"` literal.

- [ ] **Step 5: Verify no relative literal remains in these files + lint**

Run: `python -c "import re,io,sys; sites={'main_app.py','src/playlist_generator.py','src/lastfm_client.py'}; bad=[(f,i+1,l.rstrip()) for f in sites for i,l in enumerate(io.open(f,encoding='utf-8')) if re.search(r'''[\"']data/metadata\.db[\"']|get\(.library., \{\}\)\.get\(.database_path''', l)]; print('\n'.join(f'{f}:{i}: {l}' for f,i,l in bad) or 'CLEAN'); sys.exit(1 if bad else 0)"`
Expected: `CLEAN`

Run: `ruff check main_app.py src/playlist_generator.py src/lastfm_client.py` → expect clean.

- [ ] **Step 6: Commit**

```bash
git add main_app.py src/playlist_generator.py src/lastfm_client.py
git commit --only -m "fix(generation): resolve DB path via config in CLI path + thread absolute path into pipeline overrides" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- main_app.py src/playlist_generator.py src/lastfm_client.py
```

---

### Task 3: GUI worker path — resolver calls

**Files:**
- Modify: `src/playlist_gui/worker.py` (the ~10 `database_path` reads)

**Interfaces:**
- Consumes: `resolve_database_path` (Task 1).

- [ ] **Step 1: Add the import**

Add to `src/playlist_gui/worker.py` imports: `from src.config_loader import resolve_database_path`.

- [ ] **Step 2: Replace the anti-pattern reads**

Grep `src/playlist_gui/worker.py` for `database_path`. Replace every read of the form `config.get('library', {}).get('database_path', 'data/metadata.db')` / `config.get("library", {}).get("database_path", "data/metadata.db")` / `Path(cfg.get("library", {}).get("database_path", "data/metadata.db"))` and the module constant `METADATA_DB_PATH = "data/metadata.db"` with `resolve_database_path(<the config/cfg in scope>)`. For `METADATA_DB_PATH`, replace its uses with `resolve_database_path(config)` at each use site (delete the module constant if it becomes unused). The line already using the property (`merged_config.library_database_path`) may stay, but for uniformity prefer `resolve_database_path(merged_config)`.

- [ ] **Step 3: Verify + lint**

Run: `python -c "import re,io,sys; bad=[(i+1,l.rstrip()) for i,l in enumerate(io.open('src/playlist_gui/worker.py',encoding='utf-8')) if re.search(r'''[\"']data/metadata\.db[\"']|get\(.library., \{\}\)\.get\(.database_path''', l)]; print('\n'.join(f'{i}: {l}' for i,l in bad) or 'CLEAN'); sys.exit(1 if bad else 0)"`
Expected: `CLEAN`

Run: `ruff check src/playlist_gui/worker.py` → expect clean.

- [ ] **Step 4: Commit**

```bash
git add src/playlist_gui/worker.py
git commit --only -m "fix(gui-worker): resolve DB path via config (satellite GUI reads real DB)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/playlist_gui/worker.py
```

---

### Task 4: Static guard test — ban the relative literal

**Files:**
- Test: `tests/test_no_relative_db_literal.py`

**Interfaces:** none.

- [ ] **Step 1: Write the guard test**

Create `tests/test_no_relative_db_literal.py`:

```python
"""Regression guard: the in-scope generation + GUI files must not reintroduce a
relative metadata.db literal or the config.get('library', {}).get('database_path')
anti-pattern. DB paths in these files go through resolve_database_path()."""

import io
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# In-scope per the 2026-07-07 spec (CLI generation + GUI worker). Enrichment
# files (discovery.py, source_extraction.py) are intentionally excluded.
_IN_SCOPE = [
    "main_app.py",
    "src/playlist_generator.py",
    "src/lastfm_client.py",
    "src/playlist_gui/worker.py",
]

_LITERAL = re.compile(r"""["']data/metadata\.db["']""")
_ANTIPATTERN = re.compile(r"""get\(['"]library['"],\s*\{\}\)\.get\(['"]database_path""")


def test_no_relative_db_literal_in_scope_files():
    offenders = []
    for rel in _IN_SCOPE:
        for i, line in enumerate(io.open(_ROOT / rel, encoding="utf-8"), 1):
            if _LITERAL.search(line) or _ANTIPATTERN.search(line):
                offenders.append(f"{rel}:{i}: {line.rstrip()}")
    assert not offenders, "Relative DB literal / anti-pattern found:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_no_relative_db_literal.py -q` (timeout 60s)
Expected: PASS. If it FAILS, it has found a site Tasks 2–3 missed — fix that site (replace with `resolve_database_path(...)`), commit it with the owning file's task pattern, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_no_relative_db_literal.py
git commit --only -m "test(config): guard against relative metadata.db literal in generation + GUI paths" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- tests/test_no_relative_db_literal.py
```

---

### Task 5: Live acceptance (orchestrator, inline) + memory

**Files:** none in-repo (SAT1 exercised at `C:\Users\Dylan\Desktop\PG3_SAT1`); memory updated at the end.

**Interfaces:** consumes all prior tasks. This is the completeness gate — the CLI generation empirically flushes out any missed relative-open.

- [ ] **Step 1: Regression — full-ish suite on the changed surface**

Run: `python -m pytest tests/test_resolve_database_path.py tests/test_no_relative_db_literal.py -q` (timeout 120s) → expect all pass.
Run: `python -m pytest tests/integration/test_ds_pipeline_smoke.py -q` (timeout 300s) → note pass/fail counts; distinguish any pre-existing/concurrent failures (the two `test_dynamic_mode_10_tracks`/`test_narrow_mode_10_tracks` failures were concurrent-session pier_bridge, not this work) from anything this change caused.

- [ ] **Step 2: Update SAT1 to canonical master**

Run: `git -C "C:/Users/Dylan/Desktop/PG3_SAT1" fetch origin --quiet && git -C "C:/Users/Dylan/Desktop/PG3_SAT1" pull origin master --ff-only`
Expected: SAT1 fast-forwards to include Tasks 1–4.

- [ ] **Step 3: Real CLI generation from SAT1 (the gate)**

Run: `(cd "C:/Users/Dylan/Desktop/PG3_SAT1" && python main_app.py --artist "Beach House" --tracks 12) > "$TMPDIR/sat1_gen_final.log" 2>&1; echo "EXIT=$?"`
Then check the log:
- MUST NOT contain `no such table` or `Unexpected Error`.
- MUST NOT contain `BPM load failed` / `BPM gates disabled` (stub tell).
- MUST show a completed playlist with BPM/pace gate activity and transition stats.

If a NEW relative-open surfaces (a different `no such table` / a fresh 0-byte `PG3_SAT1/data/metadata.db`): read the traceback (add a temporary `import traceback; traceback.print_exc()` shim only if needed, or inspect the failing module), find the missed site, replace it with `resolve_database_path(...)`, commit it (explicit path), pull SAT1 again, and re-run. Repeat until the generation succeeds. (This is the spec's "empirical flush-out".)

- [ ] **Step 4: Confirm canonical generation unchanged**

Run: `python main_app.py --artist "Beach House" --tracks 12 > "$TMPDIR/canon_gen_final.log" 2>&1; echo "EXIT=$?"`
Expected: completes as before (canonical regression check — its relative config still resolves to the real DB).

- [ ] **Step 5: Update memory + hand off the GUI check**

Update memory `project_satellite_clones_workflow` (remove the BLOCKED caveat for CLI generations; note the resolver shipped and satellite CLI generation is validated; GUI pending Dylan's manual check). Then hand off to Dylan: the one manual step is launching SAT1's GUI (`cd C:\Users\Dylan\Desktop\PG3_SAT1 && python tools/serve_web.py --port 8771`) and generating through it to confirm the worker path reads the real DB. If that surfaces a missed worker site, it's a one-line resolver follow-up.

- [ ] **Step 6: Finish**

Use superpowers:finishing-a-development-branch. (Work is committed to `master`; no branch to merge — confirm the changed tests are green and summarize the commits.)

---

## Self-review notes

- **Spec coverage:** resolver (Task 1) ✓; entry-point resolver calls (Task 2 CLI, Task 3 GUI) ✓; overrides threading (Task 2 Step 3) ✓; no-change-to-static-defaults (constraint honored — callers pass resolved path) ✓; unit tests (Task 1) ✓; static guard (Task 4) ✓; live SAT1 CLI acceptance + canonical regression (Task 5) ✓; GUI manual (Task 5 Step 5) ✓; out-of-scope enrichment excluded (Task 4 `_IN_SCOPE`) ✓.
- **The `2ae6d06` note:** Task 2 Step 2 explicitly replaces that line with the resolver call — supersedes the earlier mis-targeted fix, as the spec states.
- **Completeness driver:** the live SAT1 generation (Task 5 Step 3) is the real gate; the guard test (Task 4) prevents regression but only over the enumerated files, so the live loop is what guarantees the generation actually works.
