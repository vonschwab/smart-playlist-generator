# Reuse-first nudge — design

**Date:** 2026-06-25
**Status:** Approved (design); pending implementation plan
**Branch:** `worktree-reuse-first-nudge`

## Motivation

We liked one specific aspect of the `ponytail` plugin — its instinct to **reuse
existing code and prefer existing native/stdlib methods before writing anything
new** — but not its always-on, global, minimize-everything posture, which fights
this project's deliberately verbose, instrumented, audit-heavy engineering
culture (Design Principles 20–21, 23, 25).

This feature extracts only the reuse-first instinct and tunes it to *this* repo:
fired at the moment new code is about to be added, pointing at this project's
actual reusable utilities and god-class hotspots, with explicit carve-outs that
protect the instrumentation this codebase treats as mandatory.

## Goals

- Nudge the agent, at the point of adding code, to check existing repo code →
  stdlib/native → already-installed deps **before** writing new code or adding a
  dependency.
- Tune the content to this repo's real reuse map (named modules below), not a
  generic "be lazy" prompt.
- Match the existing project idiom: a small Python advisory hook that injects a
  once-per-session `additionalContext` reminder and points at a skill for depth
  (exactly how `stale_state_reminder.py` references the `web-gui` skill).
- Never block; never nag more than once per session per category.
- Do **not** encourage minimizing away diagnostics, quality metrics, audit
  scaffolding, validation, graceful fallbacks, or config knobs.

## Non-goals

- No always-on / SessionStart context injection (the ponytail approach we
  rejected).
- No hard gate / blocking PreToolUse decision.
- No global/portable plugin — this is tuned and scoped to this repo. (The skill
  and hook are plain files and could be lifted later, but that is not a goal.)
- No coverage of `tests/` (nudging reuse in test code is mostly noise).

## Architecture

Two cooperating pieces plus wiring, mirroring the existing
hook → skill pattern in this repo.

### 1. Skill — `.claude/skills/reuse-first/SKILL.md`

The substance: a "reuse ladder" walked **before** writing new code, tuned to this
repo. Frontmatter `description` written so the skill triggers when the agent is
about to add a feature, a function, or a dependency.

The ladder:

- **Rung 0 — Understand first.** Trace the real flow / read the code the change
  touches before picking a rung. A small diff you don't understand is guessing,
  not reuse. (Aligns with CLAUDE.md "read the logs / grep first".)
- **Rung 1 — Already in this repo?** `grep` before writing. Named reuse targets:
  - generic utils (prime reinvention risk): `src/artist_utils.py`,
    `src/string_utils.py`, `src/logging_utils.py`, `src/playlist/utils.py`
  - identity / normalization → `src/ai_genre_enrichment/normalization.py`
  - genre reads → `src/genre/authority.py` (+ the `genre-data-authority` skill)
  - runtime / mode config → `src/playlist_gui/policy.py::derive_runtime_config`
    (never hand-roll mode strings — standing gotcha)
  - energy → `src/playlist/energy_loader.py`
- **Rung 2 — Hotspot rule.** If the change lands in a god-class
  (`src/playlist/pier_bridge_builder.py`, `src/playlist_generator.py`,
  `src/playlist/pipeline.py`, `src/playlist_gui/worker.py`), extract a helper —
  do not grow the monolith (CLAUDE.md Hotspots).
- **Rung 3 — Stdlib / native?** `itertools`, `functools`, `dataclasses`,
  `pathlib`, `statistics`, `collections`; numpy/scipy vectorization over hand
  loops (already dependencies).
- **Rung 4 — Already-installed dep?** Check `pyproject.toml` / `web/package.json`
  before reaching for a new dependency.
- **Rung 5 — Only then** write the minimum new code.

Carve-outs (the critical tuning so it does not fight the repo's culture) —
never "minimize away":

- diagnostic logging, quality metrics, opt-in audit reports (Principles 20–21)
- input validation, error handling, graceful fallbacks (Principle 25)
- config knobs / tunability (Principle 23)
- anything the user explicitly asked to keep

Output discipline: after adding code, note in 1–2 lines what existing thing was
reused, or why new code was unavoidable.

### 2. Hook — `.claude/hooks/reuse_first_reminder.py` (PreToolUse, matcher `Edit|Write`)

Structurally mirrors `.claude/hooks/stale_state_reminder.py`: read the tool-call
JSON from stdin, gate once-per-session via a marker file in the system temp dir,
emit a `hookSpecificOutput.additionalContext` message, never block.

Trigger logic:

- **Source file** — path matches `.py/.ts/.tsx/.js/.jsx/.mjs` under `src/`,
  `web/src/`, `scripts/`, or `tools/` (and not under `tests/`) — **and** the
  change is a net-add:
  - `Write`: `content` length exceeds the threshold, or
  - `Edit`: `len(new_string) - len(old_string)` exceeds the threshold, or
    `new_string` introduces a new `def `/`class `/`function `/`const ` not
    present in `old_string`.
  → fire the **general** reuse reminder, pointing at the `reuse-first` skill.
- **Dependency manifest** — path is `pyproject.toml` or `web/package.json` →
  fire a distinct, **stronger** reminder ("adding a dependency is deliberate —
  confirm stdlib + an already-installed dep + existing repo code can't do this
  first").
- Trivial edits (net change below threshold) → silent.

Net-add threshold: ~300 characters to start; tunable after observing real firing
behavior. Two marker categories (`general`, `dependency`) so the dependency
nudge is never suppressed by an earlier general one.

Output form: `{"hookSpecificOutput": {"hookEventName": "PreToolUse",
"additionalContext": <message>}}`. Advisory only — no `permissionDecision`.

### 3. Wiring — `.claude/settings.json`

Add a new matcher block under `hooks.PreToolUse` with matcher `Edit|Write`,
invoking the new hook via `python` + `${CLAUDE_PROJECT_DIR}/.claude/hooks/
reuse_first_reminder.py`, consistent with the existing `pytest_pipe_guard`
entry. The existing PreToolUse entry matches `Bash|PowerShell`, so there is no
collision.

### 4. Test — `tests/test_reuse_first_reminder.py`

Feed representative tool-call payloads to the hook's stdin-processing function
and assert fire/no-fire and category:

- `Write` of a new source file with substantial content → general reminder
- trivial one-line `Edit` → silent
- large `Edit` adding a new `def` → general reminder
- `Edit` of `pyproject.toml` → dependency reminder
- non-source file (e.g. a `.md`) → silent

The hook is pure stdin→stdout JSON logic, so the test needs no DB, artifacts, or
running app. This is the "one runnable check" the skill itself preaches.

## Known limitation

PreToolUse fires as the first qualifying edit is *being applied*, so the
reminder sets the agent's disposition for the rest of the session rather than
catching the very first add. A once-per-session advisory is the right weight; a
blocking gate would be the nagging we explicitly chose to avoid.

## Files

| File | Change |
|------|--------|
| `.claude/skills/reuse-first/SKILL.md` | new — the reuse ladder |
| `.claude/hooks/reuse_first_reminder.py` | new — PreToolUse advisory hook |
| `.claude/settings.json` | edit — register the hook under PreToolUse `Edit|Write` |
| `tests/test_reuse_first_reminder.py` | new — fire/no-fire unit test |
| `docs/superpowers/specs/2026-06-25-reuse-first-nudge-design.md` | new — this doc |
