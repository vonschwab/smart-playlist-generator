# Simultaneous sessions via standing satellite clones — design

**Date:** 2026-07-06
**Status:** Approved (Dylan, 2026-07-06)
**Supersedes:** the worktree workflow (retired 2026-07-06) and, for multi-session isolation, the shared-checkout-only stopgap.

## Problem

Multiple Claude Code sessions routinely develop this repo at the same time. Worktrees were the isolation mechanism and failed on five documented mechanisms (see Prior art). The current stopgap — everyone in one shared checkout with hook-enforced explicit-path commits (`git_shared_checkout_guard`, commit `0253ace`) — makes shared *commits* safe but provides no *file* isolation: one HEAD and one set of working files means same-file edits race, every session sees every other session's in-flight diff, and only one branch can be checked out at a time.

**Goal:** each concurrent session gets its own working tree, own branch, and full dev powers (edit, tests, real-data generations, GUI), without reintroducing any of the worktree failure mechanisms.

## Prior art (constraints this design must honor)

The five worktree killers, from memories `feedback_worktree_data_absolute_overrides`, `feedback_worktree_sqlite_wal_aliasing`, `feedback_worktree_data_junction_removal`, `feedback_subagents_run_in_main_checkout`:

1. **`data/` cannot be linked** — it has git-tracked stub content (zero-byte `metadata.db`), so symlink/junction setups silently leave a stub that *silently degrades generation* (BPM load fails inside try/except → different playlist, no error).
2. **SQLite WAL aliasing** — one DB opened via two different path strings gets two independent `-wal`/`-shm` sets that checkpoint over each other; corrupted `ai_genre_enrichment.db` on 2026-06-22. Safety condition: **same path string everywhere**.
3. **Junction teardown** — recursive deletion of a workspace containing a junction can follow it and delete the target (nearly lost the MERT shards).
4. **Mid-session directory entry** — hooks and subagents anchor to the session *launch* dir; entering a workspace mid-session leaks commits to the wrong branch and can 404 every hook (unrecoverable in-session).
5. **`scripts/analyze_library.py` hardcodes `ROOT_DIR`-relative data paths** (e.g. `ENRICHMENT_DB_PATH`), ignoring config — so data-writing stages run against whatever tree they're launched from.

What *worked* and is reused here: absolute-path data overrides (`PLAYLIST_GOLDEN_ARTIFACT`/`PLAYLIST_GOLDEN_DB`, verified bit-identical from any cwd), the same-path-string rule, launching sessions with cwd already in the workspace, and the probe-then-pin subagent pattern.

## Design

### 1. Topology

- **Canonical checkout:** `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3`. Unchanged: sits on `master`, owns all data writes, GUI on port 8770, GitHub `origin` + push gate untouched, shared-checkout guards fully strict.
- **Satellites:** two standing full clones, `C:\Users\Dylan\Desktop\PG3_SAT1` and `C:\Users\Dylan\Desktop\PG3_SAT2`. Each satellite's git `origin` is the canonical checkout (local path remote). Satellites are permanent (no per-task setup/teardown).
- **One active Claude session per workspace at a time.** Sessions are launched with cwd already in their workspace; never switch workspaces mid-session (constraint 4).
- Each clone is a complete independent repo — own working tree, index, HEAD. Canonical on `master`, SAT1 on any branch, SAT2 on any other branch; no interaction until push/merge. (Unlike worktrees, clones may even check out same-named branches — though we avoid same names by convention, see §4.)

### 2. Data access (satellites → canonical data, read-only)

Satellites get full real-data dev powers via **absolute config paths — no links of any kind** (constraints 1–3):

- Each satellite's gitignored `config.yaml` (copied once from canonical, then edited) sets:
  - `library.database_path: C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db`
  - `playlists.ds_pipeline.artifact_path: C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz`
  - Both keys are already config-driven (`src/config_loader.py:101,157`) — no app-code changes for the core data access (the single code change in this design is the MCP stub fix, §7).
- **No-aliasing rule (the real WAL safety condition):** the 2026-06-22 corruption came from a *symlink* — SQLite derives `-wal`/`-shm` sidecar locations from the path it is given, so a DB opened via a link gets a second sidecar set next to the link. With zero links anywhere, every spelling (canonical's relative `data/metadata.db`, satellites' canonical-absolute path) resolves to the one real file and the one real sidecar set — safe. "Use the same path string" remains the belt-and-suspenders convention, but safety does not depend on it; do not "fix" canonical's relative config path to match satellites'. Generation is SELECT-only against `metadata.db`; concurrent reads are ordinary SQLite behavior, and shared caches generation writes (e.g. `data/popularity_cache.db`) are lock-managed by SQLite since there is exactly one real file.
- **Local by design (stay relative):** `logs/` (per-workspace generation logs — a feature: each session's logs are its own), `web/dist`, `web/node_modules`, `docs/run_audits/` experiment outputs.
- **Stub landmine, defused at bootstrap:** a fresh clone's `data/` contains the tracked zero-byte `metadata.db` stub — the silent-BPM-kill trap (constraint 1). `tools/doctor.py` gains a satellite check (reuse, not a new tool): the *resolved* DB and artifact paths must be absolute, exist, exceed a plausible size floor (stub = 0 bytes; real DB is hundreds of MB), and must not resolve inside a clone whose own `data/metadata.db` is a stub. Bootstrap ends by running doctor; a mis-wired satellite refuses loudly (design principle: a configured knob that can't act is a startup error).

### 3. Data writes — canonical-only, hook-enforced

Because of constraint 5 (and the WAL incident behind constraint 2), **all data-writing pipeline work runs only in the canonical checkout**: `analyze_library.py` stages (scan/genres/discogs/lastfm/sonic/adjudicate/apply/publish/artifacts/verify), fold scripts, MuQ extraction.

Enforcement: new PreToolUse hook `satellite_data_write_guard.py` (same contract as the existing hook suite; fail-open). In a satellite it **denies** Bash/PowerShell commands invoking `analyze_library.py`, `fold_*.py`, or `muq_runner`, with a message directing the work to canonical. In canonical it is silent.

**GUI publish is also a data write, policy-only:** the Genre Review "Publish decided" action writes `metadata.db` from the web worker — hooks cannot see worker-internal actions, and it is WAL-safe (no aliasing), so this stays a documented rule rather than enforcement: **do genre publishing from the canonical GUI (port 8770) only.** Goes in the CLAUDE.md satellite bullet (§10).

**Workspace detection (shared helper, used by both this hook and the git guard):** `git config remote.origin.url` — a local filesystem path ⇒ satellite; a GitHub URL or unset ⇒ canonical. Automatic; no marker files to forget. Implemented once (e.g. `.claude/hooks/workspace_identity.py`) and imported by both hooks; unit-tested.

### 4. Git flow

- Satellites branch off `origin/master`: `git fetch origin` then `git switch -c <branch> origin/master`. Fetch-before-branch is a recipe step (staleness is the main satellite risk).
- **Landing:** `git push origin <branch>` from the satellite (pushing a non-checked-out branch to canonical is stock git), then merge to `master` **in canonical only**. This is the natural review point and keeps one integration authority.
- Branch names are descriptive and unique across satellites (convention; collision just makes push fail loudly, not corrupt).
- The GitHub push gate is untouched — satellites never talk to GitHub; only canonical does, deliberately.

### 5. Guard behavior per workspace

- **`git_shared_checkout_guard`** (as updated with quote-aware segmentation): gains workspace awareness via the shared detection helper.
  - **Canonical: fully strict** (unchanged) — deny `add -A/-u/.`, `commit -a`, bare `commit`, `reset --hard`, `clean -f`, `checkout/restore .`; warn on branch switches.
  - **Satellite: index-sweeper denials downgrade to one-time-per-session warnings** (a satellite's index is private; strictness there is friction without safety). `reset --hard` and `clean -f` **stay denied everywhere** — rarely legitimate, cheap insurance. Branch-switch warnings drop in satellites (switching branches is the satellite's whole point).
- **`data_safety_guard`** (commit `4f9d6ab`): works in satellites unchanged — its `metadata.db`/archive patterns match the canonical absolute paths satellites use, and each satellite's `config.yaml` carries `music_directory` for the audio-library pattern. No changes needed.
- **`satellite_data_write_guard`**: new, per §3.

### 6. Claude harness in satellites

- `CLAUDE.md`, `.claude/settings.json`, `.claude/hooks/`, `.claude/skills/` are git-tracked → satellites inherit them automatically. Each session's `CLAUDE_PROJECT_DIR` is its own clone (sessions launch in-workspace), so `${CLAUDE_PROJECT_DIR}`-relative hooks resolve correctly — constraint 4's anchor problem cannot occur.
- **Auto-memory:** memory is keyed to the launch directory, so satellite sessions would see an empty memory dir. Fix: bootstrap writes a pointer `MEMORY.md` into each satellite's memory dir (`~/.claude/projects/<munged-satellite-path>/memory/MEMORY.md`) instructing the session to Read the canonical memory index at its absolute path first. **No junctions/symlinks** (constraint 3 spirit). Memory *writes* from satellite sessions go to the canonical memory dir by absolute path (the pointer says so).
- **Ports:** canonical 8770, SAT1 8771, SAT2 8772 (`python tools/serve_web.py --port <n>`, in the recipe).
- **Python env:** shared — imports are cwd-relative (`from src....`; the `src` package resolves from the repo root the process launches in, verified 65+ occurrences), so each workspace runs its own code with one interpreter. **Node:** each satellite runs `npm --prefix web install` once at bootstrap.

### 7. Bootstrap (one-time per satellite, scripted)

`python tools/create_satellite.py --name PG3_SAT1 --port 8771` (Python, not PowerShell — the config rewrite needs robust line-targeted YAML edits, and regex surgery in PS is fragile):
1. `git clone C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3 C:\Users\Dylan\Desktop\PG3_SAT1`
2. Copy canonical `config.yaml` → satellite; rewrite `library.database_path` and set `playlists.ds_pipeline.artifact_path` (inserting the key if canonical relies on the default) to canonical absolute paths, preserving the rest of the file byte-for-byte (targeted line edits, not YAML round-trip — Dylan's comments survive). Leave `music_directory` as-is (already absolute).
3. Copy the untracked local-config files that sessions rely on: `.claude/settings.local.json` (permission allowlist) and `.mcp.json`.
4. `npm --prefix <sat>\web install` and `npm --prefix <sat>\web run build`.
5. Write the satellite memory pointer `MEMORY.md` (§6).
6. Run the doctor satellite check (§2); fail the bootstrap loudly if it fails.
7. Print the session-launch line: launch Claude Code with cwd `C:\Users\Dylan\Desktop\PG3_SAT1`; GUI = `python tools/serve_web.py --port 8771`.

**MCP stub fix (small code change, root-cause):** `tools/mcp_sqlite_readonly.py:45` hardcodes `_REPO_ROOT/data/metadata.db` — in a satellite the SQLite MCP would silently query the empty stub (read-only, so no damage, but silently-empty results are the stub-divergence failure class). Fix it to honor `config.yaml` `library.database_path` (resolved against the repo root when relative), falling back to the current default. This is the only app-code change in the design.

### 8. Failure-mode audit

| # | Worktree failure | This design |
|---|---|---|
| 1 | `data/` unlinkable; stub DB silently poisons runs | No links — absolute config paths; stub defused by doctor bootstrap gate |
| 2 | Dual-WAL aliasing corruption | One path string everywhere; data writes canonical-only, hook-enforced |
| 3 | Teardown follows junction, deletes target | Nothing linked anywhere; satellites are standing (no teardown path); archive also covered by `data_safety_guard` |
| 4 | Mid-session entry → hook 404 / commit leak | Sessions launch in-workspace; every clone carries its own tracked `.claude/` |
| 5 | `analyze_library.py` hardcoded ROOT_DIR paths | Data-writing stages denied in satellites by `satellite_data_write_guard` |

Residual risks (accepted, mitigated): two sessions accidentally in one satellite (rule + satellite warn-mode git guard as backstop); satellite branching from a stale master (fetch-first recipe step); same-named branches across satellites (push fails loudly; convention avoids it); doctor check skipped after a manual config edit (doctor is also runnable any time: first debugging step for a "weird satellite").

### 9. Testing & acceptance

- **Unit:** `workspace_identity` detection (local-path origin vs GitHub vs unset); `satellite_data_write_guard` deny/allow matrix (analyze/fold/muq denied in satellite, allowed in canonical; unrelated commands untouched); git-guard satellite downgrades (sweepers warn in satellite, still deny in canonical; `reset --hard`/`clean -f` deny in both); doctor satellite check (stub detection, relative-path rejection, missing-file rejection). Same loading pattern as `tests/test_git_shared_checkout_guard.py`.
- **Live acceptance (the real gate):**
  1. Bootstrap SAT1 with the script; doctor passes.
  2. From SAT1: run a real generation (`gui_fidelity`-faithful path) at INFO; verify the log does NOT contain the stub tell ("BPM load failed" / "BPM gates disabled") and DOES show BPM/pace gate activity, and output quality stats are sane (proves no stub poisoning — the constraint-1 failure).
  3. From SAT1: confirm `analyze_library.py` is denied; in canonical: confirm it is allowed.
  4. From SAT1: branch, commit, `git push origin <branch>`; in canonical: merge to master; confirm history is clean.
  5. Run canonical + SAT1 GUIs simultaneously on 8770/8771.

### 10. Documentation

- CLAUDE.md "Session discipline": replace the shared-checkout-is-the-norm bullets with the three-workspace model (canonical + satellites, one session per workspace, satellites = full dev minus data writes, landing flow). Keep the explicit-path discipline text for canonical.
- Memory: update `project_config_enforcement_mechanisms` (new hook, guard modes) and add a satellite-workflow memory once live-verified.

## Out of scope

- Reviving worktrees in any form; junctions/symlinks of any kind.
- Making `analyze_library.py` config-driven so satellites could write data (deliberately NOT doing this — single-writer topology is the safety property, not a limitation to engineer away).
- Changing the GitHub push gate or origin handling in canonical.
- Per-clone Python venvs (unnecessary — cwd-relative imports; revisit only if the packaging model changes).
- Automating cross-session coordination (task claiming, file locking) — YAGNI until the three-workspace model shows a real collision rate.
