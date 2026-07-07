# Config-driven database-path resolution — design

**Date:** 2026-07-07
**Status:** Approved (Dylan, 2026-07-07)
**Motivation:** Unblock satellite generations + GUI (found by the satellite-clones live acceptance — see `docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md` §9 and memory `project_satellite_clones_workflow`).

## Problem

The generation and GUI code resolve the metadata-DB path inconsistently, and many sites default to the **relative** literal `"data/metadata.db"` instead of honoring `config.yaml`'s `library.database_path`. This is invisible in the canonical checkout — the process cwd is the repo root, so the relative path resolves to the real DB by coincidence — but it breaks any workspace whose cwd is not the data root. A satellite clone (whose `config.yaml` points `database_path` at the canonical DB by absolute path) opens the relative path instead, hits its own empty `data/metadata.db` stub, and fails with `no such table: tracks`.

This is the project's documented #1 failure mode — "a configured knob that looks wired but isn't" — at scale.

**Verified root cause (from the live acceptance, not grep inference):** the CLI generation fails at `main_app.py:44`, `self.library = LocalLibraryClient(db_path="data/metadata.db")` — a hardcoded relative literal that ignores config entirely, executed right after "Initializing Playlist Generator". A speculative earlier fix to `playlist_generator.py:303` (commit `2ae6d06`) was necessary-but-not-sufficient and mis-targeted — that line isn't even reached before `main_app.py:44` fails. The acceptance (a real generation) is the only reliable way to enumerate the failing sites; static grep under-counts (hardcoded literals) and over-counts (valid dict access).

## Two kinds of site

1. **Callers that hold config** (a `Config` object or the full config dict): `main_app.py`, `playlist_generator.py`, the `worker.py` handlers. These can resolve the path themselves.
2. **Deep consumers that only receive a partial `overrides` dict** (`pipeline/core.py:401,581`, the banger-gate / beam loaders): `overrides` does not carry `library.database_path`, so `(overrides or {}).get("library", {}).get("database_path") or "data/metadata.db"` falls back to relative. These cannot resolve on their own — the value must be threaded in.

Also note the `Config.get(section, key, default)` signature (`config_loader.py:84`) is 3-arg: `config.get('library', {}).get('database_path', ...)` is **valid on a plain dict** but **raises `TypeError` on a `Config` object** (`dict.get({})` — unhashable). So the anti-pattern is either a silent relative fallback (dict) or a latent crash (Config object), depending on the caller. The resolver removes the ambiguity.

## Design

### 1. `resolve_database_path(config)` — single source of truth

New function in `src/config_loader.py` (beside the `Config` class):

- **Input:** a `Config` object **or** a plain `dict` (the two shapes that exist at call sites).
- **Reads:** `library.database_path` — via `config.get('library', 'database_path', default=None)` for a `Config` object, or `config.get('library', {}).get('database_path')` for a dict (type-detected).
- **Resolves to ABSOLUTE:** absolute input returned as-is; a relative value resolved against the **repo root** (the `Config` module's known root, e.g. `Path(__file__).resolve().parent.parent`), never the process cwd; a missing/blank value falls back to `<repo-root>/data/metadata.db` (absolute).
- **Returns:** an absolute path string (str, since sqlite/clients take str paths).
- **Never** returns a bare relative path. This is the property that fixes both canonical (unchanged — still resolves to the same real DB) and satellites (now the configured absolute path).

Resolving against the repo root, not cwd, is the crux: it makes the result independent of where the process was launched, which is exactly the satellite requirement.

### 2. Entry-point sites: call the resolver

Replace the hardcoded literal / `config.get('library', {})` access with `resolve_database_path(config)` at every caller that holds config:

- `main_app.py:44` (the blocker) — `LocalLibraryClient(db_path=resolve_database_path(self.config))`.
- `playlist_generator.py:279, 303, 1850, 1896` — route all four through the resolver (the 3-arg ones are already correct but become absolute + uniform; `:303` supersedes commit `2ae6d06`).
- `worker.py` (~10 sites: 954, 1219 already uses the property, 1632, 1690, 1736, 2017, 2036, 2063, 2089, 2446, 2489) — resolver call.

Do **not** change the static default arguments of shared client constructors (`local_library_client.py:28`, `similarity_calculator.py:36`, `discovery.py:58`, `source_extraction.py:79`) — a static `= "data/metadata.db"` default cannot resolve at import time and other callers depend on the signature. Instead, ensure every **caller in scope** passes the resolved path explicitly, so the default is never used in the generation/GUI path. Fix the direct `sqlite3.connect("data/metadata.db")` literals in `lastfm_client.py:108,141` to take the resolved path from their caller (or the resolver if they hold config).

### 3. Deep-consumer sites: thread the resolved path via `overrides`

The entry point resolves once and writes the absolute path into the overrides bag before invoking the pipeline: `overrides.setdefault("library", {})["database_path"] = resolve_database_path(config)`. Then `pipeline/core.py:401,581` read the (now-absolute) value from `overrides` unchanged. This keeps the 5.3k-LOC pier-bridge hotspot's deep signatures untouched — the smallest, safest change that makes deep code correct. (Rejected alternative: threading an explicit `metadata_db_path` parameter through the pipeline call chain — more signature churn in the hotspot for no additional correctness.)

### 4. Scope

**In scope (CLI generation + GUI, per the 2026-07-07 decision):** `main_app.py`, `playlist_generator.py`, `pipeline/core.py`, `similarity_calculator.py` (caller-side), `local_library_client.py` (caller-side), `lastfm_client.py`, `worker.py`.

**Out of scope:** `discovery.py`, `source_extraction.py` (enrichment code satellites don't run — the whole-repo sweep was explicitly declined). They keep the anti-pattern with a note; a future pass can fold them in.

## Testing & acceptance

**Unit tests** (`tests/test_resolve_database_path.py`):
- absolute input returned as-is;
- relative input resolved against repo root (not cwd) — assert the result is absolute and repo-rooted;
- `Config`-object input reads `library.database_path`;
- plain-dict input reads `library.database_path`;
- missing/blank value → `<repo-root>/data/metadata.db` absolute;
- (regression) canonical config's relative `data/metadata.db` resolves to the real repo DB path unchanged.

**Static guard test** (`tests/test_no_relative_db_literal.py`): scan the in-scope generation + GUI source files and assert no residual `sqlite3.connect("data/metadata.db")` / `db_path="data/metadata.db"` / `.get('database_path', 'data/metadata.db')` literal remains outside the resolver and the declined-scope files. Prevents regression to the anti-pattern.

**Live acceptance — automated gate (agent-run, empirical, flushes out missed CLI sites):**
1. From SAT1 (updated to canonical master), a real CLI generation completes: no `no such table` / `BPM load failed` / stub tell; BPM/pace gate activity present; sane transition stats.
2. Canonical CLI generation unchanged (regression check).

**GUI — manual smoke check (Dylan, not a blocking agent gate):** after the worker sites are fixed, Dylan launches SAT1's GUI (`serve_web.py --port 8771`) and confirms a generation through it reads the real canonical DB. The worker-side code fixes and the static guard test still ship as part of this work; only the *live GUI exercise* is manual. If the manual check surfaces a missed worker site, it's a fast follow-up (one more resolver substitution), not a re-plan.

## Risks & notes

- **Hotspot edits under concurrent work:** `playlist_generator.py` (~4.1k LOC) and `worker.py` (~2.5k LOC) are being actively edited by other sessions. Each change is a mechanical single-line resolver substitution; stage explicit paths, commit per file group, re-check `git status` before each commit (shared-checkout discipline; the git guard enforces `--only`).
- **`similarity_calculator.py` default is `"metadata.db"`** (not `data/metadata.db`) — even more relative. Confirm its generation-path caller (via `LocalLibraryClient`) passes the resolved path so the default is never hit.
- **WAL safety:** the satellite opens the canonical DB by absolute path while canonical opens it by relative path; both resolve to the same real file with no symlink, so there is no dual-WAL aliasing, and generation reads are SELECT-only (per memory `feedback_worktree_data_absolute_overrides`). No corruption vector.
- **Canonical behavior is unchanged** everywhere: canonical's `database_path` is relative, and the resolver resolves it against the repo root = the same real DB the relative literal hit by cwd coincidence.

## Out of scope

- The `discovery.py` / `source_extraction.py` enrichment sites (declined).
- Any change to `Config`'s semantics beyond adding the resolver function.
- Making `analyze_library.py` config-driven (a separate, deliberately-not-doing-it decision from the satellite design).
