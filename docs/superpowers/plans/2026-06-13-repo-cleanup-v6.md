# v6.0 Repo Cleanup & Release — Implementation Plan

> **Revised 2026-06-15** after the two divergent lanes (genre-data-quality + pace/MERT) were merged onto `cleanup/v6`. Supersedes the original single-session "land in-flight work then clean" version. Execute with superpowers:subagent-driven-development; steps use `- [ ]`.

**Goal:** One clean, professional branch carrying everything — genre + enrich + pace/MERT — with **all dead code and dead wiring removed**, docs rewritten for what actually shipped, versioned and tagged v6.0. Push is human-gated on a secret scrub.

**Mandate (user, 2026-06-15):** the repo must be clean. Deprecated/dead/stale code actively harms development — it causes miswiring and the illusion that dead wiring is functional. Take the time to be careful; remove it all. This elevates Phase B from "delete the known kill list" to "evidence-driven removal of every dead module, script, config knob, and unimplemented-but-declared stage."

**Tech stack:** Python 3.11 / pytest / ruff / mypy; React+TS+Vite (`web/`); SQLite; git on Windows + PowerShell.

---

## Where we are now (done)

- **Lanes merged** onto `cleanup/v6` (merge `f71daba`, verified both-directions: nothing lost, both lanes' symbols present, `storage.py` carries all three method-sets).
- **WIP-red settled by implementing** (not deleting): `stage_mert` analyze stage (7 tests) + enrich-pause halt-before-publish (2 tests); stale single-writer sonic-floor assertion updated for the MERT recalibration.
- **Suite green:** `pytest -m "not slow"` → **1767 passed, 6 skipped, 5 slow-deselected, 0 failed**.
- **config.yaml untracked** (pure deletion, working file kept) + **junk gitignored** (`data/*.db`, `*_test.db`, `*.bak*`, WAL/SHM, MERT scratch, `web/test-results/`).
- Branch commits on top of the merge: `dd10396`, `95b0e47`, `b94a1da`.

## Operating rules (read before every task)

- **Single-writer worktree.** All work happens in this worktree (`.claude/worktrees/v6cleanup`, branch `cleanup/v6`). Do not operate in the main checkout or touch `master` until Phase D. Stage explicit paths only — never `git add -A`/`-u`.
- **Never stage `config.yaml`.** It holds live keys (still in history). Before every commit: `git diff --cached --name-only | Select-String '^config\.yaml$'` must be empty.
- **Pytest: bounded, never piped** through `tail`/`head`/`Select-Object` (hook-blocked, hangs sessions). Redirect to a file and Read it. Full non-slow suite ≈ 2.5–5 min.
- **Irreplaceable, read-only:** `data/metadata.db`, `data/artifacts/**` (MERT shards/sidecar), audio files. No `metadata.db` writes anywhere in this plan. The worktree's `data/` lacks these (symlink didn't materialize); for the real-generation guard, point a throwaway config at the **main checkout's** absolute `data/` paths (read-only) — do not copy/junction the irreplaceable files.
- **Don't run `analyze_library` enrich** against the real DB — the merge brought the incremental guard, but a full re-derivation can still clobber the genre-quality fixes. Generation/test only.
- **Push is hard-gated** on the human's key rotation + history scrub. Nothing in this plan pushes.

---

## Phase A — Remaining hygiene

### Task A1: Stop the test-DB leak at the root
Some tests write SQLite DBs into `data/` (the source of the `data/*_test.db` junk). Redirect them to `tmp_path`.
- [ ] Find writers: `Select-String -Path (Get-ChildItem -Recurse tests -Include *.py) -Pattern 'data/ai_genre_(enriched|refinement)|data\\\\.*_test\.db|model_prior_test'`.
- [ ] Redirect each hard-coded `data/..._test.db` to a `tmp_path / "..."` fixture. Show before/after per hit.
- [ ] Verify no new `data/*.db` appears: capture `data` listing before/after running the touched test files; counts must not grow.
- [ ] Commit (explicit test paths) — `test(hygiene): redirect generated test DBs to tmp_path`.

### Task A2: Delete the `C:tmp…` root-junk yamls (main checkout)
These accidental Windows-path-bug files live in the **main checkout** working tree, not this worktree, and are untracked. They cannot be removed from here without reaching into another working tree.
- [ ] **Flag for the human** (or do once on `master` at Phase D): `Remove-Item 'C*tmpsp3a_taxonomy_handoff*.yaml'` from the repo root. No commit needed (untracked).

---

## Phase B — Dead code & dead-wiring removal (evidence-driven)

The 2026-06-10 audit is **stale** — the genre, pace, and MERT lanes all landed since. Re-derive the dead set against the **current merged tree**, then remove in careful, independently-verified commits. After every deletion commit: full non-slow suite green + (for engine/runtime changes) one real multi-pier generation.

### Task B0: Fresh dead-code + dead-wiring audit (no deletions yet)
- [ ] **Static reachability:** AST import graph over `src/` + `scripts/`, BFS from every production entrypoint (`main_app.py`, `tools/*.py`, `scripts/analyze_library.py` and the other scripts it imports, `src.playlist_gui.worker`, `src.playlist_web.app`). Modules unreachable from all entrypoints = static-dead candidates.
- [ ] **Knob audit (the "dead wiring" the mandate targets):** every leaf key in `config.example.yaml` checked for a reader — static reads, dynamic per-mode reads (`cfg.get(f"{base}_{mode}")`), policy-layer reads. Zero-reader keys = dead config. Cross-check the **new** pace/MERT/genre keys too.
- [ ] **Declared-but-unimplemented audit:** every `Literal`/enum/registry of stages, modes, variants, commands (e.g. `request_models.AnalyzeLibraryStage`, worker `STAGE_FUNCS`, web tool stages, `sonic_variant`) — confirm each declared value has a live implementation. (This is the class that produced the mert-stage/enrich-pause WIP-red tests.)
- [ ] **Zombie tests:** tests importing dead-module candidates.
- [ ] Write findings to `docs/run_audits/dead_code_2026-06-15.md` (gitignored path — scratch evidence). Each candidate cites its evidence. This drives B1–B5.

### Task B1: Legacy pre-DS engine removal — DEFERRED to its own focused pass ⚠️
**User decision (2026-06-15): do NOT rush this; treat it as a dedicated, extremely-careful effort.** It gates the v6.0 tag (Phase D) but runs as its own pass, not inline with the rest of the cleanup.

**Why it's not a simple delete (2026-06-15 finding):** the legacy cluster (`constructor`, `candidate_generator`, `ordering`, `diversity`, `history_analyzer`, `similarity_calculator` @ `src/`, `genre_similarity_v2` @ `src/`) is **not an isolated dead branch**. `playlist_generator.py`'s live methods (`create_playlist_batch/for_artist/for_genre/from_seed_tracks` — all called by `main_app` + the worker) run the **DS pipeline** (`run_ds_pipeline`) but still *delegate* to these modules for auxiliary work: `history_analyzer` (listening-history seed selection, `is_collaboration_of`, `analyze_top_artists`), `candidate_generator` (`generate_candidates`, `CandidateConfig`, `build_seed_title_set`), `diversity` (`diversify_by_artist_cap`), `similarity_calculator` (scoring). The 2026-06-10 audit rated these "5–12% runtime coverage" — mostly the dead old engine, but with a few helper functions still invoked. Done wrong, this breaks live generation.

**Method (coverage-guided, the audit's own approach):**
- [ ] Set up read-only `data/` in the worktree (point the untracked `config.yaml` at the main checkout's absolute `data/metadata.db` + artifact paths). Never write them.
- [ ] Run real multi-pier generations (GUI 10-seed + artist-mode CLI) under `coverage.py --branch`. Identify, per legacy module, exactly which functions execute vs are dead.
- [ ] For each still-executed helper: either it is genuinely needed (extract it to a live home / inline it) or its caller is itself a dead path (remove the caller). Decide per function, with evidence.
- [ ] Remove the dead engine + dormant `_get_combined_genres` seam; untangle `playlist_generator.py`'s imports; commit to `pipeline: ds` as the only engine.
- [ ] **Guards after every increment:** import clean; `ruff` no F401; full non-slow suite green; **a real multi-pier generation with sane transition stats** (playlist-testing skill). Incremental commits, each green.
- [ ] **Folds in:** the two B4 keys read only by `similarity_calculator` (`genre_similarity.use_artist_tags`, `similar_artists.boost`) are removed here, with their reader.
- [ ] **Escape hatch:** if generation destabilizes and the cause isn't fast to find, revert and raise — never ship a half-untangled engine.

### Task B2: Delete `vocab_normalization` — DONE (`3ec567c`)
Deleted module + test (0 importers). Suite green (1716).

### Task B3a: Delete dead scripts — DONE (`d0f1760`)
Deleted the A/B sweeps, `fix_compound_genres`, `rebuild_sonic_tower_weighted` + `src/features/sonic_rebuild.py`, `build_windows.ps1`, and their tests (−3706 LOC). Suite green (1712).

### Task B3b: Relocate research/audition scripts — DEFERRED (organizational, not dead-code)
- [ ] Move to `scripts/research/`: the `sonic_*` probes, `research_*`, `measure_genre_baseline.py`, `run_model_prior_album_tests.py`, and the audition harnesses (`sonic_audition_*`, `genre_audition_*`, `pace_audition_*`, `pace_calibration_sweep.py`) — these have tests, so update test import paths + the links in `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`. Lower priority than dead-code; can land late.

### Task B4: Config deprecated-settings sweep (the truly-zero-reader keys now; legacy-coupled keys in B1)
- [ ] Delete `playlists.cache_expiry_days` (0 readers anywhere). The other two known-dead keys (`use_artist_tags`, `similar_artists.boost`) are read only by `similarity_calculator` → remove them **in B1** with their reader, to keep each phase self-consistent.
- [ ] From a fresh knob audit, delete every other key with **zero readers anywhere** (not even the legacy engine). Keep the ~37 live per-mode-suffixed keys (dynamic reads) and the new pace/MERT/genre keys (verify each is read).
- [ ] `python tools/doctor.py` loads clean. Commit `config.example.yaml`; edit the untracked `config.yaml` too. Record before/after leaf-key counts.

### Task B5: Close any remaining dead wiring
- [ ] The two declared-but-unimplemented items B0 surfaced (mert analyze stage, enrich-pause) are already **implemented** (`dd10396`). Re-run the declared-but-unimplemented audit after B1; resolve anything new (implement if intended, else remove the declaration). No configured value may resolve to a no-op.

---

## Phase C — Documentation overhaul (merged reality)

### Task C1: README.md — full v6.0 rewrite
- [ ] **Pace mode:** replace the stale cosine-floor table with the BPM + onset-rate **bands** + soft rhythm penalty (values from `mode_presets.PACE_MODE_PRESETS`: strict 0.30/0.40, narrow 0.50/0.60, dynamic 0.75/0.85, off ∞; rhythm-soft 0.35/0.25/0.15).
- [ ] **Four axes** incl. `cohesion_mode` (drives the beam) vs genre/sonic/pace (pool composition).
- [ ] **Sonic:** document that the **default sonic space is now the learned MERT-v1-95M embedding** (`whiten_l2`, 768-d), with the 162-d towers as the rollback (`artifacts.sonic_variant_override`). Note the listen-test flip gate per the mert-migration memory — describe MERT as the shipped default but flag the towers fallback honestly.
- [ ] **Genre system:** enriched-genre **authority** (`release_effective_genres` via `authority.py`), SP3a **layered taxonomy graph**, multisource **Claude enrichment** (scan→genres→discogs→lastfm→**mert**→enrich→publish), `genre_source: graph` artifact bake, the **fusion rebalance + surgical delta migration** (storefront/lastfm-identity hardening), **enrich pauses** on transient rate limits before publish, the **Genre Review** GUI panel + graph-canonical chips.
- [ ] **GUI:** browser-only (`tools/serve_web.py`); Generate/Tools/Genre-Review tabs; remove all PySide6 text. Fix Quick Start + structure tree. Version history → 6.0.

### Task C2: CHANGELOG (root + docs/CHANGELOG.md)
- [ ] v6.0 entry covering: MERT-default sonic + pace bands; genre authority/graph/Claude-enrichment/analyze-graph-stages/mert-stage/enrich-pause; fusion rebalance + delta migration; genre-review panel + canonical chips; browser GUI replacing PySide6; dead-code removal (engine, scripts, config); secret untracking.

### Task C3: Supporting docs + version
- [ ] `AGENTS.md`: kill stale PySide6 / `[gui]` refs → browser GUI / `[web]`.
- [ ] `docs/README.md` index: add the new docs (MERT design + whitening findings, pace-retune, genre-review/chips, genre-data-quality findings, genre-data-authority); drop pointers to anything Phase B deleted.
- [ ] `docs/ARCHITECTURE.md`, `docs/TROUBLESHOOTING.md`, `docs/GOLDEN_COMMANDS.md`: bring to the merged era (web GUI, graph authority, MERT default, pace bands, `mert` stage).
- [ ] `pyproject.toml` version `4.0.0` → `6.0.0`.
- [ ] `CLAUDE.md`: drop notes Phase B invalidates (e.g. the `rebuild_sonic_tower_weighted.py` gotcha); reconcile the Layer-2 "sonic is multi-dimensional" / tower-weight commitments (#8/#17/#18) with MERT-as-default per the mert-migration memory's note to amend when MERT becomes permanent.

---

## Phase D — Release gate + integration

1. [ ] Full `pytest` (note slow/golden subset separately). `ruff check`; `mypy`. `python tools/doctor.py`.
2. [ ] One real multi-pier generation — sane transition stats.
3. [ ] `npm --prefix web run build` (stale-dist trap).
4. [ ] **Integrate to master (local only):** fast-forward `master` to the unified `cleanup/v6` (done in the main checkout, or by the human alongside the scrub). Tag `v6.0`.
5. [ ] **STOP — do not push.** Hand to the human for: rotate the live Last.fm + OpenAI keys, scrub `config.yaml` from history (filter-repo/BFG), then push `master` + `v6.0`.

## Out of scope (explicit)
- MERT listen-test flip decision (Dylan's ear — the real gate; this plan ships MERT-default as already committed, towers as rollback).
- Genre-similarity audition harness implementation.
- Any `metadata.db` / artifact / audio write.
- The push and history scrub (human-driven).
