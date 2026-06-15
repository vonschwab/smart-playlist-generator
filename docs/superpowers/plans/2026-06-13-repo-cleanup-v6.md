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

### Task B1: Kill list C — delete the legacy pre-DS engine (the big surgery)
The pre-pier-bridge engine is imported by `playlist_generator.py` but is the dead non-DS path. **Re-verify exact paths from B0** — they are NOT all under `src/playlist/` (e.g. the import is `from .similarity_calculator` → `src/similarity_calculator.py`; `genre_similarity_v2` likewise). Candidate cluster (verify each): `constructor`, `candidate_generator`, `ordering`, `diversity`, `history_analyzer`, `similarity_calculator`, `genre_similarity_v2`.
- [ ] Map every importer (B0 output). Confirm the only consumers are `playlist_generator.py` + each other (the legacy cluster) — if a **live** DS-path module imports one, it is NOT dead; stop and reassess.
- [ ] Remove the legacy code path from `playlist_generator.py` (the non-DS branch + its imports). Commit to `pipeline: ds` as the only engine.
- [ ] Delete the modules + their zombie tests. Deleting `similarity_calculator.py` also removes the dormant `_get_combined_genres` signature-preferring seam (genre-data-authority skill).
- [ ] **Guards:** `python -c "import src.playlist_generator"` clean; `ruff check src/playlist_generator.py` (no F401); **full non-slow suite green**; **one real multi-pier generation** via the `gui_fidelity` harness (read the playlist-testing skill first) with sane transition stats.
- [ ] Commit — `refactor(engine): delete legacy pre-DS pipeline; ds is the only engine`. Do this as 1–3 incremental commits if the untangle is large; keep each green.
- [ ] **Escape hatch:** if generation destabilizes and the cause isn't quickly found, revert the phase and raise it — do NOT ship a half-untangled engine.

### Task B2: Delete `vocab_normalization`
- [ ] Re-confirm 0 non-test importers (current: 0 in src/scripts/tools). Delete `src/genre/vocab_normalization.py` + `tests/unit/test_vocab_normalization.py`.
- [ ] Full suite green; commit.

### Task B3: Dead scripts — delete; research scripts — relocate
- [ ] **Delete** (all confirmed present): `scripts/sweep_pier_bridge_dials.py`, `run_dj_connector_bias_ab.py`, `run_dj_ladder_route_ab.py`, `run_dj_relaxation_micro_pier_demo.py`, `run_dj_union_pooling_stress_ab.py`, `fix_compound_genres.py`, `rebuild_sonic_tower_weighted.py` (dangerous post-2DFTM; also drop its CLAUDE.md gotcha), `build_windows.ps1` (PyInstaller, dead with PySide6 gone). Re-verify B0 shows no importer for each.
- [ ] **Move to `scripts/research/`** (referenced as reproducibility evidence — update the links in `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`): the `sonic_*` probes, `research_*`, `measure_genre_baseline.py`, `run_model_prior_album_tests.py`. **Also assess the new `pace_audition_*` scripts** — if they're one-shot harness scripts like the sonic audition, relocate them too.
- [ ] Commit deletions and moves separately.

### Task B4: Config deprecated-settings sweep
From B0's knob audit, against the **post-B1** code (engine removal orphans more keys):
- [ ] Delete the three known-dead keys: `playlists.cache_expiry_days`, `playlists.genre_similarity.use_artist_tags`, `playlists.similar_artists.boost` (the latter two were only read by the now-deleted `similarity_calculator`).
- [ ] Delete every other zero-reader key B0 found. Keep the ~37 live per-mode-suffixed keys (dynamic reads). Re-check the **new** pace (onset/bpm bands, rhythm soft penalty), MERT (`analyze.mert`, `artifacts.sonic_variant_override`), and genre keys are all live before pruning.
- [ ] `python tools/doctor.py` loads clean. Commit `config.example.yaml` (the local `config.yaml` is untracked — edit it too, not committed). Record before/after leaf-key counts in the message.

### Task B5: Close any remaining dead wiring B0 surfaced
- [ ] For each "declared-but-unimplemented" item from B0: either implement it (if intended, like mert-stage was) or remove the declaration (if abandoned). No configured value may resolve to a no-op. Commit per fix.

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
