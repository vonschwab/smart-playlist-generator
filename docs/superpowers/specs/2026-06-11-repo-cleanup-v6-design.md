# Repo cleanup & v6.0 release — design

**Date:** 2026-06-11 (revised 2026-06-13 for codebase drift)
**Status:** Approved (user, 2026-06-11); scope refreshed after a large body of work landed.
**Goal:** A professional, clean repo committed and tagged as v6.0 on GitHub: in-flight
work landed, secrets removed, junk and deprecated code/tests/config removed, documentation
rewritten to match what actually shipped this round.

## What this round shipped (the things docs must now reflect)

Since the original spec, a large amount landed as commits (master is now ~92 commits
ahead of origin). The cleanup and especially the documentation must cover all of it:

- **Pace mode was re-architected** (committed). The old rhythm-cosine hard floors are
  gone; pace now gates on two embedding-independent **hard bands** — BPM log-distance +
  onset-rate log-distance — plus a **soft** rhythm-cosine penalty (tower variants only).
  The README's entire v5.0 pace table (admission floor 0.55 / bridge floor 0.65, etc.) is
  now wrong. Spec: `docs/superpowers/specs/2026-06-12-pace-gate-retune-design.md`.
- **Genre authority + layered graph became the published source of truth** (committed,
  `bfebef0` + the genre-review/chips series). `release_effective_genres` (metadata.db,
  written only by the publish stage, read via `src/genre/authority.py`) is THE authority;
  the artifact bakes it in via `genre_source: graph`; the SP3a layered taxonomy
  (`data/layered_genre_taxonomy.yaml`, 455 genres) is the structural source of truth. The
  rules live in the new `genre-data-authority` skill.
- **Genre Review panel** (new GUI tab) + **graph-canonical genre chips** (sub→broad order,
  6-cap +N overflow) shipped. Specs: `2026-06-11-genre-review-panel-design.md`,
  `2026-06-11-genre-chips-granularity-design.md`.
- **Multisource Claude enrichment backend** + analyze-library graph stages
  (lastfm/enrich/publish) shipped (committed). Provider factory, Claude Agent SDK, no API
  billing.
- **MERT learned sonic embedding** — foundations shipped (extraction phases 1/2/5,
  calibration, `whiten_l2` transform selection); the **analyze `mert` stage** + build
  integration is the remaining uncommitted in-flight work. Validation doc:
  `docs/MERT_WHITEN_NEIGHBORS_20SEEDS.md`; design:
  `2026-06-11-mert-sonic-embedding-design.md`. Still experimental/opt-in for v6.0 — towers
  remain the default sonic space and the rollback path.
- **Web GUI grew** a Tools panel (analyze/enrich), a seed-track section with localStorage
  persistence, the Review tab, and the genre chips. PySide6 is fully gone.
- **New developer skills** encode the rules: `genre-data-authority`, `web-gui`,
  `evaluation-methodology`, plus updates to `playlist-testing` and `taxonomy-growth`.

## Decisions (user-confirmed)

1. **In-flight work lands first** as proper feature commits, then cleanup on top. v6.0
   includes MERT infrastructure in its current opt-in/experimental state.
2. **Version is v6.0** (tag `v6.0`, `pyproject.toml` → `6.0.0`). Drift to fix: tag `v5.0`
   exists, README says 5.0, pyproject says 4.0.0.
3. **Kill list C is in scope**, sequenced last so it can be cut to post-release if the
   `playlist_generator.py` surgery destabilizes generation.
4. **Both config files get a full deprecated-settings sweep**, run after kill list C
   (engine removal orphans more keys) and accounting for the pace re-architecture.
5. `src/genre/authority.py` is now **load-bearing** (it is THE genre authority) — the
   original "audit review-with-user delete" note is void; it stays. `src/genre/
   vocab_normalization.py` is still delete-eligible (imported only by its own test) —
   re-verify against the post-graph code before deleting.
6. **Secrets:** rotate all four leaked keys, `git rm --cached config.yaml` to untrack it
   going forward, ship a secret-free `config.example.yaml`. **No history rewrite, no
   force-push** (user-chosen; the 16 local branches and concurrent checkouts make a
   filter-repo rewrite too disruptive, and rotation makes the committed values dead).

## Phase 0 — Land in-flight work

Current uncommitted working tree (re-confirm with `git status` at execution time, and per
CLAUDE.md session discipline stage explicit paths only — never `git add -A`):

1. **feat(mert): analyze-library `mert` stage + build/calibration integration**
   `scripts/analyze_library.py`, `scripts/extract_mert_sidecar.py`,
   `scripts/calibrate_mert_transform.py`, `scripts/build_beat3tower_artifacts.py`,
   `src/playlist/request_models.py`, `src/playlist/analyze_library_results.py`,
   `src/playlist_gui/worker.py`, `src/playlist_generator.py`, `src/genre/authority.py`
   (if its diff is MERT-adjacent — inspect; otherwise group with the genre commit),
   `config.example.yaml` + `config.yaml` (`analyze.mert` block), the new MERT tests, and
   the modified analyze/worker/tools/genre tests that move with them.
2. **chore(enrichment): remove dead `_is_ai_adjudicated` helper** + associated test diffs
   (`src/ai_genre_enrichment/hybrid_evidence.py`, `layered_assignment.py`, `storage.py`
   if part of the same trim — inspect each diff before grouping).
3. **chore(skills): genre-data-authority + web-gui skills; playlist-testing update**
   `.claude/skills/genre-data-authority/` (new), `.claude/skills/web-gui/` (new),
   `.claude/skills/playlist-testing/SKILL.md` (1-line diff).

Exact file→commit grouping is resolved at implementation time by reading each diff; some
modified test files may belong to commit 1 vs 2. **Verify each commit's tests pass before
moving on** (per verification-before-completion).

## Phase 1 — Secrets, junk purge, root-cause leak fixes

### 1a. Secrets (do first — gating for GitHub)

- User rotates all four keys in `config.yaml` (Discogs token, Last.fm api_key, OpenAI
  `sk-proj-…` key, line-412 token). I cannot do this — flag it explicitly as a required
  manual step and do not push until confirmed.
- `git rm --cached config.yaml` (local file kept untouched). `.gitignore` already lists it;
  it was tracked in practice.
- Confirm `config.example.yaml` contains **only placeholders** (`token: ''`, etc.) — no
  real values. Add the `analyze.mert` block there with placeholder-safe defaults.

### 1b. Delete from disk (generated/accidental; nothing irreplaceable)

- `Ctmpsp3a_taxonomy_handofflayered_genre_taxonomy_p9_test.yaml`,
  `Ctmpsp3a_taxonomy_handofftaxonomy_pass8_isolated.yaml` (repo root; Windows path-bug artifacts)
- All generated test DBs: `data/ai_genre_enriched_*_test.db`,
  `data/ai_genre_refinement_*test.db`, `data/model_prior_test.db`,
  `data/ai_genre_enrichment_live_test.db`
- Stray `-shm`/`-wal` sidecars on backup files (e.g.
  `data/ai_genre_enrichment.db.bak_20260612_163129-shm/-wal`)
- `data/vocab.bak.yaml`, `data/genre_vocabulary.yaml.polluted_20260604.bak`
- `web/test-results/`

### 1c. Never delete (keep local, ensure gitignored)

`data/ai_genre_enrichment.db` and its `.bak_*`, `data/metadata.db` + `data/metadata.db.bak.*`,
`data/mert_cal_*.txt`, `data/mert_calibration_track_ids.txt`, everything under
`data/artifacts/` (MERT shards/sidecar are irreplaceable, ~55h CPU).

### 1d. Root causes

- `tests/unit/test_ai_genre_enrichment.py` writes test DBs into `data/` → redirect to
  `tmp_path` so the junk never regrows. (Re-verify which test(s) leak — there may be more
  than one given the count of `*_test.db` files.)
- `.gitignore` additions: `data/*.db`, `data/*.db.bak*`, `data/*.bak*`,
  `data/*-shm`, `data/*-wal`, `web/test-results/`, `data/mert_cal_*.txt`,
  `data/mert_calibration_track_ids.txt`.

## Phase 2 — Test-suite health

- Run the full suite with **zero deselects**; triage every failure (the ~12 perma-fails +
  the corrected 13-test SP3a deselect list). Each gets **fixed or deleted with a stated
  reason** — no third option. Per CLAUDE.md session discipline: bound the run with the
  tool timeout, never pipe through `tail`/`head`; quote real pass/fail counts.
- Remove the `qtbot` no-op stub from `tests/conftest.py` if still present.
- Exit criterion: `pytest -m "not slow"` green with no deselect flags on a clean checkout,
  then a full run. Zombie tests for Phase-3 modules go with their modules.

## Phase 3 — Kill list C, remaining dead code, config sweep

### 3a. Legacy pre-DS engine (kill list C)

Delete the pre-pier-bridge engine and untangle `playlist_generator.py` imports; commit to
`pipeline: ds` as the only engine. Re-verify the exact file list against current `src/`
before deleting (paths drift): `constructor.py`, `candidate_generator.py`, `ordering.py`,
`diversity.py`, `history_analyzer.py`, `similarity_calculator.py`, `genre_similarity_v2.py`.
**Bonus:** removing `similarity_calculator.py` also kills the dormant
`_get_combined_genres` signature-preferring seam flagged in the `genre-data-authority` skill.

**Guards after surgery:** full test suite, one real CLI generation (multi-pier artist mode
via the `gui_fidelity`/`generate_like_gui` harness per the playlist-testing skill), web GUI
smoke. **Escape hatch:** if generation destabilizes, revert the phase and ship v6.0 without
it (user-approved fallback).

### 3b. Audit holdovers

- `src/genre/authority.py` — **KEEP** (now THE genre authority; original delete note void).
- `src/genre/vocab_normalization.py` — re-verify it is still imported only by
  `tests/unit/test_vocab_normalization.py`; if so, delete both. If the graph lane picked it
  up, keep.

### 3c. Scripts

**Delete** (findings preserved in docs + git history): `sweep_pier_bridge_dials.py`,
`run_dj_connector_bias_ab.py`, `run_dj_ladder_route_ab.py`,
`run_dj_relaxation_micro_pier_demo.py`, `run_dj_union_pooling_stress_ab.py`,
`fix_compound_genres.py`, `rebuild_sonic_tower_weighted.py` (dangerous post-2DFTM; remove
its CLAUDE.md gotcha mention too), `build_windows.ps1` (PyInstaller desktop build, dead).

**Move to `scripts/research/`** (referenced as reproducibility evidence in
`docs/SONIC_PHASE2_HARMONY_FINDINGS.md`; update those links): the 8 `sonic_*` probe
scripts, `research_genre_similarity.py`, `research_sonic_hubness.py`,
`research_sonic_transition.py`, `measure_genre_baseline.py`, `run_model_prior_album_tests.py`.

**Keep in place:** everything imported by `scripts/analyze_library.py` (verify the import
list at execution — it now includes `extract_mert_sidecar`), the active audition harnesses
(`sonic_audition_*`, `genre_audition_*`), `calibrate_mert_transform.py`, `calibrate_*`,
`diagnose_*`, `ai_genre_enrich.py`, `build_graph_genre_similarity.py`,
`consolidate_enrichment_dbs.py`, `extract_harmony_2dftm_sidecar.py`,
`fold_2dftm_into_artifact.py`, `growth_candidate_report.py`, `import_mbids_from_csv.py`,
`publish_genres.py`, `refresh_artifact_genres.py`, `update_file_genres.py`,
`validate_published_genres.py`, `smoke_test.ps1`.

### 3d. Full config deprecated-settings sweep

Re-run the 2026-06-10 knob-audit method against the **post-3a** code, now accounting for
the pace re-architecture:

1. Enumerate every leaf key in `config.example.yaml` and the local `config.yaml`.
2. For each: static reads; dynamic per-mode reads (`cfg.get(f"{key}_{mode}")`); per-mode
   base-key expansion; policy-layer reads (`src/playlist_gui/policy.py`); worker/web request
   plumbing.
3. Zero readers → delete from both files. Readers only in 3a/3b-deleted code → delete.
4. **Pace-specific:** the rhythm-cosine admission/bridge **floors are now zeroed/removed**
   in presets — confirm whether any `config.yaml` pace-cosine keys are now dead and prune
   them; keep the live BPM-band + onset-band + rhythm-soft-penalty keys.
5. Known-live trap: the ~37 per-mode suffixed keys are live via dynamic reads — keep.
6. Known-dead from the audit (delete): `playlists.cache_expiry_days`,
   `playlists.genre_similarity.use_artist_tags`, `playlists.similar_artists.boost`.
7. Clean the local `config.yaml` too, preserving the user's live values for surviving keys.
8. Short before/after key-count note in the commit message.

The Phase-0 startup-validation guardrail stays **deferred to post-release** (behavior
change, not cleanup) — noted in the roadmap.

## Phase 4 — Documentation overhaul (the heart of this round)

The codebase moved far past the docs. Rewrites, not touch-ups:

### 4a. README.md (full rewrite for v6.0)

- **Pace mode section rewritten** to the new mechanism: BPM log-distance band + onset-rate
  log-distance band (hard) + soft rhythm-cosine penalty; bands widen on backoff; per-mode
  preset table from `PACE_MODE_PRESETS` (not the stale 0.55/0.65 cosine floors). Note the
  MERT-durability property (bands are DB features, survive the fold).
- **Four mode axes** including `cohesion_mode` (the README still says "three independent
  axes"; CLAUDE.md is authoritative: cohesion drives the beam, genre/sonic/pace drive pool
  composition).
- **Genre system rewritten**: enriched-genre authority (`release_effective_genres` via
  `authority.py`), the SP3a layered taxonomy graph (455 genres), multisource Claude
  enrichment (scan→genres→discogs→lastfm→enrich→publish→artifacts), `genre_source: graph`
  bakes the authority into the artifact. Multi-genre signatures preserved.
- **Sonic section**: current default is 162-dim tower_weighted (rhythm 9 + timbre 57 +
  2DFTM harmony 96, weights 0.20/0.50/0.30). MERT learned embedding documented as an
  **experimental opt-in** (`sonic.variant: mert`, `whiten_l2` post-processing), with the
  towers as default + rollback; link `MERT_WHITEN_NEIGHBORS_20SEEDS.md`.
- **Browser GUI as the only front-end**: remove all PySide6 right-click/Help-menu text;
  document the web Generate / Tools / Genre Review tabs, the Tools panel (Analyze Library +
  Enrich), seed staging, graph-canonical genre chips (sub→broad, 6 + `+N`), and web track
  replacement context menu.
- Corrected Quick Start (`scripts/analyze_library.py` orchestrator with stage list incl.
  `mert`; `tools/serve_web.py`). Version history table extended to 6.0. Project-structure
  tree refreshed (`src/playlist_web/`, `web/src/components/*`).

### 4b. CHANGELOG (root summary + docs/CHANGELOG.md)

v6.0 entry covering the whole round: browser GUI replacing PySide6; enriched-genre
authority + layered graph + Claude enrichment backend + analyze graph stages; genre review
panel + canonical chips; pace-gate re-architecture (BPM+onset bands, soft rhythm penalty);
MERT phases + analyze stage (experimental); dead-code cleanup (lists A/B/C); wiring fixes
(beam widths, pace gate, web dj_bridging, genre-source graph); secret-handling note.

### 4c. Other docs

- **AGENTS.md**: remove the 2 stale PySide6 / `pip install -e .[gui]` references; align to
  `[web]` and the browser GUI.
- **docs/README.md** index: add the new docs (MERT design + whitening findings, pace-retune
  spec, genre-review/chips specs, genre-data-authority); drop dead pointers.
- **docs/ARCHITECTURE.md**: bring to the web-GUI + graph-authority + pace-bands era;
  reconcile the Layer-2 "sonic is multi-dimensional" note with the MERT direction per the
  MERT design's architectural note (amend when MERT becomes default — for v6.0, document the
  current tower default and the MERT path as opt-in).
- **docs/TROUBLESHOOTING.md**: refresh stale paths (`repo_refreshed`), add web-GUI traps
  (stale `web/dist`, worker restart) cross-referencing the `web-gui` skill.
- **docs/GOLDEN_COMMANDS.md**: verify current; add the `mert` stage + review/tools commands
  if missing.
- **pyproject.toml** version → `6.0.0`.
- **CLAUDE.md**: drop the `rebuild_sonic_tower_weighted.py` gotcha (script deleted in 3c);
  any other statements cleanup invalidates.

## Phase 5 — Release gate

In order; all must pass before tagging:

1. **Secrets confirmed rotated** by the user (gating).
2. Full `pytest` (note `slow`/`golden` subset separately if long).
3. `ruff check`, `mypy`.
4. `python tools/doctor.py`.
5. One real CLI generation (multi-pier artist mode) — sanity-check transition stats.
6. `npm --prefix web run build` (stale-dist trap from the web-gui skill).
7. Version-bump commit, tag `v6.0`, push `master` + tags (carries the ~92 unpushed commits).

## Commit discipline

Each phase = one or more focused conventional commits; destructive phases (1, 3) list
deleted files in the commit body. No history rewriting; no force push. Stage explicit paths
only; re-check `git status` immediately before each commit (concurrent sessions on this
checkout — CLAUDE.md session discipline).

## Out of scope (explicitly)

- Finishing MERT extraction/calibration/fold and the default flip (post-release; towers stay
  default for v6.0).
- Startup config-validation guardrail (post-release roadmap item).
- Genre-similarity audition harness implementation (spec/plan committed, untouched).
- Git history rewrite / secret scrubbing from old commits (user chose rotate-only).
- Any write to `data/metadata.db`, MERT shards/sidecar, or audio files.
