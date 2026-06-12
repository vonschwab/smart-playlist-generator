# Repo cleanup & v6.0 release — design

**Date:** 2026-06-11
**Status:** Approved (user, 2026-06-11)
**Goal:** A professional, clean repo committed and tagged as v6.0 on GitHub: in-flight
work landed, junk and deprecated code/tests/config removed, documentation rewritten to
match what actually shipped this round (browser GUI, layered genre graph, multisource
Claude enrichment, MERT foundations).

## Decisions (user-confirmed)

1. **In-flight work lands first** as proper feature commits, then cleanup commits on top.
   The v6.0 tag includes MERT infrastructure in its current opt-in/experimental state.
2. **Version is v6.0** (tag `v6.0`, `pyproject.toml` → `6.0.0`). Existing drift: tag
   `v5.0` exists, README says 5.0, pyproject says 4.0.0.
3. **Kill list C is in scope**, sequenced last before the tag so it can be cut to
   post-release if `playlist_generator.py` surgery gets hairy.
4. **Both config files get a full deprecated-settings sweep**, not just the three keys
   from the 2026-06-10 audit. Runs after kill list C (engine removal orphans more keys).
5. `src/genre/authority.py` and `src/genre/vocab_normalization.py` (audit "review with
   user" items) are deleted. `config.yaml` is untracked (`git rm --cached`, local file kept).

## Phase 0 — Land in-flight work

All 28 affected tests pass (verified 2026-06-11). Three commits, grouped by concern:

1. **feat(mert): analyze-library `mert` stage + worker cancel concurrency**
   `scripts/analyze_library.py`, `scripts/extract_mert_sidecar.py`,
   `src/playlist/request_models.py`, `src/playlist/analyze_library_results.py`,
   `src/playlist_gui/worker.py`, `tests/fixtures/fake_worker.py`,
   `config.example.yaml` (`analyze.mert` block), `CLAUDE.md` (MERT data-safety notes),
   new tests `test_analyze_mert_stage.py`, `test_mert_extraction_cancel.py`,
   `test_worker_cancel_concurrency.py`, plus modified analyze/worker/tools tests.
2. **feat(web): seed track section with localStorage persistence**
   `web/src/components/SeedTrackSection.tsx`, `web/src/lib/useLocalStorage.ts`,
   `web/src/components/GenerateControls.tsx`, `web/tests/seeds.spec.ts`.
3. **chore(enrichment): remove dead `_is_ai_adjudicated` helper + test updates**
   `src/ai_genre_enrichment/hybrid_evidence.py` and associated hybrid/layered test diffs.

Also committed: `.claude/skills/web-gui/`, `docs/superpowers/plans/2026-06-11-web-tools-panel.md`,
`docs/TAXONOMY_EXPANSION_BRIEF.md`. Exact file→commit grouping resolved at implementation
time by reading each diff (some modified test files may belong to commit 1 vs 3).

## Phase 1 — Junk purge + root-cause leak fixes

**Delete from disk** (generated/accidental; nothing irreplaceable):

- `Ctmpsp3a_taxonomy_handofflayered_genre_taxonomy_p9_test.yaml`,
  `Ctmpsp3a_taxonomy_handofftaxonomy_pass8_isolated.yaml` (repo root; Windows path-bug artifacts)
- All generated test DBs: `data/ai_genre_enriched_*_test.db`,
  `data/ai_genre_refinement_*test.db`, `data/model_prior_test.db`,
  `data/ai_genre_enrichment_live_test.db`
- `data/vocab.bak.yaml`, `data/genre_vocabulary.yaml.polluted_20260604.bak`
- `web/test-results/`

**Never delete** (keep local, ensure gitignored): `data/ai_genre_enrichment.db` and its
`.bak_*`, `data/metadata.db.bak.*`, `data/mert_cal_chunk_*.txt`,
`data/mert_calibration_track_ids.txt`, `data/metadata.db` itself, everything under
`data/artifacts/` (MERT shards/sidecar are irreplaceable).

**Root causes:**

- `tests/unit/test_ai_genre_enrichment.py` writes test DBs into `data/` → redirect to
  `tmp_path` so the junk never regrows.
- `.gitignore` additions: `data/*.db`, `data/*.db.bak*`, `data/*.bak*`,
  `web/test-results/`, `data/mert_cal_*.txt`, `data/mert_calibration_track_ids.txt`.
- Untrack `config.yaml` (`git rm --cached config.yaml`; local file untouched). It is
  user-local config, listed in `.gitignore` but tracked in practice.

## Phase 2 — Test-suite health

- Run the full suite (`pytest -m "not slow"`, then full) with **zero deselects**.
- Triage every failure — the ~12 perma-fails plus the 13-test deselect list referenced in
  the SP3a notes. Each gets **fixed or deleted with a stated reason**; no third option.
- Remove the `qtbot` no-op stub from `tests/conftest.py` if still present (PySide6 leftover).
- Exit criterion: `pytest -m "not slow"` green with no deselect flags on a clean checkout.

## Phase 3 — Kill list C, remaining dead code, config sweep

### 3a. Legacy pre-DS engine (kill list C)

Delete `src/playlist/constructor.py`, `candidate_generator.py`, `ordering.py`,
`diversity.py`, `history_analyzer.py`, `similarity_calculator.py`,
`src/genre_similarity_v2.py` (paths verified at implementation time); untangle
`playlist_generator.py` imports; delete their zombie tests. This commits to
`pipeline: ds` as the only engine.

**Guards after surgery:** full test suite, one real CLI generation
(multi-pier artist mode per the playlist-testing skill), web GUI smoke.
**Escape hatch:** if the surgery destabilizes generation, revert the phase and ship v6.0
without it (user-approved fallback).

### 3b. Audit holdovers

- `src/genre/authority.py` — zero imports anywhere (verified 2026-06-11). Delete.
- `src/genre/vocab_normalization.py` — imported only by its own test; superseded by the
  graph/adjudicator lane. Delete with `tests/unit/test_vocab_normalization.py`.

### 3c. Scripts

**Delete** (findings documented in docs; git history preserves the code):
`sweep_pier_bridge_dials.py`, `run_dj_connector_bias_ab.py`, `run_dj_ladder_route_ab.py`,
`run_dj_relaxation_micro_pier_demo.py`, `run_dj_union_pooling_stress_ab.py`,
`fix_compound_genres.py`, `rebuild_sonic_tower_weighted.py` (dangerous post-2DFTM; the
fold script supersedes it — also remove its CLAUDE.md mention), `build_windows.ps1`
(PyInstaller desktop build, dead with PySide6 gone).

**Move to `scripts/research/`** (kept: referenced as reproducibility evidence in
`docs/SONIC_PHASE2_HARMONY_FINDINGS.md`; update those links): `sonic_gate1_blend.py`,
`sonic_gate2_rhythm.py`, `sonic_harmony_richer_probe.py`, `sonic_harmony_weight_sweep.py`,
`sonic_keyinvariance_check.py`, `sonic_beatsync_2dftm.py`, `sonic_tower_diagnostic.py`,
`sonic_phase1_metrics.py`, `research_genre_similarity.py`, `research_sonic_hubness.py`,
`research_sonic_transition.py`, `measure_genre_baseline.py`, `run_model_prior_album_tests.py`.

**Keep in place:** everything imported by `scripts/analyze_library.py` (verified:
`scan_library`, `update_sonic`, `update_genres_v3_normalized`, `update_discogs_genres`,
`fetch_mbids_musicbrainz`, `validate_published_genres`, `build_beat3tower_artifacts`,
`fold_2dftm_into_artifact`, `build_genre_embedding`, `extract_mert_sidecar`), the active
audition harnesses (`sonic_audition_*`, `genre_audition_*`), `calibrate_*`, `diagnose_*`,
`ai_genre_enrich.py`, `build_graph_genre_similarity.py`, `consolidate_enrichment_dbs.py`,
`extract_harmony_2dftm_sidecar.py`, `growth_candidate_report.py`, `import_mbids_from_csv.py`,
`publish_genres.py`, `refresh_artifact_genres.py`, `update_file_genres.py`, `smoke_test.ps1`.

### 3d. Full config deprecated-settings sweep (user addition)

Method (mirrors the 2026-06-10 knob audit, re-run against post-3a code):

1. Enumerate every leaf key in `config.example.yaml` and the local `config.yaml`.
2. For each key, check: static reads; dynamic per-mode reads
   (`cfg.get(f"{key}_{mode}")` — `config.py:144`-style); per-mode base-key expansion;
   policy-layer reads (`src/playlist_gui/policy.py`); worker/web request plumbing.
3. Zero readers → delete from both files. Readers only in code deleted by 3a/3b →
   delete from both files.
4. Known-live trap: the ~37 per-mode suffixed keys are live via dynamic reads — keep.
5. Known-dead from the audit (delete): `playlists.cache_expiry_days`,
   `playlists.genre_similarity.use_artist_tags`, `playlists.similar_artists.boost`.
6. The local `config.yaml` (untracked by now) is cleaned too, preserving the user's
   live values for every surviving key.
7. Produce a short before/after key-count note in the commit message.

The Phase-0 startup-validation guardrail ("a configured knob that can't act is a startup
error") stays **deferred to post-release** — it is a behavior change, not cleanup. Noted
in the roadmap.

## Phase 4 — Documentation overhaul

- **README.md** rewritten for v6.0: browser GUI as the only front-end (PySide6 right-click
  / Help-menu text removed; web context-menu track replacement documented), **four** mode
  axes including `cohesion_mode`, 162-dim 2DFTM tower-weighted sonic, enriched-genre
  authority + layered taxonomy graph + multisource Claude enrichment, MERT marked
  experimental/in-progress, corrected Quick Start (`analyze_library.py` orchestrator,
  `serve_web.py`), version history table with 6.0.
- **CHANGELOG** (root summary + `docs/CHANGELOG.md`): v6.0 entry covering the round —
  browser GUI replacing PySide6, enriched genre authority + layered graph + Claude
  enrichment backend, analyze-library graph stages, MERT phases 1/2/5 + analyze stage,
  dead-code cleanup (lists A/B/C), wiring fixes (beam widths, pace gate, web dj_bridging).
- **AGENTS.md**: remove stale PySide6 / `[gui]`-extra guidance; align with `[web]`.
- **docs/README.md** index refreshed; `docs/ARCHITECTURE.md` and `docs/TROUBLESHOOTING.md`
  brought to the web-GUI era; `docs/GOLDEN_COMMANDS.md` verified current (recently updated).
- **pyproject.toml** version → `6.0.0`.
- CLAUDE.md touch-ups where cleanup invalidates statements (e.g.
  `rebuild_sonic_tower_weighted.py` gotcha).

## Phase 5 — Release gate

In order, all must pass before tagging:

1. Full `pytest` (document the `slow`/`golden` subset results separately if long).
2. `ruff check`, `mypy`.
3. `python tools/doctor.py`.
4. One real CLI generation (multi-pier artist mode) — sanity-check transition stats.
5. `npm run build` in `web/` (stale-dist trap from the web-gui skill).
6. Version-bump commit, tag `v6.0`, push `master` + tags (carries the 54 unpushed commits).

## Commit discipline

Each phase = one or more focused commits with conventional-commit messages; destructive
phases (1, 3) list deleted files in the commit body. No history rewriting; no force push.

## Out of scope (explicitly)

- Finishing MERT extraction/calibration/fold (continues post-release).
- Startup config validation guardrail (post-release roadmap item).
- Genre review panel implementation (spec/plan committed, untouched).
- Any write to `data/metadata.db`, MERT shards/sidecar, or audio files.
