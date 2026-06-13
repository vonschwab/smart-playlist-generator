# v6.0 Repo Cleanup & Release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the in-flight work, strip secrets and junk, delete the last deprecated engine, sweep dead config, rewrite the docs for what actually shipped, and tag a clean v6.0.

**Architecture:** Six ordered phases. Each phase is independently committable and leaves the suite green. Phase 0 lands uncommitted feature work as coherent commits; Phases 1–3 remove cruft; Phase 4 rewrites docs; Phase 5 cuts the release. Destructive steps verify against the full test suite plus one real generation before proceeding.

**Tech Stack:** Python 3.11 / pytest / ruff / mypy; React+TS+Vite (`web/`); SQLite; git on Windows + PowerShell.

---

## Operating rules (read before every task)

- **Concurrent sessions touch `master`.** Stage explicit paths only — never `git add -A`/`-u`. Re-run `git status` immediately before each commit; if an unexpected file is modified, leave it out and leave it alone (it's another session's work).
- **Never commit `config.yaml`** — it holds live secrets and is being untracked in Phase 1. Stage `config.example.yaml` when config changes are intended.
- **Bound pytest with the tool timeout; never pipe through `tail`/`head`.** To capture output, redirect to a file (`> x.log 2>&1`) and Read it. The full non-slow suite runs ~5 min.
- **Baseline (captured 2026-06-13):** `pytest -m "not slow"` with deselects cleared = **1 failed, 1725 passed, 1 skipped**. The single failure is `tests/unit/test_calibrate_mert_transform.py::test_output_npz_has_all_transform_keys` (Task 0.1 fixes it). There is **no perma-fail deselect list** in `pyproject.toml`; the memory's "12 perma-fails / 13 deselect" is stale.
- **Irreplaceable, never touch:** `data/metadata.db`, `data/artifacts/**` (MERT shards/sidecar), audio files. No `metadata.db` writes anywhere in this plan.

---

## Phase 0 — Land in-flight work

The working tree carries several already-implemented, already-tested features intermixed. They commit as coherent units. The genre work (review Completed-view, bandcamp fusion fix, authority display) shares `storage.py` and `worker.py`, so it lands as **one** genre-subsystem commit rather than smearing those files across three — they were developed together (2026-06-12) and the commit message enumerates the three threads.

**Snapshot of uncommitted files (re-verify with `git status` — this is a moving tree):**
MERT → `scripts/analyze_library.py`, `scripts/extract_mert_sidecar.py`, `scripts/calibrate_mert_transform.py`, `scripts/build_beat3tower_artifacts.py`, `src/playlist/request_models.py`, `src/playlist/analyze_library_results.py`, `config.example.yaml`, MERT/analyze/tower-knob tests.
Pace → `src/playlist/pier_bridge_builder.py`, `src/playlist_generator.py`, `tests/unit/test_beam_pace_gate.py`, `test_builder_pace_gate_wiring.py`, `test_candidate_pool_pace_floor.py`, `test_pace_mode_presets.py`, `tests/unit/goldens/pipeline/*.json`.
Genre subsystem → `src/ai_genre_enrichment/storage.py`, `hybrid_evidence.py`, `layered_assignment.py`, `src/genre/authority.py`, `src/playlist_gui/worker.py`, `src/playlist_web/app.py`, `worker_bridge.py`, `ws.py`, `web/src/components/GenreReviewPanel.tsx`, `web/src/lib/api.ts`, `web/src/lib/types.ts`, `web/src/components/GenerateControls.tsx`, `web/src/components/ToolsPanel.tsx`, `tests/fixtures/fake_worker.py`, the `test_web_*`/`test_review_*`/`test_worker_*`/`test_playlist_gui_genre_resolver`/`test_ai_genre_hybrid_*`/`test_layered_*`/`test_user_overrides_*`/`test_analyze_graph_stages` tests.
Skills → `.claude/skills/genre-data-authority/`, `.claude/skills/web-gui/`, `.claude/skills/playlist-testing/SKILL.md`.

### Task 0.1: Fix the failing MERT calibrate test

**Files:**
- Modify: `scripts/calibrate_mert_transform.py` and/or `tests/unit/test_calibrate_mert_transform.py`

- [ ] **Step 1: Reproduce the failure**

Run: `python -m pytest tests/unit/test_calibrate_mert_transform.py::test_output_npz_has_all_transform_keys -v -o addopts="" -p no:cacheprovider`
Expected: FAIL with `TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N` originating in `umap` spectral init.

- [ ] **Step 2: Diagnose with the systematic-debugging skill**

The cause is UMAP's spectral embedding requiring `k < N`; the test fixture has too few rows for the UMAP path exercised by the transform that produces the `umap_*` keys. Read the test and the transform code. The fix is one of (in preference order): (a) raise the fixture sample count above UMAP's neighbor `k` so the real code path runs; (b) if UMAP is an optional transform, have the test assert the non-UMAP transform keys and guard the UMAP key behind a `pytest.importorskip`/size check; (c) make `calibrate_mert_transform.py` fall back to a dense eig path when `k >= N` (only if that reflects a real bug, not just a tiny-fixture artifact). Prefer (a) — a faithful fixture — unless reading shows UMAP is genuinely optional.

- [ ] **Step 3: Apply the chosen fix**

Implement the minimal change. Do not weaken the assertion to dodge the error; if (b), the UMAP key must still be asserted whenever the path runs.

- [ ] **Step 4: Verify the test passes**

Run: `python -m pytest tests/unit/test_calibrate_mert_transform.py -v -o addopts="" -p no:cacheprovider`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Do not commit yet** — this rides with the MERT commit in Task 0.2.

### Task 0.2: Commit the MERT analyze-stage group

**Files:** the MERT snapshot list above + the Task 0.1 fix.

- [ ] **Step 1: Confirm MERT tests green**

Run: `python -m pytest tests/unit/test_analyze_mert_stage.py tests/unit/test_mert_extraction_cancel.py tests/unit/test_worker_cancel_concurrency.py tests/unit/test_calibrate_mert_transform.py tests/unit/test_tower_knob_guard.py tests/unit/test_analyze_library_results.py tests/unit/test_analyze_graph_stages.py -v -o addopts="" -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 2: Inspect each diff is MERT-scoped**

Run: `git diff scripts/analyze_library.py scripts/extract_mert_sidecar.py scripts/calibrate_mert_transform.py scripts/build_beat3tower_artifacts.py src/playlist/request_models.py src/playlist/analyze_library_results.py`
Confirm the changes are the `mert` stage, extraction, calibration, and build wiring. `config.example.yaml` should show only the `analyze.mert` block.

- [ ] **Step 3: Stage explicit paths (NOT config.yaml)**

```powershell
git add scripts/analyze_library.py scripts/extract_mert_sidecar.py scripts/calibrate_mert_transform.py scripts/build_beat3tower_artifacts.py src/playlist/request_models.py src/playlist/analyze_library_results.py config.example.yaml tests/unit/test_analyze_mert_stage.py tests/unit/test_mert_extraction_cancel.py tests/unit/test_worker_cancel_concurrency.py tests/unit/test_calibrate_mert_transform.py tests/unit/test_tower_knob_guard.py tests/unit/test_analyze_library_results.py tests/unit/test_analyze_graph_stages.py
```
(Add `tests/integration/test_web_tools_api.py` and `tests/fixtures/fake_worker.py` here only if their diff is the `mert` analyze stage; otherwise they go with the genre commit. Inspect first.)

- [ ] **Step 4: Re-check status, then commit**

Run: `git status --short` (verify only intended paths staged), then:
```powershell
git commit -m "feat(mert): analyze-library mert stage + calibration/build integration"
```

### Task 0.3: Commit the pace-gate follow-up group

**Files:** pace snapshot list above.

- [ ] **Step 1: Confirm pace tests + goldens green**

Run: `python -m pytest tests/unit/test_beam_pace_gate.py tests/unit/test_builder_pace_gate_wiring.py tests/unit/test_candidate_pool_pace_floor.py tests/unit/test_pace_mode_presets.py -v -o addopts="" -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 2: Verify the goldens diff is the pace retune**

Run: `git diff --stat tests/unit/goldens/pipeline/` — expect the four JSON goldens. Spot-check one diff is pace/edge-audit related (e.g. added `bpm_log_dist`), not an unrelated regression.

- [ ] **Step 3: Stage + commit**

```powershell
git add src/playlist/pier_bridge_builder.py src/playlist_generator.py tests/unit/test_beam_pace_gate.py tests/unit/test_builder_pace_gate_wiring.py tests/unit/test_candidate_pool_pace_floor.py tests/unit/test_pace_mode_presets.py tests/unit/goldens/pipeline/discover_with_dj_bridging.json tests/unit/goldens/pipeline/dynamic_default.json tests/unit/goldens/pipeline/narrow_progress_arc_dry_run.json tests/unit/goldens/pipeline/narrow_with_pier_bridge_overrides.json
git status --short
git commit -m "feat(pace): onset/BPM band edge-audit fields + regenerated goldens"
```

### Task 0.4: Commit the genre subsystem group (review + fusion + display)

**Files:** genre-subsystem snapshot list above. One commit: `storage.py` and `worker.py` each span two of the three threads, so splitting would smear them.

- [ ] **Step 1: Confirm genre/web tests green**

Run: `python -m pytest tests/unit/test_review_queue_storage.py tests/unit/test_worker_review_queue.py tests/unit/test_playlist_gui_genre_resolver.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_user_overrides_storage.py tests/unit/test_layered_artifact_builder.py tests/unit/test_layered_genre_cli.py tests/unit/test_layered_genre_taxonomy.py tests/unit/test_web_worker_bridge.py tests/unit/test_web_ws.py tests/unit/test_worker_protocol.py tests/integration/test_web_review_api.py tests/integration/test_web_track_genres_api.py -o addopts="" -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 2: Frontend builds (stale-dist trap)**

Run: `npm --prefix web run build`
Expected: build succeeds (TS compiles). If `tsc` errors, fix before committing — a red build ships broken UI.

- [ ] **Step 3: Stage backend + frontend + tests**

```powershell
git add src/ai_genre_enrichment/storage.py src/ai_genre_enrichment/hybrid_evidence.py src/ai_genre_enrichment/layered_assignment.py src/genre/authority.py src/playlist_gui/worker.py src/playlist_web/app.py src/playlist_web/worker_bridge.py src/playlist_web/ws.py web/src/components/GenreReviewPanel.tsx web/src/components/GenerateControls.tsx web/src/components/ToolsPanel.tsx web/src/lib/api.ts web/src/lib/types.ts tests/fixtures/fake_worker.py tests/unit/test_review_queue_storage.py tests/unit/test_worker_review_queue.py tests/unit/test_playlist_gui_genre_resolver.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_user_overrides_storage.py tests/unit/test_layered_artifact_builder.py tests/unit/test_layered_genre_cli.py tests/unit/test_layered_genre_taxonomy.py tests/unit/test_web_worker_bridge.py tests/unit/test_web_ws.py tests/unit/test_worker_protocol.py tests/integration/test_web_review_api.py tests/integration/test_web_track_genres_api.py tests/integration/test_web_tools_api.py
```

- [ ] **Step 4: Re-check status, then commit**

```powershell
git status --short
git commit -m @'
feat(genre): review Completed view, readonly queue, fusion + display hardening

- review panel: get_completed_review_page + decided counts; readonly queue
  reads on the worker reader thread (2026-06-12 timeout incident)
- enrichment fusion: bandcamp label-storefront split (bandcamp_domain_artist_counts),
  pure compute_layered_assignment_rows for dry-run preview
- display: authority.display_genre_names_for_track for graph-canonical chips
'@
```

### Task 0.5: Commit the developer skills

- [ ] **Step 1: Stage + commit**

```powershell
git add .claude/skills/genre-data-authority/ .claude/skills/web-gui/ .claude/skills/playlist-testing/SKILL.md
git status --short
git commit -m "docs(skills): genre-data-authority + web-gui skills; playlist-testing update"
```

### Task 0.6: Full-suite verification gate

- [ ] **Step 1: Run the full non-slow suite**

Run: `python -m pytest -m "not slow" -o addopts="-ra --strict-markers --strict-config" -p no:cacheprovider > phase0_verify.log 2>&1` then Read the tail.
Expected: **0 failed** (1726 passed / 1 skipped, give or take concurrent-session drift). If anything fails, stop and triage before Phase 1.

- [ ] **Step 2: Delete the scratch log**

Run: `Remove-Item phase0_verify.log, test_baseline.log -ErrorAction SilentlyContinue`

---

## Phase 1 — Secrets, junk purge, leak fixes

### Task 1.1: Untrack config.yaml + confirm example is secret-free

**Files:** Modify `.gitignore` (already lists `config.yaml`; no change needed unless absent); untrack `config.yaml`.

- [ ] **Step 1: SURFACE TO USER — rotate keys (manual, gating)**

`config.yaml` (tracked, in history) contains four live secrets: Discogs token, Last.fm `api_key`, OpenAI `sk-proj-…`, and a line-412 token. Per the spec, the user rotates all four externally. **Print a clear note that this must happen before Phase 5 push.** Do not block Phase 1–4 on it, but record it as an open gate.

- [ ] **Step 2: Verify config.example.yaml has only placeholders**

Run: `Select-String -Path config.example.yaml -Pattern 'sk-proj|YOnVD|[0-9a-f]{32}'`
Expected: no matches. If any real value leaked into the example, replace with `''`.

- [ ] **Step 3: Untrack config.yaml (keep the local file)**

```powershell
git rm --cached config.yaml
git status --short config.yaml   # shows 'D' staged + '??' untracked working copy
```

- [ ] **Step 4: Commit**

```powershell
git commit -m "chore(secrets): untrack config.yaml (user-local, holds live keys)"
```

### Task 1.2: Delete junk files from disk

- [ ] **Step 1: Delete root path-bug artifacts + backups + web test-results**

```powershell
Remove-Item "C*tmpsp3a_taxonomy_handoff*.yaml" -ErrorAction SilentlyContinue
Remove-Item data/vocab.bak.yaml, data/genre_vocabulary.yaml.polluted_20260604.bak -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force web/test-results -ErrorAction SilentlyContinue
```

- [ ] **Step 2: Delete generated test DBs (and stray -shm/-wal)**

```powershell
Remove-Item data/ai_genre_enriched_*_test.db, data/ai_genre_refinement_*test.db, data/ai_genre_refinement_*test, data/model_prior_test.db, data/ai_genre_enrichment_live_test.db -ErrorAction SilentlyContinue
Remove-Item data/*.db.bak_*-shm, data/*.db.bak_*-wal -ErrorAction SilentlyContinue
```

- [ ] **Step 3: Verify the keep-list survived**

Run: `Get-ChildItem data/ai_genre_enrichment.db, data/metadata.db -ErrorAction SilentlyContinue | Select-Object Name`
Expected: both present. (These and `data/artifacts/**`, `data/metadata.db.bak.*`, `data/mert_cal_*.txt` must NOT have been deleted — they're gitignored and irreplaceable.)

### Task 1.3: Stop the test-DB leak at the root

**Files:** Modify `tests/unit/test_ai_genre_enrichment.py` (and any other test writing into `data/`).

- [ ] **Step 1: Find every test that writes a DB into `data/`**

Run: `Select-String -Path (Get-ChildItem -Recurse tests -Include *.py) -Pattern "data/ai_genre_(enriched|refinement)|data\\\\.*_test\.db|model_prior_test" | Select-Object Filename,LineNumber,Line`
Record each offending path/line.

- [ ] **Step 2: Redirect each to `tmp_path`**

For each hit, change the hard-coded `data/..._test.db` to a `tmp_path / "..._test.db"` fixture (pytest's `tmp_path`). Show the before/after for each in the diff. Do not leave any test constructing a DB path under `data/`.

- [ ] **Step 3: Run those tests, confirm no new `data/*.db` appears**

```powershell
$before = (Get-ChildItem data -Filter *_test.db -ErrorAction SilentlyContinue).Count
python -m pytest tests/unit/test_ai_genre_enrichment.py -o addopts="" -p no:cacheprovider
$after = (Get-ChildItem data -Filter *_test.db -ErrorAction SilentlyContinue).Count
"before=$before after=$after"
```
Expected: tests PASS and `after` is not greater than `before` (ideally 0/0).

### Task 1.4: Harden .gitignore + commit Phase 1

**Files:** Modify `.gitignore`.

- [ ] **Step 1: Add belt-and-suspenders data ignores**

Add under the existing data block in `.gitignore`:
```
data/*.db
data/*.db.bak*
data/*.bak*
data/*-shm
data/*-wal
data/mert_cal_*.txt
data/mert_calibration_track_ids.txt
web/test-results/
```

- [ ] **Step 2: Confirm nothing now-ignored is still tracked**

Run: `git ls-files data/ | Select-String '\.db$|\.bak'`
Expected: no matches (only the YAML data files remain tracked).

- [ ] **Step 3: Commit**

```powershell
git add .gitignore tests/unit/test_ai_genre_enrichment.py
git status --short
git commit -m "chore(hygiene): redirect test DBs to tmp_path; ignore generated data artifacts"
```

---

## Phase 2 — Test-suite health

The baseline is already effectively green (no deselect list, qtbot stub gone). This phase verifies that and closes any residual gap.

### Task 2.1: Confirm green with no deselects

- [ ] **Step 1: Full non-slow run, default addopts**

Run: `python -m pytest -m "not slow" -p no:cacheprovider > phase2.log 2>&1` then Read the tail.
Expected: 0 failed. Quote the real counts.

- [ ] **Step 2: If any failure remains, triage**

Use systematic-debugging. Each failure is **fixed or deleted with a stated reason** — no deselect, no skip-to-hide. If a test is obsolete (covers Phase-3-doomed code), note it and let Phase 3 delete it with its module.

- [ ] **Step 3: Run the slow/golden subset once**

Run: `python -m pytest -m "slow or golden" -p no:cacheprovider > phase2_slow.log 2>&1` then Read the tail.
Expected: green (or known-documented). Record the result; delete the logs.

---

## Phase 3 — Kill list C, dead code, config sweep

### Task 3.1: Delete the legacy pre-DS engine (kill list C)

**Files (re-verify paths exist before deleting):**
- Delete: `src/playlist/constructor.py`, `src/playlist/candidate_generator.py`, `src/playlist/ordering.py`, `src/playlist/diversity.py`, `src/playlist/history_analyzer.py`, `src/playlist/similarity_calculator.py`, `src/genre_similarity_v2.py`
- Modify: `src/playlist_generator.py` (remove imports of the above)
- Delete: their zombie tests (find in Step 2)

- [ ] **Step 1: Map the imports**

Run: `Select-String -Path (Get-ChildItem -Recurse src,scripts,main_app.py,tools -Include *.py) -Pattern 'constructor|candidate_generator|history_analyzer|similarity_calculator|genre_similarity_v2|from .*\bordering\b|from .*\bdiversity\b' | Select-Object Filename,LineNumber,Line`
Record every importer. `playlist_generator.py` is the expected one; if a non-dead module imports any of these, STOP — it's not actually dead, reassess.

- [ ] **Step 2: Find the zombie tests**

Run: `Select-String -Path (Get-ChildItem -Recurse tests -Include *.py) -Pattern 'constructor|candidate_generator|history_analyzer|similarity_calculator|genre_similarity_v2' -List | Select-Object Filename`
Record the test files to delete alongside the modules.

- [ ] **Step 3: Delete modules + tests; untangle playlist_generator.py**

Delete the module files and zombie tests. In `playlist_generator.py`, remove the now-dead imports and any code paths they fed (the legacy non-DS branch). The `genre-data-authority` skill notes deleting `similarity_calculator.py` also removes the dormant `_get_combined_genres` signature-preferring seam — confirm no live caller breaks.

- [ ] **Step 4: Static + import check**

Run: `python -c "import src.playlist_generator"` and `ruff check src/playlist_generator.py`
Expected: imports cleanly, no unused-import (F401) errors.

- [ ] **Step 5: Full suite + real generation guard**

Run the full non-slow suite (redirect to file, Read tail) → 0 failed.
Then one real generation via the playlist-testing harness (multi-pier artist mode — read the `playlist-testing` skill first; use `generate_like_gui`, never hand-built single-seed overrides). Confirm it returns a full playlist with sane transition stats.

- [ ] **Step 6: Commit**

```powershell
git add -- src/playlist_generator.py
git rm -- src/playlist/constructor.py src/playlist/candidate_generator.py src/playlist/ordering.py src/playlist/diversity.py src/playlist/history_analyzer.py src/playlist/similarity_calculator.py src/genre_similarity_v2.py <zombie test paths>
git status --short
git commit -m "refactor(engine): delete legacy pre-DS pipeline; ds is the only engine"
```

**Escape hatch:** if Step 5 destabilizes generation and the cause isn't quickly fixable, `git restore`/`git checkout` the phase and ship v6.0 without it (user-approved). Record the decision.

### Task 3.2: Re-verify and delete vocab_normalization (conditional)

- [ ] **Step 1: Check importers**

Run: `Select-String -Path (Get-ChildItem -Recurse src,scripts,tools -Include *.py) -Pattern 'vocab_normalization'`
Expected: only `src/genre/vocab_normalization.py` self-refs (no production importer). If a production module imports it, KEEP and skip this task.

- [ ] **Step 2: Delete module + its test**

```powershell
git rm src/genre/vocab_normalization.py tests/unit/test_vocab_normalization.py
```
(`src/genre/authority.py` is KEPT — it is the genre authority. Do not delete it.)

- [ ] **Step 3: Suite + commit**

Run the full non-slow suite → 0 failed, then:
```powershell
git status --short
git commit -m "refactor(genre): delete superseded vocab_normalization module + test"
```

### Task 3.3: Delete dead scripts

- [ ] **Step 1: Delete**

```powershell
git rm scripts/sweep_pier_bridge_dials.py scripts/run_dj_connector_bias_ab.py scripts/run_dj_ladder_route_ab.py scripts/run_dj_relaxation_micro_pier_demo.py scripts/run_dj_union_pooling_stress_ab.py scripts/fix_compound_genres.py scripts/rebuild_sonic_tower_weighted.py scripts/build_windows.ps1
```

- [ ] **Step 2: Confirm no importer/reference breaks**

Run: `Select-String -Path (Get-ChildItem -Recurse src,scripts,tools,tests -Include *.py) -Pattern 'sweep_pier_bridge_dials|run_dj_connector|run_dj_ladder|run_dj_relaxation|run_dj_union|fix_compound_genres|rebuild_sonic_tower_weighted'`
Expected: no matches. If any test imports a deleted script, delete that test too (note it).

- [ ] **Step 3: Commit**

```powershell
git status --short
git commit -m "chore(scripts): delete superseded A/B sweep + dead build scripts"
```

### Task 3.4: Move research scripts to scripts/research/

- [ ] **Step 1: Create dir + move**

```powershell
New-Item -ItemType Directory -Force scripts/research | Out-Null
git mv scripts/sonic_gate1_blend.py scripts/research/
git mv scripts/sonic_gate2_rhythm.py scripts/research/
git mv scripts/sonic_harmony_richer_probe.py scripts/research/
git mv scripts/sonic_harmony_weight_sweep.py scripts/research/
git mv scripts/sonic_keyinvariance_check.py scripts/research/
git mv scripts/sonic_beatsync_2dftm.py scripts/research/
git mv scripts/sonic_tower_diagnostic.py scripts/research/
git mv scripts/sonic_phase1_metrics.py scripts/research/
git mv scripts/research_genre_similarity.py scripts/research/
git mv scripts/research_sonic_hubness.py scripts/research/
git mv scripts/research_sonic_transition.py scripts/research/
git mv scripts/measure_genre_baseline.py scripts/research/
git mv scripts/run_model_prior_album_tests.py scripts/research/
```

- [ ] **Step 2: Update doc links**

In `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`, rewrite every `../scripts/sonic_*` and `../scripts/research_*` link to `../scripts/research/...`. Find them:
Run: `Select-String -Path docs/SONIC_PHASE2_HARMONY_FINDINGS.md -Pattern 'scripts/(sonic_|research_)'`
Update each matched path.

- [ ] **Step 3: Confirm nothing imports the moved scripts by old path**

Run: `Select-String -Path (Get-ChildItem -Recurse src,scripts,tools,tests -Include *.py) -Pattern 'from scripts.(sonic_|research_sonic|research_genre|measure_genre_baseline|run_model_prior)'`
Expected: no matches (these are standalone). If a match exists, update its import path.

- [ ] **Step 4: Commit**

```powershell
git add docs/SONIC_PHASE2_HARMONY_FINDINGS.md
git status --short
git commit -m "chore(scripts): move sonic/genre research probes to scripts/research/"
```

### Task 3.5: Config deprecated-settings sweep

**Files:** Modify `config.example.yaml` and the local `config.yaml` (untracked now — edit the file, it won't be committed).

- [ ] **Step 1: Delete the three audit-confirmed dead keys**

Confirmed dead (2026-06-13): `playlists.cache_expiry_days` (only `track_matcher.py`'s unrelated constructor param matches), `playlists.genre_similarity.use_artist_tags` and `playlists.similar_artists.boost` (only read by `similarity_calculator.py`, deleted in Task 3.1). Remove these three keys from `config.example.yaml` and from local `config.yaml`.

- [ ] **Step 2: Full leaf-key audit**

For every remaining leaf key in `config.example.yaml`, check for a reader:
Run per suspicious key: `Select-String -Path (Get-ChildItem -Recurse src,scripts,main_app.py,tools -Include *.py) -Pattern '<key>'`
Account for dynamic per-mode reads (`cfg.get(f"{base}_{mode}")`) — the ~37 per-mode suffixed keys ARE live; do not remove them. Pace check: confirm the rhythm-cosine `admission_floor`/`bridge_floor` are still referenced (they're `0.0` in presets but the keys may still be read as guards) — only remove a pace key if it has zero readers post-3.1.

- [ ] **Step 3: Remove any zero-reader keys from both files**

Delete confirmed-dead keys. Keep the user's live values for every surviving key in `config.yaml`.

- [ ] **Step 4: Smoke that config still loads**

Run: `python tools/doctor.py`
Expected: no missing-key / parse errors.

- [ ] **Step 5: Commit (example only — config.yaml is untracked)**

```powershell
git add config.example.yaml
git status --short
git commit -m "chore(config): remove deprecated keys (cache_expiry_days, use_artist_tags, similar_artists.boost, +sweep)"
```
Note the before/after leaf-key count in the commit body.

---

## Phase 4 — Documentation overhaul

### Task 4.1: Rewrite README.md pace-mode section

**Files:** Modify `README.md` (the "Pace Mode" subsection, currently the v5.0 cosine-floor table).

- [ ] **Step 1: Replace the pace table + prose**

The old table (admission floor 0.55 / bridge floor 0.65, etc.) is wrong. Replace the Pace Mode section body with the current mechanism. Use this content (numbers verified from `PACE_MODE_PRESETS`):

> Pace mode keeps a playlist's rhythmic feel consistent, independent of timbre/harmony. It gates on two embedding-independent **hard bands** — tempo (BPM log-distance) and rhythmic density (onset-rate log-distance) — plus a **soft** rhythm-cosine penalty that demotes (never rejects) off-rhythm bridge edges. Because the bands read DB features (`bpm_info`, `onset_rate`), pace survives the MERT sonic migration unchanged. Bands widen on segment backoff so an over-tight pace never blows the generation budget.
>
> | Mode | BPM band (adm/bridge, log₂) | Onset band (adm/bridge, log₂) | Rhythm soft penalty (thresh/strength) |
> |---|---|---|---|
> | `strict`  | 0.30 / 0.40 | 0.30 / 0.40 | 0.35 / 0.20 |
> | `narrow`  | 0.50 / 0.60 | 0.50 / 0.60 | 0.25 / 0.15 |
> | `dynamic` | 0.75 / 0.85 | 0.75 / 0.85 | 0.15 / 0.10 |
> | `off`     | ∞ / ∞ | ∞ / ∞ | 0 / 0 |
>
> The old rhythm-cosine hard floor is gone (it was near-noise and made beatless/ambient artists infeasible). `narrow` is now usable for an ambient seed while still tightening for rhythmic music.

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "docs(readme): rewrite pace-mode section for BPM+onset bands"
```

### Task 4.2: README — four axes, genre system, sonic/MERT, GUI

**Files:** Modify `README.md`.

- [ ] **Step 1: Four mode axes**

Change "Three Independent Axes" → four. Add `cohesion_mode` (drives the beam: strict/narrow/dynamic/discover) alongside genre/sonic/pace (drive pool composition). State that all four live at `playlists.<axis>` in config and the GUI exposes them as sliders. (Authoritative source: CLAUDE.md "Design principles" #19 and the cohesion-mode gotcha.)

- [ ] **Step 2: Genre system section**

Replace genre prose with the current architecture: enriched genres are the **authority** — `release_effective_genres` (metadata.db), written only by the publish stage, read via `src/genre/authority.py`. The SP3a **layered taxonomy graph** (`data/layered_genre_taxonomy.yaml`, 455 genres) is the structural source of truth. Multisource **Claude enrichment** collects + adjudicates tags (scan→genres→discogs→lastfm→enrich→publish), and the artifact bakes the authority in via `genre_source: graph`. Genre chips in the GUI are graph-canonical, ordered specific→broad. Multi-genre signatures are preserved. (Sources: `genre-data-authority` skill; specs `2026-06-11-genre-chips-granularity-design.md`, `2026-06-10-analyze-library-graph-claude-design.md`.)

- [ ] **Step 3: Sonic section**

State the current default: 162-dim `tower_weighted` (rhythm 9 + timbre 57 + 2DFTM harmony 96; weights 0.20/0.50/0.30). Add a paragraph: a learned **MERT** sonic embedding (MERT-v1-95M, `whiten_l2` post-processing) is in progress as an **experimental opt-in** (`sonic.variant: mert`); the towers remain the default and the rollback path. Link `docs/MERT_WHITEN_NEIGHBORS_20SEEDS.md` and `docs/superpowers/specs/2026-06-11-mert-sonic-embedding-design.md`. Do NOT claim MERT is the default.

- [ ] **Step 4: GUI section + Quick Start + structure**

Remove all PySide6 right-click/Help-menu text. Document the browser GUI (`python tools/serve_web.py`, port 8770) with its tabs: **Generate**, **Tools** (Analyze Library + Enrich), **Genre Review**; seed staging; graph-canonical genre chips (6 + `+N`); web track-replacement context menu. Fix Quick Start to use `scripts/analyze_library.py` (orchestrator; stages incl. `mert`) and `tools/serve_web.py`. Refresh the Project Structure tree (`src/playlist_web/`, `web/src/components/*`). Update the Version History table to add v6.0; set the top-of-file version to 6.0.

- [ ] **Step 5: Build sanity + commit**

Re-read the edited README top-to-bottom for contradictions (no "three axes", no PySide6, no stale pace floors). Then:
```powershell
git add README.md
git commit -m "docs(readme): v6.0 — four axes, genre authority/graph, MERT opt-in, browser GUI"
```

### Task 4.3: CHANGELOG (root + docs)

**Files:** Modify `CHANGELOG.md` (root summary) and `docs/CHANGELOG.md` (full).

- [ ] **Step 1: Add the v6.0 entry to docs/CHANGELOG.md**

Add a dated v6.0 section covering: browser GUI replacing PySide6; enriched-genre authority + layered taxonomy graph + Claude enrichment backend + analyze graph stages (lastfm/enrich/publish); genre review panel + graph-canonical chips; pace-gate re-architecture (BPM + onset bands, soft rhythm penalty); MERT foundations + analyze `mert` stage (experimental); dead-code cleanup (kill lists A/B/C); wiring fixes (beam widths, pace gate, web dj_bridging, `genre_source: graph`); secret untracking.

- [ ] **Step 2: Update the root CHANGELOG.md summary**

Add the one-line v6.0 bullet at the top; bump "Latest Release: Version 6.0".

- [ ] **Step 3: Commit**

```powershell
git add CHANGELOG.md docs/CHANGELOG.md
git commit -m "docs(changelog): v6.0 release notes"
```

### Task 4.4: AGENTS.md, docs index, ARCHITECTURE, TROUBLESHOOTING, GOLDEN_COMMANDS

**Files:** Modify `AGENTS.md`, `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/TROUBLESHOOTING.md`, `docs/GOLDEN_COMMANDS.md`.

- [ ] **Step 1: AGENTS.md — kill stale PySide6/[gui]**

Run: `Select-String -Path AGENTS.md -Pattern 'PySide6|pip install -e \.\[gui\]|PyQt'`
Replace the 2 hits: GUI is the browser app; install extra is `[web]` / `[web,dev]`.

- [ ] **Step 2: docs/README.md index**

Add entries for the new docs (MERT design + `MERT_WHITEN_NEIGHBORS_20SEEDS.md`, pace-retune spec, genre-review/chips specs, `genre-data-authority` rules). Remove pointers to anything deleted in Phase 3.

- [ ] **Step 3: ARCHITECTURE.md**

Bring to the web-GUI + graph-authority + pace-bands era. For sonic, document the tower default and note the MERT opt-in path per the MERT design's "architectural note" (don't rewrite Layer-2 commitment #8 — flag it as amended-when-MERT-defaults).

- [ ] **Step 4: TROUBLESHOOTING.md**

Fix stale paths (`repo_refreshed` → repo root). Add web-GUI traps (stale `web/dist` needs `npm run build`; worker restart after worker edits) cross-referencing the `web-gui` skill.

- [ ] **Step 5: GOLDEN_COMMANDS.md**

Run: `Select-String -Path docs/GOLDEN_COMMANDS.md -Pattern 'mert|review|analyze_library'`
Ensure the `analyze_library.py` stage list includes `mert` and the review/tools commands are present; add if missing.

- [ ] **Step 6: Commit**

```powershell
git add AGENTS.md docs/README.md docs/ARCHITECTURE.md docs/TROUBLESHOOTING.md docs/GOLDEN_COMMANDS.md
git commit -m "docs: refresh agents/index/architecture/troubleshooting/commands for v6.0"
```

### Task 4.5: pyproject version + CLAUDE.md cleanup-invalidated notes

**Files:** Modify `pyproject.toml`, `CLAUDE.md`.

- [ ] **Step 1: Bump version**

In `pyproject.toml`, set `version = "6.0.0"`.

- [ ] **Step 2: Drop the deleted-script gotcha from CLAUDE.md**

Remove the `rebuild_sonic_tower_weighted.py` mention (script deleted in Task 3.3) from the CLAUDE.md "project-specific gotchas" / sonic note; reword so the fold-script remains the documented path. Scan CLAUDE.md for any other statement Phase 3 invalidated.

- [ ] **Step 3: Commit**

```powershell
git add pyproject.toml CLAUDE.md
git commit -m "chore(version): 6.0.0; drop CLAUDE.md note for deleted rebuild script"
```

---

## Phase 5 — Release gate

### Task 5.1: Pre-tag verification

- [ ] **Step 1: Confirm secrets rotated (user gate)**

Do not proceed until the user confirms the four keys are rotated. Re-print the reminder from Task 1.1.

- [ ] **Step 2: Full suite**

Run: `python -m pytest -p no:cacheprovider > release_pytest.log 2>&1` then Read the tail. Expected: green (note any documented slow/golden exceptions). 

- [ ] **Step 3: Lint + types**

Run: `ruff check` then `mypy`. Expected: clean (or only the pre-existing intentional `[[tool.mypy.overrides]]` exemptions). Fix anything the cleanup introduced.

- [ ] **Step 4: Doctor**

Run: `python tools/doctor.py`. Expected: no errors.

- [ ] **Step 5: Real generation**

One multi-pier artist-mode generation via the playlist-testing harness; confirm a full playlist + sane transition stats (min/mean/p10/p90, distinct-artist count).

- [ ] **Step 6: Frontend build**

Run: `npm --prefix web run build`. Expected: success. Delete `release_pytest.log`.

### Task 5.2: Tag + push

- [ ] **Step 1: Final status check**

Run: `git status` (clean working tree except untracked local `config.yaml` + gitignored data) and `git log --oneline origin/master..HEAD | Measure-Object -Line` (record the commit count being pushed).

- [ ] **Step 2: Tag**

```powershell
git tag -a v6.0 -m "v6.0 — browser GUI, genre authority/graph, Claude enrichment, pace bands, MERT foundations"
```

- [ ] **Step 3: Push (surface to user first if remote is shared/public)**

Pushing publishes to GitHub. Confirm the user wants the push now (secrets gate cleared), then:
```powershell
git push origin master
git push origin v6.0
```

- [ ] **Step 4: Report**

Print: commits pushed, tag created, test/lint/doctor results, and the open post-release items (MERT extraction/fold + default flip, startup config-validation guardrail, genre-similarity audition harness).

---

## Self-review notes (coverage check)

- Spec Phase 0 (land in-flight) → Tasks 0.1–0.6, regrouped to match the real (entangled) tree: MERT, pace, genre-subsystem, skills.
- Spec Phase 1 (secrets/junk/leaks) → Tasks 1.1–1.4 (untrack, delete, tmp_path redirect, gitignore).
- Spec Phase 2 (test health) → Tasks 2.1; lightened because baseline is green and no deselect list exists.
- Spec Phase 3 (kill list C / dead code / config sweep) → Tasks 3.1–3.5; `authority.py` explicitly KEPT.
- Spec Phase 4 (docs) → Tasks 4.1–4.5; pace table given verbatim, four axes, genre authority/graph, MERT opt-in, browser GUI, version bump.
- Spec Phase 5 (release) → Tasks 5.1–5.2; secrets-rotated gate enforced before push.
- Out-of-scope items (MERT fold/default flip, startup guardrail, audition harness, history rewrite, metadata.db writes) are not tasked — correct.
