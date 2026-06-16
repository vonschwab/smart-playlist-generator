# v6.0 Cleanup — Remaining Tasks (fresh-session handoff)

**Date:** 2026-06-15. Picks up after the big merge. The detailed method for each task lives in
`docs/superpowers/plans/2026-06-13-repo-cleanup-v6.md` (Phases B/C/D) — this file is the
short, current "what's left" list. Read both.

## Where things stand (do NOT re-derive)

- **`master` = the unified everything** (commit `b93de68` at handoff). It carries: MERT-default
  sonic embedding, pace BPM/onset bands, enriched-genre authority + layered taxonomy graph +
  Genre Review panel, multisource Claude enrichment (incl. the new `mert` analyze stage +
  enrich-pause), web Tools + web-search/infinite-scroll, the browser GUI, and all cleanup so far.
- **Suite green:** `python -m pytest -m "not slow"` → ~**1674 passed**, 0 failed.
- **Done already:** lane merge; WIP-red settled (mert stage + enrich-pause implemented);
  `config.yaml` untracked + junk gitignored; `tools/dead_code_audit.py` added; **B2** deleted
  `vocab_normalization`; **B3a** deleted dead A/B-sweep scripts + `sonic_rebuild`; **B1** removed
  the dead legacy engine (`candidate_generator`, `diversity`, `ordering`, `genre_similarity_v2`
  + ~20 orphaned methods in `playlist_generator.py`, ~3,986 LOC); **C1** README v6.0; **C2**
  CHANGELOG v6.0; **C3-core** `pyproject` 6.0.0, AGENTS, docs index.
- **Kept (verified live, do not delete):** `history_analyzer` (`is_collaboration_of` via
  `artist_style`), `constructor.PlaylistResult` (used by `pipeline/core`), `similarity_calculator`
  (instantiated by `LocalLibraryClient`; its dead `genre_calc` dependency was severed to `None`).

## Operating rules (read before starting)

- **Work in a git worktree off `master`** (EnterWorktree). One writer per tree.
- **Integrate to master FREQUENTLY** — small commits, fast-forward `master` after each task or two.
  The whole pain this round came from letting a cleanup branch diverge for days. Don't repeat it.
- **Never stage `config.yaml`** — it holds live Last.fm + OpenAI keys and is now untracked.
  Before every commit: `git diff --cached --name-only | Select-String '^config\.yaml$'` must be empty.
- **Do NOT `git push`** — the keys are in git history. Push is human-gated on rotation + scrub (below).
- **Pytest:** bounded, never piped through `tail`/`head`/`Select-Object` (hook-blocked, hangs). Redirect
  to a file and read it. Full non-slow suite ≈ 2.5 min.
- **`data/` for generation gates:** a fresh worktree's `data/` lacks the heavy files (gitignored). To run
  a real generation: `New-Item -ItemType Junction data/artifacts -Target <main>/data/artifacts`, and
  `Copy-Item <main>/data/metadata.db data/metadata.db` (a disposable copy — never write the originals).
  Config paths stay relative; the junction+copy make them resolve.
- **Re-run `python tools/dead_code_audit.py`** after each deletion to catch newly-orphaned modules.

## Remaining tasks (suggested order)

### 1. B5 — dead-wiring re-audit (fast, do first; informs B4)
- [ ] Re-run `python tools/dead_code_audit.py` on current `master`. Investigate any new `[ORPHAN]`.
- [ ] Declared-but-unimplemented sweep: confirm every `request_models.AnalyzeLibraryStage`, worker
      `STAGE_FUNCS` key, web tool stage, and `sonic_variant` has a live implementation (the class that
      produced the mert-stage/enrich-pause WIP-red). Implement or remove each dangling declaration.

### 2. B4 — config deprecated-settings sweep
- [ ] The 3 audit-known dead keys (`cache_expiry_days`, `genre_similarity.use_artist_tags`,
      `similar_artists.boost`) are already absent from `config.example.yaml` — confirm.
- [ ] Full knob audit: every leaf key in `config.example.yaml` with **zero readers** (static +
      dynamic `cfg.get(f"{base}_{mode}")` + policy reads). Keep the ~37 live per-mode keys and the new
      pace/MERT/genre keys. Delete zero-reader keys from `config.example.yaml` (and the untracked local
      `config.yaml`). `python tools/doctor.py` must load clean. Note before/after key counts.

### 3. B3b — relocate research/audition scripts (organizational)
- [ ] `git mv` to `scripts/research/`: the `sonic_*` probes, `research_*`, `measure_genre_baseline.py`,
      `run_model_prior_album_tests.py`, and the audition harnesses (`sonic_audition_*`,
      `genre_audition_*`, `pace_audition_*`, `pace_calibration_sweep.py`). Update their test import
      paths and the links in `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`. Suite green.

### 4. C3 remainder — heavy doc refreshes
- [ ] `docs/ARCHITECTURE.md` — pre-MERT; bring to the merged era (MERT-default sonic + tower rollback,
      pace bands, genre authority/graph, browser GUI). Reconcile CLAUDE.md Layer-2 #8/#17/#18 only when
      MERT becomes permanent (it's committed-as-default but the listen-test flip gate is still open).
- [ ] `docs/TROUBLESHOOTING.md` — fix stale paths (`repo_refreshed`); add web-GUI traps (stale `web/dist`,
      worker restart) cross-referencing the `web-gui` skill.
- [ ] Verify `docs/GOLDEN_COMMANDS.md` is current (it has the enrich/`mert` stages).

### 5. A2 — delete root-junk (main checkout)
- [ ] In the main checkout: `Remove-Item 'C*tmpsp3a_taxonomy_handoff*.yaml'` (untracked Windows path-bug
      files). Also decide `CLAUDE.md`'s status — it's tracked yet in `.gitignore` (inconsistent); either
      `git rm --cached` it (treat as local) or drop it from `.gitignore`.

### 6. D — release gate + tag (LAST; tag only)
- [ ] Full `pytest`; `ruff check`; `mypy`; `python tools/doctor.py`; one real multi-pier generation
      (playlist-testing skill); `npm --prefix web run build`.
- [ ] Tag `v6.0` on `master` locally. **STOP — do not push.**

## The push gate (HUMAN-driven — a session cannot do this)

`config.yaml`'s Last.fm + OpenAI keys are in git history. Before any `git push`:
1. **Rotate** the Last.fm + OpenAI keys (external).
2. **Scrub** `config.yaml` from history (`git filter-repo` or BFG), force-push coordinated across the
   16 local branches / worktrees.
3. Then push `master` + the `v6.0` tag.
Until that's done, everything stays local.
