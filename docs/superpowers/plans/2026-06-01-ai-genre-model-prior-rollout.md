# AI Genre Model-Prior Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved AI genre stabilization and album-level model-prior design through three independently testable milestones.

**Architecture:** Stabilization lands first and restores a backward-compatible artifact boundary with `legacy` as the default. The model-prior prototype then adds a separate CLI-only, no-web classifier lane. The final milestone allows mapped provisional prior terms into isolated `hybrid_shadow` artifacts and emits comparison evidence without changing production generation.

**Tech Stack:** Python 3.11+, SQLite sidecar DB, NumPy NPZ artifacts, OpenAI Responses API, PyYAML, pytest

---

## Approved Design

Read before implementation:

- `docs/superpowers/specs/2026-06-01-ai-genre-model-prior-stabilization-design.md`
- `AGENTS.md`
- `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`

## Ordered Plans

Execute in order. Each plan ends in working, testable software and a commit checkpoint.

1. `docs/superpowers/plans/2026-06-01-ai-genre-stabilization.md`
   - Restore explicit `legacy | enriched | hybrid_shadow` artifact modes.
   - Fix Bandcamp source typing.
   - Quarantine Last.fm-only evidence for refreshed signatures.
   - Add policy versioning and API-free dry runs.
   - Reject stale dense sidecars with one shared validator.

2. `docs/superpowers/plans/2026-06-01-ai-genre-model-prior-prototype.md`
   - Add a reusable structured Responses API wrapper.
   - Add album-prior schema, validation, taxonomy mapping, storage, cache, CLI single-run, batch, and report commands.
   - Keep every model-prior term sidecar-only and excluded from artifacts.

3. `docs/superpowers/plans/2026-06-01-ai-genre-shadow-integration.md`
   - Add a dedicated `HybridShadowGenreResolver`.
   - Include only mapped `accepted_for_shadow=1` provisional terms.
   - Build isolated sparse and dense shadow artifacts.
   - Produce fixed-seed comparison reports.

## Safety Rules

- Do not write, migrate, or alter `data/metadata.db`.
- Do not run `scripts/normalize_genre_vocab.py --apply`.
- Do not write, move, rename, or delete music files.
- Treat the current worktree as user-owned and dirty. Do not revert unrelated changes.
- Back up `data/ai_genre_enrichment.db` before any manual live-sidecar validation run.
- Use temporary sidecar DBs in automated tests.

## Existing Working-Tree Overlap

The current worktree already contains uncommitted genre redesign files, including:

```text
scripts/build_genre_embedding.py
scripts/measure_genre_baseline.py
src/genre/blend.py
src/genre/llm_client.py
src/genre/llm_prior.py
src/genre/vocab_normalization.py
tests/unit/test_pmi_svd.py
tests/unit/test_vocab_normalization.py
```

Read and preserve those changes during implementation. Do not replace them wholesale.

## Baseline Verification

Run before Task 1 and record the output in the implementation thread:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_artifact_builder_enriched.py tests/unit/test_worker_enrich_artist.py tests/unit/test_genre_vocabulary.py tests/unit/test_user_overrides_storage.py tests/unit/test_review_panel_graduate.py tests/unit/test_vocab_normalization.py tests/unit/test_pmi_svd.py -q --basetemp C:\tmp\playlist-vnext-unit -o cache_dir=C:\tmp\playlist-vnext-unit-cache
```

Expected baseline:

```text
213 passed
```

Run the dense integration baseline separately:

```powershell
pytest tests/integration/test_dense_genre_integration.py -q --basetemp C:\tmp\playlist-vnext-dense -o cache_dir=C:\tmp\playlist-vnext-dense-cache
```

Expected baseline:

```text
23 passed, 3 failed
```

Known pre-existing failures:

```text
test_dense_pool_meets_minimum[beach_boys]
test_charli_xcx_dense_sim_distribution
test_dense_sim_niche_artists_higher_than_sparse
```

Do not attribute those three failures to this rollout unless their measured output changes.

## Execution Stop Points

- Stop after stabilization and review artifact-mode behavior before adding the model prior.
- Stop after the CLI prototype and review sample prior output before enabling shadow integration.
- Stop after shadow reporting and inspect comparison evidence before considering GUI controls or production adoption.
