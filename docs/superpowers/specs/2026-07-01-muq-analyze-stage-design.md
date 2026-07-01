# SP-A: Merge MuQ into the Analyze Library flow — Design

**Date:** 2026-07-01
**Status:** Design approved (brainstorming). Next: implementation plan.
**Depends on:** the MuQ auto-fold fix (`54d682c`, already on master) + `scripts/fold_muq_into_artifact.py`.
**Gates:** SP-B (remove MERT + Beat3Tower) — deferred to `docs/CLEANUP_LIST.md`, gated on this landing.

## Problem

MuQ is the live production sonic space (`artifacts.sonic_variant_override: muq`), but it is **not
self-sufficient in the analyze pipeline**. `X_sonic_muq` was produced by a one-off research scan
(`scripts/research/embed_muq_full.py`, not on master); the pipeline's only sonic-embedding *stage*
is `mert`. So a fresh `analyze` / artifacts rebuild cannot (re)produce MuQ — it would run MERT and,
thanks to the now-fixed auto-fold, loudly error that `muq_sidecar.npz` is missing. This blocks
re-analysis and blocks removing MERT (SP-B). Goal: a first-class `muq` extraction stage so a rebuild
under `variant=muq` produces a fresh `X_sonic_muq` end-to-end.

## Design (Approach 1 — variant-aware extraction)

### Components

**`src/analyze/muq_runner.py`** (new) — the extraction, productionized from `embed_muq_full.py`,
one clear responsibility ("refresh the MuQ sidecar for tracks lacking a vector"):
- `pending_muq(sidecar_path, db_paths) -> (pending_ids, total)` — DB `file_path` tracks whose id is
  not already in `muq_sidecar.npz`.
- `run_muq_extraction(pending, db_paths, sidecar_path, *, device, torch_threads, limit, backup=True)`
  — load `MuQ-MuLan-large` once; for each pending track embed the middle-10s window
  (`librosa.load(path, sr=24000, mono=True, offset=max(0, mid-5), duration=10)`); **back up the
  existing sidecar (timestamped) before the first write**; append incrementally + resumably so an
  interrupted run loses nothing. Mirrors the resumable design already in `embed_muq_full.py` and the
  incremental shape of `energy_runner` / the MERT `run_extraction`.

**`stage_muq(ctx)`** (new, in `analyze_library.py`) — mirrors `stage_mert` (2235):
- universe = DB tracks with `file_path` (`load_paths(ctx["db_path"])`, reused from
  `extract_mert_sidecar`); pending = universe − ids already in `muq_sidecar.npz`.
- Returns `{"skipped": True, "pending": 0}` immediately when pending == 0 and `--force` is off, so the
  model is never loaded needlessly.
- Reads an `analyze.muq` config block (`device`, `torch_threads`) mirroring `analyze.mert`.
- Calls `run_muq_extraction(...)`.

**Variant gate** — the Approach-1 core. A shared helper resolves the active variant (reuse the
existing `_mert_fold_settings(config_path) -> (enabled, active_variant)`). At the top of BOTH
extraction stages:
- `stage_mert`: if `active_variant != "mert"` → return `{"skipped": True, "reason": "variant=<v> — mert extraction skipped"}` with a loud INFO log. (Data already exists; rollback = set `variant: mert`, which re-activates it.)
- `stage_muq`: symmetric — skip unless `active_variant == "muq"`.
Result: a default rebuild runs exactly the active variant's extraction; the other is available by
switching the variant. No new "explicitly-requested" plumbing.

**Registration:** add `"muq"` to `STAGE_FUNCS`, to `ANALYZE_LIBRARY_STAGE_ORDER` (immediately after
`"mert"`), to the `AnalyzeLibraryStage` Literal, to the ToolsPanel `ALL_STAGES` mirror, and a `muq`
branch in the stage-fingerprint (`_stage_fingerprint`) that hashes the pending set exactly like
`mert` so a newly-scanned track re-triggers extraction in the same pass.

**Fold:** unchanged — `stage_artifacts` already auto-folds `fold_muq` when `variant=muq` (`54d682c`).

**Dependency:** declare `muq` (and its torch requirement) as an optional extra in `pyproject.toml`
(e.g. `[project.optional-dependencies].muq`), matching how the MERT/torch ML deps are handled;
document the install in the analyze section of `CLAUDE.md` / README if MERT's isn't already.

### Data flow (rebuild, variant=muq)

`sonic` (towers built, inert under MuQ) → `mert` **no-ops** (variant≠mert) → `muq` extracts pending →
`muq_sidecar.npz` (backed up first) → `artifacts` rebuild + `fold_muq` → **fresh `X_sonic_muq`** →
`verify` passes on the stamp *for the right reason*.

### Data safety

`muq_sidecar.npz` is ~16–29h CPU — treat it like the MERT data: **timestamped backup before any
overwrite, never delete an existing sidecar.** Extraction reads audio files strictly read-only. No
DB writes beyond what `scan` already did. Music files remain read-only.

### Testing

The audio scan cannot run in CI, so test the **seams**, not the model, with a stub embedder + a tiny
fixture sidecar/DB:
- `pending_muq` correctly returns DB-minus-sidecar and `(0, total)` when all present.
- The variant gate: `stage_mert` no-ops under `variant=muq`, `stage_muq` no-ops under `variant=mert`,
  each runs under its own variant.
- Sidecar append is additive + resumable, and a backup file is written before overwrite.
- `stage_muq` returns `skipped` (model never imported) when pending == 0 and not `--force`.
Validation (not CI): a real **incremental** run on a handful of newly-scanned tracks — confirm the
sidecar gains exactly those ids, `fold_muq` refreshes `X_sonic_muq`, `verify` passes, and a
generation runs on the fresh vectors.

## Scope guard

SP-A is **additive + the mert gate only**. It does NOT touch the `sonic`/tower stage, does NOT remove
MERT/tower code or artifact bakes, and does NOT delete any data. All removal is **SP-B** (deferred,
gated on SP-A proving a real MuQ rebuild works).

## Out of scope
- SP-B (remove MERT + Beat3Tower) — cleanup list.
- 3-window MuQ extraction (1-window middle-10s is the validated choice: 84% triplets vs 89% for 3× the time).
- GPU extraction path (CPU is the established analyze mode; `device` config knob leaves the door open).
