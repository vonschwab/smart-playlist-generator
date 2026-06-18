# Energy scan → Analyze Library integration

**Date:** 2026-06-17
**Status:** Approved (design)
**Author:** session w/ Dylan

## Problem & goal

We validated (5-track, then a stratified 34-track pre-registered probe) that Essentia's
**emoMusic arousal** (distribution: p10/p50/p90) and **danceability** track perceived
energy where loudness/onset_rate do not — zero bucket inversions across the probe. We have
a working, resumable, parallel extractor (`scripts/extract_energy_sidecar.py`) that runs
under WSL and writes `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz`.

**Goal:** make the energy scan a first-class **stage of the unified `analyze_library`
pipeline**, surfaced in both the CLI and the GUI Tools panel, so refreshing energy data is
the same one-action operation as scan/sonic/mert/genres.

## Scope

**In scope — produce the data through the standard tool:**
- A new `energy` stage in `scripts/analyze_library.py` (registered, ordered, fingerprinted).
- Stage-list mirroring in `src/playlist/request_models.py` and `web/src/.../ToolsPanel.tsx`.
- A `analyze.energy` config block (tunable, documented).
- Unit tests with WSL mocked; request-model stage test; fake-worker compatibility.

**Out of scope — consume the data (next sub-project):**
- `energy_loader` that reads the sidecar into aligned arrays at runtime.
- The pace-gate "energy" steering term (adjacency-smoothness first, then energy-arc).
- Folding energy into any artifact (it is a standalone **pace-axis sidecar**, never part of
  the sonic-similarity blend — see the architecture note from 2026-06-17).

## Key facts about the existing pipeline (verified)

- Stages are `def stage_X(ctx) -> Dict`, registered in `STAGE_FUNCS`, ordered by
  `STAGE_ORDER_DEFAULT`. `run_pipeline` dispatches by name with fingerprint-based skip
  (`compute_stage_fingerprint`), unit estimates (`estimate_stage_units`), and result
  summaries. `ctx` carries `args`, `conn`, `db_path`, `config_path`, `out_dir`, `config_hash`.
- **`stage_mert` is the closest precedent** — a heavy ML embedding step that aligns to
  artifact `track_ids`, builds a skip set from prior progress, and merges shards into a
  sidecar npz. BUT MERT runs **in-process** (torch is Windows-installable). Essentia is
  **WSL-only**, so `stage_energy` is the first stage that shells out to WSL.
- The stage list is mirrored in **three** places (web-gui "stale stage list" trap):
  `analyze_library.STAGE_ORDER_DEFAULT`/`STAGE_FUNCS`,
  `request_models.AnalyzeLibraryStage`/`ANALYZE_LIBRARY_STAGE_ORDER`,
  `web/src/components/ToolsPanel.tsx::ALL_STAGES`.
- The GUI sends `stages` to the worker; `handle_analyze_library` runs `run_pipeline` and the
  `AnalyzeLibraryProgressLogHandler` bridges `analyze_library` logger lines into GUI progress.
  So a new stage that logs progress in the standard format flows to the GUI automatically.
- `extract_energy_sidecar.py` already: scopes to artifact `track_ids`, reads paths from
  metadata.db (read-only), runs a spawn pool (TF-safe), append-only `checkpoint.jsonl`
  (resume), merges to `energy_sidecar.npz` aligned to the artifact, records empty/undecodable
  tracks as errors (non-fatal), backs up an existing sidecar before overwrite, writes only
  inside the new `energy/` subdir.

## Design

### Decision (ratified): default-on, hard-fail if WSL missing

`energy` is in the default stage set (checked in the GUI, present in `STAGE_ORDER_DEFAULT`).
When up-to-date it skips instantly (like MERT). If WSL/Essentia is unreachable it raises a
clear error with remediation — matching the codebase's "a configured knob that can't act is
an error, not a silent no-op" rule and the discogs/lastfm production-required pattern.

### `analyze.energy` config block

Added to `config.yaml` + `config.example.yaml`:

```yaml
analyze:
  energy:
    distro: "Ubuntu-22.04"          # WSL distro hosting the Essentia venv
    python: "/opt/ess/bin/python"   # interpreter inside that distro
    models_dir: "/opt/ess/models"   # used by the extractor; passed through for preflight
    workers: 14                     # parallel decode workers (20-core box)
```

`stage_energy` reads this block the same way `stage_mert` reads `analyze.mert` (direct YAML
read, defaults on any miss). A `--energy-workers` CLI override is added for parity with
`--workers` (sonic); config is the primary source.

### `stage_energy(ctx)` responsibilities

1. **Resolve config** (`analyze.energy`), compute the WSL repo path from `ROOT_DIR`
   (`C:\...` → `/mnt/c/...`).
2. **Skip-fast** (mirror `stage_mert`): pending = artifact `track_ids` − checkpoint-done
   (read `energy/checkpoint.jsonl` on the Windows side). If 0 and not `--force`, return
   `{skipped: True, pending: 0}` without invoking WSL. If the artifact npz is missing, skip
   with reason (a missing prerequisite, not a WSL failure — like MERT).
3. **Preflight WSL**: a fast `wsl.exe -d <distro> -- bash -c 'test -x <python> && test -f
   <models_dir>/msd-musicnn-1.pb'`. On failure → `RuntimeError` with remediation text
   (install/repair WSL + the `/opt/ess` venv). This is the hard-fail.
4. **Run** `Popen(["wsl.exe","-d",distro,"-u","root","--","bash","-c", f"cd {wsl_repo} &&
   {python} scripts/extract_energy_sidecar.py --workers {n}{' --force' if force else ''}"])`,
   stream stdout line-by-line. Re-emit the extractor's `N/M … trk/s … ETA` lines as the
   pipeline's `ProgressLogger`-style progress so the CLI shows them and the GUI handler parses
   stage progress.
5. **Cancellation**: `run_pipeline` currently calls `cancellation_check` only at stage
   boundaries and does **not** expose it to stages via `ctx`. Add a one-line, additive change
   — `ctx["cancellation_check"] = _check_cancelled` — so `stage_energy` can poll it between
   progress lines and **terminate the WSL subprocess** on cancel. Safe because the extractor
   checkpoints per track (resume continues). This is strictly additive: other stages ignore
   the new key, so behavior is unchanged for them. (Because energy is a subprocess, this gives
   it *more* responsive cancel than the in-process MERT stage, which still cancels only at its
   boundary.)
6. **Result**: parse the extractor's final merge line (`ok=… missing=… error=…`) and return
   `{skipped: False, pending, ok, missing, error, sidecar}`. Non-zero extractor exit →
   `RuntimeError` surfacing the stderr tail. Partial decode failures are reported, not fatal.

### Stage registration & ordering

- Place `energy` **after `artifacts`** (it needs artifact `track_ids` for scope/alignment),
  before `genre-embedding`/`verify`:
  `[…, "artifacts", "energy", "genre-embedding", "verify"]`.
- `STAGE_FUNCS["energy"] = stage_energy`.
- `compute_stage_fingerprint(ctx, "energy")`: key on artifact `track_ids` count +
  checkpoint-done count + `analyze.energy` config (workers/model). Re-run when new tracks
  appear or config changes.
- `estimate_stage_units(ctx, "energy")`: pending count, label
  "tracks needing energy descriptors".
- `request_models.py`: add `"energy"` to `AnalyzeLibraryStage` and
  `ANALYZE_LIBRARY_STAGE_ORDER` in the same position.
- `web/src/components/ToolsPanel.tsx`: add `"energy"` to `ALL_STAGES` (same position);
  rebuild `web/dist`; restart `serve_web.py`.

### Data flow

```
Analyze Library (CLI or GUI Tools panel)
  → run_pipeline reaches `energy` (after `artifacts`)
    → stage_energy: resolve cfg → skip-fast? → preflight WSL → Popen wsl … extract_energy_sidecar.py
      → extractor: paths from metadata.db (RO) + audio from /mnt/e
        → energy/checkpoint.jsonl (resume) → energy/energy_sidecar.npz (aligned to artifact)
  → progress streamed to analyze_library logger → CLI + GUI bridge
```

No writes to metadata.db or the irreplaceable sonic/MERT artifacts; output confined to
`data/artifacts/beat3tower_32k/energy/`.

## Error handling

| Condition | Behavior |
|-----------|----------|
| WSL distro / venv / models unreachable | `RuntimeError` w/ remediation (hard-fail) |
| Extractor non-zero exit | `RuntimeError` surfacing stderr tail |
| Artifact npz missing (energy before first build) | skip-with-reason (like MERT) |
| Empty / undecodable track | recorded as error in result; pipeline continues |
| User cancel mid-stage | poll `ctx["cancellation_check"]` between progress lines → terminate WSL subprocess; checkpoint resumes next run |

## Testing

- **Unit (`stage_energy`, WSL monkeypatched — no real WSL in CI):** preflight-failure raises;
  progress lines parse into pipeline progress; skip-fast when nothing pending; missing-artifact
  skip; result-dict shape; non-zero exit raises.
- **request_models:** `energy` present in `ANALYZE_LIBRARY_STAGE_ORDER`; `_clean_stages` keeps
  it; default run includes it.
- **fake_worker / web:** confirm `analyze_library` still drives end-to-end with `energy` in the
  list (the worker bridges stages generically; the fake worker simulates the command, not per
  stage). Rebuild `web/dist`; the checkbox appears.
- **Manual real-WSL smoke:** `analyze_library --stages energy --limit 30` against the live
  `/opt/ess` venv (already exercised standalone).

## Risks / assumptions

- Assumes the WSL `/opt/ess` venv + models persist (set up 2026-06-17). Preflight catches
  drift loudly.
- Assumes `wsl.exe` is on PATH for the Windows process running the pipeline (worker + CLI).
- The first GUI run with `energy` checked does the **full** scan (no `--limit` from the GUI,
  same as MERT) — expected; resumable; ~2–4 h on 14 workers.
- `extract_energy_sidecar.py` stays the single source of truth for the scan; `stage_energy`
  is a thin invoker, not a reimplementation.
