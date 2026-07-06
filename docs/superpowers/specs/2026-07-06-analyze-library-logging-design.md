# Analyze Library — progress logging & per-run rotated logs

**Date:** 2026-07-06
**Status:** Design approved, pending spec review
**Motivation:** An Analyze Library run appeared to hang during MuQ scanning, and there was
no way to tell "hung" from "working slowly" — the stage produced no output for the entire
job. This spec closes the visibility gap and gives Analyze runs the same per-run, rotated
log files that playlist generation already has.

---

## 1. Problem

### 1a. MuQ is the one stage that goes dark

The pipeline orchestrator brackets every stage with a *start* line and an *end* line
(`scripts/analyze_library.py:2675` and `:2712`). What happens between is the stage's
responsibility.

For MuQ, "between" is the whole job and it emits nothing:

- `stage_muq` (`scripts/analyze_library.py:2244-2297`) logs exactly two lines:
  `"N track(s) pending … loading MuQ-MuLan …"` (`:2289`) and, at the very end,
  `"embedded ok=N failed=M"` (`:2295`).
- `build_muq_embedder` (`src/analyze/muq_runner.py:109-126`) loads the model — silent,
  can take a minute+ on a cold cache.
- `run_muq_extraction` (`src/analyze/muq_runner.py:129-160`) loops over **every** pending
  track, embedding each (librosa load + torch forward, ~1–5 s/track on CPU), and logs
  **nothing** per-track or per-batch. It even checkpoints every `SAVE_EVERY=500` tracks
  silently.

For a few thousand pending tracks that is tens of minutes to hours of zero output between
two log lines. "Model still loading", "loop churning fine", and "genuinely hung" all look
identical: a frozen log after `loading MuQ-MuLan …`.

### 1b. There is already a heartbeat contract — MuQ just doesn't use it

`ProgressLogger` (`src/logging_utils.py:341-429`) is the de-facto contract:

```
ProgressLogger(logger, total, label, unit="tracks",
               interval_s=15.0, every_n=500, verbose_each=False)
  .update(n=1, detail=<path>)   # per item
  .finish(detail=...)           # once at end
```

It emits an INFO summary every `interval_s` seconds **or** every `every_n` items, with
percent, rate (`unit/s`), and ETA baked in. It is already wired into **scan**
(`:929`), **sonic** (`:1950`), **mbid** (`:833`), **discogs** (`:1276`), **lastfm**
(`:1443`), and **enrich** (`:1705`). **energy** streams its own per-line heartbeat
(`src/analyze/energy_runner.py:149`). The two remaining heavy stages — **genre-sim** and
**artifacts** — are vectorized/graph builders that already milestone-log, not silent
per-item loops.

**MuQ is the only per-item stage that never adopted `ProgressLogger`.**

### 1c. The GUI already surfaces a live log stream

The worker installs a `WorkerLogHandler` on the **root logger at INFO**
(`src/playlist_gui/worker.py:388-416`) that forwards every log record to the GUI as an
NDJSON event; the front-end renders them in `web/src/components/LogPanel.tsx`. So any INFO
heartbeat line MuQ emits appears live in the GUI without touching the progress-bar bridge
(`parse_analyze_library_stage_progress`), which is intentionally left unchanged.

### 1d. Analyze logs are a single ever-growing file, not per-run

`run_pipeline` configures logging to one fixed path, `logs/analyze_library.log`
(`scripts/analyze_library.py:2552`), via a plain `logging.FileHandler` in append mode
(`src/logging_utils.py:158`). There is no per-run separation and no rotation; concurrent
sessions interleave into the same file. Playlist generation, by contrast, gets a per-run
DEBUG file plus age-based cleanup (`playlist_log_file` / `cleanup_old_playlist_logs`,
`src/logging_utils.py:551-682`). Analyze should mirror that.

---

## 2. Design

### Part A — MuQ adopts the `ProgressLogger` contract (the core fix)

**`src/analyze/muq_runner.py`**
- `run_muq_extraction(..., progress: Optional["ProgressLogger"] = None)`.
- After each item (success **or** recorded failure) call `progress.update(1, detail=path)`.
- After the loop, before returning, call `progress.finish()`.
- `progress=None` is a no-op → existing unit tests remain byte-identical. To avoid a hard
  import dependency in this clean module, type the parameter structurally (any object with
  `.update`/`.finish`) or import `ProgressLogger` lazily; the runner must not require the
  logging module when `progress is None`.

**`scripts/analyze_library.py` — `stage_muq`**
- Build the reporter exactly like the sibling stages:
  `ProgressLogger(logger, total=len(pending), label="muq", unit="tracks",
  interval_s=getattr(args,"progress_interval",15.0),
  every_n=getattr(args,"progress_every",500),
  verbose_each=bool(getattr(args,"verbose",False)))` — only when
  `getattr(args,"progress",True)` (matches scan/sonic), else `None`.
- Bracket the silent model load: keep the existing `"loading MuQ-MuLan …"` line, and after
  `build_muq_embedder` returns, log `"stage_muq: MuQ-MuLan loaded in %.1fs"`.
- Pass the reporter into `run_muq_extraction`.

**Result:** a live INFO line every ~15 s, e.g.
`muq: 340/4,210 (8.1%) | 0.5 tracks/s | ETA 2h 4m`, visible in the GUI LogPanel and in the
per-run file (Part C). No changes to the progress-bar bridge.

### Part B — "No stage goes dark" confirmation pass

Verify **genre-sim**, **artifacts**, and **genre-embedding** each emit a
`"starting … (N items/tracks)"` line before any multi-second op, so no stage is ever fully
silent. Current reading indicates they already milestone-log; add a single line only where
one is genuinely missing. This part is confirmation, not new machinery.

### Part C — Per-run saved + rotated analyze logs (mirror playlists)

**Implementation note (refined during planning):** analyze does not need the playlist
contextmanager. Unlike generation — which runs many times inside one long-lived worker that
configured logging once at startup, hence a per-request attach/detach handler — `run_pipeline`
calls `configure_logging(...)` *fresh every invocation*, and that call already installs a DEBUG
`FileHandler` tagged `_HANDLER_TAG` which is torn down and replaced on the next call
(`src/logging_utils.py:129-168`). So its file handler is **already per-run in lifecycle**; only
its *path* is fixed. The clean fix is therefore to compute a per-run path and hand it to
`configure_logging`, plus add rotation — no parallel handler mechanism. (Reconfiguring is safe
for the GUI: `configure_logging` only removes handlers tagged `_HANDLER_TAG`, and the worker's
`WorkerLogHandler` is untagged, so GUI log forwarding survives — `worker.py:413-416`.)

**`src/logging_utils.py`** — add:
- `analyze_log_dir() -> Path` → `<repo_root>/logs/analyze` (ROOT-anchored, cwd-independent).
- `make_analyze_log_path(run_id, *, dir=None) -> Path` →
  `<dir>/<YYYY-MM-DD_HHMMSS>_<run_id[:6]>.log`.
- `cleanup_old_analyze_logs(dir=None, retention_days=30)` + `cleanup_old_analyze_logs_async`
  — delete `*.log` older than `retention_days`; never raise. Extract the shared body of this
  and `cleanup_old_playlist_logs` into a private `_cleanup_logs_older_than(base_dir,
  retention_days)` (DRY) with identical behavior, so the playlist cleanup tests stay green.

**Config** — new block, resolved by a new `_analyze_log_settings(config)` helper in
`scripts/analyze_library.py` mirroring `_playlist_log_settings` (`main_app.py:522-532`),
returning `(enabled, dir, retention_days, level_name)`; defaults shown:
```yaml
logging:
  analyze_logs:
    enabled: true
    dir: logs/analyze
    retention_days: 30
    level: DEBUG
```
Document these in `config.example.yaml` alongside `playlist_logs`.

**Wiring — `scripts/analyze_library.py::run_pipeline`:**
- Load `cfg = Config(args.config)` just *before* `configure_logging` (a small, safe reorder;
  nothing between the two currently uses `cfg`), resolve `analyze_logs` settings, and:
  - if `args.log_file` is explicitly set, honor it (unchanged — existing tests pass `--log-file`);
  - else if `analyze_logs.enabled`, set `log_file = make_analyze_log_path(run_id,
    dir=<configured dir>)`; else `log_file = None`.
- Pass the resolved `log_file` to `configure_logging` and pass `file_level=analyze_logs.level`.
- Kick `cleanup_old_analyze_logs_async(dir=<configured dir>, retention_days=<configured>)` once.
- This covers **both** entry points, because the GUI worker calls `run_pipeline` directly
  (`src/playlist_gui/worker.py:1907`).
- Net effect: the fixed ever-growing `logs/analyze_library.log` is gone; each run writes a
  distinct `logs/analyze/<ts>_<run_id6>.log`; console output is unchanged.

---

## 3. Testing

**Unit (`tests/unit/`):**
- `run_muq_extraction` with a fake reporter records exactly `len(items)` `.update()` calls
  and one `.finish()`; failures still counted in `.update()`; `progress=None` path unchanged.
- `make_analyze_log_path` produces the documented shape; `cleanup_old_analyze_logs` deletes
  files older than `retention_days` and keeps recent ones (mirror the existing playlist-log
  tests).
- `analyze_log_file` attaches a handler while active and removes it on exit; disabled →
  no handler added.

**Live (per CLAUDE.md "exercise the real path"):**
- Run Analyze Library through the GUI. Confirm (a) a MuQ heartbeat line appears in LogPanel
  roughly every 15 s during the MuQ stage, (b) a `MuQ-MuLan loaded in …s` line appears, and
  (c) a per-run file materializes under `logs/analyze/`. Restart `serve_web.py` first so the
  worker picks up the change.

---

## 4. Non-goals

- No change to the progress-bar bridge / `parse_analyze_library_stage_progress`.
- No new heartbeat machinery for the vectorized builders beyond a single "starting" line.
- No writes to `metadata.db`; no new third-party dependency.
- Not touching the archived MERT / MuQ sidecar data.

---

## 5. Touched files

| File | Change |
|------|--------|
| `src/analyze/muq_runner.py` | `run_muq_extraction` accepts optional `progress`; `.update()`/`.finish()` |
| `scripts/analyze_library.py` | `stage_muq` builds+passes reporter, logs model-load time; `_analyze_log_settings` helper; `run_pipeline` computes per-run `log_file` + kicks cleanup, drops fixed log path; Part B "starting" lines if missing |
| `src/logging_utils.py` | `analyze_log_dir` / `make_analyze_log_path` / `cleanup_old_analyze_logs(_async)` + shared `_cleanup_logs_older_than` |
| `config.example.yaml` | document `logging.analyze_logs.*` |
| `tests/unit/` | reporter wiring, path shape, cleanup, contextmanager |
