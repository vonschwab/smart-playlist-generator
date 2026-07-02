# Per-playlist persistent logging — design

**Date:** 2026-07-02
**Status:** approved (design)
**Scope:** persist one DEBUG log file per playlist generation, default-on, with an
async >30-day cleanup. Applies to the GUI worker (primary) and the CLI.

## Problem

The GUI worker (`src/playlist_gui/worker.py`) routes logs only through
`WorkerLogHandler` → NDJSON events to the browser; **nothing is written to disk**.
To debug a generation you must manually redirect stdout, and you only catch a
truncated tail. We want every generation to leave a complete, greppable log on
disk — full DEBUG detail (gate tallies, per-segment pools, repair refusals) that
the INFO-level GUI never shows — cleaned up after 30 days.

## Design

### 1. `logging_utils.py` helpers (new)

- `playlist_log_dir() -> Path` — ROOT-anchored `logs/playlists/` (mirrors the
  `parents[N]` path idiom used elsewhere).
- `make_playlist_log_path(artist, request_id, *, dir=None) -> Path` — returns
  `<dir>/<YYYY-MM-DD_HHMMSS>_<safe_artist>_<shortid>.log`, e.g.
  `2026-07-02_124530_Porches_a1b2c3.log`. `safe_artist` = artist with filesystem-
  unsafe chars replaced and length-capped; `shortid` = `request_id[:6]` (fallback
  to a monotonic counter if no request_id). Timestamp sorts + reads naturally;
  artist makes files findable; shortid guarantees uniqueness.
- `@contextmanager playlist_log_file(artist, request_id, *, enabled=True, dir=None,
  level=logging.DEBUG) -> Path` — when `enabled`: mkdir; build a `FileHandler`
  (`level`, the existing `_FILE_FMT_WITH_RUN_ID` format + `RunIdFilter`); tag it
  with a **distinct** tag (NOT `_HANDLER_TAG`, so console-level controls and
  `configure_logging`'s tagged-handler cleanup never touch it); attach to root;
  yield the path; in `finally` remove + `close()` the handler. When `enabled` is
  False, yield `None` and attach nothing (byte-identical to today). Never raises
  out of setup/teardown — a logging failure must never break a generation.
- `cleanup_old_playlist_logs(dir=None, retention_days=30) -> int` — delete
  `*.log` files whose mtime is older than the cutoff; returns count; never raises.
- `cleanup_old_playlist_logs_async(dir=None, retention_days=30) -> None` — run the
  above in a `daemon` thread so it never adds to generation wall-clock.

### 2. Worker wiring (`worker.py`)

- **Root level → DEBUG (critical).** `setup_worker_logging()` currently sets
  `root.setLevel(logging.INFO)`, which would drop every DEBUG record before it
  reaches the file handler — the file would be silently empty. Change root to
  `logging.DEBUG`; keep `WorkerLogHandler` at `INFO` so **the GUI stays INFO**
  while the file captures DEBUG. (This is exactly the "a configured knob that
  can't act is a startup error" trap — verify the file actually gets DEBUG lines.)
- In `handle_generate_playlist` (line ~1013; `artist` at ~1021, `request_id` from
  `_worker_state.get_request_id()`), wrap the generation body in
  `with playlist_log_file(artist, request_id, enabled=..., dir=..., level=...) as _p:`.
- **After** the `with` block closes (log flushed + handler closed), call
  `cleanup_old_playlist_logs_async(dir, retention_days)` — async, off the critical
  path, exactly as requested.

### 3. CLI wiring (`main_app.py`)

A CLI invocation generates one playlist, so its whole run is one playlist log.
Wrap the single generation in the same `playlist_log_file(...)` context manager
(artist from args, `request_id=None` → counter shortid). An explicit `--log-file`
still works and is independent.

### 4. Config (default-on, `logging:` block)

```yaml
logging:
  playlist_logs:
    enabled: true            # rollback = false (byte-identical to today)
    dir: logs/playlists
    retention_days: 30
    level: DEBUG
```
Add to `config.example.yaml` and `config.yaml`. Worker + CLI read these and pass
them through; missing block → the defaults above.

### 5. `.gitignore`

Add `logs/` (transient debug artifacts, never committed).

## Testing

- `make_playlist_log_path`: naming shape, artist sanitization, shortid, uniqueness.
- `playlist_log_file`: attaches a DEBUG FileHandler, writes DEBUG records to the
  file, detaches + closes on exit, leaves root handlers otherwise unchanged;
  `enabled=False` attaches nothing; teardown never raises on a bad dir.
- `cleanup_old_playlist_logs`: deletes files with mtime > retention_days old,
  keeps recent ones, tolerates a missing dir, returns the delete count. (Use
  `os.utime` to age fixtures — no real sleeping.)
- Worker: `setup_worker_logging()` leaves root at DEBUG and the NDJSON handler at
  INFO (so DEBUG reaches a file handler but not the GUI).

## Out of scope

Per-file size rotation, log compression, structured/JSON playlist logs, shipping
logs anywhere. Retention is a simple age sweep.
