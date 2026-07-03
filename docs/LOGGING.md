# Logging Standards

The logging policy for this codebase: what goes at each level, how loggers are named, how the
CLI controls verbosity, and how secrets stay out of log output. This doc is deliberately
**sonic/genre-independent** — it doesn't change with the embedding or genre-scoring stack. For
the system it's instrumenting, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

All of the machinery below lives in `src/logging_utils.py` — the single module every entrypoint
imports from.

---

## Levels policy

| Level | When to use | Examples |
|-------|-------------|----------|
| `ERROR` | Operation failed, requires attention | DB connection failed, API error after retries |
| `WARNING` | Recoverable issue, degraded operation | Missing optional config, skipped item, paused stage |
| `INFO` | Milestones and summaries only | Stage start/complete, final counts, run recap |
| `DEBUG` | Per-item details, diagnostic data | Per-track processing, per-candidate scoring, per-edge weights |

1. **INFO is for humans scanning logs.** A typical run should produce a manageable number of
   lines, not a stream.
2. **Never log per-item at INFO.** Per-track, per-artist, per-edge, per-album logs belong at
   DEBUG — the real code follows this: MBID/Discogs per-item failures (`scripts/analyze_library.py`)
   log at `logger.debug(...)`, only the stage totals log at `logger.info(...)`.
3. **Summaries replace streams.** Log "Processed 500/500 tracks", not each of the 500.

```python
# BAD - per-item at INFO
for track in tracks:
    logger.info(f"Processing: {track.title}")

# GOOD - detail at DEBUG, summary at INFO
for track in tracks:
    logger.debug(f"Processing: {track.title}")
logger.info(f"Processed {len(tracks)} tracks")
```

---

## Logger naming

Every module gets its own logger via `__name__`, so the hierarchy mirrors the package tree
(`src.playlist.pipeline.core`, `src.playlist.pier_bridge_builder`, `src.playlist_web.app`, ...).
This is universal in `src/` — there is no module in the codebase that instantiates a
custom-named logger instead.

```python
# CORRECT
logger = logging.getLogger(__name__)

# AVOID - breaks the hierarchy, can't be filtered by package
logger = logging.getLogger("my_custom_name")
```

Standalone scripts are the one deliberate exception — they name the logger after the script so
log lines are legible without a package prefix, e.g. `scripts/scan_library.py` uses
`logging.getLogger('scan_library')` and `scripts/analyze_library.py` re-fetches
`logging.getLogger("analyze_library")` after `configure_logging()` runs.

---

## Message format and run IDs

```
%(asctime)s | %(levelname)-5s | %(name)s | %(message)s
```

is the console default. `configure_logging()` injects a `run_id` (a `RunIdFilter` on the root
logger, defaulting to `-` when none is set) but **console output omits it by default** —
it only appears when the console level is `DEBUG` or `--show-run-id` is passed:

```
10:30:45 | INFO  | src.playlist.pipeline.core | Starting playlist generation
10:30:45 | DEBUG | src.playlist.pipeline.core | run_id=3f2a9c1e | Candidate pool: 812 tracks
```

File output always includes `run_id` **and** `funcName:lineno` — the file format is strictly
more detailed than console, never less:

```
%(asctime)s | %(levelname)-5s | %(name)s | %(funcName)s:%(lineno)d | run_id=%(run_id)s | %(message)s
```

### Message guidelines

1. Start with an action verb — "Processing...", "Found...", "Completed...".
2. Include counts — "Found 150 candidates", not "Found candidates".
3. Include timing for stages — "Completed in 1.2s".
4. No trailing punctuation.
5. Truncate long lists (`truncate_list()` in `logging_utils.py`) — "rock, pop, jazz (+5 more)".
6. For structured data, use `key=value` pairs, pipe-delimited. This is the real convention in
   `scripts/analyze_library.py`, not a hypothetical style:
   ```
   run_id=3f2a9c1e | stage=sonic | decision=ran | reason=fingerprint_changed | processed=142 | elapsed_s=38.40 | throughput=3.7/s | errors=0 | top_error_categories=-
   ```

---

## CLI controls

Every entrypoint calls `add_logging_args(parser)` from `logging_utils.py`, which adds these
flags uniformly:

| Flag | Effect |
|------|--------|
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Console level (default `INFO`) |
| `--debug` | Shortcut for `--log-level DEBUG`; wins over `--quiet` |
| `--quiet` | Shortcut for `--log-level WARNING` |
| `--log-file PATH` | Also write full-detail logs to this file |
| `--show-run-id` | Include `run_id` in console output even below DEBUG |

`resolve_log_level(args)` applies the precedence (`--debug` > `--quiet` > `--log-level`).

> **Not every flag is wired at every entrypoint.** `--verbose` exists on `main_app.py`,
> `scripts/analyze_library.py`, and `scripts/scan_library.py`, but means something different at
> each: on `main_app.py` it's a blanket `log_level = 'DEBUG'`; on the two analyze-side scripts it
> also flips `ProgressLogger` into per-item DEBUG mode (see below). The
> `--progress` / `--no-progress` / `--progress-interval` / `--progress-every` flags exist **only**
> on `scripts/analyze_library.py` and `scripts/scan_library.py` — `main_app.py` has no progress
> flags (a single generation run doesn't have a meaningful progress bar). And `main_app.py`
> parses `--show-run-id` via `add_logging_args()` but never forwards it to `configure_logging()`
> — on that entrypoint the flag is currently inert. If you need `run_id` on a CLI generation run,
> use `--log-file` (file output always includes it) or `--debug`.

### Implementation

```python
from src.logging_utils import add_logging_args, resolve_log_level, configure_logging

parser = argparse.ArgumentParser()
add_logging_args(parser)
args = parser.parse_args()

level = resolve_log_level(args)
configure_logging(level=level, log_file=args.log_file)
```

---

## Single setup rule

**Logging is configured exactly once, at the entrypoint.** Library modules only ever call
`logging.getLogger(__name__)` — never `configure_logging()`, never `logging.basicConfig()`.

```python
# main_app.py (entrypoint)
from src.logging_utils import configure_logging
configure_logging(level='INFO', log_file='playlist_generator.log')

# src/playlist/pipeline/core.py (library module)
import logging
logger = logging.getLogger(__name__)  # get a logger, never configure one
```

`configure_logging()` is itself idempotent without `force=True` — a second call is a no-op, so an
accidental double-call from a library module can't silently duplicate handlers (verified by
`tests/unit/test_logging_config_idempotent.py`). A repo-wide test
(`test_no_basicconfig_in_src_scripts`) greps every file under `src/` and `scripts/` for a literal
`basicConfig(` call and fails the suite if one exists — this is enforced, not just convention.

The browser-GUI worker (`src/playlist_gui/worker.py`, a separate subprocess) is the one place
that legitimately reconfigures the root logger itself, via its own `setup_worker_logging()` —
it *is* an entrypoint (its own process), just not one that goes through `configure_logging()`.
See "Browser GUI / worker logging" below.

### What NOT to do

```python
# BAD - module configures its own logging
import logging
logging.basicConfig(level=logging.DEBUG)

# BAD - re-configuring in a __main__ block of a library module
if __name__ == "__main__":
    logging.basicConfig(...)
```

---

## Progress + ETA

`ProgressLogger` (`logging_utils.py`) is the shared mechanism for long-running stages —
currently used by `scripts/analyze_library.py` and `scripts/scan_library.py`, not by
`main_app.py` (a single playlist generation doesn't run long enough to need one).

- Default: periodic **INFO** summaries every `interval_s` (15s) or `every_n` items (500),
  whichever comes first. Each summary includes rate and, when a total is known, percent and ETA.
- `--verbose` switches a stage into `verbose_each=True`: **DEBUG** per item. The class docstring
  says verbose mode keeps the periodic INFO summaries too, but `_should_emit()` currently
  short-circuits to `False` whenever `verbose_each` is set (`logging_utils.py`), so that branch
  is dead — in practice a verbose run gets DEBUG-per-item plus one final INFO summary from
  `finish()`, not periodic INFO summaries along the way. Worth knowing if a verbose log looks
  DEBUG-only for a long stretch — that's the current (buggy) behavior, not a hang.
- `finish()` always emits a final summary regardless of interval/count timing — this one path is
  unconditional in both modes.
- `--no-progress` disables the periodic summaries; `WARNING`/`ERROR` still emit.

Real output shapes (from `ProgressLogger._progress_msg` / `.finish()`):

```
INFO  stub_scan: 1,250/3,210 (38.9%) | 82.3 items/s | ETA 25s
INFO  stub_scan complete: 3,210 items | elapsed 1m02s | avg 51.6 items/s
DEBUG stub_scan item 125/3210: Artist - Title (song.flac)
```

---

## Analyze library logging

`scripts/analyze_library.py` (`run_pipeline`) is the most heavily instrumented entrypoint —
every stage's decision is auditable from the log alone. This is the actual, verified format
(pinned down by `tests/unit/test_analyze_library_logging.py`), not an aspirational example:

```
INFO  Analyze run start | run_id=3f2a9c1e | db=data/metadata.db | out_dir=data/artifacts/beat3tower_32k | stages=scan, sonic, artifacts, verify
INFO    config_hash=a91f... | git=d48bae9
INFO    progress=on interval=15.0s every=500 verbose_each=False
INFO    pending_estimates: sonic=142 tracks
INFO  run_id=3f2a9c1e | stage=sonic | decision=ran | reason=fingerprint_changed | pending=142
INFO  run_id=3f2a9c1e | stage=sonic | decision=ran | reason=fingerprint_changed | processed=142 | elapsed_s=38.40 | throughput=3.7/s | errors=0 | top_error_categories=-
INFO  Wrote run report to data/artifacts/beat3tower_32k/analyze_run_report.json
INFO  RUN RECAP | run_id=3f2a9c1e | config_hash=a91f... | report=...analyze_run_report.json
INFO    verify_issues=none
INFO    stage=sonic | decision=ran | reason=fingerprint_changed | pending_before=142 | processed=142 | elapsed=38.40s | rate=3.70/s | errors=0 | top_error_categories=-
INFO  Total elapsed: 41.20s
```

Key mechanics:

- **Startup line** includes `run_id`, `db`, `out_dir`, `stages`, `config_hash`, the current git
  commit, and the progress settings in effect.
- **Every stage logs a decision** before and after it runs: `decision` is one of
  `skipped` / `ran` / `forced` / `paused`; `reason` explains why (`fingerprint_same`,
  `fingerprint_changed`, `required`, `forced`, or a pause reason). A skip due to an unchanged
  fingerprint is logged explicitly — it is never a silent no-op.
- **`scan` gets an extra breakdown line** (`  scan modified breakdown: stat_changed=12, ...`);
  with `--verbose` it also logs per-reason example paths at DEBUG.
- **The run ends with a written JSON report** (`analyze_run_report.json`) plus a `RUN RECAP` line
  and a per-stage recap table at INFO, so a human can read the whole run's outcome without
  opening the JSON.
- **Cancellation is checked between stages** (`cancellation_check`), and the web worker calls
  `run_pipeline(args, console_logging=False)` to suppress stdout duplication while still writing
  the log file — the NDJSON handler (see below) is the console for that path instead.

---

## Timings and metrics: available helpers vs. what's actually used

`logging_utils.py` ships two general-purpose helpers:

```python
from src.logging_utils import stage_timer

with stage_timer("Candidate generation", logger=logger):
    candidates = generate_candidates(anchor)
# DEBUG "Candidate generation starting...", INFO "Candidate generation completed in 2.3s"
```

```python
from src.logging_utils import RunSummary

summary = RunSummary("Genre Update")
summary.add("artists_processed", 150)
summary.increment("api_calls")
summary.log()
# INFO block: "GENRE UPDATE SUMMARY" / "  Artists Processed: 150" / ... / "  Total Time: 12.3s"
```

Both are unit-tested (`tests/unit/test_logging_utils.py`) and safe to reach for in new code.
**Neither is currently called by a production entrypoint.** `main_app.py` and
`scripts/analyze_library.py` each have their own bespoke summary format instead — the analyze
pipeline's `RUN RECAP` block above is the real production pattern for that entrypoint. Don't
assume `stage_timer`/`RunSummary` output appears somewhere in a real log just because they exist
in the module.

---

## Redaction policy

### Never log

- API keys, tokens, or passwords
- Full file paths containing usernames
- Database connection strings with credentials
- Config contents wholesale — log a hash (`config_hash` in the analyze recap above) or a redacted
  render instead

### Two redaction paths — know which one actually runs

There are **two independent redaction implementations**, and only one of them is wired into a
production call site:

1. **`src.logging_utils.redact()`** — regex-based, redacts API-key/token/secret/password
   patterns, Windows/Unix home directories, and email addresses; accepts extra `keys=[...]` for
   dict-shaped text. It is unit-tested and safe to use, but as of this writing **no production
   log call in `src/` or `scripts/` actually invokes it** — it's available infrastructure, not a
   proven-in-use guard. If you add logging that might surface a path or secret from `main_app.py`
   or `scripts/analyze_library.py`, wrap it in `redact()` yourself; don't assume it already
   happens.
2. **`src.playlist_gui.utils.redaction.redact_text()` / `redact_mapping()`** — the GUI/worker
   equivalent, covering API keys, `Authorization: Bearer`, URL query tokens, and CLI-flag-style
   secrets. This one **is** on every production path: `src/playlist_gui/worker.py` calls
   `redact_text()` inside `emit_log()` and `emit_error()`, so every NDJSON log/error event sent
   to the browser GUI is redacted before it leaves the worker process.

```python
# CLI / analyze side — available, call it explicitly
from src.logging_utils import redact
logger.info(f"Config loaded from {redact(config_path)}")

# GUI worker side — already wired, redacts every emitted log line
from src.playlist_gui.utils.redaction import redact_text
emit_log("INFO", redact_text(msg))  # worker.py does this inside emit_log() itself
```

### Path redaction

Prefer relative or basename paths over absolute ones in any log line a user might paste
somewhere public:

```python
# Avoid
logger.info(f"Scanning: C:/Users/dylan/Music/library")

# Prefer
logger.info(f"Scanning: {path.relative_to(library_root)}")
```

---

## Console vs. file output

There is no color/ANSI formatting in either handler — `configure_logging()` uses a plain
`logging.Formatter` for both. The two handlers differ in **level and detail**, not styling:

| | Console | File (`--log-file`) |
|---|---------|----------------------|
| Level | Configured level (`--log-level`, default INFO) | Always `DEBUG` and above (`file_level` param) |
| `run_id` | Omitted unless level is DEBUG or `--show-run-id` | Always included |
| `funcName:lineno` | Never | Always |

```python
configure_logging(
    level='INFO',           # console shows INFO+
    log_file='run.log',     # file captures everything
    file_level='DEBUG',     # file gets DEBUG+ regardless of console level
)
```

This means `--log-file` is the right answer whenever you need full detail without cluttering the
console — the file always has more than the screen, never less.

---

## Browser GUI / worker logging

The worker (`src/playlist_gui/worker.py`, spawned once by `src/playlist_web/app.py` — see
[`ARCHITECTURE.md`](ARCHITECTURE.md) "Browser GUI wiring") is a separate process with its own
logging setup, `setup_worker_logging()`:

- It replaces the root logger's handlers with a single `WorkerLogHandler` that emits every log
  record as an NDJSON `{"type": "log", "level": ..., "msg": ...}` event over stdout, rather than
  formatting to a stream — the browser (not a terminal) is the console here.
- `emit_log()` runs every message through `redact_text()` before it's ever serialized (see
  Redaction above) — this is the one redaction path that's actually load-bearing.
- The GUI can change the worker's live log level at runtime via the `set_logging_level` command
  (`handle_set_logging_level` → `logging_utils.set_log_level()`), without restarting the worker
  subprocess.
- When the worker drives `scripts/analyze_library.py`'s `run_pipeline()` for the Tools panel, it
  passes `console_logging=False` so the analyze pipeline's own stdout logging doesn't fight with
  the NDJSON stream; a separate `AnalyzeLibraryProgressLogHandler` bridges analyze's log records
  into worker progress events instead.

---

## Migration checklist

When touching logging in an existing module:

1. Remove any `logging.basicConfig()` call (there should be none — see "Single setup rule").
2. Replace `print()` with `logger.info/debug/warning/error`.
3. Move per-item logging from INFO to DEBUG.
4. Add a summary log at INFO for anything that loops.
5. Use `logging.getLogger(__name__)`.
6. Add timing for anything that could plausibly take >1s (`stage_timer` or a manual
   `time.perf_counter()` pair).
7. Check for secrets or raw absolute paths in any new log line; redact if a user is likely to
   paste the log somewhere (see "Redaction policy" — know which redaction path actually applies
   to your entrypoint).
