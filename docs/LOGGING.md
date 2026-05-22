# Logging Standards

This document defines the logging policy for the playlist generator codebase.

## Levels Policy

| Level | When to Use | Examples |
|-------|-------------|----------|
| `ERROR` | Operation failed, requires attention | Database connection failed, API error after retries |
| `WARNING` | Recoverable issue, degraded operation | Missing optional config, skipped item, rate limit hit |
| `INFO` | Milestones and summaries only | Stage start/complete, final counts, run summary |
| `DEBUG` | Per-item details, diagnostic data | Per-track processing, per-candidate scoring, edge weights |

### Key Rules

1. **INFO is for humans scanning logs** - Should produce ~10-50 lines for a typical run
2. **DEBUG is for troubleshooting** - Enable when investigating issues
3. **Never log per-item at INFO** - Move all per-track, per-artist, per-edge logs to DEBUG
4. **Summaries replace streams** - Instead of logging each item, log "Processed 500/500 tracks"

### Before/After Examples

```python
# BAD - Per-item at INFO
for track in tracks:
    logger.info(f"Processing: {track.title}")

# GOOD - Summary at INFO, details at DEBUG
for track in tracks:
    logger.debug(f"Processing: {track.title}")
logger.info(f"Processed {len(tracks)} tracks")
```

```python
# BAD - Logging every edge
for edge in edges:
    logger.info(f"Edge: {edge.from_track} -> {edge.to_track} score={edge.score}")

# GOOD - Summary only at INFO
logger.debug(f"Edge scores: min={min_score:.3f} max={max_score:.3f} mean={mean_score:.3f}")
logger.info(f"Built {len(edges)} edges")
```

## Logger Naming

Use `__name__` for all loggers to maintain Python package hierarchy:

```python
# CORRECT
logger = logging.getLogger(__name__)

# AVOID - Breaks hierarchy
logger = logging.getLogger("my_custom_name")
```

This produces natural hierarchies like:
- `src.playlist.pipeline`
- `src.playlist.ordering`
- `src.features.librosa_analyzer`

For scripts, use the script name:
```python
logger = logging.getLogger("scan_library")
```

## Message Style

### Format

All log messages use this format:
```
%(asctime)s | %(levelname)-5s | %(name)s | run_id=%(run_id)s | %(message)s
```
`run_id` is injected automatically by `configure_logging` (defaults to `-` when not provided).

Run IDs:
- Console output **omits run_id by default** for readability.
- File logs always include run_id.
- Add `--show-run-id` (or set log level DEBUG) to include run_id on console.
Example default console output:
```
10:30:45 | INFO  | src.playlist.pipeline | Starting playlist generation
```
Example with `--show-run-id` or DEBUG:
```
10:30:45 | INFO  | src.playlist.pipeline | run_id=abcd1234 | Starting playlist generation
```

### Message Guidelines

1. **Start with action verb** - "Processing...", "Found...", "Completed..."
2. **Include counts** - "Found 150 candidates" not "Found candidates"
3. **Include timing for stages** - "Completed in 1.2s"
4. **No trailing punctuation** - "Processing tracks" not "Processing tracks."
5. **Truncate long lists** - "Genres: rock, pop, jazz (+5 more)"

### Structured Data

For complex data, use key=value format:
```python
logger.info(f"Playlist complete: tracks={len(tracks)} duration={duration:.1f}min unique_artists={n_artists}")
```

## Progress + ETA

- Default runs show **periodic INFO summaries** using `ProgressLogger` every `--progress-interval` seconds or `--progress-every` items (defaults: 15s or 500 items). Each summary includes percent, rate, and ETA when a total is known.
- Verbose runs (`--verbose`) enable **per-item DEBUG logs** while still emitting periodic INFO summaries. Use when debugging a specific item flow.
- Disable progress output with `--no-progress` if you prefer silent runs or need clean stdout for piping.
- All entrypoints accept: `--progress/--no-progress`, `--progress-interval`, `--progress-every`, `--verbose`, plus standard `--log-level/--debug/--quiet/--log-file`.

### Recommended defaults and examples

- Default scan (`python scripts/scan_library.py --quick`):
  - INFO: `scan_library: 1,250/3,210 files (38.9%) | 82.3 files/s | ETA 25s`
  - INFO final: `scan_library complete: 3,210 files | elapsed 1m02s | avg 51.6 files/s`
- Verbose scan (`python scripts/scan_library.py --verbose`):
  - DEBUG per file: `scan_library item 125/3210: Artist - Title (song.flac)`
  - INFO periodic summaries as above.
- Quiet mode (`--quiet`):
  - Suppresses INFO progress; WARNING/ERROR still emitted.
- Run IDs:
  - Normal runs: default console (no run_id), add `--log-file` to capture run_id in file.
  - Debugging or correlation: `--show-run-id` to include run_id on console, or run with `--debug`.

Always redact secrets and sensitive paths when logging; prefer basename/relative paths and avoid logging tokens, API keys, or credentials.

## Timings and Metrics

### Stage Timer Context Manager

Use `stage_timer` for automatic timing of pipeline stages:

```python
from src.logging_utils import stage_timer

with stage_timer("Candidate generation"):
    candidates = generate_candidates(anchor)
# Logs: "Candidate generation completed in 2.3s"
```

### Run Summaries

Each entrypoint should log a final summary:

```python
logger.info("=" * 60)
logger.info("RUN SUMMARY")
logger.info(f"  Tracks processed: {n_tracks}")
logger.info(f"  Success rate: {success_rate:.1%}")
logger.info(f"  Total time: {elapsed:.1f}s")
logger.info("=" * 60)
```

### Metrics to Always Include

| Workflow | Required Metrics |
|----------|------------------|
| scan_library | tracks_found, tracks_added, tracks_updated, errors |
| update_sonic | tracks_analyzed, features_extracted, failures |
| update_genres | artists_updated, albums_updated, api_calls, empty_results |
| main_app | anchor_track, candidates_considered, final_tracks, generation_time |

## Analyze Library Logging

- Startup logs include `run_id`, db path, out_dir, selected stages, config hash, git commit, and progress settings.
- Each stage logs a **plan line** with current/last fingerprints, skip eligibility, and pending counts; skips record explicit reasons (e.g., fingerprint unchanged).
- Stage execution logs start/end with durations; periodic progress/ETA comes from stage-specific `ProgressLogger` instances.
- End-of-run summary lists stage, decision, reason, items (when known), duration, plus verify issues if present.
- `--verbose` enables per-item DEBUG logs within stages; `--no-progress` disables periodic summaries.

## Redaction Policy

### Never Log

- API keys or tokens
- Full file paths containing usernames
- Database connection strings with credentials

### Redaction Helper

```python
from src.logging_utils import redact

# Redacts sensitive patterns
logger.info(f"Config loaded from {redact(config_path)}")
# Output: "Config loaded from .../config.yaml"

logger.debug(f"API response: {redact(response, keys=['token', 'key'])}")
```

### Path Redaction

Always use relative paths or redact absolute paths:
```python
# BAD
logger.info(f"Scanning: C:/Users/john/Music/library")

# GOOD
logger.info(f"Scanning: {path.relative_to(library_root)}")
```

## CLI Controls

All entrypoints support these flags:

| Flag | Effect |
|------|--------|
| `--log-level LEVEL` | Set level (DEBUG, INFO, WARNING, ERROR) |
| `--debug` | Shortcut for `--log-level DEBUG` |
| `--quiet` | Shortcut for `--log-level WARNING` |
| `--log-file PATH` | Also write logs to file |

### Implementation

```python
import argparse
from src.logging_utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
parser.add_argument('--debug', action='store_true')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--log-file', type=str)
args = parser.parse_args()

# Resolve level
level = 'DEBUG' if args.debug else 'WARNING' if args.quiet else args.log_level

configure_logging(level=level, log_file=args.log_file)
```

## Single Setup Rule

**Logging is configured exactly once, at the entrypoint.**

```python
# main_app.py (entrypoint)
from src.logging_utils import configure_logging
configure_logging(level='INFO')

# src/playlist/pipeline.py (library module)
import logging
logger = logging.getLogger(__name__)  # Just get logger, don't configure
```

### What NOT to Do

```python
# BAD - Module configures its own logging
# src/some_module.py
import logging
logging.basicConfig(level=logging.DEBUG)  # NO!

# BAD - Multiple setup calls
if __name__ == "__main__":
    logging.basicConfig(...)  # Don't do this in __main__ blocks
```

## Migration Checklist

When updating a module:

1. [ ] Remove any `logging.basicConfig()` calls
2. [ ] Replace `print()` with appropriate `logger.info/debug/warning`
3. [ ] Change per-item INFO logs to DEBUG
4. [ ] Add summary logs at INFO level
5. [ ] Use `__name__` for logger
6. [ ] Add timing for major operations
7. [ ] Verify no secrets in log output

## Console vs File Output

- **Console**: Colored output, shows INFO and above by default
- **File**: Full detail, includes DEBUG, uses plain text format

```python
configure_logging(
    level='INFO',           # Console shows INFO+
    log_file='run.log',     # File captures everything
    file_level='DEBUG'      # File gets DEBUG+
)
```
