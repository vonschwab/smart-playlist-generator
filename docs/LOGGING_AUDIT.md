# Logging Audit Report

## Summary

| Metric | Count |
|--------|-------|
| `print()` statements in production code | 147 |
| `logging.basicConfig()` calls | 15 |
| `logger.info()` calls | 478 |
| `logger.debug()` calls | 115 |
| Unique logger setup patterns | 3 |

## Current Patterns

### 1. Multiple Logging Setup Approaches

**Pattern A: `setup_logging()` from `src/logging_config.py`**
- Used by: `scan_library.py`, `update_sonic.py`, `update_genres_v3_normalized.py`, `analyze_library.py`
- Creates named loggers with console + file handlers
- Good: Centralized, handles UTF-8, removes duplicate handlers

**Pattern B: `_setup_logging()` in `main_app.py`**
- Custom per-class setup
- Manually configures handlers for 16+ modules by name
- Problem: Duplicates logic, adds handlers to each module separately

**Pattern C: `logging.basicConfig()` in module `__main__` blocks**
- Found in 15 modules (artist_cache, config_loader, librosa_analyzer, etc.)
- Problem: Inconsistent levels (some DEBUG, some INFO), conflicts with other patterns

### 2. Heavy Use of `print()`

**main_app.py** (60+ print statements):
- Summary reports (`PLAYLIST GENERATION SUMMARY`)
- Error messages (`print(f"\nError: {e}")`)
- User-facing status (`print(f"Generating playlist for: {artist_name}")`)

**Issue**: Mixes logging and print for similar purposes. User-facing output should use logging with a dedicated handler or be clearly separated.

### 3. Per-Item Logging in Loops

**Verbose INFO patterns:**
```python
# scripts/update_genres_v3_normalized.py
logger.info(f"[{i}/{total}] {artist}")

# src/playlist/reporter.py
logger.info(f"Track {i:02d}: {artist} - {title}")
```

These are acceptable for progress but some should be DEBUG.

**Per-edge logging in reporter.py:**
```python
logger.info(f"  #{i:02d}  T={t:.3f}  S={s:.3f}  G={g:.3f}  {edge_str}")
```
This logs every edge at INFO level - should be DEBUG.

### 4. Inconsistent Logger Names

```python
# Module-level loggers use different patterns:
logger = logging.getLogger(__name__)           # Most modules
logger = logging.getLogger("analyze_library")  # Named string
logger = logging.getLogger("update_genres")    # Named string
```

Named strings work but don't follow the `src.module` hierarchy.

### 5. No Secrets Logging (Good)

Only one potential issue found:
```python
self.logger.warning("Plex export enabled but base_url/token not configured")
```
This doesn't log the actual token value, just mentions it's missing. Safe.

### 6. Missing Features

- No run_id for correlating logs across a pipeline run
- No structured logging / JSON option
- No stage timing context manager
- No --log-level CLI flag (uses config file or env var)
- No redaction helper for future-proofing

## Biggest Problems

### Priority 1: Inconsistent Setup
Three different setup patterns cause:
- Duplicate handlers when modules import each other
- Inconsistent formatting between scripts and main_app
- Different log levels in different contexts

### Priority 2: 147 print() Statements
- Mixes with logging output
- Can't be redirected/filtered
- No timestamps or levels
- Breaks JSON logging if implemented

### Priority 3: INFO Overuse
- 478 INFO vs 115 DEBUG calls (4:1 ratio)
- Per-edge, per-track details at INFO level
- Running with defaults produces walls of text

### Priority 4: No CLI Flags
- Level controlled only by config or env var
- No `--quiet`, `--debug`, `--log-file` flags
- Users can't easily adjust verbosity

## Modules Needing Most Work

| Module | Issues |
|--------|--------|
| `main_app.py` | 60+ print(), custom _setup_logging(), per-module handler config |
| `src/playlist/reporter.py` | 40+ logger calls, per-edge INFO logging |
| `scripts/analyze_library.py` | Per-stage verbose logging, no summary |
| `src/playlist_generator.py` | 80+ logger calls, mixed levels |
| `src/playlist/candidate_generator.py` | Per-artist loop logging at INFO |

## Recommendations

1. **Single setup function** called once at entrypoint
2. **Convert print() to logging** with a user-facing handler
3. **Move per-item logs to DEBUG** - only summaries at INFO
4. **Add CLI flags** for --debug, --quiet, --log-file
5. **Add run_id** to correlate pipeline stages
6. **Add stage_timer** context manager for consistent timing
7. **Add redact() helper** for future-proofing secrets
