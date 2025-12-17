# Logging Optimization Plan

Strategic improvements to make logging more cohesive, informative, and less extraneous.

## Current State Summary

- **51 files** import logging
- **447 log statements** across codebase
- **35+ print statements** (inconsistency!)
- **33+ basicConfig** calls in scripts (conflicts!)
- **240+ DEBUG statements** in 2 files (verbose!)
- **Only 1 file** logs context manager exit (fragmented!)
- **No structured logging** (hard to parse/analyze)
- **No correlation IDs** (can't trace requests)

## Phase 1: Remove Extraneous Content (Quick Wins)

### 1.1 Eliminate Print Statements (8 files, ~35 instances)

**Target Files:**
- `src/artist_cache.py` - Cache stats printed
- `src/hybrid_sonic_analyzer.py` - Success messages printed (5 instances)
- `src/genre_similarity_v2.py` - Comparison debug printed
- `src/librosa_analyzer.py` - Success messages printed (5 instances)
- `src/multi_source_genre_fetcher.py` - Test data printed
- `src/similarity_calculator.py` - Stats printed
- `src/local_library_client.py` - Sample tracks printed
- `main_app.py` - User output printed (extensive)

**Action:** Replace all `print()` with `logger.info()` or `logger.debug()`

**Expected Impact:**
- All output now goes to log file + console
- Consistent formatting
- Respects log level configuration
- User can suppress with lower log level

---

### 1.2 Remove Verbose DEBUG Statements from Core Files (2 files, ~60 statements)

**Target Files:**
- `src/playlist_generator.py` - 50+ debug statements
- `src/similarity_calculator.py` - 40+ debug statements

**Strategy: Be Selective**
- Keep DEBUG for: error conditions, unusual paths, performance metrics
- Remove DEBUG for: routine operations, inner loop logging, verbose object dumps
- Example to REMOVE: `logger.debug(f"Comparing: '{artist1}' <-> '{artist2}'")`
- Example to KEEP: `logger.debug(f"Transition floor {score:.3f} < threshold {floor:.3f}, rejecting")`

**Action Items:**
1. Audit each DEBUG statement for necessity
2. Keep ~50% (most critical)
3. Move routine logging to TRACE level (requires new level)
4. OR: Move routine to `logger.debug()` but gate with conditions

**Expected Impact:**
- DEBUG mode still informative but less verbose
- Easier to read debug logs
- Can add TRACE level later if detailed debugging needed

---

### 1.3 Consolidate basicConfig in Scripts (33+ files)

**Target Files:**
- All scripts with `if __name__ == "__main__": logging.basicConfig()`
- Archive and legacy scripts

**Action:**
1. Create `src/logging_config.py` with centralized setup
2. Import in all scripts: `from src.logging_config import setup_logging`
3. Call: `setup_logging(level=args.level, file=args.log_file)`
4. Remove all individual basicConfig calls

**Expected Impact:**
- Consistent logging format across all scripts
- Single source of truth
- Environment variables work everywhere
- Command-line args work in all scripts

---

## Phase 2: Improve Cohesion & Informativeness

### 2.1 Fix Handler Duplication in main_app.py

**Current Issue (Lines 126-134):**
```python
for module in ['src.openai_client', 'src.playlist_generator', ...]:
    mod_logger = logging.getLogger(module)
    mod_logger.addHandler(console_handler)
    mod_logger.addHandler(file_handler)
```

**Problem:** Same handler objects added multiple times to multiple loggers

**Fix:**
1. Set up handlers once (already done)
2. Use logger hierarchy with root logger instead:
   ```python
   root_logger = logging.getLogger()
   root_logger.addHandler(console_handler)
   root_logger.addHandler(file_handler)
   ```
3. Remove explicit handler assignment to module loggers
4. All child loggers inherit handlers automatically

**Expected Impact:**
- Cleaner code
- No duplicate handler references
- Better memory efficiency
- Easier to add/remove handlers

---

### 2.2 Add Entry/Exit Logging to Major Functions

**Target Functions:**
- All functions >50 lines or with external I/O
- Examples: `generate_playlist()`, `analyze_track()`, `fetch_genres()`

**Pattern:**
```python
def generate_playlist(self, seed_track_id, count=50):
    logger.info(f"Generating playlist from {seed_track_id} ({count} tracks)")
    try:
        # ... implementation ...
        logger.info(f"Playlist generated successfully: {len(playlist)} tracks")
        return playlist
    except Exception as e:
        logger.error(f"Playlist generation failed: {e}", exc_info=True)
        raise
```

**Expected Impact:**
- Can trace request flow
- See function entry/exit timing
- Better error context

---

### 2.3 Add Context Manager Logging Consistently

**Target:** All context managers in src/ (4 files)

**Pattern:**
```python
def __enter__(self):
    logger.debug(f"Opening {self.__class__.__name__}")
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    logger.debug(f"Closing {self.__class__.__name__}")
    self.close()
```

**Expected Impact:**
- See resource lifecycle
- Detect resource leaks
- Better request tracing

---

### 2.4 Use exc_info=True Consistently

**Current:** Only 8 files use `exc_info=True`

**Target:** All exception logging (20+ more files)

**Pattern:**
```python
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise
```

**Expected Impact:**
- Complete stack traces in error logs
- Easier debugging
- Better error context

---

## Phase 3: Add Structure & Tracing (Advanced)

### 3.1 Add Structured Logging Configuration

**New File:** `src/logging_config.py`

```python
import logging
import json
from typing import Optional

class StructuredFormatter(logging.Formatter):
    """Outputs structured JSON logs when in 'structured' mode."""

    def __init__(self, structured: bool = False):
        super().__init__()
        self.structured = structured

    def format(self, record):
        if self.structured:
            log_obj = {
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'module': record.name,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'request_id': getattr(record, 'request_id', None),
            }
            if record.exc_info:
                log_obj['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_obj)
        else:
            return super().format(record)

def setup_logging(level='INFO', file='playlist_generator.log', structured=False):
    """Unified logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level))
    console_fmt = StructuredFormatter(structured)
    console.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG in file
    file_fmt = StructuredFormatter(structured)
    file_handler.setFormatter(file_fmt)

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    return root_logger
```

**Usage:**
```python
# In config.yaml
logging:
  level: DEBUG
  file: playlist_generator.log
  structured: false  # Set to true for JSON logs

# In code
from src.logging_config import setup_logging
logger = setup_logging(level=config['logging']['level'],
                       structured=config['logging'].get('structured', False))
```

**Expected Impact:**
- Option to switch to JSON structured logging later
- Easier log aggregation
- Better for production deployments

---

### 3.2 Add Correlation IDs for Request Tracing

**Pattern:**
```python
import uuid
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar('request_id', default='')

class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        return True

# Add to logging setup
request_id_filter = RequestIDFilter()
logger.addFilter(request_id_filter)

# In API endpoints
@app.post("/api/playlists/generate")
def generate_playlist(payload):
    rid = str(uuid.uuid4())
    request_id.set(rid)
    logger.info(f"Generating playlist (request {rid})")
    # ... can now trace all logs to this request
```

**Expected Impact:**
- Can correlate all logs from single request
- Critical for debugging multi-step operations
- Essential for async/concurrent requests

---

## Phase 4: Document & Standardize

### 4.1 Logging Standards Document

**Create:** `LOGGING_STANDARDS.md`

**Include:**
- When to use each log level (INFO, DEBUG, WARNING, ERROR)
- Entry/exit pattern for functions
- Exception logging with exc_info=True
- Context manager pattern
- Correlation ID usage
- Examples for common patterns

### 4.2 Update Configuration Documentation

**Modify:** `docs/configuration.md` logging section

**Add:**
- Explanation of logging levels
- How to enable DEBUG mode
- How to interpret log files
- Where to find logs
- Performance impact of DEBUG

---

## Implementation Priority & Effort

| Phase | Task | Files | Effort | Priority | Impact |
|-------|------|-------|--------|----------|--------|
| 1.1 | Remove print statements | 8 | 30 min | ⚠️ HIGH | Medium - Consistency |
| 1.2 | Remove verbose DEBUG | 2 | 45 min | ⚠️ HIGH | High - Readability |
| 1.3 | Consolidate basicConfig | 33+ | 1 hour | ✅ MEDIUM | High - Consistency |
| 2.1 | Fix handler duplication | 1 | 15 min | ✅ MEDIUM | Low - Maintenance |
| 2.2 | Add entry/exit logging | 15+ | 1 hour | ⚠️ HIGH | Medium - Debugging |
| 2.3 | Add context manager logging | 4 | 20 min | ✅ MEDIUM | Low - Visibility |
| 2.4 | Use exc_info consistently | 20+ | 30 min | ✅ MEDIUM | Medium - Debugging |
| 3.1 | Add structured logging | 1 | 1 hour | 🔵 LOW | High - Scalability |
| 3.2 | Add correlation IDs | 3 | 1 hour | 🔵 LOW | Medium - Tracing |
| 4.1 | Document standards | 1 | 30 min | ✅ MEDIUM | High - Education |
| 4.2 | Update config docs | 1 | 15 min | ✅ MEDIUM | Low - Docs |

**Total Estimated Effort:** 6-7 hours
**Recommended:** Phase 1 (1.5 hrs) + Phase 2 (1.5 hrs) = 3 hours essential improvements

---

## Suggested Implementation Order

### Immediate (While Sonic Rebuild Runs)
1. Phase 1.1: Remove print statements (30 min) ← START HERE
2. Phase 1.2: Audit & remove verbose DEBUG (45 min)
3. Phase 2.4: Add exc_info=True (30 min)
4. Phase 2.2: Add entry/exit logging to top functions (1 hour)

### Next Session
5. Phase 1.3: Consolidate basicConfig (1 hour)
6. Phase 2.1: Fix handler duplication (15 min)
7. Phase 2.3: Add context manager logging (20 min)
8. Phase 4.1 & 4.2: Document standards (45 min)

### Future (After Sonic Rebuild Completes)
9. Phase 3.1: Add structured logging (optional)
10. Phase 3.2: Add correlation IDs (optional, for API)

---

## Success Criteria

### Phase 1 Complete
- ✅ All print statements removed
- ✅ DEBUG verbosity reduced by ~50%
- ✅ All basicConfig consolidated
- ✅ All tests still pass

### Phase 2 Complete
- ✅ Handler duplication fixed
- ✅ Entry/exit logging added
- ✅ exc_info=True used everywhere
- ✅ Context managers log lifecycle

### Phase 3 Complete
- ✅ Structured logging option available
- ✅ Correlation IDs working
- ✅ Standards documented
- ✅ Production-ready logging infrastructure

---

## Rollback Plan

If improvements break anything:
1. Keep original logging config in git
2. Commit each phase separately
3. Can revert individual phases
4. Tests catch regressions immediately

---

## Notes

- Sonic rebuild will continue during Phase 1-2 implementation (no conflicts)
- Logging changes are non-breaking (same outputs, different sources)
- Can run all tests after each phase to validate
- Archive/legacy files can be cleaned later (Phase 4)

