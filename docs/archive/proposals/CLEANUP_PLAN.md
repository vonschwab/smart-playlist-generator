# Repository Cleanup & Modernization Plan

**Generated:** December 2025
**Objective:** Professional, maintainable codebase ready for UI integration

---

# What I Found

## Repo Inventory & Key Entrypoints

### Primary Entrypoints (Currently Used)

| Entrypoint | Purpose | Command |
|------------|---------|---------|
| `main_app.py` | CLI playlist generation | `python main_app.py` |
| `api/main.py` | FastAPI backend | `uvicorn api.main:app --reload` |
| `scripts/analyze_library.py` | Unified pipeline | `python scripts/analyze_library.py` |
| `scripts/scan_library.py` | Library scan | `python scripts/scan_library.py` |
| `scripts/update_sonic.py` | Sonic analysis | `python scripts/update_sonic.py` |
| `ui/` | React frontend (Vite) | `cd ui && npm run dev` |

### Codebase Statistics

- **src/**: 37 Python files, ~12,000 lines
- **scripts/**: 24 Python files
- **tests/**: 19 test files, 111 tests collected
- **docs/**: 11 documentation files (mixed freshness)

---

## Directory Classification Table

| Directory/File | Classification | Rationale | Action |
|----------------|----------------|-----------|--------|
| **src/** | ESSENTIAL | Core library code, all imports resolve here | Keep & polish |
| **src/playlist/** | ESSENTIAL | DS pipeline implementation | Keep |
| **src/analyze/** | ESSENTIAL | Artifact builder | Keep |
| **src/similarity/** | ESSENTIAL | Hybrid/sonic similarity | Keep |
| **src/features/** | ESSENTIAL | Artifact loader | Keep |
| **src/eval/** | SUPPORTING | Run artifact evaluation | Review usage |
| **src/genre_similarity.py** | LEGACY | Deprecated shim (emits warning) | Archive → delete |
| **api/** | ESSENTIAL | FastAPI backend for UI | Keep & polish |
| **api/services/** | ESSENTIAL | Last.fm service | Keep |
| **scripts/** | ESSENTIAL | Operational scripts | Keep, but reorganize |
| **tests/** | SUPPORTING | Test suite (111 tests) | Rationalize |
| **data/** | ESSENTIAL | metadata.db, genre_similarity.yaml | Keep |
| **docs/** | SUPPORTING | Mixed current/stale | Overhaul |
| **ui/** | ESSENTIAL | React frontend | Keep (for UI work) |
| **experiments/** | LEGACY | Lab notebooks, planning docs | Archive |
| **diagnostics/** | LEGACY | Debug outputs | Gitignore, archive |
| **discogs-genre-test/** | LEGACY | One-off test | Archive |
| `main_app.py` | ESSENTIAL | CLI entrypoint | Keep |
| `config.yaml` | ESSENTIAL | User configuration | Keep (gitignored) |
| `config.example.yaml` | ESSENTIAL | Config template | Keep |
| `requirements.txt` | ESSENTIAL | Dependencies | Keep |
| `README.md` | ESSENTIAL | Main docs | Rewrite |
| `DEV.md` | SUPPORTING | Dev quickstart | Merge into docs/ |
| `AGENTS.md` | SUPPORTING | Claude Code hints | Keep |
| `REPOSITORY_STRUCTURE.md` | LEGACY | Outdated structure doc | Delete after docs overhaul |
| `CODE_CLEANUP_AND_OPTIMIZATION.md` | LEGACY | Analysis artifact | Archive |
| `DIAL_GRID_FIX_SUMMARY.md` | LEGACY | Session artifact | Archive |
| `SCAN_LIBRARY_IMPROVEMENTS_SUMMARY.txt` | LEGACY | Session artifact | Archive |
| `discogs_misses.tsv` | REMOVE | Diagnostic output | Delete |
| `metadata.db` (root) | REMOVE | Duplicate of data/metadata.db | Delete |
| `*.log` (root) | REMOVE | Should be gitignored | Delete, add to .gitignore |

---

## scripts/ Detailed Classification

| Script | Classification | Used By | Action |
|--------|----------------|---------|--------|
| `analyze_library.py` | ESSENTIAL | Main unified pipeline | Keep |
| `scan_library.py` | ESSENTIAL | analyze_library.py | Keep |
| `update_sonic.py` | ESSENTIAL | analyze_library.py | Keep |
| `update_genres_v3_normalized.py` | ESSENTIAL | analyze_library.py | Keep |
| `validate_metadata.py` | SUPPORTING | Manual validation | Keep |
| `validate_duration.py` | SUPPORTING | Duration checks | Keep |
| `validate_title_dedupe.py` | SUPPORTING | Dedupe testing | Keep |
| `backfill_duration.py` | SUPPORTING | One-time migration | Archive after use |
| `check_duration_health.py` | SUPPORTING | Monitoring | Keep |
| `diagnose_*.py` (6 files) | LEGACY | One-off debugging | Archive |
| `ab_playlist_sonic_variants.py` | LEGACY | A/B testing | Archive |
| `batch_eval_sonic_variant.py` | LEGACY | Batch evaluation | Archive |
| `sonic_separability.py` | LEGACY | Analysis | Archive |
| `tune_dial_grid.py` | LEGACY | Tuning experiment | Archive |
| `infer_sonic_feature_schema.py` | LEGACY | Schema inference | Archive |
| `build_track_group.py` | LEGACY | Test data builder | Archive |
| `make_pairs_file.py` | LEGACY | Test data builder | Archive |
| `debug_edge_scores.py` | LEGACY | Debugging | Archive |
| `purge_lastfm_genres.py` | LEGACY | One-time cleanup | Archive |
| `update_discogs_genres.py` | LEGACY | Discogs experiment | Archive |

---

## docs/ Detailed Classification

| Document | Classification | Status | Action |
|----------|----------------|--------|--------|
| `TITLE_DEDUPLICATION.md` | ESSENTIAL | Current | Keep |
| `DURATION_SUPPORT.md` | ESSENTIAL | Current | Keep |
| `SCAN_LIBRARY_ANALYSIS.md` | SUPPORTING | Current | Keep |
| `GENRE_SIMILARITY_SYSTEM.md` | ESSENTIAL | Current | Keep |
| `GENRE_SIMILARITY_METHODS.md` | SUPPORTING | Current | Keep |
| `API_REFERENCE.md` | ESSENTIAL | Needs update | Update |
| `TUNING_WORKFLOW.md` | SUPPORTING | Current | Keep |
| `INTEGRATION_COMPLETE.md` | LEGACY | Historical | Archive |
| `ds_pipeline_integration_plan.md` | LEGACY | Planning doc | Archive |
| `NEW_LOG_MOCKUP.log` | LEGACY | Mock/example | Delete |
| `session_notes/` | LEGACY | Dev notes | Archive |

---

## tests/ Classification

| Test File | Classification | Tests | Action |
|-----------|----------------|-------|--------|
| `test_title_dedupe.py` | ESSENTIAL | 38 | Keep |
| `test_artifacts.py` | ESSENTIAL | Core functionality | Keep |
| `test_constraints.py` | ESSENTIAL | Playlist constraints | Keep |
| `test_ds_pipeline_determinism.py` | ESSENTIAL | Pipeline stability | Keep |
| `test_pipeline_switch.py` | ESSENTIAL | Pipeline mode | Keep |
| `test_duration_similarity.py` | ESSENTIAL | Duration scoring | Keep |
| `test_sonic_variant.py` | SUPPORTING | Variant handling | Keep |
| `test_sonic_variant_similarity.py` | SUPPORTING | Sonic similarity | Keep |
| `test_sonic_variant_gate.py` | SUPPORTING | Feature gate | Keep |
| `test_analyze_library_cli.py` | SUPPORTING | CLI smoke test | Keep |
| `test_batch_eval_sonic_variant.py` | LEGACY | Batch eval | Archive |
| `test_diagnose_sonic_feature_ablation.py` | LEGACY | Diagnostic test | Archive |
| `test_diagnose_sonic_geometry.py` | LEGACY | Diagnostic test | Archive |
| `test_sonic_separability.py` | LEGACY | Separability test | Archive |
| `test_infer_sonic_feature_schema.py` | LEGACY | Schema inference | Archive |
| `test_dial_grid_tuning.py` | LEGACY | Tuning test | Archive |
| `test_ds_pool_logging.py` | LEGACY | Logging test | Review |
| `test_run_artifact.py` | SUPPORTING | Artifact eval | Keep |
| `conftest.py` | ESSENTIAL | Fixtures | Keep |

---

# Proposed Target Structure

```
PLAYLIST GENERATOR/
├── README.md                    # Consolidated quickstart + overview
├── CHANGELOG.md                 # Release notes
├── AGENTS.md                    # Claude Code hints (optional)
├── requirements.txt             # Python dependencies
├── config.example.yaml          # Config template
├── pyproject.toml               # (future) Modern Python packaging
│
├── src/                         # Production library code
│   ├── __init__.py
│   ├── config_loader.py         # Configuration
│   ├── playlist_generator.py    # Core generator
│   ├── similarity_calculator.py # Hybrid scoring
│   ├── local_library_client.py  # Library interface
│   ├── metadata_client.py       # Database interface
│   ├── track_matcher.py         # Track matching
│   ├── m3u_exporter.py          # Playlist export
│   ├── title_dedupe.py          # Title deduplication
│   ├── string_utils.py          # String helpers
│   ├── artist_utils.py          # Artist helpers
│   ├── genre_similarity_v2.py   # Genre similarity (renamed from _v2)
│   ├── librosa_analyzer.py      # Audio analysis
│   ├── hybrid_sonic_analyzer.py # Multi-segment analysis
│   ├── openai_client.py         # OpenAI integration
│   ├── lastfm_client.py         # Last.fm client
│   ├── multi_source_genre_fetcher.py
│   ├── rate_limiter.py
│   ├── retry_helper.py
│   ├── artist_cache.py
│   │
│   ├── analyze/                 # Artifact building
│   │   ├── __init__.py
│   │   ├── artifact_builder.py
│   │   └── genre_similarity.py
│   │
│   ├── features/                # Feature extraction
│   │   ├── __init__.py
│   │   └── artifacts.py
│   │
│   ├── similarity/              # Similarity modules
│   │   ├── __init__.py
│   │   ├── hybrid.py
│   │   ├── sonic_schema.py
│   │   └── sonic_variant.py
│   │
│   └── playlist/                # Playlist construction
│       ├── __init__.py
│       ├── candidate_pool.py
│       ├── config.py
│       ├── constructor.py
│       ├── ds_pipeline_runner.py
│       └── pipeline.py
│
├── api/                         # FastAPI backend
│   ├── __init__.py
│   ├── __main__.py              # Module entrypoint
│   ├── main.py                  # FastAPI app
│   └── services/
│       └── lastfm_service.py
│
├── cli/                         # CLI entrypoints (new)
│   ├── __init__.py
│   ├── generate.py              # Renamed from main_app.py
│   └── analyze.py               # Thin wrapper for scripts/analyze_library.py
│
├── scripts/                     # Operational scripts (trimmed)
│   ├── README.md
│   ├── analyze_library.py       # Unified pipeline
│   ├── scan_library.py          # Library scanner
│   ├── update_sonic.py          # Sonic analysis
│   ├── update_genres_v3_normalized.py
│   ├── validate_metadata.py
│   ├── validate_duration.py
│   └── validate_title_dedupe.py
│
├── tests/                       # Test suite (rationalized)
│   ├── conftest.py
│   ├── unit/                    # Fast, isolated tests
│   │   ├── test_title_dedupe.py
│   │   ├── test_duration_similarity.py
│   │   └── test_constraints.py
│   ├── integration/             # Tests requiring DB/artifacts
│   │   ├── test_artifacts.py
│   │   ├── test_ds_pipeline_determinism.py
│   │   ├── test_pipeline_switch.py
│   │   └── test_sonic_variant.py
│   └── smoke/                   # End-to-end quick checks
│       └── test_cli_smoke.py    # New: validates CLI commands
│
├── docs/                        # Documentation (overhauled)
│   ├── index.md                 # Overview + links
│   ├── quickstart.md            # Installation + first playlist
│   ├── architecture.md          # System diagram + data flow
│   ├── data_model.md            # SQLite schema, key entities
│   ├── pipelines.md             # scan/analyze/artifacts
│   ├── playlist_generation.md   # Modes, constraints, scoring, dedupe
│   ├── api.md                   # API endpoints + usage
│   ├── dev.md                   # Dev setup, tests, troubleshooting
│   ├── configuration.md         # All config options explained
│   └── reference/               # Deep-dive docs
│       ├── genre_similarity.md
│       ├── sonic_analysis.md
│       └── title_deduplication.md
│
├── ui/                          # React frontend (unchanged structure)
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── data/                        # Runtime data (gitignored except templates)
│   ├── .gitkeep
│   ├── genre_similarity.yaml    # Committed template
│   └── metadata.db              # Gitignored
│
├── archive/                     # Deprecated code (new)
│   ├── README.md                # Explains what's here
│   ├── experiments/             # Moved from root
│   ├── diagnostics/             # Moved from root
│   ├── legacy_scripts/          # Archived scripts
│   ├── legacy_tests/            # Archived tests
│   └── session_artifacts/       # MD files from root
│
└── .github/                     # CI/CD (future)
    └── workflows/
        └── ci.yml
```

### Rationale

1. **Clear separation**: `src/` = library, `api/` = backend, `cli/` = entrypoints, `scripts/` = operational
2. **Archive strategy**: Nothing deleted, legacy code moves to `archive/`
3. **Test taxonomy**: `unit/` (fast), `integration/` (needs fixtures), `smoke/` (CLI validation)
4. **Docs overhaul**: User-focused structure with quickstart prominent
5. **Clean root**: Only essential files at top level
6. **UI boundary**: `api/` is the stable integration point for `ui/`

---

# Documentation Plan

## Proposed Docs Set

### README.md (Root - Rewrite)

```markdown
# Playlist Generator

AI-powered playlist generation using sonic analysis and genre metadata.

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Copy config: `cp config.example.yaml config.yaml`
3. Scan library: `python scripts/analyze_library.py`
4. Generate playlist: `python -m cli.generate`
5. Run API: `uvicorn api.main:app`

## Documentation

- [Quickstart Guide](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Development](docs/dev.md)

## Project Status

- Core: Stable
- API: Stable
- UI: In Development
```

### docs/quickstart.md

**Outline:**
1. Prerequisites (Python 3.10+, ffmpeg optional)
2. Installation (pip install, config setup)
3. First Run
   - Scan your library
   - Generate a playlist
   - Export to M3U
4. Running the API
5. Common Issues

### docs/architecture.md

**Outline:**
1. System Diagram (ASCII/Mermaid)
2. Component Overview
   - Library Scanner → Database
   - Sonic Analyzer → Feature Vectors
   - Genre Fetcher → Normalized Tags
   - Artifact Builder → NPZ matrices
   - Playlist Generator → DS Pipeline → M3U
3. Data Flow
4. Key Design Decisions

### docs/data_model.md

**Outline:**
1. SQLite Schema Diagram
2. Key Tables
   - `tracks` (columns, indexes)
   - `albums`
   - `artist_genres`, `album_genres`, `track_genres`
   - `sonic_features` (if separate)
3. NPZ Artifact Format
   - `track_ids`, `artist_keys`, `X_sonic`, `X_genre_raw`, etc.
4. Relationships

### docs/pipelines.md

**Outline:**
1. Unified Pipeline (`analyze_library.py`)
   - Stages: scan → genres → sonic → genre-sim → artifacts → verify
   - Flags: `--force`, `--limit`, `--stages`
2. Individual Scripts
   - scan_library.py
   - update_genres_v3_normalized.py
   - update_sonic.py
3. Artifact Building
4. Recommended Workflows
   - Fresh library setup
   - Incremental updates
   - Full rebuild

### docs/playlist_generation.md

**Outline:**
1. Modes
   - Legacy pipeline
   - DS pipeline (default)
   - Dynamic mode
2. Seed Selection
3. Candidate Pool
4. Scoring
   - Sonic similarity (60%)
   - Genre similarity (40%)
   - Duration preference
5. Constraints
   - Artist diversity (max per playlist, window)
   - Duration filtering
   - Recently played exclusion
6. Title Deduplication
7. Output Formats

### docs/api.md

**Outline:**
1. Running the API
2. Endpoints
   - `GET /api/library/status`
   - `GET /api/library/artists`
   - `GET /api/library/tracks`
   - `POST /api/playlist/generate`
   - `GET /api/settings`
   - `PUT /api/settings`
3. Request/Response Examples
4. Error Handling
5. CORS Configuration

### docs/dev.md

**Outline:**
1. Development Setup
2. Running Tests
   - `pytest tests/unit/`
   - `pytest tests/integration/`
   - `pytest tests/smoke/`
3. Code Style (formatting, linting)
4. Adding New Features
5. Debugging Tips
6. Troubleshooting

### CHANGELOG.md

**Outline:**
```markdown
# Changelog

## [Unreleased]
- Repo cleanup and reorganization
- Documentation overhaul

## [1.0.0] - 2025-12-XX
- Initial stable release
- DS pipeline as default
- Title deduplication
- Duration filtering
- FastAPI backend
```

---

## Stale Docs to Retire

| Document | Action |
|----------|--------|
| `REPOSITORY_STRUCTURE.md` | Delete after architecture.md written |
| `CODE_CLEANUP_AND_OPTIMIZATION.md` | Archive |
| `DIAL_GRID_FIX_SUMMARY.md` | Archive |
| `SCAN_LIBRARY_IMPROVEMENTS_SUMMARY.txt` | Archive |
| `docs/INTEGRATION_COMPLETE.md` | Archive |
| `docs/ds_pipeline_integration_plan.md` | Archive |
| `docs/NEW_LOG_MOCKUP.log` | Delete |
| `docs/session_notes/` | Archive |

---

# Test Strategy

## Current State

- **111 tests** collected
- Mix of unit, integration, and diagnostic tests
- No clear taxonomy
- Some tests for archived/experimental code

## Proposed Taxonomy

### tests/unit/ (Fast, No External Dependencies)

```
test_title_dedupe.py          # 38 tests - Title normalization, fuzzy matching
test_duration_similarity.py   # Duration scoring
test_constraints.py           # Playlist constraints
test_string_utils.py          # NEW: String normalization tests
test_config_loader.py         # NEW: Config parsing tests
```

### tests/integration/ (Requires Fixtures/DB)

```
test_artifacts.py             # Artifact loading
test_ds_pipeline_determinism.py # Pipeline reproducibility
test_pipeline_switch.py       # Pipeline mode switching
test_sonic_variant.py         # Sonic variant handling
test_sonic_variant_similarity.py
test_sonic_variant_gate.py
test_run_artifact.py          # Artifact evaluation
```

### tests/smoke/ (End-to-End, CLI Validation)

```
test_cli_smoke.py             # NEW: Validates main commands work
  - test_analyze_library_dry_run()
  - test_generate_playlist_help()
  - test_api_starts()
```

### tests/archive/ (Moved, Not Run by Default)

```
test_batch_eval_sonic_variant.py
test_diagnose_sonic_feature_ablation.py
test_diagnose_sonic_geometry.py
test_sonic_separability.py
test_infer_sonic_feature_schema.py
test_dial_grid_tuning.py
test_ds_pool_logging.py
```

## Missing High-Value Tests

| Test | Purpose | Priority |
|------|---------|----------|
| `test_cli_smoke.py` | Validate CLI commands execute | HIGH |
| `test_api_smoke.py` | Validate API starts, key endpoints work | HIGH |
| `test_playlist_generation_e2e.py` | Generate playlist with fixtures | MEDIUM |
| `test_config_loader.py` | Config parsing, validation | MEDIUM |
| `test_string_utils.py` | String normalization | LOW |

## Minimal CI Plan

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - name: Unit Tests
        run: pytest tests/unit/ -v
      - name: Integration Tests
        run: pytest tests/integration/ -v
      - name: Smoke Tests
        run: pytest tests/smoke/ -v

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install ruff
      - run: ruff check src/ api/ scripts/
```

---

# Phased Execution Plan

## Phase 0: Safety Nets

**Goal:** Establish baselines before any changes

### Steps

1. **Snapshot current tree**
   ```bash
   git status > CLEANUP_BASELINE_STATUS.txt
   find . -type f -name "*.py" | wc -l >> CLEANUP_BASELINE_STATUS.txt
   git log --oneline -1 >> CLEANUP_BASELINE_STATUS.txt
   ```

2. **Verify smoke commands work**
   ```bash
   # These must all succeed before proceeding
   python -c "from src.playlist_generator import PlaylistGenerator; print('OK')"
   python -c "from api.main import app; print('OK')"
   python scripts/analyze_library.py --dry-run
   python main_app.py --help
   pytest tests/ --collect-only
   ```

3. **Create archive branch**
   ```bash
   git checkout -b archive/pre-cleanup-snapshot
   git checkout main
   ```

### Validation
- All smoke commands pass
- Git status captured
- Archive branch created

### Commit Boundary
- None (no changes yet)

---

## Phase 1: Non-Breaking Reorganization

**Goal:** Move/rename without breaking imports

### Step 1.1: Create archive/ structure

```bash
mkdir -p archive/experiments
mkdir -p archive/diagnostics
mkdir -p archive/legacy_scripts
mkdir -p archive/legacy_tests
mkdir -p archive/session_artifacts
```

**Commit:** `chore: create archive directory structure`

### Step 1.2: Move experiments/

```bash
mv experiments/genre_similarity_lab archive/experiments/
mv experiments/*.md archive/experiments/
rmdir experiments
```

**Commit:** `chore: archive experiments directory`

### Step 1.3: Move diagnostics/

```bash
mv diagnostics/* archive/diagnostics/
rmdir diagnostics
echo "diagnostics/" >> .gitignore
```

**Commit:** `chore: archive diagnostics and gitignore future outputs`

### Step 1.4: Move discogs-genre-test/

```bash
mv discogs-genre-test archive/experiments/
```

**Commit:** `chore: archive discogs-genre-test`

### Step 1.5: Archive legacy scripts

```bash
# Move diagnostic/experimental scripts
mv scripts/diagnose_*.py archive/legacy_scripts/
mv scripts/ab_playlist_sonic_variants.py archive/legacy_scripts/
mv scripts/batch_eval_sonic_variant.py archive/legacy_scripts/
mv scripts/sonic_separability.py archive/legacy_scripts/
mv scripts/tune_dial_grid.py archive/legacy_scripts/
mv scripts/infer_sonic_feature_schema.py archive/legacy_scripts/
mv scripts/build_track_group.py archive/legacy_scripts/
mv scripts/make_pairs_file.py archive/legacy_scripts/
mv scripts/debug_edge_scores.py archive/legacy_scripts/
mv scripts/purge_lastfm_genres.py archive/legacy_scripts/
mv scripts/update_discogs_genres.py archive/legacy_scripts/
mv scripts/backfill_duration.py archive/legacy_scripts/
```

**Commit:** `chore: archive legacy/experimental scripts`

### Step 1.6: Archive legacy tests

```bash
mkdir -p tests/archive
mv tests/test_batch_eval_sonic_variant.py tests/archive/
mv tests/test_diagnose_*.py tests/archive/
mv tests/test_sonic_separability.py tests/archive/
mv tests/test_infer_sonic_feature_schema.py tests/archive/
mv tests/test_dial_grid_tuning.py tests/archive/
```

**Commit:** `chore: archive legacy tests`

### Step 1.7: Archive root session artifacts

```bash
mv CODE_CLEANUP_AND_OPTIMIZATION.md archive/session_artifacts/
mv DIAL_GRID_FIX_SUMMARY.md archive/session_artifacts/
mv SCAN_LIBRARY_IMPROVEMENTS_SUMMARY.txt archive/session_artifacts/
mv REPOSITORY_STRUCTURE.md archive/session_artifacts/
```

**Commit:** `chore: archive session artifacts from root`

### Step 1.8: Clean up root

```bash
# Delete duplicate/generated files
rm -f metadata.db  # Keep data/metadata.db
rm -f discogs_misses.tsv
rm -f *.log  # playlist_generator.log, sonic_analysis.log, genre_update_v3.log

# Update .gitignore
cat >> .gitignore << 'EOF'
# Logs
*.log

# Generated data
diagnostics/
*.tsv

# Root-level DB (use data/metadata.db)
/metadata.db
EOF
```

**Commit:** `chore: clean root directory, update .gitignore`

### Step 1.9: Organize tests into taxonomy

```bash
mkdir -p tests/unit tests/integration tests/smoke

# Unit tests (fast, no fixtures)
mv tests/test_title_dedupe.py tests/unit/
mv tests/test_duration_similarity.py tests/unit/
mv tests/test_constraints.py tests/unit/

# Integration tests (need fixtures)
mv tests/test_artifacts.py tests/integration/
mv tests/test_ds_pipeline_determinism.py tests/integration/
mv tests/test_pipeline_switch.py tests/integration/
mv tests/test_sonic_variant*.py tests/integration/
mv tests/test_run_artifact.py tests/integration/
mv tests/test_analyze_library_cli.py tests/integration/
mv tests/test_ds_pool_logging.py tests/integration/

# Keep conftest.py at root of tests/
```

**Commit:** `refactor: organize tests into unit/integration/smoke taxonomy`

### Validation (Phase 1)

```bash
# All imports still work
python -c "from src.playlist_generator import PlaylistGenerator; print('OK')"
python -c "from api.main import app; print('OK')"

# Scripts still run
python scripts/analyze_library.py --dry-run

# Tests still pass
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Broken imports after moving files | Only moving to archive/, not production code |
| Tests fail after reorganization | Move tests carefully, update conftest if needed |
| Lost history | Git mv preserves history; archive branch is backup |

---

## Phase 2: Documentation Overhaul

**Goal:** Current, accurate, newcomer-friendly docs

### Step 2.1: Create docs structure

```bash
mkdir -p docs/reference
```

### Step 2.2: Write core docs

Create/update in order:
1. `docs/quickstart.md`
2. `docs/architecture.md`
3. `docs/data_model.md`
4. `docs/pipelines.md`
5. `docs/playlist_generation.md`
6. `docs/api.md`
7. `docs/dev.md`
8. `docs/configuration.md`

**Commit per doc:** `docs: add {docname}.md`

### Step 2.3: Move reference docs

```bash
mv docs/GENRE_SIMILARITY_SYSTEM.md docs/reference/genre_similarity.md
mv docs/GENRE_SIMILARITY_METHODS.md docs/reference/genre_similarity_methods.md
mv docs/TITLE_DEDUPLICATION.md docs/reference/title_deduplication.md
mv docs/DURATION_SUPPORT.md docs/reference/duration_support.md
mv docs/SCAN_LIBRARY_ANALYSIS.md docs/reference/scan_library.md
mv docs/TUNING_WORKFLOW.md docs/reference/tuning.md
```

**Commit:** `docs: reorganize reference documentation`

### Step 2.4: Archive stale docs

```bash
mv docs/INTEGRATION_COMPLETE.md archive/session_artifacts/
mv docs/ds_pipeline_integration_plan.md archive/session_artifacts/
mv docs/session_notes archive/session_artifacts/
rm docs/NEW_LOG_MOCKUP.log
```

**Commit:** `docs: archive stale documentation`

### Step 2.5: Rewrite README.md

Replace with concise quickstart + links to docs/

**Commit:** `docs: rewrite README.md`

### Step 2.6: Create CHANGELOG.md

**Commit:** `docs: add CHANGELOG.md`

### Step 2.7: Update DEV.md → docs/dev.md

Merge DEV.md content into docs/dev.md, delete root DEV.md

**Commit:** `docs: consolidate DEV.md into docs/dev.md`

### Validation (Phase 2)

```bash
# Docs render correctly (if using mkdocs or similar)
# All links in README work
# quickstart steps execute successfully
```

---

## Phase 3: Test Cleanup & CI

**Goal:** Reliable test suite with CI

### Step 3.1: Add missing smoke tests

Create `tests/smoke/test_cli_smoke.py`:

```python
import subprocess
import sys

def test_main_app_help():
    result = subprocess.run([sys.executable, "main_app.py", "--help"], capture_output=True)
    assert result.returncode == 0

def test_analyze_library_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/analyze_library.py", "--dry-run"],
        capture_output=True
    )
    assert result.returncode == 0

def test_api_import():
    result = subprocess.run(
        [sys.executable, "-c", "from api.main import app"],
        capture_output=True
    )
    assert result.returncode == 0
```

**Commit:** `test: add CLI smoke tests`

### Step 3.2: Fix any broken tests

Run full suite, fix failures:

```bash
pytest tests/ -v --tb=short
```

**Commit:** `test: fix broken tests after reorganization`

### Step 3.3: Add CI workflow

Create `.github/workflows/ci.yml` (see earlier in this doc)

**Commit:** `ci: add GitHub Actions workflow`

### Validation (Phase 3)

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/smoke/ -v
# All should pass
```

---

## Phase 4: Remove Candidates (Post-Validation)

**Goal:** Final cleanup after everything is validated

### Step 4.1: Delete deprecated shim

After confirming nothing imports it:

```bash
grep -r "from.*genre_similarity import" src/ api/ scripts/
# If empty or only imports GenreSimilarityV2:
rm src/genre_similarity.py
```

**Commit:** `chore: remove deprecated genre_similarity.py shim`

### Step 4.2: Rename genre_similarity_v2.py (optional)

```bash
# If desired, drop the _v2 suffix
mv src/genre_similarity_v2.py src/genre_similarity.py
# Update all imports
```

**Commit:** `refactor: rename genre_similarity_v2 to genre_similarity`

### Step 4.3: Clean archive if confident

After 1+ month, consider:
- Deleting `archive/legacy_tests/`
- Deleting `archive/legacy_scripts/`
- Keeping `archive/experiments/` for reference

**Commit:** `chore: prune archive (optional)`

### Validation (Phase 4)

```bash
# Full test suite
pytest tests/ -v

# Smoke commands
python main_app.py --help
python scripts/analyze_library.py --dry-run
uvicorn api.main:app --help
```

---

# Summary: One-Command Happy Paths

After cleanup, these commands should work:

### (a) Analyze Library
```bash
python scripts/analyze_library.py
# Or with options:
python scripts/analyze_library.py --stages scan,sonic,artifacts --workers 4
```

### (b) Generate Playlist
```bash
python main_app.py
# Or:
python main_app.py --pipeline ds --count 5
```

### (c) Run API/Server
```bash
uvicorn api.main:app --reload --port 8000
# Or:
python -m api
```

### (d) Run UI (Development)
```bash
cd ui && npm run dev
```

---

# Acceptance Checklist

- [ ] Newcomer can run: install → scan → generate → API in few commands
- [ ] Clear separation: `src/` (library) vs `archive/` (legacy)
- [ ] Docs describe current behavior (not aspirational)
- [ ] Repo root is clean (≤10 files)
- [ ] UI integration point obvious (`api/` endpoints documented)
- [ ] Test suite organized (unit/integration/smoke)
- [ ] CI validates PRs
- [ ] No broken imports
- [ ] All smoke commands pass
