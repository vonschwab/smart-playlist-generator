# Repository Cleanup - Execution Plan

**Start Date**: 2025-12-16 (While Phase 2 rebuild running in background)
**Phases**: 0-4 (Sequential)
**Timeline**: ~2-3 hours total

---

## QUICK START: What We're Doing

The repo has accumulated experimental code, session artifacts, and legacy scripts. We're reorganizing into:

```
Clean structure:
├── src/                 → Production library code
├── api/                 → FastAPI backend
├── scripts/             → Essential operational scripts only
├── tests/               → Organized: unit/ integration/ smoke/
├── docs/                → Current, consolidated documentation
├── data/                → Runtime data (gitignored except templates)
├── archive/             → Legacy code (moved, not deleted)
└── [clean root]         → Only essential files
```

**Total effect**: More professional, easier to onboard, clear boundaries.

---

## PHASE 0: Safety Nets ✅ READY TO START

**Goal**: Establish baselines before changes

### 0.1: Capture current state
```bash
git status > CLEANUP_BASELINE_STATUS.txt
find . -type f -name "*.py" | wc -l >> CLEANUP_BASELINE_STATUS.txt
git log --oneline -1 >> CLEANUP_BASELINE_STATUS.txt
```

### 0.2: Verify smoke commands
```bash
# These must all pass
python -c "from src.playlist_generator import PlaylistGenerator; print('OK')"
python -c "from api.main import app; print('OK')"
python main_app.py --help
pytest tests/ --collect-only
```

### 0.3: Create archive branch
```bash
git checkout -b archive/pre-cleanup-snapshot
git checkout master
```

**Status**: ⏳ Ready to execute

---

## PHASE 1: Non-Breaking Reorganization ✅ MAIN WORK

**Goal**: Move/archive without breaking imports

### 1.1: Create archive/ structure
```bash
mkdir -p archive/experiments
mkdir -p archive/diagnostics
mkdir -p archive/legacy_scripts
mkdir -p archive/legacy_tests
mkdir -p archive/session_artifacts
touch archive/README.md
```

**Time**: ~1 min

### 1.2: Move experiments/
```bash
mv experiments/genre_similarity_lab archive/experiments/
mv experiments/*.md archive/experiments/
rmdir experiments 2>/dev/null || true
```

**Commit**: `chore: archive experiments directory`

### 1.3: Move diagnostics/
```bash
mv diagnostics/* archive/diagnostics/ 2>/dev/null || true
rmdir diagnostics 2>/dev/null || true
echo "diagnostics/" >> .gitignore
```

**Commit**: `chore: archive diagnostics and gitignore future outputs`

### 1.4: Move discogs-genre-test/
```bash
mv discogs-genre-test archive/experiments/ 2>/dev/null || true
```

**Commit**: `chore: archive discogs-genre-test`

### 1.5: Archive legacy scripts (13 files)
```bash
mv scripts/diagnose_*.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/ab_playlist_sonic_variants.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/batch_eval_sonic_variant.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/sonic_separability.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/tune_dial_grid.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/infer_sonic_feature_schema.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/build_track_group.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/make_pairs_file.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/debug_edge_scores.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/purge_lastfm_genres.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/update_discogs_genres.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/backfill_duration.py archive/legacy_scripts/ 2>/dev/null || true
mv scripts/refresh_effective_genres.py archive/legacy_scripts/ 2>/dev/null || true
```

**Commit**: `chore: archive legacy/experimental scripts`

**Time**: ~5 min

### 1.6: Archive legacy tests (6 files)
```bash
mkdir -p tests/archive
mv tests/test_batch_eval_sonic_variant.py tests/archive/ 2>/dev/null || true
mv tests/test_diagnose_*.py tests/archive/ 2>/dev/null || true
mv tests/test_sonic_separability.py tests/archive/ 2>/dev/null || true
mv tests/test_infer_sonic_feature_schema.py tests/archive/ 2>/dev/null || true
mv tests/test_dial_grid_tuning.py tests/archive/ 2>/dev/null || true
mv tests/test_ds_pool_logging.py tests/archive/ 2>/dev/null || true
```

**Commit**: `chore: archive legacy tests`

### 1.7: Archive root session artifacts (4 files)
```bash
mv CODE_CLEANUP_AND_OPTIMIZATION.md archive/session_artifacts/ 2>/dev/null || true
mv DIAL_GRID_FIX_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv SCAN_LIBRARY_IMPROVEMENTS_SUMMARY.txt archive/session_artifacts/ 2>/dev/null || true
mv REPOSITORY_STRUCTURE.md archive/session_artifacts/ 2>/dev/null || true
mv IMPLEMENTATION_COMPLETE.md archive/session_artifacts/ 2>/dev/null || true
mv IMPLEMENTATION_CHANGES_REFERENCE.txt archive/session_artifacts/ 2>/dev/null || true
mv NEXT_STEPS.md archive/session_artifacts/ 2>/dev/null || true
mv DIALS_IMPLEMENTATION_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv DIALS_IMPLEMENTATION_VERIFICATION.md archive/session_artifacts/ 2>/dev/null || true
mv DIAL_GRID_FIX_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv DIAL_INVESTIGATION_EXECUTIVE_SUMMARY.txt archive/session_artifacts/ 2>/dev/null || true
mv DIAL_INVESTIGATION_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv DIAL_ROUTING_ANALYSIS.md archive/session_artifacts/ 2>/dev/null || true
mv GENRE_ENFORCEMENT_QUICK_START.md archive/session_artifacts/ 2>/dev/null || true
mv GENRE_ENFORCEMENT_TEST_RESULTS.md archive/session_artifacts/ 2>/dev/null || true
mv GENRE_SYSTEM_COMPLETE_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv MINIMAL_FIX_DIFF.md archive/session_artifacts/ 2>/dev/null || true
mv PHASE_AB_DIAGNOSTIC_SUMMARY.md archive/session_artifacts/ 2>/dev/null || true
mv QUICK_START_DIALS_VERIFICATION.txt archive/session_artifacts/ 2>/dev/null || true
mv README_SONIC_FIX.md archive/session_artifacts/ 2>/dev/null || true
```

**Commit**: `chore: archive session artifacts from root`

**Time**: ~2 min

### 1.8: Clean root directory
```bash
rm -f metadata.db 2>/dev/null || true
rm -f discogs_misses.tsv 2>/dev/null || true
rm -f *.log 2>/dev/null || true

# Update .gitignore
cat >> .gitignore << 'EOF'

# Logs and generated files
*.log
*.tsv
diagnostics/

# Root-level DB (use data/metadata.db)
/metadata.db
EOF
```

**Commit**: `chore: clean root directory, update .gitignore`

**Time**: ~2 min

### 1.9: Organize tests into taxonomy
```bash
mkdir -p tests/unit tests/integration tests/smoke

# Unit tests (fast, no fixtures)
mv tests/test_title_dedupe.py tests/unit/ 2>/dev/null || true
mv tests/test_duration_similarity.py tests/unit/ 2>/dev/null || true
mv tests/test_constraints.py tests/unit/ 2>/dev/null || true

# Integration tests (need fixtures/DB)
mv tests/test_artifacts.py tests/integration/ 2>/dev/null || true
mv tests/test_ds_pipeline_determinism.py tests/integration/ 2>/dev/null || true
mv tests/test_pipeline_switch.py tests/integration/ 2>/dev/null || true
mv tests/test_sonic_variant*.py tests/integration/ 2>/dev/null || true
mv tests/test_run_artifact.py tests/integration/ 2>/dev/null || true
mv tests/test_analyze_library_cli.py tests/integration/ 2>/dev/null || true

# Smoke tests (CLI validation)
# (We'll create these in Phase 3)
```

**Commit**: `refactor: organize tests into unit/integration/smoke taxonomy`

**Time**: ~3 min

### 1.10: Validation after Phase 1
```bash
# Verify imports still work
python -c "from src.playlist_generator import PlaylistGenerator; print('✓ PlaylistGenerator')"
python -c "from api.main import app; print('✓ API app')"

# Verify key scripts
python scripts/analyze_library.py --dry-run 2>&1 | head -1
python main_app.py --help 2>&1 | head -1

# Run tests
pytest tests/unit/ -v --tb=short 2>&1 | tail -3
```

**Expected output**: All green ✓

**Time to complete Phase 1**: ~15-20 minutes

---

## PHASE 2: Documentation Overhaul ✅ NEXT

**Goal**: Current, accurate, user-friendly docs

### 2.1: Create docs structure
```bash
mkdir -p docs/reference
```

### 2.2: Write/update core docs
**Order** (why order matters: later docs link to earlier ones):

1. **docs/quickstart.md** (Installation + first playlist)
   - Prerequisites
   - Install steps
   - First run (scan → generate → export)
   - Troubleshooting

2. **docs/architecture.md** (System overview)
   - ASCII diagram
   - Component descriptions
   - Data flow

3. **docs/data_model.md** (Database schema)
   - SQLite tables
   - NPZ artifact format
   - Key relationships

4. **docs/pipelines.md** (How to use scripts)
   - Unified pipeline (analyze_library.py)
   - Individual scripts
   - Workflows

5. **docs/playlist_generation.md** (Playlist details)
   - Modes, scoring, constraints
   - Deduplication, output

6. **docs/api.md** (API reference)
   - Endpoints
   - Examples
   - Error handling

7. **docs/configuration.md** (All config options)
   - What each setting does
   - Examples

8. **docs/dev.md** (For developers)
   - Setup, testing, debugging
   - Architecture for code changes

### 2.3: Move reference docs
```bash
mkdir -p docs/reference
mv docs/GENRE_SIMILARITY_SYSTEM.md docs/reference/genre_similarity.md 2>/dev/null || true
mv docs/GENRE_SIMILARITY_METHODS.md docs/reference/genre_similarity_methods.md 2>/dev/null || true
mv docs/TITLE_DEDUPLICATION.md docs/reference/title_deduplication.md 2>/dev/null || true
mv docs/DURATION_SUPPORT.md docs/reference/duration_support.md 2>/dev/null || true
mv docs/SCAN_LIBRARY_ANALYSIS.md docs/reference/scan_library.md 2>/dev/null || true
mv docs/TUNING_WORKFLOW.md docs/reference/tuning.md 2>/dev/null || true
```

**Commit**: `docs: reorganize reference documentation`

### 2.4: Archive stale docs
```bash
mv docs/INTEGRATION_COMPLETE.md archive/session_artifacts/ 2>/dev/null || true
mv docs/ds_pipeline_integration_plan.md archive/session_artifacts/ 2>/dev/null || true
rm -rf docs/session_notes 2>/dev/null || true
rm docs/NEW_LOG_MOCKUP.log 2>/dev/null || true
```

**Commit**: `docs: archive stale documentation`

### 2.5: Rewrite README.md
```markdown
# Playlist Generator

AI-powered playlist generation using sonic analysis and genre metadata.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set up config
cp config.example.yaml config.yaml

# 3. Scan your music library
python scripts/analyze_library.py

# 4. Generate a playlist
python main_app.py --count 50

# 5. Run the API (for UI)
uvicorn api.main:app --reload
```

## Documentation

- [Quickstart Guide](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration](docs/configuration.md)
- [Development Guide](docs/dev.md)

## Project Status

✅ Core: Stable
✅ API: Stable
🔄 UI: In Development
```

**Commit**: `docs: rewrite README.md`

### 2.6: Create CHANGELOG.md
```markdown
# Changelog

## [Unreleased]
- Repository cleanup and reorganization
- Documentation overhaul
- Test suite organization

## [1.0.0] - 2025-12-16
- Initial stable release
- DS pipeline as default
- Title deduplication
- Duration filtering
- FastAPI backend
```

**Commit**: `docs: add CHANGELOG.md`

### 2.7: Consolidate DEV.md
Move DEV.md content → docs/dev.md, delete root DEV.md

```bash
mv DEV.md archive/session_artifacts/ 2>/dev/null || true
```

**Commit**: `docs: consolidate DEV.md into docs/dev.md`

**Time to complete Phase 2**: ~45-60 minutes (most time spent writing docs)

---

## PHASE 3: Test Cleanup & CI ✅ FUTURE

**Goal**: Reliable test suite with CI

### 3.1: Add smoke tests
Create `tests/smoke/test_cli_smoke.py` with basic CLI validation

### 3.2: Fix broken tests
Run full suite, fix failures

### 3.3: Add GitHub Actions CI
Create `.github/workflows/ci.yml` for automated testing

**Time to complete Phase 3**: ~20-30 minutes

---

## PHASE 4: Final Removals ✅ LATER

**Goal**: Clean up confirmed obsolete code

### 4.1: Delete deprecated shims
After confirming no imports of `src/genre_similarity.py`

### 4.2: Optionally rename genre_similarity_v2.py
Drop the `_v2` suffix if desired

### 4.3: Prune archive after 1 month
Keep `experiments/`, delete `legacy_scripts/` if confident

**Time to complete Phase 4**: ~10 minutes

---

## EXECUTION ORDER (Tonight)

### Recommended order (minimal risk):

**Now (Next 20 min)**:
1. ✅ Phase 0: Safety nets (baseline, branch)
2. ✅ Phase 1.1-1.8: Archive + reorganize (15 min, 7 commits)
3. ✅ Phase 1.9-1.10: Organize tests + validate (5 min, 1 commit)

**Later (45-60 min)**:
4. Phase 2: Documentation (write core docs, 8 commits)

**After rebuild completes (30 min)**:
5. Phase 3: Test CI setup (3 commits)

**After everything validated (10 min)**:
6. Phase 4: Final removals (2-3 commits)

---

## Git Commit Summary

After all phases, you'll have ~**25 clean commits** instead of monolithic PRs:

```
Phase 1:
- archive directory structure
- experiments → archive
- diagnostics → archive
- legacy scripts → archive
- legacy tests → archive
- session artifacts → archive
- root cleanup
- test taxonomy reorganization

Phase 2:
- quickstart doc
- architecture doc
- data model doc
- pipelines doc
- playlist generation doc
- api doc
- dev doc
- reference docs reorganization
- stale docs archive
- README rewrite
- CHANGELOG

Phase 3:
- smoke tests
- test fixes
- CI workflow

Phase 4:
- remove deprecated shims
- archive pruning
```

---

## Success Checklist

After all phases:
- [ ] Root directory has ≤12 essential files
- [ ] `archive/` contains all legacy code
- [ ] Tests organized: `unit/` `integration/` `smoke/`
- [ ] Core docs complete: quickstart, architecture, API
- [ ] README clear and concise
- [ ] All imports still work
- [ ] `pytest tests/unit/ -v` passes
- [ ] `pytest tests/integration/ -v` passes
- [ ] CLI commands work: `analyze_library.py`, `main_app.py`, `uvicorn api.main:app`
- [ ] No broken symlinks or references
- [ ] Git history clean (one commit per logical change)

---

## Starting Now: Phase 0

Ready to proceed? Let's start:

```bash
cd "C:\Users\Dylan\Desktop\PLAYLIST GENERATOR"
git status > CLEANUP_BASELINE_STATUS.txt
cat CLEANUP_BASELINE_STATUS.txt
```

Then we'll proceed through Phase 1 step-by-step.

