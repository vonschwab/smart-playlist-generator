# Changelog - Version 3.2

## Recent Updates (2026-01-02)

This changelog documents the major features, fixes, and improvements implemented in version 3.2.

---

## 🎵 New Feature: Genre Mode

### Overview
Added comprehensive genre-based playlist generation with intelligent autocomplete and similarity matching.

### Features
- **Genre Mode in GUI** - New dropdown option alongside Artist and History modes
- **Smart Autocomplete** - Shows both exact genre matches AND similar genres (similarity ≥ 0.7)
  - Type "ambient" → see "ambient", "drone (similar)", "downtempo (similar)", etc.
  - Limited to 15 suggestions for optimal UX
- **Accent-Insensitive Matching** - Works seamlessly with the existing genre database
- **Pier-Bridge Integration** - Genre playlists use 4 random seeds from the genre as anchor points
- **Configurable Weighting** - Uses sonic_weight=0.60, genre_weight=0.50 for balanced results

### Implementation
- **CLI Support**: `python main_app.py --genre "ambient" --tracks 30`
- **GUI Integration**: Full support in main window with persistent settings
- **Database Ready**: Works with existing track_effective_genres table

### Files Modified
- `src/playlist_gui/main_window.py` - Genre mode UI and validation
- `src/playlist_gui/autocomplete.py` - GenreSimilarityCompleter class
- `src/playlist_gui/worker.py` - Genre mode routing
- `src/playlist_gui/worker_client.py` - Genre parameter support
- `src/playlist_generator.py` - create_playlist_for_genre() method

### Documentation
- [GENRE_MODE_DESIGN.md](docs/GENRE_MODE_DESIGN.md) - Technical design
- [GENRE_MODE_GUI_IMPLEMENTATION.md](docs/GENRE_MODE_GUI_IMPLEMENTATION.md) - GUI implementation details
- [GENRE_MODE_AUDIT.md](docs/GENRE_MODE_AUDIT.md) - Testing and verification

---

## 🔧 Major Fixes

### 1. Compound Genres Data Quality Fix

**Problem:** Genre autocomplete showed compound strings like "indie rock, alternative" instead of clean individual genres.

**Root Cause:** Database contained 12,256 compound genre entries (7% of all data) that were never atomized:
- 2,912 with commas
- 525 with semicolons
- 3,551 with slashes
- 5,268 with ampersands

**Solution:**
- Created `scripts/fix_compound_genres.py` to atomize genres at the source
- Used existing `normalize_genre_list()` infrastructure (no redundant code)
- Properly handles ampersands: "R&B" → "r and b" as a single genre
- Executed in two rounds, processing all separators

**Results:**
- ✅ 0 compound genres remaining
- ✅ 746 unique atomized genres
- ✅ Clean autocomplete experience
- ✅ Net change: +4,000 atomized genre entries

**Files:**
- `scripts/fix_compound_genres.py` - Created
- `src/playlist_gui/autocomplete.py` - Reverted temporary parsing code
- `data/metadata.db` - Atomized all compound genres

**Documentation:** [COMPOUND_GENRES_FIX.md](docs/COMPOUND_GENRES_FIX.md)

---

### 2. Pier Seed Enforcement Fix

**Problem:** Genre playlists failed with "Out-of-pool tracks detected" error when pier seeds weren't in the allowed pool.

**Root Cause:** Pier seeds were added to `allowed_indices` (for generation) but NOT to `allowed_track_ids_set` (for enforcement check).

**Solution:** Added one line in `src/playlist/pipeline.py:174`:
```python
allowed_track_ids_set.update(exempt_ids)
```

**Impact:**
- ✅ Genre playlists now generate successfully
- ✅ Pier seeds correctly exempted from enforcement
- ✅ Final playlist includes all anchor tracks as intended

**Files Modified:**
- `src/playlist/pipeline.py` - Line 174 added

**Documentation:** [docs/run_audits/PIER_SEED_ENFORCEMENT_FIX.md](docs/run_audits/PIER_SEED_ENFORCEMENT_FIX.md)

---

### 3. Run All Button Fixes

**Problem:** "Run All" button in GUI failed on all four maintenance operations with interface mismatches.

**Errors:**
1. Library Scan: `AttributeError: 'LibraryScanner' object has no attribute 'scan'`
2. Genre Update: `ImportError: cannot import name 'update_genres_main'`
3. Sonic Analysis: `ImportError: cannot import name 'main'`
4. Artifact Build: `TypeError: main() got an unexpected keyword argument 'db_path'`

**Solution:** Fixed all four worker handlers to match actual script interfaces:

1. **Library Scanner** - Changed `scanner.scan()` → `scanner.run()`
2. **Genre Updater** - Instantiate `NormalizedGenreUpdater` class and call methods
3. **Sonic Analyzer** - Instantiate `SonicFeaturePipeline` class and call `run()`
4. **Artifact Builder** - Create `argparse.Namespace` object with proper attributes

**Results:**
- ✅ All four operations execute successfully
- ✅ Progress bars update correctly
- ✅ Detailed summary statistics shown
- ✅ One-click pipeline: Scan → Genres → Sonic → Artifacts

**Files Modified:**
- `src/playlist_gui/worker.py` - Lines 723, 757-772, 812-824, 858-880

**Documentation:** [docs/run_audits/RUN_ALL_INTERFACE_FIX.md](docs/run_audits/RUN_ALL_INTERFACE_FIX.md)

---

## 📝 Documentation Updates

### Terminology Correction
**Changed:** "AI-powered" → "Data Science-powered"
**Reason:** This system uses data science techniques (similarity computation, beam search, PCA, normalization) but does NOT call AI/LLM APIs during playlist generation.

**Files Updated:**
- `README.md` - Main project description
- `src/__init__.py` - Package docstring
- `src/playlist_generator.py` - Module docstring
- `main_app.py` - Application header
- `src/playlist_gui/main_window.py` - About dialog
- `src/playlist_gui/README.md` - GUI description

### Enhanced Documentation
- **Genre Mode Coverage** - Added to README.md, GOLDEN_COMMANDS.md
- **GUI Highlights** - Updated with genre mode, atomized genres, Run All button
- **Version Updates** - Bumped to v3.2 in About dialog

---

## 🎯 System Improvements

### Data Quality
- ✅ **Atomized Genres** - All 746 genres properly normalized
- ✅ **No Compound Strings** - Clean genre taxonomy for accurate matching
- ✅ **Consistent Normalization** - Ampersands preserved for legitimate genres (R&B, Drum & Bass)

### User Experience
- ✅ **Genre Autocomplete** - Smart suggestions with similarity matching
- ✅ **One-Click Maintenance** - Run All button executes full pipeline
- ✅ **Accent Tolerance** - Type "Joao" to find "João Gilberto"
- ✅ **Progress Tracking** - Real-time updates for all operations

### Code Quality
- ✅ **Interface Consistency** - Worker properly calls all script methods
- ✅ **Error Handling** - Pier seeds exempted from pool enforcement
- ✅ **No Redundant Code** - Genre fixing uses existing normalization infrastructure

---

## 🔍 Technical Details

### Genre Playlist Configuration

**Sonic/Genre Weights:**
```yaml
genre_similarity:
  sonic_weight: 0.60    # 60% sonic similarity
  genre_weight: 0.50    # 50% genre similarity
  min_genre_similarity: 0.30  # Minimum genre match threshold
  method: ensemble      # Uses all similarity methods
```

**Pier-Bridge Behavior:**
- Selects 4 random seeds from genre as anchors
- Uses pier-bridge algorithm with same settings as artist mode
- Applies genre filtering to candidate pool
- Maintains smooth sonic transitions between diverse tracks

### Database Schema

**Genre Tables:**
- `track_genres` - Raw genres from file tags, MusicBrainz, Discogs
- `track_effective_genres` - Normalized, atomized genres (one per row)
- `artist_genres` - Artist-level genre assignments
- `album_genres` - Album-level genre assignments

**Genre Normalization:**
- Separators: commas, semicolons, forward slashes
- Ampersands: Converted to "and" (R&B → "r and b")
- Broad filters: Removes years, decades, meta tags
- Case normalization: Lowercase with whitespace trimming

---

## 🐛 Bug Fixes Summary

| Issue | Impact | Status |
|-------|--------|--------|
| Compound genres in autocomplete | High | ✅ Fixed |
| Pier seed enforcement failure | High | ✅ Fixed |
| Run All button failures | High | ✅ Fixed |
| AI terminology in docs | Low | ✅ Fixed |

---

## 📊 Statistics

### Genre Data Quality
- **Before:** 12,256 compound entries (7% of data)
- **After:** 0 compound entries
- **Unique Genres:** 746 (down from 844 due to deduplication)
- **Total Genre Entries:** ~253,000 (up from ~249,000)

### Code Changes
- **Files Modified:** 12
- **Lines Added:** ~850
- **Lines Removed:** ~120
- **Net Change:** +730 lines
- **New Scripts:** 1 (`fix_compound_genres.py`)
- **Documentation Files:** 7 created/updated

---

## 🚀 Migration Notes

### For Existing Users

**Genre data cleanup (one-time):**
```bash
# Atomize any remaining compound genres
python scripts/fix_compound_genres.py --apply
```

**Rebuild artifacts (if needed):**
```bash
# After genre cleanup, rebuild to incorporate changes
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz
```

**No breaking changes:**
- ✅ Existing playlists still work
- ✅ All CLI commands unchanged
- ✅ Config files compatible
- ✅ Database schema unchanged (only data cleaned)

---

## 🔮 Future Enhancements

### Planned Features
- Genre combinations (e.g., "ambient + electronic")
- Genre exclusion filters
- Time-based genre evolution tracking
- Genre mood mapping

### Under Consideration
- Hybrid mode (artist + genre constraints)
- Genre blend playlists (smooth transition between genres)
- Custom genre taxonomy editor

---

## 👥 Contributors

All features and fixes implemented by the development team.

Special thanks to users for reporting:
- Compound genre autocomplete issues
- Genre playlist generation errors
- Run All button failures

---

## 📞 Support

For issues or questions:
- Check [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Review [GOLDEN_COMMANDS.md](docs/GOLDEN_COMMANDS.md)
- Run `python tools/doctor.py` to verify setup
- Check logs in `%APPDATA%\PlaylistGenerator\logs\`

---

## Version History

- **v3.2** (2026-01-02) - Genre mode, data quality fixes, Run All fixes
- **v3.1** - MBID enrichment, GUI improvements
- **v3.0** - Beat3tower sonic analysis, pier-bridge ordering
- **v2.x** - Legacy sonic analysis
- **v1.x** - Initial release

---

**Release Date:** 2026-01-02
**Status:** Stable
**Compatibility:** Windows 10/11, Python 3.8+
