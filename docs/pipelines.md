# Pipelines & Workflows

Detailed guide to the data processing pipelines.

## Unified Pipeline: `analyze_library.py`

Main orchestration script that runs all stages in sequence.

### Usage

```bash
# Full pipeline (scan → genres → sonic → artifacts)
python scripts/analyze_library.py

# Run specific stages only
python scripts/analyze_library.py --stages sonic,artifacts

# Rebuild everything (force mode)
python scripts/analyze_library.py --force

# Debug with dry-run
python scripts/analyze_library.py --dry-run

# Use custom database
python scripts/analyze_library.py --db /path/to/other.db
```

### Stages

| Stage | Script | Purpose | Time |
|-------|--------|---------|------|
| `scan` | `scan_library.py` | Discover files, extract basic metadata | 5-10 min/1k |
| `genres` | `update_genres_v3_normalized.py` | Fetch + normalize genres | 30-60 min/1k |
| `sonic` | `update_sonic.py` | Extract audio features | 60+ min/1k |
| `artifacts` | `analyze_library.py` (internal) | Build NPZ matrices | 5-10 min |

### Command Examples

```bash
# Fastest: Only missing features (incremental)
python scripts/analyze_library.py

# Rebuild everything from scratch
python scripts/analyze_library.py --force

# Stop at audio features (don't build artifacts yet)
python scripts/analyze_library.py --stages scan,genres,sonic

# Use 16 parallel workers for sonic analysis
python scripts/analyze_library.py --stages sonic --workers 16

# Analyze only first 1000 tracks
python scripts/analyze_library.py --limit 1000

# Force rebuild, but only analyze genres
python scripts/analyze_library.py --stages genres --force
```

## Individual Pipelines

### 1. Library Scanning

```bash
python scripts/scan_library.py
```

**What it does**:
1. Recursively finds all audio files in library
2. Extracts metadata (title, artist, album, duration)
3. Stores in database

**Supported formats**: `.mp3`, `.flac`, `.m4a`, `.ogg`, `.wav`

**Output**: `data/metadata.db` with `tracks` table

### 2. Genre Fetching

```bash
python scripts/update_genres_v3_normalized.py --workers 4
```

**What it does**:
1. Queries Last.fm API for genres
2. Falls back to MusicBrainz if needed
3. Normalizes and deduplicates genres
4. Computes genre co-occurrence matrix
5. Builds smoothed genre vectors

**Output**: `data/metadata.db` with genres, `data/genre_similarity.yaml`

**API Keys Required** (optional):
```yaml
# In config.yaml
lastfm:
  api_key: "your_key_here"
  username: "optional"

musicbrainz:
  enabled: true
```

### 3. Sonic Analysis

```bash
# Windowed mode (legacy)
python scripts/analyze_library.py --stages sonic

# Beat-sync mode (recommended)
python scripts/analyze_library.py --stages sonic --beat-sync

# Parallel processing
python scripts/update_sonic.py --beat-sync --workers 8
```

**What it does**:
1. Loads audio file
2. Detects beats (for beat-sync mode)
3. Extracts audio features per beat
4. Aggregates features (median + IQR)
5. Stores in database

**Output**: `data/metadata.db` with `sonic_features` column

**Fallback strategy**:
- If beat detection fails → automatic fallback to windowed mode
- If audio file is corrupted → skip track, continue
- Expected success rate: 95-99%

### 4. Artifact Building

```bash
python scripts/analyze_library.py --stages artifacts
```

**What it does**:
1. Loads all track data from database
2. Builds dense feature matrices
3. Normalizes vectors
4. Saves as NPZ file

**Output**: `data_matrices.npz` with all matrices

**Can be customized**:
- Rebuild for specific track subset
- Use different normalization
- Include/exclude certain features

## Recommended Workflows

### Workflow A: Fresh Library Setup

```bash
# 1. Scan music files
python scripts/scan_library.py

# 2. Get genre data (this will take time)
python scripts/update_genres_v3_normalized.py

# 3. Extract audio features in background
python scripts/update_sonic.py --beat-sync &

# 4. (Next day) Build artifacts
python scripts/analyze_library.py --stages artifacts

# 5. Generate playlist
python main_app.py --artist "Your Artist" --count 50
```

**Total time**: 2-3 hours wall clock (sonic runs in background)

### Workflow B: Incremental Update (New Tracks Added)

```bash
# Update with new files automatically discovered
python scripts/analyze_library.py

# Only analyzes tracks without features (very fast)
# Skips already-processed tracks
```

**Total time**: 5-10 minutes (only new tracks)

### Workflow C: Full Rebuild with New Mode

```bash
# Re-analyze all tracks with beat-sync (replaces windowed)
python scripts/analyze_library.py --force --stages sonic --beat-sync

# Then rebuild artifacts
python scripts/analyze_library.py --stages artifacts
```

**Total time**: 6-8 hours (re-processes all tracks)

### Workflow D: Development/Testing

```bash
# Analyze only first 100 tracks (for testing)
python scripts/analyze_library.py --limit 100

# Full pipeline on small dataset: ~10 minutes
# Use for testing before running on full library
```

## Performance Tuning

### Sonic Analysis (Most Time-Consuming)

```bash
# Increase workers (more CPU cores = faster)
python scripts/update_sonic.py --beat-sync --workers 16

# But CPU ≤ 100% use:
# Optimal: workers = (cpu_count - 1)
# Example: 8-core machine → --workers 7

# Expected speeds:
# 1 worker:  0.2 tracks/sec
# 4 workers: 0.7 tracks/sec
# 8 workers: 1.1 tracks/sec
# 16 workers: 1.2 tracks/sec (diminishing returns)
```

### Genre Fetching (API-Limited)

```bash
# API rate limits force sequential processing
# Parallel doesn't help much
# Expect: 0.5-1 track/sec (API-limited, not CPU-limited)

# Speed up by caching:
# 1st run: Slow (fetches all)
# 2nd run: Fast (uses cache)
```

### Memory Usage

| Stage | Memory |
|-------|--------|
| Scan | 100 MB |
| Genre fetch | 200 MB |
| Sonic (1 worker) | 500 MB |
| Sonic (8 workers) | 2 GB |
| Artifact build | 1 GB |

**If out of memory**: Use `--limit 1000` to process in batches

## Monitoring Progress

### Check Current Status

```bash
python scripts/analyze_library.py --stats
```

**Output**:
```
Sonic Analysis Statistics:
  Total tracks: 34100
  Analyzed: 31000
  Pending: 3100
  Librosa: 31000
  Fallback: 0
```

### Watch Live Progress

```bash
tail -f sonic_analysis.log
```

**Watch for**:
- `Progress: 1000/10000 (10%) - Rate: 1.1 tracks/sec - ETA: 2.5 hours`
- Number of failures (should be <1%)

### Verify Results

```bash
# Check database integrity
sqlite3 data/metadata.db "SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL"

# Should return high percentage (95%+)
```

## Troubleshooting Pipelines

### "Database is locked"
- Another instance running
- Solution: Close other processes, wait 30 seconds, retry

### "Out of memory"
- Too many workers for available RAM
- Solution: Reduce with `--workers 2` or use `--limit 1000`

### "ffmpeg not found" warning
- Optional dependency for some audio formats
- Solution: Install ffmpeg, or just proceed (slower but works)

### Genre fetch failing
- API key invalid or expired
- Solution: Update config.yaml with valid API keys
- Fallback: Will skip genre stage, continue with sonic

### Sonic analysis very slow
- Single-core processing
- Solution: Increase workers `--workers 8`
- Or: Run overnight and check progress in morning

## Resuming Interrupted Pipelines

All pipelines are resumable (idempotent):

```bash
# If sonic analysis was interrupted at 50%:
python scripts/analyze_library.py --stages sonic

# Will resume from where it stopped
# (tracks with features are skipped)

# Same for all stages
```

No data loss on interruption - only unfinished tracks need to be redone.

