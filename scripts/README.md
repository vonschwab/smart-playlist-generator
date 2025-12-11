# Scripts Directory

This directory contains utility scripts for maintaining and updating the music library metadata.

## Main Scripts

### `scan_library.py` - Music Library Scanner

Scans your music directory and populates the database with track metadata.

```bash
# Scan the music library
python scan_library.py
```

**What it does:**
- Scans the music directory specified in `config.yaml`
- Extracts metadata from audio files (MP3, FLAC, M4A, OGG, WMA, etc.)
- Generates unique track IDs (MD5 hash of file_path|artist|title)
- Creates album records with unique album IDs
- Prevents duplicates by checking file paths
- Extracts embedded genres from file tags

**Features:**
- Supports multiple audio formats via mutagen
- Duplicate prevention (checks by file_path)
- Handles track metadata changes (updates track_id)
- Progress reporting

---

### `update_genres_v3_normalized.py` - Genre Data Updater (Normalized Schema)

Fetches and updates genre data from Last.FM and MusicBrainz using the normalized database schema.

```bash
# Update artist genres (most efficient - ~2,100 artists)
python update_genres_v3_normalized.py --artists

# Update album genres (~3,757 albums)
python update_genres_v3_normalized.py --albums

# Update track-specific genres (~33,636 tracks)
python update_genres_v3_normalized.py --tracks

# Update all at once
python update_genres_v3_normalized.py

# Limit updates for testing
python update_genres_v3_normalized.py --artists --limit 10

# Show coverage statistics
python update_genres_v3_normalized.py --stats
```

**Features:**
- **Normalized schema**: Artist and album genres fetched once, not per-track
- **60% fewer API calls** vs per-track approach
- **Multi-source data**:
  - Last.FM artist tags
  - Last.FM album tags
  - Last.FM track tags
  - MusicBrainz artist genres
  - MusicBrainz release genres
- **Source tracking**: Stores which API provided each genre
- **Incremental updates**: Only fetches missing data
- **Empty markers**: Distinguishes "never checked" from "checked but empty"
- **Rate limiting**: Built-in delays (1.5s Last.FM, 1.1s MusicBrainz)
- **Resumable**: Can stop and restart without losing progress

**Efficiency:**
```
Traditional approach:  33,636 tracks × API calls = massive overhead
Normalized approach:   2,100 artists + 3,757 albums + 33,636 tracks = 60% savings
```

---

### `update_sonic.py` - Sonic Feature Analyzer (Multi-Segment)

Analyzes audio features using librosa with multi-segment analysis and parallel processing.

```bash
# Analyze with optimal workers for HDD (recommended)
python update_sonic.py --workers 6

# For SSD, can use more workers
python update_sonic.py --workers 12

# Analyze limited batch for testing
python update_sonic.py --limit 100

# Re-analyze everything (use with caution)
python update_sonic.py --force

# Show progress statistics
python update_sonic.py --stats
```

**Features:**
- **Multi-segment analysis**: Analyzes beginning (0-30s), middle (center 30s), end (last 30s)
- **Parallel processing**: Multi-worker support with configurable workers
- **Audio features extracted**:
  - MFCC (timbre, texture)
  - Spectral centroid (brightness)
  - Chroma (harmonic content)
  - Tempo/BPM
  - Key estimation
  - RMS energy (loudness)
  - Zero crossing rate
  - Spectral rolloff
  - Spectral bandwidth
- **Progress reporting**: Shows tracks/sec, ETA, percentage complete
- **Incremental**: Skips already analyzed tracks (unless --force)
- **Error handling**: Continues on failures, logs errors

**Worker Optimization:**
- **HDD users**: Use 4-8 workers (avoid disk thrashing)
- **SSD users**: Can use 12-16 workers
- Monitor disk usage - if at 100%, reduce workers
- Processing rate: 2-4 tracks/second depending on workers
- Full library (~33k tracks): 3-5 hours (HDD), 1.5-2 hours (SSD)

---

### `validate_metadata.py` - Database Validator

Validates database integrity and shows statistics.

```bash
# Validate and show statistics
python validate_metadata.py
```

**What it checks:**
- Database integrity
- Track counts
- Genre coverage
- Sonic analysis progress
- Missing data
- Anomalies

## Workflow

### Complete Setup Workflow

```bash
# 1. Configure settings
cd "C:\Users\Dylan\Desktop\PLAYLIST GENERATOR"
edit config.yaml  # Set music_directory, Last.FM API key, etc.

# 2. Scan local music library
python scripts/scan_library.py

# 3. Fetch genre data (normalized schema - most efficient order)
python scripts/update_genres_v3_normalized.py --artists  # ~2,100 artists
python scripts/update_genres_v3_normalized.py --albums   # ~3,757 albums
python scripts/update_genres_v3_normalized.py --tracks   # ~33,636 tracks

# 4. Analyze sonic features (multi-segment)
python scripts/update_sonic.py --workers 6  # Adjust workers for your disk type

# 5. Generate playlists
python main_app.py
```

### Incremental Updates (Adding New Music)

```bash
# 1. Rescan library (picks up new files)
python scripts/scan_library.py

# 2. Update genres for new items
python scripts/update_genres_v3_normalized.py --artists
python scripts/update_genres_v3_normalized.py --albums
python scripts/update_genres_v3_normalized.py --tracks

# 3. Analyze new tracks (automatically skips already analyzed)
python scripts/update_sonic.py --workers 6

# 4. Generate playlists
python main_app.py
```

### Quick Testing Workflow

```bash
# Test with small batches
python scripts/update_genres_v3_normalized.py --artists --limit 10
python scripts/update_sonic.py --limit 100

# Check statistics
python scripts/update_genres_v3_normalized.py --stats
```

## Statistics & Monitoring

Check coverage and progress:

```bash
# Genre coverage statistics
python scripts/update_genres_v3_normalized.py --stats

# Database validation
python scripts/validate_metadata.py

# Monitor live progress
tail -f logs/genre_update_v3.log
tail -f logs/sonic_analysis.log
```

Example stats output:
```
Genre Coverage Statistics (Normalized Schema)
======================================================================
Artists:
  Total: 2,100
  With Last.FM data: 1,850 (88.1%)
  With MusicBrainz data: 1,920 (91.4%)
  Pending: 180 (8.6%)

Albums:
  Total: 3,757
  With Last.FM data: 3,200 (85.2%)
  With MusicBrainz data: 3,400 (90.5%)
  Pending: 357 (9.5%)

Tracks:
  Total: 33,636
  With genres (any source): 29,672 (88.2%)
  Pending: 3,964 (11.8%)
======================================================================

Sonic Analysis Statistics:
  Total tracks: 33,636
  Analyzed: 11,772 (35.0%)
  Pending: 21,864 (65.0%)
  Analysis method: Multi-segment librosa
  Estimated time remaining: 3.2 hours @ 6 workers
======================================================================
```

## Tips & Best Practices

### For Efficiency
1. **Always start with artists** - Smallest dataset, biggest impact
2. **Run overnight** - Genre + sonic updates can take hours for large libraries
3. **Monitor disk usage** - If at 100%, reduce sonic workers
4. **Use limits for testing** - Test with `--limit 10` before full runs

### For HDD Users
- Use 4-8 workers for sonic analysis
- Don't run multiple heavy processes simultaneously
- Consider moving library to SSD for faster processing

### For SSD Users
- Use 12-16 workers for sonic analysis
- Can run genre and sonic updates in parallel
- Much faster overall (1.5-2 hours vs 3-5 hours)

### For Resume/Retry
- All scripts are resumable - just run again
- Genre updates use empty markers to track checked-but-empty items
- Sonic analysis skips already analyzed tracks
- Safe to Ctrl+C and restart
