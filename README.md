# Playlist Generator

AI-powered music playlist generator using beat3tower sonic analysis and normalized genre metadata.

## Overview

This system generates intelligent playlists by combining:
- **Beat3Tower Sonic Analysis** - 3-tower architecture (rhythm, timbre, harmony) with 137-dimensional feature vectors
- **Multi-Segment Analysis** - Captures full song dynamics (start, middle, end) for accurate similarity and transition matching
- **Normalized Genre Metadata** - Efficient artist/album/track-level genre data from MusicBrainz and file tags
- **Validated Preprocessing** - Robust whitening with PCA for optimal sonic discrimination (proven 4/4 metrics pass)

## Key Features

- ✅ **Beat3Tower Feature Extraction** - State-of-the-art 137-dim sonic features (21 rhythm + 83 timbre + 33 harmony)
- ✅ **Validated Preprocessing** - Robust scaling + PCA whitening (+1,043% improvement over raw features)
- ✅ **Multi-Segment Analysis** - Start/mid/end segments for transition-aware playlist construction
- ✅ **Normalized Database Schema** - Artist and album genres fetched once, not per-track
- ✅ **Multiple Playlist Modes** - Narrow (focused), Dynamic (balanced), Discover (exploratory)
- ✅ **Local Library Support** - Direct file scanning (no Plex dependency)
- ✅ **Parallel Processing** - Multi-worker sonic analysis (optimized for HDD/SSD)
- ✅ **M3U Export** - Portable playlist files

---

## Quick Start

### 1. Initial Setup - Scan Your Library

```bash
# Scan your music library and extract metadata
python scripts/scan_library.py
```

**What it does:**
- Scans E:\MUSIC (configurable in config.yaml)
- Extracts metadata from audio files (MP3, FLAC, M4A, etc.)
- Generates unique track IDs
- Stores in database with duplicate prevention

### 2. Populate Genre Data

```bash
# Update artist genres (most efficient first)
python scripts/update_genres_v3_normalized.py --artists

# Update album genres
python scripts/update_genres_v3_normalized.py --albums

# Update track-specific genres
python scripts/update_genres_v3_normalized.py --tracks

# Or update all at once
python scripts/update_genres_v3_normalized.py
```

**Efficiency gains:**
- Artist genres: Fetched once per artist (~2,100 artists)
- Album genres: Fetched once per album (~3,757 albums)
- Track genres: Fetched per track (~33,636 tracks)
- **Total savings: ~60% fewer API calls vs per-track approach**

### 3. Run Beat3Tower Sonic Analysis

```bash
# Recommended: Beat3Tower extraction (state-of-the-art)
python scripts/update_sonic.py --beat3tower --workers 4

# For SSD, can use more workers
python scripts/update_sonic.py --beat3tower --workers 8

# Analyze limited batch for testing
python scripts/update_sonic.py --beat3tower --limit 100
```

**Performance tips:**
- **HDD users:** Use 4-6 workers (avoid disk thrashing)
- **SSD users:** Can use 8-12 workers
- Processing rate: ~1.5-3 tracks/second depending on workers
- Full library (~33k tracks): 4-8 hours for beat3tower (more thorough than legacy)

**Why Beat3Tower?**
- 137-dimensional features vs 27-dim legacy
- 3-tower architecture: rhythm (21 dims), timbre (83 dims), harmony (33 dims)
- Validated to pass all similarity metrics when combined with robust whitening
- Better captures musical structure and dynamics

### 4. Generate Playlists

```bash
# Generate playlists from listening history
python main_app.py

# Generate for specific artist
python main_app.py --artist "Radiohead" --tracks 30

# Preview without creating
python main_app.py --dry-run

# Dynamic mode (more variety)
python main_app.py --dynamic

# Discover mode (exploratory)
python main_app.py --discover
```

---

## Beat3Tower Sonic Analysis

### Architecture

Beat3Tower uses a **3-tower architecture** to capture different aspects of musical content:

#### 1. Rhythm Tower (21 dimensions)
- **Onset detection**: Rate, strength mean/std
- **Tempogram**: Tempo distribution histogram
- **Beat intervals**: Median, std, quantiles (10th, 25th, 75th, 90th)
- **BPM**: Detected tempo

#### 2. Timbre Tower (83 dimensions)
- **MFCCs**: Mel-frequency cepstral coefficients (13 coefficients × multiple statistics)
- **Spectral features**: Centroid, rolloff, contrast, bandwidth
- **Zero-crossing rate**: Percussiveness indicator
- **RMS energy**: Overall loudness

#### 3. Harmony Tower (33 dimensions)
- **Chroma features**: 12 pitch class bins
- **Tonnetz**: Tonal centroid features (harmonic relationships)
- **Key estimation**: Detected musical key and mode

**Total: 137 dimensions per segment**

### Multi-Segment Extraction

Each track is analyzed in **4 segments**:
- **Start** (first 30s): Intro characteristics
- **Mid** (center 30s): Main body of song
- **End** (last 30s): Outro characteristics
- **Full** (entire track): Aggregate features

**Stored Structure:**
```json
{
  "full": {
    "extraction_method": "beat3tower",
    "n_beats": 1466,
    "bpm_info": {...},
    "rhythm": {...},      // 21 dims
    "timbre": {...},      // 83 dims
    "harmony": {...}      // 33 dims
  },
  "start": {...},         // Same structure
  "mid": {...},
  "end": {...}
}
```

### Benefits

1. **Validated discrimination** - Passes all 4 similarity validation metrics:
   - TopK Gap: 0.258 (top neighbors 26% more similar than random)
   - Intra-Artist Coherence: 0.115 (same artist tracks cluster together)
   - Intra-Album Coherence: 0.282 (same album tracks strongly cluster)
   - Score Flatness: 181.2 (excellent separation between similar/dissimilar)

2. **Better transition matching** - End-to-start segment matching for smooth flow

3. **Robust features** - 137 dimensions capture nuanced musical structure

4. **Preprocessing-ready** - Works optimally with robust whitening (default)

See `diagnostics/BEAT3TOWER_VALIDATION_BREAKTHROUGH.md` for full validation results.

---

## Sonic Preprocessing (Robust Whitening)

### Why Preprocessing Matters

Raw beat3tower features have vastly different scales across dimensions (BPM vs MFCC coefficients), causing some dimensions to dominate similarity calculations. **Robust whitening** solves this:

**Step 1: Robust Scaling**
- Centers by median (not mean) - robust to outliers
- Scales by IQR (interquartile range) - resistant to extreme values

**Step 2: PCA Whitening**
- Rotates to principal components (uncorrelated directions)
- Scales each component to unit variance
- Removes redundant dimensions

**Step 3: L2 Normalization**
- Normalizes vectors for cosine similarity

### Validation Results

| Metric | Raw Features | Robust Whitening | Improvement |
|--------|-------------|------------------|-------------|
| TopK Gap | 0.023 | 0.258 | **+1,043%** |
| Intra-Artist Coherence | 0.024 | 0.115 | **+375%** |
| Intra-Album Coherence | 0.011 | 0.282 | **+2,555%** |
| Score Flatness | 0.082 | 181.223 | **+221,000%** |

**Robust whitening is enabled by default** for all playlist generation.

**Override if needed:**
```bash
# Use raw features (not recommended)
export SONIC_SIM_VARIANT=raw
python main_app.py

# Or via command line
python main_app.py --sonic-variant raw
```

---

## Database Schema (Normalized)

### Core Tables

**tracks** - Individual music files
```sql
track_id          TEXT PRIMARY KEY  -- MD5 hash of file_path|artist|title
artist            TEXT
title             TEXT
album             TEXT
file_path         TEXT              -- Full path to audio file
album_id          TEXT              -- Foreign key to albums
sonic_features    TEXT              -- JSON: beat3tower multi-segment
sonic_source      TEXT              -- 'librosa'
sonic_analyzed_at INTEGER           -- Unix timestamp
norm_artist       TEXT              -- Normalized artist name (for grouping)
```

**albums** - Unique albums (one per artist/album combination)
```sql
album_id          TEXT PRIMARY KEY  -- MD5 hash of artist|album
artist            TEXT
title             TEXT
UNIQUE(artist, title)
```

**artist_genres** - Artist-level genres
```sql
artist            TEXT
genre             TEXT
source            TEXT              -- 'musicbrainz_artist'
UNIQUE(artist, genre, source)
```

**album_genres** - Album-level genres
```sql
album_id          TEXT
genre             TEXT
source            TEXT              -- 'musicbrainz_release' or 'discogs_release'
UNIQUE(album_id, genre, source)
```

**track_genres** - Track-specific genres
```sql
track_id          TEXT
genre             TEXT
source            TEXT              -- 'file'
UNIQUE(track_id, genre, source)
```

---

## Playlist Modes

The system supports three playlist generation modes:

### Narrow Mode
**Goal**: Highly focused, cohesive playlists
- Higher similarity floor (0.35)
- Moderate artist diversity (20% max per artist)
- Shorter artist gaps (min 3 tracks between repeats)
- Use case: Deep dive into a specific sound

### Dynamic Mode (Default)
**Goal**: Balanced variety and cohesion
- Moderate similarity floor (0.30)
- Higher artist diversity (12.5% max per artist)
- Longer artist gaps (min 6 tracks between repeats)
- Hard transition floor (blocks jarring transitions)
- Use case: General listening, discovery with guardrails

### Discover Mode
**Goal**: Maximum exploration
- Lower similarity floor (0.25)
- Maximum artist diversity (5% max per artist)
- Longest artist gaps (min 9 tracks between repeats)
- Soft penalties (no hard blocks)
- Use case: Breaking out of listening patterns, find new connections

---

## Configuration

Edit `config.yaml`:

```yaml
# Library settings
library:
  music_directory: "E:/MUSIC"     # Your music folder

# Last.FM API (for listening history only)
lastfm:
  api_key: "your_api_key_here"
  username: "your_username"

# Database location
metadata:
  database_path: "data/metadata.db"

# Playlist generation
playlists:
  genre_similarity:
    enabled: true
    weight: 0.4                      # 40% genre weight
    sonic_weight: 0.6                # 60% sonic weight (configurable)
    min_genre_similarity: 0.3        # Genre filter threshold
    method: "ensemble"               # Similarity method (see below)
    use_discogs_album: true          # Use Discogs album genres
```

### Genre Similarity Methods

The system supports multiple genre similarity calculation methods:

- **ensemble** (recommended): Weighted combination of all methods for robust matching
- **weighted_jaccard**: Fast, relationship-aware set overlap
- **cosine**: Vector-based similarity using genre embeddings
- **best_match**: Optimal pairing between genre lists
- **jaccard**: Pure set overlap (strict, exact matches only)
- **average_pairwise**: Average of all genre-to-genre comparisons
- **legacy**: Original maximum similarity method (for comparison)

See `docs/GENRE_SIMILARITY_SYSTEM.md` for detailed analysis and recommendations.

---

## Scripts Reference

### scan_library.py
Scans music directory and populates database.

```bash
python scripts/scan_library.py
```

**Features:**
- Duplicate prevention (checks by file_path)
- Metadata extraction from audio files
- Genre extraction from ID3/FLAC tags
- Handles track metadata changes (updates track_id)

### update_genres_v3_normalized.py
Fetches genre metadata with normalized schema.

```bash
# Update specific types
python scripts/update_genres_v3_normalized.py --artists
python scripts/update_genres_v3_normalized.py --albums
python scripts/update_genres_v3_normalized.py --tracks

# Limit updates for testing
python scripts/update_genres_v3_normalized.py --artists --limit 10

# Show statistics
python scripts/update_genres_v3_normalized.py --stats
```

**Genre sources:**
- **Artist:** MusicBrainz artist genres
- **Album:** MusicBrainz release genres + Discogs release genres
- **Track:** File tags (ID3/FLAC)

### update_sonic.py
Analyzes sonic features using beat3tower extraction.

```bash
# Recommended: Beat3Tower extraction
python scripts/update_sonic.py --beat3tower --workers 4

# For SSD
python scripts/update_sonic.py --beat3tower --workers 8

# Test on small batch
python scripts/update_sonic.py --beat3tower --limit 100

# Re-analyze everything (use with caution)
python scripts/update_sonic.py --beat3tower --force

# Check extraction statistics
tail -50 sonic_analysis.log
```

**Worker optimization:**
- **HDD:** 4-6 workers (disk I/O bottleneck)
- **SSD:** 8-12 workers (CPU bottleneck)
- Watch disk usage - 100% disk = too many workers

**Beat3Tower vs Legacy:**
- Beat3Tower: 137 dims, validated, state-of-the-art (recommended)
- Legacy: 27 dims, outdated, lower quality (deprecated)

### sonic_validation_suite.py
Validates sonic similarity quality on a given seed track.

```bash
# Validate with robust whitening (default)
python scripts/sonic_validation_suite.py \
    --artifact data_matrices.npz \
    --seed-track-id <track_id> \
    --k 30 \
    --sonic-variant robust_whiten

# Compare raw vs whitened
python scripts/sonic_validation_suite.py \
    --seed-track-id <track_id> \
    --sonic-variant raw
```

**Outputs:**
- M3U playlists (sonic-only, genre-only, hybrid)
- Metrics CSV with validation scores
- Markdown report with pass/fail assessment

**Validation Metrics:**
- Score Flatness: (p90-p10)/median (threshold ≥ 0.5)
- TopK Gap: mean(top 30) - mean(random 30) (threshold ≥ 0.15)
- Intra-Artist Coherence: same artist vs random (threshold ≥ 0.05)
- Intra-Album Coherence: same album vs random (threshold ≥ 0.08)

---

## Similarity Algorithm

### Hybrid Scoring (Configurable Weights)

```python
# Default: 60% sonic + 40% genre
final_score = (sonic_similarity * 0.6) + (genre_similarity * 0.4)

# Configurable in config.yaml
playlists:
  genre_similarity:
    sonic_weight: 0.6  # Adjust here
    genre_weight: 0.4
```

### Sonic Similarity Calculation

1. **Feature extraction**: Beat3tower extracts 137-dim vectors per segment
2. **Preprocessing**: Robust scaling + PCA whitening (default)
3. **Similarity**: Cosine similarity on L2-normalized vectors

**Formula:**
```python
# With robust whitening (default)
X_scaled = RobustScaler().fit_transform(X_sonic)
X_pca = PCA(whiten=True).fit_transform(X_scaled)
X_norm = X_pca / ||X_pca||  # L2 normalize

sonic_sim = cosine_similarity(X_norm[track_a], X_norm[track_b])
```

### Genre Similarity Calculation

Uses curated genre relationship matrix (`data/genre_similarity.yaml`):
- 1.0 = Same genre (e.g., "indie rock" ↔ "indie rock")
- 0.8 = Very similar (e.g., "indie rock" ↔ "alternative rock")
- 0.5 = Related (e.g., "indie rock" ↔ "post-punk")
- 0.0 = Unrelated (e.g., "indie rock" ↔ "jazz")

**Method: Ensemble** (default)
```python
# Combines multiple methods with weights
genre_sim = (
    0.40 * weighted_jaccard(genres_a, genres_b) +
    0.25 * cosine_similarity(genres_a, genres_b) +
    0.20 * best_match_similarity(genres_a, genres_b) +
    0.15 * average_pairwise(genres_a, genres_b)
)
```

### Filtering

**Minimum genre similarity:** 0.3 (configurable)
- Prevents cross-genre mismatches
- Example: Blocks Ahmad Jamal (jazz) from matching Duster (slowcore)
- Can be adjusted per mode (narrow/dynamic/discover)

---

## Data Sources

### Sonic Features
- **Primary:** Librosa with beat3tower extraction (local analysis)
- **Current:** 100% librosa beat3tower (local, no API dependency)
- **Deprecated:** Legacy beat-sync extraction (27 dims)

### Genre Data
- **MusicBrainz API** - Artist/release genres
- **Discogs API** - Release genres (album-level)
- **File Tags** - ID3/FLAC embedded genres

### Genre Relationships
- **Curated matrix:** `data/genre_similarity.yaml`
- Manually weighted genre relationships
- Updated based on validation and user feedback

---

## Validation & Quality

### Sonic Validation Suite

The system includes a comprehensive validation suite (`scripts/sonic_validation_suite.py`) that measures sonic similarity quality:

**Metrics:**
1. **Score Flatness** - Measures separation between similar/dissimilar tracks
2. **TopK Gap** - Tests if top neighbors are better than random
3. **Intra-Artist Coherence** - Validates same-artist tracks cluster together
4. **Intra-Album Coherence** - Validates same-album tracks cluster together

**Current Results (Beat3Tower + Robust Whitening):**
- ✅ Score Flatness: 181.2 (PASS - threshold 0.5)
- ✅ TopK Gap: 0.258 (PASS - threshold 0.15)
- ✅ Intra-Artist Coherence: 0.115 (PASS - threshold 0.05)
- ✅ Intra-Album Coherence: 0.282 (PASS - threshold 0.08)

**Status: 4/4 metrics PASS** (first configuration to pass all criteria)

### Phase C Diagnostics

Playlist constructor now includes diagnostic counters for transition scoring:
- `total_candidates_evaluated` - All candidates considered
- `rejected_by_floor` - Candidates rejected by hard transition floor
- `penalty_applied_count` - Soft penalties applied
- `p10_transition` - 10th percentile of transition scores

**Usage:**
```python
result = construct_playlist(...)
print(f"Rejected by floor: {result.stats['rejected_by_floor']}")
print(f"P10 transition: {result.stats['p10_transition']:.3f}")
```

See `diagnostics/BEAT3TOWER_VALIDATION_BREAKTHROUGH.md` for detailed validation analysis.

---

## Monitoring & Logs

### Log Files

Located in root directory:
- `sonic_analysis.log` - Sonic analysis progress
- `genre_update_v3.log` - Genre update progress
- `playlist_generator.log` - Playlist generation

### Check Progress

```bash
# Genre statistics
python scripts/update_genres_v3_normalized.py --stats

# Check sonic analysis status
tail -50 sonic_analysis.log

# Validation on test seed
python scripts/sonic_validation_suite.py --seed-track-id <id>
```

---

## Performance Tips

### For HDD Users
1. Use **4-6 workers** for beat3tower analysis
2. Run overnight (6-8 hours for full library)
3. Avoid running multiple heavy processes simultaneously
4. Consider moving library to SSD for faster processing

### For SSD Users
1. Use **8-12 workers** for beat3tower analysis
2. Can run genre + sonic updates simultaneously
3. Expect 3-4 hours for full library

### General
1. Run artist genres first (smallest dataset, biggest impact)
2. Albums and tracks can run overnight
3. Monitor disk usage - 100% = reduce workers
4. Beat3tower analysis is one-time; updates only needed for new files
5. Use `--limit 100` to test configuration before full scan

---

## Troubleshooting

### "No tracks need genre updates"
Already up to date! Check with `--stats` to verify.

### Slow sonic analysis
- Check disk usage (should be <80%)
- Reduce workers if disk at 100%
- HDD users: Use 4-6 workers max
- Beat3tower is more thorough than legacy (slower but better quality)

### Genre API errors
- MusicBrainz: Has 1.1s delays (no API key needed)
- Discogs: Requires API key in config.yaml
- Rate limiting: Built-in delays prevent issues

### Validation fails (low TopK gap, negative coherence)
- Ensure beat3tower extraction is used (not legacy)
- Verify robust_whiten is enabled (default)
- Check artifact has 137-dim features: `python -c "import numpy as np; data = np.load('artifact.npz'); print(data['X_sonic'].shape)"`
- Expected: (n_tracks, 137)

### Duplicate tracks
- Library scanner now prevents duplicates (checks by file_path)
- Old duplicates removed during migration

---

## Documentation

- `README.md` - This file (overview & quick start)
- `REPOSITORY_STRUCTURE.md` - Full codebase structure
- `scripts/README.md` - Script documentation
- `docs/GENRE_SIMILARITY_SYSTEM.md` - Genre similarity deep dive
- `diagnostics/BEAT3TOWER_VALIDATION_BREAKTHROUGH.md` - Validation results
- `diagnostics/HOUSEKEEPING_SUMMARY.md` - Recent improvements summary

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
- Last.FM API key (only for history; free at https://www.last.fm/api)
- Discogs API key (optional, for album genres)
- Local music library
- Recommended: 8GB+ RAM for sonic analysis
- Recommended: SSD for faster processing

---

## Current Status

After initial setup, your system should have:
- ✅ ~33,636 tracks scanned
- ✅ ~3,757 albums identified
- ✅ ~2,100 artists
- ✅ Genre data: Populated via normalized schema
- ⏳ Sonic analysis: Beat3tower extraction in progress (use `--beat3tower`)

Once beat3tower analysis is complete, you can generate validated, high-quality playlists with:
- Proven sonic discrimination (4/4 validation metrics pass)
- Robust preprocessing (automatic via default settings)
- Multi-mode support (narrow/dynamic/discover)
- Transition-aware construction (end-to-start matching)

---

## Migration from Legacy

If you have existing sonic features from legacy beat-sync extraction:

**Check current extraction method:**
```bash
# Count tracks by extraction method
python -c "
import sqlite3
import json
conn = sqlite3.connect('metadata.db')
c = conn.cursor()
c.execute('SELECT sonic_features FROM tracks WHERE sonic_features IS NOT NULL LIMIT 1')
row = c.fetchone()
if row:
    features = json.loads(row[0])
    if 'full' in features and features['full'].get('extraction_method') == 'beat3tower':
        print('✓ Using beat3tower')
    else:
        print('✗ Using legacy extraction')
"
```

**Migrate to beat3tower:**
```bash
# Re-analyze with beat3tower (overwrites legacy features)
python scripts/update_sonic.py --beat3tower --force --workers 4
```

**Note:** Legacy features (27 dims) have poor discrimination (TopK gap 0.030). Beat3tower (137 dims) with whitening achieves gap 0.258 (+1,043% improvement).

---

## Contributing

When making changes:
1. Run validation suite on test seeds
2. Update documentation
3. Add diagnostic counters for new features
4. Include validation results in commit messages

See git history for examples of well-documented commits.
