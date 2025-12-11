# Playlist Generator

AI-powered music playlist generator using multi-segment sonic analysis and normalized genre metadata.

## Overview

This system generates intelligent playlists by combining:
- **Multi-segment sonic analysis** (beginning, middle, end) for accurate similarity matching
- **Normalized genre metadata** from Last.FM and MusicBrainz (artist/album/track level)
- **Hybrid scoring** (60% sonic + 40% genre) with smart filtering

## Key Features

- ✅ **Normalized Database Schema** - Artist and album genres fetched once, not per-track
- ✅ **Multi-Segment Sonic Analysis** - Captures entire song dynamics (intro, middle, outro)
- ✅ **Efficient Genre Updates** - 60% fewer API calls than track-by-track approach
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

### 3. Run Sonic Analysis

```bash
# Analyze with optimal workers for HDD (recommended)
python scripts/update_sonic.py --workers 6

# For SSD, can use more workers
python scripts/update_sonic.py --workers 12

# Analyze limited batch for testing
python scripts/update_sonic.py --limit 100
```

**Performance tips:**
- **HDD users:** Use 4-8 workers (avoid disk thrashing)
- **SSD users:** Can use 12-16 workers
- Processing rate: ~2-4 tracks/second depending on workers
- Full library (~33k tracks): 3-5 hours

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
sonic_features    TEXT              -- JSON: multi-segment analysis
sonic_source      TEXT              -- 'librosa' or 'acousticbrainz'
sonic_analyzed_at INTEGER           -- Unix timestamp
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
source            TEXT              -- 'lastfm_artist', 'musicbrainz_artist'
UNIQUE(artist, genre, source)
```

**album_genres** - Album-level genres
```sql
album_id          TEXT
genre             TEXT
source            TEXT              -- 'lastfm_album', 'musicbrainz_release'
UNIQUE(album_id, genre, source)
```

**track_genres** - Track-specific genres
```sql
track_id          TEXT
genre             TEXT
source            TEXT              -- 'lastfm_track', 'file'
UNIQUE(track_id, genre, source)
```

---

## Multi-Segment Sonic Analysis

### Feature Structure

Each track is analyzed in **3 segments** (30 seconds each):
- **Beginning** (0-30s): Intro characteristics
- **Middle** (center 30s): Main body of song
- **End** (last 30s): Outro characteristics
- **Average**: Averaged features across all segments

### Stored Features Per Segment

```python
{
  "beginning": {
    "mfcc_mean": [...],           # Timbre
    "spectral_centroid": 1234.5,  # Brightness
    "bpm": 120,                   # Tempo
    "chroma_mean": [...],         # Harmonic content
    "estimated_key": "C",         # Musical key
    "rms_energy": 0.15,           # Loudness
    ...
  },
  "middle": { ... },
  "end": { ... },
  "average": { ... }              # Used for similarity matching
}
```

### Benefits

1. **Better similarity matching** - Captures whole song, not just intro
2. **Dynamic song handling** - Songs with quiet intro + loud middle properly represented
3. **Future transition matching** - Can match end-to-beginning for DJ-style flow
4. **Build detection** - Can identify crescendos and drops

See `MULTI_SEGMENT_PLAN.md` for full implementation details.

---

## Configuration

Edit `config.yaml`:

```yaml
# Library settings
library:
  music_directory: "E:/MUSIC"     # Your music folder

# Last.FM API (for genre data)
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
    weight: 0.5                      # 50% genre weight
    sonic_weight: 0.5                # 50% sonic weight
    min_genre_similarity: 0.3        # Filter threshold
    method: "ensemble"               # Similarity method (see below)
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

See `docs/GENRE_SIMILARITY_METHODS.md` for detailed analysis and recommendations.
```

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
- **Artist:** Last.FM artist tags, MusicBrainz artist genres
- **Album:** Last.FM album tags, MusicBrainz release genres
- **Track:** Last.FM track tags, file tags (ID3/FLAC)

### update_sonic.py
Analyzes sonic features using librosa.

```bash
# Optimal for HDD
python scripts/update_sonic.py --workers 6

# Optimal for SSD
python scripts/update_sonic.py --workers 12

# Test on small batch
python scripts/update_sonic.py --limit 100

# Re-analyze everything (use with caution)
python scripts/update_sonic.py --force
```

**Worker optimization:**
- **HDD:** 4-8 workers (disk I/O bottleneck)
- **SSD:** 12-16 workers (CPU bottleneck)
- Watch disk usage - 100% disk = too many workers

### migrate_to_normalized_schema.py
One-time migration script (already run).

```bash
# Dry run (see what would happen)
python scripts/migrate_to_normalized_schema.py --dry-run

# Execute migration
python scripts/migrate_to_normalized_schema.py --execute
```

**What it does:**
- Creates albums, artist_genres, album_genres tables
- Extracts unique albums from tracks
- Migrates existing genre data
- Adds album_id foreign keys

---

## Similarity Algorithm

### Hybrid Scoring (60% Sonic + 40% Genre)

```python
final_score = (sonic_similarity * 0.6) + (genre_similarity * 0.4)
```

### Genre Similarity Calculation

Uses curated genre relationship matrix (`data/genre_similarity.yaml`):
- 1.0 = Same genre (e.g., "indie rock" ↔ "indie rock")
- 0.8 = Very similar (e.g., "indie rock" ↔ "alternative rock")
- 0.5 = Related (e.g., "indie rock" ↔ "post-punk")
- 0.0 = Unrelated (e.g., "indie rock" ↔ "jazz")

### Filtering

**Minimum genre similarity:** 0.3
- Prevents cross-genre mismatches
- Example: Blocks Ahmad Jamal (jazz) from matching Duster (slowcore)

---

## Data Sources

### Sonic Features
- **Primary:** Librosa (local analysis)
- **Alternative:** AcousticBrainz API (pre-computed, faster)
- **Current:** 100% librosa (local, no API dependency)

### Genre Data
- **Last.FM API** - Artist/album/track tags
- **MusicBrainz API** - Artist/release genres
- **File Tags** - ID3/FLAC embedded genres

### Genre Relationships
- **Curated matrix:** `data/genre_similarity.yaml`
- Manually weighted genre relationships
- Updated based on testing and user feedback

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

# Database stats
python scripts/validate_metadata.py
```

---

## Performance Tips

### For HDD Users
1. Use **6 workers** for sonic analysis
2. Run overnight (3-5 hours for full library)
3. Avoid running multiple heavy processes simultaneously
4. Consider moving library to SSD for faster processing

### For SSD Users
1. Use **12-16 workers** for sonic analysis
2. Can run genre + sonic updates simultaneously
3. Expect 1.5-2 hours for full library

### General
1. Run artist genres first (smallest dataset, biggest impact)
2. Albums and tracks can run overnight
3. Monitor disk usage - 100% = reduce workers
4. Sonic analysis is one-time; updates only needed for new files

---

## Troubleshooting

### "No tracks need genre updates"
Already up to date! Check with `--stats` to verify.

### Slow sonic analysis
- Check disk usage (should be <80%)
- Reduce workers if disk at 100%
- HDD users: Use 4-6 workers max

### Genre API errors
- Check Last.FM API key in config.yaml
- Rate limiting: Script has 1.5s delays built-in
- MusicBrainz: Has 1.1s delays (no API key needed)

### Duplicate tracks
- Library scanner now prevents duplicates (checks by file_path)
- Old duplicates removed during migration

---

## Documentation

- `README.md` - This file (overview & quick start)
- `MULTI_SEGMENT_PLAN.md` - Sonic analysis implementation details
- `REPOSITORY_STRUCTURE.md` - Full codebase structure
- `scripts/README.md` - Script documentation
- `docs/session_notes/` - Development session notes

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
- Last.FM API key (free at https://www.last.fm/api)
- Local music library
- Recommended: 8GB+ RAM for sonic analysis

---

## Current Status

After initial setup, your system should have:
- ✅ ~33,636 tracks scanned
- ✅ ~3,757 albums identified
- ✅ ~2,100 artists
- ⏳ Genre data: Populating (run update_genres_v3_normalized.py)
- ⏳ Sonic analysis: Populating (run update_sonic.py)

Once both are complete, you can generate intelligent playlists with full genre + sonic awareness!
