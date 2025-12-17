# AI Playlist Generator - Repository Structure

## Overview

This repository contains a local library playlist generator that uses multi-segment sonic analysis and normalized genre metadata to create intelligent, personalized music playlists.

## Directory Structure

```
PLAYLIST GENERATOR/
├── main_app.py                          # Main application entry point
├── config.yaml                          # Configuration file
├── requirements.txt                     # Python dependencies
├── README.md                            # Main documentation
├── REPOSITORY_STRUCTURE.md              # This file
│
├── src/                                 # Core library code (18 modules)
│   ├── playlist_generator.py            # Main playlist generation logic
│   ├── similarity_calculator.py         # Hybrid sonic + genre similarity
│   ├── genre_similarity_v2.py           # Advanced genre similarity (7 methods)
│   ├── genre_similarity.py              # Legacy genre similarity
│   ├── string_utils.py                  # Shared string normalization helpers
│   ├── artist_utils.py                  # Shared artist normalization helpers
│   ├── local_library_client.py          # Local library interface
│   ├── metadata_client.py               # Database interface
│   ├── lastfm_client.py                 # Last.FM API client (listening history only)
│   ├── openai_client.py                 # OpenAI API integration
│   ├── librosa_analyzer.py              # Audio analysis wrapper
│   ├── hybrid_sonic_analyzer.py         # Multi-segment sonic analysis
│   ├── artist_cache.py                  # Artist similarity caching
│   ├── track_matcher.py                 # Track matching logic
│   ├── m3u_exporter.py                  # M3U playlist export
│   ├── multi_source_genre_fetcher.py    # Genre fetching pipeline
│   ├── config_loader.py                 # Configuration manager
│   ├── rate_limiter.py                  # API rate limiting
│   ├── retry_helper.py                  # Retry logic
│   └── __init__.py                      # Package init
│
├── scripts/                             # Utility scripts
│   ├── README.md                        # Scripts documentation
│   ├── scan_library.py                  # Scan music directory
│   ├── update_genres_v3_normalized.py   # Genre metadata updater (MusicBrainz-only)
│   ├── update_sonic.py                  # Sonic feature analyzer
│   └── validate_metadata.py             # Database validation
│
├── data/                                # Data files
│   ├── metadata.db                      # SQLite database (219MB)
│   └── genre_similarity.yaml            # Genre relationship matrix
│
├── docs/                                # Documentation
│   ├── GENRE_SIMILARITY_METHODS.md      # Comparison of 7 similarity methods
│   ├── GENRE_SIMILARITY_SYSTEM.md       # Architecture overview
│   ├── INTEGRATION_COMPLETE.md          # V2 integration notes
│   └── session_notes/                   # Development session notes
│
└── logs/                                # Log files
    ├── playlist_generator.log           # Main app logs
    ├── sonic_analysis.log               # Sonic analysis progress
    └── genre_update_v3.log              # Genre update progress
```

## Main Components

### Core Application

- **main_app.py** - Command-line interface for playlist generation
- **config.yaml** - Configuration (Last.FM API, OpenAI API, paths, playlist settings)

### Library Modules (`src/`)

All core functionality is in the `src/` directory:

#### Playlist Generation
- **playlist_generator.py** (2,884 lines) - Main playlist generation logic with genre filtering, artist distribution, duplicate prevention
- **m3u_exporter.py** - Exports playlists to M3U format

#### Similarity & Matching
- **similarity_calculator.py** (904 lines) - Hybrid scoring system combining sonic (60%) and genre (40%) similarity
- **genre_similarity_v2.py** (17,465 lines) - Advanced genre similarity with 7 methods (Jaccard, Weighted Jaccard, Cosine, Pairwise, Best Match, Ensemble, Legacy)
- **genre_similarity.py** - Legacy genre similarity implementation
- **track_matcher.py** - Matches tracks between different data sources

#### Audio Analysis
- **librosa_analyzer.py** - Audio feature extraction wrapper using librosa
- **hybrid_sonic_analyzer.py** - Multi-segment sonic analysis (beginning/middle/end + average)

#### Data Sources & APIs
- **local_library_client.py** - Local library interface (SQLite queries, track retrieval)
- **metadata_client.py** - Database management (schema creation, genre storage, queries)
- **lastfm_client.py** - Last.FM API client (listening history, artist info, genre tags)
- **openai_client.py** - OpenAI API integration
- **multi_source_genre_fetcher.py** - Genre fetching pipeline combining multiple sources

#### Caching & Utilities
- **artist_cache.py** - Performance optimization through artist similarity caching
- **config_loader.py** - Configuration file manager
- **rate_limiter.py** - API rate limiting for Last.FM and MusicBrainz
- **retry_helper.py** - Retry logic for API calls

### Utility Scripts (`scripts/`)

Four main scripts for library maintenance:

1. **scan_library.py** - Scans music directory and extracts metadata
2. **update_genres_v3_normalized.py** - Fetches genre data from Last.FM + MusicBrainz (normalized schema)
3. **update_sonic.py** - Analyzes sonic features using librosa with multi-worker support
4. **validate_metadata.py** - Validates database integrity and shows statistics

See `scripts/README.md` for detailed usage.

## Workflow

### Initial Setup

```bash
# 1. Configure settings
edit config.yaml  # Add music directory path (Last.FM API key only for history)

# 2. Scan your music library
python scripts/scan_library.py

# 3. Populate genre data (normalized schema)
python scripts/update_genres_v3_normalized.py --artists  # Start with artists (most efficient)
python scripts/update_genres_v3_normalized.py --albums   # Then albums
python scripts/update_genres_v3_normalized.py --tracks   # Finally tracks

# 4. Analyze sonic features
python scripts/update_sonic.py --workers 6   # For HDD
python scripts/update_sonic.py --workers 12  # For SSD

# 5. Generate playlists
python main_app.py
```

### Incremental Updates

```bash
# Rescan library for new files
python scripts/scan_library.py

# Update genres for new items
python scripts/update_genres_v3_normalized.py --artists
python scripts/update_genres_v3_normalized.py --albums
python scripts/update_genres_v3_normalized.py --tracks

# Analyze new tracks (skips already analyzed)
python scripts/update_sonic.py --workers 6

# Generate playlists
python main_app.py
```

### Statistics & Validation

```bash
# Check genre coverage
python scripts/update_genres_v3_normalized.py --stats

# Validate database integrity
python scripts/validate_metadata.py

# Monitor logs
tail -f logs/sonic_analysis.log
tail -f logs/genre_update_v3.log
```

## Key Features

### Multi-Segment Sonic Analysis
- Analyzes songs in 3 segments: **beginning** (0-30s), **middle** (center 30s), **end** (last 30s)
- Captures dynamic changes (quiet intro → loud chorus)
- Better whole-song representation
- Future: DJ-style transition matching (end-to-beginning)

### Normalized Database Schema
- **Artist genres** fetched once per artist (~2,100 artists)
- **Album genres** fetched once per album (~3,757 albums)
- **Track genres** fetched per track (~33,636 tracks)
- **60% fewer API calls** vs per-track approach
- Source-aware storage (MusicBrainz, file tags)

### Advanced Genre Similarity
- **7 similarity methods** implemented:
  - Jaccard (pure set overlap)
  - Weighted Jaccard (relationship-aware)
  - Cosine (vector-based)
  - Average Pairwise (comprehensive)
  - Best Match (optimal pairing)
  - **Ensemble** (recommended - weighted combination)
  - Legacy (original method)
- Configurable via `config.yaml`

### Hybrid Scoring System
- **60% Sonic Similarity** - Audio features (MFCC, chroma, tempo, spectral)
- **40% Genre Similarity** - Curated genre relationship matrix
- **Minimum genre threshold** (default: 0.2) - Prevents cross-genre mismatches

### Smart Filtering & Distribution
- **Genre filtering** - Blocks unrelated genres (e.g., jazz vs. slowcore)
- **Artist distribution** - Max 4 tracks per artist, windowed distribution
- **Duplicate prevention** - No repeated tracks
- **Broad genre filters** - Ignores generic tags (50s, 60s, rock, pop, etc.)

### Performance Optimization
- **Multi-worker sonic analysis** - Parallel processing (configurable workers)
- **Artist similarity caching** - Reduces API calls
- **Rate limiting** - Respects API limits (Last.FM, MusicBrainz)
- **Retry logic** - Handles transient failures

## Configuration

Key settings in `config.yaml`:

```yaml
library:
  music_directory: E:/MUSIC
  database_path: metadata.db

lastfm:
  api_key: [your_key]
  username: [your_username]

playlists:
  count: 8
  tracks_per_playlist: 30

  genre_similarity:
    enabled: true
    weight: 0.33                    # Genre weight in final score
    sonic_weight: 0.67              # Sonic weight in final score
    min_genre_similarity: 0.2       # Minimum threshold
    method: ensemble                # Similarity method
    broad_filters: [50s, 60s, rock, pop, ...]  # Ignore these genres

  max_tracks_per_artist: 4
  artist_window_size: 8
  export_m3u: true
  m3u_export_path: E:/PLAYLISTS
```

## Database Schema

Normalized SQLite schema (`metadata.db`):

### Core Tables
- **tracks** - Track metadata, sonic features (multi-segment), file paths
- **albums** - Unique albums (artist + title)

### Genre Tables (Source-Aware)
- **artist_genres** - Artist-level genres (lastfm_artist, musicbrainz_artist)
- **album_genres** - Album-level genres (lastfm_album, musicbrainz_release)
- **track_genres** - Track-specific genres (lastfm_track, file)

See `README.md` for full schema details.

## Technology Stack

- **Python 3.8+**
- **Audio Analysis**: librosa, mutagen
- **APIs**: MusicBrainz, OpenAI (Last.FM history optional)
- **Data Science**: numpy, scipy, scikit-learn
- **Optimization**: python-tsp (Traveling Salesman Problem)
- **Database**: SQLite3
- **Configuration**: PyYAML

## Current Status

- 33,636 tracks scanned
- 3,757 albums identified
- 2,100 artists
- Database size: 219MB
- Genre data: ~88% complete
- Sonic analysis: ~35% complete

## Documentation

- **README.md** - Main documentation, quick start, feature overview
- **REPOSITORY_STRUCTURE.md** - This file (repository structure)
- **scripts/README.md** - Scripts documentation and usage
- **docs/GENRE_SIMILARITY_METHODS.md** - Detailed comparison of 7 similarity methods
- **docs/GENRE_SIMILARITY_SYSTEM.md** - Architecture and system design
- **docs/INTEGRATION_COMPLETE.md** - V2 genre similarity integration notes
- **docs/session_notes/** - Development session notes

## Version History

- **v4.0 "Multi-Segment"** - Multi-segment sonic analysis (beginning/middle/end)
- **v3.0 "Normalized Schema"** - Normalized database, 7 genre similarity methods
- **v2.0 "Local-First"** - Removed Plex dependency, direct file scanning
- **v1.0** - Initial version

---

**Last Updated:** December 11, 2025
