# System Architecture

High-level overview of Playlist Generator components and data flow.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MUSIC LIBRARY (File System)                  │
│              MP3, FLAC, M4A, OGG, WAV files...                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────────┐           ┌────────────────────┐
│ Library Scanner   │           │ Audio Analysis     │
│ (scan_library.py) │           │ (librosa)          │
│                   │           │                    │
│ • Find files      │           │ • Beat detection   │
│ • Extract tags    │           │ • Feature extract  │
│ • Metadata DB     │           │ • Sonic features   │
└────────┬──────────┘           └────────┬───────────┘
         │                               │
         └───────────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │   Database (SQLite)    │
                │   data/metadata.db     │
                │                        │
                │ • tracks table         │
                │ • albums, artists      │
                │ • genres, ratings      │
                │ • sonic_features       │
                └────────────┬───────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ Genre    │    │ Sonic        │    │ Artifact     │
   │ Similarity│   │ Features     │    │ Builder      │
   │ Builder  │    │              │    │              │
   │          │    │ • MFCC       │    │ Builds NPZ   │
   │ • API    │    │ • Chroma     │    │ matrices for │
   │ • Cooc   │    │ • Spectral   │    │ fast lookup  │
   │ • Smooth │    │ • Beat-sync  │    │              │
   └────┬─────┘    └──────┬───────┘    └────┬─────────┘
        │                 │                  │
        └─────────────────┼──────────────────┘
                          │
                ┌─────────▼──────────┐
                │  Artifacts (NPZ)   │
                │  Generated Matrices│
                │                    │
                │ • X_sonic (dense)  │
                │ • X_genre (dense)  │
                │ • track IDs, names │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────────────────┐
                │  Playlist Generator            │
                │  (main_app.py / API)           │
                │                                │
                │ • Candidate pool building      │
                │ • Scoring & ranking            │
                │ • Deduplication               │
                │ • Constraint enforcement       │
                └─────────┬──────────────────────┘
                          │
                ┌─────────▼─────────────┐
                │   Output              │
                │                       │
                │ • M3U playlists       │
                │ • JSON metadata       │
                │ • API responses       │
                └───────────────────────┘
```

## Component Overview

### 1. **Library Scanner** (`src/local_library_client.py`)
- Discovers music files in configured directories
- Extracts metadata (title, artist, album, duration)
- Stores in SQLite database
- **Input**: File system
- **Output**: `data/metadata.db`

### 2. **Audio Analyzer** (`src/librosa_analyzer.py`)
- Extracts sonic features from audio files
- Uses librosa for audio processing
- Beat-sync extraction for robust rhythm analysis
- Features: MFCC, chroma, spectral contrast, rhythm, etc.
- **Input**: Audio files + `data/metadata.db`
- **Output**: Updated `data/metadata.db` with sonic_features

### 3. **Genre Processor** (`src/multi_source_genre_fetcher.py`)
- Fetches genres from multiple APIs (Last.fm, MusicBrainz, Discogs)
- Normalizes and deduplicates genre tags
- Builds co-occurrence matrix for genre relationships
- **Input**: Track metadata
- **Output**: Genre assignments, `data/genre_similarity.yaml`

### 4. **Artifact Builder** (`src/analyze/artifact_builder.py`)
- Builds dense matrices from sparse data
- Creates X_sonic (audio feature vectors)
- Creates X_genre (normalized genre vectors)
- Stores in NPZ format for fast loading
- **Input**: `data/metadata.db`
- **Output**: `data_matrices.npz`

### 5. **Similarity Calculator** (`src/similarity_calculator.py`)
- Computes sonic similarity (cosine distance)
- Computes genre similarity (weighted ensemble)
- Computes hybrid similarity (60% sonic + 40% genre)
- **Input**: Feature vectors
- **Output**: Similarity scores (0-1)

### 6. **Playlist Constructor** (`src/playlist/constructor.py`)
- Builds candidate pool from seed track
- Applies constraints (artist diversity, genre coherence)
- Scores candidates by seed similarity + transitions
- Performs title deduplication
- **Input**: Artifact, seed track, constraints
- **Output**: Ordered list of track IDs

### 7. **API Backend** (`api/main.py`)
- FastAPI server with REST endpoints
- Handles library status, track search, playlist generation
- CORS support for frontend integration
- **Input**: HTTP requests
- **Output**: JSON responses

### 8. **Frontend** (`ui/`)
- React + Vite application
- Playlist generation interface
- Track browser
- Settings panel
- **Input**: User interactions
- **Output**: HTTP requests to API

## Data Model

### SQLite Database (`data/metadata.db`)

```
tracks:
  - track_id (primary key)
  - file_path
  - artist, album, title
  - duration
  - sonic_features (JSON)
  - sonic_source (beat_sync or windowed)

genres (normalized):
  - track_genre_id
  - track_id (FK)
  - genre_tag
  - source (lastfm, musicbrainz, discogs)

artifacts (generated):
  - Track ID mappings
  - Normalized feature vectors
  - Dimension names
```

### Artifact Matrices (`data_matrices.npz`)

```
X_sonic:         (n_tracks, 71)   - Beat-sync audio features
X_sonic_start:   (n_tracks, 71)   - Start-of-track features
X_sonic_end:     (n_tracks, 71)   - End-of-track features
X_genre_raw:     (n_tracks, k)    - Raw genre one-hot encoding
X_genre_smoothed: (n_tracks, k)   - Smoothed genre probabilities

track_ids:       (n_tracks,)      - Track identifiers
track_artists:   (n_tracks,)      - Artist names
track_titles:    (n_tracks,)      - Track titles
artist_keys:     (n_tracks,)      - Normalized artist identifiers
```

## Data Flow: Full Pipeline

```
1. SCAN
   music_library/ → scanner → metadata.db
   Time: 5-10 min per 1000 tracks

2. ANALYZE GENRES
   metadata.db → genre_fetcher → metadata.db (genres added)
   Time: 30-60 min per 1000 tracks (API-limited)

3. ANALYZE SONIC
   metadata.db → librosa_analyzer → metadata.db (features added)
   Time: 60+ min per 1000 tracks (CPU-bound)

4. BUILD ARTIFACTS
   metadata.db → artifact_builder → data_matrices.npz
   Time: 5-10 min

5. GENERATE PLAYLIST
   data_matrices.npz + seed_track → constructor → M3U file
   Time: <1 second
```

## Key Design Decisions

### 1. **Hybrid Similarity**
- **Why**: Sonic alone is noisy, genre alone is too coarse
- **How**: 60% sonic + 40% genre weighted blend
- **Result**: Playlists feel intentional (genre coherence) + natural (sonic flow)

### 2. **Beat-Synchronized Features**
- **Why**: Fixed-window extraction loses temporal structure
- **How**: Extract features per beat, aggregate with median + IQR
- **Result**: 4.34x better discrimination, better rhythm matching

### 3. **Multi-Source Genre Fetching**
- **Why**: No single API has complete coverage
- **How**: Query Last.fm, MusicBrainz, Discogs; merge + deduplicate
- **Result**: ~90%+ genre coverage for popular music

### 4. **Artifact Caching**
- **Why**: Playlist generation is fast, feature extraction is slow
- **How**: Pre-compute matrices, reload in <1 second
- **Result**: Instant playlist generation after initial build

### 5. **SQLite for Metadata**
- **Why**: Simple, embedded, queerable
- **How**: Store track data + computed features
- **Result**: Easy backup, queryable, no external dependencies

## Performance Characteristics

| Operation | Time | Memory | Parallelizable |
|-----------|------|--------|----------------|
| Library scan (1000 tracks) | 5 min | 100MB | Yes (files) |
| Genre fetch (1000 tracks) | 30-60 min | 200MB | No (API-limited) |
| Sonic analysis (1000 tracks) | 60-120 min | 500MB | Yes (multi-core) |
| Artifact build (1000 tracks) | 5 min | 300MB | No |
| Playlist generation | <1 sec | 50MB | N/A |

## Extensibility

### Adding a New Feature Source
1. Create feature extraction module in `src/features/`
2. Add feature computation to `src/librosa_analyzer.py`
3. Update artifact builder to include new features
4. Rerun pipeline: `scripts/analyze_library.py --force`

### Adding a New Playlist Mode
1. Extend `src/playlist/config.py` with new constraints
2. Implement custom scoring in `src/playlist/constructor.py`
3. Add CLI flag to `main_app.py`
4. Test with existing artifacts

### Integrating with External APIs
1. Create service module in `api/services/`
2. Add endpoint to `api/main.py`
3. Update frontend `ui/` to call new endpoint
4. No database changes needed

