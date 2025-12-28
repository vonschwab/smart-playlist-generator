# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Playlist Generator                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Library   │───▶│   Sonic     │───▶│  Artifacts  │             │
│  │   Scanner   │    │  Analyzer   │    │   Builder   │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                                     │                     │
│         ▼                                     ▼                     │
│  ┌─────────────┐                     ┌─────────────┐               │
│  │   Genre     │                     │ DS Pipeline │               │
│  │   Fetcher   │────────────────────▶│  (Generate) │               │
│  └─────────────┘                     └─────────────┘               │
│                                             │                       │
│                                             ▼                       │
│                                      ┌─────────────┐               │
│                                      │ M3U Export  │               │
│                                      └─────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Library Scan (`scan_library.py`)
- Scans configured music directory for audio files
- Extracts metadata (artist, title, album, duration) using Mutagen
- Stores tracks in SQLite database (`data/metadata.db`)
- Generates unique track IDs from file path + metadata

### 2. Genre Enrichment (`update_genres_v3_normalized.py`)
- Fetches artist genres from MusicBrainz
- Fetches album genres from Discogs
- Normalizes genre names
- Stores in normalized tables (artist_genres, album_genres, track_genres)

### 3. Sonic Analysis (`update_sonic.py`)
- Loads audio files with Librosa
- Extracts beat3tower features (137 dimensions):
  - Rhythm tower (21 dims): tempo, beats, onsets
  - Timbre tower (83 dims): MFCCs, spectral features
  - Harmony tower (33 dims): chroma, tonnetz
- Multi-segment extraction (start, mid, end, full)
- Stores JSON features in tracks.sonic_features

### 4. Artifact Building (`build_beat3tower_artifacts.py`)
- Loads all track features from database
- Applies tower-specific normalization (PCA whitening)
- Computes similarity matrices
- Saves as NumPy archive (.npz) for fast loading

### 5. Playlist Generation (`main_app.py`)
- Loads artifact bundle
- Runs DS (Deep Sequence) pipeline:
  1. Candidate generation (similar tracks)
  2. Filtering (artist diversity, recency)
  3. Scoring (transition quality)
  4. Ordering (pier + bridge strategy)
- Exports to M3U file

## Core Components

### PlaylistGenerator (`playlist_generator.py`)
Main orchestrator that coordinates:
- Seed selection (from history or artist)
- Candidate pool building
- Track selection and ordering
- Export to various formats

### DS Pipeline (`playlist/pipeline.py`)
Deep Sequence pipeline for intelligent track ordering:
- Uses precomputed similarity matrices
- Optimizes for smooth transitions
- Enforces diversity constraints

### Beat3Tower Features (`features/beat3tower_*.py`)
Audio feature extraction using 3-tower architecture:
- Captures different musical aspects separately
- Per-tower normalization for balanced similarity
- Multi-segment support for transition matching

### Similarity System (`similarity/`)
- `sonic_variant.py`: Preprocessing variants (tower_pca, robust_whiten)
- `hybrid.py`: Combined sonic + genre embeddings

## Database Schema

```sql
-- Core track table
tracks (
    track_id TEXT PRIMARY KEY,
    artist TEXT,
    title TEXT,
    album TEXT,
    file_path TEXT,
    duration_ms INTEGER,
    sonic_features TEXT,  -- JSON blob
    sonic_source TEXT,
    sonic_analyzed_at INTEGER
)

-- Genre tables (normalized)
artist_genres (artist TEXT, genre TEXT, source TEXT)
album_genres (album_id TEXT, genre TEXT, source TEXT)
track_genres (track_id TEXT, genre TEXT, source TEXT)

-- Album lookup
albums (album_id TEXT PRIMARY KEY, artist TEXT, title TEXT)
```

## Configuration

All behavior is controlled via `config.yaml`:
- Library paths
- API credentials (Last.FM, Discogs)
- DS pipeline parameters
- Playlist generation settings

See [CONFIG.md](CONFIG.md) for full reference.

## Extension Points

### Adding a New Export Format
1. Create exporter in `src/playlist_generator/` (e.g., `spotify_exporter.py`)
2. Add to `PlaylistApp.__init__()` in `main_app.py`
3. Call in `_export_and_report_playlist()`

### Adding a New Similarity Metric
1. Add variant to `similarity/sonic_variant.py`
2. Register in `resolve_sonic_variant()`
3. Use via `--sonic-variant` flag or config

### Adding a New Playlist Mode
1. Define parameters in `playlist/config.py`
2. Add mode key to `DSPipelineConfig.from_dict()`
3. Use via `--ds-mode` flag
