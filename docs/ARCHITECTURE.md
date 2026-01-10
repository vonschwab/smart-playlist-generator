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

### DJ Bridge Mode (`playlist/pier_bridge_builder.py`)
Advanced multi-seed playlist generation with genre-aware routing between "pier" tracks:

**Core Concept:**
- Multiple seed tracks (piers) serve as anchors
- System builds smooth genre-aware bridges between them
- Each segment optimizes transitions using beam search

**Phase 2 Enhancements (2026-01-09):**
Comprehensive fix for hub genre collapse where waypoints would default to generic genres:

1. **Vector Mode**: Direct multi-genre interpolation bypassing shortest-path label selection
   - Formula: `g = (1-s)*vA + s*vB`
   - Preserves full genre signatures throughout bridge
   - No more single-label collapse

2. **IDF Weighting**: Down-weights common genres like stop-words in text retrieval
   - Formula: `idf = log((N+1)/(df+1))^power`
   - Rare genres (shoegaze, slowcore): high weight (0.8-1.0)
   - Common genres (indie rock): low weight (0.1-0.3)

3. **Coverage Bonus**: Rewards candidates matching anchor's top-K signature genres
   - Schedule decay: strong near anchors, weak in middle
   - Influences 33% of sampled steps
   - Mean bonus: 0.104 per step

**Results:**
- +400% genre diversity in targets (1 label → 4-5 genres/step)
- Rare genres preserved and emphasized
- Smoother genre transitions with better alignment

**Phase 3 Enhancements (2026-01-09):**
Fixes saturation issues where waypoint/coverage scoring would plateau at caps:

1. **Centered Waypoint Delta**: Subtract step-wise baseline to allow negative deltas (-84% mean_delta, unsaturated)
2. **Tanh Squashing**: Smooth squashing prevents hard plateaus at cap (+100% winner_changed influence)
3. **Coverage Improvements**: Raw presence source + weighted mode for continuous gradient (-60% saturation)
4. **Provenance Overlaps**: Membership-based tracking reveals pool contribution vs overlap

**Configuration:**
```yaml
pier_bridge:
  dj_bridging:
    enabled: true
    route_shape: ladder
    dj_ladder_target_mode: vector       # Phase 2: Multi-genre targets
    dj_genre_use_idf: true              # Phase 2: IDF weighting
    dj_genre_use_coverage: true         # Phase 2: Coverage bonus
```

**Complete documentation:** See [dj_bridge_architecture.md](dj_bridge_architecture.md) for full technical details, implementation notes, and diagnostic logging guide.

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
