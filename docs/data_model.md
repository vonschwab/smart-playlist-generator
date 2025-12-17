# Data Model

Complete reference for database schema and artifact format.

## SQLite Database: `data/metadata.db`

### tracks Table

Core track information and computed features.

```sql
CREATE TABLE tracks (
  track_id TEXT PRIMARY KEY,           -- Unique identifier (UUID)
  file_path TEXT NOT NULL UNIQUE,      -- Absolute path to audio file

  -- Metadata
  artist TEXT,                         -- Track artist
  title TEXT,                          -- Track title
  album TEXT,                          -- Album name
  musicbrainz_id TEXT,                 -- MusicBrainz MBID

  -- Audio properties
  duration INTEGER,                    -- Duration in seconds

  -- Sonic features (JSON)
  sonic_features TEXT,                 -- JSON: audio analysis results
  sonic_source TEXT,                   -- 'beat_sync' or 'windowed'
  sonic_analyzed_at INTEGER,           -- Unix timestamp

  -- Genres (joined via track_genres table)

  -- Metadata
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  updated_at INTEGER DEFAULT (strftime('%s', 'now'))
)

CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
```

### track_genres Table

Genre assignments with source tracking.

```sql
CREATE TABLE track_genres (
  track_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id TEXT NOT NULL UNIQUE,       -- FK to tracks
  genre_tag TEXT NOT NULL,             -- Genre (e.g., 'rock', 'jazz')
  source TEXT NOT NULL,                -- 'lastfm', 'musicbrainz', or 'discogs'
  confidence REAL DEFAULT 1.0,         -- Confidence 0-1 (if applicable)

  FOREIGN KEY (track_id) REFERENCES tracks(track_id)
);

CREATE INDEX idx_track_genres_track_id ON track_genres(track_id);
CREATE INDEX idx_track_genres_genre ON track_genres(genre_tag);
```

### albums Table

Album metadata (optional, for album-level queries).

```sql
CREATE TABLE albums (
  album_id TEXT PRIMARY KEY,
  title TEXT,
  artist TEXT,
  release_year INTEGER,
  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);
```

### Meta Table

Schema versioning and configuration.

```sql
CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

-- Example entries:
-- ('schema_version', '2')
-- ('last_sync', '1703000000')
-- ('library_path', '/path/to/music')
```

## Sonic Features Schema

Stored in `tracks.sonic_features` as JSON.

### Beat-Sync Features (Phase 2)

```json
{
  "extraction_method": "beat_sync",

  "mfcc_median": [0.1, -0.2, 0.3, ...],      // (13,)
  "mfcc_iqr": [0.05, 0.08, 0.1, ...],        // (13,)

  "chroma_median": [0.1, 0.2, 0.15, ...],    // (12,)
  "chroma_iqr": [0.05, 0.06, 0.04, ...],     // (12,)

  "spectral_contrast_median": [1.0, 2.0, ...], // (7,)
  "spectral_contrast_iqr": [0.5, 0.8, ...],    // (7,)

  "onset_strength_mean": 0.42,
  "onset_strength_std": 0.15,

  "spectral_centroid_mean": 2400,             // Hz
  "spectral_centroid_std": 600,

  "spectral_rolloff_mean": 8000,              // Hz
  "spectral_rolloff_std": 1200,

  "bpm": 103.5
}
```

**Total dimensions**: 71

### Windowed Features (Legacy)

```json
{
  "extraction_method": "windowed",

  "mfcc_mean": [...],           // (13,)
  "mfcc_std": [...],            // (13,)

  "chroma_mean": [...],         // (12,)
  "chroma_std": [...],          // (12,)

  "spectral_contrast_mean": [...], // (7,)

  "onset_strength": 0.42,
  "spectral_centroid": 2400,
  "spectral_rolloff": 8000,

  "bpm": 103.5
}
```

**Total dimensions**: 52

## Artifact Matrices: `data_matrices.npz`

Optimized matrices for fast similarity computation and playlist generation.

### X_sonic

**Shape**: `(n_tracks, 71)`
**Type**: `float32`
**Content**: Beat-sync audio features, L2-normalized

```
X_sonic[i] = normalized beat-sync features for track i
            Values: 0.0-1.0 (after normalization)
```

**Accessed via**:
```python
artifact = np.load('data_matrices.npz', allow_pickle=True)
X_sonic = artifact['X_sonic']  # (34100, 71)
```

### X_sonic_start

**Shape**: `(n_tracks, 71)`
**Type**: `float32`
**Content**: Audio features from first 30 seconds of track

**Use**: Computing transitions from previous track end to current track start

### X_sonic_end

**Shape**: `(n_tracks, 71)`
**Type**: `float32`
**Content**: Audio features from last 30 seconds of track

**Use**: Computing flow quality between consecutive tracks

### X_genre_raw

**Shape**: `(n_tracks, n_genres)`
**Type**: `float32`
**Content**: One-hot or sparse encoding of track genres

```
X_genre_raw[i] = genre vector for track i
                 One element per genre in vocabulary
                 Value = 1.0 if track has genre, 0.0 otherwise
```

### X_genre_smoothed

**Shape**: `(n_tracks, n_genres)`
**Type**: `float32`
**Content**: Smoothed genre probabilities (handles ambiguity)

```
X_genre_smoothed[i] = probability distribution over genres
                      Sum across axis = ~1.0 (normalized)
                      Handles tracks with multiple genres gracefully
```

### track_ids

**Shape**: `(n_tracks,)`
**Type**: String (object)
**Content**: Unique identifier for each track

```
track_ids[i] = track ID corresponding to X_sonic[i]
```

### track_artists

**Shape**: `(n_tracks,)`
**Type**: String (object)
**Content**: Artist name for each track

```
track_artists[i] = "Artist Name" (or "Unknown")
```

### track_titles

**Shape**: `(n_tracks,)`
**Type**: String (object)
**Content**: Track title for each track

```
track_titles[i] = "Track Title" (or "Unknown")
```

### artist_keys

**Shape**: `(n_tracks,)`
**Type**: String (object)
**Content**: Normalized artist identifier for deduplication

```
artist_keys[i] = normalized_artist_name (lowercase, ASCII)
                Used for grouping by artist
```

### genre_vocab

**Shape**: `(n_genres,)`
**Type**: String (object)
**Content**: Genre vocabulary (mapping)

```
genre_vocab[j] = genre name
                Used to map column j in X_genre* to genre string
```

## Artifact Loading Example

```python
import numpy as np
from pathlib import Path

artifact_path = 'data_matrices.npz'
artifact = np.load(artifact_path, allow_pickle=True)

# Access matrices
X_sonic = artifact['X_sonic']              # (34100, 71)
X_genre = artifact['X_genre_smoothed']    # (34100, n_genres)
track_ids = artifact['track_ids']         # (34100,)
track_artists = artifact['track_artists'] # (34100,)

# Map index to track info
idx = 0
print(f"Track {idx}:")
print(f"  ID: {track_ids[idx]}")
print(f"  Artist: {track_artists[idx]}")
print(f"  Title: {track_titles[idx]}")
print(f"  Sonic features: {X_sonic[idx]}")  # 71 dimensions
print(f"  Genre vector: {X_genre[idx]}")    # n_genres dimensions
```

## Relationships

```
┌─────────────┐
│   tracks    │
├─────────────┤
│ track_id ◄──┼───────────┐
│ file_path   │           │
│ artist      │    ┌──────┴──────────────┐
│ title       │    │                     │
│ sonic_*     │    │                     │
└─────────────┘    │                     │
                   │              ┌──────▼─────────┐
              ┌────┴──────┐       │  track_genres  │
              │  albums   │       ├────────────────┤
              ├───────────┤       │ track_genre_id │
              │ album_id  │       │ track_id ─────►├──FK
              │ title     │       │ genre_tag      │
              │ artist    │       │ source         │
              └───────────┘       └────────────────┘
```

## Genre Normalization Example

**Raw genres** (from APIs):
- Last.fm: "Rock", "Alternative Rock", "Indie Rock"
- MusicBrainz: "rock", "indie rock"
- Discogs: "Rock", "Alternative"

**After normalization**:
- Canonical: ["rock", "indie", "alternative"]
- Smoothed distribution: [0.7, 0.2, 0.1]

## Indexes for Performance

Critical indexes for queries:

```sql
-- Fast artist-based queries
CREATE INDEX idx_tracks_artist ON tracks(artist);

-- Fast genre-based queries
CREATE INDEX idx_track_genres_genre ON track_genres(genre_tag);

-- Fast file lookups
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
```

## Database Size Reference

For a typical 1000-track library:

| Component | Size |
|-----------|------|
| metadata.db | 2-5 MB |
| data_matrices.npz | 10-15 MB |
| **Total** | **15-20 MB** |

For 34,100 tracks:

| Component | Size |
|-----------|------|
| metadata.db | 50-100 MB |
| data_matrices.npz | 300-400 MB |
| **Total** | **350-500 MB** |

## Backward Compatibility

The system maintains compatibility between schema versions:

- **sonic_source** column indicates feature extraction method
- Legacy tracks with `windowed` features still work
- New tracks use `beat_sync` features
- Similarity computation handles both transparently

