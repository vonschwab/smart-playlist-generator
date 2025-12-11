# API & Module Reference

Comprehensive reference for all modules in the Playlist Generator system.

---

## Core Modules (`src/`)

### `playlist_generator.py`

**Purpose:** Main playlist generation logic with genre filtering, artist distribution, and duplicate prevention.

**Key Classes:**
- `PlaylistGenerator` - Main class for generating playlists

**Key Methods:**
```python
def __init__(self, config_path='config.yaml')
    """Initialize the playlist generator with configuration."""

def generate_playlists(self, seed_artists=None, num_playlists=8, tracks_per_playlist=30)
    """Generate multiple playlists based on listening history or seed artists.

    Args:
        seed_artists: Optional list of artist names to use as seeds
        num_playlists: Number of playlists to generate
        tracks_per_playlist: Target number of tracks per playlist

    Returns:
        List of playlist dictionaries with track information
    """

def generate_single_playlist(self, seed_track_id, num_tracks=30)
    """Generate a single playlist starting from a seed track.

    Args:
        seed_track_id: Track ID to use as seed
        num_tracks: Number of tracks to include

    Returns:
        List of track dictionaries
    """
```

**Configuration Options:**
- `playlists.count` - Number of playlists to generate
- `playlists.tracks_per_playlist` - Tracks per playlist
- `playlists.max_tracks_per_artist` - Maximum tracks from same artist
- `playlists.artist_window_size` - Window for artist distribution check
- `playlists.min_duration_minutes` - Minimum playlist duration

**File Location:** `src/playlist_generator.py:1`

---

### `similarity_calculator.py`

**Purpose:** Hybrid scoring system combining sonic (60%) and genre (40%) similarity.

**Key Classes:**
- `SimilarityCalculator` - Calculates similarity between tracks

**Key Methods:**
```python
def __init__(self, config_path='config.yaml')
    """Initialize with configuration and genre similarity calculator."""

def calculate_similarity(self, track1_id, track2_id)
    """Calculate hybrid similarity between two tracks.

    Args:
        track1_id: First track ID
        track2_id: Second track ID

    Returns:
        Float similarity score (0.0 to 1.0)
        Returns 0.0 if genre similarity below threshold
    """

def calculate_sonic_similarity(self, features1, features2)
    """Calculate sonic similarity using audio features.

    Args:
        features1: JSON dict of sonic features for track 1
        features2: JSON dict of sonic features for track 2

    Returns:
        Float similarity score (0.0 to 1.0)
    """

def calculate_genre_similarity(self, genres1, genres2)
    """Calculate genre similarity using configured method.

    Args:
        genres1: List of genre strings for track 1
        genres2: List of genre strings for track 2

    Returns:
        Float similarity score (0.0 to 1.0)
    """
```

**Configuration Options:**
- `playlists.genre_similarity.sonic_weight` - Sonic contribution (default: 0.67)
- `playlists.genre_similarity.weight` - Genre contribution (default: 0.33)
- `playlists.genre_similarity.min_genre_similarity` - Minimum threshold (default: 0.2)
- `playlists.genre_similarity.method` - Similarity method (default: ensemble)

**File Location:** `src/similarity_calculator.py:1`

---

### `genre_similarity_v2.py`

**Purpose:** Advanced genre similarity with 7 different calculation methods.

**Key Classes:**
- `GenreSimilarityV2` - Genre similarity calculator with multiple methods

**Key Methods:**
```python
def __init__(self, similarity_file='data/genre_similarity.yaml')
    """Initialize with genre similarity matrix."""

def calculate_similarity(self, genres1, genres2, method='ensemble')
    """Calculate genre similarity using specified method.

    Args:
        genres1: List of genres for first track
        genres2: List of genres for second track
        method: One of: 'ensemble', 'weighted_jaccard', 'cosine',
                'best_match', 'jaccard', 'average_pairwise', 'legacy'

    Returns:
        Float similarity score (0.0 to 1.0)
    """

# Individual methods (all return float 0.0-1.0):
def jaccard_similarity(genres1, genres2)
def weighted_jaccard_similarity(genres1, genres2)
def cosine_similarity(genres1, genres2)
def average_pairwise_similarity(genres1, genres2)
def best_match_similarity(genres1, genres2)
def ensemble_similarity(genres1, genres2)
def legacy_max_similarity(genres1, genres2)
```

**Available Methods:**
- `ensemble` - **Recommended** - Weighted combination (0.15 Jaccard + 0.35 Weighted Jaccard + 0.25 Cosine + 0.25 Best Match)
- `weighted_jaccard` - Relationship-aware set overlap (fast, effective)
- `cosine` - Vector-based similarity using embeddings
- `best_match` - Optimal pairing between genre lists
- `jaccard` - Pure set overlap (strict)
- `average_pairwise` - Average of all genre-to-genre comparisons
- `legacy` - Original maximum similarity method

**File Location:** `src/genre_similarity_v2.py:1`

---

### `local_library_client.py`

**Purpose:** Local library interface providing SQLite queries and track retrieval.

**Key Classes:**
- `LocalLibraryClient` - Interface to local music database

**Key Methods:**
```python
def __init__(self, db_path='data/metadata.db')
    """Initialize connection to local database."""

def get_all_tracks(self)
    """Get all tracks from the database.

    Returns:
        List of track dictionaries with metadata
    """

def get_track_by_id(self, track_id)
    """Get specific track by ID.

    Args:
        track_id: Track ID string

    Returns:
        Track dictionary or None
    """

def search_tracks(self, query, field='title')
    """Search tracks by field.

    Args:
        query: Search query string
        field: Field to search ('title', 'artist', 'album')

    Returns:
        List of matching track dictionaries
    """

def get_tracks_by_artist(self, artist)
    """Get all tracks by an artist.

    Args:
        artist: Artist name

    Returns:
        List of track dictionaries
    """
```

**File Location:** `src/local_library_client.py:1`

---

### `metadata_client.py`

**Purpose:** Database management including schema creation, genre storage, and complex queries.

**Key Classes:**
- `MetadataClient` - Database interface for metadata operations

**Key Methods:**
```python
def __init__(self, db_path='data/metadata.db')
    """Initialize database connection and create schema if needed."""

def get_combined_track_genres(self, track_id)
    """Get combined genres for a track from all sources.

    Priority order: lastfm_track > lastfm_album > musicbrainz_release >
                    lastfm_artist > musicbrainz_artist

    Args:
        track_id: Track ID string

    Returns:
        List of genre strings (deduplicated, priority-ordered)
    """

def add_artist_genre(self, artist, genre, source)
    """Add artist-level genre.

    Args:
        artist: Artist name
        genre: Genre string
        source: 'lastfm_artist' or 'musicbrainz_artist'
    """

def add_album_genre(self, album_id, genre, source)
    """Add album-level genre.

    Args:
        album_id: Album ID string
        genre: Genre string
        source: 'lastfm_album' or 'musicbrainz_release'
    """

def add_track_genre(self, track_id, genre, source)
    """Add track-specific genre.

    Args:
        track_id: Track ID string
        genre: Genre string
        source: 'lastfm_track' or 'file'
    """

def get_track_sonic_features(self, track_id)
    """Get sonic features for a track.

    Returns:
        JSON dict with multi-segment features or None
    """
```

**Database Schema:**
```sql
-- Core tables
tracks (track_id, artist, title, album, file_path, album_id, sonic_features, sonic_source, sonic_analyzed_at)
albums (album_id, artist, title, UNIQUE(artist, title))

-- Genre tables (source-aware)
artist_genres (artist, genre, source, UNIQUE(artist, genre, source))
album_genres (album_id, genre, source, UNIQUE(album_id, genre, source))
track_genres (track_id, genre, source, UNIQUE(track_id, genre, source))
```

**File Location:** `src/metadata_client.py:1`

---

### `lastfm_client.py`

**Purpose:** Last.FM API client for listening history, artist info, and genre tags.

**Key Classes:**
- `LastFMClient` - Last.FM API wrapper

**Key Methods:**
```python
def __init__(self, api_key, username)
    """Initialize with Last.FM credentials."""

def get_recent_tracks(self, days=14)
    """Get user's recent listening history.

    Args:
        days: Number of days to look back

    Returns:
        List of track dictionaries with play counts
    """

def get_artist_tags(self, artist)
    """Get genre tags for an artist.

    Args:
        artist: Artist name

    Returns:
        List of genre strings or None
    """

def get_album_tags(self, artist, album)
    """Get genre tags for an album.

    Args:
        artist: Artist name
        album: Album title

    Returns:
        List of genre strings or None
    """

def get_track_tags(self, artist, track)
    """Get genre tags for a track.

    Args:
        artist: Artist name
        track: Track title

    Returns:
        List of genre strings or None
    """

def get_similar_artists(self, artist, limit=50)
    """Get similar artists.

    Args:
        artist: Artist name
        limit: Maximum results

    Returns:
        List of (artist_name, similarity_score) tuples
    """
```

**Configuration:**
- `lastfm.api_key` - Last.FM API key (get free at https://www.last.fm/api)
- `lastfm.username` - Last.FM username for listening history
- `lastfm.history_days` - Days of history to fetch (default: 90)

**Rate Limiting:** Built-in 1.5s delay between requests

**File Location:** `src/lastfm_client.py:1`

---

### `librosa_analyzer.py`

**Purpose:** Audio feature extraction wrapper using librosa.

**Key Classes:**
- `LibrosaAnalyzer` - Audio analysis using librosa

**Key Methods:**
```python
def analyze_track(self, file_path)
    """Analyze audio file and extract features.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary of audio features or None on error
    """
```

**Extracted Features:**
- MFCC (Mel-frequency cepstral coefficients) - Timbre
- Spectral centroid - Brightness
- Chroma - Harmonic content
- Tempo/BPM
- Key estimation
- RMS energy - Loudness
- Zero crossing rate
- Spectral rolloff
- Spectral bandwidth

**File Location:** `src/librosa_analyzer.py:1`

---

### `hybrid_sonic_analyzer.py`

**Purpose:** Multi-segment sonic analysis (beginning/middle/end + average).

**Key Classes:**
- `HybridSonicAnalyzer` - Multi-segment audio analyzer

**Key Methods:**
```python
def analyze_track(self, file_path, segment_duration=30)
    """Analyze track in multiple segments.

    Args:
        file_path: Path to audio file
        segment_duration: Duration of each segment in seconds (default: 30)

    Returns:
        Dictionary with 'beginning', 'middle', 'end', 'average' features
    """
```

**Output Structure:**
```python
{
    "beginning": { ...features... },  # First 30 seconds
    "middle": { ...features... },     # Center 30 seconds
    "end": { ...features... },        # Last 30 seconds
    "average": { ...features... }     # Averaged across segments
}
```

**Benefits:**
- Captures song dynamics (quiet intro → loud chorus)
- Better whole-song representation
- Future: transition matching for DJ-style flow

**File Location:** `src/hybrid_sonic_analyzer.py:1`

---

### `artist_cache.py`

**Purpose:** Performance optimization through artist similarity caching.

**Key Classes:**
- `ArtistCache` - Caches similar artists from Last.FM

**Key Methods:**
```python
def __init__(self, cache_file='data/artist_cache.json', expiry_days=30)
    """Initialize cache with file path and expiry."""

def get_similar_artists(self, artist)
    """Get cached similar artists or None if not cached/expired.

    Args:
        artist: Artist name

    Returns:
        List of similar artist names or None
    """

def cache_similar_artists(self, artist, similar_artists)
    """Cache similar artists with timestamp.

    Args:
        artist: Artist name
        similar_artists: List of similar artist names
    """

def is_expired(self, artist)
    """Check if cache entry is expired."""
```

**Configuration:**
- `playlists.cache_expiry_days` - Cache expiry in days (default: 30)

**File Location:** `src/artist_cache.py:1`

---

### `m3u_exporter.py`

**Purpose:** Exports playlists to M3U format.

**Key Classes:**
- `M3UExporter` - M3U playlist file writer

**Key Methods:**
```python
def export_playlist(self, playlist, output_path)
    """Export playlist to M3U file.

    Args:
        playlist: List of track dictionaries
        output_path: Path for output .m3u file

    Returns:
        True on success, False on error
    """
```

**M3U Format:**
```
#EXTM3U
#EXTINF:240,Artist - Track Title
/path/to/track.mp3
...
```

**Configuration:**
- `playlists.export_m3u` - Enable/disable M3U export (default: true)
- `playlists.m3u_export_path` - Output directory for M3U files

**File Location:** `src/m3u_exporter.py:1`

---

### `config_loader.py`

**Purpose:** Configuration file manager for YAML config.

**Key Classes:**
- `ConfigLoader` - YAML configuration loader

**Key Methods:**
```python
def load_config(config_path='config.yaml')
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dictionary of configuration values
    """

def validate_config(config)
    """Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises exception if invalid
    """
```

**File Location:** `src/config_loader.py:1`

---

### `rate_limiter.py`

**Purpose:** API rate limiting for Last.FM and MusicBrainz.

**Key Classes:**
- `RateLimiter` - Rate limiter with configurable delay

**Key Methods:**
```python
def __init__(self, delay_seconds=1.5)
    """Initialize with delay between requests."""

def wait(self)
    """Wait if needed to respect rate limit."""
```

**Default Delays:**
- Last.FM: 1.5 seconds
- MusicBrainz: 1.1 seconds

**File Location:** `src/rate_limiter.py:1`

---

### `retry_helper.py`

**Purpose:** Retry logic for API calls with exponential backoff.

**Key Functions:**
```python
def retry_with_backoff(func, max_retries=3, initial_delay=1.0)
    """Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Function result or raises last exception
    """
```

**File Location:** `src/retry_helper.py:1`

---

## Utility Scripts (`scripts/`)

### `scan_library.py`

**Purpose:** Scans music directory and populates database.

**Usage:**
```bash
python scripts/scan_library.py
```

**Features:**
- Multi-format support (MP3, FLAC, M4A, OGG, WMA)
- Duplicate prevention by file path
- Generates unique track IDs (MD5 hash)
- Creates album records
- Extracts embedded genres

**File Location:** `scripts/scan_library.py:1`

---

### `update_genres_v3_normalized.py`

**Purpose:** Genre metadata updater using normalized schema.

**Usage:**
```bash
python scripts/update_genres_v3_normalized.py --artists
python scripts/update_genres_v3_normalized.py --albums
python scripts/update_genres_v3_normalized.py --tracks
python scripts/update_genres_v3_normalized.py --stats
```

**Features:**
- Normalized schema (60% fewer API calls)
- Multi-source: Last.FM + MusicBrainz
- Source tracking
- Incremental updates
- Empty markers
- Resumable

**File Location:** `scripts/update_genres_v3_normalized.py:1`

---

### `update_sonic.py`

**Purpose:** Sonic feature analyzer with multi-segment analysis.

**Usage:**
```bash
python scripts/update_sonic.py --workers 6
python scripts/update_sonic.py --limit 100
python scripts/update_sonic.py --stats
```

**Features:**
- Multi-segment analysis (beginning/middle/end)
- Parallel processing
- Progress reporting
- Incremental (skips analyzed)
- Error handling

**File Location:** `scripts/update_sonic.py:1`

---

### `validate_metadata.py`

**Purpose:** Database validator and statistics reporter.

**Usage:**
```bash
python scripts/validate_metadata.py
```

**Checks:**
- Database integrity
- Track counts
- Genre coverage
- Sonic analysis progress
- Missing data
- Anomalies

**File Location:** `scripts/validate_metadata.py:1`

---

## Configuration Reference

Complete configuration options in `config.yaml`:

```yaml
library:
  music_directory: "E:/MUSIC"
  database_path: "data/metadata.db"

lastfm:
  api_key: "your_api_key_here"
  username: "your_username"
  history_days: 90

openai:
  api_key: "your_openai_key"
  model: "gpt-4o-mini"

playlists:
  count: 8
  tracks_per_playlist: 30
  history_days: 14
  seed_count: 5
  similar_per_seed: 20
  name_prefix: "Auto:"

  genre_similarity:
    enabled: true
    weight: 0.33
    sonic_weight: 0.67
    min_genre_similarity: 0.2
    method: "ensemble"
    similarity_file: "data/genre_similarity.yaml"
    broad_filters: [
      "50s", "60s", "70s", "80s", "90s", "00s", "10s", "20s",
      "rock", "pop", "electronic", "metal", "jazz", "hip-hop",
      "favorites", "seen live", "albums i own", ...
    ]

  max_tracks_per_artist: 4
  artist_window_size: 8
  max_artist_per_window: 1
  min_duration_minutes: 90
  min_track_duration_seconds: 46

  export_m3u: true
  m3u_export_path: "E:/PLAYLISTS"

  cache_expiry_days: 30

  similar_artists:
    enabled: true
    boost: 0.05
```

---

## Error Handling

All modules implement consistent error handling:

1. **API Errors**: Retry with exponential backoff (max 3 attempts)
2. **File Errors**: Log and continue processing
3. **Database Errors**: Rollback transaction and log
4. **Configuration Errors**: Validate and raise clear exceptions

---

## Logging

Logs are written to the `logs/` directory:

- `playlist_generator.log` - Main application logs
- `sonic_analysis.log` - Sonic analysis progress
- `genre_update_v3.log` - Genre update progress

Log format: `[YYYY-MM-DD HH:MM:SS] [LEVEL] [module] message`

---

**Last Updated:** December 11, 2025
