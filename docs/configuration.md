# Configuration Guide

Complete reference for all configuration options in `config.yaml`.

## Setup

### Initial Configuration

Copy the example configuration and customize:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## Configuration Sections

### Music Library

Configure where your music files are stored.

```yaml
library:
  music_directory: E:/MUSIC        # Path to your music library
  database_path: data/metadata.db  # SQLite database location
```

**Options**:
- `music_directory` (required): Absolute path to music folder. Supports nested directories with 5+ audio file formats.
- `database_path` (optional): Location for SQLite metadata database. Default: `data/metadata.db`

### Last.FM Integration (Optional)

Fetch listening history and genre tags from Last.FM.

```yaml
lastfm:
  api_key: ""           # Get free from https://www.last.fm/api
  username: ""          # Your Last.FM username
  history_days: 90      # Days of history to fetch
```

**Options**:
- `api_key` (optional): Last.FM API key for genre fetching. Leave blank to skip genre APIs.
- `username` (optional): Your Last.FM username for listening history integration.
- `history_days` (default: 90): How many days back to fetch listening history.

### OpenAI Integration (Optional)

Generate smart playlist names using AI.

```yaml
openai:
  api_key: YOUR_OPENAI_API_KEY
  model: gpt-4o-mini  # Fast + low-cost
```

**Options**:
- `api_key` (optional): OpenAI API key for playlist naming.
- `model` (default: `gpt-4o-mini`): Model for generation (fast + inexpensive).

### Plex Integration (Optional)

Export generated playlists directly to Plex.

```yaml
plex:
  enabled: false
  base_url: http://localhost:32400
  token: ""              # Or set PLEX_TOKEN env var
  music_section: Music   # Name of your Plex music library
  verify_ssl: true
  replace_existing: true
  path_map: []           # Optional path remapping for Plex file paths
```

**Options**:
- `enabled` (default: false): Turn Plex export on/off.
- `base_url`: Plex server URL.
- `token`: Plex token (use `PLEX_TOKEN` env var instead of storing in config).
- `music_section`: Plex music library section name.
- `verify_ssl` (default: true): SSL verification for HTTPS servers.
- `replace_existing` (default: true): Replace playlist if title already exists.
- `path_map` (optional): List of `{from, to}` mappings to align local file paths with Plex paths.

## Playlist Generation

### Basic Settings

```yaml
playlists:
  count: 8                      # Number of playlists to generate
  tracks_per_playlist: 30       # Target tracks per playlist
  history_days: 14              # Look back for recent listening
  seed_count: 5                 # Number of seed tracks to use
  similar_per_seed: 20          # Similar tracks per seed
  name_prefix: 'Auto:'          # Prefix for generated playlists
  max_age_days: 14              # Max age of generated playlists
```

### M3U Export

```yaml
playlists:
  export_m3u: true              # Export as M3U files
  m3u_export_path: E:/PLAYLISTS # Where to save M3U files
```

### Caching

```yaml
playlists:
  cache_expiry_days: 30         # Cache validity in days
```

### Duration Filtering

```yaml
playlists:
  min_duration_minutes: 90      # Minimum total playlist duration
  min_track_duration_seconds: 46   # Skip tracks shorter than this
  max_track_duration_seconds: 720  # Skip tracks longer than this (12 min)
```

### Artist Distribution

Control how many tracks per artist appear in playlists.

```yaml
playlists:
  max_tracks_per_artist: 4      # Max tracks from same artist in entire playlist
  artist_window_size: 8         # Window size for diversity check
  max_artist_per_window: 1      # Max same artist in a window
  min_seed_artist_ratio: 0.125  # Minimum ratio of seed artist tracks (12.5%)
```

**Example**:
- `max_tracks_per_artist: 4` → No artist appears more than 4 times
- `artist_window_size: 8` → In any 8-track window, max 1 track per artist
- `min_seed_artist_ratio: 0.125` → Seed artist gets 12.5% of playlist

### Recently Played Filtering

Avoid re-playing recent songs (optional).

```yaml
playlists:
  recently_played_filter:
    enabled: true               # Enable/disable filter
    lookback_days: 30           # Don't play tracks from last 30 days
    min_playcount_threshold: 0  # Tracks played at least N times
```

## Scoring & Similarity

### Pipeline Selection

```yaml
playlists:
  pipeline: ds  # Options: "legacy" | "ds"
```

**Options**:
- `ds` (recommended): Data Science pipeline with hybrid scoring
- `legacy`: Simple sonic distance (baseline)

### Data Science Pipeline

Advanced scoring with beat-synchronized features.

```yaml
playlists:
  ds_pipeline:
    artifact_path: data/artifacts/beat3tower_32k/data_matrices_step1.npz
    mode: dynamic  # Options: dynamic, ds, narrow, discover, legacy
    random_seed: 0
    enable_logging: true
```

**Modes**:
- `ds`: Balanced hybrid (60% sonic + 40% genre) - **recommended**
- `dynamic`: Progressive emphasis on transitions
- `narrow`: Strict genre coherence
- `discover`: Genre exploration
- `legacy`: Pure sonic similarity

### Sonic Features

```yaml
playlists:
  sonic:
    sim_variant: tower_pca  # Options: raw | centered | z | z_clip | whiten_pca | robust_whiten | tower_l2 | tower_robust | tower_iqr | tower_weighted | tower_pca
```

**Variants**:
- `tower_pca`: Standardize + PCA each tower (8/16/8 components), apply weights (0.2/0.5/0.3), concatenate (default, production variant)
- `tower_weighted`: L2 normalize each tower, apply weights (0.2/0.5/0.3), then concatenate
- `tower_robust`: Robust scale each tower (median/IQR), then concatenate
- `tower_iqr`: IQR scale each tower (no centering), then concatenate
- `tower_l2`: L2 normalize rhythm/timbre/harmony towers, then concatenate
- `robust_whiten`: Robust scale (median/IQR) then PCA whitening
- `whiten_pca`: Z-score then PCA whitening
- `z_clip`: Z-score with outlier clipping
- `z`: Z-score normalization
- `centered`: Zero-centered before normalization
- `raw`: L2-normalized only (fastest but lower quality)

### Genre Similarity

Control how genre impacts playlist generation.

```yaml
playlists:
  genre_similarity:
    enabled: true                  # Enable genre filtering
    weight: 0.50                   # Genre contribution (normalized with sonic_weight)
    sonic_weight: 0.60             # Sonic contribution (normalized with weight)
    min_genre_similarity: 0.2       # Minimum genre similarity threshold
    min_genre_similarity_narrow: 0.45  # Narrow-mode override (stricter)
    method: ensemble               # Similarity method
    similarity_file: data/genre_similarity.yaml
    use_artist_tags: true          # Include artist-level genres
```

**Methods**:
- `ensemble` (recommended): Weighted combination of 4 methods
- `weighted_jaccard`: Relationship-aware set overlap
- `cosine`: Vector-based similarity
- `best_match`: Optimal pairing
- `jaccard`: Pure set overlap
- `average_pairwise`: Average of all comparisons
- `legacy`: Original max similarity

**Broad Genre Filters** (exclude these tags):

Removes non-musical tags like "50s", "favorites", "seen live", etc. These tags don't represent actual genres.

```yaml
playlists:
  genre_similarity:
    broad_filters:
      - "50s"
      - "60s"
      # ... decades ...
      - "rock"
      - "pop"
      # ... avoid broad categories ...
      - "favorites"
      - "seen live"
      - "unknown"
```

### Similarity Scoring

```yaml
playlists:
  similarity:
    min_threshold: 0.5              # Minimum similarity to include track
    artist_direct_match: 0.9        # Boost for same artist
    artist_shared_base: 0.4         # Base boost for shared base artist
    artist_shared_increment: 0.05   # Increment per shared characteristic
    artist_shared_max: 0.7          # Maximum shared artist boost
```

### Candidate Pool

```yaml
playlists:
  limits:
    similar_tracks: 50              # Max candidate pool size
    similar_artists: 30             # Max similar artists to consider
    extension_base: 10              # Base extension pool
    extension_increment: 20         # Increment for each fallback
```

### Duration Matching

Prefer tracks near a target duration.

```yaml
playlists:
  duration_match:
    enabled: true
    weight: 0.35                    # Influence on scoring (multiplicative)
    window_frac: 0.25               # ±25% treated as near-neutral
    falloff: 0.6                    # Steeper/smoother tail outside window
    min_target_seconds: 40          # Ignore tracks shorter than this
```

**Example**:
- Target: 240 seconds (4 minutes)
- Window: ±25% = 180-300 seconds (neutral penalty)
- Outside window: Penalty increases with falloff=0.6 (smoother curve)

### Title Deduplication

Prevent duplicate songs (same title, different versions).

```yaml
playlists:
  dedupe:
    title:
      enabled: true                 # Enable deduplication
      threshold: 92                 # Fuzzy match threshold (0-100)
      mode: loose                   # "strict" | "loose"
      short_title_min_len: 6        # Titles <6 chars need exact match
```

**Modes**:
- `loose`: Removes version tags (Live, Remastered, etc.)
- `strict`: Requires exact match

**Example**:
- "Hey Jude" + "Hey Jude - Live" → Deduplicated (loose mode)
- "Hey Jude" + "Hey Jude (2024 Remaster)" → Deduplicated (loose mode)

## Logging

```yaml
logging:
  level: DEBUG              # DEBUG | INFO | WARNING | ERROR
  file: playlist_generator.log
```

**Levels**:
- `DEBUG`: Verbose output (all operations)
- `INFO`: Important events only
- `WARNING`: Warnings and errors
- `ERROR`: Errors only

## Common Configurations

### Minimal Setup (No APIs)

```yaml
library:
  music_directory: /path/to/music

playlists:
  count: 5
  tracks_per_playlist: 30
  pipeline: ds

logging:
  level: INFO
```

### Full Setup (All APIs)

```yaml
library:
  music_directory: /path/to/music

lastfm:
  api_key: your_api_key
  username: your_username
  history_days: 90

openai:
  api_key: your_openai_key
  model: gpt-4o-mini

playlists:
  count: 8
  tracks_per_playlist: 50
  pipeline: ds
  ds_pipeline:
    mode: dynamic
  genre_similarity:
    enabled: true
    weight: 0.40
  recently_played_filter:
    enabled: true
    lookback_days: 30
  export_m3u: true
  m3u_export_path: /path/to/playlists

logging:
  level: INFO
```

### Performance-Focused

```yaml
playlists:
  pipeline: ds            # DS pipeline required
  sonic:
    sim_variant: raw      # Fastest (lower quality than tower_pca)
  genre_similarity:
    enabled: false        # Skip genre computation
  cache_expiry_days: 60   # Cache longer
```

### Quality-Focused

```yaml
playlists:
  pipeline: ds
  ds_pipeline:
    mode: dynamic         # Better flow
  sonic:
    sim_variant: tower_pca  # Production variant (default)
  genre_similarity:
    enabled: true
    weight: 0.50          # Genre influence
    method: ensemble      # Best accuracy
  duration_match:
    enabled: true
    weight: 0.35          # Duration preference
```

## Validation

Check your configuration:

```bash
python -c "from src.config_loader import ConfigLoader; cfg = ConfigLoader.load_config(); print('✓ Config valid')"
```

## Troubleshooting

### "music_directory not found"
- Verify path exists and is readable
- Use absolute paths (not relative)
- Use forward slashes `/` on all platforms

### "database_path permission denied"
- Ensure parent directory is writable
- Check file permissions
- Close other database connections

### "API key invalid"
- Verify key in Last.FM API dashboard
- Check for extra spaces or characters
- Keys are case-sensitive

### Config changes not taking effect
- Restart the application
- Clear cache if applicable
- Check `enable_logging: true` to debug

## Environment Variables (Optional)

Override config values with environment variables (uppercase, underscores):

```bash
# Override music directory
export LIBRARY_MUSIC_DIRECTORY=/path/to/music
export LASTFM_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## Next Steps

- [Quick Start](quickstart.md) - First-time setup
- [API Reference](api.md) - Use programmatically
- [Pipelines](pipelines.md) - Data processing workflows

