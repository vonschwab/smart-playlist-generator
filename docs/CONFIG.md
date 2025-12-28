# Configuration Reference

This document describes all configuration options in `config.yaml`.

## Library Settings

```yaml
library:
  music_directory: /path/to/music    # Root directory to scan
  database_path: data/metadata.db    # SQLite database path
```

## API Credentials

### Last.FM (optional, for listening history)
```yaml
lastfm:
  api_key: "your_api_key"            # Get from last.fm/api
  username: "your_username"          # Your Last.FM username
  history_days: 90                   # Days of history to fetch
```

### Discogs (optional, for album genres)
```yaml
discogs:
  token: "your_token"                # Get from discogs.com/settings/developers
```

### OpenAI (optional, for AI title generation)
```yaml
openai:
  api_key: "your_api_key"            # Get from platform.openai.com
  model: gpt-4o-mini                 # Model to use
```

### Plex (optional, for Plex export)
```yaml
plex:
  enabled: false
  base_url: http://localhost:32400
  token: "your_token"
  music_section: Music
  verify_ssl: true
  replace_existing: true
  path_map: []                       # Path translation for Docker
```

## Playlist Generation

```yaml
playlists:
  count: 8                           # Playlists to generate per batch
  tracks_per_playlist: 30            # Target tracks per playlist
  name_prefix: "Auto:"               # Prefix for generated playlists
  export_m3u: true                   # Enable M3U export
  m3u_export_path: /path/to/playlists # Where to save M3U files

  # Track constraints
  min_duration_minutes: 90           # Minimum playlist duration
  min_track_duration_seconds: 46     # Skip short tracks
  max_track_duration_seconds: 720    # Skip very long tracks
  max_tracks_per_artist: 3           # Artist diversity

  # DS Pipeline selection
  pipeline: ds                       # Use DS pipeline (recommended)
```

## DS Pipeline Configuration

```yaml
playlists:
  ds_pipeline:
    artifact_path: data/artifacts/beat3tower_32k/data_matrices_step1.npz
    mode: dynamic                    # narrow, dynamic, discover, sonic_only
    random_seed: 0                   # For reproducibility (0 = random)
    enable_logging: true

    # Tower weights for similarity (must sum to 1.0)
    tower_weights:
      rhythm: 0.20                   # Tempo/beat importance
      timbre: 0.50                   # Texture/tone importance
      harmony: 0.30                  # Key/chord importance

    # Transition weights (end-of-A to start-of-B)
    transition_weights:
      rhythm: 0.40
      timbre: 0.35
      harmony: 0.25

    # PCA dimensions per tower
    tower_pca_dims:
      rhythm: 8                      # Max 21
      timbre: 16                     # Max 83
      harmony: 8                     # Max 33

    # Candidate pool settings
    candidate_pool:
      similarity_floor: 0.20         # Minimum similarity to consider
      max_pool_size: 1200            # Maximum candidates
      max_artist_fraction: 0.125     # Max 12.5% from one artist

    # Scoring weights
    scoring:
      alpha: 0.55                    # Seed similarity weight
      beta: 0.55                     # Transition similarity weight
      gamma: 0.04                    # Diversity bonus
      alpha_schedule: arc            # constant or arc

    # Constraints
    constraints:
      min_gap: 6                     # Tracks between same artist
      hard_floor: true               # Reject vs penalize bad transitions
      transition_floor: 0.20         # Minimum transition quality

    # Repair pass (fixes weak transitions)
    repair:
      enabled: true
      max_iters: 5
      max_edges: 5
```

## Genre Similarity

```yaml
playlists:
  genre_similarity:
    enabled: true
    weight: 0.50                     # Genre weight in hybrid score
    sonic_weight: 0.50               # Sonic weight in hybrid score
    min_genre_similarity: 0.30       # Filter threshold
    method: ensemble                 # Similarity method
    similarity_file: data/genre_similarity.yaml
    use_artist_tags: true

    # Broad filters (excluded from matching)
    broad_filters:
      - rock
      - pop
      - indie
      # ... list of overly broad tags
```

## Sonic Preprocessing

```yaml
playlists:
  sonic:
    sim_variant: tower_pca           # Preprocessing variant
```

Available variants:
- `tower_pca` (default) - Per-tower PCA with weighting
- `robust_whiten` - Robust scaling + PCA whitening
- `raw` - No preprocessing (not recommended)

Override via CLI: `--sonic-variant tower_pca`
Override via env: `SONIC_SIM_VARIANT=tower_pca`

## Title Deduplication

```yaml
playlists:
  dedupe:
    title:
      enabled: true
      threshold: 92                  # Fuzzy match threshold (0-100)
      mode: loose                    # loose or strict
      short_title_min_len: 6         # Min length for matching
```

## Logging

```yaml
logging:
  level: INFO                        # DEBUG, INFO, WARNING, ERROR
  file: playlist_generator.log       # Log file path
```

## Environment Variable Overrides

Some settings can be overridden via environment variables:

| Variable | Purpose |
|----------|---------|
| `SONIC_SIM_VARIANT` | Sonic preprocessing variant |
| `SONIC_TOWER_PCA` | PCA dimensions (comma-separated) |
| `SONIC_TOWER_WEIGHTS` | Tower weights (comma-separated) |
| `PLEX_TOKEN` | Plex authentication token |
