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

## Mode-Based Configuration (Simplified Tuning)

**NEW in v3.2:** Instead of manually tuning genre/sonic weights and thresholds, you can use simple mode presets that configure everything automatically.

### Genre Modes

Control how strictly playlists match genre tags:

| Mode | Description | Genre Weight | Threshold | Use Case |
|------|-------------|--------------|-----------|----------|
| `strict` | Ultra-tight genre matching | 0.80 | 0.50 | Single-genre deep dives |
| `narrow` | Stay close to seed genre | 0.65 | 0.40 | Cohesive genre exploration |
| `dynamic` | Balanced exploration | 0.50 | 0.30 | General use (default) |
| `discover` | Genre-adjacent exploration | 0.35 | 0.20 | Cross-genre discovery |
| `off` | Sonic-only mode | 0.00 | 0.00 | Ignore genre tags completely |

### Sonic Modes

Control how strictly playlists match audio features (rhythm, timbre, harmony):

| Mode | Description | Sonic Weight | Pool Control | Use Case |
|------|-------------|--------------|--------------|----------|
| `strict` | Ultra-tight sonic matching | 0.85 | Tight pool | Laser-focused sound |
| `narrow` | Cohesive sound | 0.70 | Tighter pool | Consistent texture |
| `dynamic` | Balanced sonic flow | 0.50 | Standard pool | General use (default) |
| `discover` | Varied textures | 0.35 | Wider pool | Sonic variety |
| `off` | Genre-only mode | 0.00 | Maximum pool | Ignore sonic features |

### Configuration Examples

**Balanced (default):**
```yaml
playlists:
  genre_mode: dynamic      # 50% genre weight
  sonic_mode: dynamic      # 50% sonic weight
```

**Ultra-cohesive playlists:**
```yaml
playlists:
  genre_mode: strict       # 80% genre, threshold 0.50
  sonic_mode: strict       # 85% sonic, tight pool
```

**Same genre, varied sound:**
```yaml
playlists:
  genre_mode: narrow       # 65% genre, threshold 0.40
  sonic_mode: discover     # 35% sonic, wide pool
```

**Pure sonic similarity:**
```yaml
playlists:
  genre_mode: off          # Genre disabled
  sonic_mode: dynamic      # 50% sonic weight
```

**Pure genre matching:**
```yaml
playlists:
  genre_mode: dynamic      # 50% genre weight
  sonic_mode: off          # Sonic disabled
```

### CLI Override

Override modes for a single run without changing config:

```bash
# Strict genre + narrow sonic
python main_app.py --artist "Radiohead" --genre-mode strict --sonic-mode narrow

# Pure sonic (genre off)
python main_app.py --genre "ambient" --genre-mode off --sonic-mode dynamic

# Discovery mode
python main_app.py --genre "jazz" --genre-mode discover --sonic-mode discover
```

### GUI Override

The GUI playlist tab includes dropdown selectors for both modes. Selected modes override config file settings for that generation only.

### Technical Details

Mode presets automatically configure multiple underlying parameters:

**Genre modes set:**
- `playlists.genre_similarity.weight` (genre weight in hybrid score)
- `playlists.genre_similarity.sonic_weight` (sonic weight in hybrid score)
- `playlists.genre_similarity.min_genre_similarity` (filter threshold)
- `playlists.ds_pipeline.candidate_pool.min_sonic_similarity_<mode>` (mode-specific sonic floor)

**Sonic modes set:**
- `playlists.genre_similarity.sonic_weight` (sonic weight in hybrid score)
- `playlists.genre_similarity.weight` (genre weight in hybrid score)
- `playlists.ds_pipeline.candidate_pool.similarity_floor` (hybrid similarity floor)

**Mode precedence (highest to lowest):**
1. CLI flags (`--genre-mode`, `--sonic-mode`)
2. GUI selections
3. Config file (`genre_mode`, `sonic_mode`)
4. Manual parameters (`weight`, `min_genre_similarity`, etc.)

If mode settings are present, they OVERRIDE manual `genre_similarity` settings.

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
      duration_penalty_enabled: true   # Penalize long tracks vs seeds
      duration_penalty_weight: 0.60    # Penalty strength (higher = more severe)
      duration_cutoff_multiplier: 2.5 # Hard cutoff vs median seed duration

    # Scoring weights
    scoring:
      alpha: 0.55                    # Seed similarity weight
      beta: 0.55                     # Transition similarity weight
      gamma: 0.04                    # Diversity bonus
      alpha_schedule: arc            # constant or arc

    # Constraints
    constraints:
      min_gap: 6                     # Tracks between same artist

      # Artist identity resolution (optional, default: disabled)
      # Collapses ensemble variants ("Bill Evans Trio" → "bill evans")
      # and splits collaborations ("A & B" → both A and B count for min_gap)
      artist_identity:
        enabled: false               # Enable identity-based min_gap enforcement
        strip_trailing_ensemble_terms: true  # Remove "Trio", "Quartet", etc.
        trailing_ensemble_terms:     # List of terms to strip (multi-word first)
          - "big band"
          - "chamber orchestra"
          - "symphony orchestra"
          - "string quartet"
          - "orchestra"
          - "ensemble"
          - "trio"
          - "quartet"
          - "quintet"
          - "sextet"
          - "septet"
          - "octet"
          - "nonet"
          - "group"
          - "band"
        split_delimiters:            # Collaboration delimiters to split on
          - ","
          - " & "
          - " and "
          - " feat. "
          - " feat "
          - " featuring "
          - " ft. "
          - " ft "
          - " with "
          - " x "
          - " + "

      hard_floor: true               # Reject vs penalize bad transitions
      transition_floor: 0.20         # Minimum transition quality

    # Repair pass (fixes weak transitions)
    repair:
      enabled: true
      max_iters: 5
      max_edges: 5

    # Duration penalty (geometric): penalizes candidates during pool construction
    # based on percentage excess over the median seed track. This is a soft penalty
    # (not a hard filter) but reduces seed similarity so long tracks fall below
    # similarity floors before pier-bridge. Candidates longer than
    # duration_cutoff_multiplier * median seed duration are hard excluded.

    # Four-phase geometric curve (percentage-based):
    # Phase 1 (0-20% excess):   Gentle penalties (power 1.5)
    # Phase 2 (20-50% excess):  Moderate penalties (power 2.0)
    # Phase 3 (50-100% excess): Steep penalties (power 2.5)
    # Phase 4 (>100% excess):   Severe penalties (power 3.0)

    # Example with weight=0.60, reference=200s seed track (duration_cutoff_multiplier=2.5):
    #   +5% (210s):   penalty ≈ 0.006 (negligible)
    #   +20% (240s):  penalty ≈ 0.030 (gentle)
    #   +40% (280s):  penalty ≈ 0.20 (moderate)
    #   +80% (360s):  penalty ≈ 0.90 (steep)
    #   +100% (400s): penalty ≈ 1.50 (severe threshold)
    #   +150% (500s): penalty ≈ 3.19 (very severe)
    #   +160% (520s): excluded (hard cutoff at 2.5x)
```

### Pier-Bridge Tuning (per mode)
Pier-bridge uses per-mode defaults when you don’t specify overrides:
- `dynamic`: `transition_floor=0.35`, `bridge_floor=0.03`, weights `bridge=0.6`, `transition=0.4`
- `narrow`: `transition_floor=0.45`, `bridge_floor=0.08`, weights `bridge=0.7`, `transition=0.3`

Overrides are supported via:
- `playlists.ds_pipeline.constraints.transition_floor_dynamic`
- `playlists.ds_pipeline.constraints.transition_floor_narrow`
- `playlists.ds_pipeline.pier_bridge.*` (see `config.example.yaml` for the full list, including `soft_genre_penalty_threshold/strength`)

### Segment-Local Bridging + Progress Constraint
Pier-bridge now builds each bridge segment (A → B) from a **segment-local** candidate universe scored against **both endpoints** (no neighbor-list union pooling).

Config (see `config.example.yaml`):
- `playlists.ds_pipeline.pier_bridge.segment_pool_strategy`: `segment_scored` (recommended)
- `playlists.ds_pipeline.pier_bridge.segment_pool_max`: per-segment candidate cap (top-K by bridge score)
- `playlists.ds_pipeline.pier_bridge.progress.*`: optional A→B progress constraint to reduce bouncing/teleports

### Artist Identity Resolution (min_gap Enforcement)

**NEW in v3.2:** Artist identity-based min_gap enforcement collapses ensemble variants and splits collaborations.

**Problem:** Without identity resolution, `min_gap=6` treats these as distinct artists:
- "Bill Evans"
- "Bill Evans Trio"
- "Bill Evans Quintet"

This allows them to cluster together, violating the spirit of artist diversity.

**Solution:** When `playlists.ds_pipeline.constraints.artist_identity.enabled=true`:
1. **Ensemble variants collapse to core identity:**
   - "Bill Evans Trio" → `"bill evans"`
   - "Ahmad Jamal Quintet" → `"ahmad jamal"`
   - "Duke Ellington Orchestra" → `"duke ellington"`

2. **Collaboration strings split into participant identities:**
   - "Bob Brookmeyer & Bill Evans" → `{"bob brookmeyer", "bill evans"}`
   - Both participants count for min_gap enforcement

3. **Cross-segment boundary tracking:**
   - Recent identity keys from the last `min_gap` positions are tracked across pier-bridge segments
   - Prevents "Bill Evans Trio" in segment N from being followed by "Bill Evans" in segment N+1

**Configuration:**
```yaml
playlists:
  ds_pipeline:
    constraints:
      min_gap: 6
      artist_identity:
        enabled: true  # Enable identity-based matching
        strip_trailing_ensemble_terms: true
        trailing_ensemble_terms:  # Multi-word terms checked first
          - "big band"
          - "chamber orchestra"
          - "orchestra"
          - "trio"
          - "quartet"
          # ... (see full list in config)
        split_delimiters:
          - ","
          - " & "
          - " feat. "
          # ... (see full list in config)
```

**Behavior when disabled (default):**
- Falls back to raw artist string normalization (diacritics, punctuation, case)
- Ensemble variants treated as distinct artists
- Collaboration strings treated as single artist

**Debug logging:**
- When enabled, logs: `"Artist identity resolution enabled for min_gap enforcement"`
- Candidate rejections logged at DEBUG level: `"Rejected candidate idx=... due to identity_min_gap: key=... distance<=..."`

### Artist Playlists: Seed Artist = Piers Only (default)
For `--artist` playlists, the DS pipeline marks the run as an artist playlist and pier-bridge defaults to **disallowing the seed artist in bridge interiors** (the artist appears only as pier tracks).

To override for any run (optional):
- `playlists.ds_pipeline.pier_bridge.disallow_seed_artist_in_interiors: false`
- `playlists.ds_pipeline.pier_bridge.disallow_pier_artists_in_interiors: true`

### How to verify
- Run `python main_app.py --artist "Sabrina Carpenter" --ds-mode dynamic --log-level INFO --dry-run`
- Confirm the run logs include: `Pier-bridge tuning resolved: mode=dynamic ...`
- Confirm the run logs include: `Pier-bridge segment policy: artist_playlist=... strategy=...`
- For penalty visibility, re-run with `--log-level DEBUG` and look for per-segment `soft_genre_penalty_hits=... edges_scored=...` lines.

### Recency Filtering Invariant (DS)
Recency exclusions (Last.fm scrobbles and/or local history) are applied **pre-order only** during candidate selection. After pier-bridge ordering completes, the playlist is **not** filtered/shrunk; we do **validation only** and fail loudly if constraints are violated.

Log evidence (per run):
- `stage=candidate_pool | Last.fm recency exclusions: before=... after=... excluded=... lookback_days=...`
- `stage=post_order_validation | recency_overlap=0 | final_size=... | expected=...`

Run audit evidence (when `playlists.ds_pipeline.pier_bridge.audit_run.enabled=true` or `--audit-run`):
- `## 3b) Recency (Pre-Order)`
- `post_order_filters_applied: []` and `post_order_validation.recency_overlap_count == 0`

### Infeasible Segments: Bridge-Floor Backoff + Run Audits (optional)
By default, pier-bridge fails loudly if any segment is infeasible under the current `bridge_floor` and allowed pool.

Optional knobs (all default OFF) under `playlists.ds_pipeline.pier_bridge`:
- `infeasible_handling.enabled`: retry infeasible segments with lower `bridge_floor` (deterministic backoff list).
- `audit_run.enabled`: write a markdown report per run (success + failure) to `docs/run_audits/`.

CLI helpers:
- Enable audits: `--audit-run` (and optionally `--audit-run-dir <path>`)
- Enable backoff: `--pb-backoff`

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
  file: logs/playlist_generator.log  # Log file path (in logs/ directory)
```

## Environment Variable Overrides

Some settings can be overridden via environment variables:

| Variable | Purpose |
|----------|---------|
| `SONIC_SIM_VARIANT` | Sonic preprocessing variant |
| `SONIC_TOWER_PCA` | PCA dimensions (comma-separated) |
| `SONIC_TOWER_WEIGHTS` | Tower weights (comma-separated) |
| `PLEX_TOKEN` | Plex authentication token |
