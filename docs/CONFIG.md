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

**UPDATED in v3.4.1 (Phase 1-3A):** Threshold relaxation to improve feasibility for narrow/dynamic modes. Strict mode now provides the ultra-cohesive filtering that narrow previously attempted. Narrow mode allows justified exploration while maintaining coherence.

### Genre Modes

Control how strictly playlists match genre tags:

| Mode | Description | Genre Weight | Threshold (Narrow) | Use Case |
|------|-------------|--------------|-------------------|----------|
| `strict` | Ultra-tight genre matching (may fail) | 0.80 | 0.60 | Single-genre deep dives |
| `narrow` | Cohesive with justified exploration | 0.65 | 0.42 (was 0.50) | Familiar genre space |
| `dynamic` | Balanced exploration | 0.50 | 0.25 (was 0.30) | General use (default) |
| `discover` | Genre-adjacent exploration | 0.35 | 0.20 | Cross-genre discovery |
| `off` | Sonic-only mode | 0.00 | N/A | Ignore genre tags completely |

### Sonic Modes

Control how strictly playlists match audio features (rhythm, timbre, harmony):

| Mode | Description | Sonic Weight | Sonic Floor | Use Case |
|------|-------------|--------------|-------------|----------|
| `strict` | Ultra-tight sonic matching (may fail) | 0.85 | 0.20 | Laser-focused sound |
| `narrow` | Cohesive sound with flexibility | 0.70 | 0.12 (was 0.18) | Consistent texture |
| `dynamic` | Balanced sonic flow | 0.50 | 0.05 (was 0.10) | General use (default) |
| `discover` | Varied textures | 0.35 | 0.00 (disabled) | Maximum sonic variety |
| `off` | Genre-only mode | 0.00 | N/A | Ignore sonic features |

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
    cohesion_mode: dynamic           # strict, narrow, dynamic (default), discover
    random_seed: 0                   # For reproducibility (0 = random)
    enable_logging: true

    # Tower weights for similarity (must sum to 1.0)
    tower_weights:
      rhythm: 0.20                   # Tempo/beat importance
      timbre: 0.50                   # Texture/tone importance
      harmony: 0.30                  # Key/chord importance

    # Transition weights (end-of-A to start-of-B)
    # IMPORTANT: should match tower_weights so the beam's transition score
    # is computed in the same feature balance as the rest of the pipeline.
    # Mismatch causes the beam to approve edges the reporter scores poorly
    # (see docs/PLAYLIST_ORDERING_TUNING.md). Previously rhythm-dominant
    # (0.50/0.25/0.15); changed to match tower_weights in v4.1.
    transition_weights:
      rhythm: 0.20
      timbre: 0.50
      harmony: 0.30

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

      # Raw genre-compatibility penalty.
      # Applies against the raw artifact genre vocabulary. The old
      # candidate-pool genre-conflict hard gate was deleted because it rejected
      # legitimate candidates with the current high-dimensional identity
      # affinity vocabulary. The soft penalty demotes off-axis tracks without
      # a direct hard gate.
      genre_compatibility_enabled: true
      genre_compatibility_penalty_strength: 0.20
      genre_compatibility_compatible_threshold: 0.35
      genre_compatibility_conflict_threshold: 0.15

      # Hard title exclusions (drops candidates entirely; case-insensitive)
      title_exclusion_enabled: true
      title_exclusion_words: ["interlude", "skit", "acapella", "a cappella", "a capella"]

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
Pier-bridge uses per-mode defaults when you don't specify overrides:
- `strict`: `transition_floor=0.45`, `bridge_floor=0.10`, weights `bridge=0.7`, `transition=0.3`
- `narrow`: `transition_floor=0.45`, `bridge_floor=0.05` (was 0.08), weights `bridge=0.7`, `transition=0.3`
- `dynamic`: `transition_floor=0.35`, `bridge_floor=0.02` (was 0.03), weights `bridge=0.6`, `transition=0.4`

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
- Run `python main_app.py --artist "Sabrina Carpenter" --cohesion-mode dynamic --log-level INFO --dry-run`
- Confirm the run logs include: `Pier-bridge tuning resolved: mode=dynamic ...`
- Confirm the run logs include: `Pier-bridge segment policy: artist_playlist=... strategy=...`
- For penalty visibility, re-run with `--log-level DEBUG` and look for per-segment `soft_genre_penalty_hits=... edges_scored=...` lines.

### Artist-Style Clustering (artist playlists)

```yaml
playlists:
  ds_pipeline:
    artist_style:
      enabled: true
      cluster_k_min: 3
      cluster_k_max: 6
      cluster_k_heuristic_enabled: true
      piers_per_cluster: 1

      # Per-cluster external candidate pool size.
      # Increased from 800 to 2000 in v4.1 — the top 800 nearest-to-medoid
      # tracks for a narrow-style band tend to be artist-clones, leaving few
      # genuinely bridging candidates. 2000 reaches further out per cluster.
      per_cluster_candidate_pool_size: 2000
      pool_balance_mode: equal             # equal | proportional_capped

      internal_connector_priority: true
      internal_connector_max_per_segment: 2

      # Genre-neighbor candidate pool (UNION with the cluster pool).
      # Size raised from 500 to 1500 in v4.1; min_confidence set to null
      # for the same reason the old candidate-pool genre-conflict hard gate
      # was deleted.
      genre_neighbor_pool_enabled: true
      genre_neighbor_pool_size: 1500
      genre_neighbor_min_similarity: 0.25
      genre_neighbor_min_confidence: null
      genre_neighbor_compatible_threshold: 0.35
      genre_neighbor_conflict_threshold: 0.15
```

### Segment-Pool Per-Artist Collapse (recommended OFF)

Default in `config.yaml` for v4.1:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      collapse_segment_pool_by_artist: false
```

When `true` (the legacy default in `PierBridgeConfig`), the segment-pool builder collapses candidates to one-track-per-artist (best by harmonic mean of pier similarities) before beam search. This is **redundant** because the beam already enforces one-per-segment artist diversity via `used_artists`, and it **biases** the pool toward mid-projection tracks (high harmonic mean), starving the high-progress region.

Set to `false` for long narrow-style segments — the beam sees many more candidates at varied projection positions while still picking distinct artists.

### Edge Diagnostics (opt-in, default off)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true
```

When enabled, generates a "Selected-edge audit" log block after each playlist with one row per edge:

```
Edge #18: METZ - Spit You Out -> Dinosaur Jr. - It's Me
  T=0.366 T_centered_cos=-0.267 S=0.675 G=0.949 | bridge=0.447 trans_beam=0.845 title_flags=-
  progress_t=0.459 progress_jump=0.121 local_sonic_cos=0.659 local_pen=0.000 genre_pen=0.000 below_floor=False
```

Also runs `diagnose_t_mismatch` as a regression check. Beam `trans_beam` and final reporter `T` now share the same transition metric, so `WARNING: T-mismatch edge ...` indicates a bug or missing-data fallback to investigate, not an expected tuning signal.

See `docs/PLAYLIST_ORDERING_TUNING.md` for full interpretation.

### Opt-in Scoring/Filtering Knobs (default off)

All under `playlists.ds_pipeline.pier_bridge` — see `docs/PLAYLIST_ORDERING_TUNING.md` for the tuning recipe.

```yaml
# Soft title-artifact penalty
title_artifact_penalty:
  enabled: false
  weights:
    demo: 0.10
    live: 0.05
    medley: 0.20
    remix: 0.10
    instrumental: 0.08
    take: 0.10
    outtake: 0.15
    alternate: 0.10
    version: 0.05

# Scaled local-sonic-edge penalty
local_sonic_edge_penalty_mode: legacy      # legacy | scaled
local_sonic_edge_penalty_scale: 1.0        # used only in 'scaled' mode

# Worst-edge lexicographic beam objective
min_edge_objective: total_score            # total_score | min_edge

# Last-mile edge repair fallback
edge_repair:
  enabled: false
  centered_cos_floor: -0.5
  margin: 0.05
  variety_guard:
    enabled: false
    threshold: 0.85
```

| Knob | What it does |
|---|---|
| `title_artifact_penalty` | Soft-demotes candidates whose titles contain `demo`/`live`/`medley`/`remix`/`instrumental`/`take`/`outtake`/`alternate`/`version` (and `mono`/`stereo`/`remaster`/`edit` if weights provided). Hard exclusions remain in `candidate_pool.title_exclusion_words`. |
| `local_sonic_edge_penalty_mode: scaled` | Replaces the legacy penalty math (`strength × (threshold − edge_cos)`, max ≈0.03) with `scale × (threshold − edge_cos)` so the penalty is large enough to actually influence beam selection. |
| `min_edge_objective: min_edge` | Final beam selection prefers paths with the highest minimum `trans_score_in_beam` across edges, ties broken by total score. Implements "no broken moments" over "good on average". |
| `edge_repair.enabled` | Opt-in fallback that attempts conservative single-track swaps only after upstream beam scoring is already aligned with final `T`. Piers and seeds are protected; swaps must clear the transition floor and improve worst adjacent `T` by `edge_repair.margin`. |

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
| `PLAYLIST_DS_ALLOWED_TRACK_ID_LIMIT` | Maximum explicit DS allow-list size before refusing a run; default `25000`, sized above the default artist-style pool. |
| `PLEX_TOKEN` | Plex authentication token |
