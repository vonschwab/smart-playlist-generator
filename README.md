# Playlist Generator

**Version 3.3** - Data Science-powered music playlist generation with a Windows GUI, MusicBrainz MBID enrichment, and the beat3tower DS pipeline.

## Overview

This system generates intelligent playlists by combining:
- **Beat3Tower Sonic Analysis** - 137-dimensional audio feature vectors (rhythm, timbre, harmony)
- **Pier-Bridge Ordering** - Seed tracks as anchors with beam-search optimized bridges between them
- **Multi-Segment Analysis** - Captures song dynamics (start, middle, end) for smooth transitions
- **Normalized Genre Data** - Artist/album/track-level genres from MusicBrainz and Discogs
- **Multiple Playlist Modes** - Strict, Narrow, Dynamic, Discover, Off (independent sonic/genre controls)
- **Seed List Mode** - Explicit multi-track seeding for artist or mixed-artist playlists
- **Modularized Pipeline** - Refactors split candidate pools, pier/bridge, scoring, and diagnostics into focused modules

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-gui.txt
pip install -e .

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your paths

# 3. Verify environment
python tools/doctor.py

# 4. Scan your music library
python scripts/scan_library.py

# 5. (Optional but recommended) Fetch MusicBrainz MBIDs without touching audio files
python scripts/fetch_mbids_musicbrainz.py --limit 500  # add --force-no-match/--force-error to retry markers

# 6. Extract sonic features
python scripts/update_sonic.py --beat3tower --workers 4

# 7. Fetch genre metadata
python scripts/update_genres_v3_normalized.py --artists

# 8. Build artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz

# 9. Generate a playlist (CLI)
python main_app.py --artist "Radiohead" --tracks 30

# Or generate by genre
python main_app.py --genre "new age" --tracks 30

# 10. Launch the GUI (Windows)
python -m playlist_gui.app
```

See [docs/GOLDEN_COMMANDS.md](docs/GOLDEN_COMMANDS.md) for complete command reference.

## Requirements

- Python 3.8+
- ~8GB RAM for sonic analysis
- SSD recommended for faster processing

## Project Structure

```
.
├── main_app.py              # Main playlist generator CLI
├── config.example.yaml      # Configuration template
├── requirements.txt         # Python dependencies
├── scripts/                 # Production CLI tools
│   ├── scan_library.py      # Library scanner
│   ├── update_sonic.py      # Sonic feature extraction
│   ├── update_genres_v3_normalized.py  # Genre metadata
│   ├── build_beat3tower_artifacts.py   # Artifact builder
│   └── analyze_library.py   # Full pipeline
├── src/                     # Core Python package
│   ├── playlist/            # Playlist generation pipeline
│   ├── features/            # Audio feature extraction
│   ├── similarity/          # Similarity computation
│   └── genre/               # Genre processing
├── tools/
│   └── doctor.py            # Environment validator
├── tests/                   # Test suite
├── data/                    # Data files (not in git)
│   ├── metadata.db          # Track database
│   ├── genre_similarity.yaml # Genre relationship matrix
│   └── artifacts/           # DS pipeline matrices
└── docs/
    ├── GOLDEN_COMMANDS.md   # Command reference
    ├── ARCHITECTURE.md      # System architecture
    ├── CONFIG.md            # Configuration reference
    └── TROUBLESHOOTING.md   # Common issues
```

## Playlist Modes

**Updated in v3.3:** Independent genre and sonic mode controls for fine-grained playlist tuning.

### Genre Modes
Control how strictly playlists match genre tags:
- `strict` - Ultra-tight genre matching (single-genre deep dives)
- `narrow` - Stay close to seed genre (cohesive exploration)
- `dynamic` - Balanced exploration (default)
- `discover` - Genre-adjacent exploration (cross-genre discovery)
- `off` - Ignore genre tags completely (pure sonic matching)

### Sonic Modes
Control how strictly playlists match audio features:
- `strict` - Laser-focused sound (ultra-tight sonic matching)
- `narrow` - Consistent texture (cohesive sound)
- `dynamic` - Balanced sonic flow (default)
- `discover` - Sonic variety (varied textures)
- `off` - Ignore sonic features (pure genre matching)

### Examples

```bash
# Ultra-cohesive (strict genre + strict sonic)
python main_app.py --artist "Radiohead" --genre-mode strict --sonic-mode strict

# Same genre, varied sound
python main_app.py --artist "Bill Evans" --genre-mode narrow --sonic-mode discover

# Pure sonic similarity (ignore genres)
python main_app.py --genre "ambient" --genre-mode off --sonic-mode dynamic

# Discovery mode (explore connections)
python main_app.py --genre "jazz" --genre-mode discover --sonic-mode discover
```

See [docs/CONFIG.md](docs/CONFIG.md#mode-based-configuration-simplified-tuning) for full mode documentation.

## DS Run Audits (3.3)
- Per-run markdown audits: add `--audit-run` (optional `--audit-run-dir docs/run_audits`) to record pool sizes, segment gating, scoring, and post-order validation.
- Infeasible segment handling (optional): add `--pb-backoff` to retry segments with a deterministic `bridge_floor` backoff (attempts are recorded in the audit).
- Recency invariant: Last.fm/local recency exclusions are applied **pre-order only**; verify logs include exactly one `stage=candidate_pool | Last.fm recency exclusions: ...` line and one `stage=post_order_validation | ...` line.

## GUI Highlights (3.3)
- **Genre Mode** - Generate playlists by genre with smart autocomplete showing both exact matches and similar genres (similarity ≥ 0.7)
- **Seed List Mode** - Add multiple explicit seed tracks (per-row autocomplete)
- **Accent-insensitive Autocomplete** - Type "Joao" and see "João Gilberto" for both artist and genre fields
- **Atomized Genre Data** - All 746 genres properly normalized and split (no compound strings like "indie rock, alternative")
- **Track Table Export** - Export buttons fixed; context menu still available
- **Progress/Log Panels** - Wired to worker with request correlation
- **Run All Button** - One-click pipeline execution (Scan → Genres → Sonic → Artifacts)

## MBID Enrichment (3.2)
- `scripts/fetch_mbids_musicbrainz.py` queries MusicBrainz by artist/title (with collab/feature handling) and writes MBIDs to `tracks.musicbrainz_id` (no file writes). Uses skip markers (`__NO_MATCH__`, `__ERROR__`); reprocess with `--force-no-match`/`--force-error` or all with `--force-all`.
- `scripts/analyze_library.py` supports a `mbid` stage: `--stages scan,mbid,genres,...` to enrich during full runs.
- Last.FM matching now prefers MBIDs for instant, exact mapping.

## Documentation

- [Golden Commands](docs/GOLDEN_COMMANDS.md) - Production workflow reference    
- [Architecture](docs/ARCHITECTURE.md) - System design overview
- [Configuration](docs/CONFIG.md) - Config file reference
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and fixes
- [Logging](docs/LOGGING.md) - Logging configuration and audit notes

### Diagnostics & Support
- GUI logs: `%APPDATA%\PlaylistGenerator\logs\playlist_gui.log` (rotates 5 x 2MB). Secret-aware redaction is applied to all UI/worker lines.
- Copy/Save Debug Report: Help -> Copy Debug Report or Save Debug Report. Copies/saves a redacted bundle (environment, config path, preset/mode, last job, readiness checks, tail of GUI/worker logs).
- Readiness banner: non-modal warning at the top of the window when prerequisites are missing (config/DB/artifacts/worker). Shows `Last checked: HH:MM` with CTAs to re-run checks, open Jobs, queue Scan/Artifacts, copy debug report, retry queue, or dismiss.
- Layout persistence: window/dock layout, last config path, mode, artist query, preset, and track filter are restored via QSettings. Reset via View -> Reset UI Layout.
- Doctor command: `{"cmd":"doctor","request_id":"<uuid>","base_config_path":"...","overrides":{}}` returns quick check results.
- Jobs pane: Queue/History filter; Clear Pending empties the queue immediately (pending count drops to zero).

## License

MIT
