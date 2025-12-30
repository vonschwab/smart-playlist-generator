# Playlist Generator

**Version 3.1** - AI-powered music playlist generation with a Windows GUI, accent-insensitive artist matching, MusicBrainz MBID enrichment, and the beat3tower DS pipeline.

## Overview

This system generates intelligent playlists by combining:
- **Beat3Tower Sonic Analysis** - 137-dimensional audio feature vectors (rhythm, timbre, harmony)
- **Pier-Bridge Ordering** - Seed tracks as anchors with beam-search optimized bridges between them
- **Multi-Segment Analysis** - Captures song dynamics (start, middle, end) for smooth transitions
- **Normalized Genre Data** - Artist/album/track-level genres from MusicBrainz and Discogs
- **Multiple Playlist Modes** - Narrow (focused), Dynamic (balanced), Discover (exploratory)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

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

# 10. Launch the GUI (Windows)
python -m src.playlist_gui.app
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

| Mode | Description | Use Case |
|------|-------------|----------|
| `narrow` | Highly focused, cohesive | Deep dive into a specific sound |
| `dynamic` | Balanced variety (default) | General listening |
| `discover` | Maximum exploration | Finding new music connections |
| `sonic_only` | Pure audio similarity | Ignore genre constraints |

```bash
python main_app.py --artist "Radiohead" --ds-mode discover
```

## GUI Highlights (3.1)
- Accent-insensitive artist autocomplete (type “Joao” and see “João Gilberto”).
- Track table export buttons fixed; context menu still available.
- Progress/log panels wired to worker with request correlation.

## MBID Enrichment (3.1)
- `scripts/fetch_mbids_musicbrainz.py` queries MusicBrainz by artist/title (with collab/feature handling) and writes MBIDs to `tracks.musicbrainz_id` (no file writes). Uses skip markers (`__NO_MATCH__`, `__ERROR__`); reprocess with `--force-no-match`/`--force-error` or all with `--force-all`.
- `scripts/analyze_library.py` supports a `mbid` stage: `--stages scan,mbid,genres,...` to enrich during full runs.
- Last.FM matching now prefers MBIDs for instant, exact mapping.

## Documentation

- [Golden Commands](docs/GOLDEN_COMMANDS.md) - Production workflow reference    
- [Architecture](docs/ARCHITECTURE.md) - System design overview
- [Configuration](docs/CONFIG.md) - Config file reference
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and fixes
- [Logging](docs/LOGGING.md) - Logging configuration and audit notes

## License

MIT
