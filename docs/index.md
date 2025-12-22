# Documentation Overview

Complete documentation for Playlist Generator - a smart music playlist generation system with beat-synchronized audio analysis and multi-source genre integration.

## Quick Navigation

### First Time Users

Start here if you're new to Playlist Generator:

1. **[Quick Start](quickstart.md)** - Installation and first playlist (5 min)
   - Prerequisites, setup, first run
   - Generate your first playlist
   - Troubleshooting

2. **[Architecture](architecture.md)** - How it works
   - System diagram
   - Component overview
   - Data flow

### Configuration & Customization

Customize Playlist Generator for your needs:

3. **[Configuration Guide](configuration.md)** - All config options
   - Library settings
   - API integrations (Last.FM, OpenAI)
   - Playlist parameters (duration, artist distribution)
   - Scoring and similarity settings
   - Common configurations

### Operation & Workflows

Learn how to use the system:

4. **[Pipelines & Workflows](pipelines.md)** - Data processing
   - Unified pipeline overview
   - Individual stages (scan, genres, sonic, artifacts)
   - Recommended workflows
   - Performance tuning
   - Troubleshooting pipeline issues

5. **[Playlist Generation](playlist_generation.md)** - Generation modes & strategies
   - 5 playlist modes compared
   - Scoring system explained
   - Constraints and deduplication
   - CLI options and Python API
   - Output formats

### API Integration

Integrate Playlist Generator into applications:

6. **[REST API Reference](api.md)** - HTTP endpoints
   - Library endpoints (search, status)
   - Playlist generation endpoints
   - Similarity endpoints
   - Error handling
   - Client examples (Python, JavaScript)

### Development

Contribute code or extend functionality:

7. **[Development Guide](dev.md)** - Setup, testing, debugging
   - Development environment setup
   - Running tests
   - Debugging techniques
   - Adding new features
   - Code standards

### Data Model

Understand the data structure:

8. **[Data Model](data_model.md)** - Database schema & artifacts
   - SQLite schema
   - Sonic features (71 dimensions)
   - Artifact matrices
   - Relationships and indexes

## Feature Overview

### Beat3Tower Audio Analysis

- **3-Tower Architecture**: 137-dimensional feature extraction
  - Rhythm Tower (21 dims): Onset, tempo, beat intervals
  - Timbre Tower (83 dims): MFCCs, spectral features
  - Harmony Tower (33 dims): Chroma, tonnetz, key
- **Multi-Segment Analysis**: Full/start/mid/end segments for transition matching
- **Tower PCA Processing**: Per-tower standardization + PCA with weights (default)
- **Validated Quality**: 4/4 similarity metrics pass

### Hybrid Scoring System

- **60% Sonic**: Beat3tower features with tower_pca preprocessing
  - Per-tower standardization + PCA
  - Weighted combination (rhythm 0.2, timbre 0.5, harmony 0.3)
- **40% Genre**: Multi-source normalized genres
  - MusicBrainz (artists/albums), Discogs (albums), file tags (tracks)
  - Ensemble similarity method (recommended)

### 5 Playlist Modes

| Mode | Purpose | Quality | Use Case |
|------|---------|---------|----------|
| **DS** (default) | Balanced hybrid scoring | ★★★★★ | General use, best balance |
| **Dynamic** | Progressive transition emphasis | ★★★★★ | Longer playlists (100+), flow focus |
| **Narrow** | Strict genre coherence | ★★★ | Genre purists, specific moods |
| **Discover** | Genre exploration | ★★★★ | Find new genres/artists |
| **Legacy** | Pure sonic distance | ★★★ | Baseline/legacy compatibility |

### Smart Constraints

- **Artist diversity**: Max tracks per artist, windowed distribution
- **Genre coherence**: Stay within similarity bands, avoid jarring transitions
- **Duration preferences**: Target song length with smooth falloff
- **Title deduplication**: Fuzzy matching to skip "Live", "Remaster", etc.

## System Requirements

- **Python 3.10+** with pip
- **Music Library**: 100+ MP3/FLAC/M4A/OGG/WAV files
- **Database**: SQLite (included)
- **Memory**: 512MB minimum, 2GB+ for 10,000+ tracks
- **Disk Space**: ~350MB-500MB for 34,000 tracks + metadata

Optional:
- **ffmpeg**: Better audio format support
- **Last.FM API key**: Genre fetching (free tier available)
- **OpenAI API key**: AI playlist naming

## Installation Methods

### 1. Local Installation (Recommended)

```bash
git clone https://github.com/vonschwab/smart-playlist-generator.git
cd smart-playlist-generator
pip install -r requirements.txt
cp config.example.yaml config.yaml
python scripts/scan_library.py
python main_app.py --artist "Your Artist" --count 50
```

### 2. Docker

```bash
docker build -t playlist-gen .
docker run -p 8000:8000 -v /path/to/music:/music playlist-gen
```

### 3. API Server Only

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Access: http://localhost:8000/docs
```

## Common Workflows

### Scenario 1: Generate a Single Playlist

```bash
python main_app.py --artist "Pink Floyd" --count 50 --pipeline ds
```

See: [Quick Start](quickstart.md), [Playlist Generation](playlist_generation.md)

### Scenario 2: Generate Multiple Playlists

```bash
python scripts/analyze_library.py  # Ensure artifacts are built

# Then use API or programmatically:
curl -X POST http://localhost:8000/api/playlists/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_track_id":"abc123","count":50,"mode":"ds"}'
```

See: [REST API](api.md), [Development](dev.md)

### Scenario 3: Analyze Your Library

```bash
python scripts/scan_library.py                      # Find all files (5-10 min)
python scripts/update_genres_v3_normalized.py --artists  # Fetch artist genres (20-30 min)
python scripts/update_genres_v3_normalized.py --albums   # Fetch album genres (20-30 min)
python scripts/update_sonic.py --beat3tower --workers 4  # Extract audio (4-8 hours)
python scripts/build_beat3tower_artifacts.py        # Build matrices (5-10 min)
```

See: [Pipelines](pipelines.md)

### Scenario 4: Customize for Your Music

Edit `config.yaml`:

```yaml
playlists:
  max_tracks_per_artist: 3      # Fewer repeats
  min_genre_similarity: 0.30    # More genre coherence
  duration_match:
    enabled: true
    target: 240                 # Prefer 4-minute songs
```

See: [Configuration](configuration.md)

### Scenario 5: Integrate with Your App

Use REST API:

```python
import requests

# Search library
response = requests.get("http://localhost:8000/api/library/search",
                       params={"q": "fela"})
track_id = response.json()["results"][0]["track_id"]

# Generate playlist
response = requests.post("http://localhost:8000/api/playlists/generate",
                        json={"seed_track_id": track_id, "count": 50})
playlist = response.json()["playlist"]

for track in playlist["tracks"]:
    print(f"{track['artist']} - {track['title']}")
```

See: [REST API](api.md), [Development](dev.md)

## Key Concepts

### Sonic Features (Beat3Tower)

3-tower architecture producing 137 dimensions:

- **Rhythm Tower** (21 dims): Onset detection, tempogram, beat intervals, BPM
- **Timbre Tower** (83 dims): MFCCs (13 coeffs × stats), spectral features, ZCR, RMS
- **Harmony Tower** (33 dims): Chroma (12 bins), tonnetz, key estimation

Extracted in 4 segments (full/start/mid/end) for transition-aware matching.

### Genre Similarity

7 different calculation methods:

1. **Ensemble** (recommended): Weighted combination for best accuracy
2. **Weighted Jaccard**: Relationship-aware set overlap
3. **Cosine**: Vector-based similarity
4. **Best Match**: Optimal pairing
5. **Jaccard**: Pure set overlap
6. **Average Pairwise**: Average of all comparisons
7. **Legacy**: Original max similarity

### Constraints

Enforce musical coherence:

- **Artist Diversity**: `max_per_artist: 3` → No more than 3 tracks by same artist
- **Genre Drift**: `min_similarity: 0.40` → Stay within 40% similarity band
- **Duration**: `target: 240, range: [180, 300]` → Prefer 3-5 minute songs
- **Deduplication**: Skip "Zombie", "Zombie (Live)", "Zombie (Remaster)"

## Architecture Highlights

**3-Stage Pipeline**:

1. **Analysis** (slow, offline)
   - Scan library → Extract audio → Fetch genres → Build matrices
   - Once per library update

2. **Generation** (fast, real-time)
   - Select seed track → Build candidate pool → Score & rank → Apply constraints
   - <1 second per playlist

3. **Output** (flexible)
   - M3U files, JSON metadata, API responses
   - Import anywhere (Spotify, Apple Music, VLC, etc.)

## Performance Metrics

| Operation | Time | Memory | Parallelizable |
|-----------|------|--------|-----------------|
| Scan library (1000 tracks) | 5 min | 100MB | Yes |
| Genre fetch (1000 tracks) | 30-60 min | 200MB | No (API-limited) |
| Sonic analysis (1000 tracks) | 60-120 min | 500MB | Yes (multi-core) |
| Artifact build (1000 tracks) | 5 min | 300MB | No |
| **Playlist generation** | **<1 sec** | **50MB** | N/A |

## Troubleshooting

### General Issues

- **No music files found**: Check `music_directory` path in config
- **"Database locked" error**: Close other instances
- **Slow audio analysis**: Use more workers (`--workers 8`)

### Feature Issues

- **Strange playlist combinations**: Genre weight too high (increase `sonic_weight`)
- **Repetitive playlists**: Increase `candidate_pool.size` in config
- **Poor transitions**: Enable `transition_floor` or increase beta weight

See: [Quick Start Troubleshooting](quickstart.md#troubleshooting), [Pipelines](pipelines.md#troubleshooting-pipelines)

## File Organization

```
docs/
├── index.md                    ← You are here
├── quickstart.md               ← Start here
├── architecture.md             ← How it works
├── configuration.md            ← Config reference
├── pipelines.md                ← Data processing
├── playlist_generation.md      ← Generation modes
├── api.md                      ← REST API
├── data_model.md               ← Database schema
├── dev.md                      ← Development
└── reference/                  ← (Optional) Additional references
    ├── GENRE_SIMILARITY_SYSTEM.md
    ├── TITLE_DEDUPLICATION.md
    ├── TUNING_WORKFLOW.md
    └── ...
```

## Latest Updates

### Beat3Tower Production Deployment (Current - Dec 2024)

- Implemented 3-tower architecture (137 dimensions)
- Tower PCA preprocessing for optimal discrimination
- Multi-segment extraction (full/start/mid/end)
- Validated quality: 4/4 metrics pass
- Production artifact with 32K+ tracks
- Normalized genre schema (60% fewer API calls)

### Repository Cleanup (Completed - Dec 2024)

- Archived session notes and implementation plans
- Organized documentation structure
- Updated all docs to reflect beat3tower
- Removed legacy references

## Getting Help

- **Documentation**: Start with [Quick Start](quickstart.md)
- **Issues**: Check [GitHub Issues](https://github.com/vonschwab/smart-playlist-generator/issues)
- **Development**: See [Development Guide](dev.md)
- **Examples**: Check test files in `tests/`

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! See [Development Guide](dev.md#contributing) for process.

---

**Last Updated**: December 22, 2025

**Current Version**: Beat3Tower Production (137-dim tower_pca)

**Status**: Production-ready

