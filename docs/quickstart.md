# Quick Start Guide

Get up and running with Playlist Generator in 5 minutes.

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **ffmpeg** (optional, for better audio analysis - [install](https://ffmpeg.org/download.html))

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vonschwab/smart-playlist-generator.git
cd smart-playlist-generator
```

### 2. Create Python Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit config.yaml with your music library path and API keys
# At minimum, set:
#   music_library_path: /path/to/your/music
#   (API keys are optional for basic usage)
```

## First Run

### Step 1: Scan Your Music Library

```bash
python scripts/scan_library.py
```

This discovers all music files in your library and extracts metadata.

**Expected output:**
```
Found 1234 music files
Analyzing metadata...
Completed: 1234/1234 tracks
```

### Step 2: Analyze Sonic Features

```bash
python scripts/analyze_library.py --stages sonic
```

This extracts audio features (tempo, timbre, rhythm, etc.) from your tracks.

**Expected time:** ~1-2 hours for 1000+ tracks (runs in parallel)

### Step 3: Build Artifact Matrices

```bash
python scripts/analyze_library.py --stages genres,artifacts
```

This creates optimized matrices for fast playlist generation.

**Expected time:** 5-10 minutes

## Generate Your First Playlist

### Option A: Command Line (Simplest)

```bash
python main_app.py --artist "Artist Name" --count 50
```

**Output:** M3U file with 50 tracks similar to songs by "Artist Name"

### Option B: Specify a Specific Song

```bash
python main_app.py --artist "Fela Kuti" --count 50 --dry-run
```

**Flags:**
- `--count N`: Number of tracks (default 50)
- `--dry-run`: Show results without writing to file
- `--dynamic`: Use dynamic mode (genre + sonic balance)
- `--pipeline ds`: Use data science pipeline (default)

### Option C: Run the API (For UI Integration)

```bash
uvicorn api.main:app --reload --port 8000
```

Then visit: http://localhost:8000/docs

## What Gets Created

After running the full pipeline:

```
data/
├── metadata.db          # Track database (SQLite)
├── genre_similarity.yaml  # Genre relationships
└── ...other caches

Generated playlists:
├── playlist_*.m3u       # M3U playlists
└── [Your music player can import these]
```

## Troubleshooting

### "No music files found"
- Check `music_library_path` in `config.yaml`
- Verify supported formats: `.mp3`, `.flac`, `.m4a`, `.ogg`, `.wav`

### "ffmpeg not found" warning
- Optional - feature extraction will work without it, just slower
- Install ffmpeg: https://ffmpeg.org/download.html

### Sonic analysis taking too long
- Set `--workers 4` to use 4 CPU cores: `python scripts/analyze_library.py --stages sonic --workers 4`
- Reduce database size with `--limit 100` for testing

### "Database locked" error
- Close other instances of Playlist Generator
- Delete `data/metadata.db` and restart (will rescan)

## Next Steps

- [Full Architecture](architecture.md) - Understand the system
- [Configuration Guide](configuration.md) - Customize settings
- [API Reference](api.md) - Use programmatically
- [Development Guide](dev.md) - Contribute code

## Need Help?

- Check existing [GitHub Issues](https://github.com/vonschwab/smart-playlist-generator/issues)
- Review [Development Guide](dev.md) for debugging tips
- Check logs in `sonic_analysis.log` and `genre_update.log`

