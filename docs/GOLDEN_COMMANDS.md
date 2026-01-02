# Golden Commands

These are the canonical production workflows for the Playlist Generator. Run `doctor` first to verify your environment.

## Prerequisites

1. Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `config.example.yaml` to `config.yaml` and configure paths
4. Ensure `data/metadata.db` exists (created by library scan)

---

## 0. Doctor (Environment Verification)

```bash
python tools/doctor.py
```

**What it checks:**
- Python version
- Required dependencies importable
- Database exists and has valid schema
- Config file present and valid
- Artifact files exist (if configured)

**Expected output:** `All checks passed` or actionable error messages.

---

## A. Library Scan

Scan your music library and populate the metadata database.

```bash
# Full scan (first time or after adding many files)
python scripts/scan_library.py

# Quick scan (only new/modified files)
python scripts/scan_library.py --quick

# Check stats without scanning
python scripts/scan_library.py --stats

# Limit for testing
python scripts/scan_library.py --limit 100
```

**Inputs:** `config.yaml` (library.music_directory)
**Outputs:** `data/metadata.db` (tracks, albums, track_genres tables)

---

## B. Genre Update

Fetch genre metadata from MusicBrainz/Discogs.

```bash
# Update all (artists, albums, tracks)
python scripts/update_genres_v3_normalized.py

# Update specific types
python scripts/update_genres_v3_normalized.py --artists
python scripts/update_genres_v3_normalized.py --albums
python scripts/update_genres_v3_normalized.py --tracks

# Check stats
python scripts/update_genres_v3_normalized.py --stats

# Limit for testing
python scripts/update_genres_v3_normalized.py --artists --limit 50
```

**Inputs:** `data/metadata.db` (tracks)
**Outputs:** `data/metadata.db` (artist_genres, album_genres, track_genres tables)

---

## C. Sonic Analysis

Analyze audio files and extract beat3tower sonic features.

```bash
# Analyze tracks without features (safe mode)
python scripts/update_sonic.py --beat3tower

# Parallel processing (adjust workers for your disk speed)
python scripts/update_sonic.py --beat3tower --workers 4

# Check stats
python scripts/update_sonic.py --stats

# Limit for testing
python scripts/update_sonic.py --beat3tower --limit 100
```

**Inputs:** Audio files, `data/metadata.db`
**Outputs:** `data/metadata.db` (sonic_features column updated)

**Performance notes:**
- HDD: Use 4-6 workers
- SSD: Use 8-12 workers
- Large libraries may take several hours

---

## D. Build Artifacts

Build DS pipeline artifacts from sonic features.

```bash
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz
```

**Inputs:** `data/metadata.db` (sonic_features), `data/genre_similarity.yaml`
**Outputs:** `data/artifacts/beat3tower_32k/data_matrices_step1.npz`

---

## E. Full Analysis Pipeline

Run all steps (scan, genres, sonic, artifacts) in sequence.

```bash
# Full pipeline
python scripts/analyze_library.py

# Limit for testing
python scripts/analyze_library.py --limit 100

# Skip stages
python scripts/analyze_library.py --skip scan --skip discogs
```

**Stages:** scan → genres → discogs → sonic → genre-sim → artifacts → verify

---

## F. Playlist Generation

Generate playlists from your library.

```bash
# Generate for specific artist (recommended start)
python main_app.py --artist "Radiohead" --tracks 30

# With specific seed track
python main_app.py --artist "David Bowie" --track "Life On Mars" --tracks 30

# Generate for specific genre
python main_app.py --genre "new age" --tracks 30
python main_app.py --genre "ambient" --ds-mode narrow --tracks 30

# Dry run (preview without creating files)
python main_app.py --artist "Radiohead" --dry-run
python main_app.py --genre "experimental" --dry-run

# DS pipeline modes
python main_app.py --artist "Radiohead" --ds-mode narrow    # Focused
python main_app.py --artist "Radiohead" --ds-mode dynamic   # Balanced (default)
python main_app.py --artist "Radiohead" --ds-mode discover  # Exploratory

# Generate from listening history
python main_app.py

# Sonic-only mode (no genre filtering)
python main_app.py --artist "Radiohead" --ds-mode sonic_only
```

**Inputs:** `config.yaml`, `data/metadata.db`, artifacts
**Outputs:** M3U playlists to configured export path

---

## Quick Start Workflow

For a new library:

```bash
# 1. Verify environment
python tools/doctor.py

# 2. Scan library
python scripts/scan_library.py

# 3. Fetch genres (artist-level first, most efficient)
python scripts/update_genres_v3_normalized.py --artists

# 4. Analyze sonic features
python scripts/update_sonic.py --beat3tower --workers 4

# 5. Build artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz

# 6. Generate a test playlist
python main_app.py --artist "Your Favorite Artist" --tracks 20 --dry-run

# Or try genre mode
python main_app.py --genre "ambient" --tracks 20 --dry-run
```

---

## Typical Maintenance Workflow

After adding new music:

```bash
# Quick scan for new files
python scripts/scan_library.py --quick

# Update genres for new artists
python scripts/update_genres_v3_normalized.py --artists

# Analyze new tracks
python scripts/update_sonic.py --beat3tower --workers 4

# Rebuild artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz
```
