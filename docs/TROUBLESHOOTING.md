# Troubleshooting

Common issues and their solutions.

## Environment Issues

### "ModuleNotFoundError: No module named 'playlist_generator'"

**Cause:** Python can't find the package.

**Fix:**
```bash
# Ensure you're in the repo root
cd /path/to/repo_refreshed

# For scripts, they should auto-add to path
# For manual import, add src/ to PYTHONPATH:
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### "config.yaml not found"

**Cause:** Config file doesn't exist.

**Fix:**
```bash
cp config.example.yaml config.yaml
# Edit with your paths
```

### Missing dependencies

**Cause:** Required packages not installed.

**Fix:**
```bash
pip install -r requirements.txt

# For specific issues:
pip install librosa mutagen scikit-learn
```

## Database Issues

### "no such table: tracks"

**Cause:** Database not initialized.

**Fix:**
```bash
python scripts/scan_library.py
```

### "database is locked"

**Cause:** Another process is using the database.

**Fix:**
- Close other scripts accessing the database
- Wait for running scans/analyses to complete
- If stuck, kill processes and retry

### Slow queries

**Cause:** Large database without indexes.

**Fix:** Run a re-scan which recreates indexes:
```bash
python scripts/scan_library.py --force
```

## Sonic Analysis Issues

### "librosa.load failed"

**Cause:** Corrupt audio file or unsupported format.

**Fix:**
- Check the specific file with a media player
- Re-encode problematic files
- Reduce workers to avoid memory issues:
  ```bash
  python scripts/update_sonic.py --beat3tower --workers 2
  ```

### Very slow analysis

**Cause:** Too many workers for your disk speed.

**Fix:**
- HDD users: Use 4-6 workers max
- SSD users: Can use 8-12 workers
- Monitor disk usage - if 100%, reduce workers:
  ```bash
  python scripts/update_sonic.py --beat3tower --workers 4
  ```

### Out of memory

**Cause:** Too many parallel workers.

**Fix:**
```bash
python scripts/update_sonic.py --beat3tower --workers 2
```

## Genre Issues

### "No tracks need genre updates"

**Cause:** Already up to date.

**Verify:**
```bash
python scripts/update_genres_v3_normalized.py --stats
```

### API rate limiting

**Cause:** Too many requests to MusicBrainz/Discogs.

**Fix:** Built-in rate limiting should handle this. Wait and retry.

### Missing genres for many artists

**Cause:** Artists not found in databases.

**Notes:**
- Obscure artists may not have genre data
- File tags are used as fallback
- Genre matching still works via artist similarity

## Artifact Issues

### "Artifact file not found"

**Cause:** Artifacts not built or path mismatch.

**Fix:**
```bash
# Build artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz

# Verify path matches config
grep artifact_path config.yaml
```

### "Shape mismatch in artifact"

**Cause:** Artifact built with different track set than database.

**Fix:** Rebuild artifacts:
```bash
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz \
    --force
```

## Playlist Generation Issues

### "Could not create playlist for artist"

**Causes:**
- Artist not in your library
- Artist has too few tracks
- No similar tracks found

**Diagnose:**
```bash
# Check if artist exists
sqlite3 data/metadata.db "SELECT COUNT(*) FROM tracks WHERE artist LIKE '%Artist Name%'"

# Try with more relaxed mode
python main_app.py --artist "Artist Name" --ds-mode discover
```

### Playlists are too similar

**Cause:** Narrow mode too restrictive.

**Fix:**
```bash
python main_app.py --artist "Artist" --ds-mode dynamic
# or
python main_app.py --artist "Artist" --ds-mode discover
```

### Playlists have jarring transitions

**Cause:** Sonic features may need preprocessing adjustment.

**Fix:**
```bash
# Try different variant
python main_app.py --artist "Artist" --sonic-variant tower_pca

# Or increase transition floor in config:
# playlists.ds_pipeline.constraints.transition_floor: 0.25
```

### Same artist appears too often

**Fix in config.yaml:**
```yaml
playlists:
  max_tracks_per_artist: 2
  ds_pipeline:
    constraints:
      min_gap: 8  # More spacing between same artist
```

## Export Issues

### M3U paths not working

**Cause:** Path format mismatch.

**Fix:**
- Ensure paths in M3U match your player's expected format
- Check `m3u_export_path` in config
- For Windows, paths should use backslashes in some players

### Plex export fails

**Causes:**
- Invalid token
- Music section not found
- Path mapping incorrect

**Fix:**
```bash
# Get your Plex token
# Settings > Account > Copy Token

# Configure in config.yaml:
plex:
  enabled: true
  base_url: http://localhost:32400
  token: "your_token_here"
  music_section: "Music"  # Must match your library name
```

## Logs

Check logs for detailed error information:

```bash
# Playlist generation
tail -f playlist_generator.log

# Sonic analysis
tail -f sonic_analysis.log

# Genre updates
tail -f genre_update_v3.log

# Library scan
tail -f scan_library.log
```

## Getting Help

1. Run the doctor: `python tools/doctor.py --verbose`
2. Check logs for specific errors
3. Verify your config against `config.example.yaml`
