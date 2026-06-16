# Troubleshooting

Common issues and their solutions.

## Environment Issues

### "ModuleNotFoundError: No module named '…'"

**Cause:** The package isn't installed into the active environment.

**Fix:**
```bash
# From the repo root, install editable (Python 3.11+ required):
pip install -e .[web]        # users
pip install -e .[web,dev]    # contributors (adds pytest, ruff, mypy, pre-commit)
```
Scripts add the repo root to `sys.path` themselves when run directly, so once the
editable install is in place both `python main_app.py …` and `python -m pytest` resolve
`src/` and `scripts/`.

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
pip install -e .[web,dev]

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

# Try with a more relaxed beam
python main_app.py --artist "Artist Name" --cohesion-mode discover
```

### Playlists are too similar

**Cause:** Narrow mode too restrictive.

**Fix:**
```bash
python main_app.py --artist "Artist" --cohesion-mode dynamic
# or
python main_app.py --artist "Artist" --cohesion-mode discover
```

### Playlists have jarring transitions

**Cause:** Beam admitted a weak edge, or the sonic space needs a different preprocessing.

**Fix:**
```bash
# Tighten the beam (raises per-mode bridge/transition floors)
python main_app.py --artist "Artist" --cohesion-mode narrow

# Engage pace gating so tempo jumps are penalized
python main_app.py --artist "Artist" --pace-mode narrow
```
Or raise the per-mode transition floor in config (these are keyed by `cohesion_mode`):
```yaml
playlists:
  ds_pipeline:
    constraints:
      transition_floor_dynamic: 0.40   # default 0.35
      transition_floor_narrow: 0.50    # default 0.45
```
The sonic space defaults to the MERT embedding in v6.0; `--sonic-variant tower_pca`
falls back to the tower blend if you need to A/B against it.

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

## Browser GUI Issues

The GUI is React (`web/`) → FastAPI (`src/playlist_web/app.py`) → NDJSON worker
(`src/playlist_gui/worker.py`), launched by `python tools/serve_web.py` (default port
8770). Most "the GUI doesn't show / do X" reports are **stale process or build state**,
not logic bugs. Walk these first; the full trap catalog is in the `web-gui` skill.

### A front-end change doesn't appear

**Cause:** `web/dist` wasn't rebuilt; the server is still serving the old bundle.

**Fix:**
```bash
npm --prefix web run build     # rebuild the bundle
# then hard-reload the browser (cache)
```

### A worker/backend change doesn't take effect

**Cause:** `serve_web.py` (and the worker child process it spawns) wasn't restarted after
editing the worker, policy layer, or an analyze stage.

**Fix:** Stop and restart `python tools/serve_web.py`. A long-lived worker holds the old
code (and any `@lru_cache`d artifact) until the process is replaced.

### A button runs but nothing comes back

**Cause:** Usually an end-to-end wiring gap — a new API endpoint or worker command that
isn't wired through every layer (React → FastAPI route → worker command handler →
streamed result), or a result the bridge silently drops.

**Fix:** Trace the request through all four layers. The `web-gui` skill documents the
stale-dist, worker-restart, end-to-end-wiring, and silently-dropped-result traps that each
historically cost a debugging cycle.

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
