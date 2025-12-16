# Track Duration Support

## Overview

The Playlist Generator has complete track duration support, enabling:
1. **Filtering out short tracks** (skits, interludes, spoken word) via `min_track_duration_seconds`
2. **Filtering out long tracks** (full albums, extended mixes) via `max_track_duration_seconds`
3. **M3U export** with proper EXTINF duration metadata
4. **Future duration-aware scoring** (prefer tracks matching seed duration)

## Database Schema

**Column:** `duration_ms` (INTEGER) in `tracks` table

**Values:**
- `> 0`: Valid track duration in milliseconds
- `-1`: Orphaned/unreadable file (file doesn't exist or has corrupt metadata)
- `NULL`: Should not occur (see Backfill section)

**Statistics:**
- 34,164 tracks with valid duration (> 0)
- 100 tracks marked as orphaned (-1)
- 0 tracks with NULL duration

## Metadata Extraction

### During Library Scan (`scripts/scan_library.py`)

The scanner extracts duration when processing audio files:

1. **Metadata Reading** (line 125):
   ```python
   'duration': getattr(audio.info, 'length', None),  # in seconds
   ```

2. **Storage** (lines 270, 304):
   ```python
   duration_ms = int(metadata['duration'] * 1000) if metadata.get('duration') else None
   ```

3. **Supported Formats**: MP3, FLAC, M4A, OGG, OPUS, WMA, WAV, AAC
   - Uses Mutagen for fast metadata-only reading (no audio decoding)

### Database Query Returns

All database query methods return `duration` (in milliseconds):

- `LocalLibraryClient.get_all_tracks()` - line 58, 73
- `LocalLibraryClient.get_track_by_key()` - line 100, 117
- `LocalLibraryClient.get_tracks_by_ids()` - line 139, 153
- `LocalLibraryClient.get_similar_tracks()` - line 188-191 (via get_track_by_key)
- `LocalLibraryClient.get_similar_tracks_sonic_only()` - line 221-224 (via get_track_by_key)

## Filtering Configuration

### Config Options (`config.yaml`)

```yaml
playlists:
  min_track_duration_seconds: 90      # Default: 90 (skip tracks < 1.5 min)
  max_track_duration_seconds: 720     # Default: 720 (skip tracks > 12 min)
```

### Filtering Implementation (`src/playlist_generator.py`)

**Short Track Filtering** (lines 657, 857):
```python
min_track_duration_ms = self.config.min_track_duration_seconds * 1000
if min_track_duration_ms > 0 and track_duration < min_track_duration_ms:
    filtered_short += 1
    continue
```

**Long Track Filtering** (lines 565-571):
```python
def _filter_long_tracks(self, candidates):
    max_duration_ms = _convert_seconds_to_ms(max_duration_seconds)
    return [c for c in candidates if (c.get('duration') or 0) <= max_duration_ms]
```

**Filtering Occurs In:**
- `_collect_sonic_candidates()` - lines 657-671
- `_generate_dynamic_mode_tracks()` - lines 857-887

## M3U Export

The M3U exporter (`src/m3u_exporter.py`) writes proper EXTINF metadata:

```
#EXTINF:234,Artist - Title
/path/to/file.mp3
```

**Implementation** (lines 86-90):
```python
duration_sec = track_info['duration'] // 1000 if track_info['duration'] else -1
f.write(f"#EXTINF:{duration_sec},{track_info['artist']} - {track_info['title']}\n")
```

## Backfill Process

### Problem
100 tracks were missing duration values (either files were deleted or metadata wasn't readable).

### Solution
Created `scripts/backfill_duration.py` which:
1. Reads duration from audio files using Mutagen
2. Updates DB with extracted values
3. Marks unreadable/missing files with `duration_ms = -1`

### Execution
```bash
python scripts/backfill_duration.py          # Backfill all missing
python scripts/backfill_duration.py --dry-run # Preview changes
python scripts/backfill_duration.py --track-id XXX  # Single track
```

### Filtering for Orphaned Tracks
Tracks marked with `-1` are automatically filtered out during playlist generation because:
- Min duration filter: `-1 < min_track_duration_ms` → excluded
- Max duration filter: `(duration or 0) <= max_duration_ms` → `-1` treated as `0` → excluded

## Future Enhancements

### Duration-Aware Similarity Scoring
Duration can be incorporated into track selection:

```python
# Prefer tracks within ±30% of seed duration
seed_duration = seed.get('duration', 0)
if seed_duration > 0:
    min_duration = seed_duration * 0.7
    max_duration = seed_duration * 1.3
    candidate_duration = candidate.get('duration', 0)
    if not (min_duration <= candidate_duration <= max_duration):
        similarity_score *= 0.8  # Penalty for duration mismatch
```

### Duration Range Configuration
Could add duration ranges per playlist type:

```yaml
playlists:
  workout:
    min_track_duration_seconds: 120  # Prefer longer tracks
    max_track_duration_seconds: 480
  focus:
    min_track_duration_seconds: 180  # Prefer 3-10 min tracks
    max_track_duration_seconds: 600
```

## Troubleshooting

### "Duration is missing for track X"
1. Check if track file exists at the path in database
2. If file deleted: run backfill script (it will mark as -1)
3. If file exists but unreadable: check file permissions and format support

### "Track not being included in playlist despite matching seed"
1. Check `min_track_duration_seconds` setting (default: 90s)
2. Run: `sqlite3 data/metadata.db "SELECT title, duration_ms FROM tracks WHERE track_id = 'XXX'"`
3. Verify duration_ms value is > min_duration_ms threshold

### "M3U file shows -1 duration"
This indicates the track is marked as orphaned (`duration_ms = -1`). The track will be filtered out during future playlist generation.

## Code References

| Component | File | Lines |
|-----------|------|-------|
| Duration extraction | scripts/scan_library.py | 125, 270, 304 |
| Duration storage | src/local_library_client.py | 58, 100, 139 |
| Short track filtering | src/playlist_generator.py | 657, 857, 685 |
| Long track filtering | src/playlist_generator.py | 565-571 |
| M3U export | src/m3u_exporter.py | 71, 86-90 |
| Config options | src/config_loader.py | 104-111 |
| Backfill script | scripts/backfill_duration.py | Full file |
