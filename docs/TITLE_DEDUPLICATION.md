# Title Deduplication

Fuzzy title matching to prevent the same song appearing multiple times in a playlist across different releases (remasters, compilations, live versions).

## Overview

The playlist generator now supports intelligent title deduplication that recognizes when tracks are different versions of the same song. This prevents playlists from containing multiple versions like:

- "Creep" (original album)
- "Creep (Remastered 2011)"
- "Creep - Live"

## Features

- **Scoped by artist**: Only considers duplicates within the same artist
- **Two normalization modes**: "strict" (conservative) and "loose" (aggressive)
- **Configurable fuzzy threshold**: Control how similar titles must be to match
- **Short title safeguard**: Very short titles require exact match to avoid false positives
- **Version preference scoring**: When duplicates are found, prefers canonical versions

## Configuration

Add to your `config.yaml`:

```yaml
playlists:
  dedupe:
    title:
      enabled: true           # Enable fuzzy title deduplication
      threshold: 92           # Fuzzy match threshold (0-100, higher = stricter)
      mode: loose             # 'strict' or 'loose' (loose strips version tags)
      short_title_min_len: 6  # Titles shorter than this require exact match
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable title deduplication |
| `threshold` | `92` | Similarity threshold (0-100). Higher values are stricter. |
| `mode` | `loose` | Normalization mode. `loose` strips version tags, `strict` preserves them. |
| `short_title_min_len` | `6` | Titles shorter than this require exact match |

### Mode Comparison

**Strict Mode:**
- Case folding and whitespace normalization
- Preserves parenthetical content
- Best when you want to keep different versions separate

**Loose Mode (recommended):**
- All strict mode normalizations
- Strips version-related parenthetical content like `(Remastered)`, `[Live]`
- Strips featuring sections like `feat. Artist`
- Strips dash suffixes like `- Live`, `- Demo`

### Threshold Guidelines

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 85 | Aggressive matching | When you have many near-duplicates |
| 92 | Balanced (default) | Most libraries |
| 95 | Conservative | When you want to preserve minor variations |
| 100 | Exact match only | When you only want identical titles to match |

## How It Works

1. **During candidate collection**: As tracks are gathered for a playlist, each candidate is checked against already-selected tracks
2. **Artist scoping**: Tracks are only compared against other tracks by the same artist
3. **Normalization**: Titles are normalized based on the configured mode
4. **Fuzzy matching**: Normalized titles are compared using SequenceMatcher
5. **Threshold check**: If similarity exceeds the threshold, the track is filtered as a duplicate

## Version Keywords

In "loose" mode, the following keywords trigger removal of parenthetical/bracketed content:

```
live, demo, remaster, remastered, edit, version, ver, mono, stereo,
acoustic, instrumental, mix, remix, rmx, re-record, rerecord, session,
alternate, alt, alternative, radio, clean, explicit, bonus, anniversary,
deluxe, extended, single, album, original, reprise, interlude, intro,
outro, stripped, unplugged, orchestral, symphonic, take, outtake, rough,
early, late, final, master, 2011-2025 (year tags)
```

## Validation

Use the validation script to test dedupe settings on your library:

```bash
# Basic validation with current config
python scripts/validate_title_dedupe.py

# Test different threshold
python scripts/validate_title_dedupe.py --threshold 90

# Test strict mode
python scripts/validate_title_dedupe.py --mode strict

# Focus on specific artist
python scripts/validate_title_dedupe.py --artist "Radiohead"

# Show how titles are normalized
python scripts/validate_title_dedupe.py --show-normalization
```

## Logging

When running playlist generation, the dedupe system logs:
- Total fuzzy duplicates filtered (INFO level)
- Individual duplicate detections (DEBUG level)

Example log output:
```
Sonic-only pool: 250 candidates (filtered 5 short, 3 long, 2 exact dupe, 8 fuzzy dupe)
```

Enable DEBUG logging to see individual matches:
```
Title dedupe: skipping 'Artist - Song (Remastered 2011)' (fuzzy match to 'Song')
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_title_dedupe.py -v
```

Tests cover:
- Title normalization (strict and loose modes)
- Fuzzy title similarity calculation
- TitleDedupeTracker behavior
- Short title safeguards
- Version keyword stripping
- Version preference scoring
- Integration scenarios (remaster vs original, live vs studio, etc.)

## Technical Details

### Files

- `src/title_dedupe.py` - Core deduplication module
- `src/config_loader.py` - Configuration properties
- `src/playlist_generator.py` - Integration points
- `tests/test_title_dedupe.py` - Test suite
- `scripts/validate_title_dedupe.py` - Validation script

### Integration Points

Title deduplication is applied in:
1. `generate_similar_tracks()` - Non-dynamic mode sonic candidate collection
2. `_generate_similar_tracks_dynamic()` - Dynamic mode (sonic + genre tracks)

The tracker is shared across all candidate sources to ensure consistency.

### Performance

The deduplication is efficient:
- O(n) complexity for adding tracks
- O(m) lookup where m = tracks by same artist already seen
- Minimal overhead per track (string normalization + similarity calculation)

For a typical playlist (30 tracks from ~100 candidates), the overhead is negligible.

## FAQ

**Q: Will this affect MusicBrainz recording ID matching?**
A: No. If you have MusicBrainz recording IDs, exact ID matching takes precedence. Fuzzy title matching is a fallback for tracks without recording IDs.

**Q: What about cover songs by different artists?**
A: These are NOT filtered because deduplication is scoped by artist. "Yesterday" by The Beatles and "Yesterday" by Frank Sinatra are treated as different songs.

**Q: Can I disable this feature?**
A: Yes, set `dedupe.title.enabled: false` in your config.

**Q: Why is threshold 92 the default?**
A: Through testing, 92% similarity catches most version variations (remastered, live, etc.) while avoiding false positives from genuinely different songs.
