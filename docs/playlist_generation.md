# Playlist Generation

Guide to playlist generation modes, scoring, constraints, and deduplication.

## Modes Overview

| Mode | Scoring | Use Case | Quality |
|------|---------|----------|---------|
| **DS (default)** | Sonic + Genre hybrid | General use, best balance | ★★★★★ |
| **Dynamic** | Progressive sonic emphasis | Longer playlists, flow | ★★★★★ |
| **Narrow** | Strict genre coherence | Genre-focused, safe | ★★★ |
| **Discover** | Genre exploration | Find new genres | ★★★★ |
| **Legacy** | Simple sonic distance | Baseline/legacy | ★★★ |

## Mode Details

### DS Mode (Recommended)

Data Science pipeline with balanced hybrid scoring.

```bash
python main_app.py --artist "Fela Kuti" --count 50 --pipeline ds
```

**Algorithm**:
1. Start with seed track
2. Build candidate pool (top 500 most similar)
3. Score each candidate:
   - `score = 0.6 * sonic_sim + 0.4 * genre_sim`
4. Pick highest-scoring track
5. Apply constraints (artist diversity, etc.)
6. Repeat for next track

**Constraints**:
- Artist diversity: max 3 tracks per artist in 10-track window
- Genre drift: stay within 40% genre similarity band
- Duration: prefer 3-5 minute tracks

**Result**: Coherent, flowing playlists with good mix

### Dynamic Mode

Progressive shift from seed similarity to transition quality.

```bash
python main_app.py --artist "Fela Kuti" --count 50 --dynamic
```

**Algorithm**:
1. Start: Heavy seed similarity weight (`alpha=0.65`)
2. Middle: Transition quality weight (`alpha=0.45`)
3. End: Return to seed similarity (`alpha=0.60`)

**Use for**: Longer playlists where flow becomes important

**Result**: Natural progressions, good flow

### Narrow Mode

Strict genre coherence (same genre throughout).

```bash
python main_app.py --artist "Fela Kuti" --count 50 --pipeline narrow
```

**Constraints**:
- Only tracks with same primary genre
- Stricter artist diversity
- No genre drift allowed

**Use for**: Genre purists, specific moods

**Result**: Tight, focused playlists

### Discover Mode

Explore across genres while maintaining sonic similarity.

**Algorithm**:
1. Genre-relaxed pool (min 0.20 similarity)
2. Sonic-only scoring
3. Allow major genre transitions

**Use for**: Discovery, finding new artists

**Result**: Wider variety, more exploration

## Scoring System

### Similarity Metrics

#### Sonic Similarity
Based on beat-synchronized audio features (71 dimensions):
- MFCC (timbre): 26 dims
- Chroma (pitch): 24 dims
- Spectral features: 16 dims
- Rhythm/beats: 5 dims

**Range**: 0.0 (very different) to 1.0 (identical)

**Example values**:
- Same song: 1.0
- Same artist, similar tempo: 0.85
- Different genre, similar energy: 0.60
- Completely different: 0.20

#### Genre Similarity
Based on normalized genre vectors:
- Cosine distance between genre distributions
- Account for genre relationships (rock → indie rock)

**Range**: 0.0 (different genres) to 1.0 (same genres)

**Example values**:
- Exact genre match: 0.95
- Related genres: 0.60
- Very different genres: 0.20

#### Hybrid Score
```
hybrid_score = w_sonic * sonic_sim + w_genre * genre_sim

Default: w_sonic=0.60, w_genre=0.40
```

### Transition Quality

How well consecutive tracks flow together.

```
transition = cosine(end_features[track_n], start_features[track_n+1])

Good flow: > 0.45
Acceptable: 0.30-0.45
Poor: < 0.30
```

Uses segment-based features (first/last 30 seconds) for better flow detection.

## Constraints

Constraints enforce musical coherence and variety.

### Artist Diversity

Prevent repetitive playlists (too many songs by one artist).

```yaml
artist_diversity:
  max_per_artist: 3          # Max 3 tracks per artist
  max_in_window: 3           # Max 3 in any 10-track window
  enforce: true              # Hard constraint (must respect)
```

**Example** (violates constraint):
```
1. Fela Kuti - Zombie
2. Fela Kuti - Shakara       ← 2nd Fela
3. Fela Kuti - Kalakuta      ← 3rd Fela (MAX)
4. Earth, Wind & Fire - Shining Star
5. Fela Kuti - Amen          ← VIOLATES (4th Fela)
```

### Genre Drift

Prevent rapid genre changes (jarring transitions).

```yaml
genre_drift:
  min_similarity: 0.40       # Stay within this band
  window_size: 10            # Over 10-track window
  enforce: soft              # Soft constraint (try to respect)
```

### Duration Preference

Favor songs in a similar length range.

```yaml
duration:
  target_seconds: 240        # Prefer 4-minute songs
  range: [180, 300]          # Accept 3-5 minutes
  prefer_exact: true         # Weight towards target
```

### Recently Played

Avoid re-playing recent songs (if integrated with player).

```yaml
recently_played:
  exclude_days: 30           # Don't play songs from last 30 days
  enabled: false             # Requires player integration
```

## Title Deduplication

Prevents duplicate song titles in playlist (same song, different versions).

### Algorithm

```python
# For each new candidate track:
1. Normalize title: lowercase, remove special chars
2. Check against previous 50 tracks in playlist
3. If title match found:
   - Skip this candidate
   - Try next best candidate
   - Continue until no duplicates
```

### Example

```
Seed: The Beatles - Hey Jude

Candidates (in order):
1. Hey Jude (The Beatles) ← DUPLICATE, SKIP
2. Hey Jude - Live (The Beatles) ← DUPLICATE, SKIP
3. I Want to Hold Your Hand (The Beatles) ← DUPLICATE ARTIST, OK
4. Eleanor Rigby (The Beatles) ← ADD TO PLAYLIST

Final: [seed] + [Eleanor Rigby] + [continue...]
```

### Configuration

```yaml
deduplication:
  enabled: true
  window_size: 50            # Check last 50 tracks
  match_type: "smart"        # smart, exact, or loose
```

## Candidate Pool Building

Strategy for selecting candidate tracks before final scoring.

### Process

```
1. Get seed track sonic features
2. Compute similarity to ALL tracks (cosine distance)
3. Sort by similarity
4. Apply filters:
   - Genre similarity >= min_threshold
   - Artist diversity OK
5. Take top 500 tracks
6. Randomize slightly (avoid ties creating determinism)
7. Return as candidate pool
```

### Pool Size

```yaml
candidate_pool:
  size: 500                  # Larger = slower but more variety
  genre_min_sim: 0.20        # Min genre similarity to consider
  min_sonic_sim: 0.15        # Min sonic similarity
```

**Larger pools**:
- Pros: More variety, less repetitive
- Cons: Slower computation, more diverse (sometimes too diverse)

**Smaller pools**:
- Pros: Faster, more focused
- Cons: Limited variety, repetitive

## CLI Options

```bash
python main_app.py \
  --artist "Artist Name"              # Seed by artist
  --count 50                           # Playlist length
  --pipeline ds                        # Mode: ds, dynamic, narrow, discover, legacy
  --dynamic                            # Shorthand for --pipeline dynamic
  --artist-only                        # Only use artist tracks (don't explore)
  --verbose                            # Debug output
  --dry-run                            # Preview without writing file
```

## Python API

```python
from src.playlist_generator import PlaylistGenerator

gen = PlaylistGenerator()

# Generate playlist
playlist = gen.generate(
    artist="Fela Kuti",
    count=50,
    mode="ds",                         # or: "dynamic", "narrow", "discover"
    constraints={
        "artist_diversity": {"max_per_artist": 3},
        "min_sonic_similarity": 0.20,
    }
)

# Access results
for track in playlist.tracks:
    print(f"{track.artist} - {track.title}")

# Export
playlist.export_m3u("my_playlist.m3u")
```

## Output Formats

### M3U Playlist

Standard music player format:

```
#EXTM3U
#EXTINF:243,Fela Kuti - Zombie
/home/user/music/fela/zombie.mp3
#EXTINF:301,Tony Allen - Progress
/home/user/music/tony/progress.flac
```

**Import into**: Spotify, Apple Music, VLC, any music player

### JSON Metadata

```json
{
  "seed": {"artist": "Fela Kuti", "title": "Kalakuta Show"},
  "length": 50,
  "mode": "ds",
  "tracks": [
    {
      "index": 0,
      "artist": "Fela Kuti",
      "title": "Zombie",
      "duration": 243,
      "genres": ["afrobeat", "funk"],
      "score": 0.92,
      "transition_score": 0.85
    },
    ...
  ],
  "stats": {
    "total_duration": 12543,
    "avg_score": 0.87,
    "genres": ["afrobeat", "funk", "soul"],
    "avg_artist_appearance": 1.2
  }
}
```

## Troubleshooting Playlist Generation

### "Playlist has duplicates"
- Deduplication failed
- Solution: Enable with `--ensure-unique`

### "Playlist feels repetitive"
- Candidate pool too small
- Solution: Increase `candidate_pool.size` in config

### "Playlist has genre jumps"
- Genre drift constraint disabled
- Solution: Enable `genre_drift` in constraints

### "Playlist all by one artist"
- Artist diversity constraint too weak
- Solution: Decrease `max_per_artist` value

### "Too many short/long songs"
- Duration preference not set
- Solution: Configure `duration.target_seconds`

## Tips for Best Results

1. **Seed selection matters**
   - Good seed = coherent playlist
   - Bad seed = confusing playlist
   - Try different seeds to explore

2. **Use mode appropriately**
   - DS mode: general use (recommended)
   - Dynamic: longer playlists (>100 tracks)
   - Narrow: genre focus
   - Discover: exploration

3. **Constraints balance**
   - Too strict: limited variety
   - Too loose: incoherent playlists
   - Default constraints are well-tuned

4. **Export and share**
   - M3U format works everywhere
   - Name playlists meaningfully
   - Archive favorites

