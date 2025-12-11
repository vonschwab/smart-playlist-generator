# Genre Similarity System Documentation

## How the Combined MusicBrainz + Last.FM Genre Similarity System Works

### **Overview**
The system uses a **hybrid scoring approach** that combines:
1. **Sonic similarity** (audio features like timbre, tempo, harmony)
2. **Genre similarity** (metadata from Last.FM and MusicBrainz)

---

## **Step-by-Step Process**

### **1. Genre Data Collection (Per Source)**

Genres are collected from multiple sources and stored separately in the normalized database:

**Last.FM Sources:**
- `lastfm_track` - Track-specific tags (e.g., "Heroes" by David Bowie → ["glam rock", "art rock"])
- `lastfm_album` - Album tags (e.g., "Low" → ["experimental", "ambient"])
- `lastfm_artist` - Artist tags (e.g., David Bowie → ["glam rock", "art rock", "80s"])

**MusicBrainz Sources:**
- `musicbrainz_release` - Release (album) genres
- `musicbrainz_artist` - Artist genres

**File Tags:**
- `file` - Genres embedded in MP3/FLAC files

---

### **2. Genre Combination (Priority-Based)**

When calculating similarity for a track, genres are retrieved with **prioritization** via `src/metadata_client.py:210`:

```python
def get_combined_track_genres(self, track_id: str):
    # Priority order (most specific → least specific)
    priority = [
        'lastfm_track',          # 1. Most specific - this exact track
        'lastfm_album',          # 2. Album context
        'musicbrainz_release',   # 3. MusicBrainz album
        'lastfm_artist',         # 4. Artist general style
        'musicbrainz_artist'     # 5. MusicBrainz artist
    ]
```

**How it works:**
1. Starts with most specific genres (track-level)
2. Adds album-level genres that aren't duplicates
3. Adds artist-level genres that aren't duplicates
4. Returns a **deduplicated list** maintaining priority order

**Example for "Heroes" by David Bowie:**
```
Input sources:
- lastfm_track: ["glam rock", "art rock", "classic rock"]
- lastfm_album: ["experimental", "glam rock"]
- lastfm_artist: ["rock", "80s", "glam rock"]
- musicbrainz_artist: ["art rock", "new wave"]

Combined result: ["glam rock", "art rock", "classic rock", "experimental", "rock", "80s", "new wave"]
                  ^track       ^track     ^track            ^album         ^artist ^artist ^MB
```

---

### **3. Genre Similarity Calculation**

Uses the **curated similarity matrix** at `data/genre_similarity.yaml`:

**Matrix Structure:**
```yaml
indie rock:
  alternative rock: 0.9   # Very similar
  shoegaze: 0.7          # Related
  post-punk: 0.7         # Related
  lo-fi: 0.6             # Somewhat related
  garage rock: 0.55      # Loosely related
```

**Similarity Algorithm** (`src/genre_similarity.py:39`):

```python
def calculate_similarity(genres1, genres2):
    # Compare ALL genres from track 1 to ALL genres from track 2
    max_similarity = 0.0

    for g1 in genres1:
        for g2 in genres2:
            if g1 == g2:
                return 1.0  # Exact match - perfect score

            # Look up in matrix (checks both directions)
            sim = lookup_similarity(g1, g2)
            max_similarity = max(max_similarity, sim)

    return max_similarity
```

**Key behaviors:**
- **Exact match** = 1.0 (if any genre appears in both lists)
- **Matrix lookup** = finds closest relationship (e.g., "indie rock" ↔ "shoegaze" = 0.7)
- **Bidirectional** = checks both `genre1→genre2` and `genre2→genre1`
- **Returns maximum** = uses the BEST match found across all genre pairs

---

### **4. Hybrid Similarity Scoring**

The final similarity score combines sonic + genre (`src/similarity_calculator.py:325`):

```python
def calculate_hybrid_similarity(track1_id, track2_id):
    # 1. Calculate sonic similarity (MFCC, chroma, tempo, etc.)
    sonic_sim = calculate_similarity(features1, features2)

    # 2. Get combined genres for both tracks
    genres1 = get_combined_track_genres(track1_id)
    genres2 = get_combined_track_genres(track2_id)

    # 3. Calculate genre similarity
    genre_sim = genre_calc.calculate_similarity(genres1, genres2)

    # 4. Filter: Reject if genre similarity too low
    if genre_sim < min_genre_similarity:  # Default: 0.3
        return 0.0  # REJECTED - genres too different

    # 5. Weighted combination
    hybrid_sim = (sonic_sim * 0.5) + (genre_sim * 0.5)

    return hybrid_sim
```

**Current weights** (from `config.yaml`):
- **Sonic weight**: 50% (audio analysis)
- **Genre weight**: 50% (metadata similarity)
- **Minimum genre threshold**: 0.3 (blocks cross-genre matches)

---

## **Practical Example**

Let's say we're comparing two tracks:

**Track A: "Lithium" by Nirvana**
```
Combined genres: ["grunge", "alternative rock", "rock", "90s"]
Sonic features: Heavy distortion, moderate tempo, loud
```

**Track B: "Teenage Riot" by Sonic Youth**
```
Combined genres: ["noise rock", "alternative rock", "indie rock", "experimental"]
Sonic features: Distorted guitars, driving rhythm
```

**Genre Similarity Calculation:**
```
Compare all pairs:
- "grunge" vs "noise rock" → 0.65 (from matrix)
- "grunge" vs "alternative rock" → 0.75 (from matrix)
- "alternative rock" vs "alternative rock" → 1.0 (EXACT MATCH!)
- ... other pairs checked ...

Result: genre_sim = 1.0 (best match)
```

**Hybrid Score:**
```
sonic_sim = 0.82 (similar distortion, tempo, energy)
genre_sim = 1.0 (exact match on "alternative rock")

hybrid_sim = (0.82 * 0.5) + (1.0 * 0.5) = 0.91
```

**Verdict:** ✅ **Highly similar** (0.91/1.0)

---

## **Filtering Example (Blocking Cross-Genre)**

**Track A: Ahmad Jamal - "Poinciana" (Jazz)**
```
Combined genres: ["jazz", "piano jazz", "bebop"]
```

**Track B: Duster - "Inside Out" (Slowcore)**
```
Combined genres: ["slowcore", "lo-fi", "indie rock"]
```

**Genre Similarity:**
```
Check all pairs:
- "jazz" vs "slowcore" → Not in matrix → 0.0
- "jazz" vs "lo-fi" → Not in matrix → 0.0
- "jazz" vs "indie rock" → Not in matrix → 0.0
... all pairs return 0.0 ...

Result: genre_sim = 0.0
```

**Filtering:**
```
if genre_sim (0.0) < min_threshold (0.3):
    return 0.0  # BLOCKED
```

**Verdict:** ❌ **Rejected** - Even if sonic similarity is decent, genres are too different

---

## **Key Advantages of This System**

### **1. Multi-Source Redundancy**
- If Last.FM has limited data, MusicBrainz fills gaps
- Track-level tags > album tags > artist tags (specificity priority)

### **2. Smart Deduplication**
- Doesn't count "rock" twice if it appears in both Last.FM and MusicBrainz
- Maintains priority order (track-specific genres come first)

### **3. Curated Relationships**
- Manual genre similarity matrix prevents bad matches
- You control which genres are considered related

### **4. Filtering Prevents Cross-Genre Pollution**
- The `min_genre_similarity: 0.3` threshold blocks wildly different genres
- Prevents jazz from matching slowcore, even if they have similar tempo

### **5. Balanced Weighting**
- 50/50 split ensures both sonic and genre matter
- Can be tuned in `config.yaml` based on your preferences

---

## **Configuration Options**

Edit `config.yaml` to tune behavior:

```yaml
genre_similarity:
  enabled: true              # Turn genre filtering on/off
  weight: 0.5               # Genre contribution to final score
  sonic_weight: 0.5         # Sonic contribution to final score
  min_genre_similarity: 0.3  # Minimum genre match required
  similarity_file: "data/genre_similarity.yaml"
```

**Recommendations:**
- **More genre-focused**: Increase `weight` to 0.6, reduce `sonic_weight` to 0.4
- **More sonic-focused**: Increase `sonic_weight` to 0.7, reduce `weight` to 0.3
- **Stricter filtering**: Increase `min_genre_similarity` to 0.4 or 0.5
- **Looser filtering**: Decrease `min_genre_similarity` to 0.2

---

## **Implementation Details**

### **Empty Marker System**

The system uses `'__EMPTY__'` markers to distinguish between:
- **Never checked**: No entry in database
- **Checked but empty**: `'__EMPTY__'` marker present

**Why this matters:**
- Prevents redundant API calls for artists/albums with no genre data
- All genre retrieval methods automatically filter out `'__EMPTY__'` markers
- Enables true incremental updates (per-source checking)

### **Per-Source Incremental Updates**

The genre update script (`scripts/update_genres_v3_normalized.py`) checks each source independently:

```python
# Example: Artist with partial data
David Bowie:
  - lastfm_artist: ["glam rock", "art rock", "80s"]  ✓ Present
  - musicbrainz_artist: (missing)                    ✗ Needs fetch

Next update will:
  - Skip Last.FM (already have data)
  - Fetch MusicBrainz only
```

This enables:
- **Resumability** after crashes
- **Efficiency** (no redundant API calls)
- **Completeness** (fills in missing sources)

---

## **Database Schema**

### **Genre Tables**

```sql
-- Artist-level genres (one per artist)
CREATE TABLE artist_genres (
    artist TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'lastfm_artist', 'musicbrainz_artist'
    UNIQUE(artist, genre, source)
)

-- Album-level genres (one per album)
CREATE TABLE album_genres (
    album_id TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'lastfm_album', 'musicbrainz_release'
    UNIQUE(album_id, genre, source)
)

-- Track-level genres (many sources per track)
CREATE TABLE track_genres (
    track_id TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'lastfm_track', 'file', 'lastfm_artist', etc.
    UNIQUE(track_id, genre, source)
)
```

### **Genre Source Types**

| Source | Level | Description |
|--------|-------|-------------|
| `lastfm_track` | Track | Last.FM track-specific tags (most specific) |
| `lastfm_album` | Album | Last.FM album tags |
| `lastfm_artist` | Artist | Last.FM artist tags |
| `musicbrainz_release` | Album | MusicBrainz release genres |
| `musicbrainz_artist` | Artist | MusicBrainz artist genres |
| `file` | Track | Genres from file tags (ID3/FLAC) |

---

## **Code References**

| Component | File | Key Methods |
|-----------|------|-------------|
| Genre similarity calculation | `src/genre_similarity.py` | `calculate_similarity()` |
| Hybrid scoring | `src/similarity_calculator.py` | `calculate_hybrid_similarity()` |
| Genre combination | `src/metadata_client.py` | `get_combined_track_genres()` |
| Genre updates | `scripts/update_genres_v3_normalized.py` | `update_artist_genres()`, `update_album_genres()`, `update_track_genres()` |
| Similarity matrix | `data/genre_similarity.yaml` | N/A (data file) |

---

## **Future Improvements**

Potential enhancements to consider:

1. **Machine learning genre relationships** - Auto-generate similarity matrix from listening data
2. **Weighted genre importance** - Give more weight to track-level genres vs artist-level
3. **Genre evolution over time** - Handle artists that change genres (e.g., Radiohead)
4. **User feedback loop** - Learn from skipped tracks to refine genre relationships
5. **Sub-genre handling** - Better handling of specific sub-genres (e.g., "math rock" vs "rock")

---

## **Troubleshooting**

### **Problem: Too many cross-genre matches**
- **Solution**: Increase `min_genre_similarity` to 0.4 or 0.5
- **Check**: Expand `data/genre_similarity.yaml` to include missing relationships

### **Problem: Not enough variety in playlists**
- **Solution**: Decrease `min_genre_similarity` to 0.2
- **Alternative**: Increase `sonic_weight` to rely more on audio features

### **Problem: Jazz matching slowcore (or similar unwanted pairs)**
- **Solution**: Ensure these genres are NOT in `genre_similarity.yaml`
- **Verify**: Check that `min_genre_similarity` is set (should be at least 0.3)

### **Problem: Missing genres for some tracks**
- **Solution**: Run genre update script: `python scripts/update_genres_v3_normalized.py`
- **Check**: Verify Last.FM API key is configured in `config.yaml`

---

**Last Updated**: December 9, 2025
**Version**: 3.0 (Normalized Schema)
