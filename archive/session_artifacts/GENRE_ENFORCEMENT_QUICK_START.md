# Genre Enforcement System - Quick Start Guide

**Confirmed Working**: 2025-12-16 (Fela Kuti test passed)

---

## TL;DR

Genre enforcement dials are now **fully functional**. Use `min_genre_similarity` to control how strictly genre compatibility is enforced in playlists.

- **min_genre_similarity=0.30**: Strict curation (filters mismatches)
- **min_genre_similarity=0.0**: Sonic discovery (allow all)

---

## One-Minute Setup

### 1. Update Genres (One-time or periodic)
```bash
python scripts/update_genres_v3_normalized.py --artists --albums
```
✓ Auto-triggers `refresh_effective_genres.py`

### 2. Build Artifact (One-time)
```bash
python scripts/analyze_library.py --mode ds
```
✓ Now includes all 34,529 tracks (including those with zero genres)

### 3. Generate Playlist with Genre Enforcement
```bash
python scripts/tune_dial_grid.py \
    --artifact data_matrices_step1.npz \
    --seeds <track_id> \
    --mode dynamic \
    --min-genre-similarity 0.30
```

Done! Genre enforcement is now active.

---

## Key Parameters

### min_genre_similarity
Controls the genre compatibility threshold.

| Value | Mode | Effect | Use Case |
|-------|------|--------|----------|
| 0.30 | Strict | Filters incompatible genres | High-quality, genre-focused playlists |
| 0.15 | Moderate | Allows some genre mixing | Balanced sonic + genre curation |
| 0.0 | Permissive | No genre filter, sonic only | Discovery, exploring sonically similar tracks |

**Example**: Afrobeat seed with min_sim=0.30 will NOT include slowcore/indie/rock tracks, but sonic-only mode (0.0) will.

### genre_method
How to compute genre similarity.

| Method | Formula | Use |
|--------|---------|-----|
| cosine | Dot product / magnitude | Dense genre vectors |
| weighted_jaccard | Jaccard with genre weights | Sparse genre vectors |
| ensemble | 0.6×cosine + 0.4×jaccard | **Default (best balance)** |

**Recommendation**: Use `ensemble` (default) for best results.

---

## How It Works (In 30 Seconds)

1. **Database** stores genres at 4 levels:
   - File tags (highest priority)
   - Album genres (MusicBrainz/Discogs)
   - Artist genres (MusicBrainz/Discogs)
   - Inherited from collaborations (NEW!)

2. **Artifact** builds X_genre matrices:
   - X_genre_raw: Binary incidence matrix
   - X_genre_smoothed: Genre-smoothed vectors

3. **Pipeline** computes genre similarity:
   ```python
   genre_sim = 0.6 * cosine(X, seed) + 0.4 * jaccard(X, seed)
   ```

4. **Gate** filters candidates:
   ```python
   if genre_sim < min_genre_similarity:
       exclude_from_playlist()
   ```

**Result**: Only genre-compatible tracks are eligible for the playlist.

---

## Real-World Example

### Fela Kuti (Afrobeat) Playlist

**Seed**: Fela Kuti & Africa 70 - "Kalakuta Show"
**Seed genres**: funk / soul, afrobeat

#### With Strict Gate (min_genre_similarity=0.30)
```
Generated 20 tracks
Genres: afrobeat, funk/soul, disco/funk, folk/world, jazz, r&b
Problematic genres: NONE ✓
```

#### With Permissive Gate (min_genre_similarity=0.0)
```
Generated 20 tracks
Genres: afrobeat, funk/soul, disco/funk, folk/world, jazz, r&b,
        rock, shoegaze, slowcore ⚠️
Problematic genres: slowcore, rock, shoegaze
```

**Takeaway**: Strict gate prevents genre mismatches, permissive allows discovery.

---

## Understanding the Database

### track_effective_genres table

Stores final genres for each track after applying precedence rules.

```sql
-- Check how many genres each track has
SELECT track_id, COUNT(*) as genre_count
FROM track_effective_genres
GROUP BY track_id
LIMIT 10;

-- Check genre sources
SELECT DISTINCT source FROM track_effective_genres;
-- Result: file, musicbrainz_release, musicbrainz_artist, musicbrainz_artist_inherited

-- Find inherited genres (from collaborations)
SELECT track_id, genre, source
FROM track_effective_genres
WHERE source = 'musicbrainz_artist_inherited'
LIMIT 10;
```

---

## Collaboration Parsing

The system intelligently handles collaborations:

| Artist String | Parsing Result | Reason |
|---------------|----------------|--------|
| Echo & The Bunnymen | [Echo & The Bunnymen] | Band name (not split) |
| Pink Siifu & Fly Anakin | [Pink Siifu, Fly Anakin] | Real collaboration (split) |
| John Coltrane feat. Cannonball | [John Coltrane, Cannonball] | Collaboration (split) |
| Horace Silver Quintet & Trio | [Horace Silver Quintet & Trio] | Ensemble (not split) |

**How it works**: parse_collaboration() detects band name patterns and ensemble suffixes before splitting on delimiters.

---

## Testing & Verification

### Verify Genre Enforcement is Working
```python
from src.playlist.pipeline import generate_playlist_ds

# Strict
strict = generate_playlist_ds(
    artifact_path='data_matrices_step1.npz',
    seed_track_id='1c347ff04e65adf7923a9e3927ab667a',
    num_tracks=20,
    mode='dynamic',
    min_genre_similarity=0.30
)

# Permissive
perm = generate_playlist_ds(
    artifact_path='data_matrices_step1.npz',
    seed_track_id='1c347ff04e65adf7923a9e3927ab667a',
    num_tracks=20,
    mode='dynamic',
    min_genre_similarity=0.0
)

# Should get different results
assert len(set(strict.track_ids) - set(perm.track_ids)) > 0
print("Genre enforcement is working!")
```

### Check Collaboration Parsing
```python
from src.artist_utils import parse_collaboration

# Should preserve band names
assert parse_collaboration("Echo & The Bunnymen") == ["Echo & The Bunnymen"]

# Should split real collaborations
assert parse_collaboration("Pink Siifu & Fly Anakin") == ["Pink Siifu", "Fly Anakin"]

print("Collaboration parsing is working!")
```

---

## Common Workflows

### Workflow 1: Create a Genre-Focused Playlist
```bash
# Step 1: Genre update (if needed)
python scripts/update_genres_v3_normalized.py --artists --albums

# Step 2: Generate with strict gate
python scripts/tune_dial_grid.py \
    --artifact data_matrices_step1.npz \
    --seeds <your_seed_track> \
    --mode dynamic \
    --sonic-weight 0.65 \
    --genre-weight 0.35 \
    --min-genre-similarity 0.30 \
    --genre-method ensemble
```

### Workflow 2: Sonic Discovery (Genre-Permissive)
```bash
python scripts/tune_dial_grid.py \
    --artifact data_matrices_step1.npz \
    --seeds <your_seed_track> \
    --mode dynamic \
    --sonic-weight 0.9 \
    --genre-weight 0.1 \
    --min-genre-similarity 0.0 \
    --genre-method ensemble
```

### Workflow 3: Fine-Tune Genre Enforcement
```bash
# Try different gates on same seed
for min_sim in 0.0 0.15 0.30 0.45; do
    python scripts/tune_dial_grid.py \
        --artifact data_matrices_step1.npz \
        --seeds <seed> \
        --mode dynamic \
        --min-genre-similarity $min_sim \
        --output-dir results/min_sim_${min_sim}
done
```

Then listen to results and pick your preferred threshold.

---

## Troubleshooting

### Problem: "Genre enforcement doesn't seem to work"
**Solution**: Check that:
1. `track_effective_genres` table exists: `SELECT COUNT(*) FROM track_effective_genres;`
2. Artifact is built with latest effective genres: `python scripts/analyze_library.py --mode ds`
3. min_genre_similarity is > 0 and < 1.0

### Problem: "Some tracks don't have genres"
**Solution**: This is normal. 1,146 tracks (3.3%) have zero explicit genres:
- These tracks have empty genre vectors
- They're included in artifacts but excluded if min_genre_similarity > 0
- To include them, use min_genre_similarity = 0.0

### Problem: "Collaboration artist genres are empty"
**Solution**: Run genre update with inheritance:
```bash
python scripts/update_genres_v3_normalized.py --artists
```
This fetches genres from constituent artists.

---

## Architecture at a Glance

```
Database (metadata.db)
├── tracks (audio metadata)
├── artist_genres (by source)
├── album_genres (by source)
├── track_genres (file tags)
└── track_effective_genres (final precedence) ← NEW!

Effective Genres Script
├── Reads all 4 genre sources
├── Applies precedence: file > album > artist > inherited
├── Populates track_effective_genres table
└── Runtime: <2 seconds

Artifact Builder
├── Loads track_effective_genres (all 34,529 tracks)
├── Builds X_genre_raw (binary incidence matrix)
├── Builds X_genre_smoothed (similarity-smoothed)
└── Saves to .npz file

Playlist Generator
├── Loads artifact
├── Computes genre_sim via ensemble method
├── Applies hard gate: sim >= min_genre_similarity
└── Returns filtered candidates
```

---

## Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Tracks in artifact | 34,529 | ✓ All included |
| Tracks with genres | 33,383 | ✓ 96.7% coverage |
| Tracks with inherited genres | 85 | ✓ New! |
| Collaboration coverage | ~70% | ✓ Up from 25% |
| Genre enforcement test | PASS | ✓ Confirmed |

---

## Next Steps

1. **Use genre enforcement** in your playlist generation with min_genre_similarity values
2. **Listen to results** and find your preferred genre gate threshold
3. **Experiment** with sonic_weight vs genre_weight for different moods
4. **Enjoy** better-curated, genre-aware playlists!

---

## Reference Links

- **Complete architecture**: GENRE_SYSTEM_COMPLETE_SUMMARY.md
- **Test results**: GENRE_ENFORCEMENT_TEST_RESULTS.md
- **Implementation details**: IMPLEMENTATION_COMPLETE.md
- **Code changes**: See IMPLEMENTATION_COMPLETE.md for file list

---

## Support

For questions or issues:
1. Check GENRE_SYSTEM_COMPLETE_SUMMARY.md for architecture details
2. Review GENRE_ENFORCEMENT_TEST_RESULTS.md for proof it works
3. Run verification commands above to confirm your setup
4. Check troubleshooting section above

---

**Status**: READY TO USE ✓
**Last Tested**: 2025-12-16
**Test Case**: Fela Kuti playlist with strict genre gate
**Result**: All tests PASSED
