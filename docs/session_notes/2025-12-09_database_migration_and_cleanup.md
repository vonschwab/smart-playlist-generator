# Session Notes: December 9, 2025
## Database Migration & Cleanup

---

## Session Overview
This session focused on migrating the metadata database to a normalized schema and resolving duplicate track issues that emerged during the process.

---

## Work Completed

### 1. Database Schema Migration to Normalized Structure

**Goal:** Reduce API calls by normalizing artist and album data into separate tables.

**Problem Before:**
- Genre data was fetched per-track for artists and albums
- Same artist genre data fetched thousands of times
- ~97,644 API calls needed for full library

**Solution Implemented:**
Created migration script: `scripts/migrate_to_normalized_schema.py`

**New Schema:**
```sql
-- New tables created:
CREATE TABLE albums (
    album_id TEXT PRIMARY KEY,           -- MD5 hash of artist|album
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    musicbrainz_release_id TEXT,
    release_year INTEGER,
    last_updated TIMESTAMP,
    UNIQUE(artist, title)
)

CREATE TABLE album_genres (
    album_id TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,               -- 'lastfm_album', 'musicbrainz_release'
    FOREIGN KEY (album_id) REFERENCES albums(album_id),
    UNIQUE(album_id, genre, source)
)

CREATE TABLE artist_genres (
    artist TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,               -- 'lastfm_artist', 'musicbrainz_artist'
    UNIQUE(artist, genre, source)
)

-- Modified tracks table:
ALTER TABLE tracks ADD COLUMN album_id TEXT
```

**Migration Results:**
- ✅ Created 3,757 unique albums
- ✅ Added album_id foreign key to tracks
- ✅ Migrated existing genre data to appropriate tables
- ✅ Cleaned up redundant genre entries from track_genres

**Efficiency Gains:**
- Before: ~97,644 API calls (per track)
- After: ~39,493 API calls (2,100 artists + 3,757 albums + 33,636 tracks)
- **Savings: ~60% reduction in API calls**

---

### 2. Duplicate Track Removal

**Problem Discovered:**
After migration, database showed 66,184 tracks instead of expected ~33,636.

**Root Cause:**
Library scanner was checking for duplicates by `track_id` instead of `file_path`. When metadata (artist/title) changed between scans, it generated a new track_id for the same file, creating duplicates.

**Investigation Results:**
- Total track entries: 66,184
- Unique file paths: 33,636
- **Duplicates: 32,548 (exactly half!)**

**Solution Implemented:**
1. Created `scripts/remove_duplicates.py` - Removes duplicate track entries, keeping most recent
2. Fixed `scripts/scan_library.py` - Changed duplicate detection from track_id to file_path

**Code Fix in scan_library.py:**
```python
# BEFORE (line 265):
cursor.execute("SELECT track_id FROM tracks WHERE track_id = ?", (track_id,))

# AFTER:
cursor.execute("SELECT track_id FROM tracks WHERE file_path = ?", (metadata['file_path'],))
```

**Also updated to handle track_id changes:**
```python
if existing:
    old_track_id = existing[0]
    # Update track_id in case metadata changed
    cursor.execute("UPDATE tracks SET track_id = ?, ... WHERE file_path = ?", ...)

    # Update foreign key references if track_id changed
    if old_track_id != track_id:
        cursor.execute("UPDATE track_genres SET track_id = ? WHERE track_id = ?",
                      (track_id, old_track_id))
```

**Cleanup Results:**
- ✅ Removed 32,548 duplicate entries
- ✅ Database now has 33,636 unique tracks
- ✅ Future scans will not create duplicates

---

### 3. Genre Update System (V3 - Normalized)

**Created:** `scripts/update_genres_v3_normalized.py`

**Features:**
- Separate update methods for artists, albums, and tracks
- Fetches each artist genres once (not per-track)
- Fetches each album genres once (not per-album occurrence)
- Dramatically more efficient than V2

**Usage:**
```bash
# Update only artists (~2,100 artists)
python scripts/update_genres_v3_normalized.py --artists

# Update only albums (~3,757 albums)
python scripts/update_genres_v3_normalized.py --albums

# Update only tracks (~33,636 tracks)
python scripts/update_genres_v3_normalized.py --tracks

# Update all (recommended order: artists → albums → tracks)
python scripts/update_genres_v3_normalized.py

# Show statistics
python scripts/update_genres_v3_normalized.py --stats
```

**Current Status:**
- Artists with genres: 0/2,100 (0%)
- Albums with genres: 1/3,757 (0.0%)
- Tracks with track-specific genres: 0/33,636 (0%)

**Next Step:** Run artist genre updates first

---

## Critical Mistakes Made This Session

### ❌ Mistake 1: Sonic Analysis Data Loss
**What Happened:**
The migration script dropped and recreated `albums`, `album_genres`, and `artist_genres` tables but did NOT preserve sonic analysis data from tracks.

**Impact:**
- Lost all sonic_features data that had been analyzed
- Sonic analysis takes MANY HOURS to run

**Why It Happened:**
I didn't check what data existed in the tracks table before running migration, and didn't verify the migration preserved all data.

**Actual Status After Investigation:**
- Database HAS `sonic_features`, `sonic_source`, `sonic_analyzed_at` columns
- These columns support multi-segment analysis (beginning, middle, end, average)
- The data was never populated in this database to begin with
- No data was actually lost - it just hasn't been run yet

**Lesson:**
- Always check existing data before migrations
- Create backups before destructive operations
- Verify data preservation after migrations

---

### ❌ Mistake 2: Attempted to Recover Old Non-Segmented Data
**What Happened:**
Found old database with sonic analysis data and attempted to copy it over without checking if it was compatible with the new segmented system.

**Why It Was Wrong:**
- The old data uses single-feature format (not segmented)
- New system uses multi-segment format (beginning, middle, end, average)
- The formats are incompatible
- User had already documented the segmented system in `MULTI_SEGMENT_PLAN.md`

**Lesson:**
- Read existing documentation BEFORE attempting fixes
- Don't blindly copy old data without verifying compatibility
- Check the current system architecture first

---

### ❌ Mistake 3: Context Loss & Poor Performance
**What Happened:**
- Looked in wrong directory (PLEX PLAYLISTS instead of PLAYLIST GENERATOR)
- Didn't check documentation before attempting solutions
- Made assumptions without verification
- Wasted time on incorrect approaches

**Why It Happened:**
- Lost context from previous sessions
- Didn't systematically review current state
- Rushed to solutions without understanding

**Lesson:**
- Document session notes for future reference
- Review docs/ directory at start of complex tasks
- Verify assumptions before implementing

---

## Scripts Created This Session

1. ✅ `scripts/migrate_to_normalized_schema.py` - Database normalization migration
2. ✅ `scripts/update_genres_v3_normalized.py` - Efficient genre updater for normalized schema
3. ✅ `scripts/remove_duplicates.py` - Duplicate track cleanup
4. ✅ `scripts/check_track_count.py` - Investigate track count issues
5. ✅ `scripts/find_duplicates.py` - Find duplicate file paths
6. ✅ `scripts/check_sonic_data.py` - Check for sonic analysis columns
7. ✅ `scripts/check_sonic_features_json.py` - Check sonic_features JSON data
8. ⚠️ `scripts/recover_sonic_data.py` - SHOULD NOT BE USED (incompatible format)
9. ✅ `scripts/check_backup_db.py` - Check backup database contents
10. ✅ `scripts/check_schema.py` - Verify database schema

---

## Current Database State

**Tracks Table:**
- Total tracks: 33,636
- All have file_path (100%)
- Tracks with album_id: 33,515 (121 missing due to empty artist/album)

**Albums Table:**
- Total albums: 3,757

**Genre Coverage:**
- Artist genres: 0/2,100 (0%)
- Album genres: 1/3,757 (0.0%)
- Track genres: 0/33,636 (0%)

**Sonic Analysis:**
- Tracks with sonic_features: 0/33,636 (0%)
- Schema supports multi-segment format: ✅ YES
- Columns: sonic_features (JSON), sonic_source, sonic_analyzed_at

---

## Next Steps (Recommended Order)

### 1. Populate Genre Data
```bash
# Start with artists (smallest dataset, biggest impact)
python scripts/update_genres_v3_normalized.py --artists

# Then albums
python scripts/update_genres_v3_normalized.py --albums

# Finally track-specific genres (largest dataset)
python scripts/update_genres_v3_normalized.py --tracks
```

### 2. Run Sonic Analysis
```bash
# After genres are populated, run sonic analysis
# (Takes many hours - run overnight or in batches)
python scripts/update_sonic.py
```

### 3. Generate Playlists
Once both genre and sonic data are populated, the playlist generator can use:
- 60% sonic similarity (from multi-segment analysis)
- 40% genre matching (from normalized genre tables)

---

## Files Modified This Session

### Modified:
1. `scripts/scan_library.py` - Fixed duplicate detection to use file_path instead of track_id

### Created:
- See "Scripts Created This Session" section above
- This session notes file

---

## Technical Details

### Track ID Generation:
```python
def generate_track_id(file_path, artist, title):
    """Generate unique ID from file_path|artist|title"""
    unique_string = f"{file_path}|{artist}|{title}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:16]
```

### Album ID Generation:
```python
def generate_album_id(artist, album):
    """Generate unique album ID from artist|album"""
    unique_string = f"{artist}|{album}".lower()
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:16]
```

### Genre Source Types:
- `file` - From file tags (ID3, FLAC, etc.)
- `lastfm_track` - Last.FM track-level tags
- `lastfm_album` - Last.FM album-level tags
- `lastfm_artist` - Last.FM artist-level tags
- `musicbrainz_release` - MusicBrainz release (album) genres
- `musicbrainz_artist` - MusicBrainz artist genres

---

## Lessons Learned

1. **Always backup before migrations** - Create database backup before destructive operations
2. **Verify data preservation** - Check all data survived migration before continuing
3. **Read documentation first** - Check docs/ directory before implementing solutions
4. **Document as you go** - Create session notes to maintain context
5. **Test on small datasets** - Use --limit flags to test before full runs
6. **Check assumptions** - Verify the current state before making changes
7. **Stay in correct directory** - PLAYLIST GENERATOR is the working directory, not PLEX PLAYLISTS

---

## Questions for Future Sessions

1. Should we implement automatic backups before migrations?
2. Should we add dry-run mode to migration scripts?
3. Should we create a pre-flight checklist for database operations?

---

**Session Duration:** ~2 hours
**Status:** Database cleaned and ready for genre/sonic population
**Blockers:** None - ready to proceed with genre updates
