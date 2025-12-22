# Discogs Integration - Production Status

**Status**: MOVED TO PRODUCTION (December 18, 2025)

Previously, Discogs genre fetching was relegated to legacy scripts. As of this deployment, **Discogs is now a required part of the production pipeline**.

---

## Why Discogs?

The database shows Discogs is a valuable complementary source:

```
Album genre sources in current database:
- MusicBrainz releases: 3,757 albums
- Discogs releases: 3,788 albums
- Discogs masters: 2,074 albums

Coverage: Discogs adds genres for albums that may lack MusicBrainz data
Advantage: More granular style information (e.g., "synthpop" vs just "pop")
```

By integrating Discogs into the production pipeline, we ensure **comprehensive genre coverage** for all library albums, which feeds directly into:
- Genre-based candidate filtering (hard gate at 0.2 similarity)
- Genre similarity matrix computation
- DS artifact genre vectors (33.3% of hybrid embedding)

---

## Getting Started

### Step 1: Get Your Discogs Token

1. Visit https://www.discogs.com/settings/developers
2. Create a personal user token (not OAuth, just a simple token)
3. Keep it safe - this grants access to your Discogs account

### Step 2: Configure Token

Choose **one** of these methods:

**Option A: Environment Variable (Recommended for scripts)**
```bash
export DISCOGS_TOKEN="your_token_here"
python scripts/analyze_library.py
```

**Option B: config.yaml (Recommended for persistent configuration)**
```yaml
discogs:
  token: "your_token_here"
```

### Step 3: Run Pipeline

The Discogs stage is now **always included by default**:

```bash
# Full pipeline (includes discogs stage automatically)
python scripts/analyze_library.py --out-dir data/artifacts/beat3tower_32k

# Custom stages (must explicitly include if you want it)
python scripts/analyze_library.py --stages scan,genres,discogs,sonic,artifacts,verify
```

### Step 4: Monitor Progress

The Discogs stage will:
1. Fetch token from environment or config
2. Iterate through all albums in your library
3. Fuzzy-match each album against Discogs database
4. Fetch genres and styles for matched releases
5. Upsert results into `album_genres` table

Progress logged every 50 albums:
```
Discogs: 50/3788 processed (22 hits, est 284s remaining)
```

---

## Pipeline Integration

### Default Pipeline Order

```
scan → genres → discogs → sonic → genre-sim → artifacts → verify
           ↓
    Normalized artist/album genres
           ↓
           discogs ← NEW
    Additional Discogs genres
           ↓
    All genres ready for similarity computation
```

**Key**: Discogs stage runs AFTER normalized genres but BEFORE sonic, so genre similarity matrix has full data.

### Error Handling

**If Discogs token missing**:
```
ERROR: Discogs token required for production pipeline.
Set DISCOGS_TOKEN environment variable or add discogs.token to config.yaml.
Get token from: https://www.discogs.com/settings/developers
```

**If Discogs API fails**:
- Rate limiting (429 errors) → Auto-retry with exponential backoff
- Network errors → Logged as debug, album marked `__EMPTY__`
- Search failures → Album skipped, marked as miss

---

## Command Reference

### Run Full Pipeline with New Files

```bash
# Incremental: scan + process new files
python scripts/analyze_library.py \
    --out-dir data/artifacts/beat3tower_32k

# Forces full reprocessing (slower, but ensures all Discogs data)
python scripts/analyze_library.py \
    --force \
    --out-dir data/artifacts/beat3tower_32k
```

### Dry Run (Preview)

```bash
python scripts/analyze_library.py --dry-run
```

### Custom Stage Selection

```bash
# Skip Discogs (for testing/debugging only)
python scripts/analyze_library.py --stages scan,genres,sonic,genre-sim,artifacts,verify

# Only Discogs (re-fetch genres)
python scripts/analyze_library.py --stages discogs --force
```

### Limit Processing

```bash
# Process only 100 albums (useful for testing)
python scripts/analyze_library.py --limit 100

# Process only sonic/discogs stages
python scripts/analyze_library.py --stages discogs,sonic --limit 100
```

---

## Database Schema

Discogs data stored in `album_genres` table:

```sql
CREATE TABLE album_genres (
    album_id TEXT NOT NULL,
    genre TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'discogs_release', 'discogs_master', 'musicbrainz_*'
    PRIMARY KEY (album_id, genre, source)
);

-- Example Discogs entries:
-- album_id: "abc123"
-- genres (from discogs_release): "electronic", "synthpop", "indie"
-- styles (from discogs_master): "synth-pop", "darkwave", "electronic"
```

---

## Performance

- **Rate**: ~0.7 requests/second (Discogs API limit)
- **Per album**: ~3 requests (search + release + master fetch)
- **Time estimate**: ~4.3 seconds per album
- **For 3,788 albums**: ~4.5 hours (runs incrementally, only new albums)
- **Network required**: Yes (live API calls)

---

## Troubleshooting

### Q: "No Discogs token found"
**A**: Set `DISCOGS_TOKEN` env var or add `discogs.token` to config.yaml

### Q: "Discogs rate limit hit (429)"
**A**: Normal - auto-retry. May take longer. Be patient.

### Q: "Search failed for [Album]"
**A**: Album marked as miss. Discogs doesn't have it or fuzzy-match failed.
   - Try `--strict-artist` flag to be more selective (legacy script only)

### Q: Want to re-check missed albums?
**A**: Use legacy script with `--recheck-empty`:
```bash
python archive/legacy_scripts/update_discogs_genres.py \
    --recheck-empty \
    --limit 100
```

### Q: "Already processed 3788 albums; use --force to recheck"
**A**: If you want to re-fetch all Discogs data:
```bash
python scripts/analyze_library.py --stages discogs --force
```

---

## What Changed?

| Before | After |
|--------|-------|
| Discogs in `archive/legacy_scripts/` | Discogs in `scripts/` (production) |
| Optional, manual execution | Required, automatic in pipeline |
| Separate from analyze_library | Integrated as `discogs` stage |
| No config support | Config via `config.yaml` |
| Not used in artifact build | Genres feed into genre similarity matrix |

---

## Next Steps

1. Set your DISCOGS_TOKEN
2. Run: `python scripts/analyze_library.py`
3. Pipeline will fetch Discogs genres as part of normal flow
4. Genres integrated into similarity computation
5. Better genre-based filtering and diversity

---

## References

- Discogs API: https://www.discogs.com/developers
- Personal user token: https://www.discogs.com/settings/developers
- Rate limiting: 0.7 requests/second
- Fuzzy matching: Token overlap similarity (60% album + 30% artist + 10% year)
- Database schema: See `src/schema.py`

---

## Migration Notes

**Existing database**: Discogs stage will:
- Skip albums already processed (3,788 albums exist)
- Process only new library additions
- Use `--force` flag to re-process all albums

**No data loss**: Existing MusicBrainz genres preserved, Discogs added alongside

