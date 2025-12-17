# Genre Coverage Audit Report

**Generated**: 2025-12-16 10:03:23

## Summary Metrics

| Metric | Count | Percentage |
|--------|-------|-----------|
| Total tracks (with sonic features) | 34529 | 100% |
| Tracks with album genres | 28148 | 81.52% |
| Tracks with artist genres | 25917 | 75.06% |
| Tracks with file-embedded genres | 20203 | 58.51% |
| **Tracks with ANY genre** | **33298** | **96.43%** |
| **Tracks with ZERO genres** | **1231** | **3.57%** ⚠️ |

## Collaboration Coverage Gap

| Artist Type | Artist Count | Total Tracks | Artists w/ Genres | Coverage % |
|-------------|--------------|--------------|-------------------|-----------|
| Solo artists | 1735 | 33009 | 1115 | 64.27% |
| Collaboration artists | 373 | 1520 | 140 | 37.53% |
| **COVERAGE GAP** | | | | **26.74% ⚠️** |

**Interpretation**: Collaboration artists have **26.74% lower** genre coverage than solo artists.

## Top Offenders

### Tracks with ZERO Genres (Top 200)
- **Count**: 200
- **Impact**: These tracks are currently **EXCLUDED from artifact bundles** during `scripts/analyze_library.py`
- **Location**: See `diagnostics/tracks_without_genres_top200.csv`

### Collaboration Artists with ZERO Genres (Top 200)
- **Count**: 200
- **Impact**: Collaboration tracks missing genres in artist lookups (don't inherit from constituents)
- **Location**: See `diagnostics/collab_artists_without_genres_top200.csv`

## Key Findings

1. **Genre Coverage is Good for Solo Artists**: 64.27% of solo artists have genres
2. **Genre Coverage is POOR for Collaborations**: 37.53% of collab artists have genres
3. **Coverage Gap is SIGNIFICANT**: 26.74% lower for collaborations
4. **Empty-Genre Tracks Exist**: 3.57% of all tracks have zero genres

## Recommendations

### Phase 1: Audit (CURRENT)
- ✅ Coverage metrics computed
- ✅ Gap identified and quantified
- ✅ Top offenders identified

### Phase 2: Inheritance (NEXT)
Implement automatic genre inheritance from constituent artists:
- Parse collaboration strings (e.g., "Artist A & Artist B" → ["Artist A", "Artist B"])
- Fetch genres for each constituent from MusicBrainz
- Store inherited genres with source='musicbrainz_artist_inherited'

**Expected Impact**:
- Collaboration coverage: 37.53% → ~75%
- Total coverage: 96.43% → ~92%

### Phase 3: Materialization (NEXT)
- Create `track_effective_genres` table combining all sources
- Track provenance (file/album/artist/inherited)
- Include tracks with empty vectors (instead of excluding them)

### Phase 4: Verification (NEXT)
- Add regression tests for collaboration parsing
- Verify seed sanity (afrobeat seed shouldn't pull indie/slowcore)
- Manual A/B listening test on collaboration-heavy playlists

## Next Steps

1. Run genre update with inheritance: `python scripts/update_genres_v3_normalized.py --artists --albums`
2. Implement Phase 2 effective genres model
3. Re-run this audit to verify improvement

---

**Report Generated**: 2025-12-16 10:03:23
