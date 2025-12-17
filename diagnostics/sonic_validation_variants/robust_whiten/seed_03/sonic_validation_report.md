# Sonic Validation Report

**Seed Track**: Sonic Youth - Silver Rocket (33f74d3e1cd2667cb332161fd86998eb)

**Date**: 2025-12-16 14:34:48

## Metrics Summary

### Sonic-Only Nearest Neighbors

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Score Flatness | 0.041 | >= 0.5 | FAIL |
| TopK vs Random Gap | 0.021 | >= 0.15 | FAIL |
| Intra-Artist Coherence | 0.016266447724631816 | >= 0.05 | FAIL |
| Intra-Album Coherence | 0.010061541253611694 | >= 0.08 | FAIL |

### Genre-Only Nearest Neighbors

| Metric | Value |
|--------|-------|
| Score Flatness | 3.454 |
| TopK vs Random Gap | 0.872 |

## Interpretation

### What These Metrics Mean

**Score Flatness**: Measures if scores are spread out or clustered together.
- High flatness (> 1.0) = Good separation between similar and dissimilar tracks
- Low flatness (< 0.5) = Most tracks have similar scores (uninformative)

**TopK vs Random Gap**: Measures if top neighbors are actually better than random.
- High gap (> 0.2) = Top neighbors are much more similar than random
- Low gap (< 0.1) = Top neighbors barely better than random (uninformative)

**Intra-Artist/Album Coherence**: Measures if tracks from same artist/album are closer.
- Positive value = Same artist/album tracks are closer than random (GOOD)
- Negative value = Same artist/album tracks are farther than random (BAD)
- > 0.05 for artist, > 0.08 for album = PASS

### Overall Assessment

**SONIC VALIDATION: FAIL**

Sonic similarity is NOT informative (0/4 metrics pass). The validation metrics show:
- Scores too flat or no discrimination
- Same-artist/album tracks not coherent
- Pure sonic neighbors may not pass listening test

Next steps:
- MUST proceed to Phase B (feature improvements)
- Test new variants (beat_sync, multi_segment_median, robust_whiten)
- Re-run validation suite on improved variants


## Files Generated

1. **sonic_only_top30.m3u** - Top 30 tracks by pure sonic similarity
2. **genre_only_top30.m3u** - Top 30 tracks by pure genre similarity
3. **hybrid_current_top30.m3u** - Top 30 tracks by current hybrid embedding
4. **sonic_validation_metrics.csv** - Detailed metrics for aggregation

Listen to all three playlists and assess:
- Does sonic-only sound coherent? (similar vibes, flow)
- Does genre-only sound coherent? (same style/era)
- Does hybrid sound balanced? (mix of sonic + genre)

Manual listening assessment is crucial for final validation.
