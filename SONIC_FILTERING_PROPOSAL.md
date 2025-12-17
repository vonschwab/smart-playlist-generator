# Sonic Feature Filtering Proposal

**Date**: December 2024
**Status**: Proposal for Review
**Priority**: Medium (Quality Enhancement)

---

## Executive Summary

Add tempo (BPM) and brightness (spectral centroid) filtering to the candidate pool generation to eliminate jarring transitions. These filters will work alongside existing genre and transition quality gates to improve playlist coherence.

**Proposal**: Implement 2 sonic feature hard gates that filter unsuitable candidates before ranking.

**Timeline**: 1-2 weeks (Phase 1) + optional artifact enhancement later
**Effort**: ~150 lines of code
**Risk**: Low (optional, backward compatible, minimal performance impact)
**Benefit**: Higher quality playlists with fewer jarring transitions

---

## Problem Statement

### Current State
The candidate pool generation uses:
1. **Hybrid similarity** (60% sonic + 40% genre) - ✅ Fast ranking
2. **Genre similarity gate** - ✅ Ensures harmonic/style compatibility
3. **Transition quality floor** - ✅ Ensures smooth segment-level transitions
4. **Artist diversity** - ✅ Prevents same-artist clustering

### What's Missing
**Sonic feature filtering** - No hard gates based on:
- **BPM (Tempo)**: Two songs with vastly different tempos can pass all gates but sound jarring (e.g., 60 BPM ballad → 170 BPM dance)
- **Spectral Centroid (Brightness)**: A dark, warm sound → bright, harsh sound can pass gates but feels unnatural

### Impact
Without sonic filtering:
- Users can get playlists with acceptable genre/transition scores but tempo/brightness whiplash
- Example: Slow acoustic ballad (60 BPM, 2000 Hz) → Fast electronic dance (170 BPM, 8000 Hz)
  - Genre might match (both "pop")
  - Transition quality might be acceptable
  - But listener immediately notices the jarring switch

---

## Proposed Solution

### Add Two Sonic Feature Filters

Apply optional hard gates **after** genre filtering, before ranking:

```
Build Candidate Pool:
├─ Similarity Floor (0.30) ← EXISTING
├─ Genre Similarity Gate ← EXISTING
├─ BPM Filtering (NEW)
│  └─ max_bpm_ratio: 1.5 (allow up to 1.5x tempo variation)
├─ Spectral Centroid Filtering (NEW)
│  └─ max_centroid_diff_hz: 3000 (allow up to 3000Hz brightness difference)
├─ Artist Grouping & Ranking ← EXISTING
└─ Per-Artist Caps ← EXISTING
```

### Configuration

```yaml
# config.yaml
candidate_pool:
  similarity_floor: 0.30
  max_pool_size: 500
  target_artists: 20

  # NEW: Sonic feature filtering (optional)
  max_bpm_ratio: 1.5              # null to disable
  max_centroid_diff_hz: 3000      # null to disable
```

### Examples

**Balanced Config** (Recommended)
```yaml
max_bpm_ratio: 1.5              # 100 BPM → accepts 67-150 BPM
max_centroid_diff_hz: 3000      # 4000 Hz → accepts 1000-7000 Hz
```

**Conservative** (Tight Filtering)
```yaml
max_bpm_ratio: 1.2              # 100 BPM → accepts 83-120 BPM
max_centroid_diff_hz: 2000      # 4000 Hz → accepts 2000-6000 Hz
```

**Liberal** (Loose Filtering)
```yaml
max_bpm_ratio: 2.0              # 100 BPM → accepts 50-200 BPM
max_centroid_diff_hz: 5000      # 4000 Hz → accepts -1000-9000 Hz
```

**Disabled** (Current Behavior)
```yaml
max_bpm_ratio: null             # No filtering
max_centroid_diff_hz: null      # No filtering
```

---

## How It Works

### BPM Filtering
```python
# Calculate ratio of tempos
ratio = max(seed_bpm, candidate_bpm) / min(seed_bpm, candidate_bpm)

# Filter: only keep if ratio <= max_ratio
if ratio <= 1.5:
    keep_candidate()
else:
    reject_candidate()
```

**Examples**:
- Seed 100 BPM + Candidate 110 BPM → ratio 1.1 ✅ (keep)
- Seed 100 BPM + Candidate 150 BPM → ratio 1.5 ✅ (keep)
- Seed 100 BPM + Candidate 160 BPM → ratio 1.6 ❌ (reject)
- Seed 60 BPM + Candidate 90 BPM → ratio 1.5 ✅ (keep)
- Seed 60 BPM + Candidate 100 BPM → ratio 1.67 ❌ (reject)

### Spectral Centroid Filtering
```python
# Calculate difference in brightness
diff = abs(seed_centroid - candidate_centroid)

# Filter: only keep if difference <= max_diff
if diff <= 3000:
    keep_candidate()
else:
    reject_candidate()
```

**Examples**:
- Seed 4000 Hz + Candidate 4100 Hz → diff 100 Hz ✅ (keep)
- Seed 4000 Hz + Candidate 6500 Hz → diff 2500 Hz ✅ (keep)
- Seed 4000 Hz + Candidate 7500 Hz → diff 3500 Hz ❌ (reject)
- Seed 2000 Hz + Candidate 4500 Hz → diff 2500 Hz ✅ (keep)
- Seed 8000 Hz + Candidate 3500 Hz → diff 4500 Hz ❌ (reject)

---

## Impact Analysis

### Performance
- **Cost per filter**: O(N) vectorized numpy operation
- **BPM filtering**: ~1ms for 34,000 tracks
- **Centroid filtering**: ~1ms for 34,000 tracks
- **Total overhead**: <2ms per playlist generation
- **Current pool generation**: 100-200ms
- **New total**: 102-202ms (imperceptible)

### Filtering Impact (Typical)
```
Starting eligible candidates: 500
After BPM filter (ratio 1.5): 450 (-50, ~10%)
After centroid filter (diff 3000Hz): 400 (-50, ~10%)
Final pool for ranking: 400

Total exclusion: 100 candidates (20% of original)
```

### Playlist Quality
Expected improvements:
- ✅ Fewer tempo-induced whiplashes
- ✅ Smoother brightness progression
- ✅ More natural playlist flow
- ✅ Better listener experience

---

## Implementation Plan

### Phase 1: Quick Implementation (1-2 weeks)

**Approach**: Extract BPM and spectral centroid from existing X_sonic feature vectors

**Advantages**:
- ✅ Works with current artifacts (no rebuild needed)
- ✅ Simple extraction (~30 lines)
- ✅ Filtering logic (~100 lines)
- ✅ Can validate approach immediately

**Files to Modify**:
1. `src/playlist/config.py` (~20 lines)
   - Add `max_bpm_ratio` and `max_centroid_diff_hz` parameters

2. `src/playlist/candidate_pool.py` (~120 lines)
   - Extract BPM/centroid from X_sonic
   - Implement filtering logic
   - Add stats/logging

3. `src/playlist/constructor.py` (~5 lines)
   - Pass X_sonic to candidate pool builder

4. `config.yaml` (~2 lines)
   - Add filtering parameters

**Testing**:
- Unit tests for feature extraction
- Integration tests with existing candidate pool tests
- Manual testing on 10 diverse seed tracks
- Verify filtered candidates are appropriate

### Phase 2: Artifact Enhancement (Optional, during next rebuild)

**Approach**: Add BPM/centroid as separate arrays in artifact bundle

**Advantages**:
- ✅ Simpler filtering code (direct array access)
- ✅ More robust (no dependency on feature ordering)
- ✅ Supports future filters (rolloff, onset strength, etc.)

**When**: During next artifact rebuild (currently running beat-sync rebuild)

**Impact**: Negligible artifact size increase (~0.5%)

---

## Risk Assessment

### Low Risk Factors
✅ **Backward Compatible**: Filtering disabled by default (config = null)
✅ **Non-Breaking**: No changes to existing API or data structures
✅ **Isolated**: Filtering logic is independent of existing gates
✅ **Testable**: Easy to validate with manual testing
✅ **Configurable**: Users can adjust or disable in config

### Potential Concerns & Mitigation

| Concern | Mitigation |
|---------|-----------|
| Threshold tuning complexity | Provide 3 presets (conservative/balanced/liberal) with documentation |
| Smaller candidate pools | Thresholds chosen to exclude <20% of candidates; users can adjust |
| Edge cases (BPM=0) | Reject tracks with invalid BPM; fallback detection in scan_library |
| Different extraction methods | Support both beat-sync and windowed features automatically |

---

## Success Criteria

✅ BPM filtering removes obvious tempo mismatches
✅ Centroid filtering removes obvious brightness mismatches
✅ Filtering disabled by default (backward compatible)
✅ <2ms performance overhead
✅ Filtered candidates appear in stats with reasons
✅ Pool size reduction acceptable (10-20%)
✅ Manual listening confirms improved quality
✅ Unit and integration tests passing
✅ Configuration documented with examples

---

## Configuration Recommendations

### Default (Disabled)
```yaml
max_bpm_ratio: null
max_centroid_diff_hz: null
```
✅ No change to current behavior
✅ Safe for production

### Recommended (Balanced)
```yaml
max_bpm_ratio: 1.5
max_centroid_diff_hz: 3000
```
✅ Eliminates obvious clashes
✅ Maintains reasonable diversity
✅ Good balance of quality vs. pool size

### Tuning Guidelines
- Start with **balanced** config
- Listen to playlists from 5-10 diverse seed tracks
- If too restrictive (pools too small): increase ratios
- If not restrictive enough (still jarring): decrease ratios
- Log shows how many candidates filtered (helps calibration)

---

## Questions & Answers

**Q: Will this break existing playlists?**
A: No. Filtering is disabled by default. Only enabled if explicitly configured.

**Q: Why hard gates instead of soft penalties?**
A: BPM/centroid mismatches are immediately noticeable to listeners. Soft penalties might still include unsuitable candidates. Hard gates ensure quality.

**Q: What if a track has no BPM?**
A: Filtered out (invalid BPM = 0). This is expected behavior. Library scanner should detect all BPMs during initial scan.

**Q: Can I enable just BPM filtering?**
A: Yes. Set `max_centroid_diff_hz: null` (or omit). Each filter is independent.

**Q: How do threshold values translate to real playlists?**
A: See "Examples" section above. Test on seed tracks you know well to calibrate.

**Q: Does this conflict with genre or transition filtering?**
A: No. Sonic filters are applied sequentially:
1. Similarity floor
2. Genre gate
3. **BPM filter** (new)
4. **Centroid filter** (new)
5. Artist ranking

All gates must pass for a candidate to be included.

---

## Comparison: Before vs After

### Before (Current System)
```
Seed: "Warm Ballad" (60 BPM, 2000Hz, Acoustic)

Candidate Pool (top 5):
1. "Similar Ballad" (70 BPM, 2100Hz) ✅ Good match
2. "Pop Ballad" (65 BPM, 2300Hz) ✅ Good match
3. "Upbeat Pop" (180 BPM, 7500Hz) ⚠️ Jarring tempo/brightness
   - Genre: Pop ✅
   - Transition: 0.45 ✅ (passes floor)
   - BPM: 60→180 (3x ratio) ❌ Excluded with filtering
4. "Dark Electronic" (120 BPM, 3000Hz) ⚠️ Acceptable genre, but tempo jump
5. "Fast Rock" (160 BPM, 6000Hz) ⚠️ Extreme mismatch
```

### After (With Sonic Filtering)
```
Same seed, max_bpm_ratio=1.5, max_centroid_diff_hz=3000

Candidate Pool (top 5):
1. "Similar Ballad" (70 BPM, 2100Hz) ✅ Good match
2. "Pop Ballad" (65 BPM, 2300Hz) ✅ Good match
3. "Upbeat Pop" (180 BPM, 7500Hz) ❌ Filtered (tempo 3x, brightness +5500Hz)
4. "Dark Electronic" (120 BPM, 3000Hz) ❌ Filtered (tempo 2x > 1.5)
5. "Moderate Pop" (85 BPM, 2800Hz) ✅ Included (tempo 1.42, brightness 800Hz)

Result: More cohesive pool, no jarring transitions
```

---

## Next Steps

**To approve this proposal, please review**:
1. This document (SONIC_FILTERING_PROPOSAL.md)
2. Detailed plan (SONIC_FILTERING_IMPLEMENTATION_PLAN.md)
3. Configuration examples above

**If approved, implementation would proceed as**:
1. Implement Phase 1 (extract + filter) - 1-2 weeks
2. Test on diverse seed tracks - 1 week
3. Document configuration examples - 1 week
4. Deploy to production
5. (Optional) Enhance artifacts during next rebuild

**Decision needed**:
- [ ] Approve Phase 1 implementation
- [ ] Approve Phase 2 artifact enhancement (future)
- [ ] Request modifications to proposal
- [ ] Defer for later

---

## Appendix: Technical Details

### Where Data Comes From

**BPM**:
- Extracted from audio files during library scan
- Stored in database as `sonic_features` JSON
- Available in artifact as X_sonic feature vector
- Single numeric value (beats per minute)

**Spectral Centroid**:
- Computed from audio frequency spectrum
- Represents "center of mass" of frequencies
- Stored in database as `sonic_features` JSON
- Available in artifact as X_sonic feature vector
- Single numeric value (Hz)

### Why These Two Features

| Feature | Perceptibility | Jarring Factor | Filtering Value |
|---------|----------------|----------------|-----------------|
| **BPM** | ⭐⭐⭐⭐⭐ Highest | Immediate | ✅ Critical |
| **Spectral Centroid** | ⭐⭐⭐⭐ High | Very noticeable | ✅ Important |
| **MFCC (Timbre)** | ⭐⭐⭐ Medium | Gradual | ❌ Use for scoring, not filtering |
| **Chroma (Harmony)** | ⭐⭐ Low | Subtle | ❌ Use for scoring, not filtering |
| **Spectral Rolloff** | ⭐⭐ Low | Subtle | ❌ Future filtering option |

---

## References

- Full implementation details: `SONIC_FILTERING_IMPLEMENTATION_PLAN.md`
- Sonic features documentation: `docs/SONIC_FEATURES_REFERENCE.md`
- Feature glossary: `docs/FEATURE_GLOSSARY.md`
- Current candidate pool code: `src/playlist/candidate_pool.py`
- Audio analyzer: `src/librosa_analyzer.py`

---

**Proposal Status**: Ready for Review
**Created**: December 2024
**Author**: Claude Code Research Team
