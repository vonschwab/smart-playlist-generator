# Sonic Feature Filtering Implementation Plan

## Current State Analysis

### ✅ What We Have

1. **Hybrid Scoring in Candidate Pool Generation** (YES)
   - `candidate_pool.py` line 115-117: Uses hybrid embedding
   - Computes: `seed_sim_all = cosine_sim_matrix_to_vector(emb_norm, seed_vec)`
   - This is the hybrid (60% sonic + 40% genre) embedding
   - Used for fast candidate pool ranking

2. **Sonic Features Stored in Feature Vectors**
   - BPM: Embedded as single dimension in X_sonic vector
   - Spectral Centroid: Embedded as single dimension in X_sonic vector
   - Both have known slice positions in the layout: `layout['bpm']['slice']`, `layout['spectral_centroid']['slice']`

3. **Data in Artifact Bundle**
   - X_sonic: (N, D) feature matrix with all sonic dimensions concatenated
   - Sonic feature names & units (optional, available in bundle)
   - Everything needed to extract BPM and centroid

### ❌ What We're Missing

1. **Extracted BPM Array** - Not as separate (N,) array, only embedded in X_sonic
2. **Extracted Spectral Centroid Array** - Not as separate (N,) array, only embedded in X_sonic
3. **Sonic Feature Filtering** - No hard/soft gates based on BPM or centroid
4. **Filtering Parameters** - No config options for sonic thresholds

---

## Implementation Strategy

### Option A: Extract from X_sonic at Runtime (FASTER TO IMPLEMENT)

**Pros**:
- No changes to artifact builder or storage
- Works with existing artifacts
- Fast (vectorized numpy operations)
- Minimal code changes

**Cons**:
- Need to know feature positions (requires sonic_feature_names or hardcode)
- Slightly slower than pre-extracted arrays (negligible)
- Fragile if feature order changes

**Implementation Complexity**: LOW (~50 lines of code)

```python
def extract_bpm_from_sonic(X_sonic, sonic_feature_names=None):
    """Extract BPM column from X_sonic feature matrix"""
    if sonic_feature_names is not None:
        try:
            bpm_idx = list(sonic_feature_names).index('bpm')
            return X_sonic[:, bpm_idx]
        except ValueError:
            return None
    # Fallback: assume standard position (beat-sync: 26+24+14+2 = 66, windowed: different)
    # This is fragile - not recommended
    return None

def extract_spectral_centroid_from_sonic(X_sonic, sonic_feature_names=None):
    """Extract spectral_centroid_mean from X_sonic feature matrix"""
    if sonic_feature_names is not None:
        try:
            idx = list(sonic_feature_names).index('spectral_centroid_mean')
            return X_sonic[:, idx]
        except ValueError:
            return None
    return None
```

### Option B: Add as Separate Arrays in Artifact (BETTER LONG-TERM)

**Pros**:
- Direct access, no extraction needed
- Self-documenting (clear what data is)
- No dependency on feature ordering
- Easier for filtering operations
- Can be used for other features too (rhythm, harmony, etc.)

**Cons**:
- Need to rebuild artifacts
- Larger artifact files (~0.5% size increase)
- Changes artifact schema

**Implementation Complexity**: MEDIUM (~100 lines of code + artifact rebuild)

```python
# In artifact_builder.py, extract and save:
bpm_values = []
centroid_values = []
for row in rows:
    features = json.loads(row["sonic_features"])
    bpm_values.append(features.get('bpm', 0.0))
    centroid_values.append(features.get('spectral_centroid_mean', 0.0))

np.savez(out_path,
    ...,
    bpm_array=np.array(bpm_values, dtype=np.float32),
    spectral_centroid_array=np.array(centroid_values, dtype=np.float32)
)
```

---

## Recommended Approach: **HYBRID**

1. **Immediately** (Option A): Extract BPM/centroid from X_sonic at runtime
   - Implement in candidate_pool.py
   - Works with existing artifacts
   - Can ship within days
   - Validate that extraction works

2. **Next rebuild** (Option B): Add as separate arrays to artifact
   - During next `analyze_library.py` run (currently rebuilding with beat-sync)
   - Makes filtering more robust long-term
   - Can deprecate extraction logic once all new artifacts built

---

## Phase 1: Implement Sonic Filtering (Using Option A)

### Files to Modify

#### 1. `src/playlist/config.py`
Add sonic filtering parameters to CandidatePoolConfig:

```python
@dataclass
class CandidatePoolConfig:
    # ... existing fields ...

    # NEW: Sonic feature filtering (optional)
    max_bpm_ratio: Optional[float] = None  # e.g., 1.5 = allow 1.5x tempo variation
    max_centroid_diff_hz: Optional[float] = None  # e.g., 3000 = max 3000Hz brightness diff
    sonic_feature_names: Optional[np.ndarray] = None  # For extracting features from X_sonic
```

#### 2. `src/playlist/candidate_pool.py`
Add sonic filtering logic:

```python
def _extract_sonic_feature(X_sonic, feature_name, sonic_feature_names):
    """Extract a single sonic feature from X_sonic matrix"""
    if sonic_feature_names is None or X_sonic is None:
        return None
    try:
        feature_idx = list(sonic_feature_names).index(feature_name)
        return X_sonic[:, feature_idx]
    except (ValueError, IndexError):
        logger.warning(f"Could not extract {feature_name} from sonic features")
        return None

def build_candidate_pool(
    *,
    seed_idx: int,
    embedding: np.ndarray,
    artist_keys: np.ndarray,
    cfg: CandidatePoolConfig,
    random_seed: int,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    min_genre_similarity: Optional[float] = None,
    genre_method: str = "ensemble",
    mode: str = "dynamic",
    X_sonic: Optional[np.ndarray] = None,  # NEW: For feature extraction
) -> CandidatePoolResult:
    """
    ... existing docstring ...

    NEW: Sonic feature filtering
    - If max_bpm_ratio set, filter by BPM compatibility
    - If max_centroid_diff_hz set, filter by spectral brightness
    """

    # ... existing code ...

    # NEW: Extract sonic features for filtering
    below_bpm_count = 0
    below_centroid_count = 0

    bpm_array = None
    centroid_array = None
    seed_bpm = None
    seed_centroid = None

    if X_sonic is not None and cfg.sonic_feature_names is not None:
        bpm_array = _extract_sonic_feature(X_sonic, 'bpm', cfg.sonic_feature_names)
        centroid_array = _extract_sonic_feature(X_sonic, 'spectral_centroid_mean', cfg.sonic_feature_names)

        if bpm_array is not None and bpm_array[seed_idx] > 0:
            seed_bpm = bpm_array[seed_idx]
        if centroid_array is not None:
            seed_centroid = centroid_array[seed_idx]

    # ... existing similarity floor filtering ...

    # NEW: Apply sonic feature filters
    if cfg.max_bpm_ratio is not None and bpm_array is not None and seed_bpm is not None:
        eligible_before_bpm = len(eligible)
        eligible = [
            i for i in eligible
            if bpm_array[i] > 0 and  # Must have valid BPM
               max(seed_bpm, bpm_array[i]) / min(seed_bpm, bpm_array[i]) <= cfg.max_bpm_ratio
        ]
        below_bpm_count = eligible_before_bpm - len(eligible)
        logger.info(
            "BPM filtering applied: max_ratio=%.2f, seed_bpm=%.1f, excluded=%d",
            cfg.max_bpm_ratio, seed_bpm, below_bpm_count
        )

    if cfg.max_centroid_diff_hz is not None and centroid_array is not None and seed_centroid is not None:
        eligible_before_centroid = len(eligible)
        eligible = [
            i for i in eligible
            if abs(centroid_array[i] - seed_centroid) <= cfg.max_centroid_diff_hz
        ]
        below_centroid_count = eligible_before_centroid - len(eligible)
        logger.info(
            "Spectral centroid filtering applied: max_diff=%.0f Hz, seed_centroid=%.0f Hz, excluded=%d",
            cfg.max_centroid_diff_hz, seed_centroid, below_centroid_count
        )

    # ... rest of existing code ...

    # Add to stats
    stats = {
        # ... existing stats ...
        "below_bpm_ratio": below_bpm_count,
        "below_spectral_centroid": below_centroid_count,
    }

    params_effective = {
        # ... existing params ...
    }
    if cfg.max_bpm_ratio is not None:
        params_effective["max_bpm_ratio"] = cfg.max_bpm_ratio
    if cfg.max_centroid_diff_hz is not None:
        params_effective["max_centroid_diff_hz"] = cfg.max_centroid_diff_hz
```

#### 3. `src/playlist/constructor.py`
Update constructor to pass sonic features:

```python
def construct_playlist(
    artifact: ArtifactBundle,
    seed_track_id: str,
    length: int,
    cfg: DSPipelineConfig,
    ...
) -> PlaylistResult:
    # ... existing code ...

    # When calling build_candidate_pool:
    pool_result = build_candidate_pool(
        seed_idx=seed_idx,
        embedding=hybrid_embedding,
        artist_keys=artifact.artist_keys,
        cfg=cfg.candidate_pool,
        random_seed=random_seed,
        X_genre_raw=artifact.X_genre_raw,
        X_genre_smoothed=artifact.X_genre_smoothed,
        min_genre_similarity=min_genre_similarity,
        genre_method=cfg.genre_method,
        mode=mode,
        X_sonic=artifact.X_sonic,  # NEW: Pass sonic features
    )
```

#### 4. Update config files to accept sonic parameters

```yaml
# config.yaml example
candidate_pool:
  max_pool_size: 500
  target_artists: 20
  candidates_per_artist: 5
  similarity_floor: 0.30
  # NEW: Sonic filtering
  max_bpm_ratio: 1.5  # Allow up to 1.5x tempo variation
  max_centroid_diff_hz: 3000  # Allow up to 3000Hz brightness difference
```

---

## Phase 2: Validate Sonic Filtering (Testing)

### Test Cases

1. **BPM Filtering**
   - Seed: 100 BPM → Candidate 150 BPM (ratio 1.5, included ✓)
   - Seed: 100 BPM → Candidate 160 BPM (ratio 1.6, excluded ✗)
   - Seed: 60 BPM → Candidate 90 BPM (ratio 1.5, included ✓)

2. **Spectral Centroid Filtering**
   - Seed: 4000 Hz → Candidate 6500 Hz (diff 2500 Hz, included ✓)
   - Seed: 4000 Hz → Candidate 7500 Hz (diff 3500 Hz, excluded ✗)
   - Seed: 2000 Hz → Candidate 4500 Hz (diff 2500 Hz, included ✓)

3. **Combined Filtering**
   - Both BPM and centroid filters applied simultaneously
   - Track passes only if both conditions met

4. **Edge Cases**
   - Missing BPM (0 value): excluded
   - Missing centroid (0 value): excluded
   - Disabled filters (None): no filtering applied

### Test Script

```bash
python scripts/test_sonic_filtering.py \
    --artifact data_matrices.npz \
    --seed-track-id <track_id> \
    --max-bpm-ratio 1.5 \
    --max-centroid-diff 3000 \
    --output diagnostics/sonic_filtering_test.json
```

Output should show:
- Pool size before/after BPM filtering
- Pool size before/after centroid filtering
- Examples of filtered-out tracks with reasons
- Verification that remaining pool still valid

---

## Phase 3: Enhanced Artifact (Option B - Later)

When rebuilding artifacts during next sonic rebuild:

```python
# In artifact_builder.py

# Extract sonic features for easier filtering
bpm_values = []
centroid_values = []
rolloff_values = []  # For future filtering

for row in rows:
    features = json.loads(row["sonic_features"])
    bpm_values.append(features.get('bpm', 0.0))
    centroid_values.append(features.get('spectral_centroid_mean', 0.0))
    rolloff_values.append(features.get('spectral_rolloff_mean', 0.0))

# Save in artifact
np.savez(
    out_path,
    # ... existing arrays ...
    bpm_array=np.array(bpm_values, dtype=np.float32),
    spectral_centroid_array=np.array(centroid_values, dtype=np.float32),
    spectral_rolloff_array=np.array(rolloff_values, dtype=np.float32),
)

# Update ArtifactBundle to load these
@dataclass(frozen=True)
class ArtifactBundle:
    # ... existing fields ...
    bpm_array: Optional[np.ndarray] = None  # (N,)
    spectral_centroid_array: Optional[np.ndarray] = None  # (N,)
    spectral_rolloff_array: Optional[np.ndarray] = None  # (N,)
```

---

## Configuration Examples

### Conservative (Tight Filtering)
```yaml
# Very similar songs only
max_bpm_ratio: 1.2  # 100 BPM can go 80-120 BPM
max_centroid_diff_hz: 2000  # Very limited brightness change
```

### Balanced (Current Recommendation)
```yaml
# Good flow, reasonable diversity
max_bpm_ratio: 1.5  # 100 BPM can go 67-150 BPM
max_centroid_diff_hz: 3000  # Good brightness range
```

### Liberal (Loose Filtering)
```yaml
# Maximum diversity while avoiding jarring transitions
max_bpm_ratio: 2.0  # 100 BPM can go 50-200 BPM
max_centroid_diff_hz: 5000  # Very broad brightness range
```

### Disabled (Current Behavior)
```yaml
# No sonic feature filtering (only genre + transition quality)
max_bpm_ratio: null
max_centroid_diff_hz: null
```

---

## Impact Analysis

### Performance Impact

**BPM Filtering**: O(N) array comparison
- Cost: ~1ms for 34k tracks
- Negligible

**Centroid Filtering**: O(N) array comparison
- Cost: ~1ms for 34k tracks
- Negligible

**Total**: Adding sonic filtering adds <2ms to candidate pool generation
- Current pool build: ~100-200ms
- New total: ~102-202ms (no perceptible difference)

### Filtering Impact (Typical Numbers)

Expected exclusions with balanced config (1.5 BPM ratio, 3000Hz centroid):

```
Starting pool: 500 candidates
After similarity floor: 450 candidates
After genre gate: 350 candidates
After BPM filtering: 300 candidates (50 excluded)
After centroid filtering: 250 candidates (50 excluded)
After artist caps: 150 candidates (used for playlist)

Exclusion rate: 50% (conservative, depends on seed track)
```

---

## Rollout Plan

### Week 1: Implementation (Phase 1)
- [ ] Modify CandidatePoolConfig to add sonic parameters
- [ ] Implement feature extraction in candidate_pool.py
- [ ] Add filtering logic (BPM + centroid)
- [ ] Add comprehensive logging
- [ ] Unit tests for extraction and filtering
- [ ] Integration tests with existing candidate pool tests

### Week 2: Validation (Phase 2)
- [ ] Create sonic_filtering_test.py script
- [ ] Test on 10 diverse seed tracks
- [ ] Verify filtered candidates make sense musically
- [ ] Measure pool size impact
- [ ] Document tuning recommendations
- [ ] Add to CI/CD (test suite)

### Week 3: Documentation
- [ ] Document filtering parameters in config.md
- [ ] Add examples to tuning_workflow.md
- [ ] Create sonic_filtering_best_practices.md
- [ ] Update API documentation if needed

### Future (Optional): Phase 3 Artifact Enhancement
- [ ] Wait for next artifact rebuild (beat-sync currently running)
- [ ] Extract BPM/centroid/rolloff arrays
- [ ] Update artifact schema
- [ ] Deprecate feature extraction logic (keep for backward compat)

---

## Key Decisions

✅ **Decision 1: Use Option A (extract from X_sonic) for Phase 1**
- Rationale: Quick to implement, works with existing artifacts, can validate approach
- Timeline: 1-2 weeks

✅ **Decision 2: Optional filtering (config-based)**
- Rationale: Users can opt-in; doesn't break existing behavior
- Default: None (disabled)
- Users enable in config.yaml if desired

✅ **Decision 3: Hard filtering (exclude, not penalize)**
- Rationale: BPM/centroid mismatches are jarring; soft penalty still includes bad candidates
- Effect: Some candidates filtered out, but higher quality pool

---

## FAQ

**Q: Will this break existing playlists?**
A: No. Filtering is disabled by default (config = None). Only enabled if explicitly set.

**Q: What if a track has no BPM (0 value)?**
A: Excluded automatically (must have valid BPM > 0). Fallback detection should catch these in scan_library.

**Q: Can I use just BPM filtering without centroid?**
A: Yes. Set max_centroid_diff_hz = null (or omit) in config.

**Q: How do I tune the thresholds?**
A: Run test script on seed tracks, listen to playlists, adjust thresholds. Start with balanced (1.5, 3000).

**Q: Does this conflict with genre filtering?**
A: No. Sonic filtering is independent. Both apply in sequence:
1. Similarity floor
2. Genre gate
3. BPM filter
4. Centroid filter
5. Artist caps

---

## Success Criteria

✅ BPM filtering removes obvious tempo clashes
✅ Centroid filtering removes obvious brightness clashes
✅ No filtering is applied if config values are None
✅ Filtered-out tracks appear in stats with reasons
✅ Pool quality improves (fewer jarring transitions)
✅ Performance unchanged (< 2ms added)
✅ Fully backward compatible (no breaking changes)
✅ Well documented with tuning examples

---

## Summary

**To answer your questions**:

1. **"Would we be able to build out the sonic filtering features?"**
   - ✅ YES, absolutely feasible
   - Hybrid approach: Extract from X_sonic now, add as separate arrays later
   - ~150 lines of code for Phase 1

2. **"Are we still calculating hybrid genre and sonic scores for fast candidate pool generation?"**
   - ✅ YES, confirmed in candidate_pool.py line 115-117
   - Uses hybrid embedding (60% sonic + 40% genre)
   - Sonic filtering would add additional gates AFTER hybrid scoring

**Next Steps**:
1. Approve approach (Option A for Phase 1, then Phase 2 artifact enhancement)
2. Estimate timeline (1-2 weeks for Phase 1 + testing)
3. Decide on threshold defaults (recommend balanced: 1.5 BPM ratio, 3000Hz centroid)
4. Start implementation
