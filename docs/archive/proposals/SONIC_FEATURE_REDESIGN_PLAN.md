# Sonic Feature Redesign: Engineering Plan

**Date**: 2025-12-16
**Goal**: Improve sonic_flatness from 0.037 to ≥0.5, topK_gap from 0.014 to ≥0.15
**Approach**: Beat-synchronized features + proper dimension scaling
**Effort**: ~4-6 hours design + implementation + validation

---

## Problem Statement

Current 27-dimensional sonic vectors exhibit catastrophic scale imbalance:

- **BPM dimension**: scale ~1800 (variance 377,251)
- **MFCC dimensions**: scale ~50-100 (variance ~1,000-11,000)
- **Chroma dimensions**: scale ~0.35 (variance ~0.012)

After L2 normalization, BPM dominates vector direction, causing:
- All vectors nearly identical in normalized space (cos_sim ≈ 0.9998)
- TopK gap: 0.0075-0.0124 (need ≥0.15)
- Sonic flatness: 0.037 (need ≥0.5)

**Root Cause**: Features concatenated at different scales without per-feature normalization.

---

## Solution: Two-Pronged Approach

### Phase 1: Fix Scale Imbalance (Quick Win)
Apply per-feature z-score normalization BEFORE concatenation, not after.

**Expected improvement**: 2-3x flatness increase (0.037 → ~0.08-0.12)

### Phase 2: Beat-Synchronized + Higher Dimensionality (Real Fix)
Replace fixed-window features with beat-synchronized extraction + robust statistics.

**Expected improvement**: 10-20x flatness increase (0.12 → 0.5+)

---

## Implementation Detail: Phase 1 (Per-Feature Scaling)

### Current (Broken) Flow:
```
Extract MFCC (scale ~50-100)
Extract BPM (scale ~1800)
Extract Chroma (scale ~0.35)
Concatenate → [MFCC, BPM, Chroma]
L2-normalize → All dominated by BPM
```

### Fixed Flow:
```
Extract MFCC → Z-score normalize (mean 0, std 1)
Extract BPM → Z-score normalize (mean 0, std 1)
Extract Chroma → Z-score normalize (mean 0, std 1)
Concatenate → [norm_MFCC, norm_BPM, norm_Chroma]
L2-normalize → Balanced contributions
```

### Changes Required:
- **File**: `src/librosa_analyzer.py`
- **Method**: `_extract_features_from_audio()`
- **Add**: Per-feature z-score normalization before returning dict
- **Impact**: 2-3 lines per feature group

### Code Example:
```python
def _extract_features_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    features = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    # NEW: Z-score normalize
    mfcc_mean_norm = (mfcc_mean - np.mean(mfcc_mean)) / (np.std(mfcc_mean) + 1e-12)
    features['mfcc_mean'] = mfcc_mean_norm.tolist()

    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # NEW: Z-score normalize globally across library (store mean/std in artifact metadata)
    features['bpm'] = float(tempo)
    features['bpm_is_raw'] = True  # Mark for downstream z-scoring

    # ... similar for chroma, spectral features
```

---

## Implementation Detail: Phase 2 (Beat-Sync + Higher Dim)

### Current Feature Set (27 dims):
```
MFCC:        13 dims (mean only)
Spectral:    10 dims (centroid, rolloff, bandwidth, contrast×7)
Chroma:      12 dims (mean only)
BPM:          1 dim (scalar)
────────────────────────
Total:       27 dims
```

### Proposed Feature Set (60-80 dims):

#### Per-Beat Features (new):
```
MFCC:           26 dims (mean + std)
Spectral:       20 dims (centroid mean+std, rolloff mean+std, contrast×7 mean+std)
Chroma:         24 dims (mean + std, per 12 pitch classes)
Rhythm:          8 dims (BPM, onset_rate, onset_strength mean+std, tempogram entropy)
────────────────────────
Subtotal:       78 dims
```

#### Aggregation Strategy:
```
For each song segment (start/mid/end):
  1. Detect beats using librosa.beat.beat_track()
  2. Extract features per beat
  3. Aggregate using robust statistics:
     - Per-beat features → median across beats
     - Variability → IQR (Q75 - Q25) or std
     - Results: mean + std (or median + IQR)
```

#### Beat-Sync Extraction Pseudocode:
```python
def extract_beat_sync_features(y, sr):
    """Extract features per beat, aggregate robustly."""

    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Extract features per beat
    mfcc_per_beat = []
    chroma_per_beat = []
    for i in range(len(beat_frames) - 1):
        start_frame = beat_frames[i]
        end_frame = beat_frames[i+1]
        segment = y[start_frame:end_frame]

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_per_beat.append(mfcc.mean(axis=1))

        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        chroma_per_beat.append(chroma.mean(axis=1))

    # Aggregate robustly
    mfcc_per_beat = np.array(mfcc_per_beat)  # shape (num_beats, 13)
    mfcc_median = np.median(mfcc_per_beat, axis=0)
    mfcc_iqr = np.percentile(mfcc_per_beat, 75, axis=0) - np.percentile(mfcc_per_beat, 25, axis=0)

    chroma_per_beat = np.array(chroma_per_beat)  # shape (num_beats, 12)
    chroma_median = np.median(chroma_per_beat, axis=0)
    chroma_iqr = np.percentile(chroma_per_beat, 75, axis=0) - np.percentile(chroma_per_beat, 25, axis=0)

    # Combine: mean + std (or median + IQR)
    return np.concatenate([
        mfcc_median, mfcc_iqr,  # 26 dims
        chroma_median, chroma_iqr,  # 24 dims
        [tempo, onset_rate, onset_strength_mean, onset_strength_std],  # 4 dims
        # ... more rhythm features
    ])
```

### Fallback for Beat-Tracking Failures:
```python
def extract_beat_sync_features_with_fallback(y, sr):
    """Try beat-sync; fall back to windowed if beats fail."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if len(beat_frames) < 3:  # Not enough beats detected
            raise ValueError("Insufficient beats detected")
        # ... beat-sync extraction
        return beat_sync_vector, True  # success flag
    except Exception as e:
        logger.warning(f"Beat tracking failed: {e}, using windowed fallback")
        # Fall back to legacy 3-segment (start/mid/end) extraction
        return windowed_vector, False
```

### Multi-Segment Preservation:
```
For artifact building:
  X_sonic[i] = beat_sync_features(full_track)
  X_sonic_start[i] = beat_sync_features(segment_start)
  X_sonic_mid[i] = beat_sync_features(segment_mid)
  X_sonic_end[i] = beat_sync_features(segment_end)

This preserves ability to score transitions: end(prev) → start(next)
```

---

## Configuration & Flags

### New Feature Variant: "beat_sync"

**File**: `src/similarity/sonic_variant.py`

```python
_ALLOWED = {
    "raw",                  # Current
    "z_clip",              # Current (production)
    "whiten_pca",          # Current
    "robust_whiten",       # Phase B
    "beat_sync",           # NEW: Beat-synchronized features
}
```

**Usage**:
```bash
# Use new beat-sync features
export SONIC_SIM_VARIANT=beat_sync
python scripts/analyze_library.py --stages artifacts

# Or in code:
from src.similarity.sonic_variant import resolve_sonic_variant
variant = resolve_sonic_variant(explicit_variant="beat_sync")
```

### Config Entry (config.yaml):
```yaml
sonic:
  extraction_method: "beat_sync"  # Options: "windowed", "beat_sync"
  feature_dimensionality: "extended"  # Options: "compact" (27), "extended" (60+)
  beat_track_fallback: true  # Fall back to windowed if beat tracking fails
  per_feature_normalize: true  # Normalize each feature group before concatenation
```

---

## Implementation Checklist

### Phase 1: Per-Feature Scaling (1-2 hours)
- [ ] Modify `src/librosa_analyzer.py`:
  - [ ] Add per-feature z-score normalization in `_extract_features_from_audio()`
  - [ ] Add global z-score normalization in `_average_features()` (for multi-segment aggregate)
  - [ ] Test on 100 tracks
- [ ] Rebuild artifact with Phase 1 changes
- [ ] Run validation suite on 12 seeds
- [ ] Expected: flatness 0.08-0.15, topK gap 0.02-0.05

### Phase 2: Beat-Sync + Higher Dim (3-4 hours)
- [ ] Modify `src/librosa_analyzer.py`:
  - [ ] Add `extract_beat_sync_features()` function
  - [ ] Add fallback logic for beat tracking failures
  - [ ] Integrate into main extraction flow with config flag
  - [ ] Update `_average_features()` for new dimensionality
- [ ] Update `src/similarity/sonic_variant.py`:
  - [ ] Add "beat_sync" to _ALLOWED
- [ ] Update config.yaml:
  - [ ] Add sonic extraction method options
- [ ] Rebuild artifact with Phase 2 changes
- [ ] Run validation suite on 12 seeds
- [ ] Expected: flatness ≥0.5, topK gap ≥0.15, 7+/12 seeds pass

### Phase 3: Testing & Reporting (1-2 hours)
- [ ] Run full validation suite before/after
- [ ] Generate comparison tables
- [ ] Document which seeds fail & hypotheses (beat track failures, sparse audio, etc.)
- [ ] Test backward compatibility (old artifacts still load)
- [ ] Create PR with:
  - [ ] Code changes
  - [ ] Unit tests for new functions
  - [ ] Updated documentation
  - [ ] Windows command checklist

---

## Backward Compatibility

### Concerns:
1. Artifact file format unchanged (same number of fields)
2. Dimensionality change: artifacts will have different dims
3. Sonic variant flag selection

### Solution:
```
Old artifacts: X_sonic shape (N, 27) with variant flag
New artifacts: X_sonic shape (N, 60+) with variant flag

Pipeline detects shape at load time:
  if X_sonic.shape[1] == 27:
      use_legacy_normalization()
  else:
      use_new_normalization()
```

### Safe Changes:
- Keep raw/z_clip/whiten_pca variants working (no breaking changes to those)
- Add beat_sync as opt-in via config/CLI
- Old playlists still work (they reference track_ids, not specific feature versions)

---

## Success Criteria

### Acceptance Thresholds:
- ✓ sonic_flatness ≥ 0.5 (currently 0.037)
- ✓ topK_gap ≥ 0.15 (currently 0.014)
- ✓ 7+/12 seeds pass all 4 metrics
- ✓ No regression on genre or overall quality
- ✓ Backward compatible (old artifacts still load)

### Test Results Expected:
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Flatness (median) | 0.037 | ≥0.5 | 13x improvement |
| TopK gap (mean) | 0.014 | ≥0.15 | 10x improvement |
| Seeds passing | 0/12 | 7+/12 | 60%+ pass |

---

## Risk Mitigation

### Risk 1: Beat Tracking Failure
**Mitigation**: Fallback to windowed extraction; track failure rate in logs
**Flag**: `beat_track_fallback: true` in config

### Risk 2: Dimensionality Change Breaks Pipeline
**Mitigation**: Detect shape at load time; support both old and new
**Test**: Load old artifacts with new code

### Risk 3: Performance Regression
**Mitigation**: Benchmark extract/normalize time; optimize if needed
**Target**: <10% slower than current

### Risk 4: Improvement Still Insufficient
**Plan B**: Pre-trained embeddings (OpenL3 as plugin)
**Fallback**: Genre-only mode with strong gates

---

## Option B: Pre-Trained Embeddings (If Beat-Sync Insufficient)

### Plugin Architecture:
```python
class SonicFeatureExtractor:
    def extract(self, y, sr) -> np.ndarray:
        if self.method == "beat_sync":
            return self._beat_sync(y, sr)
        elif self.method == "openl3":
            return self._openl3(y, sr)
        else:
            return self._windowed(y, sr)  # legacy

class OpenL3Extractor:
    def _openl3(self, y, sr):
        import openl3
        model = openl3.models.load_embedding_model('music')
        emb, _ = openl3.get_embedding(y, sr, model=model)
        return emb  # (512,) or (2048,) dims
```

### Tradeoffs:
| Aspect | Beat-Sync | OpenL3 |
|--------|-----------|--------|
| Dimensionality | 60-80 | 512-2048 |
| Install complexity | None (librosa) | New dependency |
| Runtime | ~100ms/track | ~200-500ms/track |
| Model size | None | ~50MB |
| Caching | Simple (in memory) | Need file cache |
| Expected improvement | 10-20x | 20-50x |

**Decision**: Implement beat-sync first; propose OpenL3 as Phase 3 if needed.

---

## Command Reference (Windows)

```bash
# 1. Run diagnostic on current artifact
python scripts/diagnose_sonic_vectors.py ^
    --artifact ./experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz

# 2. Rebuild artifact with Phase 1 (per-feature scaling)
set SONIC_SIM_VARIANT=z_clip
python scripts/analyze_library.py --stages artifacts --force

# 3. Run validation on 12 seeds
python scripts/sonic_validation_suite.py ^
    --artifact ./experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz ^
    --seed-track-id 89f67bd19c13c2c481bd583206ce44d9 ^
    --k 30 ^
    --output-dir diagnostics/sonic_validation_phase1/

# 4. Rebuild with beat-sync features
set SONIC_SIM_VARIANT=beat_sync
python scripts/analyze_library.py --stages artifacts --force

# 5. Validate phase 2
python scripts/sonic_validation_suite.py ^
    --artifact ./experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz ^
    --seed-track-id 89f67bd19c13c2c481bd583206ce44d9 ^
    --k 30 ^
    --output-dir diagnostics/sonic_validation_phase2/
```

---

## Files to Modify/Create

### Modify:
- `src/librosa_analyzer.py` - Add beat-sync extraction
- `src/similarity/sonic_variant.py` - Add "beat_sync" variant

### Create:
- `tests/test_beat_sync_features.py` - Unit tests
- `BEAT_SYNC_IMPLEMENTATION.md` - Detailed notes

### Update:
- `config.yaml` - Add extraction method options
- This document → implementation PR description

---

## Success Story (Target)

**Before**:
- Sonic similarity flat (flatness 0.037)
- TopK barely beats random (gap 0.014)
- Genre dominates hybrid mode (40% genre weight controls playlist)
- Strange neighbors: indie/slowcore for afrobeat seeds

**After**:
- Sonic similarity informative (flatness ≥0.5)
- TopK strongly beats random (gap ≥0.15)
- Sonic & genre balanced (60% sonic provides signal)
- Good neighbors: consistent sonic character + same vibe

---

**Next**: Implementation PR with code, tests, and validation results.

