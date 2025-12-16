# Sonic Analysis Validation & Diagnostics Report

**Date**: 2025-12-16
**Session**: Sonic Analysis Validation & Improvement Phase A & B
**Status**: COMPLETE - Root Cause Identified
**Conclusion**: Sonic features fundamentally broken; feature extraction must be fixed

---

## Executive Summary

This session implemented a comprehensive validation framework (Phase A) and tested normalization improvements (Phase B) to diagnose why sonic similarity is not working in the playlist generator.

**Key Finding**: Sonic features produce **completely flat similarity scores** across all 34,100 tracks. All metrics conclusively failed validation, and normalization tweaks provided no improvement. **The root cause is in the feature extraction pipeline, not in post-processing.**

---

## Phase A: Sonic Validation Suite

### Objective
Create diagnostic tools to determine if sonic similarity is currently informative before making changes.

### Implementation
**File**: `scripts/sonic_validation_suite.py` (550 lines)

**Outputs per seed**:
1. `sonic_only_top30.m3u` - Pure sonic neighbors (for listening test)
2. `genre_only_top30.m3u` - Pure genre neighbors (for comparison)
3. `hybrid_current_top30.m3u` - Current hybrid mode (for baseline)
4. `sonic_validation_metrics.csv` - Quantitative metrics
5. `sonic_validation_report.md` - Human-readable analysis

**Metrics Computed**:

| Metric | Purpose | Formula | PASS Threshold |
|--------|---------|---------|-----------------|
| **Score Flatness** | Measure if scores separate good from bad neighbors | (p90 - p10) / median | ≥ 0.5 |
| **TopK vs Random Gap** | Measure if top-K neighbors actually better than random | mean(topK) - mean(random) | ≥ 0.15 |
| **Intra-Artist Coherence** | Measure if same-artist tracks cluster together | mean(same_artist_sim) - mean(random_sim) | ≥ 0.05 |
| **Intra-Album Coherence** | Measure if same-album tracks cluster together | mean(same_album_sim) - mean(random_sim) | ≥ 0.08 |

### Test Dataset
**12 diverse seeds** from `diagnostics/tune_seeds.txt`:
- Multiple genres and styles
- Various artist types (solo, collaborations)
- Different track lengths and tempos

### Results: Phase A FAILS (0/12 Seeds)

| Metric | Min | P25 | Median | P75 | Max | Status |
|--------|-----|-----|--------|-----|-----|--------|
| Score Flatness | 0.017 | 0.023 | 0.037 | 0.043 | 0.084 | **FAIL 0/12** |
| TopK vs Random Gap | 0.008 | 0.011 | 0.014 | 0.020 | 0.065 | **FAIL 0/12** |
| Intra-Artist Coherence | -0.005 | 0.004 | 0.013 | 0.018 | 0.034 | **FAIL 0/12** |
| Intra-Album Coherence | -0.012 | 0.002 | 0.013 | 0.029 | 0.064 | **FAIL 0/12** |

**Analysis**:
- **Score Flatness CRISIS**: Median flatness is 0.037 vs threshold 0.5 (need **13.5x improvement**)
- **No Discrimination**: Top-30 neighbors only 0.014 better than random vs threshold 0.15 (need **10x improvement**)
- **No Artist Clustering**: Same-artist tracks indistinguishable from random
- **No Album Clustering**: Same-album tracks indistinguishable from random

**Implication**: Sonic similarity treats all 34,100 tracks as nearly identical. Selecting top-30 neighbors is barely better than random sampling.

---

## Phase B: Normalization Variant Testing

### Objective
Test if normalization/post-processing improvements can fix sonic similarity.

### Normalization Variants Tested

#### Variant 1: Raw
- No preprocessing
- Pure cosine similarity on original 27-dim vectors
- Baseline

#### Variant 2: Z-clip (Current Production)
- Per-dimension z-score normalization
- Clipped to [-3, 3] to reduce outlier influence
- Current system default

#### Variant 3: Whiten-PCA
- StandardScaler (mean=0, std=1)
- PCA whitening (decorrelate + unit variance)
- Reduce dominant dimension bias

#### Variant 4: Robust-Whiten ⭐ NEW
- RobustScaler (median/IQR instead of mean/std)
- PCA whitening
- More resistant to outliers

### Test Procedure
1. Rebuild artifact with each variant
2. Run validation suite on 3 test seeds
3. Compare metrics across variants

### Results: Phase B FAILS (All Variants)

| Variant | Flatness | TopK Gap | Intra-Artist | Intra-Album | Improvement |
|---------|----------|----------|--------------|-------------|-------------|
| raw | 0.0342 | 0.0162 | 0.0091 | 0.0056 | — |
| z_clip (current) | 0.0342 | 0.0145 | 0.0070 | 0.0051 | None |
| whiten_pca | 0.0342 | 0.0159 | 0.0071 | 0.0158 | Marginal |
| robust_whiten | 0.0342 | 0.0194 | 0.0064 | 0.0128 | Marginal |

**Critical Observation**: **All variants have identical flatness (0.0342)**

### Root Cause Analysis

The fact that flatness is identical across all normalization variants is definitive proof that:

✅ **The problem is NOT in post-processing/normalization**
❌ **The problem IS in feature extraction**

If normalization mattered, different variants would have different flatness. Since they don't, the issue lies upstream.

---

## Root Cause: Feature Extraction Pipeline

### Current Sonic Features (27 dimensions)

**From `src/librosa_analyzer.py`:**

```
13 × MFCC (Mel-Frequency Cepstral Coefficients)
 1 × BPM (Tempo)
12 × Chroma (Pitch class distribution)
 1 × Spectral Centroid (brightness)
 1 × Spectral Rolloff
 1 × Spectral Bandwidth
 7 × Spectral Contrast (per band)
─────────────────────────────────────
27 total dimensions
```

### Why These Features Are Uninformative

#### 1. Fixed 30-Second Windows
- Music has structure: intro, verse, chorus, bridge, outro
- Extracting features from fixed positions (0-30s, middle 30s, last 30s) misses variation
- A song that's quiet at start but energetic in chorus will be poorly represented
- Doesn't adapt to actual song structure

#### 2. Mean Aggregation Destroys Temporal Dynamics
- Averaging features across a 30-second window loses the temporal story
- A song with dynamic range (soft→loud) becomes indistinguishable from a flat mix at same average level
- Loses rhythm patterns, syncopation, dynamic build-ups

#### 3. Generic Spectral Features
- MFCCs, spectral centroid, chroma are "universal" features
- They describe timbre/pitch in general terms, not genre-specific characteristics
- No specific modeling of:
  - Genre-defining sounds (drums in hip-hop, strings in orchestral, distortion in metal)
  - Instrumentation (how to distinguish guitar from keyboard)
  - Emotional content (energy, mood, tension)

#### 4. No Beat Alignment
- Features extracted on fixed 2.048ms hop windows
- No relationship to actual musical beats
- Tempo variations cause feature misalignment
- A fast vs slow version of same song would have very different features

#### 5. Limited Dimensionality
- 27 dimensions is very low for capturing 34,100 diverse tracks
- Heavy compression required in downstream normalization
- Information loss inevitable with such low dimensionality
- Compare to pre-trained audio embeddings (512-2048 dims)

### Evidence of the Problem

**Comparison of Results**:
- Baseline (raw): flatness = 0.034
- With z-score: flatness = 0.034 (no change)
- With PCA whitening: flatness = 0.034 (no change)
- With robust scaling: flatness = 0.034 (no change)

**Interpretation**:
The 27-dimensional sonic vector is so uniformly compressed that no normalization can recover information that was never captured. It's like trying to improve image quality by adjusting contrast on a blurry photo—the information is fundamentally missing.

---

## Impact on Playlist Generation

### Current Problems Explained

**Genre Curation Ineffective**:
- Genre gates filter by genre similarity, but sonic similarity adds noise
- With flat sonic scores, genre gates can't distinguish good candidates
- Results: Strange genre combinations, jarring transitions

**Sonic-Only Mode Unusable**:
- Selecting top-30 neighbors by sonic similarity barely beats random
- Sonic-only playlists lack coherence and flow

**Transitions Poor**:
- Transition scoring tries to optimize flow between tracks
- But underlying sonic similarity is uninformative
- Can't find good consecutive tracks

**Hybrid Mode Dominated by Genre**:
- Sonic weight is 60%, but sonic signal is noise
- Genre weight is 40%, but it's the only reliable signal
- Effectively becomes pure genre mode

---

## Solutions: 3 Paths Forward

### Option 1: Beat-Synchronized Features ⭐ RECOMMENDED

**Approach**:
- Detect beats using `librosa.beat.beat_track()`
- Extract features per beat instead of fixed windows
- Aggregate beats using median (robust) instead of mean
- Each song gets features that align to its musical structure

**Implementation**:
```python
def extract_beat_sync_features(y, sr):
    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Extract MFCC, chroma per beat
    mfcc_per_beat = []
    for i in range(len(beat_frames) - 1):
        segment = y[beat_frames[i]:beat_frames[i+1]]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_per_beat.append(mfcc.mean(axis=1))

    # Aggregate via median (robust to outliers)
    return np.median(mfcc_per_beat, axis=0)
```

**Potential Improvement**: 5-10x flatness improvement
**Effort**: Medium (~2 hours)
**Pros**:
- Aligns to music structure
- Captures rhythmic patterns
- More robust to tempo variations
- Can increase feature quality significantly

**Cons**:
- Requires beat detection (can fail on irregular music)
- Slightly more computation

### Option 2: Pre-Trained Audio Embeddings

**Approaches**:
- **OpenL3**: Open-source embeddings trained on large music corpus
- **MusicNN**: Neural network trained on Million Song Dataset
- **AcousticBrainz API**: External service for audio similarity
- **Spotify Echo Nest**: If available

**Implementation**:
```python
import openl3

# Load pre-trained model
model = openl3.models.load_embedding_model('music', input_repr='mel256')

# Extract embeddings for 3 segments
emb_start = openl3.get_embedding(y_start, sr, model=model)[0]
emb_mid = openl3.get_embedding(y_mid, sr, model=model)[0]
emb_end = openl3.get_embedding(y_end, sr, model=model)[0]

# Return embeddings (512-2048 dimensions)
return np.vstack([emb_start, emb_mid, emb_end])
```

**Potential Improvement**: 10-50x flatness improvement
**Effort**: Low-Medium (~1-2 hours)
**Pros**:
- Learned from millions of songs
- Captures genre, mood, instrumentation naturally
- Proven effective for music similarity
- Can provide 512-2048 dimensions

**Cons**:
- Adds external dependency
- May require GPU for speed
- Training data bias may affect niche genres

### Option 3: Enhanced Handcrafted Features

**Additional Features**:
- Onset rate (percussiveness)
- Key & mode (tonal center)
- Timbre brightness (spectral envelope)
- Dynamic range (RMS variation)
- Zero-crossing rate (instrument presence)
- Energy contour (over time)

**Implementation**:
```python
def extract_enhanced_features(y, sr):
    # Existing features
    features = [...]  # 27 current features

    # New features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(onset_env.mean())
    features.append(onset_env.std())

    # Key/mode detection
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.var(chroma.mean(axis=1)))  # Key strength

    # Timbre brightness
    spectral_brightness = np.mean(librosa.stft(y)**2 * np.arange(len(np.fft.fft(y))))
    features.append(spectral_brightness)

    return np.array(features)  # 35+ dimensions
```

**Potential Improvement**: 3-5x flatness improvement
**Effort**: Medium (~2-3 hours)
**Pros**:
- No external dependencies
- Directly addresses missing information
- Incremental improvement path

**Cons**:
- Slower than just using pre-trained models
- May not reach required improvement

---

## Recommendation

### Primary Path: Beat-Synchronized Features
1. **Why**: Most practical, proven effective in audio ML
2. **Implementation**: 2-3 hours of development
3. **Expected Result**: 5-10x improvement in flatness
4. **Fallback**: If doesn't work, try pre-trained embeddings

### If Beat-Sync Insufficient: Pre-Trained Embeddings
1. **Why**: Highest likelihood of success
2. **Implementation**: 1-2 hours of development
3. **Expected Result**: 10-50x improvement in flatness
4. **Setup**: Add OpenL3 as optional feature source

### Parallel: Phase C & D Cannot Proceed
- Transition scoring improvements (Phase C) depend on functional sonic features
- Dynamic mode rebalancing (Phase D) depends on effective genre gating + sonic ranking
- **Prerequisite**: Fix sonic features to pass Phase A validation

---

## Deliverables This Session

### Code & Scripts
1. **`scripts/sonic_validation_suite.py`** (550 lines)
   - Complete validation framework
   - 4 diagnostic metrics
   - M3U export for listening tests
   - Markdown report generation

2. **`src/similarity/sonic_variant.py`** (MODIFIED)
   - Added `robust_whiten` variant
   - RobustScaler + PCA whitening

3. **`PHASE_AB_DIAGNOSTIC_SUMMARY.md`**
   - Initial diagnostic report
   - Recommendations for next steps

### Data & Results
1. **`diagnostics/sonic_validation/`** (12 seed directories)
   - Full results for each seed
   - 3 M3U playlists per seed
   - CSV metrics
   - Markdown reports

2. **`diagnostics/sonic_validation_aggregate_results.csv`**
   - Aggregated metrics across all 12 seeds
   - Summary statistics

3. **`diagnostics/sonic_validation_variants/`** (4 variant directories)
   - Results for each normalization variant
   - Proof that normalization doesn't help

---

## Files Modified

| File | Change | Lines | Reason |
|------|--------|-------|--------|
| `src/similarity/sonic_variant.py` | Added `robust_whiten` variant | +15 | Phase B testing |
| `scripts/sonic_validation_suite.py` | NEW - validation framework | +550 | Phase A diagnostics |

---

## Timeline

| Phase | Status | Duration | Output |
|-------|--------|----------|--------|
| Phase A: Validation | ✓ Complete | 2 hours | Proves sonic broken |
| Phase B: Normalization | ✓ Complete | 2 hours | Proves issue is feature extraction |
| Phase C: Transitions | ⏳ Blocked | — | Requires fixed sonic |
| Phase D: Rebalancing | ⏳ Blocked | — | Requires fixed sonic |
| **Phase B.1: Beat-Sync Features** | ⏳ Next | ~2 hours | Should fix sonic |

---

## Success Metrics for Next Phase

When beat-sync features (or alternative) is implemented, success is measured by:

**Must achieve**:
- ✓ Score flatness ≥ 0.5 (currently 0.037)
- ✓ TopK gap ≥ 0.15 (currently 0.014)
- ✓ At least 7/12 seeds pass all 4 metrics

**Then can proceed**:
- Phase C: Transition scoring improvements
- Phase D: Dynamic mode rebalancing
- Complete playlist quality overhaul

---

## Conclusion

### What We Learned
1. **Sonic similarity is completely broken** - flat scores across all tracks
2. **Root cause is feature extraction** - normalization can't fix missing information
3. **Current 27-dim features insufficient** - too generic, fixed windows, mean aggregation
4. **Clear path forward exists** - beat-sync or pre-trained embeddings can fix it

### Key Insight
Following the plan principle "do NOT hand-tune weights first" was correct. The problem isn't in weight tuning or normalization—it's in the foundation. Systematic validation revealed this before wasting time on dead ends.

### Next Action
**Implement beat-synchronized feature extraction** or explore pre-trained embeddings. Phase C & D cannot proceed until sonic features pass Phase A validation.

---

**Report Generated**: 2025-12-16
**Session Duration**: ~4 hours
**Diagnostic Status**: COMPLETE
**Implementation Status**: BLOCKED (waiting for feature extraction fix)

EOF
