# Phase A & B Diagnostic Summary

**Date**: 2025-12-16
**Status**: SONIC FEATURES FUNDAMENTALLY BROKEN

---

## Phase A Results: VALIDATION FAILS (0/12 seeds)

Tested 12 diverse seeds from `diagnostics/tune_seeds.txt`. All failed all 4 metrics:

| Metric | Min | Median | Max | Threshold | Result |
|--------|-----|--------|-----|-----------|--------|
| Score Flatness | 0.017 | 0.037 | 0.084 | ≥ 0.5 | **FAIL 0/12** |
| TopK vs Random Gap | 0.008 | 0.014 | 0.065 | ≥ 0.15 | **FAIL 0/12** |
| Intra-Artist Coherence | -0.005 | 0.013 | 0.034 | ≥ 0.05 | **FAIL 0/12** |
| Intra-Album Coherence | -0.012 | 0.013 | 0.064 | ≥ 0.08 | **FAIL 0/12** |

**Key Finding**: Sonic scores are so compressed that all 34,100 tracks have nearly identical similarity to any seed. Top-30 neighbors are barely better than random.

---

## Phase B Results: NORMALIZATION DOESN'T HELP

Tested 4 normalization variants on 3 test seeds:

| Variant | Flatness | TopK Gap | Intra-Artist | Intra-Album | Improvement |
|---------|----------|----------|--------------|-------------|-------------|
| raw (baseline) | 0.034 | 0.016 | 0.009 | 0.006 | N/A |
| z_clip (current) | 0.034 | 0.015 | 0.007 | 0.005 | No |
| whiten_pca | 0.034 | 0.016 | 0.007 | 0.016 | Minimal |
| robust_whiten (NEW) | 0.034 | 0.019 | 0.006 | 0.013 | Minimal |

**Critical Observation**: **All variants have identical sonic flatness (0.034)**.

This proves the problem is NOT in post-processing/normalization.
The problem is in the **feature extraction pipeline**.

---

## Root Cause Analysis

The current sonic feature extraction (`src/librosa_analyzer.py`) extracts 27 dimensions:

- **13 MFCCs** (mel-frequency cepstral coefficients)
- **1 Spectral Centroid**
- **1 Spectral Rolloff**
- **1 Spectral Bandwidth**
- **7 Spectral Contrast** bands
- **1 BPM** (tempo)
- **12 Chroma** (pitch class distribution)

**Problem**: These features are too generic and compressed.

### Why Features Are Uninformative

1. **Fixed 30-second windows**: Don't adapt to song structure (verses, choruses, bridges)
2. **Mean aggregation**: Loses temporal patterns and dynamics
3. **Generic spectral features**: Don't capture genre-defining characteristics
4. **No beat alignment**: Features computed on fixed time windows, not beat-synchronized
5. **Limited dimensionality**: 27 dimensions with heavy compression loses information

---

## What Needs to Happen

### Option 1: Beat-Synchronized Features ⭐ (Recommended)
Extract features aligned to musical beats instead of fixed time windows.

**Potential improvements**:
- Captures rhythm/tempo variation
- Aligns features to musical structure
- More robust to tempo variations
- Could increase flatness score 5-10x

**Effort**: Medium (requires modify `src/librosa_analyzer.py`)

### Option 2: Pre-Trained Audio Embeddings
Use models like:
- **OpenL3**: Music/audio embeddings trained on large music corpus
- **MusicNN**: Neural network trained on Million Song Dataset
- **AcousticBrainz**: API for audio similarity

**Potential improvements**:
- Learned representations from millions of songs
- Captures genre, mood, instrumentation naturally
- Could improve flatness 10-50x

**Effort**: Low-Medium (just add as alternative feature source)

### Option 3: Additional Handcrafted Features
Add features that capture musical content better:
- **Onset rate** (percussiveness)
- **Key/mode** (tonal center)
- **Timbre brightness** (spectral envelope)
- **Dynamic range** (RMS variation)
- **Zero-crossing rate** (instrument presence)

**Potential improvements**: Modest (3-5x improvement)

**Effort**: Medium (modify feature extraction)

---

## Recommendation

**DO NOT PROCEED with current approach.**

The sonic features are fundamentally uninformative. Tweaking normalization won't fix a broken foundation.

### Next Steps:

1. **Implement beat-sync features** (Phase B Variant 1) - most practical
   - Modify `extract_beat_sync_features()` in `src/librosa_analyzer.py`
   - Extract MFCC/chroma per beat instead of fixed windows
   - Rebuild artifact and re-validate

2. **If beat-sync doesn't work**: Consider pre-trained embeddings (OpenL3)

3. **Fallback**: Add handcrafted music features to improve coverage

---

## Impact Assessment

**If beat-sync achieves 0.5 flatness** (currently 0.034, need 14x improvement):
- Genre enforcement dials become functional
- Sonic-only mode becomes usable
- Transition scoring can bind properly

**If we DON'T fix this**:
- All playlist quality issues persist
- Genre gates ineffective
- Transitions remain jarring

---

## Files Modified/Created This Session

### Phase A (Diagnostic)
- ✓ `scripts/sonic_validation_suite.py` (NEW) - 550 lines
- ✓ `diagnostics/sonic_validation/` - Results for 12 seeds
- ✓ `diagnostics/sonic_validation_aggregate_results.csv` - Summary metrics

### Phase B (Attempted Fixes)
- ✓ `src/similarity/sonic_variant.py` - Added `robust_whiten` variant
- ✓ Tested 4 normalization approaches
- ✗ All normalization approaches FAILED (proven ineffective)

---

## Conclusion

**Current Status**: BLOCKED - Sonic features too flat to improve with current approach

**Next Action**: Implement beat-synchronized feature extraction to capture musical structure

**Timeline**: Estimate 1-2 hours for beat_sync implementation + validation

---

*Report compiled: 2025-12-16*
*Sonic validation suite: 12 seeds, 0% pass rate*
*Normalization variants: 4 tested, 0% pass rate*
