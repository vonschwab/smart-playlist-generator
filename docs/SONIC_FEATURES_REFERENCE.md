# Sonic Features Reference

Canonical documentation for audio features extracted and used in the Playlist Generator system.

---

## Overview

Audio features extracted from music files are the foundation of sonic similarity matching. This document defines:
- **What** each feature represents (semantic meaning)
- **Why** it matters for music similarity
- **How** we extract and process it
- **Units** and valid ranges

### Feature Extraction Pipeline

```
Audio File
    ↓
Librosa Audio Analysis (librosa Python library)
    ↓
Beat-Sync or Windowed Feature Extraction
    ↓
JSON features stored in database
    ↓
Feature Vector Construction (SimilarityCalculator)
    ↓
Normalized 1D vector for similarity computation
```

---

## Extraction Methods

### Beat-Sync Extraction (Phase 2 - Current)

**When**: Enabled by `--beat-sync` flag or `use_beat_sync=True` in LibrosaAnalyzer

**How It Works**:
1. Detect musical beats using librosa's beat tracking algorithm
2. Extract features per beat interval (not fixed time windows)
3. Aggregate using **median + IQR** (robust statistics)
   - More resistant to outliers than mean/std
   - Captures central tendency and variability

**Advantages**:
- Aligned to actual musical structure (tempo-aware)
- Robust to tempo variations
- Better handles rhythm discontinuities
- 71 total dimensions

**Disadvantages**:
- Requires sufficient beats detected (fallback to windowed if < 3 beats)
- Slightly slower than windowed extraction

### Windowed Extraction (Legacy)

**When**: `use_beat_sync=False` in LibrosaAnalyzer

**How It Works**:
1. Extract features from fixed time windows:
   - **Beginning**: 0-30 seconds
   - **Middle**: 30 seconds from track center
   - **End**: Last 30 seconds (or from [duration-30s])
   - **Average**: Aggregated across all 3 segments

**Advantages**:
- Simple, predictable timing
- Works with all audio (no beat detection required)
- 52 total dimensions

**Disadvantages**:
- Ignores actual musical beats
- Fixed segments may split important musical moments
- Less robust to tempo variations

---

## Feature Categories & Definitions

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

**Purpose**: Timbre characterization - the "color" or "texture" of the sound

**How It's Extracted**:
1. Transform audio to mel-scale (logarithmic frequency scale that matches human hearing)
2. Apply discrete cosine transform to get cepstral coefficients
3. Extract 13 coefficients per time window
4. Beat-sync: Store **median** and **IQR** across beats (26 dimensions)
5. Windowed: Store **mean** and **std** across time (26 dimensions)

**Why It Matters**:
- MFCCs are the industry standard for audio timbre
- Different instruments have different MFCC patterns
- Used in music recognition, genre classification, etc.

**Similarity**: Cosine distance between MFCC vectors
- Close distance = similar timbral characteristics
- Weight in hybrid: **60%** (most important)

**Units**: Cepstral coefficients (no standard unit, relative values only)

**Valid Range**: Typically -30 to +10 (varies by audio content)

**Example Interpretation**:
- Synthesizer: Flat MFCC profile (limited harmonic content)
- Piano: Decaying MFCC profile (note attack then sustain)
- Vocals: Dynamic MFCC changes (formant movements)

---

### 2. Chroma / HPCP (Harmonic Content)

**Purpose**: Harmonic/tonal structure - the "pitch class" distribution

**How It's Extracted**:
1. Compute short-time Fourier transform (STFT)
2. Map energy to 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
3. Extract 12 chroma values per window
4. Beat-sync: Store **median** and **IQR** (24 dimensions)
5. Windowed: Store **mean** and **std** (24 dimensions)

**Why It Matters**:
- Captures harmonic/melodic content independent of octave
- Songs in the same key have similar chroma profiles
- Different from timbre - measures "what notes are played"

**Similarity**: Cosine distance between chroma vectors
- Close distance = similar harmonic structure
- Weight in hybrid: **20%** (secondary importance)

**Units**: Energy distribution across pitch classes (0-1 typical range)

**Valid Range**: 0-1 per bin (sum typically ~1.0-1.2)

**Example Interpretation**:
- C Major Scale: High energy in C, D, E, G, A (C-D-E-G-A pattern)
- C Minor Scale: High energy in C, Eb, G (darker profile)
- Atonal Music: Relatively even distribution across all 12 pitches

---

### 3. Spectral Contrast

**Purpose**: Relative loudness of peaks vs. valleys in the frequency spectrum

**How It's Extracted**:
1. Compute STFT and divide spectrum into sub-bands
2. For each sub-band, measure:
   - Peak height (loudest frequency)
   - Valley depth (quietest frequency)
   - Contrast = Peak - Valley
3. Extract 7 values (for 7 frequency bands)
4. Beat-sync: Store **median** and **IQR** (14 dimensions)
5. Windowed: Store **mean** only (7 dimensions)

**Why It Matters**:
- Captures "brightness" or "clarity" of the sound
- High contrast = clear, punchy tones (e.g., drums, strings)
- Low contrast = smooth, blended tones (e.g., synth pads, vocal layers)

**Similarity**: Cosine distance between spectral contrast vectors

**Units**: dB (decibels), relative loudness

**Valid Range**: Typically 0-30 dB

**Example Interpretation**:
- Snare Drum: High contrast (sharp attack, clear peaks)
- Sine Wave: Zero contrast (single frequency, no peaks vs. valleys)
- Orchestral Swell: Low contrast (many frequencies at similar levels)

---

### 4. BPM (Beats Per Minute)

**Purpose**: Tempo characterization

**How It's Extracted**:
1. Detect beat frames using `librosa.beat.beat_track()`
2. Count beats per minute
3. Single scalar value (1 dimension)

**Why It Matters**:
- Fast songs have different listener experience than slow songs
- Similar BPM often indicates compatible track transitions
- Important for DJ-style playlist continuity

**Similarity**: Absolute difference (lower = more similar)
- Typically penalized in hybrid similarity (part of 10% spectral features weight)

**Units**: Beats per minute (BPM)

**Valid Range**: 30-240 typical (extremes: 10-300+ possible)

**Example Interpretation**:
- 60 BPM: Slow ballad, one beat per second
- 120 BPM: Moderate tempo, two beats per second
- 200 BPM: Fast dance, high energy

---

### 5. Spectral Centroid

**Purpose**: Center of mass of the frequency spectrum (bright vs. dark)

**How It's Extracted**:
1. Compute STFT
2. Calculate weighted average frequency (weighted by magnitude)
3. Store mean and std across time

**Why It Matters**:
- Captures overall "brightness" or "darkness" of sound
- High centroid = bright (more high frequencies)
- Low centroid = dark (more low frequencies)

**Similarity**: Euclidean distance

**Units**: Hz (frequency in Hertz)

**Valid Range**: 0-22050 Hz (up to Nyquist frequency at 22kHz sample rate)
- Typical: 1000-8000 Hz
- Dark sound: 2000-3000 Hz (e.g., cellos, kick drums)
- Bright sound: 5000-10000 Hz (e.g., cymbals, metal)

**Example Interpretation**:
- Kick Drum: 100 Hz (very dark, low frequencies)
- Snare Drum: 4000 Hz (bright, midrange peaks)
- Hi-Hat Cymbal: 8000 Hz (very bright, high frequencies)

---

### 6. Spectral Rolloff

**Purpose**: Frequency at which 85% of energy is contained below

**How It's Extracted**:
1. Compute cumulative energy across frequencies
2. Find frequency where 85% of total energy is below
3. Store mean and std across time

**Why It Matters**:
- Similar to centroid but more robust to outliers
- Measures "high-frequency energy extent"
- Important for distinguishing clean vs. distorted sounds

**Similarity**: Euclidean distance

**Units**: Hz

**Valid Range**: 0-22050 Hz
- Typical: 2000-15000 Hz
- Clean sound: 8000-12000 Hz
- Distorted/noisy sound: 12000-20000 Hz

**Example Interpretation**:
- Acoustic Guitar: 6000 Hz (most energy below 6kHz)
- Electric Guitar (Distorted): 15000 Hz (extends to high frequencies)
- Whisper: 4000 Hz (limited high-frequency content)

---

### 7. RMS Energy (Windowed Only)

**Purpose**: Overall loudness/intensity

**How It's Extracted**:
1. Compute Root Mean Square of audio signal
2. Store mean and std across time

**Why It Matters**:
- Captures volume envelope
- Dynamic vs. static music have different energy profiles
- Useful for transition analysis

**Similarity**: Euclidean distance

**Units**: Amplitude (0-1 typically normalized)

**Valid Range**: 0-1

**Example Interpretation**:
- Silent: 0.01 RMS
- Normal dialogue: 0.1 RMS
- Shouting: 0.3-0.5 RMS
- Loud music: 0.2-0.4 RMS

---

### 8. Zero Crossing Rate (Windowed Only)

**Purpose**: "Noisiness" or "percussiveness" indicator

**How It's Extracted**:
1. Count times per frame the signal crosses zero (amplitude changes sign)
2. Normalize by frame length
3. Store as single rate value

**Why It Matters**:
- High ZCR = noisy/percussive (lots of fast amplitude changes)
- Low ZCR = tonal (smooth, sustained notes)
- Distinguishes speech/noise from music

**Similarity**: Euclidean distance

**Units**: Crossings per second

**Valid Range**: 0-0.5 (typical)

**Example Interpretation**:
- Sine Wave (1 kHz): 2000 crossings/sec (very regular)
- White Noise: Highest possible rate (random, all frequencies)
- Sustained Vocal: 3000-5000 crossings/sec (mid-range)

---

## Feature Vector Construction

### Build Process (SimilarityCalculator._vector_from_features)

Features are concatenated in fixed order:

```
Feature Vector = [MFCC | Chroma | BPM | Spectral Centroid]
```

**Concatenation Order**:
1. **MFCC** (13 dims for beat-sync, 26 dims for windowed)
2. **Chroma** (12 dims for beat-sync, 24 dims for windowed)
3. **BPM** (1 dim)
4. **Spectral Centroid** (1 dim)
5. **Spectral Rolloff** (1 dim, if available)
6. **RMS Energy** (2 dims, if available)
7. **Other features** (fallback positions)

### Feature Vector Dimensions

**Beat-Sync Mode** (Current):
- MFCC median (13) + IQR (13) = 26
- Chroma median (12) + IQR (12) = 24
- Spectral contrast median (7) + IQR (7) = 14
- BPM (1)
- Onset strength mean (1) + std (1) = 2
- Spectral centroid mean (1) + std (1) = 2
- Spectral rolloff mean (1) + std (1) = 2
- **Total: 71 dimensions**

**Windowed Mode** (Legacy):
- MFCC mean (13) + std (13) = 26
- Chroma mean (12) + std (12) = 24
- Spectral contrast mean (7) = 7
- BPM (1)
- Spectral centroid (1)
- Spectral rolloff (1)
- RMS energy (2)
- Zero crossing rate (1)
- **Total: 52-60 dimensions** (varies)

---

## Similarity Scoring

### Weighted Feature Contributions

The sonic similarity score combines weighted distances for each feature type:

| Feature | Weight | Reasoning |
|---------|--------|-----------|
| **MFCC** | 60% | Timbre is most important for sonic similarity |
| **Chroma** | 20% | Harmonic structure matters but secondary to timbre |
| **BPM** | 10% | Tempo compatibility, but not primary |
| **Spectral Centroid** | 10% | Brightness/darkness, minor contributor |

**Example**:
- Two songs with identical timbre (MFCC) but different key → 60% score
- Two songs with same key (Chroma) but different timbre → 20% score
- Two songs with everything same except tempo → 90% score

### Computation Details

1. **Per-feature normalization**: L2 norm to unit length
2. **Per-feature similarity**: Cosine distance (1 - cosine_distance)
3. **Weighted average**: sum(weight[i] * similarity[i])
4. **Final range**: 0.0 (completely different) to 1.0 (identical)

---

## Industry Standards

### Comparison to Other Systems

| System | Timbre Features | Harmonic Features | Rhythm |
|--------|-----------------|-------------------|--------|
| **Playlist Generator** | MFCC (13 dim) | Chroma (12 dim) | BPM only |
| **Spotify** | Unknown (proprietary) | Unknown | Tempo, key, energy |
| **MusicBrainz** | AcousticBrainz (low-level) | Unknown | Genre-based |
| **Last.fm** | Genre-based | Tags | Genre-based |
| **Essentia** | MFCC, MFCCs, spectral features | HPCP | Beat detection |

### Librosa vs. Essentia

Both are audio analysis libraries:
- **Librosa** (used here): Python-based, flexible, good for research
- **Essentia**: C++ backend, more optimized, more features

Our choice of librosa allows easy customization while maintaining reproducibility.

---

## Data Flow Examples

### Example 1: Comparing Two Songs

**Song A** (Fast Rock):
```json
{
  "mfcc_median": [3.2, 1.8, -0.5, ...],
  "chroma_median": [0.12, 0.08, 0.15, ...],  // C major tonality
  "bpm": 140,
  "spectral_centroid_mean": 4200
}
```

**Song B** (Slow Ballad):
```json
{
  "mfcc_median": [3.5, 1.9, -0.4, ...],      // Similar timbre
  "chroma_median": [0.11, 0.09, 0.14, ...],  // Similar key
  "bpm": 70,                                  // Different tempo!
  "spectral_centroid_mean": 4100
}
```

**Similarity Calculation**:
- MFCC similarity: 0.94 (very similar) × 0.60 = 0.564
- Chroma similarity: 0.96 (very similar) × 0.20 = 0.192
- BPM difference: 70 (10% weight) → reduced score
- Spectral centroid: 0.98 × 0.10 = 0.098

**Result**: ~0.85 overall similarity (same vibe, different energy)

---

### Example 2: Beat-Sync Advantage

**Scenario**: Song with tempo change at middle

**Windowed (Fixed 30-sec windows)**:
- Beginning: Normal tempo features
- Middle: Tempo change creates discontinuity
- End: Back to normal
- Result: Middle segment is "outlier", affects global stats

**Beat-Sync (Per-beat aggregation)**:
- Extract all beats regardless of tempo
- Median aggregation ignores outlier beats
- Result: Robust to tempo changes

---

## Quality Assurance

### Feature Extraction Checks

The system validates features via:

1. **Duration Validation** (`scripts/validate_duration.py`):
   - Ensures all tracks have duration populated
   - Checks values are reasonable (> 0)

2. **Sonic Validation Suite** (`scripts/sonic_validation_suite.py`):
   - Tests whether sonic similarity is informative
   - Metrics:
     - **Flatness**: Distribution width (higher = better separation)
     - **TopK Gap**: Difference between top-ranked and random neighbors
     - **Intra-artist coherence**: Same artist tracks closer than random
     - **Intra-album coherence**: Same album tracks closer than random

3. **Sonic Health Check** (`scripts/check_duration_health.py`):
   - Quick statistics on feature population
   - Identifies missing or malformed features

### Acceptance Criteria

✅ **Features are valid if**:
- All tracks have sonic_features JSON populated
- MFCC dimensions: 13 (beat-sync) or 26 (windowed)
- Chroma dimensions: 12 (beat-sync) or 24 (windowed)
- BPM: 30-240 range
- Spectral values: within Hz range (0-22050)

❌ **Features are invalid if**:
- Any dimension is NaN or Inf
- BPM < 10 or > 300
- Zero-length feature vectors

---

## Troubleshooting

### Problem: Sonic features too flat (low discriminative power)

**Symptom**: All songs have similarity > 0.98

**Possible Causes**:
1. BPM dimension dominating (phase 1 issue - should be fixed)
2. Features not normalized properly
3. Window size too large (mixing disparate segments)

**Solutions**:
1. Check feature normalization in artifact builder
2. Use beat-sync extraction instead of windowed
3. Verify MFCC parameters (n_mfcc=13 correct)

### Problem: Beat-sync extraction failing

**Symptom**: Fallback to windowed for many tracks

**Possible Causes**:
1. Audio has weak beat detection
2. Tempo too irregular
3. Less than 3 beats detected

**Solutions**:
1. Check beat_track() output: `tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)`
2. Increase hop_length if needed (currently 512)
3. Consider using windowed for problematic files

### Problem: High variance between beat-sync and windowed

**Symptom**: Switching extraction methods changes all similarity scores significantly

**Expected**: Some variance due to different aggregation (median vs. mean)

**Not a problem if**: Relative ranking of neighbors is stable

---

## References

### Librosa Documentation
- [Librosa MFCC](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html)
- [Librosa Chroma](https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html)
- [Librosa Spectral Features](https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html)
- [Librosa Beat Tracking](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)

### Audio Processing References
- MFCC: "Mel-Frequency Cepstral Coefficients" (Davis & Mermelstein, 1980)
- Chroma: "Music information retrieval using chroma" (Ellis, 2007)
- Spectral Contrast: "Robust Music Detection in Different Background Noise Environments" (Jiang et al., 2002)

### Related Documentation
- [Data Model](data_model.md) - Database schema for sonic features
- [Pipelines](pipelines.md) - Feature extraction workflows
- [Architecture](architecture.md) - System component overview

---

## Summary Table

| Feature | Dimensions | Unit | Weight | Meaning |
|---------|-----------|------|--------|---------|
| MFCC | 13 | Cepstral coeff | 60% | Timbre/texture |
| Chroma | 12 | Pitch class energy | 20% | Harmonic content |
| BPM | 1 | Beats/min | 10% | Tempo |
| Spectral Centroid | 1 | Hz | 10% | Brightness |
| Spectral Rolloff | 1 | Hz | - | High-freq extent |
| Spectral Contrast | 7 | dB | - | Peak vs. valley |
| RMS Energy | 1 | Amplitude | - | Loudness |
| Zero Crossing | 1 | Crossings/s | - | Noisiness |

---

**Last Updated**: December 2024

**Maintainer**: Playlist Generator Development Team

**Status**: ✅ Complete - Canonical reference for all sonic feature extraction and usage
