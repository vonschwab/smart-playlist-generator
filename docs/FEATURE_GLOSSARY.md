# Audio Feature Glossary

Quick reference for sonic features used in playlist generation. For detailed explanations, see [SONIC_FEATURES_REFERENCE.md](SONIC_FEATURES_REFERENCE.md).

---

## Quick Lookup Table

| Term | Full Name | What It Measures | Why It Matters | Units |
|------|-----------|------------------|----------------|-------|
| **MFCC** | Mel-Frequency Cepstral Coefficients | Timbre / sound "color" | Most important for similarity (60% weight) | Cepstral coefficients |
| **Chroma** | Chromatic scale representation | Harmonic content / pitch class distribution | Secondary importance (20% weight) | Energy per pitch class |
| **BPM** | Beats Per Minute | Tempo / speed | Affects transition compatibility (10% weight) | Beats per minute |
| **Spectral Centroid** | Center of mass of frequency spectrum | Brightness (high = bright, low = dark) | Minor contributor (10% weight) | Hz |
| **Spectral Rolloff** | Frequency of 85% energy cutoff | High-frequency extent | Distinguishes clean vs. distorted | Hz |
| **Spectral Contrast** | Peak-to-valley ratio per frequency band | Clarity/punchiness of sound | Captures dynamic range | dB |
| **RMS Energy** | Root Mean Square amplitude | Loudness / intensity | Volume envelope dynamics | Amplitude (0-1) |
| **Zero Crossing Rate** | Rate of amplitude sign changes | Noisiness / percussiveness | Noise vs. tonal distinction | Crossings/sec |
| **Beat** | Regularly occurring pulse in music | Musical downbeat | Foundation for beat-sync extraction | Frames/time |

---

## Common Questions

### "Why is MFCC 60% of the similarity score?"

MFCC (timbre) is the **most distinctive** attribute of sound. Two songs can have:
- Same tempo (BPM) but sound completely different
- Same key (Chroma) but sound completely different
- Different timbres (MFCC) but be instantly recognizable as different

Therefore, MFCC gets the most weight.

### "What's the difference between Spectral Centroid and Spectral Rolloff?"

- **Spectral Centroid**: Average frequency (center of mass)
- **Spectral Rolloff**: Threshold frequency where 85% of energy is below

Example:
- Song with one loud 5kHz component and rest quiet → Centroid ≈ 5kHz, Rolloff > 15kHz
- Song with even energy 0-5kHz → Centroid ≈ 2.5kHz, Rolloff ≈ 5kHz

### "When would two songs have high MFCC similarity but low Chroma similarity?"

Example: Two songs in **different keys** but made with **same instrument/voice**:
- Violin playing C Major melody → High MFCC match
- Violin playing G Major melody → Low Chroma match

Result: Similar timbre, different harmonic structure → similarity ≈ 0.60 + something

### "Is 0.90 similarity score good?"

Depends on context:
- **0.90+**: Extremely similar (likely same genre, same era, similar mood)
- **0.70-0.90**: Clearly related (compatible for transitions)
- **0.40-0.70**: Vaguely related (different vibes, might clash)
- **<0.40**: Very different (likely poor transition)

### "What extraction method should I use?"

**Use Beat-Sync if**:
- You want tempo-aware analysis
- Your audio has clear, consistent beat
- You're analyzing dance/electronic/pop music
- You have time for slower extraction

**Use Windowed if**:
- You want fast extraction
- Your audio is beat-ambiguous (classical, ambient)
- You need deterministic results (no beat detection variability)

---

## Feature Extraction Modes

### Beat-Sync (Phase 2 - Recommended)
- **Total dimensions**: 71
- **Extraction method**: Per-beat aggregation (median + IQR)
- **Robust to**: Tempo variations, beat tempo mismatches
- **When extracted**: 30-60 seconds per track (beat detection adds overhead)
- **File flag**: `extraction_method: "beat_sync"`

### Windowed (Legacy)
- **Total dimensions**: 52-60
- **Extraction method**: Fixed 30-second segment windows (mean + std)
- **Robust to**: Nothing (fixed windows)
- **When extracted**: 10-20 seconds per track (faster)
- **File flag**: `extraction_method: "windowed"`

---

## Real-World Feature Examples

### Kick Drum Sound

| Feature | Value | Why |
|---------|-------|-----|
| MFCC | [3.5, 2.1, 0.8, ...] | Percussive attack + low-frequency body |
| Spectral Centroid | 100 Hz | Most energy in low frequencies |
| Spectral Rolloff | 300 Hz | Little high-frequency content |
| RMS Energy | 0.4 | High peak amplitude |
| Zero Crossing | 0.05 | Very low (simple waveform) |

### Cymbal Crash

| Feature | Value | Why |
|---------|-------|-----|
| MFCC | [1.2, 0.5, -0.3, ...] | Bright, inharmonic partial |
| Spectral Centroid | 10000 Hz | Most energy in high frequencies |
| Spectral Rolloff | 20000 Hz | Extended high-frequency content |
| RMS Energy | 0.3 | Peak amplitude |
| Zero Crossing | 0.4 | Very high (complex, noisy) |

### Human Voice (Vocal)

| Feature | Value | Why |
|---------|-------|-----|
| MFCC | [2.8, 1.6, 0.4, ...] | Formant structure (unique to speaker) |
| Chroma | [0.15, 0.08, 0.20, ...] (varies by note) | Follows pitch/melody |
| Spectral Centroid | 2000-4000 Hz | Concentrated in "speech band" |
| RMS Energy | 0.2-0.3 | Variable based on phoneme |
| Zero Crossing | 0.15 | Moderate (mixture of voiced/unvoiced) |

### Synthesizer Pad

| Feature | Value | Why |
|---------|-------|-----|
| MFCC | [2.2, 1.1, 0.2, ...] | Relatively smooth, artificial |
| Spectral Centroid | 3000 Hz | Depends on synthesis parameters |
| Spectral Contrast | Low | Smooth, few peaks/valleys |
| RMS Energy | 0.25 (sustained) | Constant without attack |
| Zero Crossing | Low | Smooth waveforms |

---

## Mathematical Details

### MFCC Computation
```
Audio Signal
    ↓ [STFT - Short-Time Fourier Transform]
Frequency Spectrum
    ↓ [Mel-Scale Mapping - log frequency]
Mel Spectrum
    ↓ [Discrete Cosine Transform]
Cepstral Coefficients (13 values)
```

### Chroma Computation
```
Audio Signal
    ↓ [STFT]
Frequency Spectrum
    ↓ [Map to 12 Pitch Classes: C, C#, D, ..., B]
    ↓ [Octave Invariant - sum all octaves]
Chroma Vector (12 values)
```

### Similarity Computation
```
Track A Features → Normalize (L2 norm)
Track B Features → Normalize (L2 norm)
    ↓
Compute per-feature similarity:
- MFCC: cosine_distance → weight 0.60
- Chroma: cosine_distance → weight 0.20
- BPM: difference → weight 0.10
- Spectral: cosine_distance → weight 0.10
    ↓
Weighted Sum → Final Score (0.0 to 1.0)
```

---

## Database Storage

Features are stored in the `sonic_features` JSON column of the `tracks` table.

### Beat-Sync JSON Structure
```json
{
  "extraction_method": "beat_sync",
  "mfcc_median": [3.2, 1.8, -0.5, ...],        // 13 values
  "mfcc_iqr": [0.5, 0.3, 0.4, ...],            // 13 values
  "chroma_median": [0.12, 0.08, 0.15, ...],    // 12 values
  "chroma_iqr": [0.02, 0.03, 0.01, ...],       // 12 values
  "spectral_contrast_median": [1.0, 2.0, ...], // 7 values
  "spectral_contrast_iqr": [0.5, 0.8, ...],    // 7 values
  "bpm": 120.5,
  "onset_strength_mean": 0.42,
  "onset_strength_std": 0.15,
  "spectral_centroid_mean": 2400,
  "spectral_centroid_std": 600,
  "spectral_rolloff_mean": 8000,
  "spectral_rolloff_std": 1200
}
```

### Windowed JSON Structure
```json
{
  "extraction_method": "windowed",
  "mfcc_mean": [...],           // 13 values
  "mfcc_std": [...],            // 13 values
  "chroma_mean": [...],         // 12 values
  "chroma_std": [...],          // 12 values
  "spectral_contrast_mean": [...], // 7 values
  "bpm": 120.5,
  "spectral_centroid": 2400,
  "spectral_rolloff": 8000,
  "rms_energy": 0.25,
  "zero_crossing_rate": 0.05
}
```

---

## Related Documentation

- **[SONIC_FEATURES_REFERENCE.md](SONIC_FEATURES_REFERENCE.md)** - Full technical documentation
- **[data_model.md](data_model.md)** - Database schema for features
- **[architecture.md](architecture.md)** - System component overview
- **[librosa_analyzer.py](../src/librosa_analyzer.py)** - Feature extraction code

---

## Summary

✅ **Key Takeaways**:
1. MFCC (timbre) is the most important feature - 60% of similarity score
2. Chroma (harmony) is secondary - 20% of similarity score
3. Beat-sync extraction is recommended for robust, tempo-aware analysis
4. Features are normalized before similarity computation
5. All features are stored as JSON in the database for reproducibility

🎯 **For Quick Decisions**:
- "Is this song like that one?" → Compare MFCCs
- "Do they fit the same playlist?" → Check Chroma + BPM match
- "Why did it choose this transition?" → Examine segment-level (start/mid/end) features

---

**Last Updated**: December 2024

**Status**: ✅ Quick reference complete - Companion to SONIC_FEATURES_REFERENCE.md
