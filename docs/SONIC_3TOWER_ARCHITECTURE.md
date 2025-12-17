# 3-Tower Beat-Synchronous Sonic Similarity Architecture

## Design Document v1.0

**Problem**: Current sonic similarity is flat/non-informative. All tracks cluster near cosine ~0.99, causing genre to dominate playlist curation and transitions to feel random.

**Root Cause**: Single-vector embedding conflates rhythm, timbre, and harmony. BPM variance dominates. Mean aggregation loses musical structure.

**Solution**: 3-tower beat-synchronous representation with calibrated per-tower similarity.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUDIO FILE (.flac/.mp3)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEAT-SYNCHRONOUS EXTRACTION                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ Beat Track   │──│ Per-Beat     │──│ Robust       │                       │
│  │ Detection    │  │ Features     │  │ Aggregation  │                       │
│  │ (librosa)    │  │ (onset,mfcc, │  │ (median/IQR) │                       │
│  └──────────────┘  │  chroma,etc) │  └──────────────┘                       │
│                    └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   RHYTHM TOWER      │ │   TIMBRE TOWER      │ │   HARMONY TOWER     │
│   (weight: 0.45)    │ │   (weight: 0.40)    │ │   (weight: 0.15)    │
│                     │ │                     │ │                     │
│ • Onset envelope    │ │ • MFCC (20 coef)    │ │ • Chroma CQT (12)   │
│   stats             │ │   median/IQR        │ │   median/IQR        │
│ • Tempogram peaks   │ │ • Spectral contrast │ │ • Chroma entropy    │
│ • Beat stability    │ │   median/IQR        │ │ • Tonnetz (optional)│
│ • BPM (half/double  │ │ • Spectral rolloff  │ │ • Key strength      │
│   aware)            │ │ • Spectral flux     │ │                     │
│                     │ │                     │ │                     │
│ Dims: ~25-35        │ │ Dims: ~60-80        │ │ Dims: ~30-40        │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PER-TOWER NORMALIZATION                                 │
│  • Robust standardization (median/IQR, clip outliers at ±3σ)                │
│  • Optional PCA whitening per tower (retain 90-95% variance)                │
│  • L2 normalize each tower embedding                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATED MULTI-TOWER SIMILARITY                         │
│                                                                              │
│  S_total = w_r * calibrate(S_rhythm) + w_t * calibrate(S_timbre)            │
│            + w_h * calibrate(S_harmony)                                      │
│                                                                              │
│  calibrate(): Convert raw cosine to z-score relative to random baseline     │
│  Default: w_r=0.45, w_t=0.40, w_h=0.15 (configurable)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   START SEGMENT     │ │   MID SEGMENT       │ │   END SEGMENT       │
│   (first 30s)       │ │   (middle 30s)      │ │   (last 30s)        │
│                     │ │                     │ │                     │
│ For transitions:    │ │ For seed sim        │ │ For transitions:    │
│ start(next)         │ │ (full track repr)   │ │ end(prev)           │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘

TRANSITION SCORING:
  TransitionSim(prev, next) = MultiTowerSim(end(prev), start(next))
                              × tempo_compatibility(prev, next)
```

---

## Tower Specifications

### Tower 1: Rhythm (Weight: 0.45)

**Rationale**: Rhythm is the most discriminative feature for genres like funk, afrobeat, electronic. Current system under-weights it.

**Features** (~25-35 dimensions):

| Feature | Aggregation | Dims | Description |
|---------|-------------|------|-------------|
| onset_strength | median, IQR, std, p10, p90 | 5 | Attack envelope statistics |
| tempogram_peaks | peak_bpm, peak2_bpm, peak_ratio | 3 | Dominant rhythmic periodicities |
| tempogram_acf | acf_lag1-5 | 5 | Rhythmic autocorrelation (groove pattern) |
| beat_interval_cv | cv | 1 | Coefficient of variation of beat intervals |
| beat_strength | median, IQR | 2 | How "strong" the beats are |
| onset_rate | mean | 1 | Onsets per second (busyness) |
| bpm | tempo, half_tempo_conf, double_tempo_conf | 3 | BPM with half/double awareness |
| rhythm_complexity | entropy | 1 | Rhythmic entropy/unpredictability |

**Half/Double Tempo Awareness**:
```python
# Store BPM with confidence for half/double relationships
tempo_info = {
    'primary_bpm': 120,
    'half_tempo_likely': False,  # 60 BPM interpretation
    'double_tempo_likely': False,  # 240 BPM interpretation
    'tempo_stability': 0.95  # How consistent the tempo is
}
```

### Tower 2: Timbre (Weight: 0.40)

**Rationale**: Timbre captures instrument/voice character. MFCC is industry standard but needs robust aggregation.

**Features** (~60-80 dimensions):

| Feature | Aggregation | Dims | Description |
|---------|-------------|------|-------------|
| mfcc (20 coef) | median, IQR | 40 | Mel-frequency cepstral coefficients |
| mfcc_delta | median | 20 | MFCC derivatives (dynamics) |
| spectral_contrast (7 bands) | median, IQR | 14 | Peak-to-valley per frequency band |
| spectral_rolloff | median, IQR | 2 | High-frequency cutoff |
| spectral_centroid | median, IQR | 2 | Brightness |
| spectral_bandwidth | median, IQR | 2 | Spectral width |
| spectral_flux | median, IQR | 2 | Rate of spectral change |
| zero_crossing_rate | median | 1 | Noisiness indicator |

**Why 20 MFCCs instead of 13**:
- 13 is speech-optimized
- 20 captures more musical detail (higher harmonics)
- Still computationally cheap

### Tower 3: Harmony (Weight: 0.15)

**Rationale**: Harmony matters for key/chord compatibility but should NOT dominate. Lower weight prevents key from overriding genre.

**Features** (~30-40 dimensions):

| Feature | Aggregation | Dims | Description |
|---------|-------------|------|-------------|
| chroma_cqt (12 bins) | median, IQR | 24 | Pitch class distribution (constant-Q) |
| chroma_entropy | mean | 1 | How "spread" the pitch classes are |
| chroma_peak_count | mean | 1 | Number of dominant pitch classes |
| tonnetz (6 dims) | median | 6 | Tonal centroid (circle of fifths) |
| key_strength | max_chroma / mean_chroma | 1 | How "key-centric" the track is |

**Chroma CQT vs STFT**:
- CQT (constant-Q transform) has better bass resolution
- More accurate for music than STFT-based chroma
- Slightly slower but worth it for harmonic accuracy

---

## Normalization Strategy

### Per-Tower Robust Standardization

```python
def robust_standardize(X, clip_sigma=3.0):
    """
    Standardize using median/IQR (robust to outliers).
    Then clip extreme values.
    """
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.maximum(iqr, 1e-6)  # Avoid division by zero

    # IQR to "pseudo-sigma" conversion (IQR ≈ 1.35σ for normal dist)
    pseudo_sigma = iqr / 1.35

    X_std = (X - median) / pseudo_sigma
    X_clipped = np.clip(X_std, -clip_sigma, clip_sigma)

    return X_clipped, {'median': median, 'iqr': iqr}
```

### Optional PCA Whitening Per Tower

```python
def pca_whiten_tower(X, variance_retain=0.95):
    """
    Reduce dimensionality while decorrelating features.
    Whitening ensures each PCA dimension has unit variance.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance_retain, whiten=True)
    X_pca = pca.fit_transform(X)

    return X_pca, {'pca': pca, 'n_components': pca.n_components_}
```

### Final L2 Normalization

```python
def l2_normalize(X):
    """L2 normalize for cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)
```

---

## Calibrated Similarity

### Problem: Tower Similarity Scales Differ

- Rhythm similarity might have mean 0.7, std 0.1
- Timbre similarity might have mean 0.85, std 0.05
- Direct weighted average would let timbre dominate

### Solution: Z-Score Calibration

```python
def calibrate_similarity(sim_raw, tower_stats):
    """
    Convert raw similarity to z-score relative to random baseline.
    tower_stats contains pre-computed mean/std from random pairs.
    """
    z = (sim_raw - tower_stats['random_mean']) / tower_stats['random_std']
    return z

def combined_similarity(query_vec, candidate_vecs, tower_slices, tower_stats, weights):
    """
    Compute calibrated multi-tower similarity.
    """
    total_sim = np.zeros(len(candidate_vecs))

    for tower_name, (start, end) in tower_slices.items():
        query_tower = query_vec[start:end]
        cand_tower = candidate_vecs[:, start:end]

        # Raw cosine similarity
        raw_sim = cand_tower @ query_tower

        # Calibrate to z-score
        calibrated = calibrate_similarity(raw_sim, tower_stats[tower_name])

        # Weight and accumulate
        total_sim += weights[tower_name] * calibrated

    return total_sim
```

### Pre-Computing Tower Statistics

During artifact build:
```python
# Sample 10,000 random pairs
random_pairs = sample_random_pairs(n=10000)
for tower_name in ['rhythm', 'timbre', 'harmony']:
    tower_sims = [cosine_sim(X[i], X[j]) for i, j in random_pairs]
    tower_stats[tower_name] = {
        'random_mean': np.mean(tower_sims),
        'random_std': np.std(tower_sims),
        'random_p10': np.percentile(tower_sims, 10),
        'random_p90': np.percentile(tower_sims, 90),
    }
```

---

## Segment-Based Transition Scoring

### Segment Extraction

```python
def extract_segment_features(y, sr, segment):
    """
    Extract 3-tower features for a specific segment.

    segment: 'start' (0-30s), 'mid' (center 30s), 'end' (last 30s)
    """
    duration = len(y) / sr

    if segment == 'start':
        y_seg = y[:int(30 * sr)]
    elif segment == 'mid':
        mid_start = int((duration / 2 - 15) * sr)
        y_seg = y[mid_start:mid_start + int(30 * sr)]
    elif segment == 'end':
        y_seg = y[-int(30 * sr):]

    return extract_3tower_features(y_seg, sr)
```

### Transition Similarity

```python
def transition_similarity(prev_track, next_track, tower_weights, tower_stats):
    """
    Compute how well prev_track flows into next_track.
    Uses end(prev) -> start(next) embedding similarity.
    """
    # Get segment embeddings
    end_prev = prev_track['end_embedding']
    start_next = next_track['start_embedding']

    # Multi-tower similarity
    sim = combined_similarity(end_prev, [start_next], tower_weights, tower_stats)

    # Tempo compatibility penalty
    tempo_compat = compute_tempo_compatibility(
        prev_track['bpm_info'],
        next_track['bpm_info']
    )

    return sim * tempo_compat
```

### Half/Double Tempo Compatibility

```python
def compute_tempo_compatibility(bpm1_info, bpm2_info):
    """
    Returns 1.0 for compatible tempos, penalized for mismatches.
    Accounts for half/double tempo relationships.
    """
    bpm1 = bpm1_info['primary_bpm']
    bpm2 = bpm2_info['primary_bpm']

    # Check direct match
    ratio = max(bpm1, bpm2) / min(bpm1, bpm2)

    # Also check half/double relationships
    ratios_to_check = [
        ratio,
        ratio / 2,  # Half tempo
        ratio * 2,  # Double tempo
    ]

    best_ratio = min(ratios_to_check, key=lambda r: abs(r - 1.0))

    # Penalty function: 1.0 at ratio=1, decreasing as ratio deviates
    if best_ratio <= 1.2:
        return 1.0
    elif best_ratio <= 1.5:
        return 0.9
    elif best_ratio <= 2.0:
        return 0.7
    else:
        return 0.5
```

---

## Tradeoffs & Decisions

### Decision 1: 20 MFCCs vs 13
- **Chosen**: 20 MFCCs
- **Reason**: Captures more musical detail, marginal compute increase
- **Risk**: Slightly more dimensions to normalize

### Decision 2: Chroma CQT vs STFT
- **Chosen**: CQT
- **Reason**: Better bass resolution, more accurate for music
- **Risk**: ~10% slower extraction

### Decision 3: Robust aggregation (median/IQR) vs mean/std
- **Chosen**: Median/IQR throughout
- **Reason**: Mean/std is sensitive to outlier beats; median/IQR is robust
- **Risk**: Slightly less interpretable

### Decision 4: Per-tower PCA vs global PCA
- **Chosen**: Per-tower PCA (optional)
- **Reason**: Preserves tower structure; global PCA would conflate them
- **Risk**: Need to track PCA params per tower

### Decision 5: Z-score calibration vs percentile calibration
- **Chosen**: Z-score
- **Reason**: Simpler, linear transformation preserves relative distances
- **Risk**: Assumes roughly normal distribution of random similarities

### Decision 6: Default weights (0.45/0.40/0.15)
- **Chosen**: Rhythm-heavy for genres like funk/afrobeat
- **Reason**: Current system under-weights rhythm; these genres need it
- **Risk**: May under-weight timbre for acoustic/classical
- **Mitigation**: Configurable per mode (dynamic/narrow/discover)

---

## Validation Metrics

### Required (must pass for success):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| sonic_flatness | >= 0.5 | (p90 - p10) / median of similarities |
| topK_gap | >= 0.15 | mean(topK) - mean(randomK) |
| within_artist_separation | > 0 | Same artist closer than random |
| transition_improvement | > 0 | end->start scores > baseline |

### Human Validation:

- Fela Kuti seed should pull afrobeat/funk, NOT slowcore
- Electronic seed should pull electronic, NOT acoustic folk
- Ballad seed should pull ballads, NOT uptempo dance

---

## File Structure

```
src/
├── features/
│   ├── beat3tower_extractor.py     # NEW: 3-tower feature extraction
│   ├── beat3tower_normalizer.py    # NEW: Per-tower normalization
│   └── artifacts.py                # MODIFY: Add 3-tower bundle support
├── similarity/
│   ├── beat3tower_similarity.py    # NEW: Calibrated multi-tower sim
│   └── sonic_variant.py            # MODIFY: Add 'beat3tower' variant
├── analyze/
│   └── artifact_builder.py         # MODIFY: Build 3-tower artifacts
└── playlist/
    └── constructor.py              # MODIFY: Use new transition scoring

scripts/
├── validate_sonic_quality.py       # NEW: Phase 0 validation harness
├── build_beat3tower_artifacts.py   # NEW: Build 3-tower artifacts
└── tune_tower_weights.py           # NEW: Phase 5 weight tuning
```

---

## Configuration

```yaml
# config.yaml
sonic:
  variant: 'beat3tower'  # or 'legacy' for old behavior

  beat3tower:
    # Extraction
    n_mfcc: 20
    use_chroma_cqt: true

    # Tower weights (default for dynamic mode)
    weights:
      rhythm: 0.45
      timbre: 0.40
      harmony: 0.15

    # Normalization
    robust_clip_sigma: 3.0
    pca_variance_retain: 0.95
    use_pca_whitening: true

    # Transition
    tempo_penalty_threshold: 1.5
    use_segment_transitions: true
```

---

## Summary

**What we're building**:
- 3 specialized towers for rhythm, timbre, harmony
- Beat-synchronous extraction with robust aggregation
- Calibrated similarity that prevents any tower from dominating
- Segment-aware transition scoring (end->start)
- Behind feature flag for safe rollout

**Expected improvements**:
- Sonic similarity actually discriminates tracks
- Rhythm-heavy genres (funk, afrobeat) get proper neighbors
- Transitions feel intentional, not random
- Genre can return to filtering role (not driving everything)

**Total dimensions**: ~115-155 (before PCA), ~60-80 (after PCA at 95% variance)

---

## Next: Phased Implementation Plan

See `SONIC_3TOWER_IMPLEMENTATION_PHASES.md` for detailed phase breakdown.
