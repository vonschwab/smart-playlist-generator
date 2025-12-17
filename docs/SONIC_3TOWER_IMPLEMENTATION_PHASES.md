# 3-Tower Sonic Similarity: Phased Implementation Plan

## Overview

This document details the phased implementation of the 3-tower beat-synchronous sonic similarity system. Each phase is self-contained with clear acceptance criteria.

---

## Phase 0: Validation Infrastructure (PREREQ)

**Goal**: Confirm beat-sync extraction is working, build acceptance test harness.

**Duration**: 2-3 days

### Tasks

#### 0.1 Sanity Check Current Beat-Sync Extraction

Create `scripts/validate_sonic_quality.py`:

```python
"""
Validate current beat-sync sonic features are non-collapsed.

Checks:
1. Per-dimension variance across library
2. % unique vectors (hash of rounded vectors)
3. Random-pair vs topK cosine distribution
4. Within-artist vs random separation
"""
```

**Metrics to compute**:

| Metric | Formula | Pass Threshold |
|--------|---------|----------------|
| dim_variance_min | min(var(X, axis=0)) | > 1e-6 |
| dim_variance_median | median(var(X, axis=0)) | > 1e-3 |
| unique_vector_pct | len(unique(hash(round(X, 4)))) / N | > 95% |
| cosine_random_mean | mean(cosine(random_pairs)) | < 0.98 |
| cosine_random_std | std(cosine(random_pairs)) | > 0.02 |
| topK_gap | mean(topK) - mean(random) | > 0.05 (warning if < 0.15) |

#### 0.2 Build Acceptance Test Harness

The validation script should output JSON + markdown report:

```python
def run_validation_suite(artifact_path, seed_track_ids, k=30):
    """
    Full validation suite for sonic similarity quality.

    Returns:
        ValidationReport with metrics and pass/fail status
    """
    bundle = load_artifact_bundle(artifact_path)

    results = {
        'timestamp': datetime.now().isoformat(),
        'artifact_path': str(artifact_path),
        'n_tracks': len(bundle.track_ids),
        'n_seeds_tested': len(seed_track_ids),
        'metrics': {},
        'per_seed': {},
    }

    # Global metrics
    results['metrics']['dimension_variance'] = compute_dim_variance(bundle.X_sonic)
    results['metrics']['unique_vectors'] = compute_unique_vectors(bundle.X_sonic)
    results['metrics']['random_similarity_dist'] = compute_random_sim_dist(bundle.X_sonic, n_pairs=10000)

    # Per-seed metrics
    for seed_id in seed_track_ids:
        seed_results = {}
        seed_idx = bundle.track_id_to_index[seed_id]

        # Compute all similarities to seed
        sims = cosine_sim_all(bundle.X_sonic, seed_idx)

        seed_results['sonic_flatness'] = compute_flatness(sims)
        seed_results['topK_gap'] = compute_topk_gap(sims, k=k)
        seed_results['within_artist_coherence'] = compute_within_artist(
            bundle.X_sonic, bundle.artist_keys, seed_idx
        )
        seed_results['topK_tracks'] = get_topk_metadata(bundle, sims, k=10)

        results['per_seed'][seed_id] = seed_results

    # Aggregate pass/fail
    results['passed'] = evaluate_pass_fail(results)

    return results
```

#### 0.3 Test on Known Seeds

Test with diverse seed tracks:

```python
TEST_SEEDS = {
    'funk_afrobeat': '<fela_kuti_track_id>',
    'electronic_dance': '<electronic_track_id>',
    'slow_ballad': '<ballad_track_id>',
    'rock_guitar': '<rock_track_id>',
    'jazz_complex': '<jazz_track_id>',
}
```

### Deliverables

1. `scripts/validate_sonic_quality.py` - Full validation script
2. `diagnostics/sonic_validation/baseline_report.json` - Baseline metrics
3. `diagnostics/sonic_validation/baseline_report.md` - Human-readable report

### Acceptance Criteria

- [ ] Script runs without errors on current artifact
- [ ] Baseline metrics captured for comparison
- [ ] At least 3 seed tracks tested
- [ ] Report clearly identifies current weaknesses

---

## Phase 1: 3-Tower Feature Extraction

**Goal**: Implement beat-synchronous 3-tower feature extractor.

**Duration**: 5-7 days

### Tasks

#### 1.1 Create Beat3Tower Extractor Module

File: `src/features/beat3tower_extractor.py`

```python
"""
Beat-synchronous 3-tower feature extraction.

Extracts:
- Rhythm tower: onset, tempogram, beat stability
- Timbre tower: MFCC, spectral features
- Harmony tower: chroma, tonnetz
"""

class Beat3TowerExtractor:
    def __init__(self, config: Beat3TowerConfig):
        self.config = config
        self.sr = config.sample_rate or 22050
        self.n_mfcc = config.n_mfcc or 20

    def extract_full(self, y: np.ndarray) -> Beat3TowerFeatures:
        """Extract all 3 towers from audio."""
        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        if len(beat_frames) < 4:
            raise InsufficientBeatsError(f"Only {len(beat_frames)} beats detected")

        # Extract each tower
        rhythm = self._extract_rhythm_tower(y, beat_frames, beat_times, tempo)
        timbre = self._extract_timbre_tower(y, beat_frames)
        harmony = self._extract_harmony_tower(y, beat_frames)

        return Beat3TowerFeatures(
            rhythm=rhythm,
            timbre=timbre,
            harmony=harmony,
            bpm_info=self._compute_bpm_info(tempo, beat_times),
            n_beats=len(beat_frames),
        )

    def extract_segment(self, y: np.ndarray, segment: str) -> Beat3TowerFeatures:
        """Extract 3-tower features for a specific segment."""
        y_seg = self._get_segment(y, segment)
        return self.extract_full(y_seg)
```

#### 1.2 Implement Rhythm Tower

```python
def _extract_rhythm_tower(self, y, beat_frames, beat_times, tempo):
    """
    Rhythm tower features (~25-35 dims):
    - Onset envelope stats
    - Tempogram peaks
    - Beat stability
    - Rhythmic complexity
    """
    features = {}

    # 1. Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
    onset_per_beat = self._aggregate_per_beat(onset_env, beat_frames)
    features['onset_median'] = np.median(onset_per_beat)
    features['onset_iqr'] = np.percentile(onset_per_beat, 75) - np.percentile(onset_per_beat, 25)
    features['onset_std'] = np.std(onset_per_beat)
    features['onset_p10'] = np.percentile(onset_per_beat, 10)
    features['onset_p90'] = np.percentile(onset_per_beat, 90)

    # 2. Tempogram / rhythmic autocorrelation
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
    tempo_acf = np.mean(tempogram, axis=1)  # Average across time

    # Peak positions in tempogram (dominant rhythmic periods)
    peak_indices = self._find_tempo_peaks(tempo_acf, n_peaks=3)
    features['tempo_peak1_lag'] = peak_indices[0] if len(peak_indices) > 0 else 0
    features['tempo_peak2_lag'] = peak_indices[1] if len(peak_indices) > 1 else 0
    features['tempo_peak_ratio'] = (tempo_acf[peak_indices[0]] / tempo_acf[peak_indices[1]]
                                    if len(peak_indices) > 1 else 1.0)

    # ACF lags for groove pattern
    for i, lag in enumerate([1, 2, 3, 4, 5]):
        features[f'tempo_acf_lag{lag}'] = tempo_acf[lag] if lag < len(tempo_acf) else 0

    # 3. Beat interval stability
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        features['beat_interval_cv'] = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)
        features['beat_interval_median'] = np.median(beat_intervals)
    else:
        features['beat_interval_cv'] = 0
        features['beat_interval_median'] = 0

    # 4. Beat strength
    beat_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else [0]
    features['beat_strength_median'] = np.median(beat_strengths)
    features['beat_strength_iqr'] = np.percentile(beat_strengths, 75) - np.percentile(beat_strengths, 25)

    # 5. Onset rate (busyness)
    features['onset_rate'] = librosa.onset.onset_detect(y=y, sr=self.sr, units='time').shape[0] / (len(y) / self.sr)

    # 6. Rhythmic complexity (onset entropy)
    onset_probs = onset_per_beat / (np.sum(onset_per_beat) + 1e-6)
    features['rhythm_entropy'] = -np.sum(onset_probs * np.log(onset_probs + 1e-10))

    return RhythmTowerFeatures(**features)
```

#### 1.3 Implement Timbre Tower

```python
def _extract_timbre_tower(self, y, beat_frames):
    """
    Timbre tower features (~60-80 dims):
    - MFCC (20 coef) with median/IQR
    - Spectral contrast, rolloff, flux
    """
    features = {}
    hop_length = 512

    # 1. MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, hop_length=hop_length)
    mfcc_per_beat = self._aggregate_feature_per_beat(mfcc, beat_frames)

    for i in range(self.n_mfcc):
        coef_values = mfcc_per_beat[:, i]
        features[f'mfcc{i:02d}_median'] = np.median(coef_values)
        features[f'mfcc{i:02d}_iqr'] = np.percentile(coef_values, 75) - np.percentile(coef_values, 25)

    # 2. MFCC delta (dynamics)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_per_beat = self._aggregate_feature_per_beat(mfcc_delta, beat_frames)
    for i in range(self.n_mfcc):
        features[f'mfcc_delta{i:02d}_median'] = np.median(mfcc_delta_per_beat[:, i])

    # 3. Spectral contrast (7 bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, hop_length=hop_length)
    spec_contrast_per_beat = self._aggregate_feature_per_beat(spec_contrast, beat_frames)
    for i in range(7):
        features[f'spec_contrast{i}_median'] = np.median(spec_contrast_per_beat[:, i])
        features[f'spec_contrast{i}_iqr'] = np.percentile(spec_contrast_per_beat[:, i], 75) - \
                                             np.percentile(spec_contrast_per_beat[:, i], 25)

    # 4. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=hop_length)[0]
    rolloff_per_beat = self._aggregate_per_beat(rolloff, beat_frames)
    features['spec_rolloff_median'] = np.median(rolloff_per_beat)
    features['spec_rolloff_iqr'] = np.percentile(rolloff_per_beat, 75) - np.percentile(rolloff_per_beat, 25)

    # 5. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=hop_length)[0]
    centroid_per_beat = self._aggregate_per_beat(centroid, beat_frames)
    features['spec_centroid_median'] = np.median(centroid_per_beat)
    features['spec_centroid_iqr'] = np.percentile(centroid_per_beat, 75) - np.percentile(centroid_per_beat, 25)

    # 6. Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, hop_length=hop_length)[0]
    bandwidth_per_beat = self._aggregate_per_beat(bandwidth, beat_frames)
    features['spec_bandwidth_median'] = np.median(bandwidth_per_beat)
    features['spec_bandwidth_iqr'] = np.percentile(bandwidth_per_beat, 75) - np.percentile(bandwidth_per_beat, 25)

    # 7. Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=self.sr)  # Approximation
    flux_per_beat = self._aggregate_per_beat(flux, beat_frames)
    features['spec_flux_median'] = np.median(flux_per_beat)
    features['spec_flux_iqr'] = np.percentile(flux_per_beat, 75) - np.percentile(flux_per_beat, 25)

    # 8. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr_per_beat = self._aggregate_per_beat(zcr, beat_frames)
    features['zcr_median'] = np.median(zcr_per_beat)

    return TimbreTowerFeatures(**features)
```

#### 1.4 Implement Harmony Tower

```python
def _extract_harmony_tower(self, y, beat_frames):
    """
    Harmony tower features (~30-40 dims):
    - Chroma CQT (12 bins) with median/IQR
    - Chroma entropy
    - Tonnetz
    """
    features = {}
    hop_length = 512

    # 1. Chroma CQT (constant-Q transform - better for music)
    chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=hop_length)
    chroma_per_beat = self._aggregate_feature_per_beat(chroma, beat_frames)

    for i in range(12):
        features[f'chroma{i:02d}_median'] = np.median(chroma_per_beat[:, i])
        features[f'chroma{i:02d}_iqr'] = np.percentile(chroma_per_beat[:, i], 75) - \
                                          np.percentile(chroma_per_beat[:, i], 25)

    # 2. Chroma entropy (how spread the pitch classes are)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_probs = chroma_mean / (np.sum(chroma_mean) + 1e-6)
    features['chroma_entropy'] = -np.sum(chroma_probs * np.log(chroma_probs + 1e-10))

    # 3. Chroma peak count (how many dominant pitch classes)
    threshold = np.mean(chroma_mean) + np.std(chroma_mean)
    features['chroma_peak_count'] = np.sum(chroma_mean > threshold)

    # 4. Key strength (how key-centric)
    features['key_strength'] = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-6)

    # 5. Tonnetz (tonal centroid - 6 dimensions)
    tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
    tonnetz_per_beat = self._aggregate_feature_per_beat(tonnetz, beat_frames)
    for i in range(6):
        features[f'tonnetz{i}_median'] = np.median(tonnetz_per_beat[:, i])

    return HarmonyTowerFeatures(**features)
```

#### 1.5 BPM Half/Double Awareness

```python
def _compute_bpm_info(self, tempo, beat_times):
    """
    Compute BPM with half/double tempo awareness.
    """
    if len(beat_times) < 4:
        return BPMInfo(primary_bpm=tempo, tempo_stability=0.0)

    # Analyze tempo histogram from beat intervals
    beat_intervals = np.diff(beat_times)
    interval_bpms = 60.0 / beat_intervals

    # Check for half-tempo evidence (long intervals)
    half_tempo = tempo / 2
    double_tempo = tempo * 2

    # Count intervals closer to each interpretation
    close_to_primary = np.sum(np.abs(interval_bpms - tempo) < 10)
    close_to_half = np.sum(np.abs(interval_bpms - half_tempo) < 10)
    close_to_double = np.sum(np.abs(interval_bpms - double_tempo) < 10)

    total = len(interval_bpms)
    half_likely = close_to_half / total > 0.3
    double_likely = close_to_double / total > 0.3

    # Tempo stability
    tempo_stability = 1.0 - (np.std(interval_bpms) / (np.mean(interval_bpms) + 1e-6))
    tempo_stability = np.clip(tempo_stability, 0, 1)

    return BPMInfo(
        primary_bpm=tempo,
        half_tempo_likely=half_likely,
        double_tempo_likely=double_likely,
        tempo_stability=tempo_stability,
    )
```

### Deliverables

1. `src/features/beat3tower_extractor.py` - Complete extractor
2. `src/features/beat3tower_types.py` - Data classes for features
3. Unit tests in `tests/unit/test_beat3tower_extractor.py`
4. Integration test with real audio files

### Acceptance Criteria

- [ ] Extractor runs on sample tracks without errors
- [ ] All 3 towers produce non-zero features
- [ ] Feature count matches spec (~115-155 dims total)
- [ ] Segment extraction (start/mid/end) works
- [ ] BPM info includes half/double awareness

---

## Phase 2: Normalization & Artifact Building

**Goal**: Build normalized 3-tower artifacts with per-tower scaling.

**Duration**: 3-4 days

### Tasks

#### 2.1 Implement Per-Tower Normalizer

File: `src/features/beat3tower_normalizer.py`

```python
class Beat3TowerNormalizer:
    """Normalizes 3-tower features with robust statistics."""

    def __init__(self, config: NormalizerConfig):
        self.config = config
        self.tower_stats = {}

    def fit(self, X_rhythm, X_timbre, X_harmony):
        """Compute normalization statistics from training data."""
        self.tower_stats['rhythm'] = self._fit_tower(X_rhythm, 'rhythm')
        self.tower_stats['timbre'] = self._fit_tower(X_timbre, 'timbre')
        self.tower_stats['harmony'] = self._fit_tower(X_harmony, 'harmony')

    def transform(self, X_rhythm, X_timbre, X_harmony):
        """Apply normalization to feature matrices."""
        X_r_norm = self._transform_tower(X_rhythm, 'rhythm')
        X_t_norm = self._transform_tower(X_timbre, 'timbre')
        X_h_norm = self._transform_tower(X_harmony, 'harmony')

        return X_r_norm, X_t_norm, X_h_norm

    def _fit_tower(self, X, tower_name):
        """Fit robust statistics for a single tower."""
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        iqr = np.maximum(iqr, 1e-6)

        stats = {'median': median, 'iqr': iqr}

        # Optional PCA whitening
        if self.config.use_pca_whitening:
            X_robust = (X - median) / (iqr / 1.35)
            X_clipped = np.clip(X_robust, -self.config.clip_sigma, self.config.clip_sigma)

            pca = PCA(n_components=self.config.pca_variance_retain, whiten=True)
            pca.fit(X_clipped)
            stats['pca'] = pca

        return stats

    def _transform_tower(self, X, tower_name):
        """Transform a single tower."""
        stats = self.tower_stats[tower_name]

        # Robust standardization
        X_robust = (X - stats['median']) / (stats['iqr'] / 1.35)
        X_clipped = np.clip(X_robust, -self.config.clip_sigma, self.config.clip_sigma)

        # Optional PCA
        if 'pca' in stats:
            X_pca = stats['pca'].transform(X_clipped)
            return X_pca

        return X_clipped
```

#### 2.2 Update Artifact Builder

Modify `src/analyze/artifact_builder.py`:

```python
def build_beat3tower_artifacts(
    db_path: str,
    config_path: str,
    out_path: Path,
    max_tracks: int = 0,
):
    """Build artifacts with 3-tower sonic embeddings."""

    # ... load tracks with sonic_features ...

    # Separate towers
    X_rhythm = []
    X_timbre = []
    X_harmony = []

    for track in tracks:
        features = parse_beat3tower_features(track['sonic_features'])
        X_rhythm.append(features.rhythm.to_vector())
        X_timbre.append(features.timbre.to_vector())
        X_harmony.append(features.harmony.to_vector())

    X_rhythm = np.vstack(X_rhythm)
    X_timbre = np.vstack(X_timbre)
    X_harmony = np.vstack(X_harmony)

    # Normalize
    normalizer = Beat3TowerNormalizer(config)
    normalizer.fit(X_rhythm, X_timbre, X_harmony)
    X_r_norm, X_t_norm, X_h_norm = normalizer.transform(X_rhythm, X_timbre, X_harmony)

    # L2 normalize each tower
    X_r_norm = l2_normalize(X_r_norm)
    X_t_norm = l2_normalize(X_t_norm)
    X_h_norm = l2_normalize(X_h_norm)

    # Compute calibration statistics
    tower_calibration = compute_tower_calibration_stats(X_r_norm, X_t_norm, X_h_norm)

    # Save artifact
    np.savez(
        out_path,
        # Tower embeddings
        X_sonic_rhythm=X_r_norm,
        X_sonic_timbre=X_t_norm,
        X_sonic_harmony=X_h_norm,
        # Concatenated (for backward compat)
        X_sonic=np.hstack([X_r_norm, X_t_norm, X_h_norm]),
        # Segment embeddings
        X_sonic_rhythm_start=X_r_start_norm,
        X_sonic_rhythm_end=X_r_end_norm,
        # ... etc for other towers ...
        # Calibration
        tower_calibration=tower_calibration,
        # Normalizer params (for reproducibility)
        normalizer_params=normalizer.get_params(),
        # Metadata
        tower_dims={'rhythm': X_r_norm.shape[1], 'timbre': X_t_norm.shape[1], 'harmony': X_h_norm.shape[1]},
        # ... existing fields ...
    )
```

### Deliverables

1. `src/features/beat3tower_normalizer.py` - Normalizer with PCA
2. Updated `src/analyze/artifact_builder.py` - 3-tower support
3. `scripts/build_beat3tower_artifacts.py` - Artifact build script
4. Serialization format documented

### Acceptance Criteria

- [ ] Artifacts contain per-tower matrices
- [ ] Normalization params stored for reproducibility
- [ ] Calibration stats computed and stored
- [ ] Artifact loads successfully with `load_artifact_bundle()`

---

## Phase 3: Calibrated Multi-Tower Similarity

**Goal**: Implement calibrated per-tower similarity with configurable weights.

**Duration**: 3-4 days

### Tasks

#### 3.1 Implement Calibrated Similarity

File: `src/similarity/beat3tower_similarity.py`

```python
class Beat3TowerSimilarity:
    """Calibrated multi-tower similarity computation."""

    def __init__(self, bundle: ArtifactBundle, config: SimilarityConfig):
        self.bundle = bundle
        self.config = config
        self.calibration = bundle.tower_calibration
        self.weights = config.tower_weights

    def compute_similarity(self, query_idx: int, candidate_indices: np.ndarray) -> np.ndarray:
        """Compute calibrated multi-tower similarity."""
        total_sim = np.zeros(len(candidate_indices))

        for tower_name in ['rhythm', 'timbre', 'harmony']:
            # Get tower matrices
            X_tower = getattr(self.bundle, f'X_sonic_{tower_name}')

            # Raw cosine similarity
            query_vec = X_tower[query_idx]
            cand_vecs = X_tower[candidate_indices]
            raw_sim = cand_vecs @ query_vec

            # Calibrate to z-score
            calibrated = self._calibrate(raw_sim, tower_name)

            # Weight and accumulate
            total_sim += self.weights[tower_name] * calibrated

        return total_sim

    def _calibrate(self, raw_sim, tower_name):
        """Convert raw similarity to calibrated z-score."""
        stats = self.calibration[tower_name]
        z = (raw_sim - stats['random_mean']) / (stats['random_std'] + 1e-6)
        return z

    def compute_transition_similarity(self, prev_idx: int, next_idx: int) -> float:
        """Compute transition similarity (end(prev) -> start(next))."""
        total_sim = 0.0

        for tower_name in ['rhythm', 'timbre', 'harmony']:
            X_end = getattr(self.bundle, f'X_sonic_{tower_name}_end')
            X_start = getattr(self.bundle, f'X_sonic_{tower_name}_start')

            end_vec = X_end[prev_idx]
            start_vec = X_start[next_idx]
            raw_sim = float(np.dot(end_vec, start_vec))

            calibrated = self._calibrate(raw_sim, tower_name)
            total_sim += self.weights[tower_name] * calibrated

        # Tempo compatibility
        tempo_compat = self._compute_tempo_compatibility(prev_idx, next_idx)

        return total_sim * tempo_compat

    def _compute_tempo_compatibility(self, prev_idx, next_idx):
        """Compute tempo compatibility with half/double awareness."""
        bpm1 = self.bundle.bpm_array[prev_idx]
        bpm2 = self.bundle.bpm_array[next_idx]

        if bpm1 <= 0 or bpm2 <= 0:
            return 1.0

        ratio = max(bpm1, bpm2) / min(bpm1, bpm2)

        # Check half/double
        best_ratio = min([ratio, ratio/2, ratio*2], key=lambda r: abs(r - 1.0))

        if best_ratio <= 1.2:
            return 1.0
        elif best_ratio <= 1.5:
            return 0.9
        elif best_ratio <= 2.0:
            return 0.7
        else:
            return 0.5
```

#### 3.2 Register as Sonic Variant

Update `src/similarity/sonic_variant.py`:

```python
VARIANT_FUNCS = {
    'raw': compute_sonic_variant_norm,
    'centered': compute_sonic_variant_centered,
    'z': compute_sonic_variant_z,
    'beat3tower': compute_sonic_variant_beat3tower,  # NEW
}

def compute_sonic_variant_beat3tower(bundle, config):
    """3-tower calibrated similarity."""
    sim_engine = Beat3TowerSimilarity(bundle, config)
    return sim_engine
```

### Deliverables

1. `src/similarity/beat3tower_similarity.py` - Similarity engine
2. Updated `src/similarity/sonic_variant.py` - Variant registration
3. Tests for calibration and weighting

### Acceptance Criteria

- [ ] Per-tower similarities computed correctly
- [ ] Calibration produces ~N(0,1) for random pairs
- [ ] Weights sum to 1.0 and are configurable
- [ ] Transition similarity uses end->start segments

---

## Phase 4: Transition Scoring Integration

**Goal**: Integrate 3-tower transitions into playlist constructor.

**Duration**: 2-3 days

### Tasks

#### 4.1 Update Playlist Constructor

Modify `src/playlist/constructor.py`:

```python
def _compute_local_sim_beat3tower(
    prev_idx: int,
    cand_indices: np.ndarray,
    sim_engine: Beat3TowerSimilarity,
    transition_gamma: float,
) -> np.ndarray:
    """
    Compute local similarity using 3-tower transitions.
    """
    # Seed similarity (full embedding)
    seed_sim = sim_engine.compute_similarity(prev_idx, cand_indices)

    # Transition similarity (end->start)
    trans_sim = np.array([
        sim_engine.compute_transition_similarity(prev_idx, cand_idx)
        for cand_idx in cand_indices
    ])

    # Blend
    return transition_gamma * trans_sim + (1 - transition_gamma) * seed_sim
```

#### 4.2 Add Tempo Penalty Config

```yaml
# config.yaml
playlist:
  transition:
    tempo_penalty:
      enabled: true
      max_ratio: 1.5  # Above this, apply penalty
      half_double_aware: true
```

### Deliverables

1. Updated `src/playlist/constructor.py` - 3-tower transition support
2. Config options for tempo penalty
3. Integration tests

### Acceptance Criteria

- [ ] Transitions use end->start segment similarity
- [ ] Tempo penalty applied correctly
- [ ] Backward compatible with legacy variant

---

## Phase 5: Weight Tuning (Optional)

**Goal**: Learn optimal tower weights from weak supervision.

**Duration**: 3-5 days (optional)

### Tasks

#### 5.1 Prepare Training Data

```python
def prepare_weak_labels(bundle):
    """
    Create weak supervision pairs:
    - Positive: same artist (expect high similarity)
    - Semi-positive: same broad genre (expect medium similarity)
    - Negative: random (expect low similarity)
    """
    positives = []  # (idx_a, idx_b, label=1.0)
    semi_positives = []  # (idx_a, idx_b, label=0.5)
    negatives = []  # (idx_a, idx_b, label=0.0)

    # Same artist pairs
    artist_groups = group_by_artist(bundle.artist_keys)
    for artist, indices in artist_groups.items():
        if len(indices) >= 2:
            for i in range(min(5, len(indices))):
                for j in range(i+1, min(6, len(indices))):
                    positives.append((indices[i], indices[j], 1.0))

    # Random pairs (negative)
    for _ in range(len(positives) * 2):
        i, j = random_pair(len(bundle.track_ids))
        negatives.append((i, j, 0.0))

    return positives + semi_positives + negatives
```

#### 5.2 Train Simple Linear Model

```python
def tune_tower_weights(bundle, pairs):
    """
    Learn weights using logistic regression on per-tower similarities.
    """
    X = []  # Per-tower similarity features
    y = []  # Labels

    for idx_a, idx_b, label in pairs:
        sim_r = cosine(bundle.X_sonic_rhythm[idx_a], bundle.X_sonic_rhythm[idx_b])
        sim_t = cosine(bundle.X_sonic_timbre[idx_a], bundle.X_sonic_timbre[idx_b])
        sim_h = cosine(bundle.X_sonic_harmony[idx_a], bundle.X_sonic_harmony[idx_b])

        X.append([sim_r, sim_t, sim_h])
        y.append(label)

    # Logistic regression (coefficients become weights)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(fit_intercept=False)
    model.fit(X, y)

    # Normalize coefficients to sum to 1
    weights = model.coef_[0]
    weights = np.abs(weights) / np.sum(np.abs(weights))

    return {
        'rhythm': weights[0],
        'timbre': weights[1],
        'harmony': weights[2],
    }
```

### Deliverables

1. `scripts/tune_tower_weights.py` - Weight tuning script
2. Learned weights stored in config/artifact
3. Documentation of tuning process

### Acceptance Criteria

- [ ] Tuned weights improve validation metrics
- [ ] Weights are interpretable (no extreme values)
- [ ] Process is reproducible

---

## Post-Implementation: Commands

### After Scan Completes

```bash
# 1. Validate current state (Phase 0)
python scripts/validate_sonic_quality.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds "fela_kuti_id,electronic_id,ballad_id" \
    --output diagnostics/sonic_validation/baseline/

# 2. Extract 3-tower features (requires re-scan if not already extracted)
python scripts/update_sonic.py --beat-sync --beat3tower --workers 8

# 3. Build 3-tower artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz

# 4. Validate 3-tower quality
python scripts/validate_sonic_quality.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz \
    --seeds "fela_kuti_id,electronic_id,ballad_id" \
    --output diagnostics/sonic_validation/beat3tower/

# 5. Compare baseline vs beat3tower
python scripts/compare_sonic_variants.py \
    --baseline diagnostics/sonic_validation/baseline/ \
    --new diagnostics/sonic_validation/beat3tower/ \
    --output diagnostics/sonic_validation/comparison.md

# 6. (Optional) Tune weights
python scripts/tune_tower_weights.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz \
    --output config/tuned_tower_weights.yaml
```

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 2-3 days | None (can start now) |
| Phase 1 | 5-7 days | Phase 0 complete |
| Phase 2 | 3-4 days | Phase 1 complete |
| Phase 3 | 3-4 days | Phase 2 complete |
| Phase 4 | 2-3 days | Phase 3 complete |
| Phase 5 | 3-5 days | Phase 4 complete (optional) |

**Total**: ~19-26 days (excluding Phase 5)

---

## Risk Mitigation

1. **Beat detection fails**: Fallback to windowed extraction
2. **PCA loses too much variance**: Make PCA optional, default to robust standardization only
3. **Calibration doesn't help**: Try percentile calibration instead of z-score
4. **Weights don't tune well**: Use hand-tuned defaults (0.45/0.40/0.15)
5. **Runtime too slow**: Cache per-beat features in DB, parallelize extraction

---

## Success Criteria Recap

| Metric | Baseline | Target |
|--------|----------|--------|
| sonic_flatness | ~0.05 | >= 0.5 |
| topK_gap | ~0.02 | >= 0.15 |
| within_artist_sep | ~0 | > 0.05 |
| transition_p10 | TBD | > baseline |

**Human sniff test**:
- Fela/afrobeat → pulls funk, afrobeat, NOT slowcore
- Electronic → pulls electronic, NOT acoustic folk
- Ballad → pulls ballads, NOT uptempo dance
