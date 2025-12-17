"""
Librosa Analyzer - Local audio feature extraction using Librosa
"""
import librosa
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LibrosaAnalyzer:
    """Extracts audio features locally using Librosa"""

    def __init__(self, sample_rate: int = 22050, use_beat_sync: bool = False):
        """
        Initialize Librosa analyzer

        Args:
            sample_rate: Target sample rate for analysis (22050 is Librosa default)
            use_beat_sync: If True, use beat-synchronized feature extraction (Phase 2)
                          If False, use fixed-window extraction (legacy)
        """
        self.sample_rate = sample_rate
        self.use_beat_sync = use_beat_sync
        mode = "beat-sync" if use_beat_sync else "windowed"
        logger.info(f"Initialized Librosa analyzer (sr={sample_rate}, mode={mode})")

    def _extract_beat_sync_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        PHASE 2: Extract features per beat using robust aggregation (median + IQR).

        This addresses the scale imbalance issue by extracting features aligned to
        actual musical beats rather than fixed time windows.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with beat-synchronized features (60+ dimensions)
        """
        features = {}

        try:
            # Detect beats (returns frames, not samples; default hop_length=512)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            hop_length = 512  # Default librosa hop_length for beat tracking

            # Minimum beats for meaningful aggregation
            if len(beat_frames) < 3:
                logger.debug(f"Insufficient beats detected ({len(beat_frames)}), using fallback")
                return self._extract_features_from_audio(y, sr)  # Fallback to windowed

            # Extract features per beat interval
            mfcc_per_beat = []
            chroma_per_beat = []
            spectral_contrast_per_beat = []

            for i in range(len(beat_frames) - 1):
                # Convert frames to samples
                start_frame = beat_frames[i]
                end_frame = beat_frames[i + 1]
                start_sample = start_frame * hop_length
                end_sample = end_frame * hop_length
                segment = y[start_sample:end_sample]

                if len(segment) < sr // 4:  # Skip very short beats
                    continue

                # MFCC per beat
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                mfcc_per_beat.append(mfcc.mean(axis=1))

                # Chroma per beat
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
                chroma_per_beat.append(chroma.mean(axis=1))

                # Spectral contrast per beat
                spec_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
                spectral_contrast_per_beat.append(spec_contrast.mean(axis=1))

            if len(mfcc_per_beat) < 2:
                logger.debug("Not enough beat intervals extracted, using fallback")
                return self._extract_features_from_audio(y, sr)  # Fallback

            # Aggregate using robust statistics (median + IQR)
            mfcc_per_beat = np.array(mfcc_per_beat)  # (n_beats, 13)
            mfcc_median = np.median(mfcc_per_beat, axis=0)
            mfcc_iqr = np.percentile(mfcc_per_beat, 75, axis=0) - np.percentile(mfcc_per_beat, 25, axis=0)

            chroma_per_beat = np.array(chroma_per_beat)  # (n_beats, 12)
            chroma_median = np.median(chroma_per_beat, axis=0)
            chroma_iqr = np.percentile(chroma_per_beat, 75, axis=0) - np.percentile(chroma_per_beat, 25, axis=0)

            spectral_contrast_per_beat = np.array(spectral_contrast_per_beat)  # (n_beats, 7)
            spec_contrast_median = np.median(spectral_contrast_per_beat, axis=0)
            spec_contrast_iqr = np.percentile(spectral_contrast_per_beat, 75, axis=0) - np.percentile(spectral_contrast_per_beat, 25, axis=0)

            # TIMBRE FEATURES: MFCC (Mel-Frequency Cepstral Coefficients)
            # 13 coefficients represent the "color" or texture of sound
            # Median aggregation is more robust to outliers than mean
            features['mfcc_median'] = mfcc_median.tolist()  # 13 dims: timbre central tendency
            features['mfcc_iqr'] = mfcc_iqr.tolist()  # 13 dims: timbre variability

            # HARMONIC FEATURES: Chroma (pitch class distribution)
            # 12 pitch classes (C, C#, D, ..., B) represent harmonic content
            features['chroma_median'] = chroma_median.tolist()  # 12 dims: harmonic central tendency
            features['chroma_iqr'] = chroma_iqr.tolist()  # 12 dims: harmonic variability

            # SPECTRAL CONTRAST: Peak-to-valley ratio in frequency bands
            # High contrast = bright/punchy; low contrast = smooth/blended
            features['spectral_contrast_median'] = spec_contrast_median.tolist()  # 7 dims: brightness central tendency
            features['spectral_contrast_iqr'] = spec_contrast_iqr.tolist()  # 7 dims: brightness variability

            # RHYTHM FEATURES: BPM (Beats Per Minute)
            # Represents tempo; important for transition compatibility
            features['bpm'] = float(tempo)  # 1 dim: tempo in BPM

            # RHYTHM DESCRIPTORS: Onset strength (percussiveness)
            # Measures attack sharpness and rhythmic clarity
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_env))  # 1 dim: average percussiveness
            features['onset_strength_std'] = float(np.std(onset_env))  # 1 dim: percussiveness variability

            # BRIGHTNESS FEATURE: Spectral Centroid
            # Frequency (Hz) at center of mass of spectrum
            # High value = bright (more high frequencies); low value = dark
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))  # 1 dim: brightness (Hz)
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))  # 1 dim: brightness variability

            # HIGH-FREQUENCY EXTENT: Spectral Rolloff
            # Frequency (Hz) where 85% of energy is below
            # Distinguishes clean vs. distorted sounds
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))  # 1 dim: high-freq extent (Hz)
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))  # 1 dim: high-freq extent variability

            # Mark as beat-sync
            features['extraction_method'] = 'beat_sync'

            return features

        except Exception as e:
            logger.debug(f"Beat-sync extraction failed ({e}), falling back to windowed")
            return self._extract_features_from_audio(y, sr)  # Fallback on any error

    def _extract_features_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract features from loaded audio data with PHASE 1 per-feature normalization.

        PHASE 1 FIX: Apply per-feature z-score normalization to prevent scale imbalance.
        Previously, BPM (~1800 scale) dominated L2 norm; now each feature group is
        normalized independently before concatenation.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of extracted features (will be z-scored during aggregation)
        """
        features = {}

        # TIMBRE FEATURES: MFCC (Mel-Frequency Cepstral Coefficients)
        # 13 coefficients represent the "color" or texture of sound (most important for similarity)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()  # 13 dims: average timbre
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()  # 13 dims: timbre variability

        # BRIGHTNESS FEATURE: Spectral Centroid
        # Frequency (Hz) at center of mass of spectrum (high = bright, low = dark)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))  # 1 dim: brightness (Hz)

        # HIGH-FREQUENCY EXTENT: Spectral Rolloff
        # Frequency (Hz) where 85% of energy is below
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))  # 1 dim: high-freq extent (Hz)

        # SPECTRAL BANDWIDTH: Width of spectrum
        # Broader bandwidth = more complex harmonic structure
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))  # 1 dim: spectral width (Hz)

        # SPECTRAL CONTRAST: Peak-to-valley ratio per frequency band
        # High contrast = bright/punchy; low contrast = smooth/blended (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()  # 7 dims: brightness profile (dB)

        # RHYTHM FEATURES: BPM (Beats Per Minute)
        # Tempo; important for transition compatibility and energy matching
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['bpm'] = float(tempo)  # 1 dim: tempo (BPM)

        # HARMONIC FEATURES: Chroma (pitch class distribution)
        # 12 pitch classes (C, C#, D, ..., B) represent harmonic/melodic content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()  # 12 dims: average pitch distribution

        # ESTIMATED KEY: Musical key inferred from chroma distribution
        # Informational only (not used in similarity calculation)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features['estimated_key'] = keys[key_idx]  # String: estimated musical key

        # ENERGY/DYNAMICS FEATURES: RMS (Root Mean Square) energy
        # Loudness/intensity of audio; captures volume envelope
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy'] = float(np.mean(rms))  # 1 dim: average loudness (amplitude)
        features['rms_std'] = float(np.std(rms))  # 1 dim: loudness variability

        # PERCUSSIVENESS INDICATOR: Zero-Crossing Rate
        # Rate of sign changes in amplitude; high ZCR = noisy/percussive, low ZCR = tonal
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))  # 1 dim: noisiness (crossings/sec)

        return features

    def extract_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract multi-segment audio features from a file

        PHASE 2 ROUTING: If use_beat_sync=True, routes to beat-synchronized extraction.
        Otherwise, extracts features from 3 segments:
        - beginning: first 30 seconds
        - middle: 30 seconds from center
        - end: last 30 seconds

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with segment features and averaged features, or None on error
        """
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Get total duration
            total_duration = librosa.get_duration(path=file_path)

            # Load full audio once
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=None)

            # PHASE 2 ROUTING: Use beat-sync if enabled
            if self.use_beat_sync:
                logger.debug(f"Extracting beat-sync features from {Path(file_path).name}")
                # Use beat-sync method for all segments
                beat_sync_features = self._extract_beat_sync_features(y, sr)

                # Create segment structure (all use beat-sync for now)
                segments = {
                    'beginning': beat_sync_features.copy(),
                    'middle': beat_sync_features.copy(),
                    'end': beat_sync_features.copy(),
                    'average': beat_sync_features.copy(),
                }
            else:
                # Original windowed approach
                logger.debug(f"Extracting windowed features from {Path(file_path).name}")

                # Initialize segment storage
                segments = {}
                segment_duration = 30  # 30 seconds per segment

                # Define segments based on track length
                if total_duration <= 60:
                    # Short track: analyze whole thing as single segment
                    features = self._extract_features_from_audio(y, sr)
                    # Use same features for all segments
                    segments['beginning'] = features.copy()
                    segments['middle'] = features.copy()
                    segments['end'] = features.copy()
                    segments['average'] = features.copy()
                else:
                    # Extract beginning (0-30s)
                    y_begin, sr_begin = librosa.load(file_path, sr=self.sample_rate, offset=0, duration=segment_duration)
                    segments['beginning'] = self._extract_features_from_audio(y_begin, sr_begin)

                    # Extract middle (center 30s)
                    middle_offset = (total_duration - segment_duration) / 2
                    y_middle, sr_mid = librosa.load(file_path, sr=self.sample_rate, offset=middle_offset, duration=segment_duration)
                    segments['middle'] = self._extract_features_from_audio(y_middle, sr_mid)

                    # Extract end (last 30s)
                    end_offset = max(0, total_duration - segment_duration)
                    y_end, sr_end = librosa.load(file_path, sr=self.sample_rate, offset=end_offset, duration=segment_duration)
                    segments['end'] = self._extract_features_from_audio(y_end, sr_end)

                    # Calculate average features across all segments with PHASE 1 normalization
                    segments['average'] = self._average_features([
                        segments['beginning'],
                        segments['middle'],
                        segments['end']
                    ])

            # Add metadata
            segments['source'] = 'librosa'
            segments['extraction_method'] = 'beat_sync' if self.use_beat_sync else 'windowed'
            segments['segment_duration'] = 30
            segments['total_duration'] = total_duration

            logger.debug(f"Extracted features from {Path(file_path).name} via {segments['extraction_method']}")
            return segments

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None

    def _average_features(self, feature_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Average features across multiple segments

        Args:
            feature_list: List of feature dictionaries

        Returns:
            Dictionary with averaged features
        """
        averaged = {}

        # Get all keys from first feature set
        keys = feature_list[0].keys()

        for key in keys:
            values = [f[key] for f in feature_list]

            # Handle different data types
            if isinstance(values[0], list):
                # Average each element in the list
                averaged[key] = [
                    sum(v[i] for v in values) / len(values)
                    for i in range(len(values[0]))
                ]
            elif isinstance(values[0], (int, float)):
                # Simple average
                averaged[key] = sum(values) / len(values)
            else:
                # For strings (like estimated_key), use most common
                from collections import Counter
                averaged[key] = Counter(values).most_common(1)[0][0]

        return averaged

    def extract_similarity_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract multi-segment features optimized for similarity comparison

        Returns a structure with:
        - beginning, middle, end: segment-specific features
        - average: averaged features for general similarity matching
        - metadata: source, durations, etc.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with segmented features or None on error
        """
        # extract_features() now returns the full multi-segment structure
        # with 'beginning', 'middle', 'end', 'average', and metadata
        return self.extract_features(file_path)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = LibrosaAnalyzer()

    # Test with a file (you'll need to provide a valid path)
    test_file = "path/to/test.flac"
    features = analyzer.extract_similarity_features(test_file)

    if features:
        logger.info("Successfully extracted features!")
        logger.info(f"BPM: {features.get('bpm')}")
        logger.info(f"Key: {features.get('chords_key')}")
    else:
        logger.error("Failed to extract features")
