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

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Librosa analyzer

        Args:
            sample_rate: Target sample rate for analysis (22050 is Librosa default)
        """
        self.sample_rate = sample_rate
        logger.info(f"Initialized Librosa analyzer (sr={sample_rate})")

    def _extract_features_from_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract features from loaded audio data

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # 1. Timbre features (MFCCs - Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()

        # 3. Rhythm features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['bpm'] = float(tempo)

        # 4. Harmonic/tonal features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1).tolist()

        # Estimate key (simple approach using chroma)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features['estimated_key'] = keys[key_idx]

        # 5. Energy/dynamics
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # Zero-crossing rate (percussiveness indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))

        return features

    def extract_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract multi-segment audio features from a file

        Extracts features from 3 segments:
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

            # Initialize segment storage
            segments = {}
            segment_duration = 30  # 30 seconds per segment

            # Define segments based on track length
            if total_duration <= 60:
                # Short track: analyze whole thing as single segment
                y, sr = librosa.load(file_path, sr=self.sample_rate, duration=None)
                features = self._extract_features_from_audio(y, sr)
                # Use same features for all segments
                segments['beginning'] = features.copy()
                segments['middle'] = features.copy()
                segments['end'] = features.copy()
                segments['average'] = features.copy()
            else:
                # Extract beginning (0-30s)
                y_begin, sr = librosa.load(file_path, sr=self.sample_rate, offset=0, duration=segment_duration)
                segments['beginning'] = self._extract_features_from_audio(y_begin, sr)

                # Extract middle (center 30s)
                middle_offset = (total_duration - segment_duration) / 2
                y_middle, sr = librosa.load(file_path, sr=self.sample_rate, offset=middle_offset, duration=segment_duration)
                segments['middle'] = self._extract_features_from_audio(y_middle, sr)

                # Extract end (last 30s)
                end_offset = max(0, total_duration - segment_duration)
                y_end, sr = librosa.load(file_path, sr=self.sample_rate, offset=end_offset, duration=segment_duration)
                segments['end'] = self._extract_features_from_audio(y_end, sr)

                # Calculate average features across all segments
                segments['average'] = self._average_features([
                    segments['beginning'],
                    segments['middle'],
                    segments['end']
                ])

            # Add metadata
            segments['source'] = 'librosa'
            segments['segment_duration'] = segment_duration
            segments['total_duration'] = total_duration

            logger.debug(f"Extracted 3-segment features from {Path(file_path).name}")
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
        print("Successfully extracted features!")
        print(f"BPM: {features.get('bpm')}")
        print(f"Key: {features.get('chords_key')}")
    else:
        print("Failed to extract features")
