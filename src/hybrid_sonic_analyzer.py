"""
Sonic Analyzer - Local audio analysis using Librosa
"""
import logging
import json
from typing import Optional, Dict, Any
from .librosa_analyzer import LibrosaAnalyzer

logger = logging.getLogger(__name__)


class HybridSonicAnalyzer:
    """
    Sonic analyzer using Librosa for local audio feature extraction
    """

    def __init__(
        self,
        spotify_client_id: Optional[str] = None,
        spotify_client_secret: Optional[str] = None,
        use_beat_sync: bool = False,
        use_beat3tower: bool = False,
    ):
        """
        Initialize analyzer

        Args:
            spotify_client_id: Unused (kept for compatibility)
            spotify_client_secret: Unused (kept for compatibility)
            use_beat_sync: If True, use old beat-synchronized feature extraction
            use_beat3tower: If True, use 3-tower beat-synchronized extraction (recommended)
                           Takes precedence over use_beat_sync
        """
        self.librosa = LibrosaAnalyzer(use_beat_sync=use_beat_sync, use_beat3tower=use_beat3tower)

        # Determine mode for logging
        if use_beat3tower:
            mode = "beat3tower"
        elif use_beat_sync:
            mode = "beat-sync"
        else:
            mode = "windowed"

        logger.info(f"Initialized SonicAnalyzer (Librosa, mode={mode})")

    def analyze_track(self, file_path: str, artist: Optional[str] = None,
                     title: Optional[str] = None, spotify_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze a track using Librosa

        Args:
            file_path: Path to the audio file
            artist: Artist name (unused, kept for compatibility)
            title: Track title (unused, kept for compatibility)
            spotify_id: Spotify track ID (unused, kept for compatibility)

        Returns:
            Dictionary of audio features or None on error
        """
        # Analyze with Librosa
        logger.debug(f"Analyzing with Librosa: {file_path}")
        features = self.librosa.extract_similarity_features(file_path)

        if features:
            logger.info(f"Librosa features extracted from {file_path}")
            return features
        else:
            logger.error(f"Failed to analyze track: {file_path}")
            return None

    def features_to_json(self, features: Dict[str, Any]) -> str:
        """
        Convert features dictionary to JSON string for database storage

        Args:
            features: Features dictionary

        Returns:
            JSON string
        """
        return json.dumps(features)

    def json_to_features(self, json_str: str) -> Dict[str, Any]:
        """
        Convert JSON string back to features dictionary

        Args:
            json_str: JSON string from database

        Returns:
            Features dictionary
        """
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse features JSON: {e}")
            return {}

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about analysis sources

        Returns:
            Dictionary with counts of each source type
        """
        # This would be populated by tracking during analysis
        # For now, just return empty stats
        return {
            'acousticbrainz': 0,
            'librosa': 0,
            'failed': 0
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = HybridSonicAnalyzer()

    # Test with a file and optional MBID
    test_file = "path/to/test.flac"
    test_mbid = "8f3471b8-c8dd-4e63-b92d-c296b478a4c1"  # Optional

    features = analyzer.analyze_track(test_file, mbid=test_mbid)

    if features:
        logger.info(f"Analysis successful! Source: {features.get('source')}")
        logger.info(f"BPM: {features.get('bpm')}")
        logger.info(f"Key: {features.get('chords_key')}")

        # Convert to JSON for storage
        json_features = analyzer.features_to_json(features)
        logger.info(f"JSON (first 200 chars): {json_features[:200]}...")
    else:
        logger.error("Analysis failed")
