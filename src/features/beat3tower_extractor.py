"""
Beat 3-Tower Feature Extractor
===============================

SP-C (2026-07-07): the sonic space is MuQ, so this extractor no longer computes
the timbre/harmony towers (deleted as dead DSP). It now runs only to produce the
pace fields the BPM/onset gate reads -- ``bpm_info.*`` and ``rhythm.onset_rate``
-- via beat-synchronous tempo/onset detection. The ``beat3tower`` extraction
marker and the ``TimbreTowerFeatures``/``HarmonyTowerFeatures`` types are kept
(populated as empty defaults) so existing DB rows and the artifact schema still
parse.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
from src.logging_utils import configure_logging

from .beat3tower_types import (
    Beat3TowerFeatures,
    BPMInfo,
    HarmonyTowerFeatures,
    RhythmTowerFeatures,
    TimbreTowerFeatures,
)

logger = logging.getLogger(__name__)


@dataclass
class Beat3TowerConfig:
    """Configuration for 3-tower feature extraction."""
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048  # still consumed by _is_silent's RMS frame_length
    min_beats: int = 4  # Minimum beats required for extraction
    segment_duration: float = 30.0  # Duration of each segment (start/mid/end) in seconds
    default_tempo_bpm: float = 60.0  # Fallback tempo when estimation fails
    timegrid_min_period_sec: float = 0.25  # Clamp to avoid overly small windows
    silence_rms_threshold: float = 1e-4  # RMS threshold for near-silence detection


class Beat3TowerExtractor:
    """
    Beat-synchronous pace (BPM + onset) feature extractor.

    Extracts the BPM/onset-rate fields the pace axis reads, aligned to musical
    beats. Timbre/harmony are no longer computed (SP-C) -- see module docstring.
    """

    def __init__(self, config: Optional[Beat3TowerConfig] = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration. Uses defaults if None.
        """
        self.config = config or Beat3TowerConfig()
        self.sr = self.config.sample_rate
        self.hop_length = self.config.hop_length
        self.n_fft = self.config.n_fft  # used by _is_silent's RMS frame_length

        logger.debug(
            f"Initialized Beat3TowerExtractor (sr={self.sr}, hop_length={self.hop_length})"
        )

    def extract_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract 3-tower features from audio file with segments.

        Returns:
            Dictionary with:
            - 'full': Full track features
            - 'start': First segment_duration seconds
            - 'mid': Middle segment_duration seconds
            - 'end': Last segment_duration seconds
            - 'metadata': Extraction metadata
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Load full audio
            y, sr = librosa.load(file_path, sr=self.sr)
            duration = len(y) / sr

            logger.debug(f"Loaded {path.name}: {duration:.1f}s at {sr}Hz")

            # Extract full track features
            full_features, full_meta = self._extract_full_with_meta(y, segment_label="full")

            # Extract segment features
            seg_dur = self.config.segment_duration

            if duration <= seg_dur * 2:
                # Short track: use full features for all segments
                result = {
                    'full': full_features.to_dict(),
                    'start': full_features.to_dict(),
                    'mid': full_features.to_dict(),
                    'end': full_features.to_dict(),
                }
                segments_meta = {
                    "full": full_meta,
                    "start": full_meta,
                    "mid": full_meta,
                    "end": full_meta,
                }
            else:
                # Extract segment-specific features
                start_feat, start_meta = self._extract_segment(y, 'start', duration)
                mid_feat, mid_meta = self._extract_segment(y, 'mid', duration)
                end_feat, end_meta = self._extract_segment(y, 'end', duration)

                result = {
                    'full': full_features.to_dict(),
                    'start': start_feat.to_dict() if start_feat else full_features.to_dict(),
                    'mid': mid_feat.to_dict() if mid_feat else full_features.to_dict(),
                    'end': end_feat.to_dict() if end_feat else full_features.to_dict(),
                }
                segments_meta = {
                    "full": full_meta,
                    "start": start_meta if start_meta else full_meta,
                    "mid": mid_meta if mid_meta else full_meta,
                    "end": end_meta if end_meta else full_meta,
                }

            # Add metadata
            result['metadata'] = {
                'extraction_method': 'beat3tower',
                'duration': duration,
                'sample_rate': sr,
                'n_beats_full': full_features.n_beats,
                # beat_mode/sonic_source describe the fallback path for this track.
                'beat_mode': full_meta.get("beat_mode"),
                'beat_count': full_meta.get("beat_count"),
                'tempo_source': full_meta.get("tempo_source"),
                'timegrid_period_sec': full_meta.get("timegrid_period_sec"),
                'silence_flag': full_meta.get("silence_flag", False),
                'sonic_source': full_meta.get("sonic_source"),
                'segments': segments_meta,
            }
            result['source'] = full_meta.get("sonic_source")

            if full_meta.get("beat_mode") != "beats":
                logger.warning(
                    "Beat3tower fallback for %s: mode=%s beats_detected=%s",
                    path.name,
                    full_meta.get("beat_mode"),
                    full_meta.get("beat_count"),
                )

            return result

        except Exception as e:
            logger.error(f"Failed to extract features from {file_path}: {e}")
            return None

    def extract_full(self, y: np.ndarray) -> Beat3TowerFeatures:
        """
        Extract all 3 towers from audio signal.

        Args:
            y: Audio time series (mono)

        Returns:
            Beat3TowerFeatures with all towers populated

        Raises:
            InsufficientBeatsError: If too few beats detected
        """
        features, _meta = self._extract_full_with_meta(y, segment_label="full")
        return features

    def _extract_segment(
        self, y: np.ndarray, segment: str, duration: float
    ) -> Tuple[Optional[Beat3TowerFeatures], Optional[Dict[str, Any]]]:
        """Extract features for a specific segment."""
        seg_dur = self.config.segment_duration
        seg_samples = int(seg_dur * self.sr)

        if segment == 'start':
            y_seg = y[:seg_samples]
        elif segment == 'mid':
            mid_start = int((duration / 2 - seg_dur / 2) * self.sr)
            mid_start = max(0, mid_start)
            y_seg = y[mid_start:mid_start + seg_samples]
        elif segment == 'end':
            y_seg = y[-seg_samples:]
        else:
            raise ValueError(f"Unknown segment: {segment}")

        try:
            return self._extract_full_with_meta(y_seg, segment_label=segment)
        except Exception:
            return None, None

    def _extract_full_with_meta(
        self,
        y: np.ndarray,
        segment_label: str,
    ) -> Tuple[Beat3TowerFeatures, Dict[str, Any]]:
        duration = len(y) / self.sr if len(y) > 0 else 0.0
        silence_flag = self._is_silent(y)

        tempo, beat_frames, beat_times = self._detect_beats(y)
        beat_count = int(len(beat_frames))

        if silence_flag:
            features = self._extract_stats_fallback(y, silence=True)
            return features, self._build_meta(
                beat_mode="stats",
                beat_count=beat_count,
                tempo_source="default",
                timegrid_period_sec=None,
                silence_flag=True,
                tempo_bpm=self.config.default_tempo_bpm,
            )

        if beat_count >= self.config.min_beats:
            try:
                features = self._extract_with_beats(
                    y=y,
                    beat_frames=beat_frames,
                    beat_times=beat_times,
                    tempo=float(tempo),
                )
                return features, self._build_meta(
                    beat_mode="beats",
                    beat_count=beat_count,
                    tempo_source="beat_track",
                    timegrid_period_sec=None,
                    silence_flag=False,
                    tempo_bpm=float(tempo),
                )
            except Exception as exc:
                logger.debug("Beat3tower extraction failed in beats mode (%s): %s", segment_label, exc)

        tempo_est, tempo_source = self._estimate_tempo(y)
        if tempo_est is None:
            tempo_est = self.config.default_tempo_bpm
            tempo_source = "default"

        timegrid = self._make_timegrid_beats(duration, tempo_est)
        if timegrid:
            beat_frames, beat_times, beat_period = timegrid
            try:
                tempo_effective = 60.0 / beat_period if beat_period > 0 else tempo_est
                features = self._extract_with_beats(
                    y=y,
                    beat_frames=beat_frames,
                    beat_times=beat_times,
                    tempo=tempo_effective,
                )
                return features, self._build_meta(
                    beat_mode="timegrid",
                    beat_count=beat_count,
                    tempo_source=tempo_source,
                    timegrid_period_sec=beat_period,
                    silence_flag=False,
                    tempo_bpm=tempo_effective,
                )
            except Exception as exc:
                logger.debug("Beat3tower timegrid extraction failed (%s): %s", segment_label, exc)

        features = self._extract_stats_fallback(y)
        return features, self._build_meta(
            beat_mode="stats",
            beat_count=beat_count,
            tempo_source="default",
            timegrid_period_sec=None,
            silence_flag=False,
            tempo_bpm=self.config.default_tempo_bpm,
        )

    def _build_meta(
        self,
        beat_mode: str,
        beat_count: int,
        tempo_source: str,
        timegrid_period_sec: Optional[float],
        silence_flag: bool,
        tempo_bpm: float,
    ) -> Dict[str, Any]:
        if beat_mode == "beats":
            sonic_source = "beat3tower_beats"
        elif beat_mode == "timegrid":
            sonic_source = "beat3tower_timegrid"
        else:
            sonic_source = "beat3tower_stats"
        return {
            "beat_mode": beat_mode,
            "beat_count": int(beat_count),
            "tempo_source": tempo_source,
            "timegrid_period_sec": float(timegrid_period_sec) if timegrid_period_sec is not None else None,
            "silence_flag": bool(silence_flag),
            "sonic_source": sonic_source,
            "tempo_bpm": float(tempo_bpm),
        }

    def _detect_beats(
        self,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sr, hop_length=self.hop_length
        )
        # librosa.beat.beat_track returns tempo as an ndim>0 array; float() on it is a numpy
        # ndim>0->scalar deprecation (a future hard error) -- take the scalar explicitly.
        return float(np.ravel(tempo)[0]), np.asarray(beat_frames, dtype=int), np.asarray(beat_times, dtype=float)

    def _estimate_tempo(self, y: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
        try:
            onset_env = librosa.onset.onset_strength(
                y=y, sr=self.sr, hop_length=self.hop_length
            )
            tempos = librosa.beat.tempo(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
            )
            if tempos is None or len(tempos) == 0:
                return None, None
            tempo = float(np.nanmedian(tempos))
            if not np.isfinite(tempo) or tempo <= 0:
                return None, None
            return tempo, "onset_tempo"
        except Exception:
            return None, None

    def _make_timegrid_beats(
        self,
        duration: float,
        tempo_bpm: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if duration <= 0:
            return None
        beat_period = 60.0 / max(tempo_bpm, 1e-6)
        max_period = duration / float(self.config.min_beats + 1)
        beat_period = min(beat_period, max_period)
        beat_period = max(beat_period, self.config.timegrid_min_period_sec)

        beat_times = np.arange(0.0, duration, beat_period)
        beat_times = beat_times[beat_times < duration]
        if beat_times.size == 0:
            return None

        beat_frames = librosa.time_to_frames(
            beat_times, sr=self.sr, hop_length=self.hop_length
        )
        beat_frames = np.unique(beat_frames)
        if beat_frames.size < self.config.min_beats:
            return None

        beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sr, hop_length=self.hop_length
        )
        return beat_frames, beat_times, float(beat_period)

    def _make_stats_beats(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(y) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_frames = max(1, int(np.floor(len(y) / self.hop_length)))
        target = max(self.config.min_beats, 2)
        frame_positions = np.linspace(0, max(0, n_frames - 1), num=target, dtype=int)
        frame_positions = np.unique(frame_positions)
        if frame_positions.size < 2:
            frame_positions = np.array([0, max(1, n_frames - 1)], dtype=int)
        beat_times = librosa.frames_to_time(
            frame_positions, sr=self.sr, hop_length=self.hop_length
        )
        return frame_positions, beat_times

    def _compute_onset_rate(self, y: np.ndarray) -> float:
        """Onsets per second -- the rhythm 'busyness' the pace axis reads. Verbatim from
        the former rhythm tower so the value stays byte-identical to the pre-SP-C output."""
        onset_times = librosa.onset.onset_detect(
            y=y, sr=self.sr, units='time', hop_length=self.hop_length
        )
        return len(onset_times) / (len(y) / self.sr) if len(y) > 0 else 0.0

    def _extract_with_beats(
        self,
        y: np.ndarray,
        beat_frames: np.ndarray,
        beat_times: np.ndarray,
        tempo: float,
    ) -> Beat3TowerFeatures:
        # SP-C: the sonic space is MuQ, so the timbre/harmony towers are no longer
        # consumed anywhere. Compute only the fields the pace axis reads -- BPM (via
        # _compute_bpm_info) and onset_rate -- and leave timbre/harmony as empty
        # defaults. The 'beat3tower' marker is retained as the artifact universe-flag.
        bpm_info = self._compute_bpm_info(tempo, beat_times)
        rhythm = RhythmTowerFeatures(
            onset_rate=self._compute_onset_rate(y),
            bpm=bpm_info.primary_bpm,
            tempo_stability=bpm_info.tempo_stability,
        )
        return Beat3TowerFeatures(
            rhythm=rhythm,
            timbre=TimbreTowerFeatures(),
            harmony=HarmonyTowerFeatures(),
            bpm_info=bpm_info,
            n_beats=int(len(beat_frames)),
            extraction_method='beat3tower',
        )

    def _extract_stats_fallback(self, y: np.ndarray, silence: bool = False) -> Beat3TowerFeatures:
        beat_frames, beat_times = self._make_stats_beats(y)
        tempo = float(self.config.default_tempo_bpm)
        if silence:
            bpm_info = BPMInfo(primary_bpm=tempo, tempo_stability=0.0)
            rhythm = RhythmTowerFeatures(bpm=tempo, tempo_stability=0.0)
            timbre = TimbreTowerFeatures()
            harmony = HarmonyTowerFeatures()
            return Beat3TowerFeatures(
                rhythm=rhythm,
                timbre=timbre,
                harmony=harmony,
                bpm_info=bpm_info,
                n_beats=int(len(beat_frames)),
                extraction_method='beat3tower',
            )
        try:
            return self._extract_with_beats(
                y=y,
                beat_frames=beat_frames,
                beat_times=beat_times,
                tempo=tempo,
            )
        except Exception:
            bpm_info = BPMInfo(primary_bpm=tempo, tempo_stability=0.0)
            rhythm = RhythmTowerFeatures(bpm=tempo, tempo_stability=0.0)
            timbre = TimbreTowerFeatures()
            harmony = HarmonyTowerFeatures()
            return Beat3TowerFeatures(
                rhythm=rhythm,
                timbre=timbre,
                harmony=harmony,
                bpm_info=bpm_info,
                n_beats=int(len(beat_frames)),
                extraction_method='beat3tower',
            )

    def _is_silent(self, y: np.ndarray) -> bool:
        if len(y) == 0:
            return True
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        return float(np.max(rms)) < float(self.config.silence_rms_threshold)

    # =========================================================================
    # BPM ANALYSIS
    # =========================================================================

    def _compute_bpm_info(self, tempo: float, beat_times: np.ndarray) -> BPMInfo:
        """
        Compute BPM with half/double tempo awareness.

        Analyzes beat intervals to detect if half or double tempo interpretation
        is more likely.
        """
        if len(beat_times) < 4:
            return BPMInfo(primary_bpm=float(tempo), tempo_stability=0.0)

        # Analyze tempo from beat intervals
        beat_intervals = np.diff(beat_times)
        interval_bpms = 60.0 / (beat_intervals + 1e-10)

        # Primary BPM from librosa
        primary_bpm = float(tempo)
        half_tempo = primary_bpm / 2
        double_tempo = primary_bpm * 2

        # Count intervals closer to each interpretation
        close_to_primary = np.sum(np.abs(interval_bpms - primary_bpm) < 15)
        close_to_half = np.sum(np.abs(interval_bpms - half_tempo) < 15)
        close_to_double = np.sum(np.abs(interval_bpms - double_tempo) < 15)

        total = len(interval_bpms)
        half_likely = (close_to_half / total > 0.25) if total > 0 else False
        double_likely = (close_to_double / total > 0.25) if total > 0 else False

        # Tempo stability (inverse of coefficient of variation)
        mean_interval = np.mean(beat_intervals)
        std_interval = np.std(beat_intervals)
        cv = std_interval / (mean_interval + 1e-10)
        tempo_stability = float(np.clip(1.0 - cv, 0.0, 1.0))

        return BPMInfo(
            primary_bpm=primary_bpm,
            half_tempo_likely=half_likely,
            double_tempo_likely=double_likely,
            tempo_stability=tempo_stability,
        )


# ============================================================================
# Module-level convenience functions
# ============================================================================

def extract_beat3tower_features(
    file_path: str,
    config: Optional[Beat3TowerConfig] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract 3-tower features from an audio file.

    Convenience function for use in scripts.

    Args:
        file_path: Path to audio file
        config: Extraction configuration

    Returns:
        Dictionary with full, start, mid, end features, or None on failure
    """
    extractor = Beat3TowerExtractor(config)
    return extractor.extract_from_file(file_path)


# Test if run directly
if __name__ == "__main__":
    import sys

    configure_logging(level="INFO")

    if len(sys.argv) < 2:
        logger.info("Usage: python beat3tower_extractor.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    logger.info(f"Extracting 3-tower features from: {file_path}")

    result = extract_beat3tower_features(file_path)

    if result:
        logger.info("\nExtraction successful!")
        logger.info(f"  Beats detected: {result['metadata']['n_beats_full']}")
        logger.info(f"  Duration: {result['metadata']['duration']:.1f}s")

        full = Beat3TowerFeatures.from_dict(result['full'])
        logger.info("\nFeature dimensions:")
        logger.info(f"  Rhythm: {full.rhythm.to_vector().shape}")
        logger.info(f"  Timbre: {full.timbre.to_vector().shape}")
        logger.info(f"  Harmony: {full.harmony.to_vector().shape}")
        logger.info(f"  Total: {full.to_vector().shape}")

        logger.info("\nBPM Info:")
        logger.info(f"  Primary: {full.bpm_info.primary_bpm:.1f}")
        logger.info(f"  Stability: {full.bpm_info.tempo_stability:.2f}")
        logger.info(f"  Half-tempo likely: {full.bpm_info.half_tempo_likely}")
        logger.info(f"  Double-tempo likely: {full.bpm_info.double_tempo_likely}")
    else:
        logger.info("Extraction failed!")
        sys.exit(1)
