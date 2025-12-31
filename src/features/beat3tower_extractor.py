"""
Beat 3-Tower Feature Extractor
===============================

Beat-synchronous extraction of 3-tower sonic features:
- Rhythm tower: onset, tempogram, beat stability
- Timbre tower: MFCC, spectral features
- Harmony tower: chroma, tonnetz

All features are aggregated per beat using robust statistics (median/IQR).
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
    InsufficientBeatsError,
    RhythmTowerFeatures,
    TimbreTowerFeatures,
)

logger = logging.getLogger(__name__)


@dataclass
class Beat3TowerConfig:
    """Configuration for 3-tower feature extraction."""
    sample_rate: int = 22050
    n_mfcc: int = 20
    hop_length: int = 512
    n_fft: int = 2048
    min_beats: int = 4  # Minimum beats required for extraction
    segment_duration: float = 30.0  # Duration of each segment (start/mid/end) in seconds
    default_tempo_bpm: float = 60.0  # Fallback tempo when estimation fails
    timegrid_min_period_sec: float = 0.25  # Clamp to avoid overly small windows
    silence_rms_threshold: float = 1e-4  # RMS threshold for near-silence detection
    tempogram_min_frames: int = 1024  # Pad onset envelope to avoid short-signal warnings


class Beat3TowerExtractor:
    """
    Beat-synchronous 3-tower feature extractor.

    Extracts rhythm, timbre, and harmony features aligned to musical beats.
    Uses robust aggregation (median/IQR) across beats.
    """

    def __init__(self, config: Optional[Beat3TowerConfig] = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration. Uses defaults if None.
        """
        self.config = config or Beat3TowerConfig()
        self.sr = self.config.sample_rate
        self.n_mfcc = self.config.n_mfcc
        self.hop_length = self.config.hop_length
        self.n_fft = self.config.n_fft

        logger.debug(
            f"Initialized Beat3TowerExtractor (sr={self.sr}, n_mfcc={self.n_mfcc}, "
            f"hop_length={self.hop_length})"
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
        return float(tempo), np.asarray(beat_frames, dtype=int), np.asarray(beat_times, dtype=float)

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

    def _extract_with_beats(
        self,
        y: np.ndarray,
        beat_frames: np.ndarray,
        beat_times: np.ndarray,
        tempo: float,
    ) -> Beat3TowerFeatures:
        rhythm = self._extract_rhythm_tower(y, beat_frames, beat_times, tempo)
        timbre = self._extract_timbre_tower(y, beat_frames)
        harmony = self._extract_harmony_tower(y, beat_frames)

        bpm_info = self._compute_bpm_info(tempo, beat_times)
        rhythm.bpm = bpm_info.primary_bpm
        rhythm.tempo_stability = bpm_info.tempo_stability

        return Beat3TowerFeatures(
            rhythm=rhythm,
            timbre=timbre,
            harmony=harmony,
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
    # RHYTHM TOWER
    # =========================================================================

    def _extract_rhythm_tower(
        self,
        y: np.ndarray,
        beat_frames: np.ndarray,
        beat_times: np.ndarray,
        tempo: float,
    ) -> RhythmTowerFeatures:
        """
        Extract rhythm tower features.

        Features:
        - Onset envelope statistics (median, IQR, std, p10, p90)
        - Tempogram peaks and autocorrelation
        - Beat interval stability
        - Rhythmic complexity (entropy)
        """
        # 1. Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_per_beat = self._aggregate_per_beat_1d(onset_env, beat_frames)

        # 2. Tempogram / rhythmic autocorrelation
        tempogram_input = onset_env
        if onset_env.size < self.config.tempogram_min_frames:
            pad_len = self.config.tempogram_min_frames - onset_env.size
            tempogram_input = np.pad(onset_env, (0, pad_len), mode="constant")
        tempogram = librosa.feature.tempogram(
            onset_envelope=tempogram_input,
            sr=self.sr,
            hop_length=self.hop_length,
        )
        tempo_acf = np.mean(tempogram, axis=1)  # Average across time

        # Find tempo peaks
        peak_indices = self._find_tempo_peaks(tempo_acf, n_peaks=3)

        # 3. Beat interval statistics
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            beat_interval_cv = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)
            beat_interval_median = np.median(beat_intervals)
        else:
            beat_interval_cv = 0.0
            beat_interval_median = 0.0

        # 4. Beat strength (onset at beat positions)
        beat_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else np.array([0.0])

        # 5. Onset rate (busyness)
        onset_times = librosa.onset.onset_detect(y=y, sr=self.sr, units='time', hop_length=self.hop_length)
        onset_rate = len(onset_times) / (len(y) / self.sr) if len(y) > 0 else 0.0

        # 6. Rhythmic complexity (entropy)
        onset_probs = onset_per_beat / (np.sum(onset_per_beat) + 1e-10)
        rhythm_entropy = -np.sum(onset_probs * np.log(onset_probs + 1e-10))

        return RhythmTowerFeatures(
            # Onset stats
            onset_median=float(np.median(onset_per_beat)),
            onset_iqr=float(np.percentile(onset_per_beat, 75) - np.percentile(onset_per_beat, 25)),
            onset_std=float(np.std(onset_per_beat)),
            onset_p10=float(np.percentile(onset_per_beat, 10)),
            onset_p90=float(np.percentile(onset_per_beat, 90)),
            # Tempogram peaks
            tempo_peak1_lag=float(peak_indices[0]) if len(peak_indices) > 0 else 0.0,
            tempo_peak2_lag=float(peak_indices[1]) if len(peak_indices) > 1 else 0.0,
            tempo_peak_ratio=float(
                tempo_acf[peak_indices[0]] / (tempo_acf[peak_indices[1]] + 1e-10)
                if len(peak_indices) > 1 else 1.0
            ),
            # ACF lags
            tempo_acf_lag1=float(tempo_acf[1]) if len(tempo_acf) > 1 else 0.0,
            tempo_acf_lag2=float(tempo_acf[2]) if len(tempo_acf) > 2 else 0.0,
            tempo_acf_lag3=float(tempo_acf[3]) if len(tempo_acf) > 3 else 0.0,
            tempo_acf_lag4=float(tempo_acf[4]) if len(tempo_acf) > 4 else 0.0,
            tempo_acf_lag5=float(tempo_acf[5]) if len(tempo_acf) > 5 else 0.0,
            # Beat stability
            beat_interval_cv=float(beat_interval_cv),
            beat_interval_median=float(beat_interval_median),
            # Beat strength
            beat_strength_median=float(np.median(beat_strengths)),
            beat_strength_iqr=float(np.percentile(beat_strengths, 75) - np.percentile(beat_strengths, 25)),
            # Onset rate
            onset_rate=float(onset_rate),
            # Complexity
            rhythm_entropy=float(rhythm_entropy),
            # BPM (will be updated later)
            bpm=float(tempo),
            tempo_stability=1.0,
        )

    def _find_tempo_peaks(self, tempo_acf: np.ndarray, n_peaks: int = 3) -> np.ndarray:
        """Find peak indices in tempogram autocorrelation."""
        # Skip first few lags (too short periods)
        start_lag = 5
        if len(tempo_acf) <= start_lag:
            return np.array([0])

        acf_search = tempo_acf[start_lag:]

        # Find local maxima
        peaks = []
        for i in range(1, len(acf_search) - 1):
            if acf_search[i] > acf_search[i - 1] and acf_search[i] > acf_search[i + 1]:
                peaks.append((i + start_lag, acf_search[i]))

        if not peaks:
            # Fallback: use global maximum
            return np.array([np.argmax(tempo_acf[start_lag:]) + start_lag])

        # Sort by magnitude and return top n
        peaks.sort(key=lambda x: -x[1])
        return np.array([p[0] for p in peaks[:n_peaks]])

    # =========================================================================
    # TIMBRE TOWER
    # =========================================================================

    def _extract_timbre_tower(
        self, y: np.ndarray, beat_frames: np.ndarray
    ) -> TimbreTowerFeatures:
        """
        Extract timbre tower features.

        Features:
        - MFCC (20 coefficients) with median/IQR per beat
        - MFCC delta (dynamics)
        - Spectral contrast, rolloff, centroid, bandwidth, flux
        - Zero crossing rate
        """
        # 1. MFCC (20 coefficients)
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        mfcc_per_beat = self._aggregate_per_beat_2d(mfcc, beat_frames)  # (n_beats, 20)

        mfcc_median = np.median(mfcc_per_beat, axis=0)
        mfcc_iqr = np.percentile(mfcc_per_beat, 75, axis=0) - np.percentile(mfcc_per_beat, 25, axis=0)

        # 2. MFCC delta (dynamics)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_per_beat = self._aggregate_per_beat_2d(mfcc_delta, beat_frames)
        mfcc_delta_median = np.median(mfcc_delta_per_beat, axis=0)

        # 3. Spectral contrast (7 bands)
        spec_contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )
        spec_contrast_per_beat = self._aggregate_per_beat_2d(spec_contrast, beat_frames)

        spec_contrast_median = np.median(spec_contrast_per_beat, axis=0)
        spec_contrast_iqr = (
            np.percentile(spec_contrast_per_beat, 75, axis=0) -
            np.percentile(spec_contrast_per_beat, 25, axis=0)
        )

        # 4. Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )[0]
        rolloff_per_beat = self._aggregate_per_beat_1d(rolloff, beat_frames)

        # 5. Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )[0]
        centroid_per_beat = self._aggregate_per_beat_1d(centroid, beat_frames)

        # 6. Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )[0]
        bandwidth_per_beat = self._aggregate_per_beat_1d(bandwidth, beat_frames)

        # 7. Spectral flux (approximated via onset strength)
        flux = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        flux_per_beat = self._aggregate_per_beat_1d(flux, beat_frames)

        # 8. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        zcr_per_beat = self._aggregate_per_beat_1d(zcr, beat_frames)

        return TimbreTowerFeatures(
            mfcc_median=mfcc_median,
            mfcc_iqr=mfcc_iqr,
            mfcc_delta_median=mfcc_delta_median,
            spec_contrast_median=spec_contrast_median,
            spec_contrast_iqr=spec_contrast_iqr,
            spec_rolloff_median=float(np.median(rolloff_per_beat)),
            spec_rolloff_iqr=float(np.percentile(rolloff_per_beat, 75) - np.percentile(rolloff_per_beat, 25)),
            spec_centroid_median=float(np.median(centroid_per_beat)),
            spec_centroid_iqr=float(np.percentile(centroid_per_beat, 75) - np.percentile(centroid_per_beat, 25)),
            spec_bandwidth_median=float(np.median(bandwidth_per_beat)),
            spec_bandwidth_iqr=float(np.percentile(bandwidth_per_beat, 75) - np.percentile(bandwidth_per_beat, 25)),
            spec_flux_median=float(np.median(flux_per_beat)),
            spec_flux_iqr=float(np.percentile(flux_per_beat, 75) - np.percentile(flux_per_beat, 25)),
            zcr_median=float(np.median(zcr_per_beat)),
        )

    # =========================================================================
    # HARMONY TOWER
    # =========================================================================

    def _extract_harmony_tower(
        self, y: np.ndarray, beat_frames: np.ndarray
    ) -> HarmonyTowerFeatures:
        """
        Extract harmony tower features.

        Features:
        - Chroma CQT (12 bins) with median/IQR per beat
        - Chroma entropy and statistics
        - Tonnetz (6 dims)
        """
        # 1. Chroma CQT (constant-Q transform - better bass resolution)
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
        chroma_per_beat = self._aggregate_per_beat_2d(chroma, beat_frames)  # (n_beats, 12)

        chroma_median = np.median(chroma_per_beat, axis=0)
        chroma_iqr = np.percentile(chroma_per_beat, 75, axis=0) - np.percentile(chroma_per_beat, 25, axis=0)

        # 2. Chroma statistics
        chroma_mean = np.mean(chroma, axis=1)  # Average across time
        chroma_probs = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        chroma_entropy = -np.sum(chroma_probs * np.log(chroma_probs + 1e-10))

        # Peak count (how many pitch classes are dominant)
        threshold = np.mean(chroma_mean) + np.std(chroma_mean)
        chroma_peak_count = np.sum(chroma_mean > threshold)

        # Key strength (max vs mean)
        key_strength = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-10)

        # 3. Tonnetz (tonal centroid - 6 dimensions)
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        tonnetz_per_beat = self._aggregate_per_beat_2d(tonnetz, beat_frames)  # (n_beats, 6)
        tonnetz_median = np.median(tonnetz_per_beat, axis=0)

        return HarmonyTowerFeatures(
            chroma_median=chroma_median,
            chroma_iqr=chroma_iqr,
            chroma_entropy=float(chroma_entropy),
            chroma_peak_count=float(chroma_peak_count),
            key_strength=float(key_strength),
            tonnetz_median=tonnetz_median,
        )

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

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _aggregate_per_beat_1d(
        self, feature: np.ndarray, beat_frames: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate 1D feature (n_frames,) per beat interval.

        Returns:
            Array of shape (n_beats-1,) with aggregated values per beat interval.
        """
        if len(beat_frames) < 2:
            return feature if len(feature) > 0 else np.array([0.0])

        values = []
        for i in range(len(beat_frames) - 1):
            start = beat_frames[i]
            end = beat_frames[i + 1]

            if start < len(feature) and end <= len(feature) and start < end:
                segment = feature[start:end]
                if len(segment) > 0:
                    values.append(np.mean(segment))

        return np.array(values) if values else np.array([np.mean(feature)])

    def _aggregate_per_beat_2d(
        self, feature: np.ndarray, beat_frames: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate 2D feature (n_features, n_frames) per beat interval.

        Returns:
            Array of shape (n_beats-1, n_features) with aggregated values.
        """
        n_features = feature.shape[0]

        if len(beat_frames) < 2:
            return np.mean(feature, axis=1, keepdims=True).T

        values = []
        for i in range(len(beat_frames) - 1):
            start = beat_frames[i]
            end = beat_frames[i + 1]

            if start < feature.shape[1] and end <= feature.shape[1] and start < end:
                segment = feature[:, start:end]
                if segment.shape[1] > 0:
                    values.append(np.mean(segment, axis=1))

        if not values:
            return np.mean(feature, axis=1, keepdims=True).T

        return np.vstack(values)


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
        logger.info(f"\nExtraction successful!")
        logger.info(f"  Beats detected: {result['metadata']['n_beats_full']}")
        logger.info(f"  Duration: {result['metadata']['duration']:.1f}s")

        full = Beat3TowerFeatures.from_dict(result['full'])
        logger.info(f"\nFeature dimensions:")
        logger.info(f"  Rhythm: {full.rhythm.to_vector().shape}")
        logger.info(f"  Timbre: {full.timbre.to_vector().shape}")
        logger.info(f"  Harmony: {full.harmony.to_vector().shape}")
        logger.info(f"  Total: {full.to_vector().shape}")

        logger.info(f"\nBPM Info:")
        logger.info(f"  Primary: {full.bpm_info.primary_bpm:.1f}")
        logger.info(f"  Stability: {full.bpm_info.tempo_stability:.2f}")
        logger.info(f"  Half-tempo likely: {full.bpm_info.half_tempo_likely}")
        logger.info(f"  Double-tempo likely: {full.bpm_info.double_tempo_likely}")
    else:
        logger.info("Extraction failed!")
        sys.exit(1)
