"""BPM bridge band is bypassed when beats are absent.

BPM is meaningless on beatless audio (e.g. drone), yet librosa returns a
confident garbage value and `tempo_stability` is fooled (it reads ~0.96 even
for drone). `onset_rate` is the reliable beat-presence signal. So the BPM band
is gated on onset_rate: a beatless PIER disables the band for the segment (its
BPM can't set a meaningful target), and a beatless CANDIDATE bypasses the band
(its BPM can't be judged). Controlled by `bpm_trust_min_onset_rate` (0.0 = off,
backward-compatible). Onset band stays active — it is the trustworthy signal.
"""
import numpy as np

from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _flat(n: int) -> np.ndarray:
    X = np.ones((n, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _cfg(**kw):
    base = dict(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        bpm_bridge_max_log_distance=0.10,       # tight BPM band
        onset_bridge_max_log_distance=float("inf"),  # onset band off — isolate BPM
        bpm_bridge_soft_penalty_strength=0.0,   # hard gate, so out-of-band == reject
    )
    base.update(kw)
    return PierBridgeConfig(**base)


def test_beatless_pier_disables_bpm_band():
    # pier_a is drone (onset 0.05): its BPM target is garbage, so the band is off.
    X = _flat(3)  # 0=pier_a, 1=cand, 2=pier_b
    bpm = np.array([160.0, 100.0, 130.0])     # cand 100 vs target 160 -> out of band
    onset = np.array([0.05, 5.0, 5.0])        # pier_a beatless
    path, _h, _e, err = _beam_search_segment(
        0, 2, 1, [1], X, X, None, None, None, None,
        _cfg(bpm_trust_min_onset_rate=0.5), 5,
        perceptual_bpm=bpm, onset_rate=onset,
    )
    assert err is None
    assert path == [1]


def test_beatless_candidate_bypasses_bpm_band():
    # piers have beats; the only candidate is beatless with an out-of-band BPM.
    X = _flat(3)
    bpm = np.array([120.0, 220.0, 120.0])     # cand 220 vs target 120 -> out of band
    onset = np.array([5.0, 0.05, 5.0])        # candidate beatless
    path, _h, _e, err = _beam_search_segment(
        0, 2, 1, [1], X, X, None, None, None, None,
        _cfg(bpm_trust_min_onset_rate=0.5), 5,
        perceptual_bpm=bpm, onset_rate=onset,
    )
    assert err is None
    assert path == [1]


def test_bpm_trust_default_off_keeps_hard_gate():
    # Default threshold 0.0 -> legacy behavior: beatless candidate's BPM still gated.
    X = _flat(3)
    bpm = np.array([120.0, 220.0, 120.0])
    onset = np.array([5.0, 0.05, 5.0])
    path, _h, _e, err = _beam_search_segment(
        0, 2, 1, [1], X, X, None, None, None, None,
        _cfg(), 5,  # bpm_trust_min_onset_rate defaults to 0.0
        perceptual_bpm=bpm, onset_rate=onset,
    )
    assert path is None
    assert err is not None and "continuations" in err


def test_real_beat_candidate_still_gated():
    # Trust threshold on, but candidate HAS beats -> band still applies -> rejected.
    X = _flat(3)
    bpm = np.array([120.0, 220.0, 120.0])
    onset = np.array([5.0, 5.0, 5.0])         # candidate has real beats
    path, _h, _e, err = _beam_search_segment(
        0, 2, 1, [1], X, X, None, None, None, None,
        _cfg(bpm_trust_min_onset_rate=0.5), 5,
        perceptual_bpm=bpm, onset_rate=onset,
    )
    assert path is None
    assert err is not None and "continuations" in err
