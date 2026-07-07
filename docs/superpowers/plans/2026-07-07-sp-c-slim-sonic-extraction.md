# SP-C: Slim Sonic Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sonic analyze stage compute only the BPM + onset fields the pace axis uses, and delete the dead timbre/harmony tower DSP — faster new-track scans, less deprecated code, zero downstream behavior change.

**Architecture:** The sonic space is MuQ; the 137-dim beat3tower vectors are unconsumed. The stage runs only to populate 5 pace fields (`bpm_info.*` + `rhythm.onset_rate`) that `bpm_loader` reads from the `sonic_features` JSON. Task 1 rewrites `Beat3TowerExtractor._extract_with_beats` to compute only those fields (leaving timbre/harmony as empty defaults, keeping the `beat3tower` marker); Task 2 deletes the now-unreachable tower methods, helpers, and dead config.

**Tech Stack:** Python 3.11, numpy, librosa, pytest. Spec: `docs/superpowers/specs/2026-07-07-sp-c-slim-sonic-extraction-design.md`.

## Global Constraints

- The **5 pace fields must be byte-identical** to the current full path: `bpm_info.primary_bpm`, `bpm_info.half_tempo_likely`, `bpm_info.double_tempo_likely`, `bpm_info.tempo_stability`, `rhythm.onset_rate`. They come from `_compute_bpm_info` (unchanged) and the verbatim `librosa.onset.onset_detect` call — do not restructure those computations.
- Keep `extraction_method='beat3tower'` (the artifact universe-flag).
- Keep the `TimbreTowerFeatures` / `HarmonyTowerFeatures` / `RhythmTowerFeatures` **types** and their `from_dict`/`to_vector`/`n_features` unchanged — existing DB rows' tower JSON must still parse, and `to_vector()` length must stay `n_features()`.
- No `metadata.db` writes, no re-analysis, no artifact rebuild. No new dependency.
- Shared checkout: explicit-path commits only — `git add <paths>` then `git commit --only -m "…" -- <paths>`; verify `git diff --cached --name-only`. Never `git add -A/-u/.`, never a bare commit.
- Run pytest directly, never piped through `tail`/`head`: `python -m pytest -q <targets> -p no:cacheprovider`.

---

### Task 1: Slim `_extract_with_beats` to BPM + onset

**Files:**
- Modify: `src/features/beat3tower_extractor.py` (`_extract_with_beats` at `:385-407`; add `_compute_onset_rate`)
- Test: `tests/test_beat3tower_slim.py` (create)

**Interfaces:**
- Consumes: `_detect_beats(y) -> (tempo: float, beat_frames: np.ndarray, beat_times: np.ndarray)` (`:310`), `_compute_bpm_info(tempo, beat_times) -> BPMInfo` (`:707`), `RhythmTowerFeatures` / `TimbreTowerFeatures` / `HarmonyTowerFeatures` / `Beat3TowerFeatures` (`beat3tower_types.py`).
- Produces: `_compute_onset_rate(self, y: np.ndarray) -> float`; a slim `_extract_with_beats(...)` returning `Beat3TowerFeatures` with populated `rhythm` (onset_rate, bpm, tempo_stability) + `bpm_info`, empty `timbre`/`harmony`, `extraction_method='beat3tower'`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beat3tower_slim.py`:

```python
import numpy as np
import librosa
import pytest

from src.features.beat3tower_extractor import Beat3TowerExtractor, Beat3TowerConfig
from src.features.beat3tower_types import TimbreTowerFeatures, HarmonyTowerFeatures


def _synth(sr=22050, duration=4.0, bpm=120.0):
    """Deterministic tone + percussive clicks at a known tempo (real librosa, no mocks)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    period = 60.0 / bpm
    for k in range(int(duration / period)):
        i = int(k * period * sr)
        y[i:i + 200] += 0.8
    return y.astype(np.float32), sr


def test_slim_pace_fields_match_direct_librosa():
    y, sr = _synth()
    ext = Beat3TowerExtractor(Beat3TowerConfig(sample_rate=sr))
    tempo, beat_frames, beat_times = ext._detect_beats(y)
    feats = ext._extract_with_beats(y, beat_frames, beat_times, tempo)

    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", hop_length=ext.hop_length)
    expected_onset_rate = len(onset_times) / (len(y) / sr)
    assert feats.rhythm.onset_rate == pytest.approx(expected_onset_rate)

    expected = ext._compute_bpm_info(tempo, beat_times)
    assert feats.bpm_info.primary_bpm == pytest.approx(expected.primary_bpm)
    assert feats.bpm_info.tempo_stability == pytest.approx(expected.tempo_stability)
    assert feats.bpm_info.half_tempo_likely == expected.half_tempo_likely
    assert feats.bpm_info.double_tempo_likely == expected.double_tempo_likely


def test_slim_towers_are_empty_and_marked():
    y, sr = _synth()
    ext = Beat3TowerExtractor(Beat3TowerConfig(sample_rate=sr))
    tempo, beat_frames, beat_times = ext._detect_beats(y)
    feats = ext._extract_with_beats(y, beat_frames, beat_times, tempo)

    # timbre/harmony are the empty defaults -> zero vectors (dimensions preserved)
    assert np.allclose(feats.timbre.to_vector(), TimbreTowerFeatures().to_vector())
    assert np.allclose(feats.harmony.to_vector(), HarmonyTowerFeatures().to_vector())
    assert feats.extraction_method == "beat3tower"

    # downstream contract: the JSON bpm_loader reads carries the 5 pace fields
    d = feats.to_dict()
    assert "onset_rate" in d["rhythm"]
    for k in ("primary_bpm", "half_tempo_likely", "double_tempo_likely", "tempo_stability"):
        assert k in d["bpm_info"]
```

- [ ] **Step 2: Run tests to verify the discriminating one fails**

Run: `python -m pytest -q tests/test_beat3tower_slim.py -p no:cacheprovider`
Expected: `test_slim_towers_are_empty_and_marked` FAILS — the current `_extract_with_beats` computes real (non-empty) timbre/harmony, so `np.allclose(feats.timbre.to_vector(), TimbreTowerFeatures().to_vector())` is False. (`test_slim_pace_fields_match_direct_librosa` may already pass — it is the invariant guard that the change must keep green.)

- [ ] **Step 3: Add `_compute_onset_rate` and rewrite `_extract_with_beats`**

In `src/features/beat3tower_extractor.py`, replace `_extract_with_beats` (`:385-407`) with:

```python
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
```

(`RhythmTowerFeatures`, `TimbreTowerFeatures`, `HarmonyTowerFeatures`, `BPMInfo`, `Beat3TowerFeatures` are already imported at the top of the file. The `_extract_rhythm_tower` / `_extract_timbre_tower` / `_extract_harmony_tower` methods are now unreachable but remain until Task 2.)

- [ ] **Step 4: Run tests to verify they pass (incl. existing fallback suite)**

Run: `python -m pytest -q tests/test_beat3tower_slim.py tests/test_beat3tower_fallback.py -p no:cacheprovider`
Expected: PASS. The fallback tests still pass because `_assert_schema` checks `to_vector().shape[0] == n_features()` (empty towers → zero vectors of the same dimension) and the meta (`sonic_source`, `beat_mode`) is built separately.

- [ ] **Step 5: Check no other test asserts deleted tower content**

Run: `python -m pytest -q tests/unit/test_sonic_phase1_metrics.py tests/unit/test_local_sonic_scaled_mode.py -p no:cacheprovider`
Also grep for tests that assert now-empty tower fields:
`grep -rnE "mfcc|chroma|tonnetz|spec_contrast|tempo_acf|rhythm_entropy|onset_median" tests/`
Expected: green. If any test asserts non-empty timbre/harmony/rhythm-descriptor *content* (not just schema/length), it is now testing deleted behavior — report it to the controller with the file:line rather than loosening this task; the controller decides whether to update/remove it (fold into Task 2).

- [ ] **Step 6: Commit**

```bash
git add src/features/beat3tower_extractor.py tests/test_beat3tower_slim.py
git diff --cached --name-only
git commit --only -m "feat(sonic): extract only BPM+onset (SP-C); leave towers empty" -- src/features/beat3tower_extractor.py tests/test_beat3tower_slim.py
```

---

### Task 2: Delete the dead tower code, helpers, and config

**Files:**
- Modify: `src/features/beat3tower_extractor.py` (delete tower methods + helpers + dead config)
- Modify: `src/librosa_analyzer.py:44` (docstring)

**Interfaces:**
- Consumes: the slim `_extract_with_beats` from Task 1 (no longer calls the tower methods). Nothing new produced.

- [ ] **Step 1: Verify the deletion targets have no remaining callers**

Run:
`grep -rnE "_extract_rhythm_tower|_extract_timbre_tower|_extract_harmony_tower|_find_tempo_peaks|_aggregate_per_beat_1d|_aggregate_per_beat_2d|\.n_mfcc|\.n_fft|tempogram_min_frames" src/ scripts/ tests/`
Expected: matches only inside `src/features/beat3tower_extractor.py` (the definitions/usages you are about to delete) and the `Beat3TowerConfig` field defaults. If any match appears in another module or a test, STOP and report it — do not delete a still-referenced symbol.

(Note: the silence branch of `_extract_stats_fallback` builds `RhythmTowerFeatures(...)` directly — that stays. The non-silence branch calls `_extract_with_beats`, now slim — that stays.)

- [ ] **Step 2: Delete the dead methods**

In `src/features/beat3tower_extractor.py`, delete these method definitions entirely:
- `_extract_rhythm_tower` (`:456-544`)
- `_find_tempo_peaks` (`:546-567`)
- `_extract_timbre_tower` (`:573-653`)
- `_extract_harmony_tower` (`:659-701`)
- `_aggregate_per_beat_1d` and `_aggregate_per_beat_2d` (the per-beat aggregation helpers used only by the deleted towers — confirm via the Step 1 grep that nothing else calls them, then delete both).

- [ ] **Step 3: Remove now-dead config + init fields**

In `Beat3TowerConfig` (`:33-45`), delete the fields `n_mfcc` (`:37`), `n_fft` (`:39`), and `tempogram_min_frames` (`:45`). In `__init__` (`:56-72`), delete `self.n_mfcc = self.config.n_mfcc` and `self.n_fft = self.config.n_fft`, and drop `n_mfcc` from the debug log so it reads:

```python
        logger.debug(
            f"Initialized Beat3TowerExtractor (sr={self.sr}, hop_length={self.hop_length})"
        )
```

Keep `sample_rate`, `hop_length`, `min_beats`, `segment_duration`, `default_tempo_bpm`, `timegrid_min_period_sec`, `silence_rms_threshold` (still used by beat detection, the timegrid path, fallbacks, and silence detection).

- [ ] **Step 4: Update the analyzer docstring**

In `src/librosa_analyzer.py:44`, change:

```python
        Uses the Beat3TowerExtractor to extract rhythm/timbre/harmony features
```
to:
```python
        Uses the Beat3TowerExtractor to extract BPM + onset (pace) features
```

- [ ] **Step 5: Run tests + lint**

Run: `python -m pytest -q tests/test_beat3tower_slim.py tests/test_beat3tower_fallback.py -p no:cacheprovider`
Expected: PASS (behavior is unchanged from Task 1; this task only removes unreachable code).
Run: `ruff check src/features/beat3tower_extractor.py src/librosa_analyzer.py`
Expected: clean (no unused-import/name warnings from the deletions — remove any import left unused by the deleted methods, e.g. if `scipy`/a librosa submodule was only used by a tower).

- [ ] **Step 6: Commit**

```bash
git add src/features/beat3tower_extractor.py src/librosa_analyzer.py
git diff --cached --name-only
git commit --only -m "refactor(sonic): delete dead beat3tower timbre/harmony DSP (SP-C)" -- src/features/beat3tower_extractor.py src/librosa_analyzer.py
```

---

## Final verification (after both tasks)

- [ ] **Focused suite:** `python -m pytest -q tests/test_beat3tower_slim.py tests/test_beat3tower_fallback.py tests/unit/test_bpm_loader.py tests/unit/test_bpm_loader_onset.py -p no:cacheprovider` — the slim extractor + the BPM/onset loader that consumes it, green.
- [ ] **Live (per CLAUDE.md — exercise the real path):** on a few real tracks, run the sonic stage (`python scripts/update_sonic.py --limit 5 --force` or an Analyze Library run with pending sonic) and confirm (a) it completes, (b) the written `sonic_features` JSON has `full.bpm_info.*` + `full.rhythm.onset_rate` + `extraction_method: "beat3tower"`, and (c) a generation still applies the pace/BPM gate (BPM data not reported missing). Compare wall-clock per track against a pre-change run to confirm the speedup.

## Self-review notes

- **Spec coverage:** §4a slim path → Task 1; §4b deletions → Task 2; §4c keep types/fallbacks → preserved (types untouched, fallbacks untouched); §5 golden byte-identical → `test_slim_pace_fields_match_direct_librosa` (invariant guard) + `test_slim_towers_are_empty_and_marked` (discriminating); §6 non-goals respected (no marker rename, no DB write, no artifact rebuild).
- **Byte-identical guarantee:** the 5 fields are produced by `_compute_bpm_info` (untouched) and the verbatim `onset_detect` call in `_compute_onset_rate`; the towers never fed them, so removing the towers cannot change them.
- **`to_vector`/`n_features` invariant:** unchanged — empty `TimbreTowerFeatures()`/`HarmonyTowerFeatures()` yield zero vectors of the same dimension, so `_assert_schema` in the fallback test stays green.
