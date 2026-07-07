# SP-C: slim the sonic scan to BPM + onset

**Date:** 2026-07-07
**Status:** Design approved, pending spec review
**Prior art:** `project_sp_b_remove_mert_towers` named SP-C as its next step ("retire beat3tower
features — still BPM/pace source"). Layer-3 item 17 (CLAUDE.md) records the tower-decomposition
lesson we are deliberately *not* deleting (it lives in docs + git history).

---

## 1. Motivation

The sonic analyze stage runs the full `Beat3TowerExtractor` on every pending track — rhythm,
timbre, and harmony towers. Since SP-B (2026-07-01/02) the sonic *space* is MuQ
(`X_sonic_muq`), so the 137-dim tower vectors are **dead**: nothing in generation or the
artifact consumes them. The stage still runs only because it also populates the handful of
BPM/onset fields the pace axis needs. Result: every new-track scan pays for timbre + harmony
DSP (MFCC, chroma, tonnetz, tempogram/entropy) that is thrown away.

**Goal:** compute only what's still used (BPM + onset), delete the dead tower DSP, and keep
the scan's downstream contract byte-identical. Faster scans, less deprecated code.

## 2. What is actually used (verified)

`src/playlist/bpm_loader.py:68-72` is the only live consumer of the sonic stage's output. It
reads exactly **five scalar fields** from the `sonic_features` JSON:

- `$.full.bpm_info.primary_bpm`
- `$.full.bpm_info.half_tempo_likely`
- `$.full.bpm_info.double_tempo_likely`
- `$.full.bpm_info.tempo_stability`
- `$.full.rhythm.onset_rate`

The pace/BPM admission gate (`candidate_pool.py:679`, `pier_bridge/pace_gate.py`) and the
beatless-track handling (`bpm_trust_min_onset_rate`) consume those. The artifact build
(`scripts/build_beat3tower_artifacts.py`) uses `sonic_features` only as a **universe gate**
plus a shape check, `_is_beat3tower_features` (`:200`), which passes on either the tower keys
OR a `full.extraction_method == "beat3tower"` **marker** (`:209`) — no towers required.

## 3. What is computed today (`beat3tower_extractor.py`)

`_extract_with_beats` (`:385-407`) runs four things:

| Call | Produces | Status |
|------|----------|--------|
| `_detect_beats` (`:310`, `librosa.beat.beat_track`) | tempo, beat_frames | **needed** (BPM) |
| `_extract_rhythm_tower` (`:456`) | `onset_rate` (`:505`) **+** tempogram/autocorrelation/entropy tower descriptors | **onset_rate needed; rest dead** |
| `_compute_bpm_info` (`:396`) | `primary_bpm`, half/double-tempo flags, `tempo_stability` | **needed** |
| `_extract_timbre_tower` (`:573`) | MFCC/spectral tower | **dead** |
| `_extract_harmony_tower` (`:659`) | chroma/tonnetz tower | **dead** |

The extractor already has fallbacks (`_extract_stats_fallback` `:409-443`, `_is_silent`
`:446`) that emit a valid `Beat3TowerFeatures` with **empty** timbre/harmony + `bpm_info` —
they just don't compute `onset_rate`.

## 4. Design

### 4a. Core change — a lean `_extract_with_beats`

Replace the body so it computes only the used fields:

- Call `librosa.beat.beat_track` (tempo, beats) and the onset detection that yields
  `onset_rate` **with exactly the same parameters the full path used** (from `_detect_beats`
  `:310` and `_extract_rhythm_tower` `:505`), so the five fields come out byte-identical
  (§5 enforces this). Reusing a single onset envelope across both is a permitted optimization
  **only if** it leaves the numbers unchanged versus the current calls — the golden test is
  the gate.
- `bpm_info = _compute_bpm_info(tempo, beat_times)` (unchanged).
- Build `Beat3TowerFeatures(rhythm=<bpm, tempo_stability, onset_rate>, timbre=TimbreTowerFeatures(),
  harmony=HarmonyTowerFeatures(), bpm_info=bpm_info, extraction_method='beat3tower')`.

The `extraction_method='beat3tower'` marker is **kept** as the artifact universe-flag, with a
code comment noting it no longer implies towers.

### 4b. Delete (dead, no external callers)

- `_extract_timbre_tower` (`:573-658`) and `_extract_harmony_tower` (`:659-694`).
- The non-onset rhythm-tower descriptors inside `_extract_rhythm_tower` (tempogram,
  autocorrelation, rhythmic entropy) — reduce it to onset detection + `onset_rate`, or add a
  focused `_compute_onset_rate` helper and drop `_extract_rhythm_tower` if nothing else needs it.
- Now-dead config on `Beat3TowerConfig` (e.g. `n_mfcc` `:37`) and any imports left unused.

Verified callers before deletion: the tower methods are called only from `_extract_with_beats`;
`TimbreTowerFeatures`/`HarmonyTowerFeatures` are referenced only in `beat3tower_types.py` and
the extractor; the public `extract_from_file` is consumed by `librosa_analyzer.py` and
`tests/test_beat3tower_fallback.py`. The plan re-confirms with a repo grep before deleting.

### 4c. Keep

- The `TimbreTowerFeatures` / `HarmonyTowerFeatures` **types** and `Beat3TowerFeatures.from_dict`
  (`beat3tower_types.py`) — so parsing an *existing* row's tower JSON never crashes. They are
  simply never populated by the new extraction path.
- The silent/short-audio fallbacks (`_extract_stats_fallback`, `_is_silent`). These keep their
  current behavior of omitting `onset_rate` for degenerate audio — `bpm_loader` already treats a
  missing `onset_rate` as `nan` (`bpm_loader.py:92`), so no change is needed there. (Only the
  normal `_extract_with_beats` path must produce `onset_rate`; the fallbacks stay as-is.)
- The analyzer routing (`librosa_analyzer.py` → `Beat3TowerExtractor.extract_from_file`) is
  **unchanged**: the slim path lives inside the extractor, so callers see the same API.

### 4d. Data flow / backward compatibility

`to_dict()` still emits `full.{bpm_info, rhythm.onset_rate, extraction_method:'beat3tower'}`,
so `bpm_loader`'s five fields and the artifact gate are unaffected. Existing ~41k rows keep
their (unused) tower JSON — untouched, still pass the universe gate and still yield the five
pace fields. New tracks get slim features. **No re-analysis and no artifact rebuild required.**

## 5. Testing — the pace-neutrality guarantee

The slim path uses the *same* `librosa.beat.beat_track` / onset-detection calls the full path
used for the five fields, so they must be **byte-identical**.

- **Golden test:** capture the five pace fields from the *current* (full) extractor on fixed
  synthetic audio — a deterministic click/beat signal at a known tempo, plus a silent clip —
  and commit them as the expected baseline. After the slim change, assert the slim extractor
  reproduces those five fields exactly (float-exact where the same primitive is called; a tight
  tolerance only if a call is restructured). This decouples the test from the deleted code.
- **Regression:** `tests/test_beat3tower_fallback.py` stays green (silent/short/degenerate
  audio still returns a valid feature dict with the marker).
- **Shape:** assert the slim `to_dict()` still contains `full.bpm_info.*`, `full.rhythm.onset_rate`,
  and `full.extraction_method == 'beat3tower'`, and that `_is_beat3tower_features` accepts it.
- **Speed sanity:** a coarse assertion (or logged timing) that the slim path skips the tower
  work; not a strict wall-clock gate.

## 6. Non-goals

- No purge of old rows' tower JSON from `metadata.db` (harmless; a production DB write we
  don't need).
- No artifact rebuild; no MuQ/sonic-space change.
- No rename of the `beat3tower` marker or the artifact gate.
- No change to pace/BPM/onset behavior — the five fields are preserved exactly.

## 7. Touched files

| File | Change |
|------|--------|
| `src/features/beat3tower_extractor.py` | lean `_extract_with_beats`; delete timbre/harmony tower methods + non-onset rhythm descriptors; onset_rate in fallbacks; drop dead config |
| `src/features/beat3tower_types.py` | keep types for `from_dict`; drop now-unused tower helpers only if fully unreferenced (plan confirms) |
| `src/librosa_analyzer.py` | doc/comment only (routing unchanged) — update the "rhythm/timbre/harmony" docstring at `:44` to reflect BPM+onset |
| `tests/` | golden pace-field baseline (slim == full) + shape test; keep `test_beat3tower_fallback.py` green |
