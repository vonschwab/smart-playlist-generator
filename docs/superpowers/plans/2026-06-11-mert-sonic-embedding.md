# MERT Sonic Embedding — Implementation Plan

Spec: `docs/superpowers/specs/2026-06-11-mert-sonic-embedding-design.md`
Phase 0 (validation prototype) is **done** — MERT beats towers +83% on neighbor genre-coherence; centering fixes saturation. Evidence in memory `project_timbre_embedding_ceiling`; prototype scripts in `C:\tmp\diag_*.py`, cached probe embeddings `C:\tmp\mert_emb.npz`.

**Safety invariants for every phase:** audio files read-only; `metadata.db` read-only (file paths only); artifact backed up before any fold; sidecars identity-locked to `track_ids` and dropped **loudly** on mismatch.

**Discovery that shapes Phase 5:** the active variant is declared in the artifact itself (`X_sonic_variant` npz key — `src/features/artifacts.py:144-161` picks `X_sonic_{variant}` and falls back to raw with a warning). There is no config-side variant selector today. Activation therefore = fold-time `--set-active` flag + a new config override for A/B.

---

## Phase 1 — Extraction infrastructure

**Deliverable:** `scripts/extract_mert_sidecar.py` — resumable, shard-based MERT extraction.

Design:
- **Clip windows** (pure helper, TDD): `clip_windows(duration_s, clip_s=24.0) -> [(offset, dur) x3]` for start/mid/end; tracks `< ~30 s` return one window used for all three slots. Unit tests first: normal, short, very-short (< clip) tracks.
- **Shard store** (TDD): writes `data/artifacts/beat3tower_32k/mert_shards/shard_NNNN.npz` (`track_ids`, `emb_start|mid|end` float32) + `manifest.json` (done ids, failed ids w/ reason). `--resume` (default) skips done ids. Tests: resume skips, merge produces exactly-once ids, failure entries don't block.
- **Embedder isolation:** model behind a `Callable[[np.ndarray], np.ndarray]` so tests inject a fake; the real one loads `m-a-p/MERT-v1-95M` **pinned to an explicit HF revision** (record the revision hash in the manifest), mean over 13 layers × time → 768-d.
- **Audio loading:** `soundfile.info` for duration, `librosa.load(sr=24000, mono=True, offset, duration)`. File paths from `metadata.db` `tracks.file_path` (SELECT only). Missing/corrupt file → log, record in manifest, continue.
- CLI: `--limit N`, `--track-ids FILE`, `--device cpu|cuda`, `--shard-size 500`, `--merge-only`.

Acceptance: unit suite green; smoke run `--limit 25` produces a merged sidecar whose ids resolve in the artifact.

## Phase 2 — Calibration (~2k subset) + transform decision

**Deliverable:** `scripts/calibrate_mert_transform.py` + a decision.

- Subset: stratified ~2,000 tracks — all 8 diagnostic seeds + their tower-error neighbors (Metallica, Glen Campbell, Jay-Z spot list) + random stratified by artist. Extract via Phase 1 (~3 h CPU).
- Fit candidate transforms on the subset: (a) center+L2, (b) robust-whiten+L2, (c)/(d) each + PCA k ∈ {128, 256}. Persist params (`mean`, `components`, `scales`, `k`) to `mert_transform.npz`.
- Evaluate each: taxonomy-coherence of top-6 neighbors (the 0.183-vs-0.100 metric, reusing the steering provider) + hard spot-check assertions (e.g., "no metal/jazz/hip-hop in YYY's top-10") + cosine spread percentiles.
- **Gate: present the comparison table to Dylan; he picks the transform (perf note: smaller k = faster beam).**

## Phase 3 — Full-library extraction (the long pole)

- Run Phase-1 script over all 39,957 tracks. Estimate **~57 h CPU** (3 × 24 s clips @ ~1.7 s each) — resumable overnight chunks; or hours on any CUDA box (`--device cuda`, same shard format, machines can split id-ranges).
- Output: merged `mert_sidecar.npz` (raw 768-d × 3 clips; transform applied later at fold so it can be re-fit on the full library without re-extracting).
- Monitor: manifest failure rate < 1%; spot-check shard merges.

## Phase 4 — Fold into artifact

**Deliverable:** `scripts/fold_mert_into_artifact.py` (model on `fold_2dftm_into_artifact.py`).

- Re-fit chosen transform family on the **full** library (calibration chose the family; full data fits the params), apply, L2-normalize.
- Write `X_sonic_mert` (whole = renormalized mean of clips), `X_sonic_mert_start|mid|end`, plus `mert_transform` params and model revision as artifact metadata.
- Identity check: sidecar `track_ids` must match artifact exactly (order included) — hard error otherwise.
- `--set-active mert|tower_weighted` flips `X_sonic_variant`; default leaves `tower_weighted` active. Timestamped artifact backup before writing (≈ disk-size check first).
- Tests (TDD, tiny synthetic npz): fold writes keys, identity mismatch errors, `--set-active` round-trips, original keys untouched.

## Phase 5 — Runtime integration

1. **Variant resolution for start/mid/end** (`src/features/artifacts.py`): when a variant is active, prefer `X_sonic_{variant}_{start|mid|end}` and fall back to legacy keys with an INFO log. Test with synthetic artifact.
2. **Config override for A/B:** `artifacts.sonic_variant_override: mert` (config.yaml, default absent) — wins over the artifact's declared variant; **startup error** if the requested keys are missing (configured-knob-must-act rule). Test both paths.
3. **Tower-knob guard:** if active variant has no `tower_dims`, explicitly log that `transition_weights`/`tower_weights` are inert for this variant; **raise** if someone sets non-default tower weights together with `variant=mert`. Test.
4. **Pace-gate fallback:** with no rhythm dims, pace bridge gating must run on perceptual BPM (`bpm_array` machinery) instead of silently no-opping — extend the existing loud-warning path into a tested fallback. Test: `pace_bridge_floor > 0` + mert variant → BPM gate active, warning logged once.
5. Golden JSONs + any `tower_dims`-assuming tests updated; `ruff`/`mypy` clean.
6. **Perf check:** 10-seed stress generation with mert active — must stay ≤ 90 s (memory `feedback_generation_time_budget`). If 768-d blows it, PCA-k from Phase 2 is the lever.

## Phase 6 — Validation & flip

1. Re-run the diagnostic suite against the folded artifact (neighbor QA, per-edge decomposition becomes single-space cosine).
2. A/B: 8-seed and 10-seed playlists, `tower_weighted` vs `mert` (config override makes this a one-line flip). Structural metrics + tracklist diff.
3. **Listen test — Dylan's ear is the flip gate, not the metrics.**
4. Blind audition harness pass (sonic-audition precedent).
5. On pass: `--set-active mert`, update `CLAUDE.md` (commitments #17/#18 + the multi-axis note from the spec), changelog entry.
6. Reopen the edge-weight balance: `weight_genre=0` / bridge 0.40 / transition 0.60 (commit `0abfb7e`) was tuned against broken sonic data — re-run the tuning experiments with trustworthy sonic similarity.

## Order & effort

| phase | depends on | effort |
|---|---|---|
| 1 extraction infra | — | one session (TDD) |
| 2 calibration | 1 | ~3 h compute + analysis |
| 3 full extraction | 2 | ~57 h CPU background / hours GPU |
| 4 fold | 3 | small (precedented) |
| 5 runtime | 4 (testable earlier w/ synthetic artifacts) | one session |
| 6 validation | 4+5 | runs + listen tests |

Phases 1, 2 and the synthetic-artifact half of 5 can proceed immediately; 3 is the long pole and should run overnight (or on a GPU box) once 2 picks the transform family.
