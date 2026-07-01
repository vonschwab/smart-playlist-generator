# SP-B: Retire MERT + the Beat3Tower sonic embedding — Design

**Date:** 2026-07-01
**Status:** Approved design (Dylan), pending plan
**Depends on:** SP-A (MuQ analyze stage) — SHIPPED on master 2026-07-01 (`5aa4ad4`..`e1aebb5`)
**Successor:** SP-C (retire Beat3Tower *feature extraction*) — deliberately **out of scope** here, see §12

## 1. Goal & locked decisions

MuQ-MuLan (`X_sonic_muq`, 512-dim) is the sole active runtime sonic space
(`artifacts.sonic_variant_override: muq`). MERT and the Beat3Tower tower-weighted
blend are unused rollback paths, artifact bloat (~1 GB of dead baked keys), and
superseded research (MuQ 86% vs MERT 73% on trusted triplets; the MERT⊕tower fusion
never shipped). Per the project's engineering discipline — activate fixes, never
default to legacy, delete superseded paths — SP-B removes them.

Decisions locked with Dylan (2026-07-01):

1. **Keep a thin baked-variant seam** (not muq-only hardcoding): `sonic_variant_override`
   + loader selection of `X_sonic_{variant}` + the extraction-stage→sidecar→fold
   pattern survive, with **muq as the sole registered variant**. A future embedding
   = add a stage, bake `X_sonic_<new>` alongside, flip the override. No re-plumbing.
2. **Full muq-native artifact rebuild** (not surgical strip, not leave-in-place):
   the rebuilt `data_matrices_step1.npz` simply stops carrying MERT/tower keys.
3. **Bounded scope:** the Beat3Tower **sonic embedding** goes; the Beat3Tower
   **feature extraction** (`stage_sonic` → `tracks.sonic_features`) **stays** — it is
   the live source of BPM/onset/pace data (`src/playlist/bpm_loader.py` reads
   `sonic_features.full.bpm_info` from `metadata.db`) and the artifact's
   track-universe gate. Retiring it is SP-C, a separate build+migration project.
4. **Artifact contract = per-variant keys only** ("Approach 1"): the artifact carries
   `X_sonic_muq{,_start,_mid,_end}` and **no plain `X_sonic`**. The loader resolves the
   override → loads `X_sonic_{variant}` into `bundle.X_sonic`. No duplicated matrices
   (~340 MB saved vs. carrying both), and side-by-side variant coexistence — exactly
   what made the MERT→MuQ migration safe — is preserved for the next migration.

## 2. Scope boundary

**In scope (remove):**
- MERT: analyze stage, sidecar extraction, fold, transform calibration, calib entry,
  stage registrations, config block, tests.
- Tower sonic embedding: build-time bake (rhythm/timbre/harmony/tower_weighted/2DFTM),
  the `sonic_variant.py` transform pipeline, runtime tower machinery
  (`tower_weights`/`transition_weights`/`tower_pca_dims` knobs, `sonic_axes` slicing,
  tower guards), config keys, tests.
- The artifact rebuild that drops the dead keys.
- Archive (never delete) of the irreplaceable MERT/2DFTM data (§10).

**Out of scope (keep untouched):**
- `stage_sonic`, `src/hybrid_sonic_analyzer.py`, `src/features/beat3tower_extractor.py`,
  `beat3tower_types.py`, `beat3tower_normalizer.py` *as used by extraction*,
  `tracks.sonic_features`, `bpm_loader.py`, all BPM/onset/pace gating. (SP-C.)
- `build_beat3tower_artifacts.load_tracks_with_beat3tower` — the
  `WHERE sonic_features IS NOT NULL` universe gate stays; the 41,042-track universe
  must be unchanged by the rebuild.
- The MuQ path end-to-end (`muq_runner.py`, `stage_muq`, `fold_muq_into_artifact`,
  `muq_sidecar.npz`, `muq_failed.json`).
- The full documentation rewrite (separately gated); only lines SP-B makes factually
  wrong get fixed here.

## 3. End state

- One live sonic space: `bundle.X_sonic` **is** the MuQ matrix, selected via the seam.
- Artifact keys (sonic side): `X_sonic_muq`, `X_sonic_muq_start/mid/end`,
  `muq_transform_mean`, `muq_model`, `X_sonic_variant` (stamp, = `"muq"`).
  Gone: `X_sonic`, `X_sonic_start/mid/end`, `X_sonic_mert*` (×4),
  `X_sonic_rhythm/timbre/harmony*` (×4 each), `X_sonic_tower_weighted`,
  `X_sonic_raw`, `X_sonic_pre_scaled`, `X_sonic_robust_whiten`,
  `mert_transform_mean/std`, `mert_model_revision`, `tower_dims`,
  `tower_calibration`, `normalizer_params`, `bpm_array` (the last three confirmed
  zero-reader write-only).
- ~3–4k LOC and five scripts deleted; ~10 test files deleted, ~5 edited.
- Every deleted knob is gone from `config.example.yaml`, `config_loader.py`, **and**
  Dylan's live `config.yaml` (no silently-dead keys).

## 4. The thin variant seam (surviving contract)

- `artifacts.sonic_variant_override` — config knob, **default flips `"mert"` → `"muq"`**
  in `_mert_fold_settings`' successor (`scripts/analyze_library.py:146` is the current
  landmine) and anywhere else a variant default lives. Keep the explicit
  `sonic_variant_override: muq` line in the live config.
- `_variant_gate` (analyze_library.py:151-174) — kept; known-variant set becomes
  `{"muq"}` (the `sonic_variant._ALLOWED` union disappears with the transform
  pipeline). Unknown override → the SP-A loud warning, unchanged.
- Per-variant stage fingerprints folding `cfg_hash` — kept (muq branch only).
- `fold_muq_into_artifact.fold_muq` — kept, becomes the **unconditional** fold step in
  `stage_artifacts` (no more mert/muq branching).
- Loader: `set_sonic_variant_override` / `load_artifact_bundle`
  (`src/features/artifacts.py:198-208`) — kept; missing `X_sonic_{override}` key at
  load **raises** (existing fail-loud behavior, preserved).
- `TRANSITION_CALIB_BY_VARIANT` (per-variant transition calibration) — kept, muq entry
  only; unknown variant raises (existing behavior).
- Future-embedding recipe (document in the seam's docstring): add extraction stage →
  write `X_sonic_<new>` sidecar → add fold → add calib entry → register variant →
  flip override. Both variants coexist in the artifact during A/B.

## 5. Artifact/loader contract change (Approach 1)

`src/features/artifacts.py`:
- Required-keys (`:141-149`) becomes **override-aware**: require
  `X_sonic_{override}` (base matrix), `X_genre_raw/smoothed`, `genre_vocab`,
  `track_ids`, `artist_keys`, durations. The `_start/_mid/_end` window keys are loaded
  when present (the fold writes them today) and are not in the required set — this
  keeps §13.2's optional slim-down compatible. Plain `X_sonic` is no longer a
  required npz key. Error message on a missing variant key lists the `X_sonic_*`
  variants actually present in the npz.
- The loader maps `X_sonic_{override}{,_start,_mid,_end}` → `bundle.X_sonic`
  (+ window fields). **All consumers above `artifacts.py` read the bundle and are
  unchanged.**
- `X_sonic_variant` stamp stays: written by the fold, checked by `verify`
  (stamp == active variant).

Note: `X_sonic_muq_start/mid/end` are currently copies of the middle-window vector
(MuQ extraction is single-window). SP-B keeps the fold writing all four keys
(contract-compatible, no consumer change). An optional slim-down — loader aliases
missing window keys to the base matrix — is a plan-time decision, not core scope.

## 6. Removal inventory (recon-verified, file:line as of `e1aebb5`)

### Scripts (delete whole files)
| File | Notes |
|---|---|
| `scripts/extract_mert_sidecar.py` (538 LOC) | **First move `load_paths` to `src/analyze/`** — `stage_muq` imports it (analyze_library.py:2425). |
| `scripts/fold_mert_into_artifact.py` (254 LOC) | |
| `scripts/calibrate_mert_transform.py` (1022 LOC) | Research tool, no runtime imports. |
| `scripts/fold_2dftm_into_artifact.py` (271 LOC) | Writes `X_sonic_tower_weighted`, hardcodes 0.20/0.50/0.30, stamps `X_sonic_variant="tower_weighted"` unconditionally. |
| `scripts/extract_harmony_2dftm_sidecar.py` (150 LOC) | |

### `scripts/analyze_library.py`
- Delete: `stage_mert` (2283-2412), `_build_mert_embedder` (2270-2280), the MERT fold
  branch (2122-2143), the `"mert"` fingerprint branch (422-444), the 2DFTM auto-fold
  hook (2077-2090), `"mert"` in `STAGE_FUNCS` (2635) — **all together with** the gate
  edits (crash point #5, §8).
- Edit: `_mert_fold_settings` (128-148) → rename (e.g. `_sonic_fold_settings`), default
  `"muq"`; `_KNOWN_SONIC_VARIANTS` → `{"muq"}`; muq fold becomes unconditional in
  `stage_artifacts`; `stage_verify` guard (2240-2260) re-keyed from
  `mert_sidecar.exists()` to the **active variant's** sidecar (closes SP-A residual (a)).

### `src/similarity/sonic_variant.py`
Delete the transform pipeline wholesale: `_ALLOWED` (11-23), weight resolvers
(26-114), `_normalize_variant_name` (117-124, the silent →`tower_pca` trap),
`resolve_sonic_variant` (127-137), `_variant_transform` (144-393),
`compute_sonic_variant_matrix/norm` (396-419), `apply_transition_weights` (421-473).
S-score call sites (`transition_metrics.py:142-143,247`;
`pier_bridge_builder.py:550-555`; surfaced in `pier_bridge/beam.py:671,1503,1645,1912`)
simplify to plain cosine on `bundle.X_sonic` — behaviorally identical (under muq they
already degrade to the raw passthrough via `tower_fallback`). **Plan-time check:**
confirm whether S feeds live scoring or diagnostics only before simplifying (§13).

### `src/features/artifacts.py`
Delete `DEFAULT_TOWER_TRANSITION_WEIGHTS` (20), `_is_default_transition_weights`
(365-379), `_variant_lacks_tower_split` (382-401), `validate_tower_knobs` (404-439)
and its call site (`src/playlist/pipeline/core.py:732-740`). Required-keys change per §5.

### Runtime tower consumers
- `src/playlist/sonic_axes.py` (whole file) + `pier_bridge_builder.py` axis path
  (557-589; the BPM fallback it already takes under muq becomes the only path) +
  `worker.py` `_resolve_tower_pca_dims`/`_infer_tower_pca_dims` (579-616).
- `src/playlist/replacement.py` (104-119): `_sound_divergence` re-based on full-sonic
  cosine (fixes the silently-wrong-under-muq fake-tower carving);
  `_pace_divergence` keeps its BPM path.
- `apply_transition_weights` call sites removed: `pier_bridge_builder.py:629-640`,
  `transition_metrics.py:136-154`, `constructor.py:145-154` (all inert under muq).

### Transition calibration (all in ONE change — crash point #2, §8)
`src/playlist/transition_metrics.py`: delete `"mert"` from
`TRANSITION_CALIB_BY_VARIANT` (23-26); `_DEFAULT_CALIB_VARIANT` → `"muq"` (29);
`TransitionMetricContext` defaults 0.32/0.0625 → 0.594/0.092 (81-82).
`src/playlist/pier_bridge/config.py`: same default flip (88-90).

### Config
- `config.example.yaml`: delete `analyze.mert` (5-12), `tower_weights` (153-156),
  `transition_weights` (158-163 — retires the known `0.40/0.35/0.25` example bug),
  `tower_pca_dims` (165-170), `sim_variant` (700); `sonic_variant_override` example
  (815-825) → value `muq`.
- `src/config_loader.py`: delete `ds_tower_weights` (347-355), `ds_transition_weights`
  (357-365), `ds_tower_pca_dims` (367-375), dict registrations (495-497). Keep
  `sonic_variant_override` (74-82).
- **Live `config.yaml`** (gitignored, edited with Dylan's knowledge per this spec):
  remove the same keys (`transition_weights` at ~342-345, tower blocks, `analyze.mert`);
  keep `sonic_variant_override: muq`.
- `src/playlist/request_models.py`: drop `"mert"` from the Literal (25) + order (51).
- `web/src/components/ToolsPanel.tsx`: drop `mert` from `ALL_STAGES` + rebuild dist.
- `pyproject.toml`: fix stale description (line 8, "learned MERT sonic embedding");
  `muq` extra stays; no mert deps exist to remove.

### Tests
- Delete: `test_extract_mert_sidecar.py`, `test_analyze_mert_stage.py`,
  `test_mert_extraction_cancel.py`, `test_fold_mert.py`,
  `test_calibrate_mert_transform.py`, `test_artifact_tower_weighted_load.py`,
  `test_worker_tower_pca_dims.py`, `test_fold_2dftm.py`,
  `test_sonic_variant_resolution.py`, `test_tower_knob_guard.py`,
  `test_sonic_axes.py`, `test_beat3tower_fallback.py` **only if** its coverage is
  extraction-side; keep whatever guards `stage_sonic` (plan-time check).
- Edit: `test_variant_gate.py` (default-is-mert test → default-is-muq),
  `test_stage_fingerprint_variant.py` (drop mert branch),
  `test_transition_calibration.py` (drop mert-band tests, add default-is-muq),
  `test_reporter_variant_calib.py` (keep the `4e1136d` regression test, muq-only),
  `test_transition_metric_alignment.py` (re-point defaults),
  `test_muq_stage_registration.py` (stage-order assertion no longer "after mert").
- Add: override-aware required-keys tests; re-keyed verify-guard test; re-based
  divergence tests; a seam test (unknown override → warn at analyze, raise at load).

### Docs (only lines SP-B falsifies)
- `CLAUDE.md`: key-paths entries for MERT shards/sidecar → archived location + status;
  the "0.20/0.50/0.30 baked into tower_weighted" and "don't change transition_weights
  without tower_weights" gotchas → deleted/rewritten; "default is the learned MERT
  embedding" (line ~116) → muq.
- `docs/CLEANUP_LIST.md`: mark SP-B executed; add SP-C entry (retire Beat3Tower
  extraction: dedicated BPM/onset/pace extractor, universe-gate rewrite, delete
  `stage_sonic` + extractor subsystem, re-validate pace gating). **This file is
  currently in-flight in another session — append-only, coordinate at execution.**

## 7. Fixes activated by SP-B (not scope creep — consumers of deleted code)

1. `_sound_divergence` re-based on full-sonic cosine (was silently carving MuQ into
   fake tower slices via inferred `tower_pca_dims`).
2. `stage_verify` variant guard re-keyed to the active variant's sidecar (was keyed to
   `mert_sidecar.npz` existing — archiving MERT would have silently disarmed it).
3. The `config.example.yaml` `transition_weights: 0.40/0.35/0.25` fresh-clone bug
   retired with the knob itself.
4. Stale `pyproject` description + CLAUDE.md variant claims corrected.

## 8. Crash points & explicit handling (from recon)

| # | Coupling | Handling |
|---|---|---|
| 1 | `load_tracks_with_beat3tower` universe gate | **Kept** (scope boundary §2). Rebuild must reproduce the 41,042-track universe exactly. |
| 2 | `resolve_transition_calib(None)` maps `None`→`_DEFAULT_CALIB_VARIANT="mert"`; deleting the mert entry alone ⇒ raise | Delete entry + flip default to `"muq"` + flip both dataclass default sites **in one commit**. |
| 3 | `_require_keys` hard-requires plain `X_sonic` ⇒ every load crashes post-rebuild | Required-keys becomes override-aware **before** the rebuild (§5). |
| 4 | Loader raises if `X_sonic_{override}` missing | Preserved deliberately; `fold_muq` becomes the unconditional fold so a rebuilt artifact always carries it. |
| 5 | Deleting `extract_mert_sidecar` while `STAGE_FUNCS["mert"]`/gate linger ⇒ ImportError on `--stages mert --force` | Stage function, registration, order entry, Literal, gate membership, GUI list removed **in the same task**. |
| 6 | `_sound_divergence` unguarded axis carving | Re-based per §7.1. |

## 9. Artifact rebuild procedure & rollback

1. All code tasks land and the full suite is green **before** touching the artifact.
2. Timestamped backup: `data_matrices_step1.npz` → `data_matrices_step1.bak_<ts>.npz`
   (same dir). The backup is the tower/MERT-keyed artifact and pairs with pre-SP-B
   code only.
3. Rebuild: `stage_artifacts` (build minus tower bake → genre matrices + universe +
   metadata; then unconditional `fold_muq`) → `stage_verify` (re-keyed guard).
4. Verify checks: `track_ids` **identical to the backup's** (the universe gate is
   unchanged, so the universe must be too — do not assume a count, assert equality);
   `X_sonic_variant == "muq"`; no `X_sonic_mert*`/tower keys present;
   `X_sonic_muq` shape == (len(track_ids), 512).
5. **Worker restart** after rebuild (`@lru_cache` holds the old bundle — known trap).
6. Rollback = restore the backup **and** `git revert` the SP-B commits (old artifact
   requires old loader contract). Keep the backup until acceptance (§11) passes,
   then it may be archived alongside the MERT data.

## 10. Data safety & archive protocol (HARD RULES)

- **Archive, never delete.** After acceptance passes and **only with Dylan's explicit
  confirmation at that moment** (second-confirmation rule), move to
  `data/archive/mert_2026/`:
  `mert_shards/` (+manifest), `mert_sidecar.npz` + its `.bak.*` copies,
  `mert_transform_calibration.npz` (+bak), `harmony_2dftm_sidecar.npz`,
  and the orphans `mert_shards_a/`, `mert_shards_b/`, `mert_layers_seeds.npz`.
- Nothing under `data/` is deleted by SP-B. Music files remain read-only. No
  `metadata.db` writes anywhere in SP-B.
- `muq_sidecar.npz` + `muq_failed.json` are live inputs — untouched.
- If any worktree is involved, the data-junction removal rule applies
  (delete the junction link before removing a worktree).
- Deleted *code* is archived by git history — no code archive directory.

## 11. Testing & acceptance

- **Unit:** the edited/added tests in §6 pass; full suite green
  (`python -m pytest -q -m "not slow"` bounded, no piping); `ruff check` + `mypy` clean.
- **Stage acceptance** (closes SP-A residual (b)): after the rebuild, a real
  `--stages artifacts,verify` run completes green *for the right reason* (verify's
  variant check demonstrably executed — assert on its log line, not just exit 0).
- **Generation acceptance:** `gui_fidelity` generation with fixed seeds +
  `random_seed` **before vs after** the rebuild: tracklists must be **identical**
  (the live sonic space is unchanged; any diff is a defect to explain, not noise).
  Log-verified per the playlist-testing skill: `BPM loaded: N/N` present (universe/
  pace data intact), no unexpected gate-tally shifts.
- **GUI smoke:** rebuild `web/dist`, restart `serve_web.py`, confirm the Analyze
  stage list shows no `mert` and a generation completes.

## 12. Out of scope → SP-C (successor project)

Retiring Beat3Tower *feature extraction* entirely: build a dedicated lightweight
BPM/onset/pace extractor; migrate pace data off `tracks.sonic_features` JSON;
rewrite the artifact universe gate to MuQ coverage; delete `stage_sonic` +
`beat3tower_extractor` subsystem; re-validate all rhythm/pace gating. Its own
brainstorm → spec → plan cycle, on the tree SP-B simplified.

## 13. Open items deferred to the plan

1. **S-score liveness check:** confirm whether the secondary "S" score feeds live beam
   scoring or diagnostics only (beam.py:671,1503,1645,1912) before simplifying its
   plumbing — determines whether the simplification needs a behavior-preservation test.
2. **Window-key slim-down (optional):** alias absent `_start/_mid/_end` keys to the
   base matrix at load instead of baking copies. Take only if free; skip if it grows
   the loader diff.
3. **`load_paths` destination:** `src/analyze/track_paths.py` vs folding into
   `muq_runner.py` — implementer's choice, with a unit test either way.
4. **`test_beat3tower_fallback.py` split:** keep whatever covers `stage_sonic`
   extraction (in-scope-to-keep), delete what covers the artifact bake.
