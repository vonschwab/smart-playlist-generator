# Cleanup list — parked / deferred items

Running list of built-but-parked, superseded, or minor tech-debt items to revisit.
Per CLAUDE.md Layer 4 ("activate fixes, never default to legacy"), parked items are
either revived+validated later or deleted — not left inert as the default. Newest first.

## Parked features

### Energy arc / pace-contour — PARKED 2026-07-01
- **What:** the arousal-based energy-arc + energy-step scoring (`energy_arc_strength`,
  `energy_arc_band`, `energy_step_*` in `PierBridgeConfig`; consumed live in
  `beam.py:1206`). The "first-class scored contour" extension (2-D [arousal, log-z-onset]
  stacking, SDD Tasks 1-3) is wired-but-off on branch `worktree-sp1-collapse-harness`,
  NOT on master.
- **Why parked:** measured redundant with MuQ for smoothness. On a real MuQ playlist,
  MuQ-similarity predicts energy-closeness (corr −0.25) ~2.5× better than tempo-closeness
  (−0.10), and energy along the playlist is already fairly smooth (2/29 adjacent jumps vs
  5/29 for tempo). An energy arc would add only *coarse* (arousal is a blunt signal) polish
  to something MuQ mostly does for free. Low marginal ROI; would need calibration + a
  by-ear audition to ship safely.
- **Revive only if:** we want INTENTIONAL directional energy dynamics (a deliberate
  build-up / comedown across the whole playlist) — the one thing MuQ can't do (it smooths,
  it doesn't *direct*). That's a "playlist has an arc" north-star feature distinct from
  anti-whiplash, and it needs its own calibration + audition.
- **It is NOT the tool for tempo whiplash** — see the feature note below.

## Higher-value gap surfaced (a FEATURE to build, not cleanup)

### Tempo-whiplash smoothness (onset-based)
MuQ is nearly blind to tempo (corr −0.10), so a real MuQ playlist has jarring adjacent
tempo jumps (measured 129→83 and 86→144 BPM; 9/29 transitions exceed 1.5× BPM). The lever
is a SOFT onset-based penalty — **onset over BPM** (BPM is meaningless on beatless/drone
tracks; the `bpm_trust` finding) and **soft over hard-gate** (hard tempo gates detonate the
relaxation cascade; the onset-band lesson). The BPM/onset bridge bands + soft penalties
already exist (`*_bridge_max_log_distance`, `*_bridge_soft_penalty_strength`) but are OFF
in `pace_mode: dynamic`. Open design question: an *adjacent-step* penalty (the tempo mirror
of `energy_step`) vs the *pier-relative* onset bridge band — the whiplash measured is
adjacent-transition, which the pier-relative band does not directly target.

## Identity fragmentation + near-dup dedup (surfaced 2026-07-01, Green-House run)

A Green-House playlist exposed two related identity gaps. Neither breaks the arc, but
both degrade diversity accuracy and let a near-duplicate slip in. Evidence: the
`artist_counts` block + tracklist of the 2026-07-01 Green-House generation.

- **Same artist counted as distinct identities (undercounts the per-artist cap).**
  Hiroshi Yoshimura appears as both `hiroshi yoshimura 吉村弘` (1) and
  `hiroshi yoshimura` (2) — the CJK-name-suffixed variant is not normalized to the
  romanized key, so he's 3 tracks scored as 1+2 against `max_artist_fraction`. Same
  latent pattern on `takahiko ishikawa 石川鷹彦` / `kohsei morimoto 森本浩正`
  (1 track each, so no cap impact *yet*). Identity resolution should fold a
  `"<romanized> <CJK>"` suffix form onto the romanized identity.
- **Cross-moniker aliases not collapsed.** `leon todd johnson` and `airport people`
  are the same artist under two names (per Dylan) but resolve to two identities — the
  same class as the "Smog ≟ Bill Callahan" follow-up noted in the `playlist-testing`
  skill. No mechanism maps same-person / different-project names together.
- **Near-dup slipped past the pool dedup (edge case).** Tracks 31→32 were both titled
  `underscore dash apostrophe`, identical 2:58 duration, back-to-back — almost certainly
  the same recording, published under the two artist strings above. The candidate-pool
  dedup is **artist+title** keyed (`Pier bridge candidate pool deduped: 572 → 564`), so
  two different artist strings on the same recording are not caught. Confirm it's one
  recording (metadata) vs two genuine tracks; if the former, identity-normalized dedup
  (or the alias map below) would catch it.

**Proposed fix — GUI context-menu artist-alias tool.** Right-click an artist →
"alias to…" → writes `alias → canonical identity` mappings that feed
`artist_identity` resolution (min_gap, per-artist cap, seed-artist exclusion) *and*
pool dedup. Solves all three above with one authoritative, user-editable source, and
is the same capability the Smog/Bill-Callahan follow-up needs. Priority: low–medium.

## Minor tech-debt (from `docs/HANDOFF_2026-06-30_muq_collapse_merge.md`, non-blocking)
- **Analyze stage-list triple-drift (item 2):** `web/src/components/ToolsPanel.tsx`
  `ALL_STAGES` (stale — lists `enrich`, missing `adjudicate`/`apply`/`popularity`) vs the
  canonical `request_models.py:ANALYZE_LIBRARY_STAGE_ORDER` vs `analyze_library.py`
  `STAGE_FUNCS`. `_clean_stages` runs ALL stages when the result is empty (silent footgun).
- **`doctor.py` Python floor (item 4):** checks ≥3.8; `pyproject.toml` requires ≥3.11. Align.
- **`pace_admission_floor` dead knob:** defined + threaded, never read in `candidate_pool.py`.
- **Stale CLAUDE.md gotchas:** `genre_conflict_min_confidence` / `_penalty_strength` are not
  shipped keys (real: `candidate_pool.genre_compatibility_*`); "162-d tower blend" → 163-d
  (rhythm PCA dim 10); "~455 genres" → 465 active / 1010 records.

## Bugs + inert knobs (from the documentation audit, 2026-07-01)

Found while auditing the codebase for the doc rewrite; verified against current master.
(The MuQ auto-fold footgun from the merge handoff is intentionally NOT listed — it was
fixed by `54d682c`.)

**Bugs**
- **🔴 `config.example.yaml` ships `transition_weights` that raise on the default learned variant
  — fresh-clone setup breaker.** The template sets `ds_pipeline.transition_weights` to
  `0.40/0.35/0.25` (`config.example.yaml:157-160`), which is **non-default** (the default is
  `DEFAULT_TOWER_TRANSITION_WEIGHTS = 0.20/0.50/0.30`, `artifacts.py:20`). On a no-tower sonic
  variant (mert/muq), `validate_tower_knobs` (`artifacts.py:404-439`, called from
  `pipeline/core.py:736`) **raises** on non-default `transition_weights`. So a fresh clone that
  copies `config.example.yaml → config.yaml` and generates on the default MERT artifact hits a
  `ValueError` on first generation. The live `config.yaml` doesn't hit it because it aligns them
  to the default `0.20/0.50/0.30`. **Fix:** set `config.example.yaml`'s `transition_weights` to
  `0.20/0.50/0.30` (or drop the block so they default) — the shipped `0.40/0.35/0.25` is the
  stale pre-v4.1 rhythm-heavy default that predates both the alignment fix and the guard.
- **`ProgressLogger` periodic-summary branch is dead in verbose mode:** `_should_emit()`
  returns `False` whenever `verbose_each=True` (`src/logging_utils.py:363-364`), so the
  `if self._should_emit()` inside the verbose branch of `update()` (`:401`) never fires. A
  `--verbose` run gets DEBUG-per-item plus only the final `finish()` summary — contradicting
  the class docstring's "also emit periodic summaries." Fix `_should_emit` or the docstring.
- **`--show-run-id` is inert on `main_app.py`:** the flag is parsed by `add_logging_args()`
  (`src/logging_utils.py:459`) and honored by `analyze_library.py` (`:2616`), but
  `main_app.py` never forwards `show_run_id` to `configure_logging()` — so CLI generation
  runs never stamp a run_id (unless `--debug` / json logs). Forward it, or drop the flag there.

**Inert / vestigial knobs**
- **`playlists.sonic.sim_variant` (tower_weighted / tower_pca):** vestigial on the learned
  sonic variants — when a variant is baked/pre-scaled, the load path ignores this knob and
  `sonic_variant.py` falls back to a passthrough (`tower_fallback=True`, guarded at
  `sonic_variant.py:390`). Only meaningful for the raw-tower artifact.
- **`tempo_stability`:** read only as a BPM-admission bypass (`candidate_pool.py`,
  `tempo_stability < bpm_stability_min` = 0.5), but per the bpm-trust finding it reads ~0.96
  for ~all tracks, so the bypass ~never fires (effectively inert). Confirm it ever fires
  before relying on it.

**Deprecated**
- **`analyze_library.py --beat-sync`:** DEPRECATED (legacy sonic mode disabled) — dead CLI flag, remove.
- **`genre_compatibility` (raw-tag pool penalty) — superseded by the dense PMI-SVD genre gate; delete.**
  Set OFF in live `config.yaml` 2026-07-01 (was erroneously `true` — default is `False`, absent from
  `config.example.yaml`, CLAUDE.md says keep off). It's a pool-level "candidate-vs-seed genre
  compatibility" filter (same altitude as the dense admission gate) implemented as **exact raw-tag
  overlap on the 410-dim vocab with identity affinity** (`compute_raw_genre_compatibility`,
  `genre_compatibility.py:68` → `np.eye` when `genre_affinity=None`, which is how both call sites
  invoke it). No relatedness: for a narrow seed it flags ~everything as conflict (Green-House run
  penalized 4307/4459 ≈ 97%). The **dense PMI-SVD admission gate** (`candidate_pool.py:772-847`,
  co-occurrence embedding = relatedness-aware) does the same job strictly better, so this is a
  redundant pre-dense predecessor. **Deletion is not a one-liner — two consumers:**
  1. `candidate_pool.py:905-931` — uses the `penalty` output, gated by `genre_compatibility_enabled`
     (now off). Remove the block + the four `cfg.genre_compatibility_*` fields (`config.py:55-58`,
     `585-...`) + the `config.yaml`/`config.example.yaml`/`CONFIG.md` keys.
  2. `artist_style.py:850-859` — calls it with `penalty_strength=0.0` and uses only
     `confidence`/`missing_or_sparse` to gate via `min_confidence` (`:886`). **Currently inert**
     (live path logs `min_confidence=None`). Decide: drop this confidence path, or re-source the
     signal from the dense embedding. Only after this is `genre_compatibility.py` (+ `tests/unit/
     test_genre_compatibility.py`) deletable.
  Also fix the stale doc pointers: `CONFIG.md:218` and `PLAYLIST_ORDERING_TUNING.md:350-358` still
  present it as a live-supported lever.

**Correction to an item above**
- The `doctor.py` Python-floor item (Minor tech-debt) is **RESOLVED** — `doctor.py:76-85` now
  checks 3.11+, not 3.8 (verified 2026-07-01). The list entry is stale.

## Deferred sub-projects

### SP-B: Remove MERT + Beat3Tower (deferred 2026-07-01, GATED on SP-A)
Once MuQ is self-sufficient in the analyze flow (SP-A: a `muq` extraction stage lands +
a real rebuild proves it), remove the deprecated MERT + tower sonic paths: the `mert`
analyze stage, the tower decomposition (X_sonic_rhythm/timbre/harmony + tower_weights/
transition_weights — already INERT under MuQ), the MERT/tower artifact bakes (X_sonic_mert*,
X_sonic_tower_weighted, X_sonic_rhythm/timbre/harmony*), the variant-switching machinery
(sonic_variant.py tower branches, the MERT side of the variant-aware transition calib), and
fold_mert / fold_2dftm. **DATA SAFETY: ARCHIVE — never delete — the MERT shards + mert_sidecar.npz
(~55h CPU, irreplaceable). Remove only code + artifact bake.** Rationale: at runtime MuQ is the
sole active sonic space; MERT/towers are unused rollback + artifact bloat + superseded fusion
research (MuQ beat MERT 86% vs 73% on trusted triplets). Dylan: no deprecated code for fallback.

**SP-B completion note (Task 9, 2026-07-02):** Tasks 1-8 landed — the MERT analyze stage, the
MERT/tower artifact bakes, and the tower/MERT variant-switching code are gone from `src/`; MuQ
(`X_sonic_muq`) is the sole live sonic space (`artifacts.sonic_variant_override: muq`). Task 9
(this sweep) retired the matching config surface — `tower_weights`/`transition_weights`/
`tower_pca_dims` (`playlists.ds_pipeline`), `playlists.sonic.sim_variant`, and `analyze.mert` —
from both `config.example.yaml` and the live `config.yaml`, and updated CLAUDE.md's key-paths/
gotchas/Layer-3 items to the MuQ-only reality. Task 10 (artifact rebuild) and Task 11 (MERT
shard archival to `data/archive/mert_2026/`) are still open — the MERT shards + sidecar remain
in place under `data/artifacts/beat3tower_32k/` until then.

### SP-C: Retire Beat3Tower extraction (proposed 2026-07-02, follow-on to SP-B)
Once SP-B lands (Task 10 rebuild + Task 11 archive), the Beat3Tower *extraction* path itself —
the `stage_sonic` analyze stage + its extractor, which populate `tracks.sonic_features` and
gate the analyzable-track "universe" — is the next thing to retire: build a dedicated,
lightweight BPM/onset/pace extractor (rhythm-only; that's the pace gate's actual dependency per
the `bpm_trust`/`pace_admission_floor` history), migrate off `tracks.sonic_features`, rewrite
the universe/coverage gate to check MuQ coverage instead of tower coverage, delete `stage_sonic`
+ the tower extractor module, and re-validate the BPM/onset pace gates (`pace_mode: strict` /
`narrow`) against the new extractor's output before calling it done. Not started — needs its own
brainstorm/spec/plan.

## Research-harness modernization (surfaced during SP-B Task 9 review, 2026-07-02)

The following `scripts/research/*.py` audition/eval harnesses hardcode or pre-flight-check a
literal `"mert"` sonic-variant label, or read tower/MERT artifact keys that SP-B has removed
from `src/` (the artifact itself still has them until the Task 10 rebuild). They will abort, or
silently degrade, against a muq-only artifact until modernized — read the live/stamped variant
or take it as an explicit CLI arg instead of hardcoding `"mert"`. Not fixed now — SP-B follow-up:
- `collapse_eval.py:414` — aborts unless the live sonic variant == `"mert"`.
- `slider_differentiation_eval.py:428` — aborts unless the live sonic variant == `"mert"`; also
  `bpm_array` silently degrades at `:61` if its expected key is absent.
- `adaptive_admission_eval.py:476` — warns but proceeds unless variant == `"mert"`.
- `energy_neighborhood_probe.py:62-64` — hardcodes `set_sonic_variant_override("mert")`.
- `energy_spread_eval.py:117` — hardcodes the `--sonic-variant` CLI default to `"mert"`.
- `collapse_rescore.py:29,149,155` — hardcodes `VARIANTS = ["mert", "muq"]` and prints
  MERT-labeled columns unconditionally.
