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
