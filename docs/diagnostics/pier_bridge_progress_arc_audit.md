# Pier-Bridge Progress Arc Audit

Scope: audit current pier-bridge segment construction, beam search scoring, progress enforcement, genre penalty, and diagnostics/testing.

## Locations (Phase 0)
- Segment pool creation/filtering: `src/playlist/pier_bridge_builder.py` (`_build_segment_candidate_pool_scored`, `build_pier_bridge_playlist`), `src/playlist/segment_pool_builder.py`
- Beam expansion + ordering: `src/playlist/pier_bridge_builder.py` (`_beam_search_segment`)
- Transition scores: `src/playlist/pier_bridge_builder.py` (`_compute_transition_score`)
- Progress projection: `src/playlist/pier_bridge_builder.py` (`_beam_search_segment`, `progress_by_idx`)
- Genre penalty: `src/playlist/pier_bridge_builder.py` (`_beam_search_segment`, `cfg.genre_penalty_threshold/strength`)
- Run audits/diagnostics: `src/playlist/run_audit.py`, `src/playlist/pier_bridge_diagnostics.py`, `docs/run_audits/*.md`

## Feature Evaluation

### A) Progress-arc scoring term (enabled/weight/shape)
- EXISTS? partial
- WHERE? `src/playlist/pier_bridge_builder.py` (`_beam_search_segment`), `PierBridgeConfig` fields `experiment_progress_*`; wiring in `src/playlist/pipeline.py` under `pier_bridge.experiments.progress_arc`
- GAPS? Only supports enabled/weight/shape; always uses abs loss; no tolerance/dead-zone; no autoscale
- PLAN? Extend config and scoring to support tolerance, loss, autoscale, max_step as requested (feature-flagged to preserve defaults)

### B) Tolerance band / dead-zone around target progress
- EXISTS? no
- WHERE? n/a
- GAPS? No tolerance before applying progress penalty
- PLAN? Add `tolerance` to progress arc config and apply `err = max(0, abs(cand_t-target_t) - tolerance)`

### C) Loss function options (abs vs squared vs huber)
- EXISTS? no
- WHERE? n/a
- GAPS? Fixed abs loss only
- PLAN? Add `loss` + `huber_delta` config and compute per spec

### D) Max progress jump guardrail
- EXISTS? no (only monotonic gate)
- WHERE? `src/playlist/pier_bridge_builder.py` (monotonic check: `cand_t < last_progress - eps`)
- GAPS? No max forward jump control
- PLAN? Add optional `max_step` with `gate` or `penalty` mode

### E) Autoscaling arc weight (distance/steps)
- EXISTS? no
- WHERE? n/a
- GAPS? Constant weight regardless of AB distance or steps
- PLAN? Add autoscale config: disable under min distance, scale by distance, optional per-step scaling

### F) Precompute cand_t per segment pool
- EXISTS? yes
- WHERE? `src/playlist/pier_bridge_builder.py` (`progress_by_idx` computed once per segment, used in beam loop)
- GAPS? None for projection; still uses dict lookup in inner loop
- PLAN? Keep; no change needed unless we add arrays for speed

### G) Genre penalty caching
- EXISTS? no
- WHERE? n/a
- GAPS? Genre similarity computed per edge without cache
- PLAN? Add per-segment cache keyed by `(a_idx, b_idx)` and record hit/miss counters

### H) Genre penalty tie-break band
- EXISTS? no
- WHERE? n/a
- GAPS? Penalty applied universally whenever below threshold
- PLAN? Add optional `tie_break_band` (default disabled) and only apply penalty when candidates are within band of current best

### I) Arc tracking diagnostics (mean/p50/p90 dev, max jump)
- EXISTS? no
- WHERE? n/a
- GAPS? Audits do not quantify path arc adherence
- PLAN? Add per-segment diagnostics and run audit fields for chosen path deviations and max jump

### J) Tests around pier-bridge scoring
- EXISTS? partial
- WHERE? `tests/test_artist_style.py` has `test_progress_monotonicity_in_beam_search_segment`; other tests cover pool building, duration penalty, tuning
- GAPS? No tests for progress-arc loss/tolerance/autoscale/max_step/caching
- PLAN? Add unit tests for progress target shapes, tolerance, loss functions, autoscale, max_step gate/penalty, and basic cache accounting
