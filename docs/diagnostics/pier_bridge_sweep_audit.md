# Pier-Bridge Dial Sweep Audit

Phase 0 findings for the sweep harness.

## Existing A/B and audit infrastructure
- A/B reports: `docs/ab_runs/ab_summary_bowie*.md` (generated from `docs/run_audits/*.md`)
- Run audits: `docs/run_audits/*.md` (written when `pier_bridge.audit_run.enabled` or `--audit-run`)
- Entry points:
  - CLI: `main_app.py` (supports `--dry-run`, `--audit-run`, `--anchor-seed-ids`)
  - Programmatic: `src/playlist/ds_pipeline_runner.generate_playlist_ds` (supports `anchor_seed_ids`, `random_seed`, `overrides`)

## How to run with fixed piers / seed / RNG
- Fixed piers are supported via `anchor_seed_ids` (CLI `--anchor-seed-ids` or pipeline param).
- RNG is supported via `random_seed` in `generate_playlist_ds`.
- `overrides["pier_bridge"]["audit_run"]` enables run audit output.

## Metrics availability in run audits
- Available (via `### summary_stats` JSON):
  - `min_transition`, `mean_transition`, `below_floor_count`
  - `soft_genre_penalty_hits`, `soft_genre_penalty_edges_scored`
- Available (via `## 3) Pool / Gating Summary` JSON):
  - `candidate_pool_stats` and pool sizes
- Missing in markdown output:
  - Progress-arc tracking metrics (`mean_abs_dev`, `p50`, `p90`, `max_progress_jump`)
  - Genre cache stats (hit rate, hits/misses)

Sweep harness will parse the available metrics and compute post-hoc invariant metrics
from final tracklists + artifact embeddings (raw sonic smoothness, arc deviation,
max jump, monotonic violations, segment distance), so it does not rely on the
run audit markdown containing those fields.

Additional harness behaviors:
- Sweep can vary progress-arc knobs and beam width (testing-only) via pier-bridge config injection.
- Optional multi-seed mode aggregates metrics across seeds and emits an aggregate CSV/MD.
- Reports include tracklist hashes and per-hash difference diagnostics vs baseline.
