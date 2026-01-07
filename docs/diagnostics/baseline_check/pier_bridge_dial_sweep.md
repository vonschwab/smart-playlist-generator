# Pier-Bridge Dial Sweep

## Missing Metrics

- Progress arc tracking metrics are not present in the run audit markdown output.
- Genre cache hit rate is not present in the run audit markdown output.

## Sweep Results

| label | mean_transition | min_transition | p90_arc_dev | max_jump | overlap | runtime_s | audit_path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.9748472704127307 | 0.9447188433768876 | None | None | 1.0 | 1.66 | `docs\diagnostics\baseline_check\run_audits\ds_dynamic_20260106T163749Z_70ede509.md` |

## Top Configs: Smoothness

- baseline: mean_transition=0.9748472704127307, min_transition=0.9447188433768876, p90_arc_dev=None, max_jump=None, overlap=1.0

## Top Configs: Pacing


## Top Configs: Balanced


## UI Dial Proposal

The following presets are derived from sweep outcomes. If pacing metrics are missing, use smoothness + overlap until arc metrics are exposed.

| preset | label | weight | tolerance | loss | max_step | autoscale | tie_break_band |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| Loose | baseline | None | None | None | None | None | None |
| Balanced | baseline | None | None | None | None | None | None |
| Guided | baseline | None | None | None | None | None | None |
| Rail | baseline | None | None | None | None | None | None |
