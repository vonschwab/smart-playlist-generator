# Phase 2 Task 4 — narrow/discover corridor width calibration

Branch `corridor-phase2`. Replaces the provisional interpolations pinned by the
Phase 1 per-mode-width task (`.superpowers/sdd/p1-permode-width-report.md`):
`corridor_width_percentile_narrow = midpoint(strict, dynamic) = 0.9675` and
`corridor_width_percentile_discover = dynamic - 0.02 = 0.93`, both explicitly
flagged there as "provisional interpolations... their quality profile is
unverified."

## Method

3 artists (SADE, Swirlies, Alex G) x **both detent families that actually
exercise the modes**: narrow via the GUI's `close` detent (`sonic_mode=narrow`
+ `genre_mode=narrow` together, matching real GUI usage — not an isolated
sonic-only override), discover via `wander` (`sonic_mode=discover` +
`genre_mode=discover`). `scripts/corridor_baseline/runner.py`'s `DETENTS` dict
gained `close`/`wander` entries (previously only `home`/`open` existed) mapped
straight from `src/playlist_gui/policy.py::DIAL_TO_AXES`'s `range` dial.

3 candidate widths per mode, bracketing each interpolation:
narrow `{0.96, 0.9675, 0.975}`, discover `{0.92, 0.93, 0.94}` — applied via
`run_cell(set_paths={"playlists.ds_pipeline.pier_bridge.corridor_width_percentile_<mode>": w})`,
the same post-policy override seam the Phase 1 width probes used (bypasses the
sonic_mode mapping so the width lever is isolated). 18 real generations, wall
335.3s. Raw data: `.superpowers/sdd/p2_task4_width_probe_raw.json`.

## Narrow (`close`) results

| width | SADE min_T | Swirlies min_T | Alex G min_T | mean | min (worst-case) | below_floor |
|---|---|---|---|---|---|---|
| 0.96   | 0.6408 | 0.5939 | 0.5959 | 0.6102 | 0.5939 | 0/3 |
| 0.9675 | 0.6266 | 0.5939 | 0.6228 | 0.6144 | 0.5939 | 0/3 |
| **0.975** | 0.6005 | 0.6297 | 0.6190 | **0.6164** | **0.6005** | 0/3 |

Corridor sizes are real and monotonic (higher percentile = smaller pool),
e.g. SADE: 187 (0.96) -> 152 (0.9675) -> 117 (0.975); no `segment_pool_max`
cap hit at any narrow width for any artist. 0.975 wins on **both** aggregate
mean min_T and worst-case (min) min_T across the 3 artists — a more decisive
win than the small mean gap alone suggests, since it also has the best floor.

## Discover (`wander`) results

| width | SADE min_T | Swirlies min_T | Alex G min_T | mean | min (worst-case) | below_floor |
|---|---|---|---|---|---|---|
| 0.92 | 0.6408 | 0.6233 | 0.5490 | 0.6044 | 0.5490 | 0/3 |
| 0.93 | 0.6408 | 0.6233 | 0.5490 | 0.6044 | 0.5490 | 0/3 |
| **0.94** | 0.6408 | 0.6233 | 0.6228 | **0.6290** | **0.6228** | 0/3 |

0.94 wins decisively on both mean and worst-case — Alex G/wander specifically
improves +0.0738 at 0.94 vs 0.92/0.93 (tied). Swirlies and Alex G hit
`segment_pool_max=800` post-cap at **every** tested discover width (same
cap-saturation pattern the Phase 1 per-mode-width report's dial audit found
for `open`/`wander`) — the 0.94 win traces to which candidates enter the
pre-cap top-800 ranking at the tighter threshold, not to post-cap size
differentiation. SADE is not cap-bound (sizes 566 -> 495 -> 424 across
0.92 -> 0.93 -> 0.94) and stays byte-identical across the whole bracket — a
legitimate beam-convergence saturation (SADE's winning 30-track sequence's
candidates all survive well within the tightest tested width; below 0.96 in
the narrow sweep the same sequence is also still found, but tightening past
~0.9675 excludes a candidate it needs, forcing a different, worse path —
consistent with the narrow-sweep numbers above, not a probe bug).

## Pinned values

| sonic_mode | old (provisional) | new (pinned) | basis |
|---|---|---|---|
| narrow | 0.9675 | **0.975** | best mean AND worst-case min_T across 3 artists; below_floor=0 at every tested width |
| discover | 0.93 | **0.94** | best mean AND worst-case min_T across 3 artists (Alex G/wander +0.074); below_floor=0 at every tested width |

Both interpolated values were **superseded, not confirmed** — the direct
probe moved both away from their midpoint/offset derivations. Changed
`src/playlist/pier_bridge/config.py`'s `corridor_width_percentile_narrow`
(0.9675 -> 0.975) and `corridor_width_percentile_discover` (0.93 -> 0.94)
defaults; re-baselined the 4 config-snapshot goldens under
`tests/unit/goldens/pipeline/*.json` (sole diff per file: these two fields).

## Mini-corpus regression check

3 probe artists (SADE, Swirlies, Alex G) x `home`/`open`, vs
`docs/corridor_baseline/phase2_task3_corpus.json` (the current, post-Task-3
baseline). `home` = `sonic_mode=strict`/`genre_mode=strict` (uses
`corridor_width_percentile_strict`, unchanged 0.985); `open` =
`sonic_mode=dynamic`/`genre_mode=dynamic` (uses `corridor_width_percentile_
dynamic`, unchanged 0.95) — neither reads the narrow/discover fields this
task changed, so byte-identical results were the correctly-predicted outcome,
verified rather than assumed:

| Cell | baseline min_T | new min_T | delta |
|---|---|---|---|
| SADE/home | 0.6676 | 0.6676 | 0.0000 |
| SADE/open | 0.6408 | 0.6408 | 0.0000 |
| Swirlies/home | 0.7063 | 0.7063 | 0.0000 |
| Swirlies/open | 0.6233 | 0.6233 | 0.0000 |
| Alex G/home | 0.6190 | 0.6190 | 0.0000 |
| Alex G/open | 0.6228 | 0.6228 | 0.0000 |

6/6 byte-identical (below `0.03` regression bar trivially; `below_floor`/
`distinct_artists` also unchanged in every cell). Raw check:
`.superpowers/sdd/p2_task4_minicorpus_check.json`.

## Gates

- `tests/unit/test_pipeline_smoke_golden.py`: 4/4 passed (config-snapshot
  goldens re-baselined deliberately, sole diff `corridor_width_percentile_
  narrow`/`_discover` per file).
- `tests/unit/test_pier_bridge_smoke_golden.py` +
  `tests/integration/test_playlist_golden_files.py`: 15/15 passed, unaffected
  (synthetic fixtures don't exercise narrow/discover in a way this change
  moves).
- `tests/unit/test_corridor_width_percentile_mode.py`: 10/10 passed (tests the
  pure resolver against its own local fixture kwargs, not the shipped
  defaults — unaffected by design).
- `tests/unit/test_corridor_baseline_runner.py` +
  `tests/unit/test_corridor_baseline_patterns.py`: 14/14 passed after adding
  `close`/`wander` to `DETENTS`.
