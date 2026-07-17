# Phase 1 Task 6 — width pinning + corridor-vs-legacy A/B (2026-07-17)

Measurement gate for `playlists.ds_pipeline.pier_bridge.pooling: corridor`
(dev flag; default remains `legacy`). Full evidence/method in
`.superpowers/sdd/p1-task-6-report.md` (gitignored — Dylan's copy). This note
is the committed, permanent record.

## Width pinning

`corridor_width_percentile` default pinned to **0.85** (was 0.90 provisional).
Method: 4 corpus artists (Bill Evans Trio, SADE, Alex G, The Strokes) x
open/dynamic, corridor flag on at width in {0.85, 0.90, 0.95}, compared
per-segment against the SAME cells run under legacy pooling using
`pier_bridge_builder.py`'s unconditional `"Segment N: ... pool_before=X
pool_after=Y"` line (fires for both strategies — a direct, apples-to-apples
per-segment pool-size comparison, not the `admitted`-only fallback the task
brief anticipated as a backstop). 0.85 gave the closest match to legacy
(mean corridor/legacy size ratio 0.89 vs 0.68 @0.90 and 0.41 @0.95).

## 12-cell A/B vs `phase0_corpus_validation.json`

Full table: `docs/corridor_baseline/phase1_corridor_ab.json` (12/12 cells
clean, `err=None`, `min_pool_backstop` 0/12, corridor health lines present
12/12 cells x every segment).

- **THE bar — Swirlies/home `below_floor` 4 -> 0: PASS.** min_T also jumped
  0.018 -> 0.459.
- `below_floor == 0` in all 12 cells (corridor), not just Swirlies/home.
- **min_T aggregate bar (>=10/12 flat-or-better, no cell worse by >0.03):
  FAIL.** 8/12 flat-or-better; 3 cells worse by more than 0.03 (SADE/home
  -0.100, SADE/open -0.048, Alex G/open -0.063).
- **distinct_artists +/-2 bar: FAIL** in 4/12 cells (The Strokes/open +4,
  Swirlies/home +4, Swirlies/open +5, Alex G/open -3).
- **wall <=2x bar: FAIL** in 6/12 cells, up to 4.35x (Swirlies/home) — the
  widening ladder's retry cost on harder/wider-universe segments.

## Cultural-drift A/B (spec section 7)

SADE + Alex G, open, corridor on, `genre_mode` strict vs off: genre_top stays
within the seed's cultural neighborhood in both modes for both artists (no
wild drift) — SADE stays R&B/soul-adjacent, Alex G stays
indie/bedroom/dream-pop-adjacent (off mode adds shoegaze into the top-8 and
drops jangle_pop, a mild broadening, not a genre swap). Eyeball verdict:
PASS (no hard bar).

## Verdict

THE bar and the cultural-drift check pass. Three of the four remaining bars
(min_T aggregate, distinct +/-2, wall <=2x) fail on a nontrivial minority of
cells. **Hold point — reported to Dylan, not advanced to Task 7** per the
task brief.
