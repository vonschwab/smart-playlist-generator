# Phase 2 Task 4 — `progress_arc_weight` retune probe

Branch `corridor-phase2`. Task 7 finding (`.superpowers/sdd/progress.md`):
"`progress_arc_weight` 0.25 default too weak vs corridor's tighter beam
(live at 3.0)". Confirmed first: `PierBridgeConfig.progress_arc_enabled`
defaults `False` (`src/playlist/pier_bridge/config.py`) — the whole feature
is off by default, so this is a documented-recipe deliverable (per the
brief), not an activation, unless the probe showed a clear win.

## Method

`progress_arc` forced `enabled=True` via post-policy `set_paths` on the
project's two canonical `SWEEP_CELLS` (`scripts/corridor_baseline/runner.py`:
Bill Evans Trio/open, Swirlies/home — the same 2 cells every prior contract
sweep in this branch used), weight in `{0.25 (shipped default, forced on),
0.5, 1.0, 2.0, 3.0}`, vs an `enabled=False` baseline. 12 real generations.
Raw data: `.superpowers/sdd/p2_task4_arc_weight_probe_raw.json`.

## Results

| weight | Bill Evans Trio/open min_T | jaccard vs baseline | Swirlies/home min_T | jaccard vs baseline |
|---|---|---|---|---|
| baseline (off) | 0.9034 | — | 0.7063 | — |
| 0.25 | 0.9034 (unchanged) | 0.818 | 0.7207 (+0.0144) | 0.476 |
| 0.5 | 0.8976 (-0.0058) | 0.429 | 0.7207 (+0.0144) | 0.476 |
| 1.0 | 0.8976 (-0.0058) | 0.395 | 0.7207 (+0.0144) | 0.476 |
| 2.0 | 0.8560 (-0.0474) | 0.463 | 0.7207 (+0.0144) | 0.476 |
| 3.0 | 0.8560 (-0.0474) | 0.463 | 0.7207 (+0.0144) | 0.476 |

`below_floor=0` at every weight, both cells.

## Where it becomes audible: cell-dependent, not a single threshold

`progress_arc`'s effective weight is scaled by pier-to-pier `ab_distance`
(and, if `autoscale.per_step_scale` were on, by segment length — off by
default, unchanged here) — see `beam.py:723-731`. This makes the "audible"
threshold a function of segment geometry, not a single universal number:

- **Swirlies/home**: already fully saturated at the shipped default `0.25`
  — identical `min_T` (0.7207) and track-set jaccard (0.476) at every tested
  weight from 0.25 through 3.0. For this cell, Task 7's "0.25 too weak"
  characterization does **not** hold; the term is already maximally audible
  (and a real +0.0144 min_T **improvement**) at the default.
- **Bill Evans Trio/open**: `0.25` is barely perturbing (jaccard 0.818 vs
  baseline — only ~18% of tracks differ, `min_T` byte-identical to arc-off).
  `0.5` is a clear step up (jaccard drops to 0.429, `min_T` starts moving).
  A second step appears at `2.0` (`min_T` drops further, -0.047 vs baseline)
  — the arc constraint trading transition quality for pacing conformity as
  weight increases. For this cell, Task 7's finding holds: 0.25 is
  genuinely weak, and by 3.0 the effect is unambiguous.

**Corridor-era effective range**: audible starting at `weight=0.5` in both
tested cells (jaccard 0.43-0.48 vs baseline, a real ~50% track-set
reshuffle) — a safer floor to recommend for anyone who wants to
*deliberately* experiment with this lever than the shipped `0.25`, which is
inconsistent (near-inert on one cell, already-saturated on the other).
Beyond `0.5`, further increases (`1.0` -> `2.0` -> `3.0`) show diminishing/
no further movement on Swirlies but a second real step on Bill Evans Trio
at `2.0` (transition-quality cost, not just reshuffling) — so `2.0-3.0` is
where the term starts trading off transition quality on a geometry where it
was already reshuffling tracks, not just where it becomes "audible."

## No clear win across cells — no activation recommended

Swirlies/home **improves** (+0.0144 at any nonzero tested weight);
Bill Evans Trio/open **degrades monotonically** as weight increases past
0.5 (-0.006 at 0.5-1.0, -0.047 at 2.0-3.0). This is a genuine trade-off, not
a one-sided win — per the brief's own instruction ("do NOT flip enabled
without evidence it improves... recommend to controller, don't
self-activate"), this does **not** meet that bar. `progress_arc_enabled`
stays `False` by default; the shipped `weight: 0.25` is left unchanged.
This probe is a documented tuning recipe for anyone who wants to opt in
deliberately (`docs/PLAYLIST_ORDERING_TUNING.md`'s new `progress_arc`
section), not an activation.
