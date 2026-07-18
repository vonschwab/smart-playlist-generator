# Phase 1 width re-pin — corridor_width_percentile vs the real library-wide universe (2026-07-18)

Task 8's `restrict_bundle` fix (commit `13256f1`) widened Artist mode's
corridor universe from a pre-restricted slice (a few thousand tracks) to the
full ~43k-track library, invalidating Task 6's `corridor_width_percentile =
0.85` pin (calibrated against the old, amputated universe). This note
re-pins the default against the real universe. Full probe method + evidence:
`.superpowers/sdd/p1-width-repin-report.md` (gitignored — Dylan's copy).

## Width re-probe

4 corpus artists (Bill Evans Trio, SADE, Alex G, The Strokes) x open/dynamic,
`corridor_width_percentile` in {0.93, 0.95, 0.97, 0.99} (16 real generations).

| pct | mean per-segment size `\|ratio-1\|` (vs Task 6's legacy avg) | mean `\|min_T delta\|` vs legacy | worst min_T cell |
|---|---|---|---|
| 0.93 | 0.404 | 0.1225 | SADE -0.243 |
| 0.95 | 0.477 | **0.049** | SADE -0.078 |
| 0.97 | 0.441 | 0.133 | SADE -0.304 |
| 0.99 | 0.772 | 0.106 | BET -0.124 |

Size-parity narrowly favors 0.93, but the metric is confounded: Alex G and
The Strokes hit the `segment_pool_max=800` cap at every percentile <=0.97 (2
of 4 probe artists' "size" reflects the cap, not the percentile), making
size-parity noisy/non-monotonic. min_T quality recovery is the decisive,
unconfounded signal — **0.95 pinned** (smallest mean `|min_T delta|`, no
probe cell worse than -0.078). `src/playlist/pier_bridge/config.py` one-line
default change (0.85 -> 0.95) + additive re-baseline of the 4 config-snapshot
goldens (sole change per file, confirmed via `git diff`).

## Full 12-cell corpus vs legacy (`phase0_corpus_validation.json`)

`docs/corridor_baseline/phase1_final_corpus.json`, no `--set` (0.95 is now
the unconditional default). 12/12 cells clean, wall 299.2s, no stall.

| Bar | Result | Verdict |
|---|---|---|
| `below_floor == 0` all 12 cells | 12/12 (Swirlies/home 4->0 holds) | **PASS** |
| min_T flat-or-better >=10/12, no cell worse by >0.03 | **4/12** flat-or-better; 8/12 worse by >0.03, several badly (SADE/home -0.332, Aaliyah/home -0.253, Aaliyah/open -0.303, BET/home -0.138) | **FAIL** |
| distinct_artists +/-2 | 7/12 exceed (Alex G/home -7, Alex G/open -5, Swirlies/open +6, Swirlies/home +5, BET/home -3, Aaliyah/home -3, Aaliyah/open -3) | **FAIL** |
| wall <=2x per cell | 3/12 exceed (Swirlies/home 3.40x, Swirlies/open 2.92x, Strokes/home 2.40x); corpus total 299.2s vs legacy 242.8s = 1.23x | mixed — total fine, 3 cells over |
| every cell `err=None` | 12/12 | **PASS** |

**Honest read, not resolved unilaterally:** the width probe was open-only (per
this task's own scope, matching Task 6's precedent) and does not generalize
to `home` (strict/strict) cells or to a 5th artist (Aaliyah) outside the probe
set — both regress hard at 0.95 (SADE/home -0.33, Aaliyah both detents -0.25
to -0.30) despite the probe's own 4-artist/open-only signal looking clean.
**Spot-check vs AB3** (the old-universe, width=0.85 corridor baseline): SADE/home
does **not** return toward AB3's ~0.606 as expected — it lands at 0.374, a
miss vs that expectation. THE bar (Swirlies/home `below_floor`) and the
below_floor floor overall hold cleanly; the min_T and distinct bars do not.
This is a genuine, reported finding for Dylan's read, same posture as Task 6's
own hold point.
