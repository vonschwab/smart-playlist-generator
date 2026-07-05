# GUI Dials — design spec (2026-07-04)

Replace the four mode selects (cohesion / genre / sonic / pace) in the browser GUI
with three intent dials — **Range**, **Flow**, **Pace** — plus a per-generation
**receipt line**. Grounded in the 2026-07-04 slider-differentiation re-eval
(`docs/run_audits/slider_differentiation_2026-07-04/`) and a UX-research pass.

## Problem

The four mode selects earned distrust: for months they were effectively inert
(absolute floors collapsed strict==narrow / dynamic==discover; roam clamp zeroed
the percentile gate; cohesion leaked into the genre floor). Dylan's real workflow
became "leave everything on Dynamic, steer vibes with Style." The 2026-07-04
fixes (8a0d675, 3403ef9, e328a87) made the axes genuinely differentiating — but
the *surface* still speaks engine language, has 4x4-5 combinations nobody
navigates, and gives zero feedback that a setting did anything.

## Research grounding (digest)

Strongly evidenced (peer-reviewed / field studies):

1. Perceived control lifts satisfaction only when paired with visible causal
   feedback (Bostandjiev et al., TasteWeights, RecSys 2012).
2. Mechanism ambiguity erodes trust faster than bad output (Harambam et al.,
   RecSys 2019).
3. Users tune once, then stop touching controls (Ekstrand et al., RecSys 2015)
   → design for a one-time "find my setting" ritual + persistent default.
4. Use one vocabulary across controls and explanations (Millecamp et al., IUI 2019).
5. Offer control in tiers — presets/defaults (low), dials (medium); do NOT keep a
   raw parameter panel as a third tier (Jannach et al., control survey, 2016).
6. Controls with no perceptible effect boost satisfaction briefly, then collapse
   trust when discovered (Langer 1975; "placebo buttons").
7. Fewer controls reduce decision load independent of expressiveness (Hick's law).

Practitioner (NN/g, Nielsen heuristics):

8. Continuous sliders are wrong for non-perceptible parameters without realtime
   feedback (generation takes 20-60s) → discrete labeled detents.
9. Labels in the listener's language, not engine terms (heuristic #2).
10. Show what the system DID with the setting, from real run data (heuristic #1).
11. Cheap, obvious reset (heuristic #3).
12. Detent count must not exceed measurably distinct behaviors (JND argument) —
    audited by the differentiation harness, our standing instrument.

Product precedents (Spotify AI DJ, Pandora, Apple Music Autoplay, Roon): nobody
ships a raw continuous taste knob; discrete states, presets, or intent language.

## Decisions fixed during brainstorm

- Real usage: defaults + Style steering; controls died from earned distrust.
- Dylan's wished-for knobs: Familiar↔Adventurous, Smooth↔Eclectic flow,
  tempo/energy discipline. (Deep-cuts dial = separate roadmap item.)
- Feedback: **receipt line** under the result (chosen over dial annotations).
- Honesty rule: **honor + confess** — the engine applies a setting as far as it
  safely can and the receipt states any limitation explicitly. No silent no-ops,
  no silent relaxation.
- Approach A: three dials, four selects deleted from the GUI; axes remain the
  engine/config vocabulary.
- Flow poles renamed after mechanics review: the cohesion axis is
  "maximize join smoothness ↔ commit to traveling between seeds", NOT
  "smooth ↔ eclectic". Eclecticism is Range's job.

## The dials

Segmented detent controls (styled buttons, existing Tailwind idiom — not rotary
widgets, not continuous sliders). Every detent maps to a named, grid-verified
engine state. Factory default = Open / Balanced / Natural = today's all-dynamic.

### Range — "how far from home the music can come from"
Compiles `sonic_mode` + `genre_mode` jointly (pool width).

| Detent | sonic_mode | genre_mode | Evidence |
|---|---|---|---|
| Home | strict | strict | pools distinct 6/6 artists (post-fix grid) |
| Close | narrow | narrow | distinct 6/6 |
| **Open** (default) | dynamic | dynamic | baseline |
| Wander | discover | discover | sonic-discover pools distinct 6/6 |

`off` states are debug-only (config), not on the dial.

### Flow — "how the playlist moves"
Compiles `cohesion_mode` (beam objective: bridge-progress weight vs pure
transition quality).

| Detent | cohesion_mode | Meaning |
|---|---|---|
| Drift | discover | pure-T beam (weight_transition=1.0): glides join-to-join, goes wherever smooth leads; best worst-edge on 4/6 artists |
| **Balanced** (default) | dynamic | current default blend |
| Journey | strict | destination-committed: each step advances seed→seed, accepts seams for shape |

`narrow` dropped from GUI (JND collapse; config-only).

### Pace — "tempo discipline"
Compiles `pace_mode`.

| Detent | pace_mode | Note |
|---|---|---|
| Steady | narrow | chosen over strict: near-identical BPM compression, lower crater risk (Codeine minT 0.177 vs 0.063); strict = config-only |
| **Natural** (default) | dynamic | baseline |
| Free | off | no tempo constraint; receipt's seam number covers floor dips |

Dial + detent words are the single vocabulary for tooltips, receipt, and any
future explanations. Engine terms never appear in the GUI.

## The receipt line

One compact line under the result header, expandable; composed by the worker
from stats the generation already produced (never re-derived):

```
30 tracks · 24 artists
⚙ Range: 445 tracks in reach · Flow: roughest seam 0.58 (avg 0.81) · Pace: ±9 BPM
   ⚠ 2 notes                                                        [details ▾]
```

- Range → candidate-pool admitted count (pool stats).
- Flow → `min_transition` / `mean_transition` (playlist_stats; already in
  `schemas.py` today).
- Pace → BPM spread from the pier-bridge `bpm_summary` stat.
- Notes (confessions), only when true, one clause each, dial vocabulary:
  beatless seed → tempo band disabled; N sonic connectors rescued past the
  Range gate; relaxation ladder engaged (rung named); seam repaired post-build
  (tail-DP / repair / edge-delete); pier vetoed as unbridgeable.

Plumbing: worker composes a `receipt` object → additive optional field on the
NDJSON result and web response schema → GUI renders it. No engine changes.

## Defaults, reset, persistence

- Dials persist via localStorage (`pg_*` pattern, three keys). Old
  `pg_axes.*` / `pg_cohesion` keys are ignored (no migration; factory default).
- `↺` reset returns all three dials to factory default; visible whenever any
  dial is off-default.
- Off-default dials show a subtle marker (dot on label).
- No preset-management/named-profiles UI (YAGNI; Ekstrand).
- Style popover untouched (orthogonal: steers *where*; dials set *how*).

## Wiring

- **`DIAL_TO_AXES` mapping table in `policy.py`** — the single source of truth;
  each detent names its exact axis quadruple. Web layer translates dials →
  existing `UIStateModel` axis fields at the API boundary;
  `derive_runtime_config` unchanged downstream. Policy remains sole mode owner
  (modes bypassing policy go silently inert — standing lesson).
- `GenerateRequestBody`: + `range_dial`, `flow_dial`, `pace_dial`; the four
  `*_mode` request fields removed (GUI is the API's only client).
- `GenerateControls.tsx`: four selects replaced by three segmented controls.
- Untouched: `config.yaml` axes (engine tuning + CLI), all per-mode engine
  knobs, gui_fidelity harness (drives the axis level, which persists).
- Rollback: git revert of the GUI/API layer; engine unchanged.

## Honesty tests (acceptance)

1. **Placebo ban**: extend `scripts/research/slider_differentiation_eval.py`
   with dial-level sweeps (combined detents through the real policy path).
   Acceptance on Codeine + Bill Evans Trio: every adjacent detent pair produces
   measurably different playlists on ≥1 of the two (track-set Jaccard overlap
   ≤ 0.8), and no detent drops the worst live transition below the repair
   threshold (0.30) on the jazz corpus. (Closes the single-axis-sweep gap: Range
   moves two axes at once.)
2. **Mapping unit tests** (fast tier): every detent → exact axis quadruple;
   request→UIStateModel translation; receipt composer against canned stats
   including every confession trigger.
3. **Receipt truth test** (integration): rendered numbers == the producing
   run's stats.
4. **Manual click-through** after `web/dist` rebuild + worker restart
   (web-gui trap list) before declaring shipped.

## Out of scope

Kitchen crater (hybrid-floor follow-up, intersects MuQ quiet-collapse);
YLT strict≡narrow residual; deep-cuts dial (separate roadmap item);
any Style/steering changes.
