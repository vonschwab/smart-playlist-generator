# Phase 2 Task 1 — Mechanism probes: does corridor admission actually exclude the ramp?

**Branch:** `corridor-phase2` @ 6d81c19. **Status: evidence gate — read before building anything.**
Reproductions use the production loader (`src.features.artifacts.load_artifact_bundle`) and the
production corridor functions (`seed_genre_relevance_mask`, `build_eligible_universe`,
`build_corridor`) directly against the current `data/artifacts/beat3tower_32k/data_matrices_step1.npz`
— no engine code was changed. Scripts live in the session scratchpad
(`probe1_pc_corridor.py`, `probe2_sade_corridor.py`); numbers below are their actual output.

**Headline finding: the Phase 1 hypothesis (corridor min-sim excludes the ramp connector) is
REFUTED for both deep-dived cases.** In both, the better connector was already sitting inside the
admitted corridor pool. The beam picked a worse path anyway, and the two repair mechanisms that
exist to catch this (tail-DP endgame, the widening ladder) both stood down because their floor
gates (`tail_dp_floor=0.3`, `transition_floor=0.2`) are set below what "not broken" looks like, not
below what's *achievable*. **Candidate A (force-include) and Candidate B (support-triggered
pre-beam widening) both operate on pool admission — the wrong lever for these cases**, because the
pool was never the problem.

A caveat survives: across the wider 12-cell corpus (Probe 3), narrow/`home`-style corridors (width
0.93–0.98, pools of 30–290) do show real pool-admission exclusion in some segments, so the original
hypothesis is not universally false — it just isn't the dominant mechanism at the width the shipped
`dynamic` mode actually uses (0.95, and pools up to the 800 cap), which behaves like the `open`
detent below.

---

## Probe 1 — Parquet Courts, segment 4 (Human Performance → Into the garden)

**Run:** satellite `logs/playlists/2026-07-18_174422_Parquet_Courts_b06d13.log` (corridor, `mode=dynamic`,
`corridor_width_percentile_dynamic=0.95`). Anchors: pier A = idx 17762 (Human Performance), pier B =
idx 17801 (Into the garden, the outlier, logged `anchor_support_b=0.28`, `anchor_support_a=0.07`).
Logged health line: `Corridor[seg 4]: size=800 width=0.95 widened=0 support_a=0.07 support_b=0.28
threshold=0.404 capped=True`. Final emitted edge for this segment: Ava Luna "Black Dog" → Into the
garden, **T=0.394** (the playlist's global weakest edge).

**Reproduction** (`load_artifact_bundle` + `seed_genre_relevance_mask` + `build_eligible_universe` +
`build_corridor`, same call shape as `pier_bridge_builder._build_corridor_segment_pool`):

| | logged (generation time) | reproduced (now) |
|---|---|---|
| library size | 38,844 | 43,241 (library grew between the run and this probe — appended rows only; pre-existing indices are stable, confirmed by idx 17762/17801 matching the log's own `idx=` annotations) |
| corridor eligible | 20,702 | 22,556 |
| threshold | 0.404 | 0.412 |
| support_a / support_b | 0.07 / 0.28 | 0.060 / 0.300 |

Close enough (the library-growth delta explains the residual gap) to trust the qualitative result.

**Sonic Youth "Theresa's Sound-World" (idx 21331)** — the connector legacy's tail-DP swapped in
(window min 0.018 → 0.714, per the canonical log `2026-07-18_181117_Parquet_Courts_ad8689.log`):

- `sim_a` (to Human Performance) = **0.4350**, `sim_b` (to Into the garden) = **0.7309**, `min_sim = 0.4350`.
- **0.4350 > threshold (0.404 logged / 0.412 reproduced) → she PASSES corridor admission.**
  `build_corridor` confirms `theresa_idx in result.indices == True`.
- Rank among library tracks by sim-to-"Into the garden": 41st overall (of 43,241).
- **Of "Into the garden"'s top-30 sonic neighbors: 30/30 pass genre+title eligibility, only 3/30
  are corridor members at width=0.95** (the rest fail on low `sim_a`, i.e. they're close to the
  outlier but not to the segment's other anchor — exactly the hypothesized shape). Theresa's own
  min-sim (0.435) is comfortably mid-pack among the neighbors that DO clear the bar.
- **Candidate B width sweep**: Theresa is a corridor member at every tested width (0.95 down to
  0.70) — she needs **no widening at all**; she was never excluded.

**So why wasn't she used?** The satellite log's segment-4 top-10-by-final dump (DEBUG line 1187)
does *not* list her (her sonic-only hmean ≈ 0.545 ranks outside the top 10 shown, though not far
outside — several printed candidates score 0.58–0.67). The chosen final tail was
…Pardoner→Editrix→Ava Luna→Into the garden, whose window T-values are 0.957 / 0.878 / **0.394**.
`Tail-DP summary: applied=0/5 segments` — tail-DP was attempted for every segment (counter
increments unconditionally) but applied nowhere. Reading `optimize_segment_tail`
(`src/playlist/pier_bridge/tail_dp.py:109-110`):

```python
if floor is not None and float(floor) > 0.0 and old_min >= float(floor):
    return None  # landing window already >= floor: leave the beam's choice
```

`old_min` for this window = min(0.957, 0.878, 0.394) = **0.394**, and `tail_dp_floor = 0.3`. Since
0.394 ≥ 0.3, **tail-DP never even searched** — it short-circuited before looking at any candidate,
including Theresa, who is confirmed present in `td_candidates` (= `last_segment_candidates`, the
segment's full accepted corridor membership minus used/banned-artist tracks — Sonic Youth was used
nowhere else in this playlist, so nothing blocks her by identity either). The widening ladder's own
gate uses `transition_floor=0.2`, also comfortably cleared by 0.394, so it doesn't widen either
(`widened=0` is correct, matching the health line).

**Verdict: REFUTED as stated.** Corridor admission is not the exclusion mechanism here — Theresa was
admitted, reachable, and unused-elsewhere. The actual mechanism is a **floor-gate under-trigger**:
both post-beam repair mechanisms (tail-DP, widening ladder) treat "clears 0.3 / clears 0.2" as
"good enough," even when material 40+ points better (T ≈ 0.7–0.8, per legacy's realized swap) sits
in the very same pool.

---

## Probe 2 — SADE/home, segment 0 (Siempre Hay Esperanza → mini-pier waypoint)

**Run:** `logs/corridor_baseline/1784391407392665000_corpus_SADE_home_757827.log` (from
`phase1_final_corpus2.json`'s SADE/home cell — `min_transition: 0.4539109938096365`). `genre_mode=strict`
(relevance floor 0.30), `width_percentile=0.98` (this corpus harness's "home" detent, not the shipped
`dynamic` 0.95). Weakest edge (bottom of "Weakest transitions"): Sade "Siempre Hay Esperanza" → Isaac
Hayes "Let's Stay Together", **T=0.454** — this is segment 0's *first* edge (pier A → first interior),
not a landing edge into pier B.

Segment 0 health line: `Corridor[seg 0]: size=70 width=0.98 widened=0 support_a=0.46 support_b=0.53
threshold=0.506`. **Note: this is NOT the lowest-support segment in the run** (segment 1 has
support_b=0.18, segment 4 has support_a=0.22) — the weakest edge did not occur at the worst-support
segment, already a partial break from the "worst edge sits at lowest support" pattern.

**Reproduction:** rebuilt the same eligible universe (`genre_mode=strict` floor=0.30) and corridor.
Reproduced threshold 0.497 vs logged 0.506 (again, library-growth residual: 43,241 rows now vs 41,257
then). Isaac Hayes (the actual choice): `sim_a=0.563, sim_b=0.542, min_sim=0.542` — a corridor member,
comfortably above threshold.

**Top-30 library neighbors of pier A by `sim_a`:** only **8/30 pass genre eligibility** (the other 22 —
Bob James, Kamasi Washington, Kokoroko, Greg Foat, Fela Kuti, etc. — are sonically very close to
"Siempre Hay Esperanza" (`sim_a` 0.70–0.84) but fail the `genre_mode=strict` 0.30 relevance floor:
jazz-funk/broken-beat/spiritual-jazz reads as too far from the pier set's aggregate genre profile).
**This half of the mechanism — a genre-relevance-mask exclusion of high-sonic-similarity candidates —
genuinely matches the "corridor excludes good ramps" shape**, though the exclusion is the *genre*
mask, not the *sonic* min-sim threshold Probe 1 tested.

But: **the single best-scoring admitted candidate, Plunky & Oneness of Juju — "Love's Wonderland
(Extended Version)" (idx 18695, `sim_a=0.725, sim_b=0.697, min_sim=0.697`), passes every gate
(genre-eligible AND a corridor member) and still beats the actual choice (Isaac Hayes, min_sim=0.542)
by a wide margin — yet the beam did not pick it.** Same pattern as Probe 1: a materially better,
fully-admitted connector existed and wasn't used. Genre-steering routed this segment through
`['smooth soul', 'yacht rock', 'jazz pop']` waypoint labels (log line 222) — Isaac Hayes is a much
better *genre-arc* fit for "smooth soul" than the more Afro-funk/spiritual-jazz Plunky track, so this
may be a deliberate sonic/genre-arc tradeoff rather than a bug, but it's not a pool-exclusion story
either way.

**Structural note:** this weak edge is the segment's *first* edge (departure from pier A), not the
last. `optimize_segment_tail` only ever re-opens the **last** 1–2 interior slots before pier B — it
structurally cannot touch a bad first edge. There is no symmetric "head-DP." `Tail-DP summary:
applied=0/6` for this run too, but for segment 0 specifically it's moot: even an unconditional
tail-DP wouldn't reach this edge.

**Candidate B (widening) would not have helped either**: corridor widening only relaxes the sonic
`width_percentile`, never the separate genre-relevance floor — so widening 0.98→0.90→0.70 (confirmed
by direct sweep: support climbs to 0.83/0.91 by 0.90, 0.97/0.96 by 0.70) never re-admits Bob
James/Kamasi Washington/Kokoroko, because they're excluded by genre, not sonic width.

**Verdict: PARTIAL.** The genre-relevance-mask half of the mechanism is real and matches the
hypothesis's shape (many good sonic neighbors excluded) — but the specific weakest edge in this run
is explained by a *different, already-admitted* candidate not being chosen, same as Probe 1, plus a
structural gap (no head-side repair) neither Candidate A nor B addresses.

---

## Probe 3 — parameter evidence across the 12-cell corpus (`phase1_final_corpus2.json`)

All `Corridor[seg N]` health lines + weakest-transition edges pulled from the 12 authoritative cells
(SADE/BET/Strokes/Swirlies/Aaliyah/Alex G × home/open, log paths in the JSON). `min(support_a,
support_b)` per segment vs. whether that segment contains the run's globally-weakest edge:

| Cell | width | lowest-support segment (min(a,b)) | weakest-edge segment | match? |
|---|---|---|---|---|
| SADE/home | 0.98 | seg1 (0.18) | seg0 (min=0.46) | **no** |
| SADE/open | 0.95 | seg0 (0.38) | — (min_T=0.62, healthy) | n/a |
| Bill Evans Trio/home | 0.98 | seg4 (0.08) | seg4 (T=0.693) | yes |
| Bill Evans Trio/open | 0.95 | seg4 (0.17) | seg4 (T=0.776) | yes |
| The Strokes/home | 0.98 | seg2/4/5 (~0.32–0.34) | seg0-ish (T=0.739) | partial |
| The Strokes/open | 0.95 | seg2 (0.62, i.e. no low segment) | seg? (T=0.534, still weak) | **no — weak edge despite uniformly healthy support** |
| Swirlies/home | 0.98 | seg2 (0.10) | seg-with-T=0.359 | yes |
| Swirlies/open | 0.95 | none low (min 0.49) | T=0.462 still weak | **no** |
| Aaliyah/home | 0.98 | seg4 (0.20) | T=0.768 | yes |
| Aaliyah/open | 0.95 | none low (min 0.54) | T=0.561 still weak | **no** |
| Alex G/home | 0.98 | seg0 (0.23) | T=0.466 | yes |
| Alex G/open | 0.95 | none low (min 0.64) | T=0.454 still weak | **no** |

**Pattern: low anchor_support predicts the weak-edge segment reasonably well in the narrow (`home`,
width 0.93–0.98, pool size 30–290) detent — 4/6 clear matches, 1 partial, 1 miss (SADE). In the
`open` detent (width 0.95, pool often capped at 800 — the shape the shipped `dynamic` mode actually
uses) support is uniformly healthy (min 0.38–0.97, mostly >0.5) in every cell, yet weak edges (T
0.45–0.62) occur in 4/6 cells anyway.** This is the same signature Probes 1–2 found by hand: once
the corridor is wide enough (which `dynamic`'s 0.95 + 800-cap already is), admission stops being the
bottleneck, and the weak edges that remain come from the beam/tail-DP layer, not the pool.

**Support-threshold recommendation, with the caveat above:** across the `home`-detent segments where
correlation held, "problem" segments cluster at `min(support_a, support_b) ≤ 0.20` (BET home 0.08,
Aaliyah home 0.20, Swirlies home 0.10/0.14, Alex G home 0.23/0.24 boundary) while segments NOT
implicated in a weak edge mostly sit ≥ 0.30. **A `corridor_ramp_support_threshold` around 0.20–0.25
is defensible for the narrow-corridor regime** — but per Probes 1–2, this threshold would not have
fired usefully for the `dynamic`/`open`-style shipped default, where support rarely drops that low
in the first place (PC segment 4 was the exception at 0.07, and even there the pool already
contained the fix).

**Force-include N:** moot for both deep-dived cases — the best available connector was already
inside the corridor at N=0 (no force-include needed). No data point across either probe shows a
connector that (a) beats the eventual choice and (b) is excluded from the *sonic* corridor mask,
which is what force-include patches. The SADE genre-mask exclusions (Bob James et al.) are real
exclusions but sit outside force-include's scope as currently designed (it operates on the eligible
universe, i.e. post-genre-mask, per `_build_corridor_segment_pool`'s `force_include` wiring) and,
worse, none of them actually beat the best already-admitted candidate (Plunky, min_sim 0.697) on the
sonic axis. **No evidence supports a specific N.**

---

## Recommendation for Task 2 (evidence-driven re-plan)

**Do not build Candidate A (force-include) or Candidate B (support-triggered pre-beam widening) as
scoped.** Both patch pool admission; in every case examined in depth, the pool already contained a
materially better connector than what was used. Building either would very likely fix nothing on the
shipped `dynamic` config and cost real engineering + review time on a lever the evidence says is
inert there.

**The evidence instead points at the post-beam repair floor gates:**

1. `tail_dp_floor = 0.3` (and `transition_floor = 0.2` for the widening ladder) are calibrated to
   "not broken," not "not much worse than achievable." PC segment 4's 0.394 and SADE's would-be
   0.454-window both clear these floors easily while a swap 30–40 points better sits in the same
   pool, unexamined.
2. Tail-DP is one-sided: it only re-opens the last 1–2 interior slots before `pier_b`. A weak edge at
   the *head* of a segment (SADE's case) is untouched by any existing repair mechanism.
3. `anchor_support` (already computed, already logged) is a decent low-support predictor in narrow
   corridors but is silent exactly where the shipped default lives (`open`-width segments have high
   support and still produce weak edges) — so gating repair on support alone (as Candidate B implies)
   would miss most of the shipped-default failures.

A more promising Task 2 candidate, grounded in this evidence: **trigger tail-DP's search
unconditionally whenever the realized window is more than `epsilon`-comparable to the corridor's own
best-available hmean/transition ceiling for that window**, rather than gating on a fixed
`tail_dp_floor`. Concretely, something like "run the tail-DP search whenever `old_min` is more than,
say, 0.15–0.2 below the top-ranked corridor candidate's own hmean for that position" would have
fired for both PC (0.394 vs Theresa's reachable ~0.7+ ceiling) and, if a symmetric head-side repair
existed, for SADE. This needs its own probe/design pass before being scoped as Task 2 — it is a
different mechanism than either original candidate, so Task 2 should not proceed with A/B as
written without a fresh design conversation.

---

## Files

- Findings doc (this file): `docs/corridor_baseline/phase2_mechanism_probes.md`
- Reproduction scripts (scratchpad, not committed):
  `probe1_pc_corridor.py`, `probe2_sade_corridor.py`
- Logs read: `logs/playlists/2026-07-18_174422_Parquet_Courts_b06d13.log` (satellite/corridor),
  `C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/logs/playlists/2026-07-18_181117_Parquet_Courts_ad8689.log`
  (canonical/legacy), `logs/corridor_baseline/1784391407392665000_corpus_SADE_home_757827.log` +
  the other 11 cells named in `docs/corridor_baseline/phase1_final_corpus2.json`.
- Code read (not modified): `src/playlist/pier_bridge/corridor.py`,
  `src/playlist/pier_bridge/eligible_universe.py`, `src/playlist/pier_bridge/tail_dp.py`,
  `src/playlist/pier_bridge_builder.py` (`_build_corridor_segment_pool`, tail-DP call site
  ~line 3090, widening ladder ~line 2128).
