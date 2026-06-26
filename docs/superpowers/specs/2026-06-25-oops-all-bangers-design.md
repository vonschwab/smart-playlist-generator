# "Oops, All Bangers" — Popularity-Steered Bridges — Design

**Status:** design (brainstormed 2026-06-25). Successor to the artist energy-spread +
Popular Seeds program (`2026-06-23-artist-energy-spread-popular-seeds-design.md`).

**Goal:** A three-stop popularity control that biases a playlist's **bridge** tracks toward
each artist's Last.fm hits — `OFF → ON → OOPS, ALL BANGERS` — so the same engine can hand
you the weird corners of your library (default) or a greatest-hits sweep (on command).

**Architecture (one sentence):** A single soft popularity penalty applied per-candidate in
the pier-bridge beam's bridge selection, fed by a bundle-aligned popularity vector that the
orchestrator builds from the existing Last.fm cache + resolver (warmed by the eager sidecar,
topped-up per-playlist by a cache-first+TTL pool-scan).

**Tech stack:** existing — `src/analyze/popularity_runner.py` (cache, resolver, sidecar),
`src/lastfm_client.py`, `src/playlist/popularity_loader.py`, `src/playlist/pier_bridge_builder.py`
(beam), the request/schema/worker chain, and the React GenerateControls.

---

## Global Constraints

- **Default OFF = byte-identical to today.** This is a *creative preference*, not a quality
  fix — the weird-library default is the point. OFF does no scanning, no penalty, no change.
- **Soft, never a hard gate.** Popularity demotes candidates in the beam's score; it never
  excludes them and never narrows the candidate pool. Hard gates on a sparse signal detonate
  the relaxation cascade and blow the budget.
- **A playlist must never fail or exceed the ~90s budget because of this.** Popularity is one
  of the relaxable soft preferences, never the hard constraint.
- **Local-first / never gate on the network.** All popularity reads are cache-first; a failed
  fetch falls back to stale or neutral, never an error.
- **`metadata.db` read-only; irreplaceable artifacts untouched.** This feature only reads the
  cache + writes the Last.fm cache table (`ai_genre_enrichment.db`) on a miss.

---

## Background — what already exists (do not rebuild)

The artist energy/Popular-Seeds program already shipped the entire data substrate:

- **Resolver** — `resolve_top_tracks_to_rank` / `resolve_top_tracks_to_popularity`
  (mbid-first, then loose-title + version-preference; remaster NOT penalized; per-artist
  `score = 1 − rank/N`).
- **Per-artist cache** — `artist_top_tracks_cache` in `ai_genre_enrichment.db`, with
  `get_artist_top_tracks_cached_or_fetch(... max_age_days=TTL ...)` — cache-first, TTL refresh,
  graceful on failure, never raises.
- **Eager library sidecar** — built 2026-06-25 via `analyze_library.py --stages popularity`:
  1,128 qualifying artists cached, `popularity_sidecar.npz` written, **51.9% of tracks carry a
  per-artist popularity number** (~4k score ≥0.90, ~7.2k ≥0.80).
- **Loaders** — `load_artist_popularity_values` (one artist → bundle-aligned vector),
  `popularity_loader.load_popularity_vector` (read the sidecar).
- **Per-pier rank logging** — `log_seed_popularity` (diagnostic).

All-Bangers is **consumption only** — it adds a control + a beam penalty + a pool-level
popularity loader. No new data pipeline.

---

## Design decisions (locked during brainstorming)

1. **Signal = per-artist Last.fm rank** (`1 − rank/N`); a track not charting / unknown = NaN.
   "Deep cut" is inherently per-artist, and this preserves niche taste (an indie act's hit is
   not buried under a pop star's B-side). Rejected: global playcount (skews mainstream).
2. **Coverage = ruthless.** NaN is treated as a deep cut and demoted at the active strength.
   No separate "is this junk vs a legit deep cut" detection — the dial filters *everything*
   below the bar, which is the point (the weird/junk stuff is a *feature* at OFF/ON, removed
   only on command). Rejected: a separate version-preference junk-gate on bridges.
3. **Soft, beam-only penalty.** Popularity never reshapes the candidate pool — only re-ranks
   within it. This is what makes (5) true.
4. **Three-stop control, default OFF.** `OFF → ON → OOPS, ALL BANGERS` = one penalty strength
   with three values. Start discrete; a continuous slider can come later.
5. **No convergence loop — one-shot pool-scan.** The beam's `universe` is built once
   (`pier_bridge_builder.py:514`) and *all* expansion draws from it (`:1166–1383`); since
   popularity is soft/beam-only it never reshapes that universe. The universe is therefore the
   fixed, fully-knowable closure of every selectable track. So: scan the pool's distinct
   artists once (cache-first + TTL), build the vector, generate once. Regeneration could only
   re-pick artists already in that scanned set.
6. **Scope = bridges, every mode.** Bridges exist in artist / seeds / genre modes, so the
   control is mode-agnostic. In artist mode it stacks with Popular Seeds (which steers the
   *piers*); the two are orthogonal (piers vs bridges).
7. **Coverage built incrementally.** Eager sidecar seeds the cache; the per-playlist pool-scan
   (cache-first + TTL) fills the sub-8-track long tail the batch skipped *and* refreshes stale
   artists — freshness is a free side effect of the TTL, not a separate mechanism.

---

## The control

A new orthogonal axis, parallel to the existing four mode axes but with its own three-value
vocabulary:

- **Config:** `playlists.popularity_mode: off | on | oops` (default `off`).
- **Strength map** (pier-bridge config, keyed by the mode): `popularity_penalty_strength_off:
  0.0`, `popularity_penalty_strength_on: <s_on>`, `popularity_penalty_strength_oops: <s_oops>`.
  Placeholders `s_on ≈ 0.10`, `s_oops ≈ 0.30` — comparable to the existing
  `genre_penalty_strength: 0.15`; **calibration targets, not final** (tuned against the
  edge-score scale where adjacent-candidate transition gaps run ~0.05–0.15).
- **Request/schema/worker:** `popularity_mode: str = "off"` threaded the same way
  `popular_seeds` / `seed_epoch` were (request_models → schemas → worker → the generation call).
- **GUI:** a three-way segmented control (Off / On / Oops, All Bangers) in GenerateControls,
  shown in **all** modes (unlike the artist-only Popular Seeds checkbox), `useLocalStorage`,
  default Off.

---

## Runtime data flow (when `popularity_mode ∈ {on, oops}`)

1. **Orchestrator builds the candidate pool** as today (sonic/genre/pace — unchanged).
2. **Pool-scan:** collect the distinct `artist_key`s of the **admitted candidate pool** (the
   beam's `universe` — the bounded closure of selectable tracks, on the order of a few hundred
   artists; **not** the pre-gate allowed superset). For each, call
   `get_artist_top_tracks_cached_or_fetch(... max_age_days=TTL ...)` (cache hit for the ~1,128
   batched artists, fetch+cache for the long tail; TTL refreshes stale). Never raises.
3. **Build the popularity vector:** resolve the pool's tracks to per-artist scores via
   `resolve_top_tracks_to_rank`/`_to_popularity`, producing a **bundle-aligned** vector
   `popularity_values` (`np.full(len(bundle.track_ids), nan)` with scores filled in) — the same
   shape contract `load_artist_popularity_values` already returns, generalized to many artists
   (`load_pool_popularity_values`).
4. **Pass into the beam:** the pier-bridge builder receives `popularity_values` and the resolved
   `popularity_penalty_strength`.
5. **Per-candidate penalty in bridge selection** (see below).

When `popularity_mode == off`: steps 2–5 are skipped entirely. `popularity_values = None`,
strength `0.0`, and the beam path is byte-identical to today.

**Warm cache, optional fast path:** the eager `popularity_sidecar.npz` may be read first as an
instant source for the batched 52% (`load_popularity_vector`), with the pool-scan only
fetching pool artists missing/stale in the cache. The cache + resolver remain authoritative for
freshness; the sidecar is an optimization, not a dependency. (Plan may start cache-only and add
the sidecar fast-path if pool-scan latency warrants it.)

---

## Penalty mechanics

For each candidate bridge track `t` with popularity score `p(t) ∈ [0,1]` (1 = the artist's #1,
→0 = bottom of its top-N), or NaN if not charting/unknown:

```
demotion d(t)  = 1 − (p(t) if finite else 0.0)     # NaN → d = 1.0  (ruthless)
penalty(t)     = strength × d(t)                    # strength ∈ {0.0, s_on, s_oops}
```

`penalty(t)` is subtracted from the candidate's per-candidate beam objective, alongside the
existing `genre_penalty` / `duration_penalty` family (same units, same soft-penalty pattern).

- Banger (`p = 1.0`) → `d = 0` → no demotion.
- Mid cut (`p = 0.4`) → `d = 0.6` → moderate demotion.
- Deep cut / unknown (`p = 0.02` or NaN) → `d ≈ 1.0` → maximum demotion.

Strength scales the whole curve: `OFF` zeroes it; `ON` nudges (oddities demoted but still
surface when sonic/genre fit is strong); `OOPS` dominates (only bangers survive → greatest
hits). It is always additive and bounded — it lowers a score, never removes a candidate.

---

## Coverage & freshness

- **Eager sidecar** (done) warms the cache so the pool-scan is mostly cache hits → fast.
- **Pool-scan + TTL** fills the sub-8-track long tail per-playlist and auto-refreshes any pool
  artist past `max_age_days`. No separate "keep current" loop.
- **Honest ceiling:** ~52% of the library has a score today; the rest is NaN (genuine deep cuts
  of charted artists + un-fetched small artists). At `ON` this means a gentle lean; at `OOPS`
  the NaN tail is strongly demoted — exactly the intended "greatest hits."

---

## Interaction with existing features

- **Popular Seeds (artist mode):** orthogonal. Popular Seeds → popular *piers* (medoid term);
  All-Bangers → popular *bridges* (beam penalty). Both may be on simultaneously.
- **The four mode axes (cohesion/genre/sonic/pace):** orthogonal. Popularity is a new soft
  preference that rides on top; it does not touch pool gating.
- **Energy spread:** orthogonal (energy is a medoid/pier concern; this is bridges).
- **Version dedup:** still applies to piers; bridges are unchanged except for the new penalty.

---

## Error handling & invariants

- **OFF byte-identical:** no scan, `popularity_values=None`, strength 0 → the beam path is
  unchanged. Guarded by both the mode check and the None-vector check (mirrors Popular Seeds).
- **Never gate generation:** pool-scan is cache-first and wrapped so a network failure yields
  stale/neutral popularity, never an exception. A NaN-heavy vector just means weaker steering.
- **Budget:** the pool-scan adds at most a bounded number of cache-first lookups (mostly hits);
  network fetches for the long tail are TTL-amortized and capped by the pool's distinct-artist
  count. If a generation-deadline is in force, pool-scan respects it (skip remaining fetches →
  neutral for those artists).
- **No hard gate, no pool change:** the candidate pool / universe is identical with the dial at
  any setting; only per-candidate scores differ.

---

## Testing strategy

- **Opt-in invariant:** `popularity_mode="off"` ⇒ no scan, `popularity_values=None`, penalty
  inert, output byte-identical to baseline (pin with a generation test).
- **Penalty unit tests:** `d(t)` formula — banger→0, mid→partial, deep-cut→~1, **NaN→1.0**;
  strength scaling OFF/ON/OOPS; additive into the candidate score; never removes a candidate.
- **Pool loader:** `load_pool_popularity_values` returns a bundle-aligned vector; cache-first;
  never raises on fetch failure; multi-artist resolution correct (per-artist rank).
- **Direction test (via the slider-differentiation harness, multi-pier through the policy
  layer):** ON demotes deep cuts vs OFF; OOPS demotes them more; distinct-artist + worst-edge
  not wrecked; ≤90s.
- **Never-fail:** a pool with mostly-NaN popularity still generates a full playlist at OOPS.

---

## Calibration & open questions (resolve during/after implementation)

- **Calibration is mandatory after implementation, against a deliberately diverse artist
  panel** — not one or two seeds. The panel must span the axes that stress this signal
  differently:
  - **Legacy vs. active** — legacy catalogs have stable, well-separated top-N; *active*
    artists' recent releases under-rank (Last.fm playcount accrues with time), so a genuine new
    banger can score low. Verify OOPS doesn't wrongly bury current hits.
  - **Niche vs. popular** — popular artists have a deep, cleanly-ranked top-50; *niche* artists
    have sparse, noisy ranks (near-tied low playcounts) and far more NaN (often sub-8-track, so
    un-batched). Verify ON/OOPS don't gut niche playlists by over-demoting the NaN tail.
  - **High vs. low sidecar coverage** — how much of the seed's pool already carries a score.
  Reuse the multi-pier `slider_differentiation_eval.py` harness (route modes through the policy
  layer). Per-panel-artist, confirm: ON demotes deep cuts vs OFF, OOPS more so, while
  distinct-artist count + worst-edge + ≤90s hold. A finding that one global strength can't serve
  all artist types (e.g. niche needs a gentler NaN penalty) is an expected calibration output.
- Exact `s_on` / `s_oops` strengths (start ~0.10 / ~0.30; tune on real playlists).
- `max_age_days` TTL for the pool-scan (start 30; popularity drifts slowly — 60–90 may fetch
  less without much staleness cost).
- Whether to read `popularity_sidecar.npz` as a fast-path or rely on the cache+resolver only
  (start cache-only; add the sidecar path if pool-scan latency is felt).
- Exact pool-scan placement (orchestrator vs. just-before-beam) — wherever the pool track_ids
  and the Last.fm client are both in scope and the bundle alignment is available.

## Out of scope

- A continuous slider (start with three stops).
- A separate version-preference junk-gate on bridges (explicitly rejected — the popularity dial
  is the only filter, by design).
- Any change to candidate-pool / universe construction (popularity stays beam-only).
- Re-running or changing the eager batch / sidecar build (already shipped).
