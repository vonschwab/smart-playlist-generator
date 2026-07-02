# Artist-pier scarcity re-admission + even mini-pier spacing — design

**Date:** 2026-07-01
**Status:** approved (design), pending implementation plan
**Scope:** artist mode only (`mode=artist` / `artist_playlist=True`). No change to seeds/diverse modes.

## Problem

A Subsonic Eye artist playlist (2026-07-01) exposed two defects in how seed-artist
piers are chosen and placed. Subsonic Eye has 10 solo tracks; Artist Presence: High
targets 10 piers.

1. **Freshness guts the seed-artist pier set → "popular seeds aren't popular."** The
   seed **freshness filter** (`playlist_generator.py`, `seed_recency_excluded_ids`)
   removes recently-played tracks *before* pier selection (passed into clustering as
   `excluded_track_ids`, and applied to the selected/automatic seed paths). For your
   most-played artist the recently-played tracks **are** the popular ones, so the top
   hits get filtered out and only mid-tier survivors remain (log: `solo=10 … total=6`;
   piers ranked Last.fm #15/#19/#20/#39/#49 + one non-charting). This also contradicts
   the documented principle that *seed/pier tracks are explicitly requested and exempt
   from recency* (CLAUDE.md).

2. **Mini-pier placement bunches the remaining seeds.** With only 6 fresh seeds and
   segments capped at `mini_pier_max_interior=5`, `plan_pier_sequence`
   (`mini_pier_select.py`) padded 4 mini-piers — but inserted **all 4 into the first
   seed-gap**, shoving the other 5 seeds into the last 20 tracks (positions
   1, 31, 36, 41, 46, 51). Root cause: it picks the split target with
   `argmax(_even_split_lengths(interior, num_seg))`, and `_even_split_lengths` puts the
   remainder on the earliest segments, so `argmax` **always returns segment 0**. Every
   waypoint chains into gap 0.

**Relationship:** freshness is the trigger. It shrank a 10-track catalog to 6 usable
seeds, which forced mini-pier padding, which hit the placement bug. Fixing (1) removes
the padding need for track-rich artists; (2) still matters for genuinely small catalogs.

## Decisions

- **Precedence:** for the seed artist, Artist Presence wins over freshness — but only
  under scarcity, and via minimal graceful re-admission (not a blanket exemption).
- **Diversity scope:** `min_gap` governs non-seed artists only. The seed artist is
  already exempt by construction (it appears only at piers;
  `disallow_seed_artist_in_interiors=True`; `min_gap` is enforced on interiors via
  `_enforce_min_gap_global`). **No diversity/min_gap code change.**
- **Scarcity target:** the seed-artist pier target is `target_piers` (from Artist
  Presence / `max_artist_fraction`). Re-admission fills the *eligible seed pool* toward
  this target.
- **Freshness stays hard on the interior/bridge candidate pool** — that is where
  "don't replay what I just heard" matters.

## Change 1 — scarcity-gated freshness re-admission

**Where:** `src/playlist_generator.py`, the artist seed-selection path (freshness
filter ~1540–1593; artist-style clustering call ~1778–1786, which passes
`excluded_track_ids=seed_recency_excluded_ids`).

**Mechanism:**
1. Partition the seed artist's valid-duration tracks into `fresh` (not in
   `seed_recency_excluded_ids`) and `stale` (recently played).
2. If `len(fresh) >= target_piers`: behave exactly as today (freshness fully honored).
3. Else re-admit `stale` tracks until the eligible pool reaches `target_piers`
   (or the catalog is exhausted). Ordering when only a subset can return:
   **by popularity when Popular Seeds is on** (restores the hits), else **stalest-first**
   (preserve maximum freshness).
4. Re-admission only *widens the eligible pool*; the existing medoid / popularity
   selector still picks the piers from it. Clustering receives the reduced exclusion set.

**Rollback:** reuse the existing `exclude_seed_tracks_from_recency` lever, now
scarcity-gated. New behavior is the live default (Layer 4: activate the fix, keep a
rollback). Setting it to force-exclude reproduces today's behavior.

**Testability:** extract a pure helper, e.g.
`resolve_seed_eligibility(fresh_ids, stale_ids_ranked, target) -> eligible_ids`, so the
scarcity logic is unit-testable without artist-style clustering + Last.fm (which are
worker-layer and not covered by `generate_like_gui`).

## Change 2 — even mini-pier distribution

**Where:** `src/playlist/pier_bridge/mini_pier_select.py::plan_pier_sequence`.

**Insight:** in the even-split model every sub-segment ends up the same length
(`interior // num_seg`) regardless of *which* gap is split — so the split-choice
controls only seed spacing. "Split the longest segment" is degenerate here (all equal),
which is why it collapsed to always-segment-0.

**Mechanism:** distribute waypoints across the seed-gaps instead of `argmax` of a
near-uniform vector. Assign each new waypoint to the seed-gap with the **fewest
waypoints so far, ties broken leftmost** — a clean round-robin that yields even seed
spacing (for equal-length gaps this is identical to "split the gap with the largest
actual current interior"). Keep the existing terminators (every segment interior ≤
`max_interior`, no feasible waypoint from `select_waypoint`, or `max_waypoints`). Each waypoint remains a real
library track chosen by `select_waypoint` as the sonic midpoint of the gap it splits.

**Worked example (100 tracks, 10 seeds, `max_interior=5`):** 9 gaps × 10 interior each
need splitting; 8 mini-piers bring it to 18 piers / 17 segments (~4.8 interior each).
- Today: all 8 in gap 0 → one seed, a ~50-track seedless wall, then 9 seeds bunched.
- Fixed: 8 mini-piers, one per gap across 8 of 9 gaps → `S0 m S1 m … S7 m S8 · S9`,
  seeds ~every 11 tracks, segments ~5.

**Non-goal (YAGNI):** sonic-distance-weighted gap splitting (pour more waypoints into
sonically rougher gaps). The current model measures gaps by track count, where even
distribution is correct. Flag as a possible future refinement only.

## Testing

- **Change 1 (unit):** on the pure helper — no re-admission when `len(fresh) >= target`;
  re-admit up to `target` under scarcity; never exceed the catalog; correct ordering
  (popularity vs stalest) per mode.
- **Change 2 (unit):** `plan_pier_sequence` with 6 seeds/50 and 10 seeds/100 → waypoints
  land in ≥3 distinct gaps (not all gap 0); resulting seed positions are ~evenly spaced;
  terminators still hold (all segments ≤ `max_interior`).
- Reference the fixing commit in each test; add a Trap-Catalog note if a new fidelity
  gap surfaces (per the playlist-testing skill).

## Files

- `src/playlist_generator.py` — scarcity-gated re-admission + extract pure helper.
- `src/playlist/pier_bridge/mini_pier_select.py` — even split-target selection.
- `tests/unit/` — new tests for both.
- (config) `exclude_seed_tracks_from_recency` semantics documented; no new knob.
