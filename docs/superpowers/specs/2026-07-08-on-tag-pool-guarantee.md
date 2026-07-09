# On-tag pool guarantee (bridge-side surfacing) — design spec

**Date:** 2026-07-08. **Status:** approved approach, ready to plan.
**Branch:** feat/tag-steering-sonic-prototype (continues the tag-steering work).
**Prior art (read first):** `docs/superpowers/specs/2026-07-08-genre-mode-design-notes.md` (F3/F5 —
bridge-side surfacing); `docs/superpowers/specs/2026-07-08-tag-first-pier-selection.md` (the pier fix this
builds on). This is the **bridge-side complement** to tag-first pier selection.

## Problem (evidence-based)

With tag-first piers shipped, a BoC + hauntology playlist is correctly anchored on BoC's 18 hauntology
tracks — but still contains **no Ghost Box** (the roster the user expects: Belbury Poly, The Advisory
Circle, The Focus Group, Pye Corner Audio…). Diagnosis from log `2026-07-08_223912` + probe:

- Ghost Box are **on-tag** and **sonically fine bridges** to the hauntology piers: 107/120 pass the 0.36
  sonic floor; Belbury Poly → 0.77, The Advisory Circle → 0.74 sim.
- They are **NOT** lost to the sonic floor or the genre gate (the allowed-set rescue bypasses the latter).
- They ARE lost at **DS candidate-pool admission**: the pool builds via a **per-artist rank walk**
  (`candidate_pool.py` ~1244-1299) — artists are walked in sonic-rank order, taking ≤`candidates_per_artist`
  each until `max_pool_size`. Ghost Box artists rank low (BoC's hauntology tracks have hundreds of
  sonically *closer* neighbors), so the walk fills the pool (248) before reaching them. Library-wide, Ghost
  Box rank 526th at best (median 6,676 of 42,304); only **2/120 land in the pool window**.
- The existing **allowed-set rescue adds them to the universe but does NOT guarantee pool admission** — they
  still compete in the rank walk and lose. Segment pools are built from the DS pool (`segment_pool_max=400`
  > pool size), so a track absent from the DS pool is absent from every segment pool → the beam never sees
  it (confirmed earlier: beam weight 10 never surfaced them).

## Fix

**Guarantee a capped, per-artist-limited set of the authority on-tag tracks into the DS candidate pool**,
past the rank walk — so the beam can bridge through them. Key on **authority membership** (the genre-dense
discriminator: Ghost Box 0.94 vs Autechre 0.03), NOT sonic ranking (sonic can't tell them apart — F4). Only
guarantee tracks that are already **eligible** (passed the sonic/genre/BPM gates), so they are genuine
bridges, not jarring injections. Segment pools inherit automatically. This makes them *available*; the beam
+ the (opt-in) genre-dense beam term decide how many to use, and the unchanged diversity constraints
(min-gap, per-artist cap) prevent flooding the final playlist.

## Design principles served

- **#3/#6:** the user's explicit tags are the gravity; respect them all the way to the bridges.
- **One Rule:** guarantee is keyed on `release_effective_genres` (the authority), reusing the resolver
  from the pier fix.
- **#22:** live default when steering is active, config rollback retained.
- **#25:** only guarantee eligible (gate-passing) tracks; never inject sub-floor jarring bridges.
- **#11 diversity is a hard constraint:** per-artist cap on the guaranteed set + unchanged final-playlist
  diversity → no single act floods.

## Architecture

### A. Selecting the guarantee set (in `playlist_generator.py`)

`_on_tag_track_ids` (seed-excluded authority on-tag; already computed for the rescue, ~1807-1828) is the
candidate source. Pass it plus caps to `build_candidate_pool` at the artist-mode call site (~2323). No new
authority read — reuse the pier fix's resolver output.

### B. Force-admit past the rank walk (in `candidate_pool.py::build_candidate_pool`)

New params: `on_tag_guarantee_ids: Optional[set[str]] = None`, `on_tag_guarantee_max: int = 0`,
`on_tag_guarantee_per_artist: int = 0`. After the per-artist walk produces `pool_indices` (and before the
return / instrumentation), if `on_tag_guarantee_ids` and `on_tag_guarantee_max > 0`:
1. Candidates = indices in **`eligible`** (passed all gates) whose track_id ∈ `on_tag_guarantee_ids` and not
   already in `pool_indices`.
2. Rank them by `sonic_seed_sim` desc (smoothest bridges first — reuse the array already computed for
   admission).
3. Walk in that order, adding to `pool_indices`, enforcing `on_tag_guarantee_per_artist` per normalized
   artist and `on_tag_guarantee_max` total.
4. Log: `"Tag steering pool guarantee: force-admitted N on-tag track(s) across M artist(s) past the rank
   walk (cap=%d/artist=%d)"`. If 0 eligible on-tag candidates were available, log at INFO (not silent).

Guaranteed tracks are appended to the admitted pool; the instrumentation counts (`admitted_count`, etc.)
include them. `uncap_pool` (seeded mode) is unaffected — guarantee is independent of `max_pool_size`.

### C. Config (config.yaml, under `playlists.ds_pipeline.pier_bridge`)

- `tag_steering_pool_guarantee_max: 30` — total on-tag tracks force-admitted (0 = disabled/rollback).
- `tag_steering_pool_guarantee_per_artist: 3` — per-artist cap within the guaranteed set.

Live default when steering is active. `max=0` restores today's behavior exactly (rollback). If the knob is
set but no `on_tag_guarantee_ids` reach the pool call (no steering / no on-tag tracks), it is a no-op — the
caller only passes the ids when steering is active, consistent with the pier fix.

## Part 2 (validation-gated) — genre-dense beam term

The opt-in combined genre-dense+sonic beam term (`tag_steering_sonic_beam_weight`, default 0.0) was found
"inert" earlier — but ONLY because on-tag tracks weren't in the segment pool. With the guarantee, they are.
If Part A alone doesn't surface enough Ghost Box in validation (the beam still prefers closer neighbors),
calibrate `tag_steering_sonic_beam_weight` upward and set the live default. This is a **calibration step
gated on the Part-A result**, not a blind change — decide from the generation log + worst-edge.

## Error handling / edge cases

- No eligible on-tag candidates (all failed the gates) → guarantee adds nothing; INFO log. Falls back to the
  current pool (no Ghost Box, but honest).
- `on_tag_guarantee_ids` empty (no tag / no on-tag tracks) → param is None → no-op.
- Blended artist (RE + jangle): the on-tag tracks are already sonically central and mostly admitted by the
  rank walk, so the guarantee force-admits few/none → effectively a no-op → RE unchanged (verify).
- Guarantee must NOT re-admit seeds or title-excluded tracks (respect the same exclusions the walk does).

## Testing

**Unit (`tests/unit/`):** a pure helper `select_pool_guarantee(eligible_ids, sonic_seed_sim, artist_keys,
guarantee_ids, max_total, per_artist) -> list[int]` (extract the ranking+cap logic so it's testable without
a full pool): ranks by sim; per-artist cap; total cap; skips ids already in the pool / not eligible; empty
inputs → [].

**Integration (real `PlaylistGenerator`, per the pier fix's approach — `generate_like_gui` only reaches
seeds mode):**
- BoC + hauntology (off): the realized pool contains ≥ N Ghost Box/on-tag non-BoC tracks (assert the
  guarantee log fires + count on-tag artists in the pool). Then assert the **playlist** contains ≥1 non-BoC
  authority-hauntology bridge (the payoff). If 0 in the playlist with the guarantee active, that is the
  signal to engage Part 2 (documented, not a silent pass).
- Rollback (`tag_steering_pool_guarantee_max: 0`): reproduces today's no-Ghost-Box pool.
- Real Estate + jangle: pool + worst-edge unchanged-or-better (guarantee ≈ no-op).

**Manual (record real numbers):** regenerate BoC + hauntology through the worker path, read the log for the
guarantee line + the realized tracklist; count Ghost Box tracks; report worst-edge min-T vs the pre-guarantee
run. Confirm RE + jangle unchanged.

**Acceptance:** a BoC + hauntology playlist contains Ghost Box / non-BoC hauntology bridges (target ≥2–3);
worst-edge min-T within a notch of the pre-guarantee playlist; RE + jangle unchanged; rollback clean.

## Out of scope

Pure genre mode (compose from a genre with no seed artist). The segment-pool *scoring* strategy itself is
unchanged — we only change what's *admitted* to the pool it draws from.
