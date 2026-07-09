# Bridge-side Phase B — on-tag anchors (design spec)

**Date:** 2026-07-09. **Status:** approved design, ready to plan. **Branch:** feat/tag-steering-bridge-side.
**Prior art:** `docs/superpowers/specs/2026-07-08-tag-steering-architecture.md`; the Phase A spec/result
(bridges insufficient for peripheral cliques — BoC/hauntology = 0 Ghost Box even at beam weight 5);
research (Agent 2: `anchor_seed_ids`→piers is live, ungated, un-droppable; Agent 3: orienteering/waypoints).

## Problem

Phase A makes on-tag tracks *available* as bridges, but the worst-edge beam won't *place* sonically-
peripheral on-tag tracks as interiors (Ghost Box: smooth to one pier, not a stepping-stone in sequence).
The engine only guarantees a track appears if it is a **pier**. So: make representative on-tag tracks piers.

## What was tried and rejected (so the next session doesn't re-derive it)

An **adaptive trigger** (inject anchors only when the seed is "peripheral" to the tag) was the first choice,
but the peripherality metric — `cos(seed on-tag centroid, tag library sonic prototype)` — **failed
validation across 8 cases**: Charli/glitch is *clean* yet scored *lowest* (0.20; multi-modal tag), and
BoC/hauntology (the hardest case) scored 0.69 — as central as clean Real Estate (0.69). Centroid distance
is misled by multi-modal tags, exactly like aggregate cohesion. **No usable threshold exists**, so there is
no reliable binary trigger. (§6's "seed centrality" variable does not survive direct measurement.)

## Design — always inject, with *bridgeability-aware selection* doing the adaptive work

When tag steering is active, always select and inject **K on-tag anchor tracks as piers**, choosing them so
they can actually bridge in. The selection self-adapts without a trigger:
- **Clean seed (Eno/neoclassical):** the bridgeable on-tag anchors are close neighbors (a Frahm track) —
  tracks that would appear anyway; ~neutral topology change.
- **Peripheral seed (BoC/hauntology):** the selection picks the on-tag tracks *closest to the seed's piers*
  (Ghost Box at ~0.77 sonic) — guaranteeing the clique appears AND that the anchor is not an unbridgeable
  island.

### A. Anchor selection (new function)
Source = the authority on-tag library set (`_on_tag_track_ids`, seed-excluded, already computed in the
artist-mode pier block). Select up to K:
1. **Bridgeable filter:** keep only tracks whose max sonic cosine to any of the seed's chosen piers ≥
   `anchor_min_bridge` (default 0.35) — the anchor must be reachable from at least one existing pier, so it
   is not a jarring island.
2. **Rank by tag-centrality:** among the bridgeable set, rank by sonic cosine to the tag's library
   prototype (the centered sonic prototype already computed for the pool lever) — canonical-sounding
   exemplars of the genre first.
3. **Cross-artist diversity:** per-artist cap `anchor_per_artist` (default 1), so anchors span the genre's
   roster (Belbury Poly, The Advisory Circle, The Focus Group — not three Belbury tracks).
4. Take the top `anchor_max` (default 3).
- **Graceful empty:** if no on-tag track clears the bridge floor, inject none → Phase-A-only (no crash, log INFO).

### B. Injection (reuse the live pier machinery)
Append the selected anchor track_ids to the seed's pier set (`ordered_medoids` / `pier_ids` in
`playlist_generator.py`'s artist-mode block, after tag-first pier selection ~2072). They become piers via
the same ungated resolution multi-artist playlists already use (Agent 2). The existing pier ordering
(`_order_seeds_by_bridgeability`) then sequences ALL piers (seed medoids + anchors) for smoothness — no new
ordering scheme needed; anchors land where they bridge best.
- **Total-pier cap:** cap total piers at `target_pier_count + anchor_max` so interiors aren't starved.
- Anchors are non-seed → they do not violate the seed-artist diversity rules; standard min-gap/per-artist
  caps still apply to the whole playlist.

### C. Interaction with Phase A
Complementary: Phase A improves the *interiors* between piers (relaxed admission + guarantee); Phase B
guarantees the on-tag *anchors*. Both gated on steering. Together: on-tag anchors + on-tag-friendly bridges.

## Config (`playlists.ds_pipeline.pier_bridge`)

- `tag_steering_anchor_max: 3` — max on-tag anchors injected (0 = off / Phase-A-only rollback).
- `tag_steering_anchor_min_bridge: 0.35` — min sonic cosine to a seed pier for an anchor to qualify.
- `tag_steering_anchor_per_artist: 1` — per-artist cap within the anchor set.

All gated on tag steering active; `anchor_max: 0` = clean rollback (Phase-A-only). Missing-data path logs, never silent.

## Testing

- **Unit** (pure selection function): bridgeable filter (island excluded); tag-centrality rank; per-artist
  cap; K cap; empty→[]; seed-excluded.
- **Integration (real `PlaylistGenerator`):**
  - BoC + hauntology: playlist contains ≥2 Ghost Box / non-BoC hauntology tracks **as piers** (the payoff —
    this is the case Phase A couldn't crack). Worst-edge min-T within ~one notch of Phase-A baseline.
  - Bowie + krautrock: ≥2 canonical-krautrock anchors present; less off-genre drift than Phase A alone.
  - Eno + neoclassical: no-regression — still on-genre, worst-edge not worse by more than ~one notch,
    anchors are close neighbors (not jarring). Confirms "always inject" is safe for the clean case.
  - `anchor_max: 0`: byte-identical to Phase-A-only (rollback guard).
- **Manual predict-then-check:** regenerate BoC/hauntology + Bowie/krautrock + Eno; count on-genre anchors +
  worst-edge; quote vs Phase-A runs.

## Acceptance

BoC + hauntology contains ≥2 Ghost Box tracks (the long-standing goal); Bowie/krautrock more on-genre; Eno
no-regression (anchors are close neighbors, worst-edge intact); rollback clean. Worst-edge min-T within ~one
notch of Phase A across all. This is the case Phase A could not solve — anchors are the fix.

## Open risk / mitigation

"Always inject" changes the clean case's topology (a couple non-seed anchors where none were needed). The
bridgeability filter keeps them close (would-appear-anyway neighbors), K is small (3), and `anchor_max` is
tunable — if the clean cases feel over-anchored in validation, lower the default. Not gated on a
peripherality metric because none reliably exists (see above).

## Out of scope

An explicit user-facing "compose toward genre" mode (option C) — deferred; this ships inside ordinary tag
steering. A learned peripherality signal (the centroid metric failed; a better one would need its own R&D).
