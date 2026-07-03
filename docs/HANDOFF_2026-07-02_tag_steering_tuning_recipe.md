# HANDOFF — tag-steering tuning recipe (pending insertion into `docs/PLAYLIST_ORDERING_TUNING.md`)

**Status:** deferred. A concurrent docs-rewrite project owns `docs/PLAYLIST_ORDERING_TUNING.md`
and will rewrite it wholesale — this recipe was written but held out of that file to avoid
colliding with the rewrite. When the rewrite lands, fold the section below into
`docs/PLAYLIST_ORDERING_TUNING.md` using its house knob-recipe format, then delete this file.

---

## Knob: tag steering (artist-mode genre chips)

- `playlists.ds_pipeline.pier_bridge.tag_steering_pool_blend` (default **0.5**) — how far the
  candidate-pool genre-admission centroid moves from the seed centroid toward the selected
  tags. 0 = chips do nothing to the pool; 1 = pool admission ranks purely by tag affinity.
  Symptom → move: "playlist ignores my tags" → raise toward 0.7; "playlist lost the artist's
  own character" → lower toward 0.3.
- `playlists.ds_pipeline.pier_bridge.tag_steering_pier_weight` (default **0.3**) — bonus for
  on-tag artist tracks in pier (medoid) selection, composing with the sonic/duration/energy/
  popularity terms. Raise if the anchors don't reflect the tags; 0 disables pier steering only.
- Per-request tags arrive as `tag_steering_tags` via the GUI; empty = feature fully inert.
- Diagnose with the per-playlist log: `Tag steering target` (mapped count), `Tag steering
  pool lever` (blend applied), `Tag steering pool affinity` (p10/p50/p90 of the admitted
  set), `Tag steering piers` (per-pier affinity). No lines = the knob didn't act — that's a
  bug, not a tuning problem.
