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

## Knob: tag steering — sonic prototype (2026-07-08)

The genre-dense chips above steer in *genre* space, which is album-level and therefore flat
within a genre-blended artist (Real Estate's tracks are all tagged the same). The sonic
prototype adds per-track resolution: the selected tag(s) learn a **centered** MuQ centroid
from the library (the tag-*specific* sonic direction), and that per-track affinity steers
pier selection and the candidate pool. Validated on Brian Eno (bimodal) + Real Estate
(genre-blended). Spec: `docs/superpowers/specs/2026-07-07-tag-steering-sonic-prototype-design.md`.

- `...pier_bridge.tag_steering_sonic_weight` (default **0.5**) — weight of the sonic-prototype
  term in pier (medoid) scoring. **Note:** the effective influence on within-cluster medoid
  choice is `sonic_weight × medoid_tag_weight` (0.5 × 0.3 = 0.15); cross-cluster pier *slot*
  allocation uses it at full weight. Raise if the anchors don't reflect the tag.
- `...pier_bridge.tag_steering_sonic_blend` (default **0.35**) — **the binding lever.** How far
  the pool's sonic admission similarity is pulled toward the centered tag affinity. This is what
  actually moves the playlist's on-tag lean (the beam only reorders within the pool it's handed,
  so the pool sets the ceiling). Symptom → move: "not leaning toward my tag" → raise toward 0.5;
  "lost the artist's character / rougher transitions" → lower toward 0.25. Uses the **centered**
  (tag-specific) direction — the earlier uncentered prototype pulled toward the generic-genre
  centroid and was retired 2026-07-08 (it made Real Estate/jangle *worse*: worst-edge 0.463→0.315;
  centered fixed it: 0.463→**0.716**, lean −0.048→**+0.020**).
- `...pier_bridge.tag_steering_sonic_beam_weight` (default **0.0 = OFF**) — opt-in per-candidate
  on-tag bonus in the beam's ranking score. **Left off by default:** a weight sweep (0.0/0.15/0.5/1.0)
  showed it cannot raise the on-tag lean at any weight (the beam is bounded by the pool's ceiling)
  and only jitters the worst edge. Wired/tested/documented for opt-in experimentation only.
- `...pier_bridge.tag_steering_prototype_min_support` (default **25**) — min library tracks
  carrying the tag(s) to trust the prototype; below this it falls back to genre-dense + WARNs.
- `...pier_bridge.tag_steering_prototype_min_cohesion` (default **0.15**) — min intra-set cohesion;
  below this the tag is sonically multimodal and the sonic levers disable + WARN.
- Expect distinct genres (ambient) to lean strongly *and* improve worst-edge; genre-blended
  artists whose tag is sonically adjacent to their neighborhood (RE jangle ≈ dream-pop) lean
  modestly but honestly, with the worst edge improved. Diagnose with `Tag steering sonic
  prototype: support=.. cohesion=..` and `Tag steering CENTERED pool lever: blend=..` log lines.
