# Tag-first pier selection (genre-steering seed choice) — design spec

**Date:** 2026-07-08. **Status:** approved design, ready to plan.
**Branch:** feat/tag-steering-sonic-prototype (continues the tag-steering work).
**Prior art (read first):** `docs/superpowers/specs/2026-07-08-genre-mode-design-notes.md` (finding **F2** is
the bug this fixes); memory `project_tag_steering`; `.superpowers/sdd/progress.md`.

## Problem

When a genre/tag is selected in artist mode, the piers (seeds) are supposed to be the artist's **most
on-tag** tracks. They are not. `cluster_artist_tracks` (src/playlist/artist_style.py) is **sonic-cluster
first**: it clusters the artist sonically, then picks the most-on-tag *medoid within each cluster*, with a
tag-skewed slot allocation on top (`allocate_piers_by_tag_affinity`). The tag is only a within-cluster
tiebreak + a slot skew, so sonic clusters that contain **no** on-tag track still emit a non-on-tag pier via
the ≥1-pier-per-cluster arc floor. Measured (Boards of Canada + hauntology): BoC's 18 authority-hauntology
tracks all sit at genre-dense affinity +0.638 (rank 1–18), yet the selected piers are **1/4 on-tag at
default knobs, 2/4 even at max knobs**. This is structural — no knob setting fixes it.

## Fix

Make pier selection **tag-first**: draw piers from the artist's **on-tag member set** (authority-defined),
then use the existing sonic machinery to spread/order them into an arc. This is a prerequisite for the
future genre mode (which reuses this pier-selection component).

## Design principles served

- **One Rule (genre authority):** "what counts as on-tag" is the published authority
  (`release_effective_genres` via `src/genre/authority.py`), not a sonic proxy. This whole fix exists
  because we once read the wrong genre layer.
- **#3 / #6:** the user's explicit tags are the gravity; the user should feel listened-to.
- **#9 / #12:** preserve multi-genre signatures; rare niche tags outweigh common labels.
- **#2:** the arc still matters — piers are sonically spread *within* the on-tag set.
- **#22:** the fix is the **live default** when steering is active (config rollback retained), never merged
  inactive.
- **#25:** graceful degradation — never crash; fall back when the artist has no on-tag tracks.

## Architecture

### A. On-tag member set `M` (authority + soft top-up)

New authority read (honors the One Rule):

- `src/genre/authority.py` → `on_tag_track_ids_for_artist(conn, artist_name, genre_ids: set[str]) ->
  dict[str, int]`: the seed artist's tracks whose album is authority-tagged with **any** id in `genre_ids`
  (`release_effective_genres` join `tracks.album_id`, `assignment_layer NOT LIKE 'inferred%'`, exact
  case-insensitive `tracks.artist` match). Value = **count of distinct selected genres** that track's album
  carries (for multi-tag ranking). Empty dict if none. Seed-*included* (distinct from the existing
  seed-*excluded* `_on_tag_track_ids` used for the bridge rescue).

- `src/playlist/tag_steering.py` → `resolve_artist_on_tag_membership(tags, artist_name, db_path,
  track_id_to_row) -> dict[int, int]`: maps chip names → canonical `genre_id`s (reusing the existing
  `genre_graph_canonical_genres` + aliases mapping already in `resolve_tag_sonic_prototype_rows`), calls the
  authority helper, and remaps track_id → bundle row. Returns `{bundle_row: tag_hit_count}`.

**Multi-tag semantics = union.** A track is on-tag if its album carries *any* selected genre. Union matches
how the steering signal already blends (`resolve_tag_steering_target` blends the tag embeddings;
`resolve_tag_sonic_prototype_rows` unions the rows) and avoids the near-always-empty intersection. Ranking
rewards multi-tag hits: order by **(tag_hit_count desc, then combined genre-dense+sonic affinity desc)**.

**Building `M`** (set of bundle indices), given `members = keys(membership)`, `target_pier_count`, and the
artist's total track count `A`:
1. `floor = min(A, max(style_cfg.cluster_k_min, ceil(topup_mult * target_pier_count)))`.
2. If `members` is **empty** → `M = None` (→ legacy fallback; we only go tag-first when the artist actually
   has authority on-tag tracks — never fabricate membership from a sonic proxy).
3. Else if `len(members) < floor` → **top up**: add the artist's tracks *not* in `members`, ranked by the
   combined affinity `combined[i] = (X_genre_dense[i] @ steering_target) + (sonic_tag_weight *
   sonic_tag_affinity[i] if sonic_tag_affinity is not None else 0.0)` (both terms already computed in the
   caller; the sonic term drops out when the cohesion gate disabled it), until `floor` is reached. This
   guarantees enough candidates for a spread arc without ever admitting an untagged track when the tagged set
   already suffices.
4. Else `M = members`.

`topup_mult` (default 2.0) aims for ~2× target piers of on-tag candidates so clustering has room to spread.
`floor` is capped at `A` and is `≥ cluster_k_min`, so clustering on `M` is always valid when the artist has
enough tracks for artist mode at all (the <`cluster_k_min`-total case already errors upstream).

### B. Three-mode dispatch (in `playlist_generator.py`, replacing the ~1952–2025 block)

Let `M_ids = {track_ids of M}`. `steering active` = `_steering_tag_list` non-empty and `steering_target`
present.

| `popular_seeds_mode` | Behavior |
|---|---|
| **fire** | **Unchanged.** `select_popular_piers` over *all* artist members (tag ignored on seeds). Bridges lean toward the tag via the existing pool lever + rescue. (Verify those fire under Fire — see Testing.) |
| **off** / **on**, `M is None` | **Legacy fallback** (today's behavior): full-artist clustering + `allocate_piers_by_tag_affinity` tag-skew (off) or medoid popularity weight (on). |
| **off**, `M` present | `cluster_artist_tracks(..., restrict_to_track_ids=M_ids)` → `allocate_piers_by_tag_affinity` over the on-tag clusters → `order_clusters`. Every pier ∈ M, sonically spread. |
| **on**, `M` present | `select_popular_piers(M_indices, popularity_values, target_pier_count)` → most popular *within* on-tag → `order_clusters`. If `popularity_values is None` (uncached), fall back to the **off**-on-`M` path. |

### C. Clustering restriction (`src/playlist/artist_style.py`)

`cluster_artist_tracks` gains one optional param `restrict_to_track_ids: Optional[set[str]]`. After deriving
`artist_indices` (and after the existing `excluded_track_ids` subtractive filter), intersect with it:
`artist_indices = [i for i in artist_indices if str(track_ids[i]) in restrict_to_track_ids]`. Everything
downstream — the **bridgeability veto**, k-means, medoid selection (`tag_slice` still combines genre-dense +
sonic), and `medoids_by_cluster` — then operates on the on-tag subset unchanged. The arc floor now spreads
piers *within on-tag clusters*, which is the fix. `None` = today's behavior, byte-identical.

## Config (config.yaml, under `playlists.ds_pipeline.pier_bridge`)

- `tag_first_pier_selection: true` — the live default when steering is active; `false` restores the legacy
  tag-skew allocation (rollback per #22).
- `tag_first_topup_mult: 2.0` — top-up floor multiplier on `target_pier_count`.

A configured knob that can't act must warn/raise, not silently no-op (project gotcha): if
`tag_first_pier_selection` is true but `steering_target`/`X_genre_dense` is absent at runtime, log at WARNING
and take the legacy path (this is the same missing-data path the existing steering guards already handle).

## Unchanged

Bridges are untouched: the allowed-set rescue, centered sonic pool lever, and opt-in combined beam term (F3/
F4/F5 machinery) stay as-is — they govern what fills *between* piers. Blended artists (Real Estate, every
album tagged jangle) → `M ≈ all tracks` → natural no-op → today's good RE result preserved.

## Error handling / edge cases

- Empty `M` (no authority on-tag tracks) → legacy fallback (B). No crash, best-effort lean.
- `popularity_values is None` under **on** → OFF-path on `M`.
- Bridgeability veto drops `M` below `target_pier_count` → fewer piers (existing warning path); top-up
  already over-provisions to reduce this.
- Multi-tag, sonically multimodal (e.g. ambient + noise) → existing prototype **cohesion gate** disables the
  sonic term → genre-dense-only steering (correct: don't force a bogus sonic centroid).
- Chip → canonical id unmapped for a tag → that tag contributes no members (existing WARN in the mapper);
  union means other tags still work.

## Testing

**Unit (`tests/unit/`):**
- `on_tag_track_ids_for_artist` on a synthetic authority DB: union across multiple genre_ids; hit-count
  value; non-inferred filter; case-insensitive artist match; empty result.
- `resolve_artist_on_tag_membership`: chip→id mapping + bundle remap; multi-tag union; unmapped tag warns.
- `M`-building: empty → None; small members → top-up to floor by combined affinity; large members → members
  unchanged; floor capped at artist track count.
- `cluster_artist_tracks(restrict_to_track_ids=...)`: piers ⊆ restriction; `None` byte-identical to today.

**Integration (`gui_fidelity` harness, multi-pier — mandatory per playlist-testing skill):**
- BoC + hauntology (OFF): piers are authority-hauntology BoC tracks (assert via `pier_check.py` logic ≥ 3/4
  on-tag, up from 1/4); worst-edge min-T within a notch of the no-steer baseline.
- BoC + hauntology + a second tag (multi-tag union): members = union; multi-tag-hit tracks rank first.
- Real Estate + jangle (OFF): result unchanged-or-better (M ≈ all RE).
- BoC + hauntology (ON): piers = most-popular hauntology BoC tracks (differ from OFF; all on-tag).
- BoC + hauntology (Fire): piers ignore the tag (pure popular); confirm the bridge pool lever still fires.
- Artist with zero on-tag tracks: legacy fallback path taken (log asserts).

**Acceptance:** BoC+hauntology piers on-tag (≥3/4); RE+jangle unchanged-or-better; worst-edge within a notch;
Fire unchanged on seeds with tag-leaned bridges; full fast suite green (the 3 known pre-existing failures
excepted, quoted).

## Out of scope

Genre mode (the pure/artist+genre 3rd mode) — this fix is only its pier-selection component. Bridge-side
surfacing of peripheral cliques (Ghost Box *between* piers) remains the segment-pool problem documented in
the genre-mode notes (F3/F5), not addressed here.
