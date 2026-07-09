# Genre mode — design notes & findings (for a future build)

**Date:** 2026-07-08. **Status:** findings + design sketch, NOT yet built. Captured from the
tag-steering investigation (branch `feat/tag-steering-sonic-prototype`) so genre mode is easy to build
later. Motivating case throughout: **Boards of Canada seed + "hauntology" tag → the user expects a
hauntology-flavored playlist anchored on BoC's hauntological tracks and pulling in the Ghost Box roster
(Belbury Poly, The Focus Group, The Advisory Circle). Today it does not.**

Genre mode = a third generation mode alongside **artist mode** and **seeds mode**: compose *from a
genre* (using genre centroids / the tag's library-wide tracks) rather than from a seed artist's sonic
neighborhood. Two flavors: **pure** ("play me hauntology") and **artist+genre** ("Boards of Canada's
hauntology").

---

## Why artist-mode + tag steering can't do this (5 findings, each with data)

### F1 — Genre reads MUST use the published authority (fixed, but the root of the confusion)
The genre authority is **`release_effective_genres`** (album-keyed, via `src/genre/authority.py`);
`track_effective_genres` is a raw/partial input (MusicBrainz-polluted + missing file-sourced tags).
Tag steering originally read `track_effective_genres` → user-tagged Ghost Box `hauntology` was invisible
(all 99 Ghost Box tracks absent there, while the authority had them as `observed_leaf` 0.95). Fixed
2026-07-08 (`resolve_tag_sonic_prototype_rows` reads the authority via `album_id`; chip name → canonical
`genre_id` via `genre_graph_canonical_genres` + `genre_graph_aliases`; non-inferred layers only).
Data: authority hauntology = 174 tracks (Advisory Circle 35, Focus Group 32, Belbury 29, Plone 24, BoC
18…) vs `track_effective_genres` = 162 (BoC 73 via an MB *artist* tag, no Ghost Box). See
`docs/CLEANUP_LIST.md` (stale-data cleanup) + memory `project_enriched_genre_authority`.

### F2 — Pier selection is cluster-THEN-tag (sonic-first) — the core reason piers aren't on-tag
`cluster_artist_tracks` (artist_style.py) clusters the artist's tracks **sonically** first, then picks
the most-on-tag medoid *within each cluster*; `allocate_piers_by_tag_affinity` skews slot counts by
per-cluster tag affinity. The tag is only a within-cluster tiebreak + a slot skew. When a tag is
concentrated in a sonic sub-region (hauntology ⊂ BoC's Tomorrow's Harvest sound), the sonic clusters
that contain no hauntology track still emit a (non-hauntology) pier via the ≥1-pier-per-cluster arc
floor. **Data (BoC + hauntology):** BoC's 18 authority-hauntology tracks all sit at genre-dense
hauntology affinity +0.638 (rank 1–18; the rest at +0.07–0.24). Selected piers at DEFAULT knobs:
Telephasic Workshop (rank 23, not hauntology), Pete Standing Alone (23, no), Memory Death (13, **yes**),
Skimming Stones (**57**, no) → **1/4 on-tag**. At MAX knobs (`tag_steering_pier_weight=3.0`,
`pier_tag_skew=1.0`): All Reason Departs (rank 1, yes), Memory Death (13, yes), XYZ (19, no), Triangles
& Rhombuses (23, no) → **2/4** — the knobs help but the sonic-cluster-first structure caps it. **Genre
mode must select piers tag-FIRST**, then order for arc.

### F3 — Four stacked seed-proximity gates squeeze out peripheral genre cliques
For an artist seed, the candidate universe is built by gates keyed on similarity to the *seed artist*,
not the requested tag. A genre-tagged clique that is sonically *peripheral* to the seed (Ghost Box vs
BoC) is squeezed at each stage:
1. **Pier-bridgeability genre gate** (`artist_style.seed_genre_relevance_mask`, `genre_floor=0.30`):
   scores full genre-vector cosine to the seed's genre centroid. Ghost Box → BoC = 0.09–0.19 << 0.30
   → cut. (Their *sonic* sim to BoC is 0.49–0.55, closer than admitted artists — irrelevant to this
   gate.) 42304 → 9637 eligible.
2. **Genre-neighbor pool** (`build_genre_neighbor_candidate_pool`, `min_similarity=0.25`): same genre
   cosine, `break`s below the floor → Ghost Box never selected. → `allowed_ids ≈ 7021`.
3. **DS candidate-pool genre gate** (`candidate_pool.py`, admission percentile): more permissive
   (effective_floor ~0.05, hauntology-blended centroid) — Ghost Box actually PASS this (cos to blended
   centroid 0.66–0.73). So they reach the global DS pool.
4. **Segment-pool builder** (per pier-pair, ~800 candidates) + the **beam**: Ghost Box are dropped from
   the per-segment candidate sets (confirmed: even beam weight 10 never surfaces them — see F4). The
   beam is a worst-edge/transition optimizer; a per-candidate tag bonus is a weak signal there.

**Mitigation shipped:** an **allowed-set rescue** (`playlist_generator.py`) unions the authority on-tag
tracks into `allowed_ids` so they bypass gates 1–2 (log: "Tag steering allowed-set rescue: +52 …").
They then reach the DS pool but still lose at gate 4.

### F4 — Genre-dense discriminates on-tag membership; the sonic prototype does NOT (for cultural genres)
Two per-track signals: **genre-dense affinity** `cos(X_genre_dense, tag_target)` and **sonic-prototype
affinity**. For surfacing on-tag tracks *across artists*, genre-dense is the clean discriminator; the
sonic prototype cannot separate a cultural genre from sonic look-alikes. **Data (hauntology):**
| artist | SONIC-haunt | GENRE-haunt |
|---|---|---|
| Advisory Circle / Focus Group / Belbury | 0.57 / 0.70 / 0.61 | **0.94–0.97** |
| Aphex Twin / Autechre | 0.55 / 0.66 | **0.01 / 0.03** |
Sonic can't tell Ghost Box from Autechre; genre-dense nails it. **But** genre-dense is FLAT *within* a
genre-blended artist (Real Estate's tracks all ~0.6 to jangle), where the sonic prototype provides the
within-artist resolution. → **Use the combined signal: genre-dense leads (cross-artist membership),
sonic adds within-blended-artist resolution.** The beam term was upgraded to this combined signal
(default weight 0.0/opt-in); the pier lever already combines (Task 3). `X_genre_dense` is bimodal &
clean because it is baked from the **album-level** authority — "is this track on a tag-tagged album" is
a crisp per-track signal, ideal for tag-first pier selection.

### F5 — The engine optimizes the smoothest sonic journey FROM the seed
Even with on-tag tracks in the pool, the beam (worst-edge minimax; break-glass repair) picks the
smoothest-transition path from the seed's piers. BoC's smoothest neighbors are IDM (Aphex/Autechre),
not the sonically-peripheral Ghost Box. Soft tag steering layered on artist mode cannot override this.
**Contrast that works:** Real Estate + jangle produces a coherent jangle playlist (Seapony, The
Courtneys, Peach Kelli Pop, DUCKS LTD., Belle & Sebastian, The Telephone Numbers) *because jangle lives
inside RE's sonic neighborhood* — the on-tag tracks survive gates 1–4. The feature works when the tag ⊂
the artist's neighborhood; it fails for peripheral genre cliques. That failure is the case for genre mode.

---

## Genre mode — design sketch

**Entry:** a genre (+ optional artist filter). Reuse the mode plumbing (policy → `cohesion/genre/sonic/
pace` axes). Add `mode="genre"` alongside `artist`/`seeds`.

**1. Pier selection — tag-FIRST (fixes F2):**
- Candidate piers = the authority's on-tag tracks (`release_effective_genres` via `album_id`,
  non-inferred), ranked by (genre-dense affinity, optional sonic centrality within the tag).
- **Artist+genre flavor:** restrict to the artist's on-tag tracks (BoC → the 18 hauntology tracks:
  Introit, Prophecy At 1420 MHz, Memory Death, Deep Time, All Reason Departs…). **Pure flavor:** the
  library's on-tag tracks, deduped by artist for diversity.
- Select K piers by tag rank, then **order for arc** with the existing sonic-progression ordering
  (`order_clusters` / balance_gaps) so the arc discipline is preserved.
- Do NOT sonic-cluster first. (Optionally sub-cluster *within* the on-tag set for spacing.)

**2. Candidate pool — genre-centroid, not seed-neighborhood (fixes F3):**
- Universe = the tag's library-wide tracks (all 174 hauntology) + their sonic neighbors as connectors,
  NOT the seed artist's genre neighborhood. Skip the pier-bridgeability + genre-neighbor gates (F3
  gates 1–2) — or key them on the tag centroid instead of the seed centroid.
- Keep the sonic floor + diversity/min-gap (the hard constraints) so transitions stay listenable.

**3. Bridges/beam:** unchanged beam, but now the pool is on-tag-dominant, so the smoothest path runs
through on-tag tracks (fixes F5 by changing what's available, not by fighting the beam). The combined
genre-dense+sonic beam term (already built, opt-in) can add a mild on-tag preference.

**Reuse already shipped (this branch):** authority resolver (`resolve_tag_sonic_prototype_rows`, F1);
combined genre-dense+sonic signal (pier lever + opt-in beam term, F4); centered sonic pool lever;
allowed-set rescue (F3 mitigation → generalize to "compose from the tag set"); `genre_graph_canonical_genres`
name→id mapping.

**Acceptance:** BoC + hauntology → piers are BoC's hauntology tracks (Tomorrow's Harvest / Geogaddi);
bridges include the Ghost Box roster; worst-edge stays within one notch of artist mode. Real Estate +
jangle unchanged-or-better.

**Open questions:** (a) pure vs artist+genre as separate UI modes or one with an optional artist filter;
(b) how many piers for a pure-genre playlist (no single artist to anchor count); (c) diversity cap when
one artist dominates a niche tag (BoC is 18/174 hauntology; Ghost Box acts dominate); (d) does the
album-level genre-dense signal need track-level refinement for within-album variation.
