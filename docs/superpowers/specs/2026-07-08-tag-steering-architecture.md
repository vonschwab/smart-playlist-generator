# Tag / Genre Steering — Architecture & Findings (2026-07-08)

**Status:** authoritative consolidation of the tag-steering work on branch
`feat/tag-steering-sonic-prototype`. Read this before touching any tag/genre-steering code.
**Motivating case throughout:** *Boards of Canada seed + "hauntology" tag → the user expects a
hauntology-flavored playlist anchored on BoC's hauntological tracks and pulling in the Ghost Box roster
(Belbury Poly, The Advisory Circle, The Focus Group, Pye Corner Audio).*

**One-line status:** piers are now correctly on-tag (shipped win); **bridges are not** — Ghost Box are
blocked from the beam's effective candidate set by the segment-pool stage, the last unpinned gate.

---

## 1. What tag steering is, and the pipeline it flows through

A user selects genre chips (multi-select). Each request carries `tag_steering_tags`. Steering is a **soft
bias** — it never hard-gates. The signal has two parts, resolved once per run (`tag_steering.py`):

- **Genre-dense affinity** = `cos(X_genre_dense, tag_target)`. The tag_target is the dense-vocab embedding
  of the tags (`resolve_tag_steering_target`). This is the **clean cross-artist discriminator** for tag
  membership: for hauntology, Ghost Box score **0.94** vs Aphex/Autechre **0.01–0.03**. `X_genre_dense` is
  baked from the **album-level** authority (`release_effective_genres`), so it is bimodal/crisp: a track's
  affinity is high iff its album is authority-tagged.
- **Centered sonic prototype** = the global-mean-subtracted MuQ centroid of the tag's library tracks. This
  gives *within-blended-artist* resolution (Real Estate/jangle, where genre-dense is flat) but **cannot**
  discriminate a cultural genre from sonic look-alikes (Ghost Box sonic ≈ Autechre). Disabled when the
  tag is sonically multimodal (cohesion gate).

The **combined signal** (genre-dense leading + sonic additive) is what steers. It is consumed at every
stage of the artist-mode generation pipeline:

```
seed artist
  │
  ├─(A) PIER SELECTION        artist_style.cluster_artist_tracks  → which of the artist's tracks anchor
  │                            [tag-first: FIXED this session]
  │
  ├─(B) ALLOWED-SET           playlist_generator: external_pool + genre_neighbor_pool + pier_ids
  │      (candidate universe)  + on-tag RESCUE                     → the universe handed to the DS pipeline
  │
  ├─(C) DS CANDIDATE POOL      candidate_pool.build_candidate_pool → ~250-315 admitted (sonic floor +
  │                            [pool GUARANTEE: added this session]   per-artist rank walk + max_pool_size)
  │
  ├─(D) SEGMENT POOL           segment_pool_builder (dj_union)     → per pier-pair beam candidates (~400-800)
  │                            [THE UNPINNED GATE]                    universe = C, filtered by allowed_set/score
  │
  └─(E) BEAM                   pier_bridge/beam._beam_search_segment → picks interior bridges per segment
                               [worst-edge BAND: added this session]   objective = worst-edge minimax
```

Every stage independently ranks/filters by **sonic proximity to the seed**. A genre clique that is
sonically *peripheral* to the seed (Ghost Box vs BoC) is squeezed at each one. That emergent property — the
engine is architected end-to-end for a *smooth sonic journey from the seed* — is the real subject of this
document.

---

## 2. The gate map (every stage, with evidence)

| Stage | Mechanism | Does it drop Ghost Box? | Evidence |
|---|---|---|---|
| **Genre authority read** | which layer defines "on-tag" | Was YES (bug), now NO | Originally read `track_effective_genres` (MB-polluted, no file tags) → Ghost Box invisible. **Fixed:** read `release_effective_genres` via `authority.py`. Authority hauntology = 174 tracks incl. 120 Ghost Box. |
| **(A) Pier selection** | sonic-cluster-first medoids | Was YES, now NO | Piers were BoC's *sonically central* tracks (1/4 on-tag default, 2/4 max knobs). **Fixed (tag-first):** 4/4 on-tag (All Reason Departs, Blood In The Labyrinth, Memory Death, Somewhere Right Now In The Future). |
| **(B) Allowed-set genre gates** | pier-bridgeability genre floor 0.30; genre-neighbor min_sim 0.25 — keyed on the SEED's genre profile | YES without rescue | Ghost Box→BoC genre-cos 0.09–0.19 << 0.30. **Mitigated:** allowed-set RESCUE unions the authority on-tag tracks into the universe (+77 logged). |
| **(C) DS candidate pool** | sonic floor (percentile, in the 32-dim PCA sonic+genre embedding) + per-artist **rank walk** to max_pool_size | YES | `below_floor=3227/4003`; Ghost Box rank 526+ (median 6676) library-wide, only 2/120 in the pool window. Eligible-only guarantee found just **3/77** on-tag tracks eligible. **Addressed:** guarantee force-admits PAST the floor (21 on-tag into the pool), keeping the BPM hard gate. |
| **(D) Segment pool** | `dj_union` per pier-pair; `universe = C`, filtered by `allowed_set` + segment scoring | **YES — the live blocker** | Even with 21 on-tag tracks in pool C, **BEAMW=100 BAND=0.4 → 0 Ghost Box**. They are absent from the beam's effective candidate set → the DS-pool guarantee does not survive stage D. **UNPINNED.** |
| **(E) Beam objective** | worst-edge minimax; tag term added to `state.score` | Structurally inert | `worst_edge_minimax_enabled: true` makes `state.score` (incl. the tag term) a **never-firing tie-break** (beam.py:131, 1665: lexicographic on continuous `min_edge`). **Addressed:** worst-edge BAND makes the term able to act (worst-edge now responds: 0.176→0.132 at band 0.2) — but moot until stage D is fixed. |

**Key cross-cutting fact (finding F4):** genre-dense discriminates tag membership; sonic does not for
cultural genres. Any stage that ranks the on-tag set by *sonic* proximity (C rank walk, D dj_union, E edge
quality) will mis-rank Ghost Box as far, because sonic can't see the tag. Only a membership-keyed
(genre-dense / authority) admission can carry them — which is why the pool guarantee is keyed on authority
membership, not sonic.

**A genuine inconsistency worth noting (finding for stage D/E):** by the beam's *own* calibrated transition
metric, top Ghost Box → hauntology-pier edges are **smooth** — Belbury Poly "From an Ancient Star" T=0.868,
Advisory Circle "Seasons" T=0.822 — *better* than the accepted playlist's mean edge (0.683). So the beam is
not avoiding weak edges; the tracks simply never reach it. If they did, they would bridge well.

---

## 3. What we built this session (and its exact status)

| Change | Commits | Status |
|---|---|---|
| **Genre authority read** for tag steering | 59f4027, 932d65a | ✅ Shipped, correct |
| **Tag-first pier selection** (authority on-tag member set + soft top-up; 3-mode off/on/fire; multi-tag union) | 61f3e3c…9bb5008 | ✅ **Shipped, works** (piers 4/4 on-tag; RE+jangle no-regression, worst-edge improved 0.140→0.790; rollback verified) |
| **Allowed-set rescue** (on-tag tracks → candidate universe) | dcb5d41 | ✅ Shipped (necessary; +77 on-tag into the universe) |
| **On-tag pool guarantee** (force-admit past the rank walk + sonic floor; capped, per-artist-limited) | 16d0786, bd2f160, b2b64ec | ⚠️ **Built, correct, INSUFFICIENT alone** — puts 21 on-tag in pool C but they don't survive stage D |
| **Combined genre-dense+sonic beam term** | dcb5d41 | ⚠️ Built, inert under minimax (see band) |
| **Worst-edge band** (relax the minimax tie-break so the tag term can act) | b2b64ec | ⚠️ **Built, correct** (beam invariants green; worst-edge responds), **moot until stage D** |

**Config knobs (all under `playlists.ds_pipeline.pier_bridge`):**
- `tag_first_pier_selection: true` (LIVE), `tag_first_topup_mult: 2.0` — the shipped pier fix.
- `tag_steering_pool_guarantee_max: 30`, `tag_steering_pool_guarantee_per_artist: 3` — pool guarantee.
- `tag_steering_sonic_beam_weight: 0.0`, `tag_steering_worst_edge_band: 0.0` — OFF (inert until stage D).

---

## 4. Dead-ends & corrections (so the next session doesn't repeat them)

Intellectual honesty — several confident conclusions were **wrong** and corrected by the next probe:
1. **"Worst-edge wall"** (claimed the beam rejects Ghost Box as weak edges) — **WRONG.** Their edges are
   T=0.82–0.87, smoother than the accepted mean. Corrected by computing the actual calibrated transitions.
2. **"Collapse-prevention / genre-arc is the gate"** (Dylan's hypothesis, worth testing) — **DISPROVEN.**
   Disabling genre steering AND `anti_center` still yields 0 Ghost Box (though genre steering *was* hurting
   the worst edge, 0.176→0.527 with it off).
3. **"The beam term is a wiring bug"** — **NO.** It is correctly added to `state.score`; it's inert because
   `worst_edge_minimax` relegates `score` to a never-firing tie-break.
4. **"Pool guarantee (availability) is the fix"** — **INSUFFICIENT.** Availability in pool C ≠ availability
   in the beam (stage D drops them). Confirmed by BEAMW=100 BAND=0.4 → 0.

**Methodology lesson:** the decisive probe (compute the actual per-track transition T, then test extreme
weight+band) should have come first. "Read the logs / measure the metric the runtime consumes" (per the
`playlist-testing` + `evaluation-methodology` skills) would have short-circuited two wrong conclusions.

---

## 5. The remaining gate + recommended paths

**Next, precise task:** pin stage **D**. Instrument `segment_pool_builder` / the `dj_union` path
(`_build_segment_candidate_pool_scored`, `_build_dj_union_pool`) to log how many authority on-tag tracks
survive into the per-segment candidate set, and *where* they are dropped (`allowed_set` filter at
candidate_pool.py-analog line 448/619-621, the `dj_pooling_k_*` sonic-neighbor caps, or the segment
scoring). The universe is `C` (candidate_pool_indices, pier_bridge_builder.py:544), so either `allowed_set`
excludes the guaranteed tracks or the dj sonic-neighbor selection never picks them. Fix = carry the on-tag
guarantee into stage D the same way it was carried into C (a membership-keyed force-include, not sonic).

**If stage D is fixed, the full chain should light up:** on-tag tracks reach the beam (D) → the worst-edge
band lets the tag term prefer them (E) → and they bridge smoothly (T=0.87). All the pieces are built and
waiting on D.

**Strategic alternatives (if the layered fix keeps revealing gates):**
- **Waypoint steering** (roadmap `project_feature_roadmap_fun_features`): pin Ghost Box tracks as *anchors*
  the beam must include ("BoC → toward Belbury Poly"). Bypasses stages C/D/E entirely — the cleanest
  mechanism for "I want this specific clique."
- **Genre mode** (`docs/superpowers/specs/2026-07-08-genre-mode-design-notes.md`): a 3rd mode that composes
  *from the tag's library-wide tracks* as the primary structure, rather than from the seed's neighborhood —
  inverts the whole "proximity to seed" architecture that fights genre-concentration.
- **Transition-metric / embedding investigation:** raw MuQ says Ghost Box ≈ BoC-hauntology (0.75); the
  32-dim PCA sonic+genre embedding used for pool admission rates them far. If that reduction is losing MuQ
  signal, fixing it would help *all* genre-adjacent bridging, not just this case.

**The deep truth:** the engine is architected as a smooth-sonic-journey-from-the-seed optimizer, and
genre-concentration is resisted at pier, allowed-set, pool, segment-pool, and beam stages independently.
Tag-first pier selection fixed the anchor; surfacing a peripheral genre clique in the *bridges* is either a
last segment-pool fix (D) or a mode that stops composing from the seed's neighborhood (waypoints / genre
mode).
