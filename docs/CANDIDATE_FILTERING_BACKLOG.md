# Candidate Filtering — Backlog

Captured 2026-05-21 during the candidate-filtering optimization brainstorm.

The current pipeline has accumulated cruft, half-disabled subsystems, and several real asymmetries that bias filtering toward the primary seed. The five categories below were surfaced together; this brainstorm scopes a fix for the immediate priority (category B's IDF rework) and parks the rest here.

---

## Filtering stages today (reference map)

**Upstream (artist mode only, `src/playlist/artist_style.py`):**
1. Per-cluster sonic pools (default 2000 × cluster_k clusters)
2. Genre-neighbor pool (default 1500 tracks; has its own `min_confidence` gate, currently null)
3. Union of (1) + (2) → `allowed_track_ids`

**Upstream (all modes, `src/playlist/candidate_pool.py`):**
4. Hybrid embedding similarity to seeds — max over seeds → `similarity_floor: 0.20`
5. Sonic-only PCA similarity to seeds — max over seeds → `min_sonic_similarity: 0.12` (dynamic)
6. Title hard-exclusion list (`title_exclusion_words`: interlude/skit/acapella only)
7. Genre similarity vs primary seed only → `min_genre_similarity: 0.40`
8. Genre overlap guard (raw shared-tag check)
9. Genre conflict penalty vs primary seed's raw tags → `strength: 0.20`
10. Genre conflict hard gate (`min_confidence`, currently null/off)
11. Duration penalty + cutoff
12. Artist diversity cap (`candidates_per_artist`, `target_artists`)

**Per-segment (inside pier-bridge, `src/playlist/segment_pool_builder.py`):**
13. `min(sim_to_pier_a, sim_to_pier_b) >= bridge_floor: 0.02`
14. Harmonic mean of pier similarities → ranking
15. Artist policy + track-key dedup
16. 1-per-artist collapse (default off in `config.yaml`)

---

## Category A — Asymmetries (real bugs)

### A1. Genre admission filter uses primary seed only
`candidate_pool.py:311` does `seed_genres = genre_matrix[seed_idx]`. Steps 4 and 5 (hybrid + sonic) use `np.max(... seed_vecs.T, axis=1)` across all seeds. The genre filter doesn't — it filters only against the first seed's genre tags. In multi-seed playlists with diverse seeds, the other seeds' genre profiles are ignored at admission.

### A2. Genre conflict penalty / gate uses primary seed only
Same bug at `candidate_pool.py:327`. `seed_raw = np.max(genre_raw_matrix[seed_list], axis=0)` does aggregate over seeds — so the conflict math actually *does* use max-over-seeds (good). But the similarity filter at step 7 (A1 above) does not. The two are out of sync.

### A3. Genre-neighbor pool in artist mode uses primary seed only
`artist_style.py:build_genre_neighbor_candidate_pool` compares candidates against a single artist's smoothed genre vector. For multi-seed artist playlists this is fine because there is one seed artist. Worth re-checking for consistency.

---

## Category B — Duplicate / overlapping systems

### B1. Two title-quality systems coexist
- `candidate_pool.py:is_title_excluded(...)` — hard list of 5 keywords (`interlude`, `skit`, `acapella`, `a cappella`, `a capella`)
- `src/playlist/title_quality.py:detect_title_artifacts(...)` — richer detector (live, demo, medley, remix, instrumental, remaster, version, take, mono, stereo, edit, outtake, alternate) with word-boundary regex

The hard exclusion at admission uses the weak one. The soft beam penalty and the edge-repair candidate filter use the rich one. Consolidate: have `candidate_pool` use `detect_title_artifacts` with a config-driven set of "always exclude" flags (a subset; default just the 5 current ones, opt-in to broader exclusion).

### B2. Two genre-similarity systems
- Candidate admission uses `_compute_genre_similarity` (cosine / weighted-jaccard / ensemble) on smoothed vectors **without IDF weighting**
- DJ bridging waypoint code uses **IDF-weighted vector mode** for routing

The base admission filter treats `indie` and `slowcore` as equally important. The waypoint code knows `slowcore` is rare and meaningful. **Scope B of this brainstorm covers this.**

---

## Category C — Half-disabled systems

### C1. Genre conflict — half a system
The penalty (step 9) is on at `strength: 0.20`. The hard gate (step 10) is null/off. The penalty itself was reduced from 0.30 to 0.20 in v4.1 after the gate-at-0.50 caused regressions. The whole subsystem now does *some* work but is confusingly named "genre_conflict_*" because it originally was a gate-plus-penalty.

Options: (a) remove the gate code path entirely, (b) keep the penalty but rename the public config field to `genre_conflict_penalty_*`, (c) re-enable the gate with a much lower threshold informed by the v4.1 lessons (e.g. 0.10 instead of 0.50).

### C2. Overlap guard partially redundant with broad_filters
`candidate_pool.py:412` (overlap guard) is justified as "prevents smoothed vectors from admitting tracks that only match through generic tags such as 'rock' or 'pop'." But `broad_filters: ["rock", "indie", "alternative", "pop"]` already masks those tags *out* of the genre vector before similarity is computed. The guard's job is mostly already done by the mask. Worth confirming with a counter-example and either documenting the residual case or removing the guard.

---

## Category D — Missing capabilities

### D1. Broad-genre collapse prevention is passive only
`broad_filters` is a static block-list. It doesn't *reward* candidates with rare-tag matches; it just hides common tags. **IDF weighting in admission (Scope B) is the positive-pressure version of this.**

### D2. Easy-out sonic collapse — no prevention
No mechanism prevents the system from picking 3+ consecutive ambient/noise/drone tracks. The pier-bridge progress constraint ensures monotonic sonic motion, but doesn't prevent the *texture* (timbre) from staying flat for long stretches.

Options: a "subgenre run guard" (no more than N consecutive tracks sharing top-1 subgenre/timbre cluster), or a per-edge texture-divergence reward (favors timbre change within sonic continuity).

### D3. Subgenre arc planning doesn't exist
The pier-bridge picks tracks that bridge well between pier pairs, but doesn't actively ensure the *journey* visits varied subgenres. We have IDF-weighted waypoints in dj_bridging but they're not constraints, just bonuses.

---

## Category E — Underused per-segment gate

### E1. bridge_floor is doing essentially nothing
`bridge_floor: 0.02` (dynamic mode) is a near-zero hybrid-cosine threshold. Combined with a typical candidate pool that's already filtered to high-similarity tracks, almost everything passes. Per-segment "filtering" is effectively just "rank by harmonic mean".

Options: (a) raise `bridge_floor` to a meaningful value (e.g. 0.15) and let it gate; (b) drop `bridge_floor` entirely and document that segment filtering is purely rank-based.

---

## Scope completed (2026-05-21)

**Scope B — Bugs + IDF-weighted admission genre filter — done.**

Items completed:
- A1 (genre admission max-over-seeds) ✓
- A2 (genre compatibility consistency check) ✓ — already correct; documented
- B1 (title quality consolidation) ✓
- B2 (IDF weighting in admission genre similarity) ✓
- C1 (genre conflict → genre compatibility rename; dead gate code removed) ✓

Deferred to follow-up brainstorms (see categories above):
- A3 (genre-neighbor pool primary-seed-only filter)
- C2 (overlap guard redundancy investigation with broad_filters)
- D1, D2, D3 (positive-pressure subgenre diversity, easy-out sonic prevention, subgenre arc planning)
- E1 (bridge_floor cleanup; currently 0.02, doing essentially no filtering)
