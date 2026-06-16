# Genre System Redesign — Handoff

**Date:** 2026-06-16
**Author:** investigative session (triggered by a "broken" 5-seed playlist)
**For:** the genre-redesign session (end-to-end redesign of enrichment + genre traversal)
**Decision that prompted this:** current enrichments carry too much noise and the genre
traversal/steering system is structurally fragile. Patching is no longer the plan; redesign is.

> Legend: **[V]** = verified/measured in this session. **[M]** = from project memory or a
> prior doc; treat as a strong lead but re-verify before building on it.

---

## 0. TL;DR

The genre system fails in **two independent places**, and they compound:

1. **Enrichment is noisy and incomplete.** The published authority mixes a small set of clean
   observed tags with a large mass of *inferred* hub-family tags, and ~7.5% of generation-eligible
   tracks have no real signature at all. The smoothed vectors the engine scores on are densified to
   ~40% of the vocabulary. **[V]**
2. **Genre traversal goes infeasible easily.** Pier-to-pier steering routes through hub genres
   ("pop") whenever two anchors lack a shared neighborhood; the genre-gated beam then finds **zero**
   feasible paths and falls through to a last-resort fill. **[V]**

The trigger case: a single-artist album (Shabason & Krgovich — *At Scaramouche*) that is **legacy-only**
(3 stale tags, no enrichment) sat at a pier; steering toward it collapsed to the `pop` hub, the
segment went infeasible, and a genre-blind fallback filled it with sonically-near but genre-incoherent
tracks. A band-aid for the fallback shipped this session (see §9); the underlying genre problems are
untouched and are this redesign's subject.

**Most important measured facts for the redesign:**
- Smoothed genre space is **over-densified**: median **179/442** nonzero dims per track; random-pair
  cosine p50=0.121, **p90=0.645**, p99=0.881. Raw tags: median **5** nonzero, p50=**0.000**. **[V]**
- Authority is **inferred-heavy**: 19,452 `inferred_family` + 6,985 `inferred_parent` vs **16,460**
  `observed_leaf` + 918 `legacy` rows (43,815 total across 3,407 albums). **[V]**
- Coverage gap: of 40,393 generation-eligible tracks, **92.5% healthy / 4.2% legacy-only /
  3.3% unpublished**. Of the 294 degraded **albums**, only **26 are publish-away**; **268 need real
  enrichment** (no adjudicated data in the sidecar at all). **[V]**

---

## 1. Current architecture (the map you're replacing)

Authoritative reference: the `genre-data-authority` skill (`.claude/skills/genre-data-authority/`) —
it is the layer map and lists the past miswiring incidents. Read it first. Summary below. **[M/V]**

### Data flow
```
scan / genres / discogs / lastfm   →  raw tags (metadata.db: track_genres, album_genres,
                                       artist_genres, track_effective_genres)   [pipeline INPUT]
        │
        ▼  enrich (adjudication; now Claude-Max-backed)
ai_genre_enrichment.db (the "sidecar"):
   enriched_genres (12,229 rows, all status='accepted')           [pipeline-internal working data]
   genre_graph_release_genre_assignments (graph layer, keyed by release_id/artist/album)
   ... + review queue, source pages/tags, model priors, graph tables
        │
        ▼  publish  (ONLY writer of the authority)
metadata.db: release_effective_genres  (album-keyed; layers observed_leaf / inferred_family /
   inferred_parent / legacy; confidence; source graph|user|legacy)   ← THE AUTHORITY
        │
        ▼  artifacts (build_beat3tower_artifacts.py; genre_source: graph)
data/artifacts/.../data_matrices_step1.npz: X_genre_raw (442), X_genre_smoothed (442),
   + dense PMI-SVD sidecar X_genre_dense (64)                       ← what generation reads
```

### Who reads what at generation time
- **Candidate-pool genre gating:** `src/playlist/candidate_pool.py` — dense PMI-SVD (64-dim)
  per-seed admission, percentile floor. **[V]**
- **Genre edge penalty / steering during bridging:** `src/playlist/pier_bridge_builder.py` —
  uses `X_genre_smoothed` (442-dim, normalized) for the pairwise genre-edge floor/penalty, plus a
  **taxonomy graph** steering provider. **[V]**
- **Taxonomy graph:** `data/layered_genre_taxonomy.yaml` (SP3a, frozen at
  `v0.12.1-group1-pass9-edge-upgrade`, 408 canonical genres) via `src/genre/graph_adapter.py`;
  similarity matrix `data/genre_similarity_graph.npz`. Run log: "408 genres (455 path nodes),
  1965 edges, 21.5% off-diag nonzero." **[V]**
- **Authority read API:** `src/genre/authority.py` (`release_effective_genres`). Display/export only;
  generation reads the baked artifact. **[V]**

---

## 2. Problem 1 — Enrichment noise (evidence)

### 2a. Inferred hub-families dominate the authority
The authority stores far more inferred than observed rows (§0). Concrete example — **Bruce
Springsteen, "I'm on Fire"** has 2 clean user tags (`heartland rock`, `singer-songwriter`, conf 1.0)
and ~13 inferred family/parent rows including `electronic` (0.75) and `pop` (0.75). **[V]** These
inferred hubs are what let steering route anything to anything through `pop`.

> Note: a 2026-06-12 fix ("Mechanism-2") reportedly **excluded inferred layers from the artifact
> genre vectors** (observed_leaf + legacy only) and moved G p50 0.42→0.12. The measured p50=0.121
> below is consistent with that fix having landed. But the authority *display/storage* is still
> inferred-heavy, and the smoothed vector still has a noisy tail (see §4). Re-confirm exactly what
> the current `build_beat3tower_artifacts.py` bakes. **[M → re-verify]**

### 2b. Known collection-side noise sources (documented, partially fixed) **[M]**
From `docs/GENRE_DATA_QUALITY_FINDINGS_2026-06-12.md` and memory:
- **Stranger-over-user:** a Bandcamp *label storefront* page tagged a hardcore record "indie
  rock/pop" at 0.95 and fusion **replaced** the user's correct file tags. Fix implemented
  (bandcamp artist/label split, `ai_enriched_accepted` corroborating-only, never-drop local tags)
  but was "awaiting enrich re-run + publish" as of the last note — **verify it actually re-ran.**
- **Last.fm name-collision:** tags fetched by artist-name string mis-identified acts (e.g.
  "Green-House" ambient got a Ukrainian hip-hop act's tags) and published on all albums. ~76 artists.
  Source-side identity fix (stamp `probable`, verify identity) was still **open**.
- **Self-corroboration double-counting** of the same page.

### 2c. The legacy/unpublished hole has shape
The degraded 7.5% is **not random** — it clusters in exactly the adventurous corners that get seeded:
reissues / compilations / OSTs / world / jazz / dub / city-pop / ambient (FF8 OST, Hosono *BGM*,
King Tubby, Floating Points & Pharoah Sanders *Promises*, Flying Lotus *Flamagra*, Mulatu Astatke,
*Tokyo Glow* comp) plus **major unpublished artists** (Beyoncé ×6, The Replacements, Sonic Youth,
Hüsker Dü, Rosalía, OutKast, João Gilberto). **[V]** Impact is amplified at **pier positions**: a
thin vector at a seed breaks steering for the whole bridge segment.

### 2d. Fix-cost of the gap is mostly *collection*, not *publish* **[V]**
| Population | albums | publish-away (sidecar has data) | needs real enrichment |
|---|---:|---:|---:|
| legacy-only | 191 | 2 | 189 |
| unpublished | 103 | 24 | 79 |
| **total degraded** | **294** | **26** | **268** |

So a publish run fixes ~9% of the gap; the rest needs the collection+adjudication pipeline to run on
these releases at all. This is part of why "redesign" beats "patch."

---

## 3. Problem 2 — Traversal/steering is structurally fragile (evidence)

Mechanism, observed in the trigger run **[V]**:
1. Pier-bridge orders seeds to maximize *pairwise pier closeness* (`pier_bridge/seeds.py`, weights
   0.6 sonic / 0.2 genre / 0.2 bridge) — **feasibility-blind**: it placed the two least-bridgeable
   anchors (heartland-rock Springsteen ↔ legacy-only Shabason) adjacent, scoring it 0.9926.
2. Per segment, genre steering computes a taxonomy "route" between the two piers. With a thin
   destination vector, the only route was `via ['heartland rock', 'pop']` — one narrow leaf + the
   `pop` hub.
3. Under the pairwise genre-edge floor (0.10) + genre-arc floor, the beam assembled **no** valid
   path: `pool_after=0` after exhausting bridge-floor backoff + genre-arc relaxation (the repeated
   `attempt 1/2/3` blocks in the log).
4. The never-fail terminal fallback then filled the segment — historically on **sonic cosine alone**
   (now genre-aware; §9), producing genre-incoherent edges (G=0.025, 0.119).

**Redesign implications:**
- Steering routes through hubs whenever anchors are far; the taxonomy similarity is `pop`-permeable.
- "Infeasible" is reached easily and silently; the system leans on a last-resort fill more than it
  should (in the trigger run, **multiple** segments dropped to the fallback, not just one). **[V]**
- The genre signal *is* the right arbiter here (it correctly flagged the bad edges; sonic/MERT did
  not — MERT geometry is healthy, see §6), but its noise + traversal fragility means it can't be
  trusted to gate transitions as currently built.

---

## 4. Measured genre-space geometry (the "noise" quantified) **[V]**

Artifact `data_matrices_step1.npz`, 40,393 tracks, vocab 442:

| matrix | dim | nonzero/track (median / p95) | random-pair cos (mean / p50 / p90 / p99) |
|---|---:|---:|---:|
| `X_genre_raw` | 442 | **5** / 10 | 0.063 / **0.000** / 0.254 / 0.563 |
| `X_genre_smoothed` | 442 | **179** / 268 | 0.234 / 0.121 / **0.645** / 0.881 |

The **smoothing/propagation step over-densifies**: every track is spread across ~40% of the
vocabulary, giving two *random* tracks a p90 cosine of 0.645. The raw tags are clean and
discriminative but sparse. Any redesign should treat "how much inference/smoothing to bake into the
scored representation" as a first-class decision — the current answer is "far too much."

---

## 5. What is NOT the problem (so you don't chase it)

- **Mode wiring** is correct. narrow/narrow applied (sonic floor 0.18, weights 0.519/0.481). The
  `mode=dynamic` string in pool-gating logs is the *cohesion_mode* label, not the genre/sonic gate
  mode. **[V]**
- **The sonic embedding (MERT)** is healthy: well-centered, isotropic, no hub (random-pair cos
  mean≈0.001, hubness std 0.009; piers are not hubs). MERT *is* perceptually unreliable across
  dissimilar anchors (it rated Springsteen≈nu-jazz), but that is a known sonic limitation, not a
  genre defect, and not a botched implementation. The relevance to genre: genre is supposed to *catch*
  those sonic errors, and right now it can't reliably. **[V]**
- **The library lacks bridge material** — false. The correct soft-pop/sophisti bridge tracks exist
  and are correctly tagged (Nicholas Krgovich, Blood Orange, Kevin Krauter, Yumi Zouma); the engine
  just couldn't select them. **[V]**

---

## 6. In-flight / partial work — coordinate, don't redo **[M — verify status]**

- **Claude-adjudication roadmap** (ratified 2026-06-16, doc `608ca2e`): make Claude (Max) the primary
  genre adjudicator, eval-gated. ~70% of the plumbing reportedly already exists but dormant
  (`claude_client` + `model_prior`, 0 rows). Next planned step was a Phase-0 eval corpus; nothing
  writes the authority until a Phase-2 blind A/B gate. **This is the most directly relevant prior
  plan — read it before designing enrichment.**
- **Genre enrichment program (SP1–SP5):** make the layered graph the single genre authority;
  interaction via taxonomy, not co-occurrence.
- **Graph similarity integration (SP4):** artifact genres now sourced from `release_effective_genres`
  via `authority.py` (shipped `bfebef0`); remaining: legacy-fallback canonicalization, floor
  recalibration, layered runtime flip.
- **Genre similarity audition harness:** blind QA (graph vs co-occurrence vs decoy), spec committed,
  implementation pending. Use the `evaluation-methodology` skill for any "X beats Y" claim.
- **Mechanism-1 fusion fix** (bandcamp split / never-drop local / no self-corroboration): implemented,
  was **awaiting enrich re-run + publish** — confirm whether it landed.
- **SP3a taxonomy** is frozen complete (762 records); `taxonomy-growth` skill documents the growth loop.

---

## 7. Durable constraints the redesign MUST honor

From `CLAUDE.md` design principles (Layers 1–2 are non-negotiable) and project gotchas:
- **Sonic ⊗ genre fusion is the value prop**; multi-genre signatures must be preserved (don't collapse
  "shoegaze+dreampop+slowcore" to "indie rock"); **rare > common** when expressing taste.
- **Local-first:** external APIs enrich offline + export; they never gate runtime generation.
- **A configured knob that can't act is a startup error** — this codebase's recurring failure mode is
  config that looks wired but isn't. New gates must warn/raise on missing data, never silently no-op.
- **90-second hard ceiling** on generation; prefer **soft penalties over hard gates** (hard gates
  detonate the relaxation/expansion cascade into minutes).
- **Data safety:** `metadata.db` is irreplaceable (2× confirm + timestamped backup before any write);
  MERT shards/sidecar irreplaceable (~55h CPU); audio files permanently read-only.
- **Publish is the ONLY writer of `release_effective_genres`** (with a timestamped metadata.db backup
  on first run). Don't rewire readers to internal layers — fix the writer.

---

## 8. Open questions / decisions for the redesign

1. **Representation:** what exactly should the *scored* genre vector be? Raw observed tags (clean,
   sparse) vs smoothed (dense, noisy)? Is graph-propagation/inference helping or hurting transition
   scoring? (Measured: it hurts discrimination badly — §4.)
2. **Inference layers:** keep `inferred_family`/`inferred_parent` for *display/recall* but exclude
   from *scoring*? Or drop hub-inference entirely and rely on the taxonomy graph for adjacency?
3. **Traversal:** replace hub-permeable taxonomy routing with something feasibility-aware? Should
   seed ordering and segment construction be *genre-feasibility-aware* up front rather than relying on
   a last-resort fill?
4. **Enrichment scope:** 268 degraded albums need real enrichment. Is the Claude-Max adjudication
   pipeline the path, and what's the cost/throughput? Which releases are even *enrichable*
   (single-artist albums) vs structurally hard (Various-Artists comps/OSTs)?
5. **Authority vs artifact coherence:** published graph albums (3,214) exceed sidecar-enriched albums
   (1,585) — the graph absorbs legacy tokens via a path other than `enriched_genres`. Map that path;
   it affects what "re-enrich everything" actually means.
6. **Eval gate:** what blind A/B / audition proves the new genre signal beats the current one before
   it touches the authority? (See `evaluation-methodology` skill + the audition-harness work.)

---

## 9. This session's band-aid (context, not part of the redesign)

Committed on branch `worktree-genre-aware-greedy-fallback` (`0b028d4`):
`fix(pier-bridge): make never-fail greedy fallback genre-aware`. The terminal greedy placement
(`_greedy_terminal_path`) now blends genre-cosine-to-piers into selection
(`(1-w)·sonic + w·genre`, knob `infeasible_handling.greedy_genre_weight`, default 0.5, never-fail
preserved). A/B on the trigger run flips the Springsteen→Shabason interior from Brainfeeder/lo-fi/grunge
to Belle & Sebastian / Yumi Zouma / Sufjan / Billie Marten (mean genre-cos-to-Springsteen 0.36→0.52).
**This only makes the fallback less wrong; it does not address either root problem.** The redesign may
keep, retune, or supersede it.

---

## 10. Code & doc pointers

| What | Where |
|---|---|
| Authority read API | `src/genre/authority.py` |
| Published authority table | `metadata.db: release_effective_genres` (+ `genre_graph_canonical_genres`) |
| Enrichment sidecar | `data/ai_genre_enrichment.db` (`enriched_genres`, `genre_graph_release_genre_assignments`, review queue) |
| Enrichment code | `src/ai_genre_enrichment/` |
| Taxonomy graph (structure) | `data/layered_genre_taxonomy.yaml`, `src/genre/graph_adapter.py`, `data/genre_similarity_graph.npz` |
| Candidate-pool genre gate | `src/playlist/candidate_pool.py` (dense PMI-SVD 64-dim) |
| Genre steering + edge penalty + fallback | `src/playlist/pier_bridge_builder.py`; seed ordering `src/playlist/pier_bridge/seeds.py` |
| Artifact build (bakes genre vectors) | `scripts/build_beat3tower_artifacts.py` |
| Skills | `genre-data-authority` (the map), `taxonomy-growth` (SP3a loop), `evaluation-methodology` (A/B rigor) |
| Prior findings | `docs/GENRE_DATA_QUALITY_FINDINGS_2026-06-12.md`, genre-resources audit (`608ca2e`), `audit/07-roadmap.md` |
| Relevant memories | `project_genre_claude_adjudication_roadmap`, `project_genre_enrichment_program`, `project_graph_similarity_integration`, `project_enriched_genre_authority`, `project_genre_embedding_anisotropy`, `project_playlist_quality_diagnosis`, `project_timbre_embedding_ceiling` |

### Reproduce the key measurements
- Authority layer split: `SELECT assignment_layer, COUNT(*) FROM release_effective_genres GROUP BY 1`.
- Coverage gap + fix-cost: classify artifact `track_ids` → `album_id` → presence of non-legacy rows in
  `release_effective_genres`, then join `album_id` against `enriched_genres` in the sidecar.
- Genre geometry: random-pair cosine + nonzero-per-track on `X_genre_raw` vs `X_genre_smoothed` in
  `data_matrices_step1.npz` (script used this session is in the conversation log).
