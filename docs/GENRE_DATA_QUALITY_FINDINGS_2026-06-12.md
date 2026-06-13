# Genre Data Quality — Diagnosis, Fixes, and Lessons

**Date:** 2026-06-12 (investigation + fixes); verified 2026-06-13
**Branch:** `master` (changes uncommitted at time of writing — see §9)
**Status:** **SHIPPED.** Three of four root-cause mechanisms fixed and live;
the fourth (sonic embedding) is the MERT lane, out of scope here.
**Scope:** Why playlists kept getting *worse* as we added genre machinery, what
was actually broken in the genre data pipeline, the fixes we shipped, the
dead-ends we hit on the way, and the methodology lessons that matter more than
any single fix. Read-only diagnosis throughout; every write (`metadata.db`,
sidecar, artifact) was backed up and gated behind a dry-run.

---

## TL;DR

1. **"More methods made playlists worse" was real, but not because of the
   methods.** Every new gate/steering objective was expressed in one of two
   genre representations that were quietly corrupt. Better machinery optimizing
   broken inputs produces *confidently* worse output.

2. **Four mechanisms, each measured, not guessed:**
   - **(1) Enrichment trusted strangers over the user.** A single Bandcamp
     *label storefront* page tagged a hardcore record "indie rock / pop" at
     confidence 0.95 and *replaced* the user's correct file tags. Self-corroboration
     (the same page counted twice) and a blanket skip of `track:file` evidence
     made it worse.
   - **(1c) Last.fm name-collision contamination.** Last.fm tags are fetched by
     artist-*name string*; "Green-House" (LA ambient) matched a Ukrainian
     hip-hop act and published `hip hop` / `underground hip-hop` on all six
     albums. ~76 artists affected, overwhelmingly generic names.
   - **(2) Inferred hub-families saturated genre similarity.** The artifact baked
     inferred ancestors (`rock`, `pop`, `indie/alternative`) into every release's
     genre vector. Random-pair genre cosine sat at **p50 ≈ 0.42** — the genre
     signal carried almost no information.
   - **(3) The sonic embedding can't hear.** Even at rank #1, the tower embedding
     ranks a production twin below cross-genre noise. This is the timbre ceiling;
     the MERT lane addresses it. **Not fixed here.**

3. **The QC was blind because it measured the same broken spaces it was scoring.**
   A run with min-T 0.53 and `below_floor=0` reported "healthy" while serving the
   worst playlist. Centered T + saturated G hid every bad edge.

4. **The genre fixes shipped and the named failures are gone.** On the original
   10-diverse-seed playlist: VV Torso (hardcore) now sits in the punk cluster
   instead of bridging to indie rock; Springsteen is approached from Magnolia
   Electric Co. (heartland→heartland) instead of Bill Callahan (folk→synth-pop);
   Green-House is ambient again. Genre cosine de-saturated (random-pair p50
   0.42 → 0.12; in-playlist G min 0.55 → 0.33 — i.e. genre can finally say
   "these are *different*").

5. **The biggest process lesson: wholesale re-derivation is the wrong vehicle for
   a policy fix.** Re-running the whole library through the corrected fusion
   policy *un-decided good past calls* (it would have stripped Duster's correct
   Last.fm-only "dream pop", and re-materialization caused catastrophic collateral
   — Beach Fossils losing 6 correct tags to gain 1 generic). The right vehicle was
   a **surgical, additive/subtractive delta** that touches only the broken
   observed-leaf rows and grandfathers everything else.

---

## 1. The symptom and the misframing

The trigger was a 10-diverse-seed playlist that was "still pretty bad" despite
the genre graph, tuned sonic similarity, and genre steering all being live. The
user's instinct: *"our enrichment process is not being discriminatory enough —
we allow albums to be tagged with genres that are technically genres but have
nothing to do with the music. VV Torso is tagged 'pop' but it is hardcore punk."*

That instinct was correct about the symptom but pointed at the wrong layer.
"Too many genres inferred" was a reasonable read, but the inference was largely
innocent: `underground hip-hop` on Green-House was a fully **observed** leaf at
0.95, not an inference. The rot was upstream, in collection and fusion, and it
was being *amplified* by inference and *hidden* by saturation.

**Lesson:** a user-reported symptom localizes the *pain*, not the *cause*.
Trace the full provenance chain (collection → fusion → publish → artifact →
generation) before acting. We found the cause three layers above where it hurt.

---

## 2. The diagnosis (measured)

All four mechanisms were confirmed with data, not asserted.

### Mechanism 1 — enrichment replaces correct tags with junk
Natural experiment: VV Torso `LPVV` (enriched) published `indie rock`; sister
album `LPVVII` (never enriched) kept the correct `hardcore / post-punk / punk`.
**Enrichment strictly degraded the data.** Trace: a single label-storefront page
(`jurassicpop.bandcamp.com`) → tags `indie, indie rock, pop, rock, cleveland` →
`bandcamp_release` weight 0.95 (joint-highest) → auto-accept. The user's file
tags scored ~0.70 and went to a 1,312-release / 34k-row review queue that was
never applied. The same page was counted twice (`bandcamp_release` +
`ai_enriched_accepted`, `evidence_count=2`).

### Mechanism 1c — Last.fm wrong-identity (found later, same root family)
Green-House published `hip hop` + `underground hip-hop` on all six albums,
provenance `sources=['lastfm_tags']`. The Last.fm fetch resolves by artist-name
string and `extract-lastfm` stamps `identity_status="confirmed"` unconditionally
(`scripts/ai_genre_enrich.py` ~line 841). A zero-overlap detector (artist's
Last.fm tags share nothing with ≥2 other-evidence tags) flagged **~76 artists**,
overwhelmingly generic names (Bob, Chad, Betty, Domi, Archetype, "green house").

### Mechanism 2 — inferred families saturate the genre vector
`build_beat3tower_artifacts.py` baked `inferred_family`/`inferred_parent` rows
into `X_genre_*` at 0.5×conf. Measured random-pair cosine of `X_genre_smoothed`:
**p50 = 0.42, p90 = 0.78**. Edge G of ~0.8 therefore carried almost no signal.
A latent bug compounded it: the `legacy` layer wasn't in the weight map, so the
silent `.get(layer, 0.5)` default *halved* the correct tags of every un-enriched
album.

### Mechanism 3 — the sonic embedding is perceptually coarse
For "Dancing in the Dark," the same-album production twin "Working on the
Highway" scored S=0.366 while Cap'n Jazz (midwest emo) was the global #1 at
0.470, with a hip-hop collage and a 90s R&B ballad in the top 10. Edge-S mean of
the playlist (0.327) was already the top 0.4% of random pairs — the generator
operates *at the ceiling* of a space whose ceiling is wrong. This is the MERT
lane; nothing in this work changes it.

### The meta-failure — QC blind to all of the above
The run reported min-T 0.53, `below_floor=0`, "healthy," while serving the worst
playlist. T-centering compressed the scale and saturated G propped T up across
genre-incoherent edges. **A metric computed in the space you changed cannot
validate that space.**

---

## 3. The fixes, in execution order (including the dead-ends)

The dead-ends are the most instructive part. Documented honestly.

### Fix A — inferred families out of the genre vector (Mechanism 2). SHIPPED.
`build_beat3tower_artifacts.py`: vectorize **observed_leaf + legacy at full
weight only**; exclude inferred layers (they re-enter via the similarity matrix,
the controlled mechanism); keep inferred-only albums at a damped fallback so they
don't become zero vectors; **raise on unknown layers** (fixes the silent
legacy-halving bug). Result: random-pair G p50 0.42 → 0.12. A/B on the diverse
seeds moved the worst-edge S floor 0.155 → 0.203 with the median held.

### Fix B — fusion policy rebalance (Mechanism 1). SHIPPED (code).
`src/ai_genre_enrichment/hybrid_evidence.py`:
- **Bandcamp artist/label split** (`classify_bandcamp_source`): a page whose
  subdomain matches the artist is self-reported and stays top-tier (0.95); a
  multi-artist domain is a label storefront (0.60, never auto-accepts); a
  mismatched single-artist domain is "unknown" (0.70, no solo auto-accept).
- **`ai_enriched_accepted` demoted to corroborating-only** — no more
  self-corroboration from the same page.
- **`local_metadata` raised to 0.80** with a **never-drop rule**: the user's
  file tags always survive to at least provisional (which publishes).
- **`track:file` / `album:file` injected** into fusion (was skipped wholesale).
- **Last.fm-only → review at any confidence** (a later tightening, after the
  publish dry-run caught `baroque` promoting onto a Debussy record).

### DEAD-END 1 — wholesale re-materialization (rejected twice at the gate).
The obvious deployment: re-run `graph-build-assignments` over the whole library
so the authority matches the new fusion policy. **Both attempts were rejected at
the publish dry-run.** Pass 1 surfaced Last.fm junk still promoting; after the
Last.fm tightening, Pass 2 exploded removals to **5,525** — because much *correct*
published data has Last.fm-only or graduated-acceptance ancestry that the strict
policy now (rightly, for new decisions) distrusts. Duster's "dream pop" is
Last.fm-only **and correct**. You cannot fix VV Torso by re-litigating Duster.
Sidecar rolled back both times; `metadata.db` never touched.

### DEAD-END 2 — assignment-level rebuild from observed leaves.
Next attempt: surgically edit observed leaves, then re-derive inferred rows from
them. Broke on Green-House: its only correct genre (`ambient`) lives at the
*family* level from direct Bandcamp evidence, not derivable from any observed
leaf — the rebuild dropped it. **Re-deriving inferred structure loses
direct-evidence broad genres.**

### DEAD-END 3 — re-materialize only the touched releases (collateral).
Targeted re-materialization through the real materializer fixed Green-House but
caused **catastrophic collateral on Delta-A releases**: Beach Fossils gained
`indie_rock` but lost `dream_pop, shoegaze, post_punk, noise_pop, jangle_pop,
surf_rock`; Azymuth gained a wrong `hip_hop` and lost its jazz. Re-materialization
applies Last.fm-only→review to the *whole* release, so it drops correct
Last.fm-only tags as a side effect. **Caught by inspecting the per-release
dry-run diff, not the summary counts.**

### Fix C — surgical delta migration (Mechanism 1 + 1c legacy cleanup). SHIPPED.
`src/ai_genre_enrichment/assignment_migration.py` +
`scripts/migrate_assignments_delta.py`. Pure observed-leaf surgery, never a
recompute — only `observed_leaf` is vectorized for generation, so this is both
correct and collateral-free:
- **Delta A (additive):** add taxonomy-leaf local file tags missing from observed.
- **Delta B (subtractive):** remove storefront-only observed leaves **only when**
  the user has contradicting local tags (reissue-label comps without local
  curation, e.g. Nigeria 70, are grandfathered by *not* being selected).
- **Delta C (subtractive, unconditional):** remove Last.fm-only observed leaves
  on contradicted artists, even if it empties the observed set — a wrong-identity
  tag is worse than sonic-only steering. Drops the matching inferred chip too,
  but preserves unrelated inferred rows (Green-House's `ambient` survives).
- Guards: storefront removal never empties; user-added (override) tags never
  removed; `__empty__` sentinel filtered.

Full dry-run: **267 releases touched, 178 pure additions, 209 adds / 182
removes** — vs. the 5,525-removal wholesale disaster. Verified zero collateral
(Beach Fossils keeps all 8 tags; Acetone keeps slowcore).

### Execution sequence (each step backed up, each write gated)
1. `migrate --apply` → sidecar (`bak_20260612_225917`).
2. `publish` real → `metadata.db` (manual backup `bak.20260612_231315`; publish
   does **not** auto-backup once `release_effective_genres` exists).
3. Rebuild artifact + dense sidecar (`bak_20260612_233322`).
4. Restart GUI (worker `@lru_cache`s the bundle).

---

## 4. Verification (before → after, original 10 diverse seeds)

| Signal | Original (broken) | After |
|---|---|---|
| VV Torso placement | → Courtney Barnett (hardcore→indie rock) | in punk cluster (Tyvek → VV Torso → We Are Scientists) |
| Springsteen approach | from Bill Callahan (folk→synth-pop) | from Magnolia Electric Co. (heartland→heartland) |
| Green-House | `hip hop` + `underground hip-hop` | `ambient` (no hip-hop on any of 6 albums) |
| Random-pair genre cosine p50 | 0.42 | 0.12 |
| In-playlist G min / mean | 0.55 / 0.795 | 0.33 / 0.728 |
| In-playlist S mean | 0.327 | 0.305 |
| min T | 0.531 | 0.503 |

The lower mean-T/min-T is **not** a regression: with genre de-saturated, T
reflects real sonic texture instead of being propped up by a rubber-stamp G. The
remaining weak edges are genuine *style seams between disparate seeds*
(Springsteen→Krgovich), which is the MERT problem, not a data problem. The easy
case (Green-House artist mode, narrow) produced a fully coherent ambient set with
the canonical neighbors (Hiroshi Yoshimura, Inoyama Land, Emily Sprague).

---

## 5. Lessons (the part to actually remember)

1. **A metric computed in the space you're changing cannot validate that space.**
   Genre QC scored inside the enriched-genre space said "fine" while the tags
   were junk. At least one arm of any quality check must be independent: ears,
   held-out labels, or a different modality. (Now codified in the
   `evaluation-methodology` skill.)

2. **Distributions over means; the floor is the product.** A healthy mean T hid a
   broken worst edge. Saturated/centered metrics hide floor failures. Report
   min / p10 / p50 / p90, and treat the worst edge as the experience (north-star
   #5).

3. **Trust the user's own data over a stranger's.** Hand-curated file tags are the
   ground truth for niche releases — often the *only* correct source. A web
   scrape, especially a storefront or a name-string match, is weak evidence by
   construction. Source quality must cap confidence; richer-looking ≠ truer.

4. **A policy fix is not a data migration.** Fixing the fusion *rules* (forward)
   and fixing the *legacy state* are different jobs. Re-deriving history through
   new rules un-decides good past calls. Use a surgical delta that changes only
   what's provably wrong and grandfathers the rest.

5. **Inspect the diff, not the summary.** The Beach Fossils collateral was
   invisible in the headline counts (267 touched, X removed) and obvious in the
   per-release lines. Before any irreversible write, read the actual rows.

6. **"More tags" is usually saturation, not richness.** Inferred ancestors and
   multi-source echoes inflate similarity without adding signal. Keep them for
   display; keep them *out* of similarity vectors.

7. **Direct-evidence broad genres are signal; inherited hub families are noise —
   and the data model conflates them.** "ambient" from a Bandcamp page and "rock"
   inferred as a distant ancestor of hardcore are both stored as
   `inferred_family`. Today we exclude both from vectors (correct for the hub
   noise, lossy for the direct-evidence broad genre). See §6.

8. **Back up before every write; dry-run before every irreversible step; gate the
   `metadata.db` write on explicit confirmation.** This discipline turned three
   wrong approaches into rolled-back experiments instead of corrupted production.

---

## 6. What remains (non-breaking follow-ups)

- **MERT sonic embedding** — the real quality ceiling now that genre is clean.
  Within a coherent genre, G can't order tracks and the sonic space is too coarse
  to. This is the queued lane; validate with neighbor-list sniff tests
  (same-album production twins must outrank cross-genre noise).
- **Direct-evidence broad genres should count in vectors.** Green-House currently
  steers sonic-only because `ambient` is family-level and excluded. Distinguish
  "broad genre from direct release evidence" from "hub family inherited via
  ancestry" and admit the former to `X_genre_*`. (Lesson 7.)
- **Enrichment Last.fm identity gate** — stamp `probable`, not `confirmed`, on
  name-string matches; verify against other evidence before publishing. Stops the
  Green-House class at the source (currently only cleaned in legacy state).
- **Stop injecting artist-level MB/Discogs catalog tags as release evidence** —
  contributed the over-tagging still visible on Bill Callahan (dance-pop/funk).
- **Genre steering contributes zero winning picks** — every segment logs
  `baseline_only`; steering acts only through the pairwise penalty. Separate
  "is steering actually steering?" investigation.
- **Per-edge BPM audit shows `n/a`** — cosmetic; the reported `edge_scores` don't
  carry `bpm_a/bpm_b` forward (`reporter.py:179`, populated only in
  `pier_bridge_builder.py:2360`). BPM *is* gating admission. Wire if the
  diagnostic is wanted.
- **Review queue** (~1,312 releases / 34k rows) is still unapplied; the surgical
  migration deliberately did not touch it. Future enrich runs under the new
  fusion policy will drain it correctly.

---

## 7. Tests

TDD throughout (~40 new/updated cases, all green; lint clean):
- `tests/unit/test_ai_genre_hybrid_evidence.py` — bandcamp classification, no
  self-corroboration, never-drop local, Last.fm→review, LPVV regression.
- `tests/unit/test_ai_genre_hybrid_cli.py` — end-to-end fusion via the CLI.
- `tests/unit/test_assignment_migration.py` — the three delta selectors +
  surgical apply (Green-House keeps ambient, Beach Fossils zero collateral).
- `tests/unit/test_artifact_builder_graph.py` — inferred-layer exclusion,
  legacy full weight, unknown-layer raises, inferred-only fallback.

---

## 8. Related docs / skills

- `.claude/skills/evaluation-methodology` — now encodes the QC-blindness and
  inspect-the-diff lessons.
- `.claude/skills/genre-data-authority` — the layer map (authority =
  `release_effective_genres` via `src/genre/authority.py`).
- `docs/SONIC_PHASE2_HARMONY_FINDINGS.md` — the sonic lane (Mechanism 3 context).
- `docs/AI_GENRE_ENRICHMENT.md` — enrichment pipeline overview.

---

## 9. Changed files (uncommitted at time of writing)

- `src/ai_genre_enrichment/hybrid_evidence.py` — fusion policy.
- `src/ai_genre_enrichment/assignment_migration.py` — delta selectors + surgical apply (new).
- `src/ai_genre_enrichment/storage.py` — `bandcamp_domain_artist_counts`, source_domain/normalized_artist in the hybrid read.
- `src/ai_genre_enrichment/layered_assignment.py` — split `compute_layered_assignment_rows` out of `materialize_*`.
- `scripts/build_beat3tower_artifacts.py` — inferred-layer exclusion + legacy weight + unknown-layer raise.
- `scripts/migrate_assignments_delta.py` — the migration runner (new).
- Tests as in §7.

**Backups created (keep until the change is committed and confirmed stable):**
`metadata.db.bak.20260612_231315`, `ai_genre_enrichment.db.bak_20260612_225917`
(+ `_163129`), `data_matrices_step1*.bak_20260612_233322`.
