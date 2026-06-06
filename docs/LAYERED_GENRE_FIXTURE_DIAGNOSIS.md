# Layered Genre Fixture Diagnosis

This document records the first practical taxonomy-hardening pass after the
initial layered graph foundation. The goal of this pass was visibility and
regression coverage, not broad new architecture.

## What Changed

- `graph-show-release` now supports exact `--release-key` inspection.
- Release inspection now reports raw evidence, normalized evidence, accepted
  leaf terms, accepted broad terms, inferred terms, review terms, rejected terms,
  missing taxonomy terms, model-prior presence, and zero-assignment status.
- Inferred terms include the explicit taxonomy edge when the stored provenance
  can be traced back to a direct curated edge.
- `graph-fixture-report` runs a repeatable smoke suite from
  `data/layered_genre_smoke_fixtures.yaml`.
- `graph-fixture-report --build-assignments` materializes layered assignments in
  the selected sidecar before evaluating fixtures.
- Fusion policy no longer rejects real broad genres just because they are broad.
- Fusion policy explicitly rejects standalone `indie` and fake `pop/rock`.
- The taxonomy now includes canonical `indie rock`.
- `dream pop -> pop` parent inference was removed. `dream pop` no longer infers
  `pop` merely because the term contains "pop".

## Current Fixture Results

The latest smoke run used production metadata read-only and a disposable sidecar:

```text
C:\tmp\ai_genre_enrichment_layered_hardening_smoke.db
```

Command:

```powershell
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe scripts\ai_genre_enrich.py --metadata-db C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db --sidecar-db C:\tmp\ai_genre_enrichment_layered_hardening_smoke.db graph-fixture-report --build-assignments
```

Summary: 4 pass, 2 fail.

## Per-Fixture Diagnosis

### Duster - Stratosphere

Status: pass.

Accepted terms:
- `dream pop`
- `indie rock`
- `post-rock`
- `shoegaze`
- `slowcore`

Inferred:
- `rock`

Rejected:
- `indie`

Still missing from taxonomy/review clarity:
- `slacker rock`
- `space rock`
- `space rock revival`
- noise/year terms such as `1998`, `drone`, `drone rock`

Diagnosis: core behavior is now much better. `dream pop` no longer infers `pop`;
`rock` is retained through explicit style context. Next step is taxonomy seeding
for `space rock` and likely `slacker rock`.

### Mount Eerie - Sauna

Status: pass by current minimum criteria, but still under-described.

Accepted terms:
- `indie folk`
- `indie rock`

Inferred:
- `folk`
- `rock`

Rejected:
- `indie`
- `black metal`
- `psychedelic folk`

Review/missing taxonomy includes:
- `avant garde`
- `drone`
- `lofi`
- `singer songwriter`
- `psychedelic folk`

Diagnosis: no longer collapses to one assignment, but the taxonomy/review policy
is still too crude. `psychedelic folk`, `singer-songwriter`, and `lo-fi` need
proper handling as genre/facet/alias decisions rather than ad hoc rejection or
review.

### Ada Lea - One Hand on the Steering Wheel the Other Sewing a Garden

Status: pass.

Accepted:
- `indie rock`

Inferred:
- `rock`

Rejected noise:
- `mixtaperoom`
- `rare sad girl`
- `rare sads`
- `indie`
- `folk rock`

Review/missing taxonomy includes:
- `folk rock`
- `singer songwriter`
- `canada`

Diagnosis: noise rejection works. The useful assignment is still thin. Next step
is to decide whether `folk rock` and `singer-songwriter` are canonical genres,
aliases, or review-only for this source mix.

### Stereolab - Dots and Loops

Status: pass by current minimum criteria, but taxonomy is clearly under-seeded.

Accepted:
- `indie pop`
- `indie rock`
- `post-rock`
- `electronic`
- `pop`
- `rock`

Rejected:
- `indie`

Review/missing taxonomy includes:
- `art pop`
- `avant garde pop`
- `experimental rock`
- `indietronica`
- `krautrock`
- `noise pop`
- `electronica`

Diagnosis: assignment no longer silently fails, but this is the clearest taxonomy
gap. Stereolab should drive the next seed update for art pop, experimental pop,
krautrock, noise pop, indietronica, and related aliases/facets.

### Explosions In The Sky - All of a Sudden I Miss Everyone

Status: fail.

Evidence status:
- `no_evidence`

Diagnosis: this is not currently a taxonomy rejection. The sidecar has no usable
evidence/model prior for this release, so the enrichment path needs to ingest or
generate evidence before the graph can classify it. Expected minimum once
evidence exists: `post-rock` and `rock`.

### The Clientele - Strange Geometry

Status: fail.

Failure:
- `ambiguous_release_match`

Matched release keys:
- `the clientele::a sense of falling strange geometry outtakes`
- `the clientele::strange geometry`

Diagnosis: the exact release inspection path exists now, but the fixture still
uses artist/album matching. This fixture should either use
`release_key: the clientele::strange geometry` or the matching logic should gain
an exact-title mode.

## Next Work

1. Add exact `release_key` to ambiguous fixtures where the intended release is
   known.
2. Seed real fixture vocabulary:
   `space rock`, `slacker rock`, `art pop`, `experimental pop`,
   `avant-garde pop`, `krautrock`, `noise pop`, `indietronica`,
   `folk rock`, `psychedelic folk`, `singer-songwriter`.
3. Add facet/descriptor handling for `lo-fi`, years, places, vocalist tags, and
   source-scene trivia.
4. Investigate missing evidence for Explosions In The Sky.
5. Decide whether broad accepted terms like `pop`, `rock`, and `electronic`
   should appear in `accepted_broad_terms`, `inferred_terms`, or both. The current
   report exposes both, but the data model should make that distinction explicit.
