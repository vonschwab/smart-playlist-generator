# Layered Genre Taxonomy Import Hardening

## Taxonomy Source

The reviewed structured taxonomy seed is installed at:

`data/layered_genre_taxonomy.yaml`

Current taxonomy version:

`0.1.1-layered-seed-reviewed`

This pass keeps behavior sidecar-only. It does not mutate `data/metadata.db` and does not change playlist generation behavior.

## Supported Schema Fields

The importer now supports the reviewed `records` schema in addition to the prior split seed shape.

Supported record fields include:

- `name`
- `kind`
- `role`
- `status`
- `facet_type`
- `specificity_score`
- `canonical_target`
- `parent_edges`
- `secondary_roles`
- `reject_reason`
- `alias_policy`
- `source_policy`
- `possible_context_target`
- `notes`

Supported top-level bridge fields include:

- `source`
- `target`
- `edge_type`
- `weight`
- `mode_allowed`
- `required_family_min`
- `required_facet_overlap`
- `required_sonic_similarity`
- `required_transition_quality`
- `required_facets_any`
- `notes`

## Import Mapping

- `kind: family` records become broad/family canonical genres.
- `kind: umbrella | genre | subgenre | microgenre` records become canonical genre records.
- `kind: facet` records become canonical facet records and require `facet_type`.
- `kind: alias` records become alias records and may target either a canonical genre or a canonical facet.
- `kind: reject` or `status: rejected` records become rejected terms and require `reject_reason`.
- `status: review` genre records stay loadable but classify to review, not automatic assignment.
- Explicit `parent_edges` infer parents/families only when the target is a canonical genre.
- `parent_edges` that target facets are validated as existing facet references but are not stored in the existing genre-to-genre edge table yet.

## Validation Added

The loader now rejects:

- missing taxonomy version
- malformed/non-list `records`
- duplicate canonical genre names after normalization
- aliases with missing or unknown `canonical_target`
- parent edges with unknown targets
- bridge rules with unknown source/target genres
- bridge rules with unknown required facets
- facet records missing `facet_type`
- rejected records missing `reject_reason`
- unsupported reject reasons or facet types when enumerated by the YAML
- conditional aliases with no context requirements

## Policy Coverage

The tests now cover reviewed taxonomy behavior for:

- `pop/rock` rejected as `retail_bucket`
- standalone `indie` rejected as `source_noise`
- `dream pop` not inferring `pop` by token containment
- `lo-fi` and `lofi` mapping to production facet behavior
- `instrumental` remaining a function facet, not a leaf genre
- `singer songwriter` aliasing to `singer-songwriter`
- conditional aliases such as `modal` and `twee` not firing without context
- conditional aliases firing with sufficient context
- exact `release_key` graph inspection
- reject reasons appearing in diagnostics

## Fixture Updates

`data/layered_genre_smoke_fixtures.yaml` now uses exact lookup for:

`the clientele::strange geometry`

The Explosions in the Sky fixture is marked with `expected_no_evidence: true` so the report distinguishes the current evidence ingestion gap from taxonomy rejection. If evidence becomes available later, the fixture still expects `post-rock` and `rock`.

The Clientele fixture uses exact `release_key: the clientele::strange geometry`. The copied production sidecar currently has no evidence for that exact key, so it is also marked `expected_no_evidence: true`. This keeps the ambiguity regression fixed while preserving the evidence-ingestion gap as an explicit diagnostic.

Fixture expectations were expanded around the reviewed vocabulary for Duster, Mount Eerie, Ada Lea, and Stereolab.

## Smoke Command

The final disposable-sidecar smoke used a copy of the production sidecar so fixture evaluation had real evidence without writing to production:

```powershell
Copy-Item `
  -LiteralPath C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\ai_genre_enrichment.db `
  -Destination C:\tmp\layered_genre_taxonomy_smoke_sidecar_20260605_copy.db `
  -Force

python scripts/ai_genre_enrich.py `
  --metadata-db C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db `
  --sidecar-db C:\tmp\layered_genre_taxonomy_smoke_sidecar_20260605_copy.db `
  graph-fixture-report `
  --build-assignments
```

Do not run this against the production sidecar.

## Verification Status

Final verification:

- Focused layered genre suite: `26 passed`
- Broader layered unit suite: `50 passed`
- Fixture smoke report: `6 pass / 0 fail`

The smoke report confirmed:

- Duster accepts the expected slowcore/shoegaze/post-rock/dream-pop/slacker-rock/space-rock territory and rejects standalone `indie`.
- Mount Eerie no longer hides psychedelic folk, singer-songwriter, lo-fi, drone, avant-garde, and black metal as unexplained noise; they surface as review/facet/context diagnostics.
- Ada Lea rejects `mixtaperoom`, `rare sad girl`, and `rare sads`, and surfaces `canada` as a region facet.
- Stereolab accepts/sees art pop, indie pop, krautrock, post-rock, broad pop/rock/electronic context, and routes indietronica/noise-pop/avant-garde/experimental territory to review.
- Explosions in the Sky remains a diagnosed no-evidence gap.
- The Clientele exact key no longer fails from ambiguous matching, but remains a diagnosed no-evidence gap in the copied sidecar.

## Deferred

Deferred deliberately until playlist-generation integration:

- changing candidate admission behavior
- changing transition scoring
- building graph vectors into generation artifacts
- learning taxonomy weights from GUI feedback
- persisting facet-target style modifier edges in a dedicated table
