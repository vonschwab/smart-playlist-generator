---
name: genre-data-authority
description: Use when reading, wiring, displaying, or exporting genres anywhere (GUI, API, worker, artifact build, export), when choosing which table or store to query for a track/album/release's genres, or when debugging sparse, missing, or stale genres that enrichment "should have" provided.
---

# Genre data authority

Repeated incidents (three in two days, 2026-06-11/12) came from consumers wired to the wrong genre layer. This skill is the map. Read the One Rule and the layer table before touching any genre read path.

## The One Rule

**Consumers read the published authority: `release_effective_genres` in metadata.db, via `src/genre/authority.py`.** Everything else is either pipeline *input* or pipeline-*internal working data*.

If some other layer looks richer or fresher than the authority, that is a **publish-stage bug** — fix the writer, never rewire the reader to an internal layer. The pull toward "this internal table has more tags, read it instead" is exactly how every past miswiring happened.

There is no "or". Answers like "read the authority *or* query `album_genres` directly" are wrong — raw tables are pipeline **inputs**, acceptable only as the documented last link of the display fallback chain, never as an alternative primary. If the authority is missing genres that raw tables have, the fix is a publish run (or a publish bug report), not a raw read.

## Layer map

| Layer | Where | Role | New consumers may read? |
|---|---|---|---|
| `data/layered_genre_taxonomy.yaml` | repo | Curated SP3a graph (455 genres). Source of truth for taxonomy *structure* | Build/steering code only, via `src/genre/graph_adapter.py` |
| `track_genres` / `album_genres` / `artist_genres` / `track_effective_genres` | metadata.db | Raw tags from files/MusicBrainz/Discogs/Last.fm. **Inputs** to enrichment | ❌ — last-resort display fallback only |
| `ai_genre_source_pages`, `ai_genre_source_tags`, `ai_genre_tag_classifications`, **`enriched_genres`**, hybrid evidence, review queue — i.e. everything read via `SidecarStore` | ai_genre_enrichment.db | Pipeline-internal working data (collection + adjudication). `enriched_genres` looks like "the source of truth for collected tags" — it is not; it's pre-publish working data | ❌ never — not even when they "have more tags" |
| `enriched_genre_signatures` | ai_genre_enrichment.db | **Deprecated** bandcamp-era layer. Never refreshed (`skipped_existing`); often 1–2 stale tags while the authority has 10–34 | ❌ — exists only as a documented mid-chain fallback in old display paths |
| **`release_effective_genres`** (+ `genre_graph_canonical_genres` for id→name) | metadata.db | **THE published output**: graph-resolved + user overrides, layers (`observed_leaf`/`inferred_family`), confidence, source. Written ONLY by the publish stage | ✅ **yes — via `src/genre/authority.py`** |
| `data_matrices_step1.npz` `X_genre_*` | artifacts dir | Generation-time snapshot built FROM the authority (`genre_source: graph` in config.yaml) | Generation runtime only |

## Reading recipes (`src/genre/authority.py`)

- Display names for a track's album: `display_genre_names_for_track(conn, track_id)` — deduped, id→name mapped, `[]` if unpublished. Order for chips with `src/genre/granularity.py::order_genres_for_display`.
- Structured rows (layer/confidence/source): `resolved_genres_for_album(conn, album_id)` / `resolved_genres_for_track(conn, track_id)`.
- Bulk (no N+1): `resolved_genres_by_album(conn)`, `canonical_genre_names(conn)`.
- Artist-level aggregation (tag-steering chips): `resolved_genres_for_artist(conn, artist_name)`
  — observed_leaf+legacy only (inferred hub families excluded by design), exact
  case-insensitive match on `tracks.artist`, ordered by (release_count, max_confidence).

Display fallback chain (legacy tolerance, NOT preference): **authority → signature → raw tags**. Never reorder it; never add an internal layer to it.

## Who writes what

`scan/genres/discogs/lastfm` collect raw tags → `enrich` adjudicates into the sidecar → **`publish` is the ONLY writer of `release_effective_genres`** (timestamped metadata.db backup on first run) → `artifacts` bakes the authority into the artifact (`genre_source: graph`) → generation/display read artifact/authority.

## Trap catalog (real incidents — don't repeat them)

| Incident | Wrong layer read | Fix |
|---|---|---|
| Artifact builder hardcoded `genre_source="legacy"` in `stage_artifacts`; enrichment never reached generation (fixed 2026-06-11, `bfebef0`) | raw tag tables | config-driven `graph` source |
| GUI chips showed 1 genre for Blood Orange (authority had 34) — worker + 2 web endpoints read the signature first (fixed 2026-06-12) | `enriched_genre_signatures` | authority-first via `display_genre_names_for_track` |
| Baseline test agent, asked to fix sparse genres, proposed reading `hybrid_source_terms_for_release` + `fuse_hybrid_evidence` | pipeline-internal evidence | the authority (it never found it) |
| Dormant seam (found 2026-06-12, not live): `similarity_calculator._get_combined_genres` prefers signature genres when an `enriched_resolver` is passed — no production caller passes one. Do NOT wire it up as an "upgrade"; if runtime similarity should see published genres, that's the `genre_graph.source: layered` lane | would be `enriched_genre_signatures` | leave unwired |
| **Enrichment trusted a stranger over the user** (2026-06-12): a Bandcamp *label storefront* page tagged a hardcore record "indie rock/pop" at 0.95 and the fusion **replaced** the user's correct file tags (which went to an unapplied review queue). Same page counted twice (`bandcamp_release` + `ai_enriched_accepted`). | n/a — collection/fusion bug | `hybrid_evidence.py`: bandcamp artist/label/unknown split, `ai_enriched_accepted` corroborating-only, `local_metadata` raised + never-drop, inject `track:file`. See `docs/GENRE_DATA_QUALITY_FINDINGS_2026-06-12.md` |
| **Last.fm name-collision** (2026-06-12): tags fetched by artist-name string → "Green-House" (ambient) got a Ukrainian hip-hop act's `hip hop`/`underground hip-hop`, published on all 6 albums. ~76 artists, mostly generic names. `extract-lastfm` stamps `identity_status="confirmed"` unconditionally. | `lastfm_tags` mis-identified at collection | Fusion now sends Last.fm-only → review; legacy cleaned by the surgical delta (Delta C). Source-side fix (stamp `probable`, verify identity) still open. |
| **Inferred hub-families saturated the genre vector** (2026-06-12): artifact baked `inferred_family`/`parent` into `X_genre_*` → random-pair cosine p50≈0.42, genre signal near-useless. Plus `legacy` layer silently half-weighted (`.get(layer,0.5)`). | over-inclusion in artifact, not a wrong read | `build_beat3tower_artifacts.py`: observed_leaf+legacy full weight only; inferred excluded from vectors (re-enter via similarity matrix); unknown layer raises |
| **Wholesale re-derivation un-decides good past calls** (2026-06-12): re-running the library through corrected fusion stripped correct Last.fm-only tags (Duster "dream pop") and re-materialization caused collateral (Beach Fossils lost 6 correct tags). | process trap, not a layer read | Surgical observed-leaf delta (`assignment_migration.py`): additive A / subtractive B+C, grandfather the rest. Never re-derive to deploy a policy fix. |

## Red flags — stop and re-read the layer map

- Importing `EnrichedGenreResolver`, `SidecarStore`, or `hybrid_evidence` outside `src/ai_genre_enrichment/` or the analyze stages
- A new feature querying `track_genres`/`album_genres`/`artist_genres` directly
- "The signature is stale — regenerate signatures" (nothing user-facing should prefer them)
- "This internal table has the full evidence — read it at display time"
- A `genre_source`-style knob honored in one entry path but hardcoded in another (CLAUDE.md: a configured knob that can't act is a startup error)

## Maintenance protocol

New genre layer, consumer, or authority function → update the layer map and recipes. New miswiring found → add a trap row with date and fix. This skill is the index of genre wiring, not a one-time doc.
