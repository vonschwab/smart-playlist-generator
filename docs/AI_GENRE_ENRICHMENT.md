# AI Genre Enrichment

## Purpose

The AI genre enrichment backend recommends album-level genre metadata for canonical artist+album releases. It is an offline refinement tool: playlist generation remains local-first and never depends on OpenAI availability.

The current implementation is recommendation-only. It does not apply AI output to genre tables or change playlist generation behavior.

This layer exists to fix the cases that deterministic genre ingestion cannot reliably solve: releases with generic, missing, descriptor-only, contradictory, or under-specific metadata. It should improve specificity while preserving the multi-genre signatures that make playlist generation work.

## Safety Model

- `data/metadata.db` is opened read-only during discovery.
- The tool never modifies `artist_genres`, `album_genres`, `track_genres`, or any existing metadata table.
- Music files are read-only and are not touched.
- Results are stored in the sidecar database `data/ai_genre_enrichment.db`.
- API keys are read from `OPENAI_API_KEY`; no secrets are committed to config.
- Use `--dry-run` to inspect routing, payloads, and prompt-size estimates without calling OpenAI.

## Relationship To Existing Genre Ingestion

MusicBrainz and Discogs are already handled by the deterministic library analysis pass. AI Genre Enrichment should not spend web-search budget re-querying those same sources.

The AI refinement layer treats existing MusicBrainz/Discogs/local tags as the baseline payload to audit, not as web sources to rediscover. Its job is to answer questions like:

- Are the existing tags too generic to be useful?
- Are descriptor tags polluting the genre vector?
- Are obvious official-source genres missing?
- Should a broad but valid parent genre be kept while more specific tags are added?
- Is the release too obscure or ambiguous for safe automation?

## Two-Lane Architecture

Each release is classified before any API call:

- `skip_well_tagged`: the album already has a compact, specific multi-genre signature and no obvious descriptor pollution.
- `no_web_adjudication`: existing local metadata is enough for cheap structured adjudication.
- `authoritative_source_enrichment`: metadata is missing, generic, contradictory, descriptor-only, or likely under-tagged, and official/release-specific source evidence may materially improve the answer.
- `needs_review`: evidence is too weak, identity is ambiguous, authoritative sources cannot be found, or web budget is exhausted.

Lane A (`web_mode=off`) is batch-friendly and uses the local payload only: file metadata, existing normalized genre tables, existing MusicBrainz/Discogs results, and any previously stored source evidence.

Lane B (`web_mode=auto|required`) is synchronous/queued. It may use the Responses API web search tool when available in the installed OpenAI SDK. Batch output is intentionally kept no-web until web-search Batch compatibility is verified.

Lane B is not a general web research pass. It is an authoritative-source lookup pass.

## Deterministic-First Workflow

The refined workflow is designed to minimize token use and keep durable decisions in Python:

1. Discover canonical artist+album releases from `metadata.db` in read-only mode.
2. Use AI source discovery only when a release needs authoritative source evidence and no confirmed URL is already supplied.
3. Extract source tags deterministically from confirmed release URLs, starting with deterministic Bandcamp release tag extraction.
4. Classify obvious source tags deterministically first: genre/style, descriptor, instrument, place, format, or mood/function.
5. Use AI tag adjudication only for ambiguous source tags that deterministic rules cannot classify safely.
6. Build the sidecar `enriched_genres` authority layer from accepted genre/style tags and source provenance.

This workflow does not modify artist_genres, album_genres, or track_genres; `enriched_genres` is the nondestructive genre authority layer used for future similarity experiments.

## `enriched_genres` Authority Layer

`enriched_genres` is the future source of truth for genre similarity experiments, but it remains sidecar-only in this phase. It stores accepted genre/style rows with provenance, confidence, basis, and source references. The original deterministic genre sources remain untouched so every decision can be re-evaluated later.

`enriched_genre_signatures` stores a compact sorted JSON signature per release for inspection and later read paths. Playlist generation does not read this table yet. The builder does not write empty enriched genre signatures: if a Bandcamp page only yields places, instruments, descriptors, formats, or review-only tags, the release has no enriched signature and future opt-in read paths must fall back to the legacy local/MusicBrainz/Discogs genre sources.

## Deterministic Bandcamp Extraction

Bandcamp release pages often expose valuable artist/label-supplied tags that broad metadata providers miss. After a Bandcamp release URL is supplied or confidently confirmed, the tool can parse the visible release tags deterministically and store only the resulting tags and source URL. It does not crawl Bandcamp, follow pagination, search artist pages, or store full page text.

Source-backed specificity is preserved. Tags such as `ambient jazz`, `electroacoustic`, `electronica`, and `fourth world` should remain distinct genre/style candidates when extracted from an authoritative release page.

## AI Source Discovery

AI source discovery is a narrow web-search step for finding candidate authoritative release URLs. It should return candidate source pages only, not genre decisions. It must not spend tokens rediscovering MusicBrainz, Discogs, Last.fm, streaming services, review sites, or generic search-result metadata.

Useful source discovery output is a short list of release-specific official artist, label, publisher, release, Bandcamp, label catalog, release note, press release, or liner-note pages with identity confidence and warnings.

## AI Tag Adjudication

AI tag adjudication is a cheap no-web classification step for ambiguous extracted tags. Its input is the local payload plus extracted source tags, not raw web pages. It decides whether each tag is a usable genre/style or a descriptor, instrument, place, format, mood/function tag, or review-only case.

Niche subgenres are not automatically review-only; source-backed specificity is the main value of this enrichment layer.

## Source Policy

Use local metadata first. When web/source evidence is needed, use only the most authoritative release-specific sources.

### Primary Authoritative Sources

Prefer sources in this order:

1. Official release page from the artist, label, or publisher.
2. Bandcamp release page, especially artist- or label-run Bandcamp pages.
3. Official artist website release/discography page.
4. Official label catalog page, release notes, press release, or shop page.
5. Official distributor page only when it clearly reproduces label/artist release metadata.
6. Liner notes, artist-provided release notes, or label-provided one-sheet text when included in the payload.

These sources can support high-confidence additions when they provide clear release-specific style/genre language or artist/label tags.

### Baseline Sources, Not Web-Enrichment Targets

The following are already part of the deterministic metadata ecosystem or are too redundant for this web lane. Do not spend web-search effort on them during authoritative-source enrichment:

- MusicBrainz
- Discogs
- Last.fm

Existing MusicBrainz/Discogs tags may still appear in the local payload and should be evaluated as baseline metadata.

### Excluded By Default

Do not use these as primary evidence for AI genre refinement:

- Last.fm tags or other open user-tag clouds
- Spotify/Apple/streaming mood tags
- Wikipedia/Wikidata genre summaries
- generic SEO music pages
- lyrics pages
- scraped mirror sites
- broad artist biographies that are not release-specific
- review aggregators
- recommendation sites with unsourced genre labels

These sources may be useful for human context, but they are too noisy for automated genre mutation and should not make a suggestion auto-apply eligible.

### Review-Only Secondary Context

Specialist reviews, record shops, blogs, and archival pages can be useful for human review, especially for rare/private-press/reissue material, but they are not primary authoritative sources for this tool.

Examples:

- Boomkat
- Forced Exposure
- All Night Flight
- Light in the Attic
- Aquarium Drunkard
- The Quietus
- Pitchfork
- Post-Trash
- Bandcamp Daily
- Resident Advisor

When used at all, treat them as review-only context unless the page clearly quotes or reproduces label/artist release language.

### Bandcamp Policy

The tool does not implement a deterministic Bandcamp crawler or scraper. Web-grounded enrichment may use search result citations or the Responses API web search tool, but the app should not crawl Bandcamp pages itself.

Store source URLs, short evidence summaries, and derived genre recommendations. Do not store full page text.

## Source-Grounded Prompt Requirements

The model prompt should make this source policy explicit:

- Do not perform broad web research.
- Do not use web search to rediscover MusicBrainz or Discogs data already present in the payload.
- Prefer official, release-specific artist/label/Bandcamp evidence.
- If no authoritative source is found, lower confidence and set `should_escalate=true`.
- Never claim a source says something unless that source was supplied or returned in the request.
- Treat artist/label/Bandcamp tags as strong evidence, but still classify joke, SEO, format, country, era, or descriptor tags correctly.
- Keep valid broad parent genres and add specific tags alongside them when supported.
- Do not prune broad valid genres merely because better specific tags can be added.

## Recommendation Basis

Every recommendation should declare its basis:

- `local_metadata`: supported by existing file/local/MusicBrainz/Discogs baseline payload.
- `authoritative_source`: supported by official artist, label, publisher, release, or Bandcamp evidence.
- `hybrid`: supported by both local metadata and authoritative source evidence.
- `model_knowledge`: based on general music knowledge only; never auto-apply eligible.
- `review_context`: supported only by secondary reviews/shops/blogs; review-only.

Only `authoritative_source` and `hybrid` additions should be eligible for future auto-apply consideration.

## Routing Guidance

Route to `skip_well_tagged` when:

- the release already has 3-8 specific usable genre/style tags,
- broad parent tags are supplemented by specific tags,
- no obvious descriptors are polluting the genre list,
- and the local metadata is internally consistent.

Route to `no_web_adjudication` when:

- existing metadata is mostly useful but needs normalization,
- descriptor tags need to be separated from genre tags,
- broad tags should be kept but not expanded without stronger evidence,
- or a cheap structured check can produce review recommendations without web evidence.

Route to `authoritative_source_enrichment` when:

- existing metadata is empty, generic, contradictory, or descriptor-only,
- the release is likely under-tagged because it is niche, local, Bandcamp-era, tape-label, private-press, experimental, ambient/new age, indie/punk, or otherwise poorly represented in MusicBrainz/Discogs,
- the album has only broad tags such as `rock`, `pop`, `indie`, `indie rock`, `alternative`, `electronic`, `experimental`, `folk`, `jazz`, or `hip hop`,
- the album has only descriptors such as `instrumental`, `live`, `soundtrack`, `Japanese`, `remastered`, `compilation`, or `demo`,
- or official/Bandcamp/label evidence is likely to materially improve genre specificity.

Route to `needs_review` when:

- no authoritative release-specific source can be found,
- the release identity remains ambiguous,
- the only available evidence is secondary, noisy, or user-tag-based,
- suggested genres are mostly microgenres or scene tags,
- or any prune would remove a major existing genre.

## Sidecar Database

`ai_genre_release_checks` stores one check per release/input/prompt/taxonomy/model/web/schema identity. The cache identity includes `release_key`, `input_hash`, `prompt_version`, `taxonomy_version`, `model`, `web_mode`, source evidence hash, response schema version, and authoritative source URL hash when available.

Key fields include normalized artist/album names, optional `album_id`, optional MBID/Discogs identifiers from the baseline payload, status, response JSON, confidence, evidence quality, token usage, estimated cost, web mode, source evidence hash, authoritative source URL hash, and schema version.

`ai_genre_suggestions` normalizes keep/prune/add/descriptor rows for reporting, including recommendation basis and supporting source indexes.

`ai_genre_run_log` records run-level counts, including cache hits, skipped well-tagged releases, no-web checks, authoritative-source checks, failures, and review cases.

## Confidence And Auto-Apply

The MVP still does not apply anything automatically. The `auto_apply_eligible` flag is only a recommendation for future review tooling.

`auto_apply_eligible=true` is allowed only for new genre/style additions and requires:

- release confidence `>= 0.85`
- `evidence_quality == high`
- individual genre confidence `>= 0.85`
- direct support from an authoritative release-specific source or strong hybrid agreement between local metadata and authoritative source evidence
- a real genre/style, not a descriptor

Never auto-apply broad parent genres, descriptors, disputed microgenres, country/language/era tags, mood/function tags, joke tags, SEO tags, or recommendations based only on model knowledge.

Prunes remain review-only unless a tag is malformed, duplicated, or clearly descriptor-only.

## Commands

```bash
python scripts/ai_genre_enrich.py discover --limit 50 --dry-run
python scripts/ai_genre_enrich.py run --limit 100 --web-mode off
python scripts/ai_genre_enrich.py run --generic-only --limit 50 --web-mode auto --max-web-enrichment 10
python scripts/ai_genre_enrich.py run-one --artist "Takashi Kokubo" --album "太陽風～オーロラの神秘～" --web-mode required --dry-run
python scripts/ai_genre_enrich.py report
```

Install the optional SDK before real API calls:

```bash
pip install -e .[ai]
```

## Inspecting Recommendations

Use the report command for summary counts, lane counts, source domains, top proposed additions/prunes, token usage, and low-confidence examples:

```bash
python scripts/ai_genre_enrich.py report
```

For direct inspection, open `data/ai_genre_enrichment.db` with SQLite and query `ai_genre_release_checks` or `ai_genre_suggestions`.

## Batch-Ready Design

`prepare-batch` writes no-web JSONL request objects for the Responses API to `data/ai_genre_batches/batch_<timestamp>.jsonl`. `collect-batch` is intentionally stubbed for a later pass that will import Batch API results into the sidecar database.

Authoritative-source enrichment is kept separate from batch by default because it may require web search, stricter budgeting, and source URL caching.

## Album Model Prior

`model-prior-one`, `model-prior`, and `model-prior-report` are CLI-only commands that generate
provisional album-level genre hypotheses without web access.

- Provider: OpenAI (default model: `gpt-4o-mini`)
- Web access: off — only local release metadata is supplied to the model
- Prior terms are provisional classifier signals: never authoritative, never auto-apply eligible,
  and never inserted into normal `enriched_genre_signatures`
- All terms persist with `auto_apply_eligible=0` and a dedicated identity key
  (release, input hash, provider, model, prompt version, taxonomy version, schema version, policy)

```bash
# Preview the payload that would be sent (no API call, no sidecar write)
python scripts/ai_genre_enrich.py model-prior-one --artist "Duster" --album "Stratosphere" --dry-run

# Generate a prior for one album (writes to dedicated sidecar tables)
python scripts/ai_genre_enrich.py model-prior-one --artist "Duster" --album "Stratosphere"

# Bounded batch run (skip albums that already have a cached complete prior)
python scripts/ai_genre_enrich.py model-prior --limit 20 --missing-only

# Report prior and mapping counts
python scripts/ai_genre_enrich.py model-prior-report
```

Prior terms are stored in `ai_genre_model_priors` and `ai_genre_model_prior_terms` in the sidecar
database. These tables are isolated from normal enrichment tables and do not affect artifact builds.

## Non-Goals

This refinement does not apply AI suggestions to existing genre tables, rebuild genre artifacts, scrape Bandcamp directly, duplicate the existing MusicBrainz/Discogs genre pass, use Last.fm/user-tag clouds as authoritative evidence, or change playlist generation behavior.
