# AI Genre Enrichment: Sources Expansion & Review Workflow

## Goal

Expand the AI genre enrichment pipeline with three new capabilities:

1. **Three-tier classifier vocabulary** — shrink the `review_only` bucket by bootstrapping the deterministic classifier from existing genre vocabularies
2. **Review workflow** — PySide6 panel for human-in-the-loop tag review at library scale, with feedback graduation into the deterministic classifier
3. **Additional enrichment sources** — mine existing metadata.db, integrate Last.fm tags with AI denoising, and wire up AI source discovery for Bandcamp URL resolution

End state: `enriched_genres` becomes a merged authority layer combining deterministic ingestion, Bandcamp extraction, Last.fm evidence, and human review decisions — all with provenance tracking.

## Sub-projects

This spec covers three ordered sub-projects. Each is independently deployable and testable:

- **Sub-project A**: Three-tier classifier vocabulary + metadata.db mining
- **Sub-project B**: Last.fm tag integration with AI denoising
- **Sub-project C**: Review workflow (PySide6 panel + graduation pipeline)

AI source discovery (finding Bandcamp URLs via the existing `source_locator.py` contract) is already spec'd in `docs/AI_GENRE_ENRICHMENT.md` and wired into the CLI. It becomes useful at scale once the classifier vocabulary is solid — no new spec needed, just operational use of existing commands.

---

## Sub-project A: Three-Tier Classifier Vocabulary + Metadata Mining

### Problem

`tag_classification.py` has a hand-curated ~88-term `_GENRE_STYLE_TAGS` allowlist. Meanwhile:

- `normalize_unified.py` has 150+ canonical genre tokens (SYNONYM_MAP values, PHRASE_MAP outputs) that the playlist engine already recognizes
- `metadata.db` contains every genre ever ingested from MusicBrainz, Discogs, and file tags across 3000+ artists

These vocabularies don't talk to each other. A Bandcamp tag like `psychedelic rock` is legitimate but lands in `review_only` because it's not in the hand-curated set.

### Design

#### Three-tier genre vocabulary

Replace the flat `_GENRE_STYLE_TAGS` set with a `GenreVocabulary` class that checks three tiers in order:

| Tier | Source | Confidence | Mutability |
|------|--------|-----------|------------|
| 1 — Curated | Static allowlist (current `_GENRE_STYLE_TAGS` + stress-test-graduated terms) | 0.95 | Manual edits + review graduation |
| 2 — Engine | Canonical tokens from `normalize_genre_token` (SYNONYM_MAP targets, PHRASE_MAP outputs) | 0.85 | Auto-derived at import time |
| 3 — Library | Distinct normalized genres from `metadata.db` genre tables | 0.80 | Auto-derived at enrichment time |

Non-genre category allowlists (instruments, places, descriptors, formats, mood/function, labels) stay hand-curated — those categories are small, stable, and benefit from precision.

#### Vocabulary file format

Tier 1 moves from inline Python sets to a loadable YAML file at `data/genre_vocabulary.yaml`:

```yaml
version: 1
genre_style:
  - ambient
  - ambient jazz
  - art pop
  - balearic
  # ... full curated list
descriptor:
  - acoustic
  - beats
  - cosmic
  # ...
instrument:
  - drums
  - guitar
  - saxophone
  # ...
place:
  - atlanta
  - brooklyn
  # ...
format:
  - demo
  - live
  # ...
mood_function:
  - meditation
  - chillout
  # ...
label_or_org:
  - american football
  - soundway records
  # ...
```

`tag_classification.py` loads this file at import time. The review workflow (Sub-project C) writes graduated terms back to this file.

#### Tier 2: engine vocabulary bootstrap

At import time, `GenreVocabulary` collects all canonical output tokens from `normalize_unified.py`:

```python
def _collect_engine_genres() -> set[str]:
    from src.genre.normalize_unified import SYNONYM_MAP, PHRASE_MAP
    genres = set(SYNONYM_MAP.values())
    for outputs in PHRASE_MAP.values():
        genres.update(outputs)
    genres.discard("")
    return genres
```

These are genres the playlist engine already recognizes — if a Bandcamp tag normalizes to one, it's safe to classify as `genre_style`.

#### Tier 3: library vocabulary

At enrichment time (not import time, since it requires a DB connection), query `metadata.db` read-only:

```sql
SELECT DISTINCT genre FROM (
    SELECT genre FROM artist_genres
    UNION
    SELECT genre FROM album_genres
    UNION
    SELECT genre FROM track_genres
)
```

Each result is normalized via `normalize_genre_token` before comparison. Tags matching this set get `genre_style` at confidence 0.80.

#### Metadata.db mining into enriched_genres

A new CLI command `ingest-local` reads existing genre tables for a release and feeds them through the source_pages → source_tags → classification → enriched_genres pipeline, just like Bandcamp tags. This keeps provenance tracking consistent:

```
python scripts/ai_genre_enrich.py ingest-local --limit 100
python scripts/ai_genre_enrich.py ingest-local --artist "Slowdive" --album "Souvlaki"
```

Pipeline:
1. Creates an `ai_genre_source_pages` entry with `source_type: "local_metadata"`, `source_url: "local://metadata.db"`, `identity_status: "confirmed"`
2. Inserts each genre as an `ai_genre_source_tags` row (raw_tag = original genre string from the table)
3. Runs `classify_source_tags()` — the three-tier vocabulary classifies them
4. `rebuild_enriched_genres_for_release()` picks up the accepted `genre_style` tags

Rules:
- Only ingest genres that pass the three-tier vocabulary check as `genre_style`
- Skip tags in `GENERIC_OR_DESCRIPTOR_TAGS` or `DESCRIPTOR_ONLY_TAGS` unless the release has fewer than 3 specific tags (keep broad tags as fallback)
- Skip `META_TAGS` and `DROP_TOKENS` from `normalize_unified.py`
- `confidence` = 0.90 for MusicBrainz/Discogs sources, 0.75 for file-tag sources
- Never overwrites existing Bandcamp-sourced enriched_genres rows (Bandcamp is higher authority for release-specific tags)
- The `source_type: "local_metadata"` must be added to the `AUTHORITATIVE_SOURCE_TYPES` list in `models.py` so `rebuild_enriched_genres_for_release()` accepts these rows

#### Changes to existing files

| File | Change |
|------|--------|
| `src/ai_genre_enrichment/tag_classification.py` | Replace inline sets with `GenreVocabulary` class loading from YAML + engine + library tiers |
| `scripts/ai_genre_enrich.py` | Add `ingest-local` command |
| `src/ai_genre_enrichment/storage.py` | Add `ingest_local_genres_for_release()` method |
| `data/genre_vocabulary.yaml` | New file: externalized vocabulary |
| `tests/unit/test_ai_genre_enrichment.py` | Tests for three-tier lookup, YAML loading, ingest-local |

---

## Sub-project B: Last.fm Tag Integration

### Problem

Last.fm has the richest crowdsourced genre vocabulary available, but the tags are noisy: meta-tags ("seen live", "favorites"), decade tags, mood descriptors, and user-specific junk mixed in with legitimate genres. The existing library analysis pass may already have Last.fm data stored, but it's not surfaced in the enrichment pipeline.

### Design

#### Source type

Last.fm tags enter the pipeline as a new `source_type: "lastfm_tags"` in `ai_genre_source_pages`, alongside the existing `"bandcamp_release"` type. They get lower base authority than Bandcamp (crowdsourced vs artist/label-supplied).

#### Extraction

A new function `extract_lastfm_tags_from_metadata(release_key, metadata_db)` reads Last.fm-sourced genres from the existing genre tables in `metadata.db`:

```sql
SELECT DISTINCT genre
FROM artist_genres
WHERE artist = ? AND source LIKE '%lastfm%'
UNION
SELECT DISTINCT genre
FROM album_genres
WHERE album_id = ? AND source LIKE '%lastfm%'
UNION
SELECT DISTINCT genre
FROM track_genres
WHERE track_id IN (SELECT track_id FROM tracks WHERE album_id = ?)
  AND source LIKE '%lastfm%'
```

If Last.fm tags aren't already in the genre tables (depends on the library analysis pass), a future step could add a `fetch_lastfm_tags(artist, album)` using the Last.fm API. But mining what's already stored is the zero-cost starting point.

#### Denoising pipeline

Last.fm tags go through the same `classify_source_tag()` pipeline as Bandcamp tags, but with these adjustments:

1. **Pre-filter**: `META_TAGS` and `DROP_TOKENS` from `normalize_unified.py` are rejected before classification (these are Last.fm-specific noise that Bandcamp pages don't have)
2. **Deterministic classifier**: same three-tier vocabulary lookup, but confidence is reduced by 0.10 for Last.fm sources (e.g., Tier 1 match → 0.85 instead of 0.95)
3. **Agreement boost**: if a Last.fm tag matches a Bandcamp tag for the same release, confidence is boosted to the Bandcamp level. Cross-source agreement is strong signal.
4. **AI adjudication**: Last.fm `review_only` tags that don't match any Bandcamp tag are candidates for AI adjudication (same contract as `tag_adjudicator.py`), but with a flag indicating Last.fm provenance so the model can weight accordingly

#### Enriched genres integration

Last.fm-sourced genres that pass classification get `basis: "lastfm_tags"` in `enriched_genres`. When building the merged authority signature:

- Bandcamp-sourced genres take precedence (higher authority)
- Last.fm genres that agree with Bandcamp get `basis: "hybrid"` and boosted confidence
- Last.fm-only genres are accepted but at lower confidence, making them review-eligible if the confidence falls below the auto-apply threshold

#### CLI integration

```
python scripts/ai_genre_enrich.py extract-lastfm --artist "Slowdive" --album "Souvlaki"
python scripts/ai_genre_enrich.py extract-lastfm --limit 100
```

The existing `classify-tags`, `build-enriched`, and `show-enriched` commands work unchanged — they operate on whatever source pages exist for a release.

#### Changes to existing files

| File | Change |
|------|--------|
| `scripts/ai_genre_enrich.py` | Add `extract-lastfm` command |
| `src/ai_genre_enrichment/source_extraction.py` | Add `extract_lastfm_tags_from_metadata()` |
| `src/ai_genre_enrichment/tag_classification.py` | Add Last.fm confidence reduction and agreement boost logic |
| `src/ai_genre_enrichment/storage.py` | Support `source_type: "lastfm_tags"`, agreement-boost logic in `rebuild_enriched_genres_for_release()` |
| `tests/unit/test_ai_genre_enrichment.py` | Tests for Last.fm extraction, denoising, agreement boost |

---

## Sub-project C: Review Workflow

### Problem

At library scale (3000+ artists, thousands of albums), human review is necessary for:

1. Tags in `review_only` after both deterministic and AI classification
2. Releases with empty or suspiciously thin enriched signatures
3. AI-classified tags at moderate confidence (0.6–0.8) that need human confirmation
4. Tags the AI adjudicator promoted but the user may disagree with

### Design

#### Review decisions table

New sidecar table `ai_genre_review_decisions`:

```sql
CREATE TABLE ai_genre_review_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_tag_id INTEGER,
    release_key TEXT NOT NULL,
    raw_tag TEXT NOT NULL,
    normalized_tag TEXT NOT NULL,
    original_classification TEXT NOT NULL,
    reviewed_classification TEXT NOT NULL,
    reviewer TEXT NOT NULL DEFAULT 'human',
    decided_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (source_tag_id) REFERENCES ai_genre_source_tags(source_tag_id),
    UNIQUE (source_tag_id, reviewer)
);
```

#### Review queue logic

The review queue is populated by querying the sidecar DB for:

```sql
-- Tags needing review: review_only or low-confidence classifications without a human decision
SELECT t.source_tag_id, p.release_key, p.normalized_artist, p.normalized_album,
       t.raw_tag, t.normalized_tag, c.classification, c.confidence, p.source_url
FROM ai_genre_source_tags t
JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
LEFT JOIN ai_genre_review_decisions d ON d.source_tag_id = t.source_tag_id
WHERE d.decision_id IS NULL
  AND (c.classification = 'review_only' OR c.confidence < 0.80)
ORDER BY c.confidence ASC, p.release_key, t.tag_position
```

Filters available: by release, by classification, by confidence band, by source type, by "new tags only" (tags whose normalized form has never appeared in any prior review decision).

#### PySide6 review panel

A new `ReviewPanel` widget integrated into the existing GUI (accessible from the menu bar or a toolbar button). Layout:

```
+------------------------------------------------------------------+
| Review Queue                                    [Filter ▼] [Stats]|
+------------------------------------------------------------------+
| Release: Slowdive — Souvlaki                                     |
| Source:  https://slowdive.bandcamp.com/album/souvlaki             |
| Current: review_only (0.50)                                      |
|                                                                   |
| Tag: "noise pop"                                                  |
|                                                                   |
| Context: Other tags on this release:                              |
|   shoegaze (genre_style, 0.95)                                   |
|   dream pop (genre_style, 0.95)                                  |
|   indie (genre_style, 0.95)                                      |
|   uk (place, 0.95)                                               |
|                                                                   |
| ─────────────────────────────────────────────────────────────────  |
|  [A] Accept genre   [D] Descriptor   [I] Instrument              |
|  [P] Place          [S] Skip         [R] Reject                  |
+------------------------------------------------------------------+
| Progress: 12 / 47 reviewed | 8 accepted | 2 descriptors | 2 skip |
+------------------------------------------------------------------+
```

Keyboard shortcuts:
- `A` — accept as `genre_style`, write decision, advance
- `D` — classify as `descriptor`, write decision, advance
- `I` — classify as `instrument`, write decision, advance
- `P` — classify as `place`, write decision, advance
- `S` — skip (defer), advance without writing
- `R` — reject (not a genre, not useful), write decision with `reviewed_classification: "rejected"`, advance
- `Ctrl+Z` — undo last decision

Each keystroke writes to `ai_genre_review_decisions` and triggers `rebuild_enriched_genres_for_release()` for the affected release.

#### Graduation pipeline

After a review session, a CLI command graduates human-accepted genre_style tags into the Tier 1 vocabulary:

```
python scripts/ai_genre_enrich.py graduate-reviewed
```

This command:
1. Queries all review decisions where `reviewed_classification = 'genre_style'`
2. Collects the distinct `normalized_tag` values
3. Adds them to `data/genre_vocabulary.yaml` under `genre_style:` (sorted, deduplicated)
4. Reports what was added

After graduation, those tags are recognized deterministically at 0.95 confidence — they never hit `review_only` again.

Similarly, tags reviewed as `descriptor`, `instrument`, `place`, etc. are added to their respective categories in the vocabulary file.

#### CLI review fallback

For terminal-only use, a simple CLI review mode:

```
python scripts/ai_genre_enrich.py review --limit 20
```

Presents tags one at a time with the same context, accepts single-character input. Same decision storage and graduation pipeline.

#### Changes to existing files

| File | Change |
|------|--------|
| `src/ai_genre_enrichment/storage.py` | Add `ai_genre_review_decisions` table, review queue query, decision recording |
| `src/playlist_gui/widgets/review_panel.py` | New file: PySide6 review panel widget |
| `src/playlist_gui/main_window.py` | Add review panel access (menu item or toolbar button) |
| `scripts/ai_genre_enrich.py` | Add `review` and `graduate-reviewed` commands |
| `data/genre_vocabulary.yaml` | Written to by graduation pipeline |
| `tests/unit/test_ai_genre_enrichment.py` | Tests for review queue, decision recording, graduation |

---

## Ordering and Dependencies

```
Sub-project A (vocabulary + metadata mining)
    ↓
Sub-project B (Last.fm integration)
    ↓
Sub-project C (review workflow)
```

A must come first because it establishes the `GenreVocabulary` class and `genre_vocabulary.yaml` file that B and C depend on. B should come before C because the review queue is most useful when it contains tags from multiple sources (Bandcamp + Last.fm). C is last because it's a consumer of everything the other sub-projects produce.

Each sub-project is independently testable and deployable — B and C add value even without the others, but the ordering maximizes the impact of each step.

## Out of Scope

- AI tag adjudication prompt engineering (brainstormed separately in ChatGPT session)
- Auto-apply pipeline (enriched_genres remains recommendation-only in this phase)
- Playlist generation reading enriched_genres (separate roadmap item)
- Genre artifact rebuild from enriched_genres
- Bandcamp crawling or scraping (policy: no crawling)
- MusicBrainz/Discogs re-querying (already handled by deterministic ingestion)
