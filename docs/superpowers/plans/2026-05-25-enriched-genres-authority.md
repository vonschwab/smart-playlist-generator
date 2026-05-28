# Enriched Genres Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic-first AI genre refinement workflow that extracts authoritative source tags, classifies them cheaply, and stores nondestructive `enriched_genres` as the future genre source of truth.

**Architecture:** Keep `metadata.db` and legacy genre tables untouched. Store all source evidence, extracted tags, tag classifications, and accepted enriched genre rows in the sidecar database. Use AI only for source discovery and ambiguous tag classification; Python owns extraction, normalization, caching, validation, reporting, and persistence.

**Tech Stack:** Python 3.11+, SQLite sidecar DB, pytest, ruff, OpenAI Responses API for source-location/tag-adjudication only, existing `src.ai_genre_enrichment` and `src.genre.normalize_unified` modules.

---

## Lessons From May 25 Testing

These are requirements, not preferences:

- Niche/specific source tags are the purpose of this project. Tags such as `fourth world`, `ambient jazz`, `electroacoustic`, and `electronica` must be preserved when source-backed; rarity is not a reason to demote a tag.
- Do not collapse source tags through legacy synonym mappings when building `enriched_genres`. In particular, `electronica` must not become only `electronic`.
- Existing local genres remain `local_metadata` unless the authoritative source exactly supports the same tag. Do not relabel local keeps as `model_knowledge`.
- A recommendation may cite a source only when that source actually contains the exact normalized tag. `ambient jazz` does not support generic `jazz`; `electronica` does not support generic `electronic`.
- Source-only tags are not existing local genres. They must not be written as `existing_genres_to_prune`; if they are real genre/style tags, they become add candidates.
- Broad parent genres may be kept, but they must not erase specific tags.
- Streaming/storefront/review/context pages such as Audiomack, Qobuz, Spotify, Tidal, Deezer, SoundCloud, AllMusic, Pitchfork, Discogs, MusicBrainz, Wikipedia, and Last.fm are not authoritative source evidence.
- Bandcamp extraction is deterministic only after a URL has been explicitly supplied or confidently confirmed. No crawling, no pagination, no broad scraping.
- The new source of truth is `enriched_genres`, but it is nondestructive sidecar data. It does not mutate `artist_genres`, `album_genres`, or `track_genres`.

---

## File Structure

- Create `src/ai_genre_enrichment/source_extraction.py`
  - Deterministic extraction from confirmed authoritative URLs.
  - First implementation supports Bandcamp release pages from supplied/confirmed URLs only.
  - No crawling, no URL discovery, no pagination.
  - Stores raw extracted tags and normalized tags, but not full page text.

- Create `src/ai_genre_enrichment/tag_classification.py`
  - Deterministic first-pass tag classifier.
  - Preserves specific tags such as `fourth world`, `ambient jazz`, `electroacoustic`, and `electronica`.
  - Separates obvious descriptors/instruments/places/formats from genre/style tags.
  - Does not call legacy `normalize_genre_token()` for source-tag canonicalization unless doing so cannot collapse specificity.

- Create `src/ai_genre_enrichment/source_locator.py`
  - Structured schema/request builder for AI source discovery.
  - Returns only candidate official/label/Bandcamp URLs and confidence, not genre decisions.

- Create `src/ai_genre_enrichment/tag_adjudicator.py`
  - Structured schema/request builder for cheap no-web ambiguous tag adjudication.
  - Input is local payload plus extracted tags, not raw web pages.

- Modify `src/ai_genre_enrichment/storage.py`
  - Add sidecar tables:
    - `ai_genre_source_pages`
    - `ai_genre_source_tags`
    - `ai_genre_tag_classifications`
    - `enriched_genres`
    - `enriched_genre_signatures`
  - Preserve existing check/suggestion tables.

- Modify `scripts/ai_genre_enrich.py`
  - Add subcommands:
    - `find-sources`
    - `extract-tags`
    - `classify-tags`
    - `build-enriched`
    - `show-enriched`
  - Keep existing `run`, `run-one`, `report`, and `prepare-batch` working.

- Modify `docs/AI_GENRE_ENRICHMENT.md`
  - Document deterministic-first workflow and `enriched_genres`.

- Modify `tests/unit/test_ai_genre_enrichment.py`
  - Add unit tests for all new deterministic pieces.
  - Continue asserting no writes to `metadata.db`.

---

### Task 1: Add Sidecar Authority Tables

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing schema test**

Add:

```python
def test_sidecar_initializes_enriched_genre_authority_tables(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    assert "ai_genre_source_pages" in tables
    assert "ai_genre_source_tags" in tables
    assert "ai_genre_tag_classifications" in tables
    assert "enriched_genres" in tables
    assert "enriched_genre_signatures" in tables
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_sidecar_initializes_enriched_genre_authority_tables -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `FAILED` because the new tables do not exist.

- [ ] **Step 3: Add the schema**

In `SidecarStore.initialize()`, add:

```sql
CREATE TABLE IF NOT EXISTS ai_genre_source_pages (
    source_page_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_key TEXT NOT NULL,
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    source_url TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    identity_status TEXT NOT NULL,
    identity_confidence REAL NOT NULL DEFAULT 0,
    fetched_at TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_hash TEXT NOT NULL DEFAULT 'none',
    evidence_summary TEXT,
    UNIQUE (release_key, source_url)
);

CREATE TABLE IF NOT EXISTS ai_genre_source_tags (
    source_tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_page_id INTEGER NOT NULL,
    raw_tag TEXT NOT NULL,
    normalized_tag TEXT NOT NULL,
    tag_position INTEGER,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY (source_page_id) REFERENCES ai_genre_source_pages(source_page_id) ON DELETE CASCADE,
    UNIQUE (source_page_id, normalized_tag)
);

CREATE TABLE IF NOT EXISTS ai_genre_tag_classifications (
    classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_tag_id INTEGER NOT NULL,
    classification TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    classifier TEXT NOT NULL,
    reason TEXT NOT NULL,
    classified_at TEXT NOT NULL,
    FOREIGN KEY (source_tag_id) REFERENCES ai_genre_source_tags(source_tag_id) ON DELETE CASCADE,
    UNIQUE (source_tag_id, classifier)
);

CREATE TABLE IF NOT EXISTS enriched_genres (
    enriched_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_key TEXT NOT NULL,
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    genre TEXT NOT NULL,
    basis TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    source_tag_id INTEGER,
    source_page_id INTEGER,
    source_ref TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'accepted',
    added_at TEXT NOT NULL,
    FOREIGN KEY (source_tag_id) REFERENCES ai_genre_source_tags(source_tag_id),
    FOREIGN KEY (source_page_id) REFERENCES ai_genre_source_pages(source_page_id),
    UNIQUE (release_key, genre, basis, source_ref)
);

CREATE TABLE IF NOT EXISTS enriched_genre_signatures (
    release_key TEXT PRIMARY KEY,
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    signature_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Use `source_ref` values such as `source_tag:123` or `local_payload:album:discogs_release` so the uniqueness rule remains valid for rows with and without `source_tag_id`.

- [ ] **Step 4: Run the schema test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_sidecar_initializes_enriched_genre_authority_tables -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

- [ ] **Step 5: Run existing AI genre tests**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: all tests pass.

---

### Task 2: Deterministic Bandcamp Tag Extraction

**Files:**
- Create: `src/ai_genre_enrichment/source_extraction.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing parser test**

Add:

```python
def test_extract_bandcamp_tags_from_release_html():
    from src.ai_genre_enrichment.source_extraction import extract_bandcamp_release_tags

    html = """
    <div class="tralbumData tralbum-tags tralbum-tags-nu">
      <a class="tag" href="https://bandcamp.com/discover/ambient">ambient</a>
      <a class="tag" href="https://bandcamp.com/discover/ambient-jazz">ambient jazz</a>
      <a class="tag" href="https://bandcamp.com/discover/fourth-world">fourth world</a>
      <a class="tag" href="https://bandcamp.com/discover/oakland">Oakland</a>
    </div>
    """

    assert extract_bandcamp_release_tags(html) == [
        "ambient",
        "ambient jazz",
        "fourth world",
        "Oakland",
    ]
```

Add:

```python
def test_fetch_bandcamp_release_tags_uses_supplied_url_only(monkeypatch):
    from src.ai_genre_enrichment.source_extraction import fetch_bandcamp_release_tags

    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return """
        <div class="tralbumData tralbum-tags tralbum-tags-nu">
          <a class="tag" href="https://bandcamp.com/discover/electronica">electronica</a>
        </div>
        """

    tags = fetch_bandcamp_release_tags(
        "https://artist.bandcamp.com/album/release",
        fetch_html=fake_fetch,
    )

    assert calls == ["https://artist.bandcamp.com/album/release"]
    assert tags == ["electronica"]
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_extract_bandcamp_tags_from_release_html -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `ImportError` for missing module/function.

- [ ] **Step 3: Implement HTML extraction**

Create `src/ai_genre_enrichment/source_extraction.py`:

```python
from __future__ import annotations

from html.parser import HTMLParser


class _BandcampTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_tag_anchor = False
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())
        href = attr_map.get("href", "")
        if "tag" in classes and "/discover/" in href:
            self._in_tag_anchor = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_tag_anchor = False

    def handle_data(self, data: str) -> None:
        if not self._in_tag_anchor:
            return
        text = " ".join(data.split())
        if text:
            self.tags.append(text)


def extract_bandcamp_release_tags(html: str) -> list[str]:
    parser = _BandcampTagParser()
    parser.feed(html)
    seen: set[str] = set()
    tags: list[str] = []
    for tag in parser.tags:
        key = tag.casefold()
        if key not in seen:
            tags.append(tag)
            seen.add(key)
    return tags


def fetch_bandcamp_release_tags(url: str, *, fetch_html: object) -> list[str]:
    html = fetch_html(url)
    if not isinstance(html, str):
        raise TypeError("fetch_html must return HTML text")
    return extract_bandcamp_release_tags(html)
```

The CLI can pass a small `urllib.request` based fetcher later. Unit tests must inject `fetch_html` and must not perform network calls.

- [ ] **Step 4: Run parser test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_extract_bandcamp_tags_from_release_html -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

Also run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_fetch_bandcamp_release_tags_uses_supplied_url_only -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 3: Preserve-Specific Source Tag Normalization

**Files:**
- Create: `src/ai_genre_enrichment/tag_classification.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing normalization/classification test**

Add:

```python
def test_source_tag_classification_preserves_specific_bandcamp_tags():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    cases = {
        "ambient": ("genre_style", "ambient"),
        "ambient jazz": ("genre_style", "ambient jazz"),
        "electroacoustic": ("genre_style", "electroacoustic"),
        "electronica": ("genre_style", "electronica"),
        "fourth world": ("genre_style", "fourth world"),
        "saxophone": ("instrument", "saxophone"),
        "Oakland": ("place", "oakland"),
        "meditation": ("mood_function", "meditation"),
    }

    for raw, expected in cases.items():
        result = classify_source_tag(raw)
        assert (result.classification, result.normalized_tag) == expected
```

Add:

```python
def test_source_tag_classification_does_not_demote_niche_subgenres():
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    result = classify_source_tag("fourth world")

    assert result.classification == "genre_style"
    assert result.normalized_tag == "fourth world"
    assert result.confidence >= 0.9
    assert "niche" not in result.reason.casefold()
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_tag_classification_preserves_specific_bandcamp_tags -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `ImportError`.

- [ ] **Step 3: Implement deterministic classification**

Create `src/ai_genre_enrichment/tag_classification.py`:

```python
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


GENRE_STYLE_OVERRIDES = {
    "ambient",
    "ambient jazz",
    "electroacoustic",
    "electronica",
    "fourth world",
}

INSTRUMENT_TAGS = {"saxophone", "guitar", "piano", "synthesizer", "drums"}
PLACE_TAGS = {"oakland", "new york", "nyc", "los angeles", "london", "tokyo"}
FORMAT_TAGS = {"live", "demo", "remastered", "compilation", "reissue"}
MOOD_FUNCTION_TAGS = {"meditation", "study", "sleep", "workout", "relaxation"}


@dataclass(frozen=True)
class TagClassification:
    raw_tag: str
    normalized_tag: str
    classification: str
    confidence: float
    classifier: str
    reason: str


def normalize_source_tag(raw_tag: str) -> str:
    text = unicodedata.normalize("NFKC", raw_tag or "").casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def classify_source_tag(raw_tag: str) -> TagClassification:
    normalized = normalize_source_tag(raw_tag)
    if normalized in GENRE_STYLE_OVERRIDES:
        return TagClassification(raw_tag, normalized, "genre_style", 0.95, "deterministic", "Known source-backed genre/style tag; specificity is preserved.")
    if normalized in INSTRUMENT_TAGS:
        return TagClassification(raw_tag, normalized, "instrument", 0.95, "deterministic", "Instrument tag.")
    if normalized in PLACE_TAGS:
        return TagClassification(raw_tag, normalized, "place", 0.95, "deterministic", "Location tag.")
    if normalized in FORMAT_TAGS:
        return TagClassification(raw_tag, normalized, "format", 0.95, "deterministic", "Release format/context tag.")
    if normalized in MOOD_FUNCTION_TAGS:
        return TagClassification(raw_tag, normalized, "mood_function", 0.9, "deterministic", "Mood or listening-function tag.")
    return TagClassification(raw_tag, normalized, "ambiguous", 0.5, "deterministic", "Needs AI adjudication.")
```

- [ ] **Step 4: Run classification test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_tag_classification_preserves_specific_bandcamp_tags -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

Also run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_tag_classification_does_not_demote_niche_subgenres -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 4: Source Locator Schema

**Files:**
- Create: `src/ai_genre_enrichment/source_locator.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing schema test**

Add:

```python
def test_source_locator_schema_returns_only_candidate_sources():
    from src.ai_genre_enrichment.source_locator import source_locator_response_format

    schema = source_locator_response_format()["schema"]
    props = schema["properties"]

    assert "candidate_sources" in props
    assert "new_genres_to_add" not in props
    assert "existing_genres_to_keep" not in props
```

Add:

```python
def test_source_locator_prompt_excludes_baseline_and_streaming_sources():
    from src.ai_genre_enrichment.source_locator import SOURCE_LOCATOR_INSTRUCTIONS

    text = SOURCE_LOCATOR_INSTRUCTIONS.casefold()
    assert "bandcamp" in text
    assert "musicbrainz" in text
    assert "discogs" in text
    assert "last.fm" in text
    assert "spotify" in text
    assert "qobuz" in text
    assert "do not return" in text
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_locator_schema_returns_only_candidate_sources -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `ImportError`.

- [ ] **Step 3: Implement source locator schema**

Create `src/ai_genre_enrichment/source_locator.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any


SOURCE_LOCATOR_INSTRUCTIONS = """Find candidate official release-specific source URLs for one local album.

Return only artist/label/official release pages and Bandcamp release pages.
Do not return genre recommendations.
Do not return MusicBrainz, Discogs, Last.fm, Spotify, Qobuz, Audiomack, Tidal, Deezer, SoundCloud, Wikipedia, review sites, lyrics pages, or generic SEO pages.
Prefer exact artist+album release pages over artist biographies or generic discographies.
If identity is ambiguous, return an empty candidate list with a warning.
"""


SOURCE_LOCATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "release_key": {"type": "string"},
        "candidate_sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_url": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": ["bandcamp_release", "official_release", "official_artist", "official_label"],
                    },
                    "identity_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "identity_status": {
                        "type": "string",
                        "enum": ["confirmed", "probable", "ambiguous", "wrong_release"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["source_url", "source_type", "identity_confidence", "identity_status", "reason"],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["release_key", "candidate_sources", "warnings"],
    "additionalProperties": False,
}


def source_locator_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ai_genre_source_locator",
        "strict": True,
        "schema": deepcopy(SOURCE_LOCATOR_SCHEMA),
    }
```

- [ ] **Step 4: Run schema test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_locator_schema_returns_only_candidate_sources -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

Also run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_source_locator_prompt_excludes_baseline_and_streaming_sources -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 5: Tag Adjudicator Schema

**Files:**
- Create: `src/ai_genre_enrichment/tag_adjudicator.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing schema test**

Add:

```python
def test_tag_adjudicator_schema_classifies_tags_without_web_fields():
    from src.ai_genre_enrichment.tag_adjudicator import tag_adjudicator_response_format

    schema = tag_adjudicator_response_format()["schema"]
    item = schema["properties"]["tag_classifications"]["items"]

    assert "tag_classifications" in schema["properties"]
    assert "source_url" not in item["properties"]
    assert item["properties"]["classification"]["enum"] == [
        "genre_style",
        "descriptor",
        "instrument",
        "place",
        "format",
        "mood_function",
        "review_only",
    ]
```

Add:

```python
def test_tag_adjudicator_instructions_preserve_narrow_source_backed_genres():
    from src.ai_genre_enrichment.tag_adjudicator import TAG_ADJUDICATOR_INSTRUCTIONS

    text = TAG_ADJUDICATOR_INSTRUCTIONS.casefold()
    assert "fourth world" in text
    assert "niche" in text
    assert "not a reason to demote" in text
    assert "do not collapse" in text
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_tag_adjudicator_schema_classifies_tags_without_web_fields -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `ImportError`.

- [ ] **Step 3: Implement tag adjudicator schema**

Create `src/ai_genre_enrichment/tag_adjudicator.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any


TAG_ADJUDICATOR_INSTRUCTIONS = """Classify extracted release-source tags.

Use only the supplied tag list and local payload. Do not browse the web.
Niche or narrow genre/style tags are valuable; niche is not a reason to demote a tag.
For example, fourth world, ambient jazz, electroacoustic, and electronica are legitimate genre/style tags when source-backed.
Do not collapse source-backed specific tags into broad parents.
Separate genres/styles from instruments, places, release formats, moods/functions, and descriptors.
Return strict JSON only.
"""


TAG_CLASSIFICATIONS = [
    "genre_style",
    "descriptor",
    "instrument",
    "place",
    "format",
    "mood_function",
    "review_only",
]

TAG_ADJUDICATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "release_key": {"type": "string"},
        "tag_classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw_tag": {"type": "string"},
                    "normalized_tag": {"type": "string"},
                    "classification": {"type": "string", "enum": TAG_CLASSIFICATIONS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["raw_tag", "normalized_tag", "classification", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["release_key", "tag_classifications", "warnings"],
    "additionalProperties": False,
}


def tag_adjudicator_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ai_genre_tag_adjudicator",
        "strict": True,
        "schema": deepcopy(TAG_ADJUDICATOR_SCHEMA),
    }
```

- [ ] **Step 4: Run schema test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_tag_adjudicator_schema_classifies_tags_without_web_fields -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

Also run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_tag_adjudicator_instructions_preserve_narrow_source_backed_genres -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 6: Build `enriched_genres` Deterministically

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing storage test**

Add:

```python
def test_build_enriched_genres_from_source_tag_classifications(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://colepulice.bandcamp.com/album/gloam-2",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(page_id, ["ambient", "ambient jazz", "saxophone"])
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("cole pulice::gloam")

    rows = sqlite3.connect(db_path).execute(
        """
        SELECT genre, basis, status
        FROM enriched_genres
        WHERE release_key = ?
        ORDER BY genre
        """,
        ("cole pulice::gloam",),
    ).fetchall()

    assert rows == [
        ("ambient", "authoritative_source", "accepted"),
        ("ambient jazz", "authoritative_source", "accepted"),
    ]
```

Add:

```python
def test_build_enriched_genres_preserves_all_specific_genre_style_tags(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="cole pulice::gloam",
        normalized_artist="cole pulice",
        normalized_album="gloam",
        album_id="a1",
        source_url="https://colepulice.bandcamp.com/album/gloam-2",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.99,
        evidence_summary="Official Bandcamp release page.",
    )
    store.replace_source_tags(
        page_id,
        [
            "ambient",
            "ambient jazz",
            "electroacoustic",
            "electronica",
            "fourth world",
            "improvisation",
            "meditation",
            "saxophone",
            "Oakland",
        ],
    )
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("cole pulice::gloam")

    rows = [
        row[0]
        for row in sqlite3.connect(db_path).execute(
            """
            SELECT genre
            FROM enriched_genres
            WHERE release_key = ?
            ORDER BY genre
            """,
            ("cole pulice::gloam",),
        )
    ]

    assert rows == [
        "ambient",
        "ambient jazz",
        "electroacoustic",
        "electronica",
        "fourth world",
    ]
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_build_enriched_genres_from_source_tag_classifications -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `AttributeError` for missing store methods.

- [ ] **Step 3: Implement store methods**

Add methods to `SidecarStore` with these signatures:

```python
def upsert_source_page(
    self,
    *,
    release_key: str,
    normalized_artist: str,
    normalized_album: str,
    album_id: str | None,
    source_url: str,
    source_type: str,
    identity_status: str,
    identity_confidence: float,
    evidence_summary: str,
) -> int:
    """Insert or update one confirmed source page and return source_page_id."""

def replace_source_tags(self, source_page_id: int, raw_tags: list[str]) -> None:
    """Replace extracted source tags for one source page."""

def classify_source_tags(self, source_page_id: int) -> None:
    """Run deterministic source-tag classification for one source page."""
    from .tag_classification import classify_source_tag

def rebuild_enriched_genres_for_release(self, release_key: str) -> None:
    """Rebuild accepted enriched_genres and release signature for one release."""
```

Implementation requirements:
- `replace_source_tags()` deletes previous tags for the page and inserts normalized tags.
- `classify_source_tags()` runs deterministic classification first.
- `rebuild_enriched_genres_for_release()` deletes existing `enriched_genres` rows for that release and inserts only classifications where `classification = 'genre_style'`.
- It must not write descriptor/instrument/place/mood/function tags into `enriched_genres`.
- It must not collapse `electronica` to `electronic`.
- It must not demote `fourth world` because it is specific.
- It writes `enriched_genre_signatures.signature_json` as sorted JSON:

```json
{
  "genres": ["ambient", "ambient jazz"],
  "sources": [{"source_type": "bandcamp_release", "source_url": "https://colepulice.bandcamp.com/album/gloam-2"}]
}
```

- [ ] **Step 4: Run storage test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_build_enriched_genres_from_source_tag_classifications -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

Also run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_build_enriched_genres_preserves_all_specific_genre_style_tags -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 7: Add CLI Commands

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing CLI dry-run test**

Add:

```python
def test_cli_extract_tags_dry_run_does_not_write_sidecar(tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    result = ai_genre_main(
        [
            "--metadata-db", str(metadata_db),
            "--sidecar-db", str(sidecar_db),
            "extract-tags",
            "--artist", "The Bill Evans Trio",
            "--album", "Waltz For Debby",
            "--source-url", "https://example.bandcamp.com/album/waltz",
            "--dry-run",
        ]
    )

    assert result == 0
    conn = sqlite3.connect(sidecar_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_genre_source_pages").fetchone()[0] == 0
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_cli_extract_tags_dry_run_does_not_write_sidecar -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: argparse error for missing command.

- [ ] **Step 3: Add CLI command skeletons**

Add parsers:

```python
extract = sub.add_parser("extract-tags", help="Extract deterministic tags from confirmed source URLs")
add_release_filters(extract)
extract.add_argument("--source-url", action="append", dest="source_urls", required=True)
extract.add_argument("--dry-run", action="store_true")

classify = sub.add_parser("classify-tags", help="Classify extracted source tags")
add_release_filters(classify)
classify.add_argument("--dry-run", action="store_true")

build = sub.add_parser("build-enriched", help="Build enriched_genres from classified source tags")
add_release_filters(build)
build.add_argument("--dry-run", action="store_true")

show = sub.add_parser("show-enriched", help="Show enriched genre signature for a release")
add_release_filters(show)
```

Add command dispatch entries:

```python
if args.command == "extract-tags":
    return cmd_extract_tags(args)
if args.command == "classify-tags":
    return cmd_classify_tags(args)
if args.command == "build-enriched":
    return cmd_build_enriched(args)
if args.command == "show-enriched":
    return cmd_show_enriched(args)
```

- [ ] **Step 4: Implement dry-run behavior first**

`cmd_extract_tags()` should:
- initialize sidecar schema,
- discover matching release read-only,
- print JSON containing release key and URLs,
- write nothing when `--dry-run`.

- [ ] **Step 5: Run CLI dry-run test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_cli_extract_tags_dry_run_does_not_write_sidecar -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 8: Documentation And Verification

**Files:**
- Modify: `docs/AI_GENRE_ENRICHMENT.md`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Add doc assertions**

Add:

```python
def test_ai_genre_docs_describe_enriched_genres_authority():
    text = Path("docs/AI_GENRE_ENRICHMENT.md").read_text(encoding="utf-8")

    assert "enriched_genres" in text
    assert "deterministic Bandcamp" in text
    assert "does not modify artist_genres, album_genres, or track_genres" in text
    assert "source discovery" in text
    assert "tag adjudication" in text
    assert "Niche subgenres are not automatically review-only" in text
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_ai_genre_docs_describe_enriched_genres_authority -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `FAILED` until docs are updated.

- [ ] **Step 3: Update docs**

Add sections:
- “Deterministic-First Workflow”
- “`enriched_genres` Authority Layer”
- “Deterministic Bandcamp Extraction”
- “AI Source Discovery”
- “AI Tag Adjudication”
- “No Legacy Genre Table Mutation”
- “Niche Genre Policy”

Required sentence:

```markdown
This workflow does not modify artist_genres, album_genres, or track_genres; `enriched_genres` is the nondestructive genre authority layer used for future similarity experiments.
```

Required sentence:

```markdown
Niche subgenres are not automatically review-only; source-backed specificity is the main value of this enrichment layer.
```

- [ ] **Step 4: Run doc test**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py::test_ai_genre_docs_describe_enriched_genres_authority -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: `1 passed`.

---

### Task 9: Full Verification

**Files:**
- All touched files.

- [ ] **Step 1: Run AI genre unit tests**

Run:

```powershell
py -m pytest tests\unit\test_ai_genre_enrichment.py -q --basetemp .\.pytest-tmp-ai -o cache_dir=.\.pytest-cache-ai
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

Run:

```powershell
py -m ruff check src\ai_genre_enrichment scripts\ai_genre_enrich.py tests\unit\test_ai_genre_enrichment.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Compile touched Python modules**

Run:

```powershell
py -m compileall src\ai_genre_enrichment scripts\ai_genre_enrich.py
```

Expected: no compile errors.

- [ ] **Step 4: Confirm `metadata.db` remains untouched**

Run before manual/live testing:

```powershell
$metadataHashBefore = (Get-FileHash .\data\metadata.db -Algorithm SHA256).Hash
```

Run after manual/live testing:

```powershell
$metadataHashAfter = (Get-FileHash .\data\metadata.db -Algorithm SHA256).Hash
$metadataHashBefore
$metadataHashAfter
```

Expected: hashes match.

- [ ] **Step 5: Manual happy-path smoke test**

Run:

```powershell
if (Test-Path .\data\ai_genre_enriched_workflow_test.db) { Remove-Item .\data\ai_genre_enriched_workflow_test.db }

python .\scripts\ai_genre_enrich.py --sidecar-db .\data\ai_genre_enriched_workflow_test.db extract-tags --artist "Cole Pulice" --album "Gloam" --source-url "https://colepulice.bandcamp.com/album/gloam-2"

python .\scripts\ai_genre_enrich.py --sidecar-db .\data\ai_genre_enriched_workflow_test.db classify-tags --artist "Cole Pulice" --album "Gloam"

python .\scripts\ai_genre_enrich.py --sidecar-db .\data\ai_genre_enriched_workflow_test.db build-enriched --artist "Cole Pulice" --album "Gloam"

python .\scripts\ai_genre_enrich.py --sidecar-db .\data\ai_genre_enriched_workflow_test.db show-enriched --artist "Cole Pulice" --album "Gloam"
```

Expected enriched genres include:

```text
ambient
ambient jazz
electroacoustic
electronica
fourth world
```

Expected enriched genres exclude:

```text
Oakland
saxophone
meditation
```

---

## Self-Review

- Spec coverage: Covers deterministic source extraction, source discovery boundary, tag classification, sidecar evidence tables, `enriched_genres`, CLI commands, docs, tests, and metadata safety.
- Placeholder scan: No `TBD` or unspecified test steps remain.
- Type consistency: Uses `enriched_genres`, `source_page_id`, `source_tag_id`, `classification`, `genre_style`, and `release_key` consistently across tasks.
- Scope check: This plan does not wire playlist generation to `enriched_genres`; that remains a later opt-in integration after this authority layer is proven.
