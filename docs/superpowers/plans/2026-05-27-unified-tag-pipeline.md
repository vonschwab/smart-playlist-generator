# Unified Tag Pipeline: AI-Informed Deterministic Classification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge Bandcamp, Last.fm, and local metadata tag classification onto a single deterministic-first pipeline where AI adjudicates unknowns once per unique tag and human review handles genuine ambiguity — driving AI token usage toward zero as the vocabulary matures.

**Architecture:** All source tags (Bandcamp, Last.fm, local metadata) flow through: deterministic 3-tier vocabulary → AI adjudication cache → AI API call (tag_adjudicator contract via OpenAI) → human review queue. AI decisions are cached by normalized_tag so each unique tag costs at most one AI call ever. Two graduation paths (human review + AI decisions) promote terms into tier-1 YAML vocabulary.

**Tech Stack:** Python 3.11+, SQLite (sidecar DB), OpenAI API (gpt-4o-mini via existing `OpenAIEnrichmentClient`), Last.fm API (existing `LastFMClient` skeleton), PySide6 (existing ReviewPanel), PyYAML

---

## File Structure

| File | Responsibility | Task |
|------|---------------|------|
| `src/ai_genre_enrichment/storage.py` | Add `ai_tag_adjudication_cache` table, `lookup_cached_adjudication()`, `cache_adjudication()`, `get_ai_graduated_terms()` methods | 1, 2 |
| `src/ai_genre_enrichment/tag_adjudicator.py` | Add `adjudicate_tags()` function that calls OpenAI with the existing contract | 2 |
| `src/ai_genre_enrichment/tag_classification.py` | Add `classify_with_adjudication()` that chains: deterministic → cache → AI → review_only | 2 |
| `src/ai_genre_enrichment/storage.py:classify_source_tags()` | Wire `classify_with_adjudication()` into the shared pipeline so ALL sources benefit | 2 |
| `scripts/ai_genre_enrich.py` | Add `graduate-ai` command, rewrite `extract-lastfm` to call Last.fm API directly, add `--adjudicate` flag to `classify-tags` | 3, 4 |
| `src/ai_genre_enrichment/lastfm_enrichment.py` | New: slim Last.fm tag fetcher using existing `LastFMClient` skeleton | 4 |
| `src/ai_genre_enrichment/source_extraction.py` | Keep `extract_lastfm_tags_from_metadata()` as fallback; add `fetch_lastfm_tags()` that calls the API | 4 |
| `tests/unit/test_ai_genre_enrichment.py` | Tests for cache, adjudication, graduation, Last.fm fetch | 1–4 |
| `data/genre_vocabulary.yaml` | Written to by graduation pipeline (both human and AI) | 3 |

---

## Context for Implementers

### Sidecar DB safety

The enrichment pipeline uses a **sidecar database** (`data/ai_genre_enrichment.db`) — all writes go here. `data/metadata.db` is **read-only** (open with `?mode=ro` URI). Never write to metadata.db.

### Current classification flow

`storage.classify_source_tags(source_page_id)` iterates source tags and calls `tag_classification.classify_source_tag(raw_tag)` which does:
1. `normalize_source_tag(raw_tag)` → Unicode + casefold
2. `GenreVocabulary.classify_genre(tag)` → tier 1 (YAML, 0.95) / tier 2 (engine, 0.85) / tier 3 (library, 0.80)
3. `GenreVocabulary.classify_non_genre(tag)` → descriptor/instrument/place/format/mood_function
4. Fallback → `review_only` at 0.50

The result is stored in `ai_genre_tag_classifications` with `classifier = "deterministic"`.

### What changes

After step 3, before the `review_only` fallback, we insert:
- Step 3.5: Check `ai_tag_adjudication_cache` for `normalized_tag` → if found, use cached decision
- Step 3.6: If `adjudicate=True` and cache miss, call AI tag adjudicator → cache result

This benefits ALL source types equally because `classify_source_tags()` is the shared bottleneck.

### OpenAI client

`src/ai_genre_enrichment/client.py` has `OpenAIEnrichmentClient` using the OpenAI Responses API. It reads `OPENAI_API_KEY` from environment. The tag adjudicator needs a simpler call path — no web search, no validation against the full release schema. We'll add a `adjudicate_tags()` function in `tag_adjudicator.py` that uses the SDK directly (same pattern as `client.py._call_openai` but without the release-level response format).

### Last.fm client

`src/lastfm_client.py` has `LastFMClient` with `get_artist_tags(artist)` and `get_album_tags(artist, album)`. These call `artist.gettoptags` and `album.gettoptags`. The API key comes from `config.yaml` via `config_loader.py:lastfm_api_key`. Rate limit is 5 req/s.

---

### Task 1: AI Adjudication Cache Table

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py:156-335` (initialize method)
- Modify: `src/ai_genre_enrichment/storage.py` (add 3 new methods)
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing tests for cache operations**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_adjudication_cache_stores_and_retrieves(tmp_path):
    store = SidecarStore(tmp_path / "test.db")
    store.initialize()

    assert store.lookup_cached_adjudication("ambient pop") is None

    store.cache_adjudication(
        normalized_tag="ambient pop",
        classification="genre_style",
        confidence=0.82,
        classifier="ai",
    )

    cached = store.lookup_cached_adjudication("ambient pop")
    assert cached is not None
    assert cached["classification"] == "genre_style"
    assert cached["confidence"] == 0.82
    assert cached["classifier"] == "ai"
    assert cached["times_seen"] == 1


def test_adjudication_cache_increments_times_seen(tmp_path):
    store = SidecarStore(tmp_path / "test.db")
    store.initialize()

    store.cache_adjudication(
        normalized_tag="ambient pop",
        classification="genre_style",
        confidence=0.82,
        classifier="ai",
    )
    store.cache_adjudication(
        normalized_tag="ambient pop",
        classification="genre_style",
        confidence=0.82,
        classifier="ai",
    )

    cached = store.lookup_cached_adjudication("ambient pop")
    assert cached["times_seen"] == 2


def test_get_ai_graduated_terms_filters_by_times_seen(tmp_path):
    store = SidecarStore(tmp_path / "test.db")
    store.initialize()

    # Seen once — not enough
    store.cache_adjudication(
        normalized_tag="rare tag",
        classification="genre_style",
        confidence=0.80,
        classifier="ai",
    )
    # Seen 3 times — qualifies
    for _ in range(3):
        store.cache_adjudication(
            normalized_tag="common tag",
            classification="genre_style",
            confidence=0.85,
            classifier="ai",
        )
    # Rejected — excluded
    for _ in range(5):
        store.cache_adjudication(
            normalized_tag="junk tag",
            classification="review_only",
            confidence=0.40,
            classifier="ai",
        )

    terms = store.get_ai_graduated_terms(min_times_seen=3)
    assert "genre_style" in terms
    assert "common tag" in terms["genre_style"]
    assert "rare tag" not in terms.get("genre_style", set())
    assert "junk tag" not in terms.get("review_only", set())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_adjudication_cache_stores_and_retrieves tests/unit/test_ai_genre_enrichment.py::test_adjudication_cache_increments_times_seen tests/unit/test_ai_genre_enrichment.py::test_get_ai_graduated_terms_filters_by_times_seen -v`
Expected: FAIL — `SidecarStore` has no `lookup_cached_adjudication` method

- [ ] **Step 3: Add table to `initialize()` in `storage.py`**

In `storage.py`, add to the `conn.executescript(...)` block inside `initialize()`, after the `ai_genre_review_decisions` table:

```sql
CREATE TABLE IF NOT EXISTS ai_tag_adjudication_cache (
    normalized_tag TEXT PRIMARY KEY,
    classification TEXT NOT NULL,
    confidence REAL NOT NULL,
    classifier TEXT NOT NULL DEFAULT 'ai',
    times_seen INTEGER NOT NULL DEFAULT 1,
    decided_at TEXT NOT NULL
);
```

- [ ] **Step 4: Add `lookup_cached_adjudication()` method**

Add to `SidecarStore` class in `storage.py`:

```python
def lookup_cached_adjudication(self, normalized_tag: str) -> dict[str, Any] | None:
    """Look up a cached AI adjudication result by normalized tag."""
    with self.connect() as conn:
        row = conn.execute(
            "SELECT classification, confidence, classifier, times_seen FROM ai_tag_adjudication_cache WHERE normalized_tag = ?",
            (normalized_tag,),
        ).fetchone()
        return dict(row) if row else None
```

- [ ] **Step 5: Add `cache_adjudication()` method**

Add to `SidecarStore` class in `storage.py`:

```python
def cache_adjudication(
    self,
    *,
    normalized_tag: str,
    classification: str,
    confidence: float,
    classifier: str = "ai",
) -> None:
    """Cache an AI adjudication result, incrementing times_seen on conflict."""
    with self.connect() as conn:
        conn.execute(
            """
            INSERT INTO ai_tag_adjudication_cache (normalized_tag, classification, confidence, classifier, times_seen, decided_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(normalized_tag)
            DO UPDATE SET
                times_seen = ai_tag_adjudication_cache.times_seen + 1,
                decided_at = excluded.decided_at
            """,
            (normalized_tag, classification, confidence, classifier, _now_iso()),
        )
```

- [ ] **Step 6: Add `get_ai_graduated_terms()` method**

Add to `SidecarStore` class in `storage.py`:

```python
def get_ai_graduated_terms(self, min_times_seen: int = 3) -> dict[str, set[str]]:
    """Return AI-adjudicated terms grouped by classification, filtered by frequency."""
    with self.connect() as conn:
        rows = list(conn.execute(
            """
            SELECT normalized_tag, classification
            FROM ai_tag_adjudication_cache
            WHERE times_seen >= ?
              AND classification NOT IN ('review_only', 'rejected')
            """,
            (min_times_seen,),
        ))
        result: dict[str, set[str]] = {}
        for row in rows:
            result.setdefault(row["classification"], set()).add(row["normalized_tag"])
        return result
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_adjudication_cache_stores_and_retrieves tests/unit/test_ai_genre_enrichment.py::test_adjudication_cache_increments_times_seen tests/unit/test_ai_genre_enrichment.py::test_get_ai_graduated_terms_filters_by_times_seen -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -x -q`
Expected: all pass (existing tests unaffected)

- [ ] **Step 9: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add ai_tag_adjudication_cache table and lookup/cache/graduate methods"
```

---

### Task 2: Wire AI Adjudication Into the Shared Classification Pipeline

This is the core change: `classify_source_tags()` gains the ability to check the AI cache and optionally call the AI adjudicator for `review_only` tags. This benefits ALL source types (Bandcamp, Last.fm, local metadata).

**Files:**
- Modify: `src/ai_genre_enrichment/tag_adjudicator.py` (add `adjudicate_tags()` function)
- Modify: `src/ai_genre_enrichment/tag_classification.py` (add `classify_with_adjudication()`)
- Modify: `src/ai_genre_enrichment/storage.py:classify_source_tags()` (wire cache + adjudication)
- Modify: `scripts/ai_genre_enrich.py` (add `--adjudicate` flag to `classify-tags`, `ingest-local`, `extract-lastfm`)
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for `adjudicate_tags()` with mocked OpenAI**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_adjudicate_tags_returns_classifications(monkeypatch):
    from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags

    fake_response = {
        "tag_classifications": [
            {
                "raw_tag": "ambient pop",
                "normalized_tag": "ambient pop",
                "classification": "genre_style",
                "confidence": 0.85,
                "reason": "Recognized subgenre.",
            },
            {
                "raw_tag": "stage & screen",
                "normalized_tag": "stage & screen",
                "classification": "review_only",
                "confidence": 0.30,
                "reason": "Market category, not a genre.",
            },
        ],
        "warnings": [],
    }

    def mock_call(self, prompt, response_format, *, instructions):
        class FakeResponse:
            output_text = json.dumps(fake_response)
            usage = None
        return FakeResponse()

    monkeypatch.setattr(
        "src.ai_genre_enrichment.tag_adjudicator.OpenAIEnrichmentClient._call_openai",
        mock_call,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    results = adjudicate_tags(
        [("ambient pop", "ambient pop"), ("stage & screen", "stage & screen")],
        model="gpt-4o-mini",
    )

    assert len(results) == 2
    assert results["ambient pop"]["classification"] == "genre_style"
    assert results["stage & screen"]["classification"] == "review_only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_adjudicate_tags_returns_classifications -v`
Expected: FAIL — `adjudicate_tags` not found in `tag_adjudicator`

- [ ] **Step 3: Implement `adjudicate_tags()` in `tag_adjudicator.py`**

Add to `src/ai_genre_enrichment/tag_adjudicator.py`:

```python
import json
import logging
import os

logger = logging.getLogger(__name__)


def adjudicate_tags(
    tags: list[tuple[str, str]],
    *,
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Call AI to classify a batch of unknown tags.

    Args:
        tags: List of (raw_tag, normalized_tag) pairs to classify.
        model: OpenAI model to use.
        dry_run: If True, return empty results without calling API.

    Returns:
        Dict keyed by normalized_tag → {"classification", "confidence", "reason"}.
    """
    if not tags or dry_run:
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping AI adjudication")
        return {}

    tag_list = "\n".join(f"- raw: {raw!r}, normalized: {norm!r}" for raw, norm in tags)
    prompt = f"Classify the following source tags:\n\n{tag_list}"

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=model,
            instructions=TAG_ADJUDICATOR_INSTRUCTIONS,
            input=[{"role": "user", "content": prompt}],
            text={"format": tag_adjudicator_response_format()},
        )

        output_text = getattr(response, "output_text", None)
        if not output_text:
            logger.warning("AI adjudicator returned no output")
            return {}

        data = json.loads(output_text)
        results: dict[str, dict[str, Any]] = {}
        for item in data.get("tag_classifications", []):
            norm = item.get("normalized_tag", "").strip().casefold()
            if norm:
                results[norm] = {
                    "classification": item["classification"],
                    "confidence": item["confidence"],
                    "reason": item.get("reason", ""),
                }

        usage = getattr(response, "usage", None)
        if usage:
            input_t = getattr(usage, "input_tokens", 0) or 0
            output_t = getattr(usage, "output_tokens", 0) or 0
            logger.info("AI adjudication: %d tags, %d input + %d output tokens", len(tags), input_t, output_t)

        return results

    except ImportError:
        logger.warning("OpenAI SDK not installed — skipping AI adjudication")
        return {}
    except Exception:
        logger.exception("AI adjudication failed")
        return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_adjudicate_tags_returns_classifications -v`
Expected: PASS

- [ ] **Step 5: Commit adjudicator implementation**

```bash
git add src/ai_genre_enrichment/tag_adjudicator.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: implement adjudicate_tags() — AI classification for unknown source tags"
```

- [ ] **Step 6: Write failing test for cache-aware classification in `classify_source_tags`**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_classify_source_tags_uses_adjudication_cache(tmp_path):
    """When a tag is in the adjudication cache, classify_source_tags uses the cached result
    instead of falling back to review_only."""
    from src.ai_genre_enrichment.tag_classification import reset_vocabulary

    store = SidecarStore(tmp_path / "test.db")
    store.initialize()
    reset_vocabulary()

    # Pre-populate cache with a decision for "ambient pop"
    store.cache_adjudication(
        normalized_tag="ambient pop",
        classification="genre_style",
        confidence=0.82,
        classifier="ai",
    )

    # Create a source page with "ambient pop" tag
    page_id = store.upsert_source_page(
        release_key="test::album",
        normalized_artist="test",
        normalized_album="album",
        album_id=None,
        source_url="https://test.bandcamp.com/album/test",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="test",
    )
    store.replace_source_tags(page_id, ["ambient pop"])

    # Classify — should pick up cached adjudication
    store.classify_source_tags(page_id, adjudicate=False)

    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT c.classification, c.confidence, c.classifier
            FROM ai_genre_tag_classifications c
            JOIN ai_genre_source_tags t ON t.source_tag_id = c.source_tag_id
            WHERE t.source_page_id = ? AND t.normalized_tag = 'ambient pop'
            """,
            (page_id,),
        ).fetchone()

    assert row is not None
    assert row["classification"] == "genre_style"
    assert row["confidence"] == 0.82
    assert row["classifier"] == "cached_ai"
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_classify_source_tags_uses_adjudication_cache -v`
Expected: FAIL — `classify_source_tags()` doesn't accept `adjudicate` parameter

- [ ] **Step 8: Modify `classify_source_tags()` in `storage.py`**

Change the signature and body of `classify_source_tags` in `storage.py`. The existing method at line 914 becomes:

```python
def classify_source_tags(self, source_page_id: int, *, adjudicate: bool = False, model: str = "gpt-4o-mini") -> None:
    """Run deterministic source-tag classification for one source page.

    If adjudicate=True and OPENAI_API_KEY is set, unknown tags are batched
    and sent to the AI tag adjudicator. Results are cached in
    ai_tag_adjudication_cache so each unique tag costs at most one AI call.
    """
    from .tag_classification import classify_source_tag

    with self.connect() as conn:
        tags = list(
            conn.execute(
                """
                SELECT source_tag_id, raw_tag
                FROM ai_genre_source_tags
                WHERE source_page_id = ?
                ORDER BY tag_position, source_tag_id
                """,
                (source_page_id,),
            )
        )
        conn.execute(
            """
            DELETE FROM ai_genre_tag_classifications
            WHERE source_tag_id IN (
                SELECT source_tag_id
                FROM ai_genre_source_tags
                WHERE source_page_id = ?
            )
            """,
            (source_page_id,),
        )
        rows = []
        review_only_batch: list[tuple[int, str, str]] = []  # (source_tag_id, raw_tag, normalized_tag)
        seen_normalized_tags: set[str] = set()
        deleted_source_tag_ids: set[int] = set()
        for tag in tags:
            source_tag_id = int(tag["source_tag_id"])
            if source_tag_id in deleted_source_tag_ids:
                continue
            classification = classify_source_tag(tag["raw_tag"])
            if classification.normalized_tag in seen_normalized_tags:
                conn.execute("DELETE FROM ai_genre_source_tags WHERE source_tag_id = ?", (source_tag_id,))
                continue
            duplicate_rows = list(
                conn.execute(
                    """
                    SELECT source_tag_id
                    FROM ai_genre_source_tags
                    WHERE source_page_id = ?
                      AND normalized_tag = ?
                      AND source_tag_id != ?
                    """,
                    (source_page_id, classification.normalized_tag, source_tag_id),
                )
            )
            for duplicate in duplicate_rows:
                duplicate_id = int(duplicate["source_tag_id"])
                deleted_source_tag_ids.add(duplicate_id)
                conn.execute("DELETE FROM ai_genre_source_tags WHERE source_tag_id = ?", (duplicate_id,))
            conn.execute(
                """
                UPDATE ai_genre_source_tags
                SET normalized_tag = ?
                WHERE source_tag_id = ?
                """,
                (classification.normalized_tag, source_tag_id),
            )
            seen_normalized_tags.add(classification.normalized_tag)

            # Check adjudication cache for review_only tags
            if classification.classification == "review_only":
                cached = self.lookup_cached_adjudication(classification.normalized_tag)
                if cached is not None:
                    rows.append((
                        source_tag_id,
                        cached["classification"],
                        cached["confidence"],
                        "cached_ai",
                        f"Cached AI adjudication (seen {cached['times_seen']}x).",
                        _now_iso(),
                    ))
                    # Increment times_seen
                    self.cache_adjudication(
                        normalized_tag=classification.normalized_tag,
                        classification=cached["classification"],
                        confidence=cached["confidence"],
                        classifier=cached["classifier"],
                    )
                    continue
                # Collect for batch AI adjudication
                review_only_batch.append((source_tag_id, tag["raw_tag"], classification.normalized_tag))
                continue

            rows.append(
                (
                    source_tag_id,
                    classification.classification,
                    classification.confidence,
                    "deterministic",
                    classification.reason,
                    _now_iso(),
                )
            )

        # Batch AI adjudication for review_only tags
        if review_only_batch and adjudicate:
            from .tag_adjudicator import adjudicate_tags

            ai_input = [(raw, norm) for _, raw, norm in review_only_batch]
            ai_results = adjudicate_tags(ai_input, model=model)

            for source_tag_id, raw_tag, normalized_tag in review_only_batch:
                ai_result = ai_results.get(normalized_tag)
                if ai_result is not None:
                    self.cache_adjudication(
                        normalized_tag=normalized_tag,
                        classification=ai_result["classification"],
                        confidence=ai_result["confidence"],
                        classifier="ai",
                    )
                    rows.append((
                        source_tag_id,
                        ai_result["classification"],
                        ai_result["confidence"],
                        "ai",
                        ai_result.get("reason", "AI adjudication."),
                        _now_iso(),
                    ))
                else:
                    rows.append((
                        source_tag_id,
                        "review_only",
                        0.5,
                        "deterministic",
                        "Unknown source tag requires adjudication before use.",
                        _now_iso(),
                    ))
        elif review_only_batch:
            # No adjudication — fall back to review_only
            for source_tag_id, raw_tag, normalized_tag in review_only_batch:
                rows.append((
                    source_tag_id,
                    "review_only",
                    0.5,
                    "deterministic",
                    "Unknown source tag requires adjudication before use.",
                    _now_iso(),
                ))

        if rows:
            conn.executemany(
                """
                INSERT INTO ai_genre_tag_classifications (
                    source_tag_id, classification, confidence, classifier, reason, classified_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
```

- [ ] **Step 9: Run tests to verify cache-aware classification passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_classify_source_tags_uses_adjudication_cache -v`
Expected: PASS

- [ ] **Step 10: Write failing test for AI adjudication during classify**

```python
def test_classify_source_tags_calls_ai_adjudicator_on_unknown(tmp_path, monkeypatch):
    """When adjudicate=True and a tag is unknown, AI is called and result is cached."""
    from src.ai_genre_enrichment.tag_classification import reset_vocabulary

    store = SidecarStore(tmp_path / "test.db")
    store.initialize()
    reset_vocabulary()

    fake_response = {
        "tag_classifications": [
            {
                "raw_tag": "witch house",
                "normalized_tag": "witch house",
                "classification": "genre_style",
                "confidence": 0.88,
                "reason": "Recognized electronic subgenre.",
            }
        ],
        "warnings": [],
    }

    def mock_call(self, prompt, response_format, *, instructions):
        class FakeResponse:
            output_text = json.dumps(fake_response)
            usage = None
        return FakeResponse()

    monkeypatch.setattr(
        "src.ai_genre_enrichment.tag_adjudicator.OpenAI",
        type("FakeOpenAI", (), {"__init__": lambda self: None, "responses": type("R", (), {"create": mock_call})()}),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    page_id = store.upsert_source_page(
        release_key="test::album",
        normalized_artist="test",
        normalized_album="album",
        album_id=None,
        source_url="https://test.bandcamp.com/album/test",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="test",
    )
    store.replace_source_tags(page_id, ["witch house"])
    store.classify_source_tags(page_id, adjudicate=True)

    # Check classification was written
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT c.classification, c.confidence, c.classifier
            FROM ai_genre_tag_classifications c
            JOIN ai_genre_source_tags t ON t.source_tag_id = c.source_tag_id
            WHERE t.source_page_id = ?
            """,
            (page_id,),
        ).fetchone()

    assert row["classification"] == "genre_style"
    assert row["classifier"] == "ai"

    # Check result was cached
    cached = store.lookup_cached_adjudication("witch house")
    assert cached is not None
    assert cached["classification"] == "genre_style"
```

- [ ] **Step 11: Run test to verify it passes** (implementation from step 8 should already handle this)

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_classify_source_tags_calls_ai_adjudicator_on_unknown -v`
Expected: PASS

- [ ] **Step 12: Add `--adjudicate` flag to `classify-tags`, `ingest-local`, and `extract-lastfm` CLI commands**

In `scripts/ai_genre_enrich.py`, add to the parser setup for each command:

```python
# In build_parser(), after each sub.add_parser for classify, ingest_local, extract_lastfm:
classify.add_argument("--adjudicate", action="store_true", help="Use AI to classify unknown tags (costs tokens)")
ingest_local.add_argument("--adjudicate", action="store_true", help="Use AI to classify unknown tags (costs tokens)")
extract_lastfm.add_argument("--adjudicate", action="store_true", help="Use AI to classify unknown tags (costs tokens)")
```

Then in `cmd_classify_tags`, change `store.classify_source_tags(page_id)` to:
```python
store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
```

And in `cmd_ingest_local`, change `store.classify_source_tags(page_id)` to:
```python
store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
```

And in `cmd_extract_lastfm`, change `store.classify_source_tags(page_id)` to:
```python
store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
```

- [ ] **Step 13: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -x -q`
Expected: all pass

- [ ] **Step 14: Commit**

```bash
git add src/ai_genre_enrichment/tag_adjudicator.py src/ai_genre_enrichment/tag_classification.py src/ai_genre_enrichment/storage.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: wire AI adjudication into shared classify_source_tags pipeline

All source types (Bandcamp, Last.fm, local metadata) benefit from
deterministic → cache → AI → review_only classification chain.
Use --adjudicate flag to enable AI calls for unknown tags."
```

---

### Task 3: `graduate-ai` CLI Command

**Files:**
- Modify: `scripts/ai_genre_enrich.py` (add `graduate-ai` command)
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for `graduate-ai` command**

```python
def test_graduate_ai_writes_to_vocab_yaml(tmp_path):
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary, reset_vocabulary

    store = SidecarStore(tmp_path / "test.db")
    store.initialize()

    # Simulate an AI-adjudicated tag seen 5 times
    for _ in range(5):
        store.cache_adjudication(
            normalized_tag="witch house",
            classification="genre_style",
            confidence=0.88,
            classifier="ai",
        )
    # And a descriptor seen 3 times
    for _ in range(3):
        store.cache_adjudication(
            normalized_tag="lo-fi recording",
            classification="descriptor",
            confidence=0.90,
            classifier="ai",
        )
    # And one seen only once — should not graduate
    store.cache_adjudication(
        normalized_tag="rare genre",
        classification="genre_style",
        confidence=0.80,
        classifier="ai",
    )

    # Create a minimal vocab YAML
    vocab_path = tmp_path / "genre_vocabulary.yaml"
    import yaml
    yaml.dump(
        {"version": 1, "genre_style": ["ambient"], "descriptor": ["acoustic"], "aliases": {}},
        vocab_path.open("w"),
        default_flow_style=False,
    )

    reset_vocabulary()

    result = ai_genre_cli.main([
        "--sidecar-db", str(tmp_path / "test.db"),
        "--metadata-db", str(tmp_path / "empty.db"),
        "graduate-ai",
        "--vocab-yaml", str(vocab_path),
        "--min-times-seen", "3",
    ])

    assert result == 0

    vocab = GenreVocabulary(vocab_path)
    assert vocab.classify_genre("witch house") is not None
    assert vocab.classify_non_genre("lo-fi recording") == "descriptor"
    assert vocab.classify_genre("rare genre") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_graduate_ai_writes_to_vocab_yaml -v`
Expected: FAIL — unknown command `graduate-ai`

- [ ] **Step 3: Add `graduate-ai` command to CLI**

In `scripts/ai_genre_enrich.py`, add parser entry in `build_parser()`:

```python
graduate_ai = sub.add_parser("graduate-ai", help="Graduate AI-adjudicated tags into vocabulary YAML")
graduate_ai.add_argument(
    "--vocab-yaml",
    type=Path,
    default=ROOT / "data" / "genre_vocabulary.yaml",
)
graduate_ai.add_argument(
    "--min-times-seen",
    type=int,
    default=3,
    help="Minimum times a tag must have been seen to graduate (default: 3)",
)
```

Add command dispatch in `main()`:

```python
if args.command == "graduate-ai":
    return cmd_graduate_ai(args)
```

Add implementation:

```python
def cmd_graduate_ai(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import reset_vocabulary

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    terms = store.get_ai_graduated_terms(min_times_seen=args.min_times_seen)
    if not terms:
        print("No AI-adjudicated tags meet the graduation threshold.")
        return 0

    vocab = GenreVocabulary(args.vocab_yaml)
    added = 0
    for classification, tags in sorted(terms.items()):
        category = classification
        # Map label_or_org → label_or_org in vocab (it's a valid category)
        for tag in sorted(tags):
            try:
                vocab.add_term(category, tag)
                added += 1
                print(f"  graduated {tag!r} → {category}")
            except ValueError:
                print(f"  skipped {tag!r} — unknown category {category!r}")

    if added:
        vocab.save()
        print(f"\nGraduated {added} term(s) into {args.vocab_yaml}.")
    else:
        print("No new terms to graduate.")
    reset_vocabulary()
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_graduate_ai_writes_to_vocab_yaml -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add graduate-ai command to promote AI-adjudicated tags into vocabulary"
```

---

### Task 4: Last.fm API Integration

Replace the metadata.db-mining approach in `extract-lastfm` with direct Last.fm API calls. Tags flow through the same unified pipeline as Bandcamp.

**Files:**
- Create: `src/ai_genre_enrichment/lastfm_enrichment.py`
- Modify: `scripts/ai_genre_enrich.py` (rewrite `cmd_extract_lastfm`)
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for `fetch_lastfm_tags()`**

```python
def test_fetch_lastfm_tags_calls_api_and_returns_tags(monkeypatch):
    from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags

    fake_api_response = {
        "toptags": {
            "tag": [
                {"name": "shoegaze", "count": "100", "url": ""},
                {"name": "dream pop", "count": "80", "url": ""},
                {"name": "noise pop", "count": "60", "url": ""},
                {"name": "seen live", "count": "50", "url": ""},
                {"name": "indie", "count": "40", "url": ""},
                {"name": "favorites", "count": "30", "url": ""},
            ],
            "@attr": {"artist": "Slowdive", "album": "Souvlaki"},
        }
    }

    def mock_get(url, params=None, timeout=None):
        class FakeResp:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return fake_api_response
        return FakeResp()

    monkeypatch.setattr("src.ai_genre_enrichment.lastfm_enrichment.requests.get", mock_get)

    tags = fetch_lastfm_tags(
        artist="Slowdive",
        album="Souvlaki",
        api_key="test-key",
        limit=20,
    )

    # "seen live" and "favorites" should be filtered out
    assert "shoegaze" in tags
    assert "dream pop" in tags
    assert "noise pop" in tags
    assert "indie" in tags
    assert "seen live" not in tags
    assert "favorites" not in tags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_fetch_lastfm_tags_calls_api_and_returns_tags -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create `lastfm_enrichment.py`**

Create `src/ai_genre_enrichment/lastfm_enrichment.py`:

```python
"""Slim Last.fm tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.genre.normalize_unified import DROP_TOKENS, META_TAGS

logger = logging.getLogger(__name__)

BASE_URL = "https://ws.audioscrobbler.com/2.0/"

LASTFM_NOISE_TAGS = {
    "seen live", "favorite", "favorites", "favourites", "albums i own",
    "love", "loved", "beautiful", "awesome", "great", "amazing",
    "check out", "cool", "nice", "good", "best",
}


def fetch_lastfm_tags(
    *,
    artist: str,
    album: str | None = None,
    api_key: str,
    limit: int = 20,
) -> list[str]:
    """Fetch top tags for an artist (and optionally album) from Last.fm API.

    Calls album.gettoptags if album is provided, otherwise artist.gettoptags.
    Pre-filters Last.fm noise tags and META_TAGS/DROP_TOKENS.

    Returns:
        List of tag name strings, filtered and deduplicated.
    """
    noise = LASTFM_NOISE_TAGS | META_TAGS | DROP_TOKENS
    all_tags: list[str] = []

    if album:
        album_tags = _fetch_toptags("album.gettoptags", api_key, artist=artist, album=album, limit=limit)
        all_tags.extend(album_tags)

    artist_tags = _fetch_toptags("artist.gettoptags", api_key, artist=artist, limit=limit)
    all_tags.extend(artist_tags)

    seen: set[str] = set()
    filtered: list[str] = []
    for tag in all_tags:
        key = tag.strip().casefold()
        if key and key not in noise and key not in seen and len(key) > 2:
            seen.add(key)
            filtered.append(tag.strip())
    return filtered


def _fetch_toptags(method: str, api_key: str, *, limit: int = 20, **params: str) -> list[str]:
    """Call a Last.fm gettoptags endpoint and return raw tag names."""
    request_params: dict[str, Any] = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
        **params,
    }

    try:
        response = requests.get(BASE_URL, params=request_params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Last.fm API request failed for %s", method)
        return []

    toptags = data.get("toptags", {})
    tags = toptags.get("tag", [])
    if not isinstance(tags, list):
        tags = [tags] if tags else []

    return [tag.get("name", "") for tag in tags[:limit] if tag.get("name")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_fetch_lastfm_tags_calls_api_and_returns_tags -v`
Expected: PASS

- [ ] **Step 5: Commit fetch implementation**

```bash
git add src/ai_genre_enrichment/lastfm_enrichment.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add lastfm_enrichment.py — slim API fetcher for genre enrichment"
```

- [ ] **Step 6: Rewrite `cmd_extract_lastfm` to call API**

In `scripts/ai_genre_enrich.py`, update the parser entry for `extract-lastfm`:

```python
extract_lastfm = sub.add_parser("extract-lastfm", help="Fetch Last.fm tags via API and classify")
add_release_filters(extract_lastfm)
extract_lastfm.add_argument("--dry-run", action="store_true")
extract_lastfm.add_argument("--adjudicate", action="store_true", help="Use AI to classify unknown tags")
extract_lastfm.add_argument("--lastfm-api-key", help="Last.fm API key (default: from config.yaml or LASTFM_API_KEY env)")
```

Replace `cmd_extract_lastfm`:

```python
def cmd_extract_lastfm(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary

    import os
    api_key = getattr(args, "lastfm_api_key", None) or os.environ.get("LASTFM_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            config = Config()
            api_key = config.lastfm_api_key
        except Exception:
            pass
    if not api_key:
        print("Error: Last.fm API key required. Set LASTFM_API_KEY env var, use --lastfm-api-key, or configure in config.yaml.")
        return 1

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    vocab = GenreVocabulary(library_db_path=args.metadata_db)
    set_vocabulary(vocab)

    extracted = 0
    for release in releases:
        tags = fetch_lastfm_tags(
            artist=release.artist,
            album=release.album if hasattr(release, "album") else release.normalized_album,
            api_key=api_key,
            limit=20,
        )
        if not tags:
            continue

        if args.dry_run:
            print(json.dumps({
                "release_key": release.release_key,
                "lastfm_tags": tags,
                "dry_run": True,
            }, ensure_ascii=False, sort_keys=True))
            continue

        page_id = store.upsert_source_page(
            release_key=release.release_key,
            normalized_artist=release.normalized_artist,
            normalized_album=release.normalized_album,
            album_id=release.album_id,
            source_url=f"lastfm://artist/{release.normalized_artist}/album/{release.normalized_album}",
            source_type="lastfm_tags",
            identity_status="confirmed",
            identity_confidence=0.9,
            evidence_summary="Last.fm top tags via API.",
        )
        store.replace_source_tags(page_id, tags)
        store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
        store.rebuild_enriched_genres_for_release(release.release_key)
        extracted += 1
        print(f"extracted-lastfm {release.release_key} tags={len(tags)}")

    print(f"Extracted Last.fm tags for {extracted} release(s).")
    return 0
```

- [ ] **Step 7: Write test for the rewritten extract-lastfm command**

```python
def test_extract_lastfm_command_calls_api(tmp_path, monkeypatch):
    from src.ai_genre_enrichment.tag_classification import reset_vocabulary

    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"
    reset_vocabulary()

    fake_api_response = {
        "toptags": {
            "tag": [
                {"name": "shoegaze", "count": "100", "url": ""},
                {"name": "dream pop", "count": "80", "url": ""},
            ],
            "@attr": {"artist": "Slowdive"},
        }
    }

    def mock_get(url, params=None, timeout=None):
        class FakeResp:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return fake_api_response
        return FakeResp()

    monkeypatch.setattr("src.ai_genre_enrichment.lastfm_enrichment.requests.get", mock_get)

    result = ai_genre_cli.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "extract-lastfm",
        "--artist", "Slowdive",
        "--lastfm-api-key", "test-key",
    ])

    assert result == 0

    store = SidecarStore(sidecar_db)
    with store.connect() as conn:
        pages = list(conn.execute(
            "SELECT * FROM ai_genre_source_pages WHERE source_type = 'lastfm_tags'"
        ))
    assert len(pages) >= 1
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_command_calls_api -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -x -q`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add src/ai_genre_enrichment/lastfm_enrichment.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: rewrite extract-lastfm to call Last.fm API directly

Replaces the metadata.db mining approach (which had no Last.fm data)
with direct album.gettoptags + artist.gettoptags API calls.
Tags flow through the same unified classification pipeline as
Bandcamp and local metadata sources."
```

---

## Unified Pipeline Summary

After all 4 tasks, the classification flow for ANY source is:

```
Source (Bandcamp / Last.fm API / local metadata / future sources)
  ↓
replace_source_tags() — store raw tags
  ↓
classify_source_tags(adjudicate=True)
  ├─ 1. normalize_source_tag()
  ├─ 2. GenreVocabulary.classify_genre() → tier 1/2/3 → DONE
  ├─ 3. GenreVocabulary.classify_non_genre() → descriptor/etc → DONE
  ├─ 4. ai_tag_adjudication_cache lookup → cached? → DONE
  ├─ 5. AI adjudicator batch call → cache result → DONE
  └─ 6. review_only → human review queue
  ↓
rebuild_enriched_genres_for_release()
  ↓
enriched_genres (merged authority layer)
```

Graduation paths (both write to `data/genre_vocabulary.yaml`):
- `graduate-reviewed` — human decisions → tier-1
- `graduate-ai` — AI cache entries seen ≥N times → tier-1

Token convergence: each unique normalized_tag costs at most one AI call. After graduation, it costs zero.
