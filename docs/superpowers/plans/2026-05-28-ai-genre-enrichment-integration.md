# AI Genre Enrichment Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AI genre enrichment pipeline end-to-end usable: add Bandcamp as a tag source, route every genre consumer (GUI display, candidate pool, scoring, artifact build) through enriched signatures when present, add a GUI panel that lets users trigger enrichment per artist, and expose the human-review queue in the GUI.

**Architecture:** A new `bandcamp_enrichment` module mirrors `lastfm_enrichment` and shares the downstream classification pipeline. A new `EnrichedGenreResolver` opens the sidecar `ai_genre_enrichment.db` read-only and provides forward lookups (`(artist, album) → list[genre] | None`) and reverse lookups (`genre → set[release_key]`). Three call-site groups become resolver-aware: the GUI worker (display), `LocalLibraryClient.get_tracks_for_genre` (candidate pool gating), and `SimilarityCalculator._get_combined_genres` / `_get_combined_genres_with_weights` (per-track genre lookup used by every scoring path: beam search, genre vectors, transition scoring). The artifact builder gains the same hook so rebuilt `data_matrices_step1.npz` reflects enriched genres for IDF. Two new GUI panels (`EnrichmentPanel`, plus wiring of the existing `ReviewPanel`) expose enrichment status, per-artist trigger, and review queue.

**Tech Stack:** Python 3.11+, SQLite (sidecar DB, read-only), OpenAI API (existing `OpenAIEnrichmentClient` for Bandcamp URL discovery via `source_locator`), PySide6 (GUI), pytest

---

## Scope Note

This plan covers **full read-side integration**: the GUI shows enriched genres, the candidate pool query honors enriched signatures, and the resolver primitive is available to future consumers. Since enriched signatures already absorb the raw metadata at enrichment time, there are no overlapping authorities at query time — for each release, enriched applies if present, raw applies otherwise. No set arithmetic, just two cases.

## File Structure

| File | Responsibility | Task |
|------|---------------|------|
| `src/ai_genre_enrichment/bandcamp_enrichment.py` | NEW: `fetch_bandcamp_tags(artist, album, ...)` — find URL via `source_locator` (AI) → scrape via `source_extraction.fetch_bandcamp_release_tags` | 1 |
| `scripts/ai_genre_enrich.py` | Add `cmd_extract_bandcamp` + `extract-bandcamp` subparser; mirrors `cmd_extract_lastfm` | 2 |
| `src/ai_genre_enrichment/genre_resolver.py` | NEW: `EnrichedGenreResolver` class — read-only access to `enriched_genre_signatures` | 3 |
| `src/playlist_gui/worker.py` | Modify `_top_genres_for_index` and the `formatted_tracks` build (lines 750-760, 1240-1290) to prefer resolver output when available | 4 |
| `src/playlist_gui/widgets/enrichment_panel.py` | NEW: PySide6 widget showing per-artist enrichment status + enrich button | 5 |
| `src/playlist_gui/worker.py` | Add `enrich_artist` command handler that shells out to the CLI pipeline | 6 |
| `src/playlist_gui/worker_client.py` | Add `enrich_artist(artist)` method on `WorkerClient` | 6 |
| `src/playlist_gui/main_window.py` | Wire `EnrichmentPanel` into the layout, connect signals to `WorkerClient` | 7 |
| `src/ai_genre_enrichment/genre_resolver.py` | Extend with `get_release_keys_with_genre(genre)` and `get_all_enriched_release_keys()` reverse lookups | 8 |
| `src/local_library_client.py:256-329` | Make `get_tracks_for_genre` sidecar-aware: skip tracks on enriched releases that don't match, add tracks from enriched releases that do | 8 |
| `tests/unit/test_ai_genre_enrichment.py` | Add tests for `bandcamp_enrichment`, `extract-bandcamp` CLI, `EnrichedGenreResolver` (forward + reverse lookups) | 1, 2, 3, 8 |
| `tests/unit/test_playlist_gui_genre_resolver.py` | NEW: tests for worker integration of resolver into formatted_tracks | 4 |
| `tests/unit/test_enrichment_panel.py` | NEW: tests for the panel widget (pytest-qt) | 5 |
| `tests/unit/test_worker_enrich_artist.py` | NEW: tests for `enrich_artist` command dispatch | 6 |
| `tests/unit/test_local_library_client_enriched.py` | NEW: tests for sidecar-aware `get_tracks_for_genre` | 8 |
| `src/similarity_calculator.py:22-32` (init), `:1029` (`_get_combined_genres`), `:901` (`_get_combined_genres_with_weights`) | Accept optional `enriched_resolver`; route per-track genre lookups through it when an enriched signature exists for `(artist, album)` | 9 |
| `tests/unit/test_similarity_calc_enriched.py` | NEW: tests for resolver-aware combined genre lookups | 9 |
| `src/playlist_gui/widgets/review_panel.py` | Add "Graduate to YAML" button + "Open CLI review" button; wire `review_completed` signal | 10 |
| `src/playlist_gui/main_window.py` | Mount `ReviewPanel` as a dock or tab; refresh `EnrichmentPanel` when graduation completes | 10 |
| `tests/unit/test_review_panel_graduate.py` | NEW: tests for graduate button + CLI launcher | 10 |
| `src/analyze/artifact_builder.py:280-350` (per-track genre lookup) | Accept optional `enriched_resolver` and use it before falling back to `metadata.db` reads | 11 |
| `scripts/ai_genre_enrich.py` | Add `rebuild-artifacts` subcommand that invokes the artifact builder with the resolver | 11 |
| `tests/unit/test_artifact_builder_enriched.py` | NEW: tests for resolver-aware artifact build | 11 |

---

## Context for Implementers

### Sidecar DB Safety

- `data/metadata.db` is **read-only** (irreplaceable production data). Open with `?mode=ro` URI.
- `data/ai_genre_enrichment.db` is the sidecar — all writes go here.
- Enriched genres are the **authoritative source** when present, falling back to raw `metadata.db` genres for unenriched releases.

### Existing Pipeline Shape

`SidecarStore.classify_source_tags(page_id)` and `SidecarStore.rebuild_enriched_genres_for_release(release_key)` are the shared downstream pipeline. Any new source just needs to:
1. `store.upsert_source_page(...)` → returns `page_id`
2. `store.replace_source_tags(page_id, raw_tags)` → writes to `ai_genre_source_tags`
3. `store.classify_source_tags(page_id, ...)` → applies deterministic+AI classification
4. `store.rebuild_enriched_genres_for_release(release_key)` → produces enriched_genre_signatures

See `cmd_extract_lastfm` at `scripts/ai_genre_enrich.py:510` for the full pattern.

### Bandcamp URL Discovery

`src/ai_genre_enrichment/source_locator.py` defines `SOURCE_LOCATOR_INSTRUCTIONS` and `source_locator_response_format()` for an OpenAI structured-output call that returns candidate source URLs (preferring Bandcamp). The schema yields:
```json
{
  "candidate_sources": [
    {"source_url": "https://artist.bandcamp.com/album/x", "source_type": "bandcamp_release", "identity_status": "confirmed", "identity_confidence": 0.9, ...}
  ],
  "warnings": [...]
}
```

The Bandcamp fetcher calls `OpenAIEnrichmentClient._call_openai` with that schema/instructions, filters for `source_type == "bandcamp_release"` and `identity_confidence >= 0.7`, picks the highest-confidence URL, then calls `fetch_bandcamp_release_tags(url)`.

### `enriched_genre_signatures` Schema

```sql
CREATE TABLE enriched_genre_signatures (
    release_key TEXT PRIMARY KEY,           -- "artist::album", both casefold-normalized
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    signature_json TEXT NOT NULL,           -- {"genres": [...], "sources": [...]}
    updated_at TEXT NOT NULL
);
```

`release_key` is `f"{normalized_artist}::{normalized_album}"`. Normalization: `unicodedata.NFKD` + casefold + strip combining marks + collapse whitespace (see `normalize_source_tag` in `tag_classification.py`).

### GUI Worker IPC

`src/playlist_gui/worker.py` is run as a subprocess by `WorkerClient` (`src/playlist_gui/worker_client.py`). It speaks NDJSON over stdout via `emit_event`, `emit_log`, `emit_progress`, `emit_result`, `emit_done`. New commands are dispatched in `main()` (~line 1919).

The pattern: `WorkerClient.<command>()` builds an NDJSON command, sends it to the worker, the worker emits events tagged with `request_id`, the GUI routes them to Qt signals.

### Current Genres Display

`src/playlist_gui/worker.py:1242-1290` populates `formatted_tracks` with a `"genres"` field. The current source is either `track.get('genres', [])` (set by the playlist engine from the candidate pool) or a fallback to `generator.similarity_calc.get_filtered_combined_genres_for_track(rating_key)`. Both paths read from `metadata.db`. The resolver wraps these calls: prefer enriched genres by `(artist, album)`, fall back to the existing logic.

---

## Task 1: Bandcamp Tag Fetcher

**Files:**
- Create: `src/ai_genre_enrichment/bandcamp_enrichment.py`
- Test: `tests/unit/test_ai_genre_enrichment.py` (append new tests)

- [ ] **Step 1: Write failing test for `fetch_bandcamp_tags` happy path**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_fetch_bandcamp_tags_uses_source_locator_and_extractor(monkeypatch):
    from src.ai_genre_enrichment import bandcamp_enrichment

    fake_locator_response = {
        "candidate_sources": [
            {
                "source_url": "https://duster.bandcamp.com/album/stratosphere",
                "source_type": "bandcamp_release",
                "source_name": "Bandcamp",
                "identity_status": "confirmed",
                "identity_confidence": 0.95,
                "release_specific": True,
                "reason": "Official artist Bandcamp page",
            }
        ],
        "warnings": [],
    }

    def fake_locate(*, artist, album, model, api_key):
        assert artist == "Duster"
        assert album == "Stratosphere"
        return fake_locator_response

    def fake_fetch_html(url):
        assert url == "https://duster.bandcamp.com/album/stratosphere"
        return (
            '<a class="tag" href="https://bandcamp.com/discover/slowcore">slowcore</a>'
            '<a class="tag" href="https://bandcamp.com/discover/space-rock">space rock</a>'
        )

    monkeypatch.setattr(bandcamp_enrichment, "_locate_bandcamp_url", fake_locate)
    tags = bandcamp_enrichment.fetch_bandcamp_tags(
        artist="Duster",
        album="Stratosphere",
        api_key="test-key",
        model="gpt-4o-mini",
        fetch_html=fake_fetch_html,
    )
    assert tags == ["slowcore", "space rock"]


def test_fetch_bandcamp_tags_returns_empty_when_no_confirmed_url(monkeypatch):
    from src.ai_genre_enrichment import bandcamp_enrichment

    def fake_locate(*, artist, album, model, api_key):
        return {
            "candidate_sources": [
                {
                    "source_url": "https://example.com/x",
                    "source_type": "official_release",
                    "source_name": "x",
                    "identity_status": "ambiguous",
                    "identity_confidence": 0.4,
                    "release_specific": False,
                    "reason": "low confidence",
                }
            ],
            "warnings": [],
        }

    monkeypatch.setattr(bandcamp_enrichment, "_locate_bandcamp_url", fake_locate)
    tags = bandcamp_enrichment.fetch_bandcamp_tags(
        artist="X",
        album="Y",
        api_key="test-key",
        model="gpt-4o-mini",
        fetch_html=lambda url: "",
    )
    assert tags == []
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_fetch_bandcamp_tags_uses_source_locator_and_extractor tests/unit/test_ai_genre_enrichment.py::test_fetch_bandcamp_tags_returns_empty_when_no_confirmed_url -v`
Expected: FAIL — `bandcamp_enrichment` module does not exist.

- [ ] **Step 3: Implement `bandcamp_enrichment.py`**

Create `src/ai_genre_enrichment/bandcamp_enrichment.py`:

```python
"""Slim Bandcamp tag fetcher for the genre enrichment pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from .source_extraction import fetch_bandcamp_release_tags, is_bandcamp_release_url
from .source_locator import SOURCE_LOCATOR_INSTRUCTIONS, source_locator_response_format

logger = logging.getLogger(__name__)

MIN_LOCATOR_CONFIDENCE = 0.7


def fetch_bandcamp_tags(
    *,
    artist: str,
    album: str | None,
    api_key: str,
    model: str = "gpt-4o-mini",
    fetch_html: Callable[[str], str] | None = None,
) -> list[str]:
    """Fetch Bandcamp tags for a release.

    Uses the AI source locator to find a confirmed Bandcamp release URL,
    then scrapes the tag list from the page.

    Returns:
        List of raw tag strings (deduplicated). Empty if no confirmed Bandcamp URL found.
    """
    locator_response = _locate_bandcamp_url(
        artist=artist, album=album, model=model, api_key=api_key
    )
    url = _pick_bandcamp_url(locator_response)
    if not url:
        return []
    try:
        tags = fetch_bandcamp_release_tags(url, fetch_html=fetch_html)
    except OSError:
        logger.exception("Bandcamp HTML fetch failed for %s", url)
        return []

    seen: set[str] = set()
    filtered: list[str] = []
    for tag in tags:
        key = tag.strip().casefold()
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            filtered.append(tag.strip())
    return filtered


def _pick_bandcamp_url(locator_response: dict[str, Any]) -> str | None:
    candidates = locator_response.get("candidate_sources", []) or []
    bandcamp_candidates = [
        c for c in candidates
        if c.get("source_type") == "bandcamp_release"
        and c.get("identity_status") == "confirmed"
        and (c.get("identity_confidence") or 0) >= MIN_LOCATOR_CONFIDENCE
        and is_bandcamp_release_url(c.get("source_url") or "")
    ]
    if not bandcamp_candidates:
        return None
    bandcamp_candidates.sort(key=lambda c: c.get("identity_confidence") or 0, reverse=True)
    return bandcamp_candidates[0]["source_url"]


def _locate_bandcamp_url(
    *, artist: str, album: str | None, model: str, api_key: str
) -> dict[str, Any]:
    """Call OpenAI source locator to find a Bandcamp URL for the release."""
    os.environ["OPENAI_API_KEY"] = api_key  # client picks this up
    from .client import OpenAIEnrichmentClient

    client = OpenAIEnrichmentClient(model=model)
    payload = {"artist": artist, "album": album or ""}
    prompt = f"artist: {artist}\nalbum: {album or ''}"
    result = client.enrich(
        payload,
        prompt,
        source_locator_response_format(),
        instructions=SOURCE_LOCATOR_INSTRUCTIONS,
    )
    if result.status != "complete":
        logger.warning("Source locator failed for %s — %s", artist, result.error_message)
        return {"candidate_sources": [], "warnings": []}
    return result.response_json
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k "bandcamp_tags" -v`
Expected: PASS — both new tests green.

- [ ] **Step 5: Run full enrichment test suite to verify no regressions**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -v`
Expected: PASS — all 121 tests pass (119 prior + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/bandcamp_enrichment.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add bandcamp_enrichment.fetch_bandcamp_tags using source_locator + source_extraction"
```

---

## Task 2: extract-bandcamp CLI Command

**Files:**
- Modify: `scripts/ai_genre_enrich.py:54` (dispatch) and add new function + subparser
- Test: `tests/unit/test_ai_genre_enrichment.py` (append new test)

- [ ] **Step 1: Write failing test for `extract-bandcamp` command**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_extract_bandcamp_command_calls_fetcher_and_stores_tags(monkeypatch, tmp_path):
    import importlib
    import sys
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    _create_test_metadata_db(metadata_db, [("Duster", "Stratosphere")])  # existing helper

    sidecar_db = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar_db))
    store.initialize()

    def fake_fetch(*, artist, album, api_key, model, fetch_html=None):
        assert artist == "duster"
        assert album == "stratosphere"
        return ["slowcore", "space rock", "shoegaze"]

    mod_name = "src.ai_genre_enrichment.bandcamp_enrichment"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    monkeypatch.setattr(
        sys.modules[mod_name] if mod_name in sys.modules else importlib.import_module(mod_name),
        "fetch_bandcamp_tags",
        fake_fetch,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from scripts import ai_genre_enrich
    argv = [
        "extract-bandcamp",
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "--artist", "Duster",
    ]
    monkeypatch.setattr(sys, "argv", ["ai_genre_enrich.py", *argv])
    rc = ai_genre_enrich.main()
    assert rc == 0

    with store.connect() as conn:
        tags = [row["raw_tag"] for row in conn.execute(
            "SELECT raw_tag FROM ai_genre_source_tags ORDER BY raw_tag"
        )]
    assert tags == ["shoegaze", "slowcore", "space rock"]
```

(If `_create_test_metadata_db` doesn't already cover the schema needed, copy from `test_extract_lastfm_command_calls_api` in the same file.)

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_bandcamp_command_calls_fetcher_and_stores_tags -v`
Expected: FAIL — `extract-bandcamp` subcommand not registered.

- [ ] **Step 3: Add subparser registration**

In `scripts/ai_genre_enrich.py`, find the existing `extract_lastfm = sub.add_parser(...)` block (~line 163) and immediately after add:

```python
    extract_bandcamp = sub.add_parser("extract-bandcamp", help="Find Bandcamp URL via AI and ingest release tags")
    add_release_filters(extract_bandcamp)
    extract_bandcamp.add_argument("--dry-run", action="store_true")
    extract_bandcamp.add_argument("--adjudicate", action="store_true", help="Send unknown tags to AI for adjudication")
    extract_bandcamp.add_argument("--model", default=DEFAULT_MODEL)
    extract_bandcamp.add_argument("--openai-api-key", help="OpenAI API key (overrides env/config.yaml)")
```

- [ ] **Step 4: Add dispatch entry**

In `scripts/ai_genre_enrich.py`, find the dispatch chain (~line 33-55) and add (placement consistent with `extract-lastfm`):

```python
    if args.command == "extract-bandcamp":
        return cmd_extract_bandcamp(args)
```

- [ ] **Step 5: Implement `cmd_extract_bandcamp`**

Place after `cmd_extract_lastfm` (~line 588). Mirror its shape closely:

```python
def cmd_extract_bandcamp(args: argparse.Namespace) -> int:
    import os
    from src.ai_genre_enrichment.bandcamp_enrichment import fetch_bandcamp_tags
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.tag_classification import set_vocabulary, reset_vocabulary

    api_key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.config_loader import Config
            api_key = Config().openai_api_key
        except (FileNotFoundError, AttributeError):
            pass
    if not api_key:
        print(
            "Error: OpenAI API key required. "
            "Set OPENAI_API_KEY env var, use --openai-api-key, or configure in config.yaml."
        )
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
    try:
        for release in releases:
            album_name = release.normalized_album or None
            tags = fetch_bandcamp_tags(
                artist=release.normalized_artist,
                album=album_name,
                api_key=api_key,
                model=args.model,
            )
            if not tags:
                continue

            if getattr(args, "dry_run", False):
                print(json.dumps({
                    "release_key": release.release_key,
                    "bandcamp_tags": tags,
                    "dry_run": True,
                }, ensure_ascii=False, sort_keys=True))
                continue

            album_segment = f"/album/{release.normalized_album}" if release.normalized_album else ""
            page_id = store.upsert_source_page(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                source_url=f"bandcamp://artist/{release.normalized_artist}{album_segment}",
                source_type="bandcamp_tags",
                identity_status="confirmed",
                identity_confidence=0.9,
                evidence_summary="Bandcamp release tags via AI source locator.",
            )
            store.replace_source_tags(page_id, tags)
            store.classify_source_tags(page_id, adjudicate=getattr(args, "adjudicate", False), model=args.model)
            store.rebuild_enriched_genres_for_release(release.release_key)
            extracted += 1
            print(f"extracted-bandcamp {release.release_key} tags={len(tags)}")
    finally:
        reset_vocabulary()

    print(f"Extracted Bandcamp tags for {extracted} release(s).")
    return 0
```

- [ ] **Step 6: Run test to verify pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_bandcamp_command_calls_fetcher_and_stores_tags -v`
Expected: PASS.

- [ ] **Step 7: Manual smoke test (optional)**

```bash
python scripts/ai_genre_enrich.py extract-bandcamp --artist "Duster" --dry-run
```
Expected: prints JSON with `bandcamp_tags` array or "No matching release found" if Duster not in metadata.db.

- [ ] **Step 8: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add extract-bandcamp CLI command using AI source locator"
```

---

## Task 3: EnrichedGenreResolver

**Files:**
- Create: `src/ai_genre_enrichment/genre_resolver.py`
- Test: `tests/unit/test_ai_genre_enrichment.py` (append new tests)

- [ ] **Step 1: Write failing tests for the resolver**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_resolver_returns_enriched_genres_when_present(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                "duster::stratosphere",
                "duster",
                "stratosphere",
                None,
                '{"genres": ["slowcore", "space rock", "shoegaze"], "sources": []}',
                "2026-05-28T00:00:00",
            ),
        )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    genres = resolver.get_enriched_genres(artist="Duster", album="Stratosphere")
    assert genres == ["slowcore", "space rock", "shoegaze"]


def test_resolver_returns_none_when_unenriched(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()

    resolver = EnrichedGenreResolver(str(sidecar))
    assert resolver.get_enriched_genres(artist="Unknown", album="Album") is None


def test_resolver_normalizes_artist_and_album(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "sigur ros::átta",
                "sigur ros",
                "átta",
                None,
                '{"genres": ["ambient", "post-rock"], "sources": []}',
                "2026-05-28T00:00:00",
            ),
        )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    assert resolver.get_enriched_genres(artist="Sigur Rós", album="Átta") == ["ambient", "post-rock"]


def test_resolver_artist_status_reports_per_album(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        for album in ["stratosphere", "together"]:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    f"duster::{album}",
                    "duster",
                    album,
                    None,
                    '{"genres": ["slowcore"], "sources": []}',
                    "2026-05-28T00:00:00",
                ),
            )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    status = resolver.get_artist_enrichment_status("Duster")
    assert status["enriched_count"] == 2
    assert sorted(status["enriched_albums"]) == ["stratosphere", "together"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k "resolver" -v`
Expected: FAIL — `EnrichedGenreResolver` does not exist.

- [ ] **Step 3: Implement the resolver**

Create `src/ai_genre_enrichment/genre_resolver.py`:

```python
"""Read-only access to enriched genre signatures from the sidecar DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .tag_classification import normalize_source_tag


class EnrichedGenreResolver:
    """Resolves enriched genres for a (artist, album) tuple.

    Opens the sidecar DB read-only. Returns None when no enriched signature
    exists for the release — callers are expected to fall back to raw metadata.
    """

    def __init__(self, sidecar_db_path: str | Path):
        self._db_path = Path(sidecar_db_path).resolve()

    def get_enriched_genres(self, *, artist: str, album: str | None) -> list[str] | None:
        if not album:
            return None
        release_key = self._release_key(artist, album)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
                (release_key,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["signature_json"])
        genres = payload.get("genres") or []
        return list(genres) if genres else None

    def is_enriched(self, *, artist: str, album: str | None) -> bool:
        return self.get_enriched_genres(artist=artist, album=album) is not None

    def get_artist_enrichment_status(self, artist: str) -> dict:
        """Return enrichment status for an artist.

        Result keys: enriched_count (int), enriched_albums (list[str] — normalized album names).
        """
        normalized_artist = normalize_source_tag(artist)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT normalized_album FROM enriched_genre_signatures "
                "WHERE normalized_artist = ? ORDER BY normalized_album",
                (normalized_artist,),
            ).fetchall()
        albums = [row["normalized_album"] for row in rows]
        return {"enriched_count": len(albums), "enriched_albums": albums}

    def _release_key(self, artist: str, album: str) -> str:
        return f"{normalize_source_tag(artist)}::{normalize_source_tag(album)}"

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            # Return a connection to an in-memory empty DB so callers always get None
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE enriched_genre_signatures(release_key TEXT, "
                "normalized_artist TEXT, normalized_album TEXT, signature_json TEXT)"
            )
            return conn
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k "resolver" -v`
Expected: PASS — 4 new tests.

- [ ] **Step 5: Full enrichment suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -v`
Expected: PASS — all tests including new resolver tests.

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/genre_resolver.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add EnrichedGenreResolver for read-only enriched genre lookup"
```

---

## Task 4: GUI Playlist Viewer Uses Resolver

**Files:**
- Modify: `src/playlist_gui/worker.py:1242-1290` (genres population)
- Modify: `src/playlist_gui/worker.py:750-760` (preview pool genres)
- Test: `tests/unit/test_playlist_gui_genre_resolver.py` (new)

- [ ] **Step 1: Write failing integration test**

Create `tests/unit/test_playlist_gui_genre_resolver.py`:

```python
"""Test that the GUI worker uses EnrichedGenreResolver to populate playlist genres."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _seed_sidecar(sidecar_path: Path) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "duster::stratosphere",
                "duster",
                "stratosphere",
                None,
                json.dumps({"genres": ["slowcore", "space rock"], "sources": []}),
                "2026-05-28T00:00:00",
            ),
        )
        conn.commit()


def test_resolve_track_genres_prefers_enriched(tmp_path):
    """When an album is enriched, _resolve_track_genres returns enriched genres."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)

    track = {"artist": "Duster", "album": "Stratosphere", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]  # noqa: E731

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["slowcore", "space rock"]


def test_resolve_track_genres_falls_back_when_unenriched(tmp_path):
    """When an album is not enriched, falls back to the provided source."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar_path)

    track = {"artist": "Unknown", "album": "Album", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]  # noqa: E731

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["indie rock", "rock"]


def test_resolve_track_genres_no_sidecar_uses_fallback(tmp_path):
    """When sidecar DB doesn't exist, always falls back."""
    from src.playlist_gui import worker

    sidecar_path = tmp_path / "nonexistent.db"  # never created

    track = {"artist": "Duster", "album": "Stratosphere", "rating_key": "t1"}
    fallback = lambda: ["indie rock", "rock"]  # noqa: E731

    result = worker._resolve_track_genres(
        track,
        sidecar_db_path=str(sidecar_path),
        fallback=fallback,
    )
    assert result == ["indie rock", "rock"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_playlist_gui_genre_resolver.py -v`
Expected: FAIL — `_resolve_track_genres` not defined in worker.

- [ ] **Step 3: Add `_resolve_track_genres` helper to worker.py**

In `src/playlist_gui/worker.py`, add near the other helper functions (around the existing `_top_genres_for_index` definition, ~line 750):

```python
def _resolve_track_genres(
    track: Dict[str, Any],
    *,
    sidecar_db_path: str,
    fallback: callable,
) -> List[str]:
    """Return enriched genres if available for (artist, album), else call fallback.

    fallback is a no-arg callable returning the raw genres list.
    """
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    artist = track.get("artist") or ""
    album = track.get("album") or ""
    if not artist or not album:
        return list(fallback() or [])
    resolver = EnrichedGenreResolver(sidecar_db_path)
    enriched = resolver.get_enriched_genres(artist=artist, album=album)
    if enriched:
        return enriched
    return list(fallback() or [])
```

(Place near the top of worker.py with other utility helpers, before any `class` definitions.)

- [ ] **Step 4: Define the sidecar DB path constant**

In `src/playlist_gui/worker.py`, find existing constants (top of file) and add:

```python
SIDECAR_DB_PATH = "data/ai_genre_enrichment.db"
```

- [ ] **Step 5: Wire helper into `formatted_tracks` build (~line 1242)**

Replace the existing genres-resolution block in the `formatted_tracks` loop. Find:

```python
            for i, track in enumerate(tracks, 1):
                genres = track.get('genres', [])
                rating_key = track.get('rating_key') or track.get('id') or track.get('track_id')
                # Fill genres lazily from similarity calculator if missing
                if (not genres) and rating_key and getattr(generator, "similarity_calc", None):
                    try:
                        genres = generator.similarity_calc.get_filtered_combined_genres_for_track(str(rating_key)) or []
                    except Exception:
                        genres = track.get('genres', []) or []
```

Replace with:

```python
            for i, track in enumerate(tracks, 1):
                rating_key = track.get('rating_key') or track.get('id') or track.get('track_id')

                def _raw_genres() -> List[str]:
                    raw = track.get('genres', []) or []
                    if raw:
                        return raw
                    if rating_key and getattr(generator, "similarity_calc", None):
                        try:
                            return generator.similarity_calc.get_filtered_combined_genres_for_track(str(rating_key)) or []
                        except Exception:
                            return []
                    return []

                genres = _resolve_track_genres(
                    track,
                    sidecar_db_path=SIDECAR_DB_PATH,
                    fallback=_raw_genres,
                )
```

- [ ] **Step 6: Wire helper into `_top_genres_for_index` callers (~line 754)**

Find:

```python
                "genres": _top_genres_for_index(cache, int(entry["index"]), limit=3),
```

This is in a different code path (preview pool, not playlist build). Since the candidate-pool integration is out of scope per the plan's scope note, leave this call site untouched — it will continue using `_top_genres_for_index`. Add a comment marking it:

```python
                # NOTE: preview pool still uses raw genres; full pool-level enrichment integration is Phase 2.
                "genres": _top_genres_for_index(cache, int(entry["index"]), limit=3),
```

- [ ] **Step 7: Run new tests to verify pass**

Run: `pytest tests/unit/test_playlist_gui_genre_resolver.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 8: Run full test suite to verify no regressions**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_playlist_gui_genre_resolver.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/playlist_gui/worker.py tests/unit/test_playlist_gui_genre_resolver.py
git commit -m "feat: playlist viewer uses EnrichedGenreResolver for genre column"
```

---

## Task 5: EnrichmentPanel Widget

**Files:**
- Create: `src/playlist_gui/widgets/enrichment_panel.py`
- Test: `tests/unit/test_enrichment_panel.py` (new)

- [ ] **Step 1: Write failing test for panel rendering**

Create `tests/unit/test_enrichment_panel.py`:

```python
"""Test EnrichmentPanel: status display and enrich button."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def _seed_sidecar(sidecar_path: Path, artist: str, albums: list[str]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as conn:
        for album in albums:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    f"{artist}::{album}",
                    artist,
                    album,
                    None,
                    json.dumps({"genres": ["slowcore"], "sources": []}),
                    "2026-05-28T00:00:00",
                ),
            )
        conn.commit()


def test_panel_shows_enrichment_status_for_artist(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", ["stratosphere", "together"])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")

    assert panel.artist_label.text() == "Duster"
    assert "2 album" in panel.status_label.text().lower()


def test_panel_emits_enrich_requested_when_button_clicked(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", [])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")

    received: list[str] = []
    panel.enrich_requested.connect(lambda artist: received.append(artist))
    panel.enrich_button.click()

    assert received == ["Duster"]


def test_panel_disables_button_while_running(tmp_path, app):
    from src.playlist_gui.widgets.enrichment_panel import EnrichmentPanel

    sidecar = tmp_path / "sidecar.db"
    _seed_sidecar(sidecar, "duster", [])

    panel = EnrichmentPanel(sidecar_db_path=str(sidecar))
    panel.set_artist("Duster")
    panel.set_running(True)
    assert not panel.enrich_button.isEnabled()
    panel.set_running(False)
    assert panel.enrich_button.isEnabled()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_enrichment_panel.py -v`
Expected: FAIL — `EnrichmentPanel` does not exist.

- [ ] **Step 3: Implement `EnrichmentPanel`**

Create `src/playlist_gui/widgets/enrichment_panel.py`:

```python
"""Panel showing per-artist genre enrichment status and triggering enrichment runs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


class EnrichmentPanel(QWidget):
    """Compact panel: shows artist + enrichment status + Enrich button.

    Signals:
        enrich_requested(str): User clicked Enrich. Argument is the artist name.
    """

    enrich_requested = Signal(str)

    def __init__(self, *, sidecar_db_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._sidecar_db_path = sidecar_db_path
        self._artist: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.artist_label = QLabel("(no artist)")
        self.artist_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.artist_label)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.enrich_button = QPushButton("Enrich genres")
        self.enrich_button.clicked.connect(self._on_enrich_clicked)
        button_row.addStretch(1)
        button_row.addWidget(self.enrich_button)
        layout.addLayout(button_row)

    def set_artist(self, artist: str) -> None:
        self._artist = artist
        self.artist_label.setText(artist or "(no artist)")
        self._refresh_status()

    def set_running(self, running: bool) -> None:
        self.enrich_button.setEnabled(not running)
        if running:
            self.status_label.setText("Enriching...")

    def refresh(self) -> None:
        """Re-read enrichment status from the sidecar."""
        self._refresh_status()

    def _refresh_status(self) -> None:
        if not self._artist:
            self.status_label.setText("")
            return
        resolver = EnrichedGenreResolver(self._sidecar_db_path)
        status = resolver.get_artist_enrichment_status(self._artist)
        count = status["enriched_count"]
        if count == 0:
            self.status_label.setText("Not enriched")
        else:
            self.status_label.setText(f"{count} album(s) enriched")

    def _on_enrich_clicked(self) -> None:
        if self._artist:
            self.enrich_requested.emit(self._artist)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_enrichment_panel.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/widgets/enrichment_panel.py tests/unit/test_enrichment_panel.py
git commit -m "feat: add EnrichmentPanel widget for per-artist enrichment status"
```

---

## Task 6: Worker `enrich_artist` Command

**Files:**
- Modify: `src/playlist_gui/worker.py` (add command handler + dispatch)
- Modify: `src/playlist_gui/worker_client.py` (add client method)
- Test: `tests/unit/test_worker_enrich_artist.py` (new)

- [ ] **Step 1: Write failing test for the worker command handler**

Create `tests/unit/test_worker_enrich_artist.py`:

```python
"""Test worker.enrich_artist command runs CLI pipeline via subprocess."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


def test_handle_enrich_artist_runs_pipeline_steps_in_order():
    from src.playlist_gui.worker import handle_enrich_artist

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.playlist_gui.worker.subprocess.run", return_value=completed) as mock_run:
        result = handle_enrich_artist(artist="Duster", request_id="req-1")

    assert result["ok"] is True
    # Five CLI invocations in order:
    expected_commands = [
        "ingest-local",
        "extract-lastfm",
        "extract-bandcamp",
        "classify-tags",
        "build-enriched",
    ]
    actual_commands = []
    for call in mock_run.call_args_list:
        argv = call.args[0]
        # argv = ["python", "scripts/ai_genre_enrich.py", "<command>", ...]
        actual_commands.append(argv[2])
    assert actual_commands == expected_commands


def test_handle_enrich_artist_stops_on_first_failure():
    from src.playlist_gui.worker import handle_enrich_artist

    def fake_run(argv, **kwargs):
        result = MagicMock()
        result.returncode = 0 if argv[2] == "ingest-local" else 1
        result.stdout = ""
        result.stderr = "boom"
        return result

    with patch("src.playlist_gui.worker.subprocess.run", side_effect=fake_run) as mock_run:
        result = handle_enrich_artist(artist="Duster", request_id="req-2")

    assert result["ok"] is False
    assert "extract-lastfm" in result.get("error", "")
    # First two calls only (ingest-local succeeded, extract-lastfm failed)
    assert len(mock_run.call_args_list) == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_worker_enrich_artist.py -v`
Expected: FAIL — `handle_enrich_artist` not defined.

- [ ] **Step 3: Add `handle_enrich_artist` to worker.py**

In `src/playlist_gui/worker.py`, add this function (and the `subprocess` import at module level if not already imported):

```python
import subprocess
import sys


def handle_enrich_artist(*, artist: str, request_id: str) -> Dict[str, Any]:
    """Run the full enrichment pipeline for an artist as subprocess invocations.

    Steps: ingest-local → extract-lastfm → extract-bandcamp → classify-tags → build-enriched.
    Stops at the first failure and reports which step failed.
    """
    steps = [
        ("ingest-local", []),
        ("extract-lastfm", []),
        ("extract-bandcamp", []),
        ("classify-tags", ["--adjudicate"]),
        ("build-enriched", []),
    ]
    for i, (command, extra_args) in enumerate(steps, 1):
        argv = [
            sys.executable,
            "scripts/ai_genre_enrich.py",
            command,
            "--artist", artist,
            *extra_args,
        ]
        emit_progress(stage=f"enrich:{command}", current=i, total=len(steps), detail=artist)
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": f"{command} failed: {completed.stderr.strip()[:200]}",
                "step": command,
            }
        emit_log("INFO", f"{command} completed for {artist}", request_id=request_id)
    return {"ok": True, "artist": artist}
```

- [ ] **Step 4: Wire dispatch in worker `main()`**

In `src/playlist_gui/worker.py` find the existing command dispatch in `main()` (~line 1919) and add a branch for `enrich_artist`:

```python
        elif cmd == "enrich_artist":
            artist = command_payload.get("artist", "")
            if not artist:
                emit_error("enrich_artist requires 'artist' field")
            else:
                result = handle_enrich_artist(artist=artist, request_id=request_id)
                if result["ok"]:
                    emit_result("enrich_artist", result)
                else:
                    emit_error(result.get("error", "enrichment failed"))
            emit_done(cmd="enrich_artist", ok=result.get("ok", False), detail=artist)
```

(Match the exact dispatch pattern of an existing command nearby — read the surrounding code at line 1919 first.)

- [ ] **Step 5: Add `enrich_artist` to WorkerClient**

In `src/playlist_gui/worker_client.py`, add a method on `WorkerClient`:

```python
    def enrich_artist(self, artist: str) -> str:
        """Send an enrich_artist command to the worker. Returns the request_id."""
        request_id = str(uuid.uuid4())
        command = {
            "cmd": "enrich_artist",
            "request_id": request_id,
            "protocol_version": PROTOCOL_VERSION,
            "artist": artist,
        }
        self._send_command(command)
        return request_id
```

(Use the existing `_send_command` pattern — read other command methods like `generate_playlist` first to match the exact dispatch.)

- [ ] **Step 6: Run tests to verify pass**

Run: `pytest tests/unit/test_worker_enrich_artist.py -v`
Expected: PASS — both tests.

- [ ] **Step 7: Commit**

```bash
git add src/playlist_gui/worker.py src/playlist_gui/worker_client.py tests/unit/test_worker_enrich_artist.py
git commit -m "feat: add worker enrich_artist command running CLI pipeline subprocesses"
```

---

## Task 7: Wire EnrichmentPanel into MainWindow

**Files:**
- Modify: `src/playlist_gui/main_window.py`
- No new tests (UI wiring; covered by panel + worker tests in Tasks 5 & 6)

- [ ] **Step 1: Import the panel**

In `src/playlist_gui/main_window.py`, near existing widget imports (~line 57), add:

```python
from .widgets.enrichment_panel import EnrichmentPanel
```

- [ ] **Step 2: Construct the panel in `__init__`**

Find where other side-panel widgets are constructed (look around the `_track_table` construction at ~line 317). Add the panel adjacent to where the user selects an artist (e.g., near the autocomplete-driven artist input):

```python
        self._enrichment_panel = EnrichmentPanel(sidecar_db_path="data/ai_genre_enrichment.db")
        self._enrichment_panel.enrich_requested.connect(self._on_enrich_requested)
```

Place the widget into the existing controls layout — read the layout setup near line 279 to choose the right container.

- [ ] **Step 3: Connect WorkerClient signals to panel state**

In `MainWindow.__init__` after the worker client is constructed (search for `WorkerClient(` in the file), add:

```python
        self._worker_client.busy_changed.connect(self._on_enrichment_busy_changed)
        self._worker_client.result_received.connect(self._on_enrichment_result)
```

- [ ] **Step 4: Add the handler methods**

Add to `MainWindow`:

```python
    def _on_enrich_requested(self, artist: str) -> None:
        self._enrichment_panel.set_running(True)
        self._worker_client.enrich_artist(artist)

    def _on_enrichment_busy_changed(self, is_busy: bool) -> None:
        if not is_busy:
            self._enrichment_panel.set_running(False)
            self._enrichment_panel.refresh()

    def _on_enrichment_result(self, result_type: str, data: dict, job_id) -> None:
        if result_type == "enrich_artist":
            self._enrichment_panel.refresh()
```

- [ ] **Step 5: Hook artist autocomplete selection to update panel**

If the GUI already has a signal when the user picks an artist (search for `artist_selected`, `_on_artist_changed`, or look at how `DatabaseCompleter` is wired ~line 143), connect it to `self._enrichment_panel.set_artist(artist)`. If no such signal exists, add a manual trigger to the artist input's `textChanged` or `editingFinished`:

```python
        # In the input setup block:
        artist_input.editingFinished.connect(
            lambda: self._enrichment_panel.set_artist(artist_input.text())
        )
```

(Adapt to the actual artist input widget name — read line 143 onwards to find it.)

- [ ] **Step 6: Manual smoke test**

Launch the GUI:

```bash
python -m playlist_gui.app
```

Expected: The new panel appears next to the artist input. Typing an artist updates the status label. Clicking Enrich disables the button and shows "Enriching...". When the subprocess completes the status refreshes with the new enriched album count.

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests/unit/ -v --tb=short`
Expected: PASS — no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/playlist_gui/main_window.py
git commit -m "feat: wire EnrichmentPanel into MainWindow with worker enrich_artist"
```

---

## Task 8: Sidecar-Aware `get_tracks_for_genre`

**Files:**
- Modify: `src/ai_genre_enrichment/genre_resolver.py` (add reverse lookups)
- Modify: `src/local_library_client.py:22-42` (accept resolver), `:256-329` (filter and union)
- Test: `tests/unit/test_ai_genre_enrichment.py` (resolver reverse lookups)
- Test: `tests/unit/test_local_library_client_enriched.py` (NEW)

### Design

Two cases per release:
1. **Enriched release** → include track iff enriched signature contains the queried genre
2. **Unenriched release** → include track iff raw metadata UNION matches

Algorithm in `get_tracks_for_genre(genre, limit)`:

```
enriched_with_genre = resolver.get_release_keys_with_genre(genre)   # release_keys to INCLUDE
enriched_all        = resolver.get_all_enriched_release_keys()      # release_keys to EXCLUDE from raw

raw_tracks = existing UNION query
filtered_raw = [t for t in raw_tracks if release_key(t) not in enriched_all]

enriched_tracks = SELECT tracks WHERE release_key IN enriched_with_genre
                  (computed by joining tracks to (normalized_artist, normalized_album))

return dedupe(filtered_raw + enriched_tracks, key=track_id)[:limit]
```

`release_key(track) = normalize(track.artist) + "::" + normalize(track.album)` using `normalize_source_tag` from `tag_classification`.

The reverse lookup caches a `{genre → set[release_key]}` index built once per resolver instance — enriched data is small (low thousands of releases) so memory cost is negligible.

- [ ] **Step 1: Write failing tests for the new resolver methods**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_resolver_release_keys_with_genre(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        for release_key, normalized_artist, normalized_album, genres in [
            ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "space rock"]),
            ("duster::together", "duster", "together", ["slowcore", "shoegaze"]),
            ("sigur ros::átta", "sigur ros", "átta", ["ambient", "post-rock"]),
        ]:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    release_key,
                    normalized_artist,
                    normalized_album,
                    None,
                    f'{{"genres": {genres!r}, "sources": []}}'.replace("'", '"'),
                    "2026-05-28T00:00:00",
                ),
            )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    assert resolver.get_release_keys_with_genre("slowcore") == {"duster::stratosphere", "duster::together"}
    assert resolver.get_release_keys_with_genre("ambient") == {"sigur ros::átta"}
    assert resolver.get_release_keys_with_genre("nonexistent") == set()


def test_resolver_all_enriched_release_keys(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            ("duster::stratosphere", "duster", "stratosphere", None,
             '{"genres": ["slowcore"], "sources": []}', "2026-05-28T00:00:00"),
        )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    assert resolver.get_all_enriched_release_keys() == {"duster::stratosphere"}


def test_resolver_reverse_index_is_case_normalized(tmp_path):
    """Reverse lookups index by the casefold-normalized genre, matching how queries arrive."""
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            ("duster::stratosphere", "duster", "stratosphere", None,
             '{"genres": ["Slowcore", "Space Rock"], "sources": []}', "2026-05-28T00:00:00"),
        )
        conn.commit()

    resolver = EnrichedGenreResolver(str(sidecar))
    # Query with lowercase — matches even though signature has mixed case
    assert resolver.get_release_keys_with_genre("slowcore") == {"duster::stratosphere"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k "reverse_index or release_keys" -v`
Expected: FAIL — methods not defined.

- [ ] **Step 3: Extend `EnrichedGenreResolver` with reverse lookups**

In `src/ai_genre_enrichment/genre_resolver.py`, add to the class:

```python
    def get_release_keys_with_genre(self, genre: str) -> set[str]:
        """Return the set of enriched release_keys whose signature contains the given genre."""
        index = self._build_reverse_index()
        return index.get(genre.casefold(), set())

    def get_all_enriched_release_keys(self) -> set[str]:
        """Return the set of all release_keys with an enriched signature."""
        if self._all_enriched_cache is None:
            with self._connect() as conn:
                rows = conn.execute("SELECT release_key FROM enriched_genre_signatures").fetchall()
            self._all_enriched_cache = {row["release_key"] for row in rows}
        return self._all_enriched_cache

    def _build_reverse_index(self) -> dict[str, set[str]]:
        if self._reverse_index_cache is not None:
            return self._reverse_index_cache
        index: dict[str, set[str]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT release_key, signature_json FROM enriched_genre_signatures"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["signature_json"])
            for genre in (payload.get("genres") or []):
                index.setdefault(genre.casefold(), set()).add(row["release_key"])
        self._reverse_index_cache = index
        return index
```

And initialize the caches in `__init__`:

```python
    def __init__(self, sidecar_db_path: str | Path):
        self._db_path = Path(sidecar_db_path).resolve()
        self._reverse_index_cache: dict[str, set[str]] | None = None
        self._all_enriched_cache: set[str] | None = None
```

- [ ] **Step 4: Run resolver tests to verify pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k "reverse_index or release_keys" -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Write failing test for sidecar-aware `get_tracks_for_genre`**

Create `tests/unit/test_local_library_client_enriched.py`:

```python
"""Test that LocalLibraryClient.get_tracks_for_genre honors enriched signatures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_metadata_db(path: Path, rows: list[dict]) -> None:
    """Create a minimal metadata.db with tracks + raw genre tables."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            artist_key TEXT,
            title TEXT,
            album TEXT,
            album_id TEXT,
            duration_ms INTEGER,
            file_path TEXT,
            musicbrainz_id TEXT,
            is_blacklisted INTEGER DEFAULT 0
        );
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO tracks(track_id, artist, artist_key, title, album, album_id, "
            "duration_ms, file_path, musicbrainz_id, is_blacklisted) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (r["track_id"], r["artist"], r["artist_key"], r["title"], r["album"],
             r["album_id"], 200000, f"/music/{r['track_id']}.mp3", None),
        )
        for g in r.get("album_genres", []):
            conn.execute("INSERT INTO album_genres(album_id, genre) VALUES(?, ?)", (r["album_id"], g))
        for g in r.get("artist_genres", []):
            conn.execute("INSERT INTO artist_genres(artist, genre) VALUES(?, ?)", (r["artist"], g))
    conn.commit()
    conn.close()


def _make_sidecar(path: Path, signatures: list[tuple[str, str, str, list[str]]]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore
    store = SidecarStore(str(path))
    store.initialize()
    with store.connect() as conn:
        for release_key, normalized_artist, normalized_album, genres in signatures:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (release_key, normalized_artist, normalized_album, None,
                 json.dumps({"genres": genres, "sources": []}), "2026-05-28T00:00:00"),
            )
        conn.commit()


def test_get_tracks_for_genre_excludes_enriched_release_when_genre_not_in_signature(tmp_path):
    """Stratosphere has raw 'indie rock' but enriched signature lacks it → exclude."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Duster", "artist_key": "duster",
         "title": "Topical Solution", "album": "Stratosphere", "album_id": "a1",
         "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere",
         ["slowcore", "space rock"]),  # NO "indie rock"
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("indie rock")
    assert tracks == []


def test_get_tracks_for_genre_includes_enriched_release_when_genre_in_signature(tmp_path):
    """Stratosphere's raw lacks 'slowcore' but enriched signature has it → include."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Duster", "artist_key": "duster",
         "title": "Topical Solution", "album": "Stratosphere", "album_id": "a1",
         "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere",
         ["slowcore", "space rock"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("slowcore")
    assert len(tracks) == 1
    assert tracks[0]["title"] == "Topical Solution"


def test_get_tracks_for_genre_unenriched_release_uses_raw_query(tmp_path):
    """An album with no enriched signature falls back to the existing UNION query."""
    from src.local_library_client import LocalLibraryClient
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "Random", "artist_key": "random",
         "title": "Song", "album": "Album", "album_id": "a1",
         "album_genres": ["indie rock"]},
    ])

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])  # no enrichment

    resolver = EnrichedGenreResolver(str(sidecar))
    client = LocalLibraryClient(db_path=str(metadata), enriched_resolver=resolver)
    tracks = client.get_tracks_for_genre("indie rock")
    assert len(tracks) == 1
    assert tracks[0]["title"] == "Song"


def test_get_tracks_for_genre_no_resolver_uses_raw_only(tmp_path):
    """Backwards compatible: no resolver → existing behavior."""
    from src.local_library_client import LocalLibraryClient

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata, [
        {"track_id": "t1", "artist": "X", "artist_key": "x", "title": "T",
         "album": "A", "album_id": "a1", "album_genres": ["rock"]},
    ])

    client = LocalLibraryClient(db_path=str(metadata))  # no resolver
    tracks = client.get_tracks_for_genre("rock")
    assert len(tracks) == 1
```

- [ ] **Step 6: Run tests to verify failure**

Run: `pytest tests/unit/test_local_library_client_enriched.py -v`
Expected: FAIL — `LocalLibraryClient.__init__` doesn't accept `enriched_resolver`.

- [ ] **Step 7: Modify `LocalLibraryClient.__init__` to accept resolver**

In `src/local_library_client.py`, modify the constructor:

```python
    def __init__(
        self,
        db_path: str = "data/metadata.db",
        *,
        enriched_resolver: "EnrichedGenreResolver | None" = None,
    ):
        """
        Initialize local library client

        Args:
            db_path: Path to metadata database
            enriched_resolver: Optional EnrichedGenreResolver; when provided,
                get_tracks_for_genre uses enriched signatures as the
                authoritative source for processed releases.
        """
        self.db_path = db_path
        self.conn = None
        self._enriched_resolver = enriched_resolver
        self.similarity_calc = SimilarityCalculator(db_path)
        self._init_db_connection()
        logger.info("Initialized LocalLibraryClient (local library mode)")
```

(Add the import for type hint at top of file: `from .ai_genre_enrichment.genre_resolver import EnrichedGenreResolver` inside `if TYPE_CHECKING:` block, or just use string annotation as shown.)

- [ ] **Step 8: Rewrite `get_tracks_for_genre` to honor enriched signatures**

Replace the body of `get_tracks_for_genre` (lines 256-329) with:

```python
    def get_tracks_for_genre(self, genre: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get tracks matching a specific genre.

        When an enriched_resolver is configured, enriched signatures are the
        authoritative source: tracks on enriched releases are included iff the
        enriched signature contains the genre; tracks on unenriched releases
        fall back to the UNION query over raw metadata tables.

        Args:
            genre: Normalized genre string (already lowercase)
            limit: Maximum number of tracks to return

        Returns:
            List of track dictionaries
        """
        raw_tracks = self._raw_tracks_for_genre(genre, limit)

        if self._enriched_resolver is None:
            return raw_tracks

        from .ai_genre_enrichment.tag_classification import normalize_source_tag

        enriched_with_genre = self._enriched_resolver.get_release_keys_with_genre(genre)
        enriched_all = self._enriched_resolver.get_all_enriched_release_keys()

        def release_key(track: Dict[str, Any]) -> str:
            return f"{normalize_source_tag(track['artist'] or '')}::{normalize_source_tag(track['album'] or '')}"

        # 1. Keep raw tracks whose release is NOT enriched (authoritative fallback)
        filtered_raw = [t for t in raw_tracks if release_key(t) not in enriched_all]

        # 2. Add tracks from enriched releases whose enriched signature contains the genre
        enriched_tracks = self._tracks_for_release_keys(enriched_with_genre, limit)

        # 3. Dedupe by track_id (raw query and enriched query may overlap when enriched is in enriched_with_genre)
        seen: set[str] = {t["rating_key"] for t in filtered_raw}
        combined = list(filtered_raw)
        for t in enriched_tracks:
            if t["rating_key"] not in seen:
                combined.append(t)
                seen.add(t["rating_key"])

        logger.debug(
            "Found %d tracks for genre '%s' (raw=%d, enriched=%d, filtered=%d)",
            len(combined), genre, len(raw_tracks), len(enriched_tracks), len(raw_tracks) - len(filtered_raw),
        )
        return combined[:limit]

    def _raw_tracks_for_genre(self, genre: str, limit: int) -> List[Dict[str, Any]]:
        """Original UNION query over raw metadata tables."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   1 as priority
            FROM tracks t
            JOIN track_effective_genres teg ON t.track_id = teg.track_id
            WHERE teg.genre = ? AND t.file_path IS NOT NULL AND t.is_blacklisted = 0

            UNION

            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   2 as priority
            FROM tracks t
            JOIN album_genres ag ON t.album_id = ag.album_id
            WHERE ag.genre = ? AND t.file_path IS NOT NULL AND t.is_blacklisted = 0

            UNION

            SELECT t.track_id as rating_key,
                   t.artist, t.artist_key, t.title, t.album,
                   t.duration_ms as duration, t.file_path, t.musicbrainz_id as mbid,
                   3 as priority
            FROM tracks t
            JOIN artist_genres ag ON t.artist = ag.artist
            WHERE ag.genre = ? AND t.file_path IS NOT NULL AND t.is_blacklisted = 0

            ORDER BY priority ASC, rating_key ASC
            LIMIT ?
        """, (genre, genre, genre, limit))

        tracks = []
        for row in cursor.fetchall():
            artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
            tracks.append({
                'rating_key': row['rating_key'],
                'artist': row['artist'] or '',
                'artist_key': artist_key,
                'title': row['title'] or '',
                'album': row['album'] or '',
                'duration': row['duration'],
                'file_path': row['file_path'],
                'mbid': row['mbid'],
            })
        return tracks

    def _tracks_for_release_keys(self, release_keys: set, limit: int) -> List[Dict[str, Any]]:
        """Fetch tracks for the given set of normalized (artist::album) release keys."""
        if not release_keys:
            return []
        from .ai_genre_enrichment.tag_classification import normalize_source_tag

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT track_id, artist, artist_key, title, album, duration_ms, file_path, musicbrainz_id
            FROM tracks
            WHERE file_path IS NOT NULL AND is_blacklisted = 0
            LIMIT ?
        """, (limit * 10,))  # over-fetch since we filter in Python

        tracks = []
        for row in cursor.fetchall():
            rk = f"{normalize_source_tag(row['artist'] or '')}::{normalize_source_tag(row['album'] or '')}"
            if rk in release_keys:
                artist_key = row["artist_key"] if "artist_key" in row.keys() and row["artist_key"] else normalize_artist_key(row["artist"] or "")
                tracks.append({
                    'rating_key': row['track_id'],
                    'artist': row['artist'] or '',
                    'artist_key': artist_key,
                    'title': row['title'] or '',
                    'album': row['album'] or '',
                    'duration': row['duration_ms'],
                    'file_path': row['file_path'],
                    'mbid': row['musicbrainz_id'],
                })
                if len(tracks) >= limit:
                    break
        return tracks
```

- [ ] **Step 9: Run new tests to verify pass**

Run: `pytest tests/unit/test_local_library_client_enriched.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 10: Wire the resolver at the LocalLibraryClient construction sites**

Find where `LocalLibraryClient` is constructed in the codebase:

```bash
grep -rn "LocalLibraryClient(" src/ --include="*.py" | grep -v test
```

For each construction site (typically in `playlist_generator.py` and the worker), add the resolver:

```python
from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

resolver = EnrichedGenreResolver("data/ai_genre_enrichment.db")
client = LocalLibraryClient(db_path="data/metadata.db", enriched_resolver=resolver)
```

(Adapt to actual paths used at each site.)

- [ ] **Step 11: Run full enrichment + integration test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_local_library_client_enriched.py tests/unit/test_playlist_gui_genre_resolver.py -v`
Expected: PASS — all tests.

- [ ] **Step 12: Run full test suite to verify no regressions**

Run: `pytest tests/unit/ -v --tb=short`
Expected: PASS — no regressions in unrelated tests.

- [ ] **Step 13: Manual smoke test**

```bash
python main_app.py --artist "Duster" --tracks 30
```

Expected: Playlist generates successfully. The candidate pool for genres found in Duster's enriched signature (slowcore, space rock, shoegaze) should pull in correctly-tagged tracks; tracks on enriched albums where the enriched signature drops "indie rock" should no longer appear when generating from an indie rock seed.

- [ ] **Step 14: Commit**

```bash
git add src/ai_genre_enrichment/genre_resolver.py src/local_library_client.py tests/unit/test_ai_genre_enrichment.py tests/unit/test_local_library_client_enriched.py
git commit -m "feat: get_tracks_for_genre honors enriched signatures as authoritative source"
```

---

## Task 9: Sidecar-Aware SimilarityCalculator

**Files:**
- Modify: `src/similarity_calculator.py` (init, `_get_combined_genres`, `_get_combined_genres_with_weights`)
- Test: `tests/unit/test_similarity_calc_enriched.py` (new)

### Why

`SimilarityCalculator` is the per-track genre source used by every scoring path: beam-search transition scoring, genre vector building, genre arc, and the GUI fallback when the playlist engine doesn't pre-populate genres. Routing this through the resolver ensures the *scorer* and *pool gating* (Task 8) see the same authoritative genre signal — otherwise the beam approves edges based on raw tags while the pool was gated on enriched.

### Design

`_get_combined_genres(track_id)` and `_get_combined_genres_with_weights(track_id)` are the two private methods that produce the genre list/weights for a track. Both currently read from `metadata.db` via `_get_track_genres`-style lookups. New flow:

```
def _get_combined_genres(track_id):
    if self._enriched_resolver is not None:
        artist, album = self._lookup_artist_album(track_id)
        enriched = self._enriched_resolver.get_enriched_genres(artist=artist, album=album)
        if enriched:
            return enriched  # already noise-filtered, no further filtering needed
    return self._existing_raw_combination(track_id)
```

For the weighted variant, enriched genres get uniform weight `1.0` (the signature is a flat list). Mixed-weight schemes can come later if needed.

`_lookup_artist_album(track_id)` is a small helper that SELECTs `(artist, album)` from `tracks` by `track_id`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_similarity_calc_enriched.py`:

```python
"""Test SimilarityCalculator routes through EnrichedGenreResolver when configured."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_metadata_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            album TEXT,
            album_id TEXT
        );
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    conn.execute("INSERT INTO tracks VALUES('t1', 'Duster', 'Stratosphere', 'a1')")
    conn.execute("INSERT INTO track_genres VALUES('t1', 'indie rock', 'file', 1.0)")
    conn.execute("INSERT INTO album_genres VALUES('a1', 'rock')")
    conn.commit()
    conn.close()


def _make_sidecar(path: Path, signatures: list[tuple]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore
    store = SidecarStore(str(path))
    store.initialize()
    with store.connect() as conn:
        for release_key, normalized_artist, normalized_album, genres in signatures:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (release_key, normalized_artist, normalized_album, None,
                 json.dumps({"genres": genres, "sources": []}), "2026-05-28T00:00:00"),
            )
        conn.commit()


def test_combined_genres_uses_enriched_when_present(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere",
         ["slowcore", "space rock", "shoegaze"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    genres = calc.get_filtered_combined_genres_for_track("t1")
    assert sorted(genres) == ["shoegaze", "slowcore", "space rock"]


def test_combined_genres_falls_back_when_unenriched(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])  # nothing enriched

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    genres = calc.get_filtered_combined_genres_for_track("t1")
    # Raw metadata had "indie rock" (track) and "rock" (album). Both should appear.
    assert "indie rock" in genres
    assert "rock" in genres


def test_weighted_genres_uses_enriched_with_uniform_weight(tmp_path):
    from src.similarity_calculator import SimilarityCalculator
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)

    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere", ["slowcore", "shoegaze"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    calc = SimilarityCalculator(db_path=str(metadata), enriched_resolver=resolver)
    weighted = calc.get_weighted_genres_for_track("t1")
    weights = {g: w for g, w in weighted}
    assert weights == {"slowcore": 1.0, "shoegaze": 1.0}


def test_no_resolver_preserves_existing_behavior(tmp_path):
    """Construction without resolver continues to work as before."""
    from src.similarity_calculator import SimilarityCalculator

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    calc = SimilarityCalculator(db_path=str(metadata))  # no resolver
    genres = calc.get_filtered_combined_genres_for_track("t1")
    assert "indie rock" in genres
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_similarity_calc_enriched.py -v`
Expected: FAIL — `SimilarityCalculator.__init__` doesn't accept `enriched_resolver`.

- [ ] **Step 3: Modify `SimilarityCalculator.__init__`**

In `src/similarity_calculator.py`, change the signature:

```python
    def __init__(
        self,
        db_path: str = "metadata.db",
        config: Dict[str, Any] = None,
        *,
        enriched_resolver: "EnrichedGenreResolver | None" = None,
    ):
        ...existing body...
        self._enriched_resolver = enriched_resolver
```

(Add `from typing import TYPE_CHECKING` then `if TYPE_CHECKING: from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver` if not already imported.)

- [ ] **Step 4: Add `_lookup_artist_album` helper**

In `src/similarity_calculator.py`, add near the other `_get_track_*` helpers:

```python
    def _lookup_artist_album(self, track_id: str) -> tuple[str, str]:
        """Return (artist, album) for a track_id, or empty strings if missing."""
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT artist, album FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        if not row:
            return ("", "")
        return (row[0] or "", row[1] or "")
```

- [ ] **Step 5: Make `_get_combined_genres` resolver-aware**

In `src/similarity_calculator.py`, modify `_get_combined_genres` (line 1029). Wrap the existing body:

```python
    def _get_combined_genres(self, track_id: str) -> List[str]:
        if self._enriched_resolver is not None:
            artist, album = self._lookup_artist_album(track_id)
            if artist and album:
                enriched = self._enriched_resolver.get_enriched_genres(artist=artist, album=album)
                if enriched:
                    return [self._normalize_genre(g) for g in enriched]
        # ...existing body that combines track/album/artist genres + broad-tag filter...
```

(Wrap the existing body in an `else` or just precede it — preserve the existing return path unchanged.)

- [ ] **Step 6: Make `_get_combined_genres_with_weights` resolver-aware**

In `src/similarity_calculator.py`, modify `_get_combined_genres_with_weights` (line 901):

```python
    def _get_combined_genres_with_weights(self, track_id: str) -> Tuple[Tuple[str, float], ...]:
        if self._enriched_resolver is not None:
            artist, album = self._lookup_artist_album(track_id)
            if artist and album:
                enriched = self._enriched_resolver.get_enriched_genres(artist=artist, album=album)
                if enriched:
                    return tuple((self._normalize_genre(g), 1.0) for g in enriched)
        # ...existing body...
```

- [ ] **Step 7: Run tests to verify pass**

Run: `pytest tests/unit/test_similarity_calc_enriched.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 8: Wire resolver at SimilarityCalculator construction sites**

```bash
grep -rn "SimilarityCalculator(" src/ --include="*.py" | grep -v test
```

For each site, pass the resolver. The most common path is via `LocalLibraryClient.__init__` (which constructs `SimilarityCalculator(db_path)` at line 31). Update `LocalLibraryClient.__init__` to forward the resolver:

```python
        self.similarity_calc = SimilarityCalculator(db_path, enriched_resolver=enriched_resolver)
```

- [ ] **Step 9: Full regression suite**

Run: `pytest tests/unit/ -v --tb=short`
Expected: PASS — no regressions in scoring/similarity tests.

- [ ] **Step 10: Commit**

```bash
git add src/similarity_calculator.py src/local_library_client.py tests/unit/test_similarity_calc_enriched.py
git commit -m "feat: SimilarityCalculator routes genre lookups through EnrichedGenreResolver"
```

---

## Task 10: Wire ReviewPanel + Graduate Buttons

**Files:**
- Modify: `src/playlist_gui/widgets/review_panel.py` (add graduate + CLI launcher buttons)
- Modify: `src/playlist_gui/main_window.py` (mount the panel)
- Test: `tests/unit/test_review_panel_graduate.py` (new)

### Why

The existing CLI `review` command works but isn't accessible from the GUI. `review_panel.py` already exists with single-keystroke review for the queue; this task wires it into MainWindow and adds two action buttons:
1. **Graduate to YAML** — runs `graduate-ai` + `graduate-reviewed` via subprocess
2. **Open CLI review** — launches an interactive terminal with `python scripts/ai_genre_enrich.py review` (for power users who prefer the CLI)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_review_panel_graduate.py`:

```python
"""Test the graduate-to-YAML and CLI launcher buttons on ReviewPanel."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_graduate_button_runs_graduate_ai_and_graduate_reviewed(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.playlist_gui.widgets.review_panel.subprocess.run", return_value=completed) as mock_run:
        panel.graduate_button.click()

    commands = [call.args[0][2] for call in mock_run.call_args_list]
    assert "graduate-ai" in commands
    assert "graduate-reviewed" in commands


def test_cli_review_button_spawns_terminal_with_review_command(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    with patch("src.playlist_gui.widgets.review_panel.subprocess.Popen") as mock_popen:
        panel.cli_review_button.click()

    assert mock_popen.call_count == 1
    argv = mock_popen.call_args.args[0]
    assert "review" in argv  # the subcommand is in the argv list


def test_graduate_emits_signal_after_success(tmp_path, app):
    from src.playlist_gui.widgets.review_panel import ReviewPanel

    sidecar = tmp_path / "sidecar.db"
    panel = ReviewPanel(sidecar_db_path=str(sidecar))
    panel.show()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    fired: list[bool] = []
    panel.vocab_graduated.connect(lambda: fired.append(True))

    with patch("src.playlist_gui.widgets.review_panel.subprocess.run", return_value=completed):
        panel.graduate_button.click()

    assert fired == [True]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_review_panel_graduate.py -v`
Expected: FAIL — `graduate_button`, `cli_review_button`, `vocab_graduated` not defined.

- [ ] **Step 3: Add buttons and signal to `ReviewPanel`**

In `src/playlist_gui/widgets/review_panel.py`, add at the top of the file:

```python
import subprocess
import sys
```

Add the new signal next to `review_completed`:

```python
class ReviewPanel(QWidget):
    review_completed = Signal()
    vocab_graduated = Signal()
```

In `_setup_ui`, append after the existing decision buttons:

```python
        # Graduation actions
        action_row = QHBoxLayout()
        self.graduate_button = QPushButton("Graduate to YAML")
        self.graduate_button.setToolTip("Promote AI- and human-reviewed tags into the vocabulary YAML.")
        self.graduate_button.clicked.connect(self._on_graduate_clicked)
        action_row.addWidget(self.graduate_button)

        self.cli_review_button = QPushButton("Open CLI review")
        self.cli_review_button.setToolTip("Launch an interactive terminal for the review CLI.")
        self.cli_review_button.clicked.connect(self._on_cli_review_clicked)
        action_row.addWidget(self.cli_review_button)
        layout.addLayout(action_row)
```

Add the handler methods:

```python
    def _on_graduate_clicked(self) -> None:
        for command in ("graduate-ai", "graduate-reviewed"):
            argv = [sys.executable, "scripts/ai_genre_enrich.py", command]
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                # Surface error via the existing UI label; keep graceful
                if hasattr(self, "_status_label"):
                    self._status_label.setText(f"Graduation step '{command}' failed.")
                return
        self.vocab_graduated.emit()
        self.load_queue()  # refresh after graduation

    def _on_cli_review_clicked(self) -> None:
        argv = [sys.executable, "scripts/ai_genre_enrich.py", "review"]
        # On Windows, spawn a new console window so the user can interact.
        if sys.platform == "win32":
            subprocess.Popen(argv, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # On other platforms, launch in a terminal emulator if available; otherwise detach.
            subprocess.Popen(argv)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_review_panel_graduate.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Mount ReviewPanel in MainWindow**

In `src/playlist_gui/main_window.py`, add:

```python
from .widgets.review_panel import ReviewPanel
```

In `MainWindow.__init__`, construct the panel and place it (as a dock widget or a tab next to `EnrichmentPanel`):

```python
        self._review_panel = ReviewPanel(sidecar_db_path="data/ai_genre_enrichment.db")
        self._review_panel.vocab_graduated.connect(self._on_vocab_graduated)
```

Add the handler:

```python
    def _on_vocab_graduated(self) -> None:
        # After graduation, refresh the enrichment panel and any cached resolver-aware lookups.
        if hasattr(self, "_enrichment_panel"):
            self._enrichment_panel.refresh()
```

Place the panel in the layout (location is a UX call — recommend a side tab or a collapsible dock to keep the main view uncluttered):

```python
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget
        review_dock = QDockWidget("Genre Review", self)
        review_dock.setWidget(self._review_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, review_dock)
        review_dock.setVisible(False)  # opt-in via View menu
```

- [ ] **Step 6: Manual smoke test**

```bash
python -m playlist_gui.app
```

Expected: Review dock available from View menu (or wherever placed). Loading queue shows pending `review_only` tags. Decisions get recorded. Graduate button promotes vocab. CLI review button opens a console (Windows) or detached subprocess (others).

- [ ] **Step 7: Commit**

```bash
git add src/playlist_gui/widgets/review_panel.py src/playlist_gui/main_window.py tests/unit/test_review_panel_graduate.py
git commit -m "feat: wire ReviewPanel into MainWindow with graduate-to-YAML and CLI launcher"
```

---

## Task 11: Resolver-Aware Artifact Builder + `rebuild-artifacts` Command

**Files:**
- Modify: `src/analyze/artifact_builder.py` (resolver-aware genre lookup)
- Modify: `scripts/ai_genre_enrich.py` (`rebuild-artifacts` subparser)
- Test: `tests/unit/test_artifact_builder_enriched.py` (new)

### Why

`data/artifacts/beat3tower_32k/data_matrices_step1.npz` contains the pre-computed `X_genre_raw` matrix used by candidate-pool genre gating, beam-search transition scoring, and IDF computation in `src/playlist/genre_idf.py`. While Tasks 8 and 9 make read-side queries resolver-aware, the *artifact itself* is built from raw genres. After a batch of enrichments, the artifact should be rebuilt so IDF weights reflect the cleaner genre distribution.

### Design

`artifact_builder.py` populates `genre_lists: List[List[Tuple[str, float]]]` — one entry per track. The artifact-building code that produces this list calls into `_get_track_genres_with_weights`-style helpers. Inject the resolver here: when building per-track genres, check enriched first.

The new `rebuild-artifacts` CLI command:
- Constructs an `EnrichedGenreResolver(sidecar_db)` and passes it to the artifact builder.
- Invokes the existing build pipeline (no UI changes needed — this is a maintenance command).
- Useful between batches of enrichment; also exposes the integration cleanly for the playlist generator.

- [ ] **Step 1: Find the per-track genre lookup in `artifact_builder.py`**

Open `src/analyze/artifact_builder.py` and locate where each track's genres are gathered into `genre_lists`. Searching for the assignment near line 320 will show the loop and the function it calls. Read 50 lines of context to identify the exact function — it is the genre-fetching equivalent of `SimilarityCalculator._get_combined_genres`.

- [ ] **Step 2: Write failing test**

Create `tests/unit/test_artifact_builder_enriched.py`:

```python
"""Test that the artifact builder uses enriched genres when a resolver is provided."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _make_metadata_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            artist TEXT,
            album TEXT,
            album_id TEXT
        );
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT);
    """)
    conn.execute("INSERT INTO tracks VALUES('t1', 'Duster', 'Stratosphere', 'a1')")
    conn.execute("INSERT INTO track_genres VALUES('t1', 'indie rock', 'file', 1.0)")
    conn.commit()
    conn.close()


def _make_sidecar(path: Path, signatures: list[tuple]) -> None:
    from src.ai_genre_enrichment.storage import SidecarStore
    store = SidecarStore(str(path))
    store.initialize()
    with store.connect() as conn:
        for release_key, normalized_artist, normalized_album, genres in signatures:
            conn.execute(
                "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
                "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (release_key, normalized_artist, normalized_album, None,
                 json.dumps({"genres": genres, "sources": []}), "2026-05-28T00:00:00"),
            )
        conn.commit()


def test_artifact_builder_uses_enriched_genres(tmp_path):
    """When a resolver is provided, genre_lists reflect enriched signatures."""
    from src.analyze.artifact_builder import collect_track_genres  # see Step 3
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [
        ("duster::stratosphere", "duster", "stratosphere",
         ["slowcore", "space rock"]),
    ])

    resolver = EnrichedGenreResolver(str(sidecar))
    genres = collect_track_genres(
        track_id="t1",
        metadata_db_path=str(metadata),
        enriched_resolver=resolver,
    )
    # Genres are (name, weight) tuples; enriched signatures get uniform weight 1.0.
    names = [g[0] for g in genres]
    assert sorted(names) == ["slowcore", "space rock"]


def test_artifact_builder_falls_back_when_no_signature(tmp_path):
    from src.analyze.artifact_builder import collect_track_genres
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    metadata = tmp_path / "metadata.db"
    _make_metadata_db(metadata)
    sidecar = tmp_path / "sidecar.db"
    _make_sidecar(sidecar, [])

    resolver = EnrichedGenreResolver(str(sidecar))
    genres = collect_track_genres(
        track_id="t1",
        metadata_db_path=str(metadata),
        enriched_resolver=resolver,
    )
    names = [g[0] for g in genres]
    assert "indie rock" in names
```

- [ ] **Step 3: Refactor artifact builder to expose `collect_track_genres`**

If the per-track lookup is currently inlined in the build loop, extract it into a module-level `collect_track_genres(track_id, metadata_db_path, enriched_resolver=None)` function that:
1. If `enriched_resolver` is provided and the track's `(artist, album)` is enriched, returns `[(g, 1.0) for g in enriched]`.
2. Otherwise, runs the existing per-track lookup (track_genres + album_genres + artist_genres with weighting).

Modify the build loop in `artifact_builder.py` (~line 320) to call this helper instead of inline lookups.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_artifact_builder_enriched.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Add `rebuild-artifacts` CLI command**

In `scripts/ai_genre_enrich.py`, add subparser registration:

```python
    rebuild = sub.add_parser("rebuild-artifacts", help="Rebuild data_matrices_step1.npz using enriched genres for processed releases")
    rebuild.add_argument("--artifacts-dir", default="data/artifacts/beat3tower_32k")
```

Add the dispatch line:

```python
    if args.command == "rebuild-artifacts":
        return cmd_rebuild_artifacts(args)
```

Implement `cmd_rebuild_artifacts`:

```python
def cmd_rebuild_artifacts(args: argparse.Namespace) -> int:
    from pathlib import Path
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
    from src.analyze.artifact_builder import build_artifacts

    resolver = EnrichedGenreResolver(args.sidecar_db)
    artifacts_dir = Path(args.artifacts_dir)
    build_artifacts(
        metadata_db_path=args.metadata_db,
        artifacts_dir=artifacts_dir,
        enriched_resolver=resolver,
    )
    print(f"Rebuilt artifacts at {artifacts_dir}")
    return 0
```

(Adapt `build_artifacts` to accept and forward the resolver — exact signature depends on the existing entry point. Read `artifact_builder.py`'s public functions to match.)

- [ ] **Step 6: Manual smoke test**

```bash
python scripts/ai_genre_enrich.py rebuild-artifacts
```

Expected: rebuilds the artifact in-place. The new file is timestamped after the run. The vocab dimension may grow if enrichment surfaced previously-unseen genres.

- [ ] **Step 7: Commit**

```bash
git add src/analyze/artifact_builder.py scripts/ai_genre_enrich.py tests/unit/test_artifact_builder_enriched.py
git commit -m "feat: resolver-aware artifact builder + rebuild-artifacts CLI command"
```

---

## Full-Library Enrichment Batch Strategy

Once Tasks 1–11 land, the full-library sweep should be structured to let the vocabulary mature progressively. Below is the recommended sequence — not a TDD task, but the operational plan for after the integration ships.

### Phase A — Seed run

Pick ~50 artists spanning your genre clusters (jazz, indie/rock, ambient, electronic, pop, niche). Run the full pipeline:

```bash
for artist in <list>; do
    python scripts/ai_genre_enrich.py ingest-local --artist "$artist"
    python scripts/ai_genre_enrich.py extract-lastfm --artist "$artist"
    python scripts/ai_genre_enrich.py extract-bandcamp --artist "$artist"
    python scripts/ai_genre_enrich.py classify-tags --artist "$artist" --adjudicate
    python scripts/ai_genre_enrich.py build-enriched --artist "$artist"
done
```

**Goal:** surface the long tail of `review_only` tags. Measure: per-artist time, OpenAI cost, number of unique unknown tags.

**Cost validation:** if Bandcamp URL discovery dominates cost (likely 80%+ of OpenAI spend), redesign as "find canonical Bandcamp artist page once, enumerate releases" before Phase C. This is the most important measurement from Phase A.

### Phase B — Review + graduate

1. Open the GUI Review dock (Task 10).
2. Process the review queue: classify each unknown tag as `genre_style`, `descriptor`, `instrument`, `place`, `format`, `mood_function`, or reject.
3. Click **Graduate to YAML** — promotes both AI-adjudicated stable terms and human-reviewed terms.
4. Re-run `classify-tags` on Phase A artists (no `--adjudicate` needed) to flush stale `review_only` rows now covered by the updated vocab.
5. `python scripts/ai_genre_enrich.py rebuild-artifacts` — recompute IDF and the genre vocab matrix.
6. Manual spot-check: generate a few playlists seeded by Phase A artists and confirm cohesion improvements.

### Phase C — Alphabetical batch sweep

Process the remaining library in batches of ~100 artists, alphabetically.

Between each batch:
1. **Graduate** — `python scripts/ai_genre_enrich.py graduate-ai` (cheap, fast)
2. **Review human queue** — clear new `review_only` entries in the GUI
3. **Rebuild artifacts** — only every ~5 batches (rebuild is expensive), or whenever the vocab grows by ≥10 terms

**Why alphabetical:**
- Predictable progress, resumable after interruption
- Doesn't bias toward a single genre cluster (which would skew vocab graduation timing)
- Sidecar stores last-completed artist; resume picks up there

**Why batches of 100:**
- Small enough that vocab updates between batches help downstream batches
- Large enough to amortize startup overhead
- Cost cap per batch is bounded and predictable

### Phase D — Final consolidation

1. Final `graduate-ai` pass.
2. Final `rebuild-artifacts`.
3. Validate: generate playlists from a sampling of seeds across the library, compare cohesion metrics (transition stats, distinct-artist count) vs. pre-enrichment baseline.
4. Document the final vocab size and `review_only` residual count.

### Resumability

Add a `--resume-from` flag to whatever batch driver script we use (or expose `ingest-local`'s artist filter to accept a starting letter). The sidecar's `ai_genre_release_checks` table already records what's processed; a one-liner reports which artists are done.

---

## Self-Review

**Spec coverage:**
- ✅ extract-bandcamp CLI command → Tasks 1 & 2
- ✅ Genre resolution layer (forward + reverse lookups) → Tasks 3 & 8
- ✅ Playlist viewer genre column → Task 4
- ✅ GUI enrichment panel → Tasks 5, 6, 7
- ✅ Candidate pool integration via `get_tracks_for_genre` → Task 8
- ✅ Per-track scoring uses enriched genres (beam search, transition scoring, genre vectors) → Task 9
- ✅ Human review accessible from GUI with graduate action → Task 10
- ✅ Artifact rebuild for IDF + matrix reflects enriched genres → Task 11
- ✅ Operational batch strategy for full-library enrichment → Batch Strategy section

**Type consistency check:**
- `EnrichedGenreResolver(sidecar_db_path: str | Path)` constructor: used as `EnrichedGenreResolver(str(sidecar))` in tests and `EnrichedGenreResolver(sidecar_db_path)` in worker.py — consistent.
- `get_enriched_genres(*, artist: str, album: str | None) -> list[str] | None` — keyword-only args, used consistently.
- `get_artist_enrichment_status(artist: str) -> dict` — return shape `{"enriched_count": int, "enriched_albums": list[str]}` used by both EnrichmentPanel.set_artist (Task 5) and tests.
- `get_release_keys_with_genre(genre: str) -> set[str]` — positional arg, used consistently between Task 8 tests and `LocalLibraryClient.get_tracks_for_genre`.
- `get_all_enriched_release_keys() -> set[str]` — same.
- `fetch_bandcamp_tags` signature matches between Task 1 implementation and Task 2 monkeypatch.
- `handle_enrich_artist(*, artist: str, request_id: str)` consistent in Task 6 tests and implementation.
- `_resolve_track_genres(track, *, sidecar_db_path, fallback)` consistent in Task 4 tests and worker.py modification.
- `LocalLibraryClient(db_path, *, enriched_resolver=None)` — new kwarg, default None preserves backwards compatibility (Task 8 step 7).
- `SimilarityCalculator(db_path, config=None, *, enriched_resolver=None)` — new kwarg, same back-compat pattern. Used by Task 9 step 3 and forwarded from `LocalLibraryClient` in step 8.
- `_lookup_artist_album(track_id) -> tuple[str, str]` — internal helper added in Task 9; consistent return type across Tasks 9 callers.
- `collect_track_genres(track_id, metadata_db_path, *, enriched_resolver=None) -> list[tuple[str, float]]` — extracted helper in Task 11, returns `(genre, weight)` tuples (matching the existing `genre_lists` shape in `artifact_builder.py`).
- `ReviewPanel.vocab_graduated` — new Signal, fired on successful graduation; consumed by `MainWindow._on_vocab_graduated` (Task 10 step 5).
- Graduate button method `_on_graduate_clicked` and CLI button method `_on_cli_review_clicked` consistent between Task 10 tests and implementation.

**Placeholder scan:**
- No "TBD", "implement later" placeholders.
- Task 7 Step 5 hedges on the artist input widget name because that depends on the actual GUI layout — flagged with "read line 143 onwards" as the resolution path. Acceptable since the layout is opaque without reading the file.
- Task 8 Step 10 ("Wire the resolver at the LocalLibraryClient construction sites") is a search-and-modify step rather than a single concrete edit because the construction sites are spread across the codebase. The grep command produces the list. Acceptable since each site is a one-line change.
- Task 9 Step 8 (`SimilarityCalculator` construction sites) is the same search-and-modify pattern. The primary site is inside `LocalLibraryClient.__init__`, which is updated explicitly; other direct constructions are rare but should be checked via grep.
- Task 11 Step 1 ("Find the per-track genre lookup in artifact_builder.py") asks the implementer to read 50 lines of context because the function shape isn't fully visible from the line ranges cited in the file structure table — a deliberate read step rather than a guess. Acceptable.
- Task 11 Step 5 references `build_artifacts(metadata_db_path, artifacts_dir, enriched_resolver=...)` — the exact public entry-point name in `artifact_builder.py` should be confirmed and matched in Step 5 rather than guessed.

**Performance note for Task 8:**
- `_tracks_for_release_keys` over-fetches and filters in Python (`LIMIT ? * 10`). For very large libraries this may need a tighter SQL approach (e.g., temp table with release_keys, JOIN). The current shape is correct for the typical library size (tens of thousands of tracks). If profiling shows it slow, swap to a SQL temp-table JOIN in a follow-up.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-ai-genre-enrichment-integration.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints

Which approach?
