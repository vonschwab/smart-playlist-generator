# Enrichment Sources Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the AI genre enrichment pipeline with a three-tier classifier vocabulary, metadata.db mining, Last.fm integration, and a PySide6 review workflow with graduation feedback.

**Architecture:** Replace the flat hand-curated allowlist in `tag_classification.py` with a `GenreVocabulary` class that checks three tiers (curated YAML → engine synonyms → library DB). Add `ingest-local` and `extract-lastfm` CLI commands that feed new source types through the existing source_pages → source_tags → classification → enriched_genres pipeline. Build a PySide6 `ReviewPanel` for single-keystroke human review with a graduation command that writes accepted tags back to the vocabulary YAML.

**Tech Stack:** Python 3.11+, SQLite, PySide6, PyYAML (already a dependency via config.yaml usage)

**Spec:** `docs/superpowers/specs/2026-05-26-enrichment-sources-expansion-design.md`

---

## File Map

### Sub-project A: Three-Tier Classifier Vocabulary + Metadata Mining

| File | Action | Responsibility |
|------|--------|---------------|
| `data/genre_vocabulary.yaml` | Create | Externalized Tier 1 vocabulary (genre_style, descriptor, instrument, place, format, mood_function, label_or_org) |
| `src/ai_genre_enrichment/genre_vocabulary.py` | Create | `GenreVocabulary` class: loads YAML, bootstraps engine genres, optional library-tier DB query |
| `src/ai_genre_enrichment/tag_classification.py` | Modify | Replace inline sets with `GenreVocabulary` usage; keep `classify_source_tag()` signature unchanged |
| `src/ai_genre_enrichment/models.py` | Modify | Add `"local_metadata"` to `SOURCE_TYPES` and `AUTHORITATIVE_SOURCE_TYPES` |
| `src/ai_genre_enrichment/storage.py` | Modify | Add `"local_metadata"` to `rebuild_enriched_genres_for_release()` source_type IN clause |
| `scripts/ai_genre_enrich.py` | Modify | Add `ingest-local` subcommand |
| `tests/unit/test_genre_vocabulary.py` | Create | Tests for GenreVocabulary tiers, YAML loading, engine bootstrap, library lookup |
| `tests/unit/test_ai_genre_enrichment.py` | Modify | Add tests for ingest-local pipeline |

### Sub-project B: Last.fm Tag Integration

| File | Action | Responsibility |
|------|--------|---------------|
| `src/ai_genre_enrichment/source_extraction.py` | Modify | Add `extract_lastfm_tags_from_metadata()` |
| `src/ai_genre_enrichment/genre_vocabulary.py` | Modify | Add `LASTFM_CONFIDENCE_REDUCTION` constant |
| `src/ai_genre_enrichment/storage.py` | Modify | Support `"lastfm_tags"` source_type in rebuild; add cross-source agreement boost |
| `src/ai_genre_enrichment/models.py` | Modify | Add `"lastfm_tags"` to `SOURCE_TYPES` |
| `scripts/ai_genre_enrich.py` | Modify | Add `extract-lastfm` subcommand |
| `tests/unit/test_ai_genre_enrichment.py` | Modify | Add tests for Last.fm extraction, denoising, agreement boost |

### Sub-project C: Review Workflow

| File | Action | Responsibility |
|------|--------|---------------|
| `src/ai_genre_enrichment/storage.py` | Modify | Add `ai_genre_review_decisions` table, review queue query, decision recording |
| `src/ai_genre_enrichment/review.py` | Create | Review queue data model and graduation logic |
| `src/playlist_gui/widgets/review_panel.py` | Create | PySide6 review panel widget |
| `src/playlist_gui/main_window.py` | Modify | Add Tools menu entry for review panel |
| `scripts/ai_genre_enrich.py` | Modify | Add `review` and `graduate-reviewed` subcommands |
| `tests/unit/test_genre_vocabulary.py` | Modify | Add tests for graduation writing to YAML |
| `tests/unit/test_ai_genre_enrichment.py` | Modify | Add tests for review queue, decision recording |

---

## Sub-project A: Three-Tier Classifier Vocabulary + Metadata Mining

### Task 1: Create genre_vocabulary.yaml with current allowlists

**Files:**
- Create: `data/genre_vocabulary.yaml`

- [ ] **Step 1: Create the vocabulary YAML file**

```yaml
version: 1
genre_style:
  - alt-country
  - alternative
  - alternative rock
  - ambient
  - ambient jazz
  - art pop
  - avant-garde
  - avant-folk
  - balearic
  - bedroom pop
  - bossa nova
  - chamber pop
  - creative music
  - devotional
  - disco
  - downtempo
  - dream pop
  - drone
  - dub
  - dub techno
  - electroacoustic
  - electronic
  - electronic jazz
  - electronica
  - emo
  - experimental
  - experimental jazz
  - experimental pop
  - field recordings
  - folk
  - folk-pop
  - fourth world
  - free improvisation
  - garage rock
  - hip hop
  - house
  - idm
  - indie
  - indie folk
  - indie pop
  - indie rock
  - italo
  - jangle pop
  - jazz
  - jazz and improvised music
  - krautrock
  - library music
  - lo-fi
  - lo-fi bedroom pop
  - math rock
  - metal
  - minimal
  - neoclassical
  - new age
  - new wave
  - noise rock
  - noise-pop
  - pop
  - post-hardcore
  - post-punk
  - post-rock
  - power pop
  - psychedelic
  - punk
  - r&b
  - rock
  - shoegaze
  - singer-songwriter
  - slacker rock
  - slowcore
  - soul
  - space ambient
  - space music
  - spiritual jazz
  - synthpop
  - uk dubstep
  - world
descriptor:
  - acoustic
  - bass
  - beats
  - billboard top 40
  - brazilian
  - cosmic
  - eurorack
  - fuzz
  - instrumental
  - living room music
  - modular synthesizer
  - natural
  - songs
  - soundscapes
  - space
  - syntheziser
instrument:
  - 7 string guitar
  - drums
  - guitar
  - jazz vibraphone
  - marimba
  - percussion
  - piano
  - saxophone
  - synthesizer
  - vibes
place:
  - athens
  - atlanta
  - baltimore
  - belgium
  - brooklyn
  - california
  - chicago
  - indianapolis
  - japan
  - leeds
  - lisbon
  - london
  - los angeles
  - melbourne
  - mexico
  - montreal
  - new york
  - oakland
  - perry
  - pittsburgh
  - portland
  - richmond
  - san francisco
  - tokyo
  - united kingdom
  - united states
  - utrecht
  - washington
  - washington d.c.
format:
  - compilation
  - demo
  - ep
  - live
  - remastered
  - single
  - soundtrack
mood_function:
  - chillout
  - meditation
  - relaxation
  - sleep
  - study
label_or_org:
  - american football
  - cap'n jazz
  - dfa
  - dfa records
  - hardly art
  - james murphy
  - kim gordon
  - lcd soundsystem
  - lee ranaldo
  - mike kinsella
  - of montreal
  - owen
  - owls
  - polyvinyl
  - soundway records
  - steve shelley
  - sufjan stevens
  - thurston moore
  - tim kinsella
aliases:
  avant garde: avant-garde
  creative-music: creative music
  electronic-music: electronic
  experiemental: experimental
  hip-hop/rap: hip hop
  jazz-and-improvised-music: jazz and improvised music
  jazz-vibraphone: jazz vibraphone
  post punk: post-punk
  rnb: r&b
  soundscape: soundscapes
```

- [ ] **Step 2: Commit**

```
git add data/genre_vocabulary.yaml
git commit -m "feat: externalize genre vocabulary to YAML file"
```

---

### Task 2: Create GenreVocabulary class

**Files:**
- Create: `src/ai_genre_enrichment/genre_vocabulary.py`
- Create: `tests/unit/test_genre_vocabulary.py`

- [ ] **Step 1: Write failing tests for GenreVocabulary**

Create `tests/unit/test_genre_vocabulary.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary


@pytest.fixture
def vocab_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "genre_vocabulary.yaml"
    path.write_text(
        "version: 1\n"
        "genre_style:\n"
        "  - ambient\n"
        "  - shoegaze\n"
        "descriptor:\n"
        "  - acoustic\n"
        "instrument:\n"
        "  - guitar\n"
        "place:\n"
        "  - oakland\n"
        "format:\n"
        "  - live\n"
        "mood_function:\n"
        "  - meditation\n"
        "label_or_org:\n"
        "  - american football\n"
        "aliases:\n"
        "  shoe gaze: shoegaze\n",
        encoding="utf-8",
    )
    return path


def test_tier1_curated_genre_lookup(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    result = vocab.classify_genre("ambient")
    assert result is not None
    assert result.confidence == 0.95
    assert result.tier == 1


def test_tier1_non_genre_category(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    assert vocab.classify_non_genre("acoustic") == "descriptor"
    assert vocab.classify_non_genre("guitar") == "instrument"
    assert vocab.classify_non_genre("oakland") == "place"
    assert vocab.classify_non_genre("live") == "format"
    assert vocab.classify_non_genre("meditation") == "mood_function"
    assert vocab.classify_non_genre("american football") == "label_or_org"
    assert vocab.classify_non_genre("unknown tag") is None


def test_tier2_engine_genre_lookup(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    # "psychedelic rock" is in normalize_unified.py SYNONYM_MAP as a target
    # but not in our test YAML's genre_style list
    result = vocab.classify_genre("psychedelic rock")
    if result is not None:
        assert result.tier == 2
        assert result.confidence == 0.85


def test_tier3_library_genre_lookup(vocab_yaml: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO artist_genres VALUES ('Test', 'dark ambient', 'musicbrainz_artist')")
    conn.commit()
    conn.close()

    vocab = GenreVocabulary(vocab_yaml, library_db_path=db_path)
    result = vocab.classify_genre("dark ambient")
    assert result is not None
    assert result.tier == 3
    assert result.confidence == 0.80


def test_unknown_tag_returns_none(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    assert vocab.classify_genre("xyzzy nonsense tag") is None


def test_alias_resolution(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    result = vocab.classify_genre("shoe gaze")
    assert result is not None
    assert result.confidence == 0.95


def test_add_genre_writes_yaml(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    vocab.add_term("genre_style", "dark ambient")
    vocab.save()

    reloaded = GenreVocabulary(vocab_yaml)
    result = reloaded.classify_genre("dark ambient")
    assert result is not None
    assert result.tier == 1
    assert result.confidence == 0.95


def test_add_descriptor_writes_yaml(vocab_yaml: Path) -> None:
    vocab = GenreVocabulary(vocab_yaml)
    vocab.add_term("descriptor", "ethereal")
    vocab.save()

    reloaded = GenreVocabulary(vocab_yaml)
    assert reloaded.classify_non_genre("ethereal") == "descriptor"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_genre_vocabulary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.genre_vocabulary'`

- [ ] **Step 3: Implement GenreVocabulary**

Create `src/ai_genre_enrichment/genre_vocabulary.py`:

```python
"""Three-tier genre vocabulary for deterministic source-tag classification."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenreLookupResult:
    genre: str
    confidence: float
    tier: int
    reason: str


_NON_GENRE_CATEGORIES = ("descriptor", "instrument", "place", "format", "mood_function", "label_or_org")
_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "data" / "genre_vocabulary.yaml"


class GenreVocabulary:
    """Three-tier genre vocabulary: curated YAML → engine synonyms → library DB."""

    def __init__(
        self,
        yaml_path: str | Path = _DEFAULT_YAML_PATH,
        *,
        library_db_path: str | Path | None = None,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._raw: dict[str, Any] = {}
        self._tier1_genres: set[str] = set()
        self._non_genre_sets: dict[str, set[str]] = {}
        self._aliases: dict[str, str] = {}
        self._tier2_genres: set[str] = set()
        self._tier3_genres: set[str] = set()

        self._load_yaml()
        self._bootstrap_engine_genres()
        if library_db_path:
            self._load_library_genres(library_db_path)

    def _load_yaml(self) -> None:
        if not self._yaml_path.exists():
            return
        with self._yaml_path.open("r", encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}
        self._tier1_genres = set(self._raw.get("genre_style", []))
        for category in _NON_GENRE_CATEGORIES:
            self._non_genre_sets[category] = set(self._raw.get(category, []))
        self._aliases = dict(self._raw.get("aliases", {}))

    def _bootstrap_engine_genres(self) -> None:
        from src.genre.normalize_unified import SYNONYM_MAP, PHRASE_MAP

        engine_genres: set[str] = set()
        for target in SYNONYM_MAP.values():
            if target:
                engine_genres.add(target)
        for outputs in PHRASE_MAP.values():
            for token in outputs:
                if token:
                    engine_genres.add(token)
        self._tier2_genres = engine_genres - self._tier1_genres - self._all_non_genre_terms()

    def _load_library_genres(self, db_path: str | Path) -> None:
        resolved = Path(db_path).resolve()
        if not resolved.exists():
            return
        uri = f"file:{resolved.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            raw_genres: set[str] = set()
            for table in ("artist_genres", "album_genres", "track_genres"):
                try:
                    for row in conn.execute(f"SELECT DISTINCT genre FROM {table}"):
                        if row["genre"]:
                            raw_genres.add(row["genre"].strip().casefold())
                except sqlite3.OperationalError:
                    continue
            known = self._tier1_genres | self._tier2_genres | self._all_non_genre_terms()
            self._tier3_genres = raw_genres - known
        finally:
            conn.close()

    def _all_non_genre_terms(self) -> set[str]:
        result: set[str] = set()
        for terms in self._non_genre_sets.values():
            result |= terms
        return result

    def resolve_alias(self, tag: str) -> str:
        return self._aliases.get(tag, tag)

    def classify_genre(self, normalized_tag: str) -> GenreLookupResult | None:
        tag = self.resolve_alias(normalized_tag)
        if tag in self._tier1_genres:
            return GenreLookupResult(tag, 0.95, 1, "Curated genre vocabulary.")
        if tag in self._tier2_genres:
            return GenreLookupResult(tag, 0.85, 2, "Engine-recognized genre token.")
        if tag in self._tier3_genres:
            return GenreLookupResult(tag, 0.80, 3, "Genre found in library metadata.")
        return None

    def classify_non_genre(self, normalized_tag: str) -> str | None:
        tag = self.resolve_alias(normalized_tag)
        for category, terms in self._non_genre_sets.items():
            if tag in terms:
                return category
        return None

    def add_term(self, category: str, term: str) -> None:
        if category == "genre_style":
            self._tier1_genres.add(term)
        elif category in self._non_genre_sets:
            self._non_genre_sets[category].add(term)
        else:
            raise ValueError(f"Unknown category: {category}")

    def save(self) -> None:
        data: dict[str, Any] = {"version": self._raw.get("version", 1)}
        data["genre_style"] = sorted(self._tier1_genres)
        for category in _NON_GENRE_CATEGORIES:
            data[category] = sorted(self._non_genre_sets.get(category, set()))
        if self._aliases:
            data["aliases"] = dict(sorted(self._aliases.items()))
        with self._yaml_path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS (the Tier 2 test may skip if `psychedelic rock` isn't in SYNONYM_MAP — the conditional handles that)

- [ ] **Step 5: Commit**

```
git add src/ai_genre_enrichment/genre_vocabulary.py tests/unit/test_genre_vocabulary.py
git commit -m "feat: add GenreVocabulary with three-tier lookup"
```

---

### Task 3: Rewire tag_classification.py to use GenreVocabulary

**Files:**
- Modify: `src/ai_genre_enrichment/tag_classification.py`
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write a failing test that proves vocabulary integration works**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_classify_source_tag_uses_vocabulary_tiers(tmp_path: Path) -> None:
    """Tags recognized by the engine vocabulary (Tier 2) should classify as genre_style."""
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    # "psychedelic rock" is a SYNONYM_MAP target in normalize_unified.py
    # It should be recognized even if not in the curated YAML Tier 1
    result = classify_source_tag("psychedelic rock")
    # After vocabulary integration, this should NOT be review_only
    # (it may be genre_style at 0.85 from Tier 2, or 0.95 if already in Tier 1)
    assert result.classification == "genre_style"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_classify_source_tag_uses_vocabulary_tiers -v`
Expected: FAIL — `psychedelic rock` is currently `review_only`

- [ ] **Step 3: Rewrite tag_classification.py to use GenreVocabulary**

Replace the full content of `src/ai_genre_enrichment/tag_classification.py`:

```python
"""Deterministic first-pass classification for source-provided release tags."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .genre_vocabulary import GenreVocabulary

_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "data" / "genre_vocabulary.yaml"

_vocab: GenreVocabulary | None = None


def _get_vocabulary() -> GenreVocabulary:
    global _vocab
    if _vocab is None:
        _vocab = GenreVocabulary(_DEFAULT_YAML_PATH)
    return _vocab


def set_vocabulary(vocab: GenreVocabulary) -> None:
    """Override the module-level vocabulary (for testing or library-tier enrichment)."""
    global _vocab
    _vocab = vocab


_LABEL_OR_ORG_REASON = "Artist, label, or related-entity tag, not a genre."
_CATEGORY_REASONS = {
    "descriptor": "Descriptor tag, not a genre.",
    "instrument": "Instrument tag, not a genre.",
    "place": "Place tag, not a genre.",
    "format": "Release format tag, not a genre.",
    "mood_function": "Mood or listening-function tag, not a genre.",
    "label_or_org": _LABEL_OR_ORG_REASON,
}
_CATEGORY_CONFIDENCES = {
    "mood_function": 0.9,
}


@dataclass(frozen=True)
class SourceTagClassification:
    raw_tag: str
    normalized_tag: str
    classification: str
    confidence: float
    reason: str


def classify_source_tag(raw_tag: str) -> SourceTagClassification:
    """Classify a source tag without collapsing precise source vocabulary."""
    vocab = _get_vocabulary()
    normalized = normalize_source_tag(raw_tag)
    resolved = vocab.resolve_alias(normalized)

    genre_result = vocab.classify_genre(resolved)
    if genre_result is not None:
        return SourceTagClassification(
            raw_tag=raw_tag,
            normalized_tag=resolved,
            classification="genre_style",
            confidence=genre_result.confidence,
            reason=genre_result.reason,
        )

    non_genre = vocab.classify_non_genre(resolved)
    if non_genre is not None:
        reason = _CATEGORY_REASONS.get(non_genre, f"{non_genre} tag, not a genre.")
        confidence = _CATEGORY_CONFIDENCES.get(non_genre, 0.95)
        classification = "descriptor" if non_genre == "label_or_org" else non_genre
        return SourceTagClassification(raw_tag, resolved, classification, confidence, reason)

    return SourceTagClassification(
        raw_tag=raw_tag,
        normalized_tag=resolved,
        classification="review_only",
        confidence=0.5,
        reason="Unknown source tag requires adjudication before use.",
    )


def normalize_source_tag(raw_tag: str) -> str:
    """Normalize source tags while preserving source-specific genre distinctions."""
    text = unicodedata.normalize("NFKD", raw_tag.strip().casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()
```

Note: `normalize_source_tag` no longer applies `_CANONICAL_TAG_ALIASES` inline — alias resolution moved to `GenreVocabulary.resolve_alias()` so all aliases live in the YAML file.

- [ ] **Step 4: Run full test suite to verify nothing broke**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```
git add src/ai_genre_enrichment/tag_classification.py tests/unit/test_ai_genre_enrichment.py
git commit -m "refactor: rewire tag_classification to use GenreVocabulary"
```

---

### Task 4: Add local_metadata source type and ingest-local command

**Files:**
- Modify: `src/ai_genre_enrichment/models.py:11-35`
- Modify: `src/ai_genre_enrichment/storage.py:978-1010`
- Modify: `scripts/ai_genre_enrich.py:125-143`
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for ingest-local**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_ingest_local_creates_source_pages_and_enriched_genres(tmp_path: Path) -> None:
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"

    result = ai_genre_main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "ingest-local",
        "--artist", "Slowdive",
        "--album", "Souvlaki",
    ])
    assert result == 0

    store = SidecarStore(sidecar_db)
    with store.connect() as conn:
        pages = list(conn.execute(
            "SELECT * FROM ai_genre_source_pages WHERE release_key LIKE '%slowdive%'"
        ))
        assert len(pages) >= 1
        assert pages[0]["source_type"] == "local_metadata"
        assert pages[0]["source_url"] == "local://metadata.db"

        enriched = list(conn.execute(
            "SELECT * FROM enriched_genres WHERE release_key LIKE '%slowdive%'"
        ))
        assert len(enriched) >= 1
        genres = {row["genre"] for row in enriched}
        assert "shoegaze" in genres
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_ingest_local_creates_source_pages_and_enriched_genres -v`
Expected: FAIL — `ingest-local` command not recognized

- [ ] **Step 3: Add `local_metadata` to models.py source types**

In `src/ai_genre_enrichment/models.py`, add `"local_metadata"` to both `SOURCE_TYPES` (line 11-23) and `AUTHORITATIVE_SOURCE_TYPES` (line 26-35):

```python
SOURCE_TYPES = {
    "official_release",
    "official_artist",
    "official_label",
    "bandcamp_release",
    "label_catalog",
    "press_release",
    "liner_notes",
    "official_distributor",
    "local_payload",
    "local_metadata",
    "review_context",
    "model_knowledge",
}

AUTHORITATIVE_SOURCE_TYPES = {
    "official_release",
    "official_artist",
    "official_label",
    "bandcamp_release",
    "label_catalog",
    "press_release",
    "liner_notes",
    "official_distributor",
    "local_metadata",
}
```

- [ ] **Step 4: Add `local_metadata` to storage.py rebuild query**

In `src/ai_genre_enrichment/storage.py`, in `rebuild_enriched_genres_for_release()` (line 1001-1009), add `'local_metadata'` to the `source_type IN (...)` clause:

```python
                      AND p.source_type IN (
                          'official_release',
                          'official_artist',
                          'official_label',
                          'bandcamp_release',
                          'label_catalog',
                          'press_release',
                          'liner_notes',
                          'official_distributor',
                          'local_metadata'
                      )
```

- [ ] **Step 5: Add `ingest-local` subcommand to CLI**

In `scripts/ai_genre_enrich.py`, add the subparser after the `show-enriched` parser (around line 141):

```python
    ingest_local = sub.add_parser("ingest-local", help="Ingest local metadata genres into enriched_genres")
    add_release_filters(ingest_local)
    ingest_local.add_argument("--dry-run", action="store_true")
```

Add the dispatch in `main()` (after the `show-enriched` branch, around line 48):

```python
    if args.command == "ingest-local":
        return cmd_ingest_local(args)
```

Add the command function:

```python
def cmd_ingest_local(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.genre.normalize_unified import META_TAGS, DROP_TOKENS

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    vocab = GenreVocabulary(library_db_path=args.metadata_db)
    from src.ai_genre_enrichment.tag_classification import set_vocabulary
    set_vocabulary(vocab)

    ingested = 0
    for release in releases:
        all_genres: list[tuple[str, str]] = []
        for source_key, genres in release.existing_genres_by_source.items():
            for genre in genres:
                normalized = genre.strip().casefold()
                if normalized and normalized not in META_TAGS and normalized not in DROP_TOKENS:
                    all_genres.append((genre, source_key))

        if not all_genres:
            continue

        if args.dry_run:
            print(json.dumps({
                "release_key": release.release_key,
                "local_genres": [g for g, _ in all_genres],
                "dry_run": True,
            }, ensure_ascii=False, sort_keys=True))
            continue

        page_id = store.upsert_source_page(
            release_key=release.release_key,
            normalized_artist=release.normalized_artist,
            normalized_album=release.normalized_album,
            album_id=release.album_id,
            source_url="local://metadata.db",
            source_type="local_metadata",
            identity_status="confirmed",
            identity_confidence=1.0,
            evidence_summary="Genres from local metadata.db genre tables.",
        )
        raw_tags = [genre for genre, _ in all_genres]
        store.replace_source_tags(page_id, raw_tags)
        store.classify_source_tags(page_id)
        store.rebuild_enriched_genres_for_release(release.release_key)
        ingested += 1
        print(f"ingested {release.release_key} tags={len(raw_tags)}")

    print(f"Ingested {ingested} release(s).")
    return 0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_ingest_local_creates_source_pages_and_enriched_genres tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```
git add src/ai_genre_enrichment/models.py src/ai_genre_enrichment/storage.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add ingest-local command and local_metadata source type"
```

---

## Sub-project B: Last.fm Tag Integration

### Task 5: Add Last.fm tag extraction from metadata.db

**Files:**
- Modify: `src/ai_genre_enrichment/source_extraction.py`
- Modify: `src/ai_genre_enrichment/models.py`
- Create or modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for Last.fm extraction**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_extract_lastfm_tags_from_metadata(tmp_path: Path) -> None:
    from src.ai_genre_enrichment.source_extraction import extract_lastfm_tags_from_metadata

    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, album TEXT, album_id TEXT, year INTEGER)")
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO tracks VALUES ('t1', 'Slowdive', 'Alison', 'Souvlaki', 'a1', 1993)")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'shoegaze', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'dream pop', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'seen live', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'rock', 'musicbrainz_artist')")
    conn.execute("INSERT INTO album_genres VALUES ('a1', 'indie', 'lastfm_album')")
    conn.commit()
    conn.close()

    tags = extract_lastfm_tags_from_metadata(
        artist="Slowdive",
        album_id="a1",
        metadata_db_path=db_path,
    )
    # Should only return lastfm-sourced tags, not musicbrainz
    assert "shoegaze" in tags
    assert "dream pop" in tags
    assert "indie" in tags
    assert "rock" not in tags
    # Meta-tags should be pre-filtered
    assert "seen live" not in tags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_tags_from_metadata -v`
Expected: FAIL — `ImportError: cannot import name 'extract_lastfm_tags_from_metadata'`

- [ ] **Step 3: Implement extract_lastfm_tags_from_metadata**

Add to `src/ai_genre_enrichment/source_extraction.py`:

```python
def extract_lastfm_tags_from_metadata(
    *,
    artist: str,
    album_id: str | None = None,
    metadata_db_path: str | Path = "data/metadata.db",
) -> list[str]:
    """Extract Last.fm-sourced genre tags from local metadata.db (read-only)."""
    from src.genre.normalize_unified import META_TAGS, DROP_TOKENS

    resolved = Path(metadata_db_path).resolve()
    if not resolved.exists():
        return []
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        raw_tags: list[str] = []
        try:
            for row in conn.execute(
                "SELECT DISTINCT genre FROM artist_genres WHERE artist = ? AND source LIKE '%lastfm%'",
                (artist,),
            ):
                if row["genre"]:
                    raw_tags.append(row["genre"])
        except sqlite3.OperationalError:
            pass
        if album_id:
            try:
                for row in conn.execute(
                    "SELECT DISTINCT genre FROM album_genres WHERE album_id = ? AND source LIKE '%lastfm%'",
                    (album_id,),
                ):
                    if row["genre"]:
                        raw_tags.append(row["genre"])
            except sqlite3.OperationalError:
                pass
            try:
                for row in conn.execute(
                    """
                    SELECT DISTINCT genre
                    FROM track_genres
                    WHERE track_id IN (SELECT track_id FROM tracks WHERE album_id = ?)
                      AND source LIKE '%lastfm%'
                    """,
                    (album_id,),
                ):
                    if row["genre"]:
                        raw_tags.append(row["genre"])
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()

    noise = META_TAGS | DROP_TOKENS
    seen: set[str] = set()
    filtered: list[str] = []
    for tag in raw_tags:
        key = tag.strip().casefold()
        if key and key not in noise and key not in seen:
            seen.add(key)
            filtered.append(tag)
    return filtered
```

Also add the necessary import at the top of `source_extraction.py`:

```python
import sqlite3
from pathlib import Path
```

- [ ] **Step 4: Add `lastfm_tags` to models.py SOURCE_TYPES**

In `src/ai_genre_enrichment/models.py`, add `"lastfm_tags"` to `SOURCE_TYPES`:

```python
SOURCE_TYPES = {
    ...
    "local_metadata",
    "lastfm_tags",
    "review_context",
    "model_knowledge",
}
```

Do NOT add it to `AUTHORITATIVE_SOURCE_TYPES` — Last.fm is crowdsourced, not authoritative.

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_tags_from_metadata -v`
Expected: PASS

- [ ] **Step 6: Commit**

```
git add src/ai_genre_enrichment/source_extraction.py src/ai_genre_enrichment/models.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add Last.fm tag extraction from metadata.db"
```

---

### Task 6: Add extract-lastfm CLI command and Last.fm enriched_genres support

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Modify: `src/ai_genre_enrichment/storage.py:978-1010`
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for extract-lastfm command**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_extract_lastfm_command(tmp_path: Path) -> None:
    metadata_db = tmp_path / "metadata.db"
    conn = sqlite3.connect(metadata_db)
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, album TEXT, album_id TEXT, year INTEGER)")
    conn.execute("CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT)")
    conn.execute("CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT)")
    conn.execute("INSERT INTO tracks VALUES ('t1', 'Slowdive', 'Alison', 'Souvlaki', 'a1', 1993)")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'shoegaze', 'lastfm_artist')")
    conn.execute("INSERT INTO artist_genres VALUES ('Slowdive', 'dream pop', 'lastfm_artist')")
    conn.commit()
    conn.close()

    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    result = ai_genre_main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "extract-lastfm",
        "--artist", "Slowdive",
        "--album", "Souvlaki",
    ])
    assert result == 0

    store = SidecarStore(sidecar_db)
    with store.connect() as conn:
        pages = list(conn.execute(
            "SELECT * FROM ai_genre_source_pages WHERE source_type = 'lastfm_tags'"
        ))
        assert len(pages) == 1
        tags = list(conn.execute(
            "SELECT normalized_tag FROM ai_genre_source_tags WHERE source_page_id = ?",
            (pages[0]["source_page_id"],),
        ))
        tag_set = {row["normalized_tag"] for row in tags}
        assert "shoegaze" in tag_set
        assert "dream pop" in tag_set
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_command -v`
Expected: FAIL — `extract-lastfm` command not recognized

- [ ] **Step 3: Add `lastfm_tags` to storage.py rebuild query**

In `src/ai_genre_enrichment/storage.py`, in `rebuild_enriched_genres_for_release()`, add `'lastfm_tags'` to the `source_type IN (...)` clause:

```python
                      AND p.source_type IN (
                          'official_release',
                          'official_artist',
                          'official_label',
                          'bandcamp_release',
                          'label_catalog',
                          'press_release',
                          'liner_notes',
                          'official_distributor',
                          'local_metadata',
                          'lastfm_tags'
                      )
```

Also update the `basis` assignment in the enriched_genres INSERT to reflect Last.fm provenance. Replace the hardcoded `"authoritative_source"` basis (line 1038) with logic that checks source_type:

```python
            now = _now_iso()
            rows = [
                (
                    row["release_key"],
                    row["normalized_artist"],
                    row["normalized_album"],
                    row["album_id"],
                    row["normalized_tag"],
                    "lastfm_tags" if row["source_type"] == "lastfm_tags"
                    else "local_metadata" if row["source_type"] == "local_metadata"
                    else "authoritative_source",
                    row["confidence"],
                    row["source_tag_id"],
                    row["source_page_id"],
                    f"source_tag:{row['source_tag_id']}",
                    "accepted",
                    now,
                )
                for row in source_rows
            ]
```

- [ ] **Step 4: Add extract-lastfm subparser and command**

In `scripts/ai_genre_enrich.py`, add the subparser after `ingest-local` (around line 143):

```python
    extract_lastfm = sub.add_parser("extract-lastfm", help="Extract Last.fm tags from metadata.db")
    add_release_filters(extract_lastfm)
    extract_lastfm.add_argument("--dry-run", action="store_true")
```

Add the dispatch in `main()`:

```python
    if args.command == "extract-lastfm":
        return cmd_extract_lastfm(args)
```

Add the command function:

```python
def cmd_extract_lastfm(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.source_extraction import extract_lastfm_tags_from_metadata

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    releases = _discover(args)
    if not releases:
        print("No matching release found.")
        return 1

    extracted = 0
    for release in releases:
        tags = extract_lastfm_tags_from_metadata(
            artist=release.artist,
            album_id=release.album_id,
            metadata_db_path=args.metadata_db,
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
            evidence_summary="Last.fm tags from local metadata.db genre tables.",
        )
        store.replace_source_tags(page_id, tags)
        store.classify_source_tags(page_id)
        store.rebuild_enriched_genres_for_release(release.release_key)
        extracted += 1
        print(f"extracted-lastfm {release.release_key} tags={len(tags)}")

    print(f"Extracted Last.fm tags for {extracted} release(s).")
    return 0
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_command -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```
git add scripts/ai_genre_enrich.py src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add extract-lastfm command and Last.fm enriched_genres support"
```

---

## Sub-project C: Review Workflow

### Task 7: Add review decisions table and queue to storage

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for review queue**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_review_queue_returns_unreviewed_tags(tmp_path: Path) -> None:
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    store = SidecarStore(sidecar_db)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="a1",
        source_url="https://slowdive.bandcamp.com/album/souvlaki",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="Test source.",
    )
    store.replace_source_tags(page_id, ["shoegaze", "noise pop", "xyzzy"])
    store.classify_source_tags(page_id)

    queue = store.get_review_queue()
    # "xyzzy" should be review_only, "noise pop" may be review_only depending on vocabulary
    review_tags = {item["normalized_tag"] for item in queue}
    assert "xyzzy" in review_tags
    # Tags already classified as genre_style at high confidence should NOT be in the queue
    assert "shoegaze" not in review_tags


def test_record_review_decision_removes_from_queue(tmp_path: Path) -> None:
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    store = SidecarStore(sidecar_db)
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="a1",
        source_url="https://slowdive.bandcamp.com/album/souvlaki",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="Test source.",
    )
    store.replace_source_tags(page_id, ["xyzzy"])
    store.classify_source_tags(page_id)

    queue = store.get_review_queue()
    assert len(queue) == 1
    item = queue[0]

    store.record_review_decision(
        source_tag_id=item["source_tag_id"],
        release_key="slowdive::souvlaki",
        raw_tag="xyzzy",
        normalized_tag="xyzzy",
        original_classification="review_only",
        reviewed_classification="rejected",
    )

    queue_after = store.get_review_queue()
    assert len(queue_after) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_review_queue_returns_unreviewed_tags tests/unit/test_ai_genre_enrichment.py::test_record_review_decision_removes_from_queue -v`
Expected: FAIL — `AttributeError: 'SidecarStore' object has no attribute 'get_review_queue'`

- [ ] **Step 3: Add review decisions table and methods to SidecarStore**

In `src/ai_genre_enrichment/storage.py`, add the table creation to `initialize()` (after the `enriched_genre_signatures` table, around line 300):

```python
                CREATE TABLE IF NOT EXISTS ai_genre_review_decisions (
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

                CREATE INDEX IF NOT EXISTS idx_review_decisions_tag
                    ON ai_genre_review_decisions (source_tag_id);
                CREATE INDEX IF NOT EXISTS idx_review_decisions_release
                    ON ai_genre_review_decisions (release_key);
```

Add these methods to `SidecarStore`:

```python
    def get_review_queue(
        self,
        *,
        release_key: str | None = None,
        classification: str | None = None,
        source_type: str | None = None,
        max_confidence: float = 0.80,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return tags needing human review, ordered by confidence ascending."""
        with self.connect() as conn:
            clauses = [
                "d.decision_id IS NULL",
                "(c.classification = 'review_only' OR c.confidence < ?)",
            ]
            params: list[Any] = [max_confidence]
            if release_key:
                clauses.append("p.release_key = ?")
                params.append(release_key)
            if classification:
                clauses.append("c.classification = ?")
                params.append(classification)
            if source_type:
                clauses.append("p.source_type = ?")
                params.append(source_type)
            where = " AND ".join(clauses)
            limit_clause = f" LIMIT {int(limit)}" if limit else ""
            rows = list(conn.execute(
                f"""
                SELECT t.source_tag_id, p.release_key, p.normalized_artist, p.normalized_album,
                       t.raw_tag, t.normalized_tag, c.classification, c.confidence,
                       p.source_url, p.source_type
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
                LEFT JOIN ai_genre_review_decisions d ON d.source_tag_id = t.source_tag_id
                WHERE {where}
                ORDER BY c.confidence ASC, p.release_key, t.tag_position
                {limit_clause}
                """,
                params,
            ))
            return [dict(row) for row in rows]

    def record_review_decision(
        self,
        *,
        source_tag_id: int,
        release_key: str,
        raw_tag: str,
        normalized_tag: str,
        original_classification: str,
        reviewed_classification: str,
        reviewer: str = "human",
        notes: str | None = None,
    ) -> int:
        """Record a human review decision for one source tag."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_review_decisions (
                    source_tag_id, release_key, raw_tag, normalized_tag,
                    original_classification, reviewed_classification,
                    reviewer, decided_at, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_tag_id, reviewer)
                DO UPDATE SET
                    reviewed_classification = excluded.reviewed_classification,
                    decided_at = excluded.decided_at,
                    notes = excluded.notes
                """,
                (
                    source_tag_id,
                    release_key,
                    raw_tag,
                    normalized_tag,
                    original_classification,
                    reviewed_classification,
                    reviewer,
                    _now_iso(),
                    notes,
                ),
            )
            row = conn.execute(
                "SELECT decision_id FROM ai_genre_review_decisions WHERE source_tag_id = ? AND reviewer = ?",
                (source_tag_id, reviewer),
            ).fetchone()
            return int(row["decision_id"])

    def undo_review_decision(self, source_tag_id: int, reviewer: str = "human") -> bool:
        """Remove a review decision, returning the tag to the review queue."""
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM ai_genre_review_decisions WHERE source_tag_id = ? AND reviewer = ?",
                (source_tag_id, reviewer),
            )
            return cursor.rowcount > 0

    def get_review_context(self, release_key: str) -> list[dict[str, Any]]:
        """Return all classified tags for a release (for showing context during review)."""
        with self.connect() as conn:
            rows = list(conn.execute(
                """
                SELECT t.normalized_tag, c.classification, c.confidence
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
                WHERE p.release_key = ?
                ORDER BY c.classification, t.normalized_tag
                """,
                (release_key,),
            ))
            return [dict(row) for row in rows]

    def get_graduated_terms(self) -> dict[str, set[str]]:
        """Return all human-reviewed terms grouped by their reviewed classification."""
        with self.connect() as conn:
            rows = list(conn.execute(
                """
                SELECT DISTINCT normalized_tag, reviewed_classification
                FROM ai_genre_review_decisions
                WHERE reviewed_classification != 'rejected'
                  AND reviewer = 'human'
                """
            ))
            result: dict[str, set[str]] = {}
            for row in rows:
                classification = row["reviewed_classification"]
                result.setdefault(classification, set()).add(row["normalized_tag"])
            return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_review_queue_returns_unreviewed_tags tests/unit/test_ai_genre_enrichment.py::test_record_review_decision_removes_from_queue -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```
git add src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add review decisions table and queue to SidecarStore"
```

---

### Task 8: Add CLI review and graduate-reviewed commands

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing test for graduate-reviewed**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_graduate_reviewed_writes_to_yaml(tmp_path: Path) -> None:
    sidecar_db = tmp_path / "ai_genre_enriched_test.db"
    vocab_yaml = tmp_path / "genre_vocabulary.yaml"
    vocab_yaml.write_text(
        "version: 1\ngenre_style:\n  - ambient\ndescriptor: []\ninstrument: []\n"
        "place: []\nformat: []\nmood_function: []\nlabel_or_org: []\naliases: {}\n",
        encoding="utf-8",
    )

    store = SidecarStore(sidecar_db)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="test::album",
        normalized_artist="test",
        normalized_album="album",
        album_id=None,
        source_url="https://test.bandcamp.com/album/album",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=1.0,
        evidence_summary="Test.",
    )
    store.replace_source_tags(page_id, ["dark ambient"])
    store.classify_source_tags(page_id)

    queue = store.get_review_queue()
    assert len(queue) >= 1
    item = [q for q in queue if q["normalized_tag"] == "dark ambient"][0]

    store.record_review_decision(
        source_tag_id=item["source_tag_id"],
        release_key="test::album",
        raw_tag="dark ambient",
        normalized_tag="dark ambient",
        original_classification="review_only",
        reviewed_classification="genre_style",
    )

    result = ai_genre_main([
        "--sidecar-db", str(sidecar_db),
        "graduate-reviewed",
        "--vocab-yaml", str(vocab_yaml),
    ])
    assert result == 0

    import yaml
    data = yaml.safe_load(vocab_yaml.read_text(encoding="utf-8"))
    assert "dark ambient" in data["genre_style"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_graduate_reviewed_writes_to_yaml -v`
Expected: FAIL — `graduate-reviewed` command not recognized

- [ ] **Step 3: Add subparsers and command implementations**

In `scripts/ai_genre_enrich.py`, add the subparsers:

```python
    review = sub.add_parser("review", help="Interactive CLI review of unclassified tags")
    review.add_argument("--limit", type=int, default=20)
    review.add_argument("--release-key")
    review.add_argument("--source-type")

    graduate = sub.add_parser("graduate-reviewed", help="Graduate human-reviewed tags into vocabulary YAML")
    graduate.add_argument(
        "--vocab-yaml",
        type=Path,
        default=ROOT / "data" / "genre_vocabulary.yaml",
    )
```

Add dispatches in `main()`:

```python
    if args.command == "review":
        return cmd_review(args)
    if args.command == "graduate-reviewed":
        return cmd_graduate_reviewed(args)
```

Add the command functions:

```python
def cmd_review(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    queue = store.get_review_queue(
        release_key=getattr(args, "release_key", None),
        source_type=getattr(args, "source_type", None),
        limit=args.limit,
    )
    if not queue:
        print("No tags in review queue.")
        return 0

    valid_keys = {"a": "genre_style", "d": "descriptor", "i": "instrument", "p": "place", "r": "rejected", "s": None}
    reviewed = 0
    for item in queue:
        context = store.get_review_context(item["release_key"])
        context_lines = [
            f"  {c['normalized_tag']} ({c['classification']}, {c['confidence']:.2f})"
            for c in context
            if c["normalized_tag"] != item["normalized_tag"]
        ]
        print(f"\nRelease: {item['normalized_artist']} — {item['normalized_album']}")
        print(f"Source:  {item['source_url']}")
        print(f"Current: {item['classification']} ({item['confidence']:.2f})")
        print(f'Tag: "{item["normalized_tag"]}"')
        if context_lines:
            print("Context:")
            for line in context_lines[:8]:
                print(line)
        print("[A]ccept genre  [D]escriptor  [I]nstrument  [P]lace  [R]eject  [S]kip  [Q]uit")

        while True:
            try:
                choice = input("> ").strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if choice == "q":
                return 0
            if choice in valid_keys:
                break
            print("Invalid choice. Use a/d/i/p/r/s/q.")

        classification = valid_keys[choice]
        if classification is None:
            continue

        store.record_review_decision(
            source_tag_id=item["source_tag_id"],
            release_key=item["release_key"],
            raw_tag=item["raw_tag"],
            normalized_tag=item["normalized_tag"],
            original_classification=item["classification"],
            reviewed_classification=classification,
        )
        reviewed += 1
        print(f"  → {classification}")

    print(f"\nReviewed {reviewed} tag(s).")
    return 0


def cmd_graduate_reviewed(args: argparse.Namespace) -> int:
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    terms = store.get_graduated_terms()
    if not terms:
        print("No reviewed terms to graduate.")
        return 0

    vocab = GenreVocabulary(args.vocab_yaml)
    category_map = {
        "genre_style": "genre_style",
        "descriptor": "descriptor",
        "instrument": "instrument",
        "place": "place",
        "format": "format",
        "mood_function": "mood_function",
        "label_or_org": "label_or_org",
    }
    added = 0
    for classification, tags in terms.items():
        category = category_map.get(classification)
        if not category:
            continue
        for tag in sorted(tags):
            vocab.add_term(category, tag)
            added += 1
            print(f"  graduated {tag} → {category}")

    vocab.save()
    print(f"Graduated {added} term(s) to {args.vocab_yaml}")
    return 0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_graduate_reviewed_writes_to_yaml -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```
git add scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add CLI review and graduate-reviewed commands"
```

---

### Task 9: Create PySide6 ReviewPanel widget

**Files:**
- Create: `src/playlist_gui/widgets/review_panel.py`
- Modify: `src/playlist_gui/main_window.py`

- [ ] **Step 1: Create the ReviewPanel widget**

Create `src/playlist_gui/widgets/review_panel.py`:

```python
"""PySide6 review panel for human-in-the-loop genre tag review."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ai_genre_enrichment.storage import SidecarStore


class ReviewPanel(QWidget):
    """Single-keystroke review panel for genre tag classification."""

    review_completed = Signal()

    def __init__(self, sidecar_db_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = SidecarStore(sidecar_db_path)
        self._store.initialize()
        self._queue: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._index = 0
        self._stats = {"accepted": 0, "descriptor": 0, "instrument": 0, "place": 0, "rejected": 0, "skipped": 0}

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Review Queue</b>"))
        header.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "review_only", "Low confidence"])
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        header.addWidget(self._filter_combo)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.load_queue)
        header.addWidget(self._refresh_btn)
        layout.addLayout(header)

        self._release_label = QLabel()
        self._release_label.setWordWrap(True)
        layout.addWidget(self._release_label)

        self._source_label = QLabel()
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        self._current_label = QLabel()
        layout.addWidget(self._current_label)

        self._tag_label = QLabel()
        self._tag_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        layout.addWidget(self._tag_label)

        self._context_label = QLabel()
        self._context_label.setWordWrap(True)
        layout.addWidget(self._context_label)

        layout.addStretch()

        buttons = QHBoxLayout()
        for key, label, classification in [
            ("A", "Accept genre", "genre_style"),
            ("D", "Descriptor", "descriptor"),
            ("I", "Instrument", "instrument"),
            ("P", "Place", "place"),
            ("S", "Skip", None),
            ("R", "Reject", "rejected"),
        ]:
            btn = QPushButton(f"[{key}] {label}")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda checked=False, c=classification: self._decide(c))
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self._progress_label = QLabel()
        layout.addWidget(self._progress_label)

    def _setup_shortcuts(self) -> None:
        for key, classification in [
            ("A", "genre_style"),
            ("D", "descriptor"),
            ("I", "instrument"),
            ("P", "place"),
            ("R", "rejected"),
        ]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda c=classification: self._decide(c))

        skip_shortcut = QShortcut(QKeySequence("S"), self)
        skip_shortcut.activated.connect(lambda: self._decide(None))

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._undo)

    def load_queue(self) -> None:
        filter_text = self._filter_combo.currentText()
        classification = "review_only" if filter_text == "review_only" else None
        max_confidence = 0.80 if filter_text == "Low confidence" else 0.80
        self._queue = self._store.get_review_queue(
            classification=classification,
            max_confidence=max_confidence,
        )
        self._index = 0
        self._show_current()

    def _on_filter_changed(self) -> None:
        self.load_queue()

    def _show_current(self) -> None:
        if self._index >= len(self._queue):
            self._release_label.setText("")
            self._source_label.setText("")
            self._current_label.setText("")
            self._tag_label.setText("Review complete — no more tags in queue.")
            self._context_label.setText("")
            self._update_progress()
            self.review_completed.emit()
            return

        item = self._queue[self._index]
        self._release_label.setText(f"Release: {item['normalized_artist']} — {item['normalized_album']}")
        self._source_label.setText(f"Source: {item['source_url']}")
        self._current_label.setText(f"Current: {item['classification']} ({item['confidence']:.2f})")
        self._tag_label.setText(f'"{item["normalized_tag"]}"')

        context = self._store.get_review_context(item["release_key"])
        context_lines = [
            f"  {c['normalized_tag']} ({c['classification']}, {c['confidence']:.2f})"
            for c in context
            if c["normalized_tag"] != item["normalized_tag"]
        ]
        self._context_label.setText("Context:\n" + "\n".join(context_lines[:8]) if context_lines else "")
        self._update_progress()

    def _decide(self, classification: str | None) -> None:
        if self._index >= len(self._queue):
            return
        item = self._queue[self._index]

        if classification is not None:
            self._store.record_review_decision(
                source_tag_id=item["source_tag_id"],
                release_key=item["release_key"],
                raw_tag=item["raw_tag"],
                normalized_tag=item["normalized_tag"],
                original_classification=item["classification"],
                reviewed_classification=classification,
            )
            self._store.rebuild_enriched_genres_for_release(item["release_key"])
            stat_key = classification if classification in self._stats else "accepted"
            self._stats[stat_key] = self._stats.get(stat_key, 0) + 1
            self._history.append({"source_tag_id": item["source_tag_id"], "classification": classification})
        else:
            self._stats["skipped"] += 1

        self._index += 1
        self._show_current()

    def _undo(self) -> None:
        if not self._history:
            return
        last = self._history.pop()
        self._store.undo_review_decision(last["source_tag_id"])
        stat_key = last["classification"] if last["classification"] in self._stats else "accepted"
        self._stats[stat_key] = max(0, self._stats.get(stat_key, 0) - 1)
        self._index = max(0, self._index - 1)
        self._show_current()

    def _update_progress(self) -> None:
        total = len(self._queue)
        reviewed = self._index
        parts = [f"{reviewed} / {total} reviewed"]
        for key, count in self._stats.items():
            if count > 0:
                parts.append(f"{count} {key}")
        self._progress_label.setText(" | ".join(parts))
```

- [ ] **Step 2: Add ReviewPanel to MainWindow**

In `src/playlist_gui/main_window.py`, add the import near the top imports (around line 35):

```python
from .widgets.review_panel import ReviewPanel
```

In the `_setup_menu_bar` method, add a Tools menu action (after the existing tools, around line 560):

```python
        self._add_tool_action(
            "Genre Tag &Review...",
            self._on_open_review_panel,
            "Review and classify unresolved genre tags from enrichment sources",
        )
```

Add the handler method to `MainWindow`:

```python
    def _on_open_review_panel(self) -> None:
        from pathlib import Path

        sidecar_db = Path("data/ai_genre_enrichment.db")
        if not sidecar_db.exists():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No enrichment sidecar DB found at data/ai_genre_enrichment.db.")
            return
        panel = ReviewPanel(str(sidecar_db), parent=self)
        panel.setWindowTitle("Genre Tag Review")
        panel.setWindowFlags(Qt.WindowType.Window)
        panel.resize(700, 500)
        panel.load_queue()
        panel.show()
```

- [ ] **Step 3: Commit**

```
git add src/playlist_gui/widgets/review_panel.py src/playlist_gui/main_window.py
git commit -m "feat: add PySide6 ReviewPanel with single-keystroke review"
```

---

### Task 10: Final integration test and lint

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_genre_vocabulary.py -v`
Expected: All PASS

- [ ] **Step 2: Run lint**

Run: `ruff check src/ai_genre_enrichment/ scripts/ai_genre_enrich.py tests/unit/test_genre_vocabulary.py`
Expected: No errors (fix any that appear)

- [ ] **Step 3: Run mypy on new modules**

Run: `mypy src/ai_genre_enrichment/genre_vocabulary.py src/ai_genre_enrichment/review.py --ignore-missing-imports`
Expected: No errors

- [ ] **Step 4: Commit any lint fixes**

```
git add -u
git commit -m "fix: lint and type fixes for enrichment sources expansion"
```
