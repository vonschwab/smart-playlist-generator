# Pre-Integration Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix YAML save formatting, add tests for untested features, clean up stale cache entries, re-classify all releases against the current vocabulary, and commit — so the enrichment pipeline is ready for end-to-end integration with the playlist generator.

**Architecture:** Five sequential tasks: fix the YAML serializer (prerequisite for graduation), write tests for four features added during live testing, clean up questionable/stale adjudication cache entries, re-classify + rebuild all enriched releases, and commit everything.

**Tech Stack:** Python 3.11+, PyYAML, SQLite, pytest

---

## File Map

| File | Responsibility | Tasks |
|------|----------------|-------|
| `src/ai_genre_enrichment/genre_vocabulary.py` | `save()` method — custom YAML serializer | 1 |
| `tests/unit/test_ai_genre_enrichment.py` | Unit tests for new features | 2 |
| `data/genre_vocabulary.yaml` | Vocabulary YAML (add `abstract`, `acid` to descriptors) | 3 |
| `data/ai_genre_enrichment.db` | Sidecar DB — cache cleanup | 3, 4 |
| `scripts/ai_genre_enrich.py` | CLI for re-classify + rebuild | 4 |

---

### Task 1: Fix `vocab.save()` YAML Formatting

The current `save()` uses `yaml.dump()` which reformats the entire file — list items lose their 2-space indentation and the diff becomes unreadable. Replace it with a custom serializer that produces output matching the hand-written YAML style.

**Files:**
- Modify: `src/ai_genre_enrichment/genre_vocabulary.py:144-156`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_vocab_save_preserves_indentation_style(tmp_path):
    """save() must produce 2-space-indented list items, not flush-left."""
    yaml_path = tmp_path / "vocab.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "genre_style:\n"
        "  - ambient\n"
        "  - shoegaze\n"
        "descriptor:\n"
        "  - acoustic\n"
        "instrument:\n"
        "  - piano\n"
        "place:\n"
        "  - oakland\n"
        "format:\n"
        "  - ep\n"
        "mood_function:\n"
        "  - chillout\n"
        "label_or_org:\n"
        "  - dfa\n"
        "aliases:\n"
        "  post punk: post-punk\n"
        "decompose:\n"
        "  funk / soul:\n"
        "    - funk\n"
        "    - soul\n",
        encoding="utf-8",
    )
    vocab = GenreVocabulary(yaml_path)
    vocab.add_term("genre_style", "drone")
    vocab.save()

    saved = yaml_path.read_text(encoding="utf-8")
    # Every list item must be indented 2 spaces
    for line in saved.splitlines():
        if line.strip().startswith("- "):
            assert line.startswith("  - ") or line.startswith("    - "), (
                f"List item not indented: {line!r}"
            )
    # Round-trip: reload and verify the new term is present
    vocab2 = GenreVocabulary(yaml_path)
    result = vocab2.classify_genre("drone")
    assert result is not None
    assert result.tier == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_vocab_save_preserves_indentation_style -v`

Expected: FAIL — `yaml.dump()` writes `- ambient` flush-left, not `  - ambient`.

- [ ] **Step 3: Implement custom YAML serializer**

Replace the `save()` method in `src/ai_genre_enrichment/genre_vocabulary.py` (lines 144–156) with:

```python
    def save(self) -> None:
        # Tier-2 (engine) and tier-3 (library DB) genres are derived at load time and not saved —
        # they're re-bootstrapped from their sources on the next instantiation.
        lines: list[str] = [f"version: {self._raw.get('version', 1)}"]

        # Flat list sections
        section_order = ["genre_style"] + list(_NON_GENRE_CATEGORIES)
        section_data: dict[str, list[str]] = {"genre_style": sorted(self._tier1_genres)}
        for cat in _NON_GENRE_CATEGORIES:
            section_data[cat] = sorted(self._non_genre_sets.get(cat, set()))

        for section in section_order:
            lines.append(f"{section}:")
            for item in section_data[section]:
                lines.append(f"  - {item}")

        # Aliases: key-value dict
        if self._aliases:
            lines.append("aliases:")
            for key, value in sorted(self._aliases.items()):
                lines.append(f"  {key}: {value}")

        # Decompose: dict of lists
        if self._decompose:
            lines.append("decompose:")
            for key, values in sorted(self._decompose.items()):
                lines.append(f"  {key}:")
                for v in values:
                    lines.append(f"    - {v}")

        with self._yaml_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_vocab_save_preserves_indentation_style -v`

Expected: PASS

- [ ] **Step 5: Verify no-diff round-trip on the real vocabulary file**

Run this command to confirm `save()` produces byte-identical output when nothing changes:

```bash
python -c "
import shutil, sys; sys.path.insert(0, '.')
shutil.copy('data/genre_vocabulary.yaml', '/tmp/vocab_backup.yaml')
from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
v = GenreVocabulary()
v.save()
import subprocess
result = subprocess.run(['diff', '/tmp/vocab_backup.yaml', 'data/genre_vocabulary.yaml'], capture_output=True, text=True)
print(result.stdout or 'IDENTICAL')
"
```

Expected: `IDENTICAL` (no diff). If there is a diff, inspect it — the serializer's output must match the hand-written style exactly. Common issues: trailing newline, sort order, special characters in values (e.g., `cap'n jazz`, `r&b`). Fix the serializer until output is identical.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -x -q`

Expected: All 116+ tests pass (115 existing + 1 new).

- [ ] **Step 7: Commit**

```bash
git add src/ai_genre_enrichment/genre_vocabulary.py tests/unit/test_ai_genre_enrichment.py
git commit -m "fix: replace yaml.dump with custom serializer to preserve indentation style"
```

---

### Task 2: Write Tests for Untested Features

Four features were added during live testing without dedicated tests: year-pattern guard, `decompose_tag()`, alias resolution at enrichment time, and decompose expansion at enrichment time.

**Files:**
- Modify: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write the year-pattern guard tests**

Add to `tests/unit/test_ai_genre_enrichment.py`:

```python
from src.ai_genre_enrichment.tag_classification import classify_source_tag, set_vocabulary, reset_vocabulary


def test_year_tags_classified_as_descriptor():
    """Year-based tags from Last.fm should be caught deterministically as descriptors."""
    cases = [
        ("2016", "descriptor"),
        ("2021", "descriptor"),
        ("2016 albums", "descriptor"),
        ("2016 releases", "descriptor"),
        ("best of 2023", "descriptor"),
        ("best albums of the 2000s", "descriptor"),
    ]
    for raw_tag, expected_cls in cases:
        result = classify_source_tag(raw_tag)
        assert result.classification == expected_cls, (
            f"{raw_tag!r}: expected {expected_cls}, got {result.classification}"
        )
```

- [ ] **Step 2: Write the decompose_tag test**

```python
def test_decompose_tag_returns_list_or_none(tmp_path):
    yaml_path = tmp_path / "vocab.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "genre_style:\n"
        "  - ambient\n"
        "descriptor: []\n"
        "instrument: []\n"
        "place: []\n"
        "format: []\n"
        "mood_function: []\n"
        "label_or_org: []\n"
        "decompose:\n"
        "  funk / soul:\n"
        "    - funk\n"
        "    - soul\n",
        encoding="utf-8",
    )
    vocab = GenreVocabulary(yaml_path)
    assert vocab.decompose_tag("funk / soul") == ["funk", "soul"]
    assert vocab.decompose_tag("ambient") is None
    assert vocab.decompose_tag("nonexistent") is None
```

- [ ] **Step 3: Write alias + decompose at enrichment time test**

This test verifies that `rebuild_enriched_genres_for_release()` applies `resolve_alias()` and `decompose_tag()` when building enriched genre rows.

```python
def test_rebuild_enriched_applies_alias_and_decompose(tmp_path, monkeypatch):
    """rebuild_enriched_genres_for_release applies alias resolution and decompose expansion."""
    # Create a vocabulary YAML with an alias and a decompose rule
    yaml_path = tmp_path / "vocab.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "genre_style:\n"
        "  - space rock\n"
        "  - funk\n"
        "  - soul\n"
        "descriptor: []\n"
        "instrument: []\n"
        "place: []\n"
        "format: []\n"
        "mood_function: []\n"
        "label_or_org: []\n"
        "aliases:\n"
        "  spacerock: space rock\n"
        "decompose:\n"
        "  funk / soul:\n"
        "    - funk\n"
        "    - soul\n",
        encoding="utf-8",
    )
    # Patch GenreVocabulary in storage module to use our test yaml
    from src.ai_genre_enrichment import genre_vocabulary as gv_mod
    original_default = gv_mod._DEFAULT_YAML_PATH
    monkeypatch.setattr(gv_mod, "_DEFAULT_YAML_PATH", yaml_path)

    store = SidecarStore(tmp_path / "test.db")
    store.initialize()

    release_key = "testartist::testalbum"
    with store.connect() as conn:
        # Insert source page
        conn.execute(
            """INSERT INTO ai_genre_source_pages
               (source_url, source_type, release_key, normalized_artist, normalized_album,
                album_id, identity_status, identity_confidence, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            ("http://test", "local_metadata", release_key, "testartist", "testalbum",
             "a1", "confirmed", 1.0),
        )
        page_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert two source tags: one that needs alias, one that needs decompose
        for pos, (raw, norm) in enumerate([("spacerock", "spacerock"), ("Funk / Soul", "funk / soul")]):
            conn.execute(
                """INSERT INTO ai_genre_source_tags
                   (source_page_id, raw_tag, normalized_tag, tag_position, extracted_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (page_id, raw, norm, pos),
            )
            tag_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                """INSERT INTO ai_genre_tag_classifications
                   (source_tag_id, classification, confidence, classifier, reason, classified_at)
                   VALUES (?, 'genre_style', 0.95, 'deterministic', 'test', datetime('now'))""",
                (tag_id,),
            )

    store.rebuild_enriched_genres_for_release(release_key)

    with store.connect() as conn:
        genres = sorted(
            row["genre"]
            for row in conn.execute(
                "SELECT genre FROM enriched_genres WHERE release_key = ?", (release_key,)
            )
        )
    # "spacerock" → alias → "space rock"
    # "funk / soul" → decompose → "funk", "soul"
    assert genres == ["funk", "soul", "space rock"]

    # Restore
    monkeypatch.setattr(gv_mod, "_DEFAULT_YAML_PATH", original_default)
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_year_tags_classified_as_descriptor tests/unit/test_ai_genre_enrichment.py::test_decompose_tag_returns_list_or_none tests/unit/test_ai_genre_enrichment.py::test_rebuild_enriched_applies_alias_and_decompose -v`

Expected: All 3 PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -x -q`

Expected: All 119+ tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_ai_genre_enrichment.py
git commit -m "test: add coverage for year-pattern guard, decompose, and alias-at-enrichment"
```

---

### Task 3: Clean Up Adjudication Cache and Vocabulary

Resolve four questionable cache entries and remove stale entries now handled by the deterministic classifier.

**Files:**
- Modify: `data/genre_vocabulary.yaml`
- Modify: `data/ai_genre_enrichment.db` (cache table)

- [ ] **Step 1: Add `abstract`, `acid`, and `stage & screen` to vocabulary as descriptors**

`abstract` is too generic standalone; `acid` is ambiguous without a qualifier; `stage & screen` is a market category. Add all three to the `descriptor` section of `data/genre_vocabulary.yaml`, maintaining alphabetical order:

In `data/genre_vocabulary.yaml`, add to the `descriptor:` section:
- `abstract` (between `acoustic` and `art`)
- `acid` (between `abstract` and `art`)
- `stage & screen` (between `space` and `syntheziser`)

- [ ] **Step 2: Add `leftfield` to `label_or_org`**

`leftfield` is primarily a record label name (Leftfield / Hard Hands). Add it to the `label_or_org:` section in `data/genre_vocabulary.yaml`, maintaining alphabetical order (between `kim gordon` and `lcd soundsystem`).

- [ ] **Step 3: Delete resolved entries from the adjudication cache**

Run:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/ai_genre_enrichment.db')

# Questionable entries now pinned in vocabulary
to_delete = ['abstract', 'acid', 'leftfield', 'stage & screen']

# Stale entries now handled by deterministic classifier (aliases/descriptors/formats)
to_delete += ['avantgarde', 'orchestral', 'spacerock', 'electronica dance', 'vidwo', 'twee pop']

placeholders = ','.join('?' for _ in to_delete)
conn.execute(f'DELETE FROM ai_tag_adjudication_cache WHERE normalized_tag IN ({placeholders})', to_delete)
conn.commit()
deleted = conn.execute('SELECT changes()').fetchone()[0]
print(f'Deleted {deleted} stale cache entries')

remaining = conn.execute('SELECT COUNT(*) FROM ai_tag_adjudication_cache').fetchone()[0]
print(f'{remaining} entries remain in cache')
conn.close()
"
```

Expected: Deletes ~10 entries. Remaining entries should all be legitimate genre_style terms (ambient techno, breakbeat, chillwave, etc.) or valid non-genre classifications (descriptor, mood_function, etc.).

- [ ] **Step 4: Verify vocabulary loads cleanly**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
v = GenreVocabulary()
assert v.classify_non_genre('abstract') == 'descriptor'
assert v.classify_non_genre('acid') == 'descriptor'
assert v.classify_non_genre('stage & screen') == 'descriptor'
assert v.classify_non_genre('leftfield') == 'label_or_org'
print('All 4 questionable terms now deterministic non-genre')
"
```

- [ ] **Step 5: Commit**

```bash
git add data/genre_vocabulary.yaml
git commit -m "fix: pin abstract, acid, stage & screen, leftfield as non-genre in vocabulary"
```

---

### Task 4: Re-Classify and Rebuild All Enriched Releases

After vocabulary changes, re-run classification and enrichment for every artist that's been processed. This flushes stale enrichment data.

**Files:**
- None modified — this is a data operation using existing CLI commands

- [ ] **Step 1: Re-classify all processed artists**

Run `classify-tags` for every artist (no `--adjudicate` since vocab changes make deterministic classification correct; AI-cached entries are already stable):

```bash
for artist in "Slowdive" "Aphex Twin" "Makaya McCraven" "Nala Sinephro" "Porter Ricks" "Monolake" "Pere Ubu" "North Americans" "Corntuth" "Bachelor" "The Legends" "Teethe" "Tommy Oeffling" "Sam Evian" "Orchid Mantis" "THiQ" "Bleary Eyed" "Cate Le Bon" "The Aislers Set"; do
  echo "=== $artist ==="
  python scripts/ai_genre_enrich.py classify-tags --artist "$artist" 2>&1 | tail -1
done
```

Expected: All artists classify without errors. Tags previously hitting AI should now resolve deterministically for `abstract`, `acid`, `leftfield`, `stage & screen`, `avantgarde`, `orchestral`, `vidwo`, year tags, etc.

- [ ] **Step 2: Rebuild enriched genres for all artists**

```bash
for artist in "Slowdive" "Aphex Twin" "Makaya McCraven" "Nala Sinephro" "Porter Ricks" "Monolake" "Pere Ubu" "North Americans" "Corntuth" "Bachelor" "The Legends" "Teethe" "Tommy Oeffling" "Sam Evian" "Orchid Mantis" "THiQ" "Bleary Eyed" "Cate Le Bon" "The Aislers Set"; do
  python scripts/ai_genre_enrich.py build-enriched --artist "$artist" 2>&1 | tail -1
done
```

Expected: All releases rebuilt. Alias resolution (`spacerock` → `space rock`, `twee pop` → `twee`) and decompose expansion (`funk / soul` → `funk` + `soul`, `electronica dance` → `electronica` + `dance`) are applied.

- [ ] **Step 3: Spot-check enriched results**

Verify a few key artists no longer have stale data:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/ai_genre_enrichment.db')
for artist in ['the aislers set', 'orchid mantis', 'teethe', 'sam evian']:
    genres = sorted(set(
        row[0] for row in conn.execute(
            'SELECT DISTINCT genre FROM enriched_genres WHERE normalized_artist = ?', (artist,)
        )
    ))
    print(f'{artist}: {genres}')
conn.close()
"
```

Verify:
- `the aislers set`: should have `twee` (not `twee pop`), should NOT have `twee pop` separately
- `orchid mantis`: should NOT have `tape music`
- `teethe`: `spacerock` should appear as `space rock`
- `sam evian`: `funk / soul` should appear as separate `funk` and `soul`

- [ ] **Step 4: Run full test suite to confirm nothing broke**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -x -q`

Expected: All tests pass.

---

### Task 5: Commit All Remaining Changes

Create a clean commit with all the pipeline work from this session that isn't covered by the per-task commits above.

**Files:**
- `src/ai_genre_enrichment/genre_vocabulary.py` (decompose support — already committed in Task 1)
- `src/ai_genre_enrichment/tag_classification.py` (year pattern guard)
- `src/ai_genre_enrichment/storage.py` (alias + decompose at enrichment time, classify_source_tags adjudication chain)
- `src/ai_genre_enrichment/tag_adjudicator.py` (OpenAI API wiring)
- `src/ai_genre_enrichment/lastfm_enrichment.py` (Last.fm API fetcher)
- `src/config_loader.py` (openai_api_key property)
- `scripts/ai_genre_enrich.py` (graduate-ai, extract-lastfm rewrite, --adjudicate flags)
- `data/genre_vocabulary.yaml` (all vocabulary additions)
- `tests/unit/test_ai_genre_enrichment.py` (all new tests)

- [ ] **Step 1: Check what's uncommitted**

```bash
git status
git diff --stat
```

Review the output. Identify any files that should NOT be committed (e.g., `.db` files, `config.yaml` with API keys).

- [ ] **Step 2: Stage source files only**

```bash
git add \
  src/ai_genre_enrichment/genre_vocabulary.py \
  src/ai_genre_enrichment/tag_classification.py \
  src/ai_genre_enrichment/storage.py \
  src/ai_genre_enrichment/tag_adjudicator.py \
  src/ai_genre_enrichment/lastfm_enrichment.py \
  src/config_loader.py \
  scripts/ai_genre_enrich.py \
  data/genre_vocabulary.yaml \
  tests/unit/test_ai_genre_enrichment.py
```

Do NOT stage:
- `data/ai_genre_enrichment.db` (sidecar DB is local data, not source code)
- `config.yaml` (contains API keys — gitignored)
- `data/*.db` test databases (gitignored or local artifacts)

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: unified tag pipeline with AI adjudication cache, Last.fm API, and vocabulary curation

- Wire tag_adjudicator.py to OpenAI Responses API via Config.openai_api_key
- Add ai_tag_adjudication_cache table with first-answer-wins semantics
- Chain classify_source_tags: deterministic → cache → AI → review_only
- Rewrite extract-lastfm to call Last.fm API directly via lastfm_enrichment.py
- Add graduate-ai CLI command for promoting cached terms to vocabulary
- Add decompose rules and alias resolution at enrichment time
- Add year-pattern deterministic guard in tag_classification.py
- Custom YAML serializer to preserve indentation style on save
- Vocabulary: curate aliases, descriptors, labels, formats from live testing"
```

- [ ] **Step 4: Verify clean state**

```bash
git status
pytest tests/unit/test_ai_genre_enrichment.py -x -q
```

Expected: Working tree shows only untracked `.db` files and `config.yaml`. All tests pass.
