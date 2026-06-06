# Unified Genre Store Implementation Plan (SP1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the authoritative layered graph genre assignments (plus taxonomy and user overrides) from the enrichment sidecar into clean, `album_id`-linked tables in `metadata.db`, exposing one materialized `release_effective_genres` table (graph-where-present, legacy-elsewhere) and a small read API.

**Architecture:** A `publish` step ATTACHes the read-only sidecar to a `metadata.db` connection and, in one transaction, (1) copies the taxonomy, (2) resolves `release_key → album_id`, (3) stamps `album_id` onto the graph authority, (4) builds a per-album resolved table choosing graph or legacy, (5) applies user overrides on the resolved rows. All published tables are additive; reversal is a `DROP`. Develop and prove against a *copy* of `metadata.db`; the live run is a gated manual step with a backup.

**Tech Stack:** Python 3.11, stdlib `sqlite3`, pytest. Reuses `src/ai_genre_enrichment/normalization.py`, `src/genre/normalize.py`, `src/ai_genre_enrichment/layered_taxonomy.py`, `src/ai_genre_enrichment/layered_assignment.py`.

Spec: `docs/superpowers/specs/2026-06-06-unified-genre-store-design.md`.

---

## File Structure

- **Create** `src/genre/genre_publish.py` — all publish logic: DDL constants, schema creation, taxonomy copy, key resolution, authority population, legacy aggregation, override classification, resolved-table build, `publish()` orchestrator, `unpublish()`.
- **Create** `src/genre/authority.py` — read API over the published tables.
- **Create** `scripts/publish_genres.py` — thin CLI wrapper.
- **Create** `tests/unit/test_genre_publish.py` — unit tests against synthetic fixture DBs.
- **Create** `tests/unit/test_genre_authority.py` — read API tests.
- **Create** `scripts/validate_published_genres.py` — real-data validation against a copy (safety gate).

### Refinement vs spec (call out to reviewer)

The spec (§Schema (b)) said the **authority** tables carry overrides. For DRYness the plan applies overrides in exactly **one** place — when building `release_effective_genres` — and keeps the published authority tables as a faithful, pure record of what the graph produced. The resolved table is the final answer (`source` ∈ `graph`/`legacy`/`user`). Net behavior for consumers is identical; debugging is easier (authority = graph said X, resolved = final). Flag this when presenting the plan.

---

## Task 1: Published schema module + `create_published_schema`

**Files:**
- Create: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_publish.py
import json
import sqlite3
import pytest
from src.genre import genre_publish


def _meta_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_create_published_schema_creates_all_tables():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {
        "genre_graph_canonical_genres",
        "genre_graph_canonical_facets",
        "genre_graph_edges",
        "genre_graph_aliases",
        "genre_graph_bridge_rules",
        "genre_graph_rejected_terms",
        "genre_graph_taxonomy_meta",
        "genre_graph_release_genre_assignments",
        "genre_graph_release_facet_assignments",
        "release_effective_genres",
    }
    assert expected <= names


def test_create_published_schema_is_idempotent():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    genre_publish.create_published_schema(conn)  # must not raise
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info('release_effective_genres')"
    ).fetchall()}
    assert {"album_id", "release_key", "genre_id", "assignment_layer",
            "confidence", "source"} <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genre.genre_publish'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/genre/genre_publish.py
"""Publish authoritative layered genres from the enrichment sidecar into metadata.db.

See docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Taxonomy + authority DDL mirrors src/ai_genre_enrichment/storage.py so the
# published tables are schema-faithful copies. Authority tables add `album_id`.
_PUBLISHED_DDL = """
CREATE TABLE IF NOT EXISTS genre_graph_canonical_genres (
    genre_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    specificity_score REAL NOT NULL,
    status TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_aliases (
    alias TEXT PRIMARY KEY,
    canonical_genre_id TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_edges (
    source_genre_id TEXT NOT NULL,
    target_genre_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id, edge_type)
);
CREATE TABLE IF NOT EXISTS genre_graph_canonical_facets (
    facet_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    facet_type TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_bridge_rules (
    source_genre_id TEXT NOT NULL,
    target_genre_id TEXT NOT NULL,
    required_family_min REAL NOT NULL,
    required_facet_overlap REAL NOT NULL,
    required_sonic_similarity REAL NOT NULL,
    required_transition_quality REAL NOT NULL,
    mode_allowed TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id)
);
CREATE TABLE IF NOT EXISTS genre_graph_rejected_terms (
    term TEXT PRIMARY KEY,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_taxonomy_meta (
    version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    published_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_release_genre_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
    artist TEXT NOT NULL,
    album TEXT NOT NULL,
    genre_id TEXT NOT NULL,
    assignment_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_reliability REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    rejected_by_user INTEGER NOT NULL DEFAULT 0,
    provenance_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (release_id, genre_id, assignment_layer)
);
CREATE TABLE IF NOT EXISTS genre_graph_release_facet_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
    artist TEXT NOT NULL,
    album TEXT NOT NULL,
    facet_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (release_id, facet_id, source)
);
CREATE TABLE IF NOT EXISTS release_effective_genres (
    album_id TEXT NOT NULL,
    release_key TEXT,
    genre_id TEXT NOT NULL,
    assignment_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (album_id, genre_id, assignment_layer)
);
CREATE INDEX IF NOT EXISTS idx_release_effective_genres_album
    ON release_effective_genres (album_id);
"""

# Tables this sub-project owns. Order matters for DROP (children first is N/A
# since FKs are not enforced, but keep a stable list for unpublish()).
PUBLISHED_TABLES = [
    "release_effective_genres",
    "genre_graph_release_genre_assignments",
    "genre_graph_release_facet_assignments",
    "genre_graph_taxonomy_meta",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
    "genre_graph_canonical_facets",
    "genre_graph_canonical_genres",
]


def create_published_schema(conn: sqlite3.Connection) -> None:
    """Create all published genre tables in metadata.db (idempotent).

    Executes statements individually (NOT executescript, which would implicitly
    COMMIT and break the publish() dry-run rollback). The DDL contains no
    embedded semicolons, so a simple split is safe.
    """
    for stmt in _PUBLISHED_DDL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): published schema for unified genre store (SP1 task 1)"
```

---

## Task 2: Copy taxonomy + write `taxonomy_meta`

The orchestrator ATTACHes the sidecar as schema `side`. This task copies the six taxonomy tables verbatim and writes one `taxonomy_meta` row.

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


def _make_sidecar(tmp_path):
    """A sidecar with the real taxonomy loaded, nothing else."""
    side = tmp_path / "sidecar.db"
    store = SidecarStore(str(side))
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())
    return side


def _attach(meta_conn, side_path):
    # Plain path (no file: URI) to avoid ATTACH URI-mode pitfalls. The sidecar is
    # only ever read (SELECT) by the publish code, so writable-attach is harmless.
    meta_conn.execute("ATTACH DATABASE ? AS side", (str(side_path),))


def test_copy_taxonomy_copies_rows_and_writes_meta(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)

    side_genres = conn.execute(
        "SELECT COUNT(*) FROM side.genre_graph_canonical_genres"
    ).fetchone()[0]
    main_genres = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_canonical_genres"
    ).fetchone()[0]
    assert main_genres == side_genres > 0

    meta = conn.execute(
        "SELECT version, fingerprint, published_at FROM genre_graph_taxonomy_meta"
    ).fetchall()
    assert len(meta) == 1
    assert meta[0]["version"]  # non-empty
    assert len(meta[0]["fingerprint"]) == 64


def test_copy_taxonomy_is_idempotent(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    genre_publish.copy_taxonomy(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM genre_graph_taxonomy_meta"
    ).fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k copy_taxonomy -v`
Expected: FAIL — `AttributeError: module 'src.genre.genre_publish' has no attribute 'copy_taxonomy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py

_TAXONOMY_COPY_TABLES = [
    "genre_graph_canonical_genres",
    "genre_graph_canonical_facets",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
]


def _taxonomy_fingerprint(conn: sqlite3.Connection) -> str:
    """Stable hash of the published taxonomy (genres + edges)."""
    genres = conn.execute(
        "SELECT genre_id, name, kind, specificity_score, status "
        "FROM genre_graph_canonical_genres ORDER BY genre_id"
    ).fetchall()
    edges = conn.execute(
        "SELECT source_genre_id, target_genre_id, edge_type, weight "
        "FROM genre_graph_edges ORDER BY source_genre_id, target_genre_id, edge_type"
    ).fetchall()
    payload = json.dumps(
        {"genres": [tuple(r) for r in genres], "edges": [tuple(r) for r in edges]},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_taxonomy(conn: sqlite3.Connection) -> None:
    """Copy taxonomy tables from attached `side` DB; rewrite taxonomy_meta.

    Requires the sidecar attached as schema `side`.
    """
    for table in _TAXONOMY_COPY_TABLES:
        conn.execute(f"DELETE FROM {table}")
        conn.execute(f"INSERT INTO {table} SELECT * FROM side.{table}")
    version_row = conn.execute(
        "SELECT taxonomy_version FROM genre_graph_canonical_genres LIMIT 1"
    ).fetchone()
    version = version_row[0] if version_row else "unknown"
    conn.execute("DELETE FROM genre_graph_taxonomy_meta")
    conn.execute(
        "INSERT INTO genre_graph_taxonomy_meta (version, fingerprint, published_at) "
        "VALUES (?, ?, ?)",
        (version, _taxonomy_fingerprint(conn), _now_iso()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k copy_taxonomy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): copy taxonomy + taxonomy_meta into metadata.db (SP1 task 2)"
```

---

## Task 3: Resolve `release_key → album_id`

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def _make_metadata(tmp_path):
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
                             album_id TEXT, norm_artist TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        """
    )
    conn.commit()
    conn.close()
    return meta


def test_resolve_keys_uses_signature_then_albums(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    # signature row maps a release_key to an album_id
    sconn = sqlite3.connect(side)
    sconn.execute(
        "INSERT INTO enriched_genre_signatures "
        "(release_key, normalized_artist, normalized_album, album_id, signature_json) "
        "VALUES (?,?,?,?,?)",
        ("acetone::york blvd", "acetone", "york blvd", "ALB_SIG", "{}"),
    )
    sconn.commit(); sconn.close()
    # albums table can recompute a different key via normalizers
    mconn = sqlite3.connect(meta)
    mconn.execute("INSERT INTO albums VALUES ('ALB_CALC', 'Rocket', '(Sandy) Alex G')")
    mconn.commit(); mconn.close()

    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    _attach(conn, side)
    mapping, collisions = genre_publish.resolve_release_key_to_album_id(conn)
    assert mapping["acetone::york blvd"] == "ALB_SIG"      # signature path
    # recomputed path: normalize_release_artist/name of the albums row
    from src.ai_genre_enrichment.normalization import (
        normalize_release_artist, normalize_release_name)
    calc_key = f"{normalize_release_artist('(Sandy) Alex G')}::{normalize_release_name('Rocket')}"
    assert mapping[calc_key] == "ALB_CALC"
    assert collisions == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k resolve_keys -v`
Expected: FAIL — no attribute `resolve_release_key_to_album_id`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py
from src.ai_genre_enrichment.normalization import (
    normalize_release_artist,
    normalize_release_name,
)


def resolve_release_key_to_album_id(
    conn: sqlite3.Connection,
) -> tuple[dict[str, str], int]:
    """Build release_key -> album_id. Signatures win; albums recompute fills gaps.

    Returns (mapping, collision_count). Requires sidecar attached as `side`.
    """
    mapping: dict[str, str] = {}

    # 1) exact from signatures
    for row in conn.execute(
        "SELECT release_key, album_id FROM side.enriched_genre_signatures "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        mapping[row[0]] = row[1]

    # 2) recompute from albums for keys not already mapped
    collisions = 0
    computed: dict[str, str] = {}
    for album_id, title, artist in conn.execute(
        "SELECT album_id, title, artist FROM albums "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        key = f"{normalize_release_artist(artist)}::{normalize_release_name(title)}"
        if not key or key == "::":
            continue
        if key in computed and computed[key] != album_id:
            collisions += 1
            # deterministic: keep the lexicographically smaller album_id
            computed[key] = min(computed[key], album_id)
        else:
            computed[key] = album_id
    for key, album_id in computed.items():
        mapping.setdefault(key, album_id)

    return mapping, collisions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k resolve_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): resolve release_key to album_id (SP1 task 3)"
```

---

## Task 4: Populate authority tables with `album_id`

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def _insert_graph_assignment(side_path, release_id, artist, album, genre_id, layer):
    sconn = sqlite3.connect(side_path)
    sconn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (release_id, artist, album, genre_id, layer, 0.9, 0.7, 2, 0, "{}", "t"),
    )
    sconn.commit(); sconn.close()


def test_populate_authority_stamps_album_id(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    _insert_graph_assignment(side, "acetone::york blvd", "acetone", "york blvd",
                             "alternative_rock", "observed_leaf")
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    mapping = {"acetone::york blvd": "ALB1"}
    genre_publish.populate_authority(conn, mapping)
    rows = conn.execute(
        "SELECT album_id, genre_id FROM genre_graph_release_genre_assignments"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["album_id"] == "ALB1"
    assert rows[0]["genre_id"] == "alternative_rock"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k populate_authority -v`
Expected: FAIL — no attribute `populate_authority`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py

def populate_authority(conn: sqlite3.Connection, key_to_album: dict[str, str]) -> None:
    """Copy graph genre + facet assignments into metadata.db, stamping album_id.

    Requires sidecar attached as `side`. Pure graph (no overrides here).
    """
    conn.execute("DELETE FROM genre_graph_release_genre_assignments")
    conn.execute("DELETE FROM genre_graph_release_facet_assignments")

    for row in conn.execute(
        "SELECT release_id, artist, album, genre_id, assignment_layer, confidence, "
        "source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at "
        "FROM side.genre_graph_release_genre_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_genre_assignments "
            "(release_id, album_id, artist, album, genre_id, assignment_layer, confidence, "
            " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5],
             row[6], row[7], row[8], row[9], row[10]),
        )

    for row in conn.execute(
        "SELECT release_id, artist, album, facet_id, confidence, source, "
        "provenance_json, updated_at FROM side.genre_graph_release_facet_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_facet_assignments "
            "(release_id, album_id, artist, album, facet_id, confidence, source, "
            " provenance_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5], row[6], row[7]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k populate_authority -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): populate authority tables with album_id (SP1 task 4)"
```

---

## Task 5: Legacy album-grain aggregation

Mirrors `load_genres_for_tracks` weights (track 1.0 / album 0.8 / artist 0.5) and `normalize_and_split_genre`, aggregated to `album_id` with max weight per token.

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def test_legacy_genres_by_album_aggregates_sources(tmp_path):
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'Some Album', 'Some Artist')")
    conn.execute("INSERT INTO tracks VALUES ('T1', 'Some Artist', 'Some Album', 'ALB1', 'some artist')")
    conn.execute("INSERT INTO track_genres VALUES ('T1', 'Slowcore', 'file', 1.0)")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', 'Indie Rock', 'discogs_release')")
    conn.execute("INSERT INTO artist_genres VALUES ('Some Artist', 'Rock', 'musicbrainz_artist')")
    conn.commit()
    result = genre_publish.legacy_genres_by_album(conn)
    assert "ALB1" in result
    genres = {g for g, _w in result["ALB1"]}
    # normalized tokens from the three sources are all present
    assert "slowcore" in genres
    assert "rock" in genres


def test_legacy_genres_skip_empty_marker(tmp_path):
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'A', 'X')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', '__EMPTY__', 'discogs_release')")
    conn.commit()
    result = genre_publish.legacy_genres_by_album(conn)
    assert "ALB1" not in result or result["ALB1"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k legacy_genres -v`
Expected: FAIL — no attribute `legacy_genres_by_album`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py
from collections import defaultdict

try:
    from src.genre.normalize import normalize_and_split_genre
    _NORMALIZE_AVAILABLE = True
except Exception:  # pragma: no cover - normalization optional
    _NORMALIZE_AVAILABLE = False

_WEIGHT_TRACK = 1.0
_WEIGHT_ALBUM = 0.8
_WEIGHT_ARTIST = 0.5


def _split(raw: str) -> list[str]:
    if not raw or raw == "__EMPTY__":
        return []
    if _NORMALIZE_AVAILABLE:
        return [t for t in normalize_and_split_genre(raw) if t]
    token = raw.strip().casefold()
    return [token] if token else []


def legacy_genres_by_album(conn: sqlite3.Connection) -> dict[str, list[tuple[str, float]]]:
    """Album-grain legacy genres: track(1.0)+album(0.8)+artist(0.5), max weight/token."""
    acc: dict[str, dict[str, float]] = defaultdict(dict)

    def add(album_id: str, raw: str, base_weight: float) -> None:
        tokens = _split(raw)
        if not tokens:
            return
        per = base_weight / len(tokens)
        for tok in tokens:
            if per > acc[album_id].get(tok, 0.0):
                acc[album_id][tok] = per

    for album_id, genre in conn.execute(
        "SELECT t.album_id, tg.genre FROM tracks t "
        "JOIN track_genres tg ON tg.track_id = t.track_id "
        "WHERE t.album_id IS NOT NULL AND t.album_id != ''"
    ):
        add(album_id, genre, _WEIGHT_TRACK)

    for album_id, genre in conn.execute(
        "SELECT album_id, genre FROM album_genres "
        "WHERE album_id IS NOT NULL AND album_id != '' AND genre != '__EMPTY__'"
    ):
        add(album_id, genre, _WEIGHT_ALBUM)

    for album_id, genre in conn.execute(
        "SELECT a.album_id, ag.genre FROM albums a "
        "JOIN artist_genres ag ON ag.artist = a.artist "
        "WHERE a.album_id IS NOT NULL AND a.album_id != '' AND ag.genre != '__EMPTY__'"
    ):
        add(album_id, genre, _WEIGHT_ARTIST)

    return {aid: sorted(toks.items()) for aid, toks in acc.items() if toks}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k legacy_genres -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): album-grain legacy genre aggregation (SP1 task 5)"
```

---

## Task 6: Override term classification helper

Maps override add/remove genre *names* to graph `genre_id`s via the taxonomy classifier.

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def test_classify_override_terms_maps_names_to_ids():
    taxonomy = load_default_layered_taxonomy()
    add_ids, remove_ids = genre_publish.classify_override_terms(
        taxonomy, add=["slowcore"], remove=["indie rock"]
    )
    # slowcore is a known leaf -> mapped to its genre_id
    assert any("slowcore" in gid for gid in add_ids)
    # indie rock maps to a known canonical genre id (e.g. indie_rock)
    assert remove_ids  # non-empty


def test_classify_override_terms_skips_unmappable():
    taxonomy = load_default_layered_taxonomy()
    add_ids, remove_ids = genre_publish.classify_override_terms(
        taxonomy, add=["zzzz not a genre zzzz"], remove=[]
    )
    assert add_ids == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k classify_override -v`
Expected: FAIL — no attribute `classify_override_terms`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py
from src.ai_genre_enrichment.layered_assignment import classify_layered_term


def _term_to_genre_id(taxonomy, term: str) -> str | None:
    """Return a graph genre_id for a term, or None if unmappable/non-genre."""
    classification = classify_layered_term(taxonomy, term)
    if classification.term_kind in {"reject", "review", "facet", "alias"}:
        # facets/aliases handled elsewhere or skipped; reject/review never applied
        if classification.term_kind == "alias" and classification.canonical_id:
            # alias to a genre is acceptable as the canonical genre id
            if taxonomy.genre_by_id(classification.canonical_id) is not None:
                return classification.canonical_id
        return None
    return classification.canonical_id


def classify_override_terms(
    taxonomy, add: list[str], remove: list[str]
) -> tuple[list[str], list[str]]:
    """Map override add/remove names to graph genre_ids (unmappable skipped)."""
    add_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in add) if gid]
    remove_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in remove) if gid]
    # de-dup preserving order
    return list(dict.fromkeys(add_ids)), list(dict.fromkeys(remove_ids))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k classify_override -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): override term classification (SP1 task 6)"
```

---

## Task 7: Build `release_effective_genres` (graph/legacy + overrides)

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def _insert_override(side_path, release_key, artist, album, add, remove):
    sconn = sqlite3.connect(side_path)
    sconn.execute(
        "INSERT INTO ai_genre_user_overrides "
        "(release_key, normalized_artist, normalized_album, genres_add_json, "
        " genres_remove_json, updated_at) VALUES (?,?,?,?,?,?)",
        (release_key, artist, album, json.dumps(add), json.dumps(remove), "t"),
    )
    sconn.commit(); sconn.close()


def test_build_resolved_graph_album_uses_graph_rows(tmp_path):
    import json
    meta = _make_metadata(tmp_path); side = _make_sidecar(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone')")
    conn.commit()
    genre_publish.create_published_schema(conn); _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    # graph authority for ALB1
    conn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, album_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES ('acetone::york blvd','ALB1','acetone','york blvd','alternative_rock',"
        " 'observed_leaf',0.9,0.7,2,0,'{}','t')"
    )
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={"acetone::york blvd": "ALB1"},
                                       taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB1'").fetchall()
    assert ("alternative_rock", "graph") in [(r["genre_id"], r["source"]) for r in rows]


def test_build_resolved_legacy_album_uses_legacy_rows(tmp_path):
    meta = _make_metadata(tmp_path); side = _make_sidecar(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB2', 'B', 'Y')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release')")
    conn.commit()
    genre_publish.create_published_schema(conn); _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={}, taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB2'").fetchall()
    assert rows and all(r["source"] == "legacy" for r in rows)
    assert "slowcore" in {r["genre_id"] for r in rows}


def test_build_resolved_applies_overrides(tmp_path):
    meta = _make_metadata(tmp_path); side = _make_sidecar(tmp_path)
    _insert_override(side, "y::b", "y", "b", add=["slowcore"], remove=["jangle pop"])
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB3', 'B', 'Y')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB3', 'Jangle Pop', 'discogs_release')")
    conn.commit()
    genre_publish.create_published_schema(conn); _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={"y::b": "ALB3"}, taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB3'").fetchall()
    ids = {r["genre_id"] for r in rows}
    # added (as user) present; removed legacy token gone
    assert any("slowcore" in gid for gid in ids)
    assert "jangle pop" not in ids
    assert ("slowcore", "user") in [(r["genre_id"], r["source"]) for r in rows] \
        or any(r["source"] == "user" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k build_resolved -v`
Expected: FAIL — no attribute `build_resolved_table`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py

def _overrides_by_album(conn, key_to_album, taxonomy):
    """album_id -> (add_genre_ids, remove_match) from ai_genre_user_overrides.

    `remove_match` covers BOTH vocab spaces: graph genre_ids (to drop graph rows)
    AND normalized free-text tokens (to drop legacy rows), since graph and legacy
    genres live in different vocabularies.
    """
    out: dict[str, tuple[list[str], set[str]]] = {}
    for release_key, add_json, remove_json in conn.execute(
        "SELECT release_key, genres_add_json, genres_remove_json FROM side.ai_genre_user_overrides"
    ):
        album_id = key_to_album.get(release_key)
        if not album_id:
            continue
        add = json.loads(add_json or "[]")
        remove = json.loads(remove_json or "[]")
        add_ids, remove_ids = classify_override_terms(taxonomy, add, remove)
        remove_tokens: set[str] = set()
        for name in remove:
            remove_tokens.update(_split(name))
        out[album_id] = (add_ids, set(remove_ids) | remove_tokens)
    return out


def build_resolved_table(conn, key_to_album: dict[str, str], taxonomy) -> None:
    """Build release_effective_genres: graph-where-present else legacy, + overrides."""
    conn.execute("DELETE FROM release_effective_genres")

    # invert key_to_album for release_key provenance per album
    album_to_key: dict[str, str] = {}
    for key, aid in key_to_album.items():
        album_to_key.setdefault(aid, key)

    graph_album_ids = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT album_id FROM genre_graph_release_genre_assignments "
            "WHERE album_id IS NOT NULL"
        )
    }
    legacy = legacy_genres_by_album(conn)
    overrides = _overrides_by_album(conn, key_to_album, taxonomy)

    all_album_ids = [
        r[0] for r in conn.execute(
            "SELECT album_id FROM albums WHERE album_id IS NOT NULL AND album_id != ''"
        )
    ]

    for album_id in all_album_ids:
        rows: dict[tuple[str, str], tuple[float, str]] = {}  # (genre_id,layer)->(conf,source)
        if album_id in graph_album_ids:
            for genre_id, layer, conf in conn.execute(
                "SELECT genre_id, assignment_layer, confidence "
                "FROM genre_graph_release_genre_assignments WHERE album_id = ?",
                (album_id,),
            ):
                rows[(genre_id, layer)] = (conf, "graph")
        elif album_id in legacy:
            for genre_id, weight in legacy[album_id]:
                rows[(genre_id, "legacy")] = (weight, "legacy")

        # overrides: remove (both vocab spaces) then add (as user)
        if album_id in overrides:
            add_ids, remove_match = overrides[album_id]
            rows = {k: v for k, v in rows.items() if k[0] not in remove_match}
            for gid in add_ids:
                rows[(gid, "observed_leaf")] = (1.0, "user")

        release_key = album_to_key.get(album_id)
        for (genre_id, layer), (conf, source) in rows.items():
            conn.execute(
                "INSERT OR REPLACE INTO release_effective_genres "
                "(album_id, release_key, genre_id, assignment_layer, confidence, source) "
                "VALUES (?,?,?,?,?,?)",
                (album_id, release_key, genre_id, layer, conf, source),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k build_resolved -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): build resolved release_effective_genres (SP1 task 7)"
```

---

## Task 8: `publish()` orchestrator + `unpublish()` + dry-run + idempotency

**Files:**
- Modify: `src/genre/genre_publish.py`
- Test: `tests/unit/test_genre_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def _full_fixture(tmp_path):
    meta = _make_metadata(tmp_path); side = _make_sidecar(tmp_path)
    mconn = sqlite3.connect(meta)
    mconn.execute("INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone')")
    mconn.execute("INSERT INTO albums VALUES ('ALB2', 'B', 'Y')")
    mconn.execute("INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release')")
    mconn.commit(); mconn.close()
    _insert_graph_assignment(side, "acetone::york blvd", "acetone", "york blvd",
                             "alternative_rock", "observed_leaf")
    return meta, side


def test_publish_end_to_end_and_stats(tmp_path):
    meta, side = _full_fixture(tmp_path)
    stats = genre_publish.publish(str(meta), str(side), dry_run=False)
    assert stats.graph_albums == 1
    assert stats.legacy_albums == 1
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    src = {r["album_id"]: r["source"] for r in conn.execute(
        "SELECT album_id, source FROM release_effective_genres GROUP BY album_id")}
    assert src["ALB1"] == "graph"
    assert src["ALB2"] == "legacy"


def test_publish_is_idempotent(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    first = conn.execute("SELECT * FROM release_effective_genres ORDER BY album_id, genre_id, assignment_layer").fetchall()
    conn.close()
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    second = conn.execute("SELECT * FROM release_effective_genres ORDER BY album_id, genre_id, assignment_layer").fetchall()
    conn.close()
    assert first == second


def test_publish_dry_run_writes_nothing(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=True)
    conn = sqlite3.connect(meta)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "release_effective_genres" not in names  # rolled back


def test_unpublish_drops_published_only(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    genre_publish.unpublish(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "release_effective_genres" not in names
    assert "albums" in names and "album_genres" in names  # legacy intact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k "publish or unpublish" -v`
Expected: FAIL — no attribute `publish`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/genre/genre_publish.py
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


@dataclass
class PublishStats:
    total_albums: int = 0
    graph_albums: int = 0
    legacy_albums: int = 0
    unlinked_releases: int = 0
    collisions: int = 0
    overrides_applied: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def unpublish(conn: sqlite3.Connection) -> None:
    """Drop all published tables. Legacy tables untouched."""
    for table in PUBLISHED_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def publish(metadata_db: str, sidecar_db: str, dry_run: bool = False) -> PublishStats:
    """Publish authoritative genres from sidecar into metadata.db (one transaction)."""
    taxonomy = load_default_layered_taxonomy()
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    # Autocommit mode: we control BEGIN/COMMIT/ROLLBACK manually so the dry-run
    # rollback (including transactional DDL) works. ATTACH must precede BEGIN
    # (SQLite forbids ATTACH inside a transaction). Plain path, read-only by use.
    conn.isolation_level = None
    try:
        conn.execute("ATTACH DATABASE ? AS side", (sidecar_db,))
        conn.execute("BEGIN")
        create_published_schema(conn)
        copy_taxonomy(conn)
        mapping, collisions = resolve_release_key_to_album_id(conn)
        populate_authority(conn, mapping)
        build_resolved_table(conn, mapping, taxonomy)

        # stats
        graph = {r[0] for r in conn.execute(
            "SELECT DISTINCT album_id FROM release_effective_genres WHERE source='graph'")}
        legacy = {r[0] for r in conn.execute(
            "SELECT DISTINCT album_id FROM release_effective_genres WHERE source='legacy'")}
        total = conn.execute(
            "SELECT COUNT(*) FROM albums WHERE album_id IS NOT NULL AND album_id != ''"
        ).fetchone()[0]
        unlinked = conn.execute(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments WHERE album_id IS NULL"
        ).fetchone()[0]
        overrides_applied = conn.execute(
            "SELECT COUNT(*) FROM release_effective_genres WHERE source='user'"
        ).fetchone()[0]
        stats = PublishStats(
            total_albums=total, graph_albums=len(graph), legacy_albums=len(legacy),
            unlinked_releases=unlinked, collisions=collisions,
            overrides_applied=overrides_applied, dry_run=dry_run,
        )
        if dry_run:
            conn.execute("ROLLBACK")
        else:
            conn.execute("COMMIT")
        return stats
    finally:
        try:
            conn.execute("DETACH DATABASE side")
        except sqlite3.OperationalError:
            pass
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -v`
Expected: PASS (all tests in file).

- [ ] **Step 5: Commit**

```bash
git add src/genre/genre_publish.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): publish orchestrator + unpublish + dry-run (SP1 task 8)"
```

---

## Task 9: Read API — `src/genre/authority.py`

**Files:**
- Create: `src/genre/authority.py`
- Test: `tests/unit/test_genre_authority.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_authority.py
import sqlite3
import pytest
from src.genre import genre_publish, authority


def _published_db(tmp_path):
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
                             album_id TEXT, norm_artist TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone');
        INSERT INTO albums VALUES ('ALB2', 'B', 'Y');
        INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release');
        INSERT INTO tracks VALUES ('T1', 'Acetone', 'York Blvd', 'ALB1', 'acetone');
        """
    )
    conn.commit(); conn.close()
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    side = tmp_path / "sidecar.db"
    store = SidecarStore(str(side)); store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())
    sconn = sqlite3.connect(side)
    sconn.execute(
        "INSERT INTO enriched_genre_signatures "
        "(release_key, normalized_artist, normalized_album, album_id, signature_json) "
        "VALUES ('acetone::york blvd','acetone','york blvd','ALB1','{}')")
    sconn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES ('acetone::york blvd','acetone','york blvd','alternative_rock',"
        " 'observed_leaf',0.9,0.7,2,0,'{}','t')")
    sconn.commit(); sconn.close()
    genre_publish.publish(str(meta), str(side), dry_run=False)
    return meta


def test_resolved_genres_for_album(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    rows = authority.resolved_genres_for_album(conn, "ALB1")
    assert any(r.genre_id == "alternative_rock" and r.source == "graph" for r in rows)


def test_resolved_genres_for_track(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    rows = authority.resolved_genres_for_track(conn, "T1")
    assert any(r.genre_id == "alternative_rock" for r in rows)


def test_genre_source_for_album(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    assert authority.genre_source_for_album(conn, "ALB1") == "graph"
    assert authority.genre_source_for_album(conn, "ALB2") == "legacy"
    assert authority.genre_source_for_album(conn, "NOPE") == "none"


def test_taxonomy_helpers(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta); conn.row_factory = sqlite3.Row
    # alternative_rock should have at least one family ancestor
    fams = authority.families_for(conn, "alternative_rock")
    assert isinstance(fams, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_authority.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genre.authority'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/genre/authority.py
"""Read API for the published unified genre store in metadata.db.

Single import point for playlist features (SP2+). All reads come from the
materialized release_effective_genres table; taxonomy-structure helpers delegate
to the loaded LayeredTaxonomy (version-matched to the published taxonomy_meta).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GenreRow:
    genre_id: str
    assignment_layer: str
    confidence: float
    source: str


def resolved_genres_for_album(conn: sqlite3.Connection, album_id: str) -> list[GenreRow]:
    rows = conn.execute(
        "SELECT genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres WHERE album_id = ? "
        "ORDER BY assignment_layer, genre_id",
        (album_id,),
    ).fetchall()
    return [GenreRow(r[0], r[1], r[2], r[3]) for r in rows]


def resolved_genres_for_track(conn: sqlite3.Connection, track_id: str) -> list[GenreRow]:
    row = conn.execute(
        "SELECT album_id FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()
    if not row or not row[0]:
        return []
    return resolved_genres_for_album(conn, row[0])


def genre_source_for_album(conn: sqlite3.Connection, album_id: str) -> str:
    row = conn.execute(
        "SELECT source FROM release_effective_genres WHERE album_id = ? LIMIT 1",
        (album_id,),
    ).fetchone()
    if not row:
        return "none"
    # 'user' overrides ride on top of a base source; report the base
    base = conn.execute(
        "SELECT source FROM release_effective_genres "
        "WHERE album_id = ? AND source != 'user' LIMIT 1",
        (album_id,),
    ).fetchone()
    return base[0] if base else "none"


@lru_cache(maxsize=1)
def _taxonomy():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    return load_default_layered_taxonomy()


def parents_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().parents_for_genre(genre_id)]


def families_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().families_for_genre(genre_id)]


def is_facet(conn: sqlite3.Connection, genre_id: str) -> bool:
    return _taxonomy().facet_by_id(genre_id) is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_authority.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/genre/authority.py tests/unit/test_genre_authority.py
git commit -m "feat(genre): unified genre store read API (SP1 task 9)"
```

---

## Task 10: CLI — `scripts/publish_genres.py`

**Files:**
- Create: `scripts/publish_genres.py`
- Test: `tests/unit/test_genre_publish.py` (add CLI smoke test)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_genre_publish.py

def test_cli_main_runs_dry_run(tmp_path, capsys):
    meta, side = _full_fixture(tmp_path)
    from scripts import publish_genres
    rc = publish_genres.main([
        "--metadata-db", str(meta), "--sidecar-db", str(side), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "graph_albums" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genre_publish.py -k cli_main -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.publish_genres'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/publish_genres.py
#!/usr/bin/env python3
"""Publish authoritative layered genres from the sidecar into metadata.db.

SAFETY: develop/validate against a COPY of metadata.db first. The live run
requires a fresh timestamped backup and explicit confirmation. See
docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.genre.genre_publish import publish


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish unified genre store into metadata.db")
    parser.add_argument("--metadata-db", default=str(ROOT / "data" / "metadata.db"))
    parser.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print stats, then roll back (no writes).")
    args = parser.parse_args(argv)
    stats = publish(args.metadata_db, args.sidecar_db, dry_run=args.dry_run)
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genre_publish.py -k cli_main -v`
Expected: PASS.

- [ ] **Step 5: Run the full SP1 suite + ruff**

Run: `pytest tests/unit/test_genre_publish.py tests/unit/test_genre_authority.py -v && ruff check src/genre/genre_publish.py src/genre/authority.py scripts/publish_genres.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/publish_genres.py tests/unit/test_genre_publish.py
git commit -m "feat(genre): publish-genres CLI (SP1 task 10)"
```

---

## Task 11: Real-data validation gate (against a COPY) + live-run runbook

This task is the safety gate. It does **not** write to the live `metadata.db`. It creates a throwaway copy, publishes against it, and validates.

**Files:**
- Create: `scripts/validate_published_genres.py`

- [ ] **Step 1: Write the validation script**

```python
# scripts/validate_published_genres.py
#!/usr/bin/env python3
"""Validate a published copy of metadata.db. Read-only checks; no live writes."""
import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def validate(meta_db: str) -> int:
    conn = sqlite3.connect(meta_db); conn.row_factory = sqlite3.Row
    problems = []

    # every album resolves to exactly one base source (graph xor legacy), or none
    bad = conn.execute(
        "SELECT album_id, COUNT(DISTINCT source) c FROM release_effective_genres "
        "WHERE source != 'user' GROUP BY album_id HAVING c > 1"
    ).fetchall()
    if bad:
        problems.append(f"{len(bad)} albums have >1 base source")

    n_albums = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
    n_resolved = conn.execute(
        "SELECT COUNT(DISTINCT album_id) FROM release_effective_genres").fetchone()[0]
    print(f"albums={n_albums} resolved_albums={n_resolved}")
    by_source = conn.execute(
        "SELECT source, COUNT(DISTINCT album_id) FROM release_effective_genres GROUP BY source"
    ).fetchall()
    print("by source:", [(r[0], r[1]) for r in by_source])

    # spot-checks (only assert if the album is present)
    spot = {
        "a tribe called quest::midnight marauders japanese": {"east_coast_hip_hop", "jazz_rap", "boom_bap"},
        "antonio carlos jobim::wave": {"bossa_nova", "latin_jazz", "mpb"},
    }
    for rk, expected in spot.items():
        got = {r[0] for r in conn.execute(
            "SELECT genre_id FROM release_effective_genres WHERE release_key = ?", (rk,))}
        missing = expected - got
        if got and missing:
            problems.append(f"{rk}: missing {missing}")
        elif got:
            print(f"OK spot-check {rk}: {sorted(expected & got)}")

    conn.close()
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("VALIDATION OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-db", required=True, help="Path to the COPY to validate")
    return validate(p.parse_args(argv).metadata_db)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the full pipeline against a copy (PowerShell)**

```powershell
$py = "C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe"
Copy-Item data\metadata.db data\metadata.db.worktest -Force
& $py scripts\publish_genres.py --metadata-db data\metadata.db.worktest --dry-run
& $py scripts\publish_genres.py --metadata-db data\metadata.db.worktest
& $py scripts\validate_published_genres.py --metadata-db data\metadata.db.worktest
```
Expected: dry-run prints stats; real run against the copy prints stats; validation prints `VALIDATION OK` and the spot-checks.

- [ ] **Step 3: Inspect counts for sanity**

Confirm `graph_albums + legacy_albums ≈ total_albums`, `collisions` is small, `unlinked_releases` is understood (graph releases whose key didn't resolve to an album). If counts look wrong, STOP and diagnose before any live run.

- [ ] **Step 4: Commit the validator**

```bash
git add scripts/validate_published_genres.py
git commit -m "feat(genre): real-data validation script for published store (SP1 task 11)"
```

- [ ] **Step 5: LIVE RUN — gated, manual, do NOT automate**

This is the only step that writes the real `metadata.db`. Per project rules it
requires a backup and a second explicit confirmation **from the user at that
moment**. Present the validated copy's stats, then:

```powershell
# Only after the user explicitly confirms:
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item data\metadata.db "data\metadata.db.bak.$ts" -Force
& $py scripts\publish_genres.py --metadata-db data\metadata.db
& $py scripts\validate_published_genres.py --metadata-db data\metadata.db
```

Do not run Step 5 as part of automated plan execution. Stop and hand back to the
user when Steps 1–4 are green.

---

## Self-Review

**Spec coverage:**
- Schema (taxonomy/authority/resolved + taxonomy_meta) → Tasks 1, 2.
- Publish algorithm steps 1–5 → Tasks 2 (taxonomy), 3 (keys), 4 (authority), 5 (legacy), 6+7 (overrides+resolved), 8 (orchestrator/dry-run).
- album_id linkage (signature + recompute + collisions) → Task 3.
- Override application (map names→ids, remove/add, win) → Tasks 6, 7.
- Read API (album/track/source + taxonomy helpers) → Task 9.
- Safety/reversibility (copy-first, backup, dry-run, unpublish) → Tasks 8, 11.
- Testing (unit + real-data validation) → Tasks 1–10 (unit), 11 (real-data).
- Out-of-scope items remain untouched (no artifact/engine changes).

**Refinement flagged:** overrides applied only at resolved-table build (one place), authority kept pure — noted in File Structure section; surface to reviewer.

**Type/name consistency:** `publish(metadata_db, sidecar_db, dry_run)`, `PublishStats` fields (`total_albums`, `graph_albums`, `legacy_albums`, `unlinked_releases`, `collisions`, `overrides_applied`, `dry_run`), `GenreRow(genre_id, assignment_layer, confidence, source)`, `resolve_release_key_to_album_id → (mapping, collisions)`, `build_resolved_table(conn, key_to_album, taxonomy)`, `classify_override_terms(taxonomy, add, remove) → (add_ids, remove_ids)` — all used consistently across tasks.

**Placeholder scan:** none — every step contains runnable code and exact commands.
