# Analyze-Library Graph Stages Implementation Plan (Phase 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three incremental stages — `lastfm`, `enrich`, `publish` — to `scripts/analyze_library.py` so a routine "Analyze Library" run scrapes Last.fm tags, runs Claude-backed genre enrichment into the sidecar (`ai_genre_enrichment.db`), and publishes authoritative genres into `release_effective_genres` in `metadata.db`.

**Architecture:** Each new stage follows the existing `stage_<name>(ctx) -> dict` contract: a function registered in `STAGE_FUNCS`, a branch in `compute_stage_fingerprint`, a branch in `estimate_stage_units`, and a slot in `STAGE_ORDER_DEFAULT`. Stages chain through the existing **fingerprint** mechanism (each stage's fingerprint reads live DB state the previous stage just changed), so no new dirty flags are required. The `enrich` stage de-duplicates unknown tags across all pending releases and adjudicates them in chunked Claude calls (one classification per distinct tag, library-wide), filling the `ai_tag_adjudication_cache`; per-release classification then becomes pure cache hits. The `publish` stage backs up `metadata.db` once (first publish only) and writes only derived genre tables via the atomic, idempotent `genre_publish.publish()`.

**Tech Stack:** Python 3.11, SQLite (sidecar + metadata.db), the Phase-1 Claude provider factory, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-analyze-library-graph-claude-design.md` (Phase 2 section). Phase 1 (Claude-Code backend) shipped (commits `c770050`..`0274e66`). Phase 3 (web Tools panel) is a later plan.

---

## Key facts established by exploration (do not re-derive)

**Stage machinery in `scripts/analyze_library.py`:**
- `STAGE_ORDER_DEFAULT` (line 43) lists the default stage names in order.
- `STAGE_FUNCS` (lines 1316-1326) maps name → `stage_<name>` function.
- `compute_stage_fingerprint(ctx, stage)` (line 146) has one `if stage == "...":` branch per stage, returning `_hash_obj(key_dict)`. **Bug to avoid: the existing `verify` branch (lines 302-313) falls through to a second `return` that is dead code — match the pattern of `scan`/`genres` (each branch returns its own `_hash_obj(...)`).**
- `estimate_stage_units(ctx, stage)` (line 316) returns `(count, label)`; unknown → `(None, None)`.
- `ctx` keys available to every stage: `conn` (a `sqlite3.Row`-row metadata.db connection, autocommit), `db_path` (str), `config_path` (str), `out_dir` (Path), `args` (argparse.Namespace), `config_hash`, `library_root`, plus the mutable dirty flags `genres_dirty`/`sonic_dirty`/`artifacts_dirty`/`force_stage`.
- The run loop (lines 1474-1590) computes `fingerprint_before` **before** calling the stage, skips when `fingerprint_before == last_success_fingerprint` (unless `--force`), runs the stage, then stores `fingerprint_after`. Because fingerprints read live DB state, a stage that changes the sidecar makes the *next* stage's `fingerprint_before` differ from its last success → the next stage runs. This is the chaining mechanism.
- `ENRICHMENT_DB_PATH = ROOT_DIR / "data" / "ai_genre_enrichment.db"` already exists (line 42).

**Reusable callables (exact import paths, all verified):**
- `from src.ai_genre_enrichment.storage import SidecarStore` — `.initialize()`, `.upsert_source_page(...)`, `.replace_source_tags(page_id, tags)`, `.classify_source_tags(page_id, *, adjudicate=False, model=None) -> bool`, `.rebuild_enriched_genres_for_release(release_key)`, `.release_keys_with_source_type(source_type) -> set[str]`, `.lookup_cached_adjudication(normalized_tag) -> dict|None`, `.cache_adjudication(*, normalized_tag, classification, confidence, classifier="ai")`, `.upsert_layered_taxonomy(taxonomy)`.
- `from src.ai_genre_enrichment.discovery import discover_releases` — `discover_releases(metadata_db_path, *, limit=None, artist=None, album=None, generic_only=False, min_existing_specific_genres=None, track_title_cap=25) -> list[ReleasePayload]`. `ReleasePayload` fields used here: `.release_key`, `.normalized_artist`, `.normalized_album`, `.album_id`, `.existing_genres_by_source` (dict).
- `from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags` — `fetch_lastfm_tags(artist, album, api_key, limit=20) -> list[str]`.
- `from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags` — `adjudicate_tags(tags: list[tuple[str,str]], *, model=None, dry_run=False, client=None) -> dict[norm, {"classification","confidence","reason"}]`. Builds a Claude client via the factory when `client=None`. **Raises** on backend failure.
- `from src.ai_genre_enrichment.provider import resolve_enrichment_model` — `resolve_enrichment_model(None) -> str`.
- `from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy`.
- `from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments` — `materialize_layered_assignments(store, *, release_id, artist, album, report, taxonomy) -> summary` with `.genre_assignment_count`, `.facet_assignment_count`, `.rejected_term_count`, `.review_term_count`.
- `from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, collect_hybrid_evidence, fuse_hybrid_evidence`.
- `from src.genre.genre_publish import publish` — `publish(metadata_db, sidecar_db, dry_run=False) -> PublishStats`, `.as_dict()` → `{total_albums, graph_albums, legacy_albums, unlinked_releases, collisions, overrides_applied, dry_run}`. One transaction; dry-run rolls back. Writes only derived genre tables (`release_effective_genres`, `genre_graph_*`); never touches `tracks`/`sonic_features`/`track_genres`/`albums`.
- `from scripts.validate_published_genres import validate` — `validate(meta_db) -> int` (0=OK, 1=problems). Prints; spot-checks only assert when the album is present.

**`_fuse_hybrid_for_release` (in `scripts/ai_genre_enrich.py:2379`)** injects artist/album-level metadata.db genres as evidence then calls `fuse_hybrid_evidence`. Task 1 extracts this into `hybrid_evidence.py` so both the CLI and the new `enrich` stage share it.

**Lastfm/enrich/publish are wrapped from these CLI commands** (read for reference, do not import the `cmd_*` functions): `cmd_extract_lastfm` (line 743), `cmd_classify_tags` (585), `cmd_graph_build_assignments` (2046), `_fuse_hybrid_for_release` (2379).

**Design decisions locked for this plan:**
1. **No new dirty flags; rely on fingerprint chaining** (see machinery note above).
2. **`enrich`/`publish` do NOT set `genres_dirty`.** The artifact builder reads `genre_source="legacy"` (`stage_artifacts`, line 1185), which enrich/publish never change. Forcing a genre-sim/artifacts rebuild would burn minutes for identical inputs (violates the warm-path principle). When SP4 flips the artifact to consume the graph, `stage_publish` should set `ctx["genres_dirty"]=True` — a one-line follow-up noted in Task 4.
3. **`publish` backs up `metadata.db` on first publish only** — when the `release_effective_genres` table does not yet exist. Subsequent runs write without backup (publish is atomic + idempotent and scoped to derived tables). Approved by the user 2026-06-10.
4. **Enrich batching = library-wide tag de-duplication + chunked `adjudicate_tags`.** A tag like `"shoegaze"` appearing in 50 releases is adjudicated **once**, cached, then reused. This is the amortization the spec asks for. (`call_structured_batch`'s single-session prefix caching is a future optimization if chunk counts grow; not needed now.)

---

## File structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ai_genre_enrichment/hybrid_evidence.py` | Modify | add `fuse_release_evidence(store, release)` shared helper |
| `scripts/ai_genre_enrich.py` | Modify | `_fuse_hybrid_for_release` delegates to the new helper |
| `scripts/analyze_library.py` | Modify | three new stages + fingerprints + estimates + registration + order + new CLI args |
| `tests/unit/test_ai_genre_hybrid_evidence.py` | Modify | test the extracted helper |
| `tests/unit/test_analyze_graph_stages.py` | Create | unit tests for the three stages + integration order test |

---

### Task 1: Extract `fuse_release_evidence` (DRY prep for the enrich stage)

**Files:**
- Modify: `src/ai_genre_enrichment/hybrid_evidence.py`
- Modify: `scripts/ai_genre_enrich.py:2379-2415`
- Test: `tests/unit/test_ai_genre_hybrid_evidence.py`

- [ ] **Step 1: Read the current `_fuse_hybrid_for_release` body**

Read `scripts/ai_genre_enrich.py:2379-2415` so the extracted function is a faithful copy (the `_SKIP_PREFIXES`, the musicbrainz/discogs confidence constants `0.75`/`0.78`, and the `EvidenceTerm(... classifier="metadata_db")` construction must be preserved exactly).

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_ai_genre_hybrid_evidence.py`:

```python
def test_fuse_release_evidence_injects_metadata_genres(tmp_path):
    """fuse_release_evidence pulls artist/album metadata.db genres in as evidence."""
    from types import SimpleNamespace
    from src.ai_genre_enrichment.hybrid_evidence import fuse_release_evidence
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(tmp_path / "side.db"))
    store.initialize()
    release = SimpleNamespace(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="alb1",
        existing_genres_by_source={"artist:musicbrainz_artist": ["shoegaze", "dream pop"]},
    )
    report = fuse_release_evidence(store, release)
    accepted = {d.term for d in report.accepted_genres}
    provisional = {d.term for d in report.provisional_genres}
    assert "shoegaze" in (accepted | provisional)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ai_genre_hybrid_evidence.py::test_fuse_release_evidence_injects_metadata_genres -v`
Expected: FAIL with `ImportError: cannot import name 'fuse_release_evidence'`

- [ ] **Step 4: Add `fuse_release_evidence` to `hybrid_evidence.py`**

Add to `src/ai_genre_enrichment/hybrid_evidence.py` (after `fuse_hybrid_evidence`; `EvidenceTerm`, `collect_hybrid_evidence`, `fuse_hybrid_evidence` are already defined in this module). The `release` parameter is any object exposing `.release_key`, `.normalized_artist`, `.normalized_album`, `.album_id`, `.existing_genres_by_source` (so both `ReleasePayload` and test `SimpleNamespace` work):

```python
def fuse_release_evidence(store, release):
    """Fuse sidecar evidence + metadata.db artist/album genres for one release.

    Shared by the `hybrid-enrich-one` / `graph-build-assignments` CLI commands and
    the analyze `enrich` stage. `release` exposes release_key, normalized_artist,
    normalized_album, album_id, and existing_genres_by_source.
    """
    evidence = collect_hybrid_evidence(store, release.release_key)

    # Artist/album-level MusicBrainz + Discogs tags from metadata.db are reliable
    # genre signals below the "strong" threshold; inject them as provisional.
    _skip_prefixes = ("artist:lastfm", "album:lastfm", "track:")
    for source_key, genres in release.existing_genres_by_source.items():
        if any(source_key.startswith(p) for p in _skip_prefixes):
            continue
        parts = source_key.split(":", 1)
        if len(parts) != 2:
            continue
        src = parts[1]
        if "musicbrainz" in src:
            source_type, conf = "musicbrainz", 0.75
        elif "discogs" in src:
            source_type, conf = "discogs", 0.78
        else:
            continue
        for genre in genres:
            genre_norm = genre.strip().casefold()
            if genre_norm:
                evidence.append(EvidenceTerm(
                    term=genre_norm,
                    source_type=source_type,
                    confidence=conf,
                    canonical_slug=genre_norm,
                    mapping_status="mapped",
                    classifier="metadata_db",
                ))

    return fuse_hybrid_evidence(
        release_key=release.release_key,
        evidence=evidence,
        sparse_release=not release.existing_genres_by_source,
    )
```

- [ ] **Step 5: Make `_fuse_hybrid_for_release` delegate**

In `scripts/ai_genre_enrich.py`, replace the entire body of `_fuse_hybrid_for_release` (lines 2379-2415) with a one-line delegation, and add `fuse_release_evidence` to the existing `hybrid_evidence` import on line 22:

```python
from src.ai_genre_enrichment.hybrid_evidence import (
    EvidenceTerm,
    collect_hybrid_evidence,
    fuse_hybrid_evidence,
    fuse_release_evidence,
)
```

```python
def _fuse_hybrid_for_release(store: SidecarStore, release: ReleasePayload):
    return fuse_release_evidence(store, release)
```

- [ ] **Step 6: Run the affected suites**

Run: `python -m pytest tests/unit/test_ai_genre_hybrid_evidence.py -v`
Expected: the new test PASSES. (Pre-existing failures in this file documented in Phase 1 remain; no NEW failures.)

- [ ] **Step 7: Commit**

```bash
git add src/ai_genre_enrichment/hybrid_evidence.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_hybrid_evidence.py
git commit -m "refactor(enrichment): extract fuse_release_evidence shared by CLI + analyze enrich"
```

---

### Task 2: `lastfm` stage

**Files:**
- Modify: `scripts/analyze_library.py` (imports, `stage_lastfm`, fingerprint branch, estimate branch, `STAGE_FUNCS`, `STAGE_ORDER_DEFAULT`, new CLI args)
- Test: `tests/unit/test_analyze_graph_stages.py` (new file)

**Behavior:** for each release without a `lastfm_tags` source page, fetch top tags via `fetch_lastfm_tags`, store as a source page, and run **deterministic** classification (`adjudicate=False` — AI adjudication is the `enrich` stage's job). Missing Last.fm key → raise loudly (like `stage_discogs`). No LLM.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_analyze_graph_stages.py`:

```python
"""Unit tests for the analyze_library graph stages (lastfm, enrich, publish)."""
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.analyze_library as al
from src.ai_genre_enrichment.storage import SidecarStore


def _metadata_db(tmp_path: Path) -> str:
    """Minimal metadata.db: one album with one track and an artist genre."""
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
            album_id TEXT, title TEXT, file_path TEXT);
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        INSERT INTO tracks VALUES ('t1','Slowdive','Souvlaki','alb1','Alison','/x/a.flac');
        INSERT INTO albums VALUES ('alb1','Souvlaki','Slowdive');
        INSERT INTO artist_genres VALUES ('Slowdive','shoegaze','musicbrainz_artist');
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _ctx(tmp_path: Path, db_path: str, sidecar: str, **arg_overrides):
    """Build a minimal stage ctx with a live metadata.db connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    args = Namespace(
        force=False, limit=None, dry_run=False, progress=False, verbose=False,
        progress_interval=15.0, progress_every=500, max_tracks=0, model=None,
        enrich_chunk_size=50, lastfm_api_key="FAKEKEY",
        **arg_overrides,
    )
    return {
        "config_path": str(tmp_path / "config.yaml"),
        "db_path": db_path,
        "out_dir": tmp_path,
        "args": args,
        "conn": conn,
        "config_hash": "test",
        "library_root": str(tmp_path),
        "genres_dirty": False, "sonic_dirty": False,
        "artifacts_dirty": False, "force_stage": False,
    }


def test_stage_lastfm_fetches_stores_and_classifies(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))

    captured = {}

    def fake_fetch(artist, album, api_key, limit=20):
        captured["args"] = (artist, album, api_key, limit)
        return ["shoegaze", "dream pop", "ambient"]

    monkeypatch.setattr(al, "fetch_lastfm_tags", fake_fetch)

    ctx = _ctx(tmp_path, db_path, sidecar)
    result = ctx_result = al.stage_lastfm(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["extracted"] == 1
    # tags landed in the sidecar as a lastfm_tags source page
    store = SidecarStore(sidecar)
    assert "slowdive::souvlaki" in store.release_keys_with_source_type("lastfm_tags")


def test_stage_lastfm_missing_key_raises(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    ctx = _ctx(tmp_path, db_path, sidecar, lastfm_api_key=None)
    # also ensure config lookup can't supply a key
    monkeypatch.setattr(al, "_resolve_lastfm_api_key", lambda ctx: None)
    with pytest.raises(RuntimeError, match="Last.fm API key"):
        al.stage_lastfm(ctx)
    ctx["conn"].close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -v`
Expected: FAIL with `AttributeError: module 'scripts.analyze_library' has no attribute 'stage_lastfm'`

- [ ] **Step 3: Add imports for the graph stages**

In `scripts/analyze_library.py`, after the existing `from src.logging_utils import ProgressLogger` (line 39), add:

```python
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.discovery import discover_releases
from src.ai_genre_enrichment.lastfm_enrichment import fetch_lastfm_tags
```

(The `enrich` and `publish` stages add their own imports lazily inside their functions in later tasks, to keep import cost off the warm path when only sonic/artifact stages run.)

- [ ] **Step 4: Add the Last.fm key resolver and `stage_lastfm`**

Add to `scripts/analyze_library.py` (after `stage_discogs`, before `stage_sonic`):

```python
def _resolve_lastfm_api_key(ctx: Dict) -> Optional[str]:
    """Resolve the Last.fm API key from args, env, or config (in that order)."""
    args = ctx["args"]
    key = getattr(args, "lastfm_api_key", None) or os.environ.get("LASTFM_API_KEY")
    if key:
        return key
    try:
        return Config(ctx["config_path"]).lastfm_api_key or None
    except Exception:
        return None


def stage_lastfm(ctx: Dict) -> Dict:
    """Fetch Last.fm top tags into the sidecar for releases that lack them.

    No LLM. Deterministic classification only (adjudicate=False); the `enrich`
    stage owns AI adjudication. Missing API key raises (production-required,
    like the discogs stage).
    """
    import time

    args = ctx["args"]
    api_key = _resolve_lastfm_api_key(ctx)
    if not api_key:
        raise RuntimeError(
            "Last.fm API key required for the lastfm stage. Set LASTFM_API_KEY, "
            "pass --lastfm-api-key, or add lastfm.api_key to config.yaml."
        )

    store = SidecarStore(str(ENRICHMENT_DB_PATH))
    store.initialize()

    limit = args.limit if args.limit and args.limit > 0 else None
    releases = discover_releases(ctx["db_path"], limit=limit)
    if not releases:
        return {"skipped": True, "reason": "no_releases", "extracted": 0}

    already = store.release_keys_with_source_type("lastfm_tags")
    pending = [r for r in releases if args.force or r.release_key not in already]
    skipped_existing = len(releases) - len(pending)
    if not pending:
        logger.info("Skipping lastfm stage (all %d releases already scraped)", len(releases))
        return {"skipped": True, "reason": "all_scraped", "extracted": 0,
                "skipped_existing": skipped_existing}

    prog = (
        ProgressLogger(
            logger, total=len(pending), label="lastfm", unit="releases",
            interval_s=getattr(args, "progress_interval", 15.0),
            every_n=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
        if getattr(args, "progress", True) else None
    )

    extracted = empty = failed = 0
    for release in pending:
        if prog:
            prog.update(detail=release.release_key)
        try:
            tags = fetch_lastfm_tags(
                artist=release.normalized_artist,
                album=release.normalized_album or None,
                api_key=api_key,
                limit=20,
            )
            if not tags:
                empty += 1
                time.sleep(0.25)
                continue
            album_segment = f"/album/{release.normalized_album}" if release.normalized_album else ""
            page_id = store.upsert_source_page(
                release_key=release.release_key,
                normalized_artist=release.normalized_artist,
                normalized_album=release.normalized_album,
                album_id=release.album_id,
                source_url=f"lastfm://artist/{release.normalized_artist}{album_segment}",
                source_type="lastfm_tags",
                identity_status="confirmed",
                identity_confidence=0.9,
                evidence_summary="Last.fm top tags via API.",
            )
            store.replace_source_tags(page_id, tags)
            store.classify_source_tags(page_id, adjudicate=False, model=None)
            extracted += 1
        except Exception as exc:  # network blip / API error — log and continue
            failed += 1
            logger.debug("Last.fm failed for %s: %s", release.release_key, exc)
        time.sleep(0.25)  # ~5 req/s courtesy limit

    if prog:
        prog.finish(detail=f"lastfm extracted {extracted:,} of {len(pending):,}")
    logger.info("lastfm stage: extracted=%d empty=%d failed=%d skipped_existing=%d",
                extracted, empty, failed, skipped_existing)
    return {"skipped": False, "extracted": extracted, "empty": empty,
            "failed": failed, "skipped_existing": skipped_existing,
            "total": len(pending), "errors": failed}
```

- [ ] **Step 5: Add the fingerprint and estimate branches**

In `compute_stage_fingerprint`, add this branch (after the `discogs` branch, before `sonic`). It opens the sidecar read-only so the fingerprint reflects how many releases still need scraping:

```python
    if stage == "lastfm":
        total_albums = _safe_count(conn, "SELECT COUNT(DISTINCT album_id) FROM albums "
                                         "WHERE album_id IS NOT NULL AND album_id != ''")
        lastfm_pages = _sidecar_count(
            "SELECT COUNT(*) FROM ai_genre_source_pages WHERE source_type='lastfm_tags'")
        key = {"stage": stage, "total_albums": total_albums, "lastfm_pages": lastfm_pages}
        return _hash_obj(key)
```

In `estimate_stage_units`, add (inside the `try`, alongside the other branches):

```python
        if stage == "lastfm":
            total_albums = _safe_count(conn, "SELECT COUNT(DISTINCT album_id) FROM albums "
                                            "WHERE album_id IS NOT NULL AND album_id != ''")
            scraped = _sidecar_count(
                "SELECT COUNT(*) FROM ai_genre_source_pages WHERE source_type='lastfm_tags'")
            return max(0, total_albums - scraped), "releases needing Last.fm tags"
```

Add the `_sidecar_count` helper near `_safe_count` (line 58). It must tolerate a missing sidecar / missing table:

```python
def _sidecar_count(query: str, params: tuple = ()) -> int:
    """COUNT(*) against the enrichment sidecar, 0 if absent/unreadable."""
    try:
        if not ENRICHMENT_DB_PATH.exists():
            return 0
        conn = sqlite3.connect(f"file:{ENRICHMENT_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0
```

- [ ] **Step 6: Register the stage and insert it into the default order**

Update `STAGE_ORDER_DEFAULT` (line 43) — insert `lastfm` after `discogs`:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "genre-sim", "artifacts", "genre-embedding", "verify"]
```

Add to `STAGE_FUNCS` (line 1316):

```python
    "lastfm": stage_lastfm,
```

- [ ] **Step 7: Add the `--lastfm-api-key` and enrich args to `parse_args`**

In `parse_args`, after the `--verbose` argument (line 1363), add:

```python
    parser.add_argument("--lastfm-api-key", default=None,
                        help="Last.fm API key for the lastfm stage (else env LASTFM_API_KEY / config)")
    parser.add_argument("--model", default=None,
                        help="LLM model override for the enrich stage (default: provider default)")
    parser.add_argument("--enrich-chunk-size", type=int, default=50,
                        help="Tags per adjudication chunk in the enrich stage (default: 50)")
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -v`
Expected: the two `lastfm` tests PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_graph_stages.py
git commit -m "feat(analyze): lastfm stage scrapes top tags into the enrichment sidecar"
```

---

### Task 3: `enrich` stage

**Files:**
- Modify: `scripts/analyze_library.py` (`stage_enrich`, fingerprint branch, estimate branch, registration)
- Test: `tests/unit/test_analyze_graph_stages.py` (append)

**Behavior:**
1. Discover releases that have at least one source page in the sidecar.
2. **Pre-pass:** deterministically classify each pending page (`adjudicate=False`) — this resolves known tags and leaves uncached `review_only` tags.
3. Collect the **distinct, uncached** `review_only` normalized tags across all pending pages.
4. Chunk them (`enrich_chunk_size`) and adjudicate each chunk with `adjudicate_tags` (Claude via the factory). Cache every **definitive** (non-`review_only`) result via `store.cache_adjudication`. One Claude call per chunk, library-wide — a tag is adjudicated once regardless of how many releases use it.
5. Per release: re-run deterministic classification (now cache hits), `rebuild_enriched_genres_for_release`, `fuse_release_evidence`, `materialize_layered_assignments`.

The Claude client is injectable for tests via `ctx["args"].enrich_client` (a test seam); production builds it through the factory.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_analyze_graph_stages.py`:

```python
def _seed_sidecar_with_pages(sidecar: str):
    """One release with a lastfm source page carrying a known + an unknown tag."""
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="alb1",
        source_url="lastfm://artist/slowdive/album/souvlaki",
        source_type="lastfm_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="seed",
    )
    # 'shoegaze' classifies deterministically; 'zzz unknown thing' is review_only.
    store.replace_source_tags(page_id, ["shoegaze", "zzz unknown thing"])
    return store


class _RecordingAdjudicator:
    """Stand-in for adjudicate_tags: records calls, returns canned classifications."""
    def __init__(self):
        self.calls = []

    def __call__(self, tags, *, model=None, dry_run=False, client=None):
        self.calls.append([norm for _, norm in tags])
        return {
            norm: {"classification": "genre_style", "confidence": 0.8, "reason": "ok"}
            for _, norm in tags
        }


def test_stage_enrich_dedupes_adjudicates_and_materializes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    rec = _RecordingAdjudicator()
    monkeypatch.setattr(al, "adjudicate_tags", rec)

    ctx = _ctx(tmp_path, db_path, sidecar)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["releases_enriched"] == 1
    # exactly one distinct unknown tag adjudicated, in a single chunk
    assert rec.calls == [["zzz unknown thing"]]
    assert result["tags_adjudicated"] == 1


def test_stage_enrich_no_pending_skips(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()  # empty sidecar, no source pages
    monkeypatch.setattr(al, "adjudicate_tags", _RecordingAdjudicator())

    ctx = _ctx(tmp_path, db_path, sidecar)
    result = al.stage_enrich(ctx)
    ctx["conn"].close()
    assert result["skipped"] is True
    assert result.get("releases_enriched", 0) == 0


def test_stage_enrich_propagates_adjudication_failure(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    _seed_sidecar_with_pages(sidecar)

    def boom(tags, *, model=None, dry_run=False, client=None):
        raise RuntimeError("Claude Code request failed after retries: rate window")

    monkeypatch.setattr(al, "adjudicate_tags", boom)
    ctx = _ctx(tmp_path, db_path, sidecar)
    with pytest.raises(RuntimeError, match="failed after retries"):
        al.stage_enrich(ctx)
    ctx["conn"].close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -k enrich -v`
Expected: FAIL with `AttributeError: ... has no attribute 'stage_enrich'`

- [ ] **Step 3: Add the module-level import and `stage_enrich`**

In `scripts/analyze_library.py`, add to the import block from Task 2 (so the test seam `monkeypatch.setattr(al, "adjudicate_tags", ...)` patches a module global):

```python
from src.ai_genre_enrichment.tag_adjudicator import adjudicate_tags
```

Add `stage_enrich` after `stage_lastfm`:

```python
def _pending_pages_for_releases(store: SidecarStore, release_keys: list[str]) -> List[Tuple[str, int]]:
    """[(release_key, source_page_id)] for all source pages of the given releases."""
    if not release_keys:
        return []
    pairs: List[Tuple[str, int]] = []
    with store.connect() as conn:
        placeholders = ",".join("?" for _ in release_keys)
        rows = conn.execute(
            f"SELECT release_key, source_page_id FROM ai_genre_source_pages "
            f"WHERE release_key IN ({placeholders}) ORDER BY source_page_id",
            release_keys,
        ).fetchall()
    for row in rows:
        pairs.append((row[0], int(row[1])))
    return pairs


def _uncached_review_only_tags(store: SidecarStore, page_ids: List[int]) -> List[Tuple[str, str]]:
    """Distinct (raw_tag, normalized_tag) review_only tags not yet in the adjudication cache.

    Re-derives classification from the raw tag via the canonical classifier (the
    same `classify_source_tag` that `classify_source_tags` uses), so this depends
    only on `ai_genre_source_tags` (columns confirmed: source_tag_id, raw_tag,
    source_page_id) — not on the classifications table's exact schema.
    """
    from src.ai_genre_enrichment.tag_classification import classify_source_tag

    if not page_ids:
        return []
    seen: dict[str, str] = {}
    with store.connect() as conn:
        placeholders = ",".join("?" for _ in page_ids)
        rows = conn.execute(
            f"SELECT raw_tag FROM ai_genre_source_tags "
            f"WHERE source_page_id IN ({placeholders})",
            page_ids,
        ).fetchall()
    for (raw_tag,) in rows:
        c = classify_source_tag(raw_tag)
        norm = c.normalized_tag
        if (
            c.classification == "review_only"
            and norm
            and norm not in seen
            and store.lookup_cached_adjudication(norm) is None
        ):
            seen[norm] = raw_tag
    return [(raw, norm) for norm, raw in seen.items()]


def stage_enrich(ctx: Dict) -> Dict:
    """Adjudicate unknown tags (chunked Claude calls) and materialize graph genres.

    Writes only to the enrichment sidecar. De-duplicates unknown tags library-wide:
    a tag is sent to Claude once, cached, then reused across every release.
    Raises if the LLM backend fails (explicitly-requested work that cannot run is
    an error, not a silent skip).
    """
    args = ctx["args"]
    store = SidecarStore(str(ENRICHMENT_DB_PATH))
    store.initialize()

    limit = args.limit if args.limit and args.limit > 0 else None
    releases = discover_releases(ctx["db_path"], limit=limit)
    by_key = {r.release_key: r for r in releases}
    page_pairs = _pending_pages_for_releases(store, list(by_key.keys()))
    if not page_pairs:
        logger.info("Skipping enrich stage (no source pages to enrich)")
        return {"skipped": True, "reason": "no_source_pages", "releases_enriched": 0}

    page_ids = [pid for _, pid in page_pairs]
    pending_keys = sorted({rk for rk, _ in page_pairs})

    # Pre-pass: deterministic classification populates known tags + cache hits.
    for _, page_id in page_pairs:
        store.classify_source_tags(page_id, adjudicate=False, model=None)

    # Collect distinct uncached review_only tags across ALL pending pages, adjudicate
    # in chunks, and cache definitive results so per-release classification is cache-only.
    unknown = _uncached_review_only_tags(store, page_ids)
    model = getattr(args, "model", None) or resolve_enrichment_model(None)
    chunk_size = max(1, int(getattr(args, "enrich_chunk_size", 50)))
    injected_client = getattr(args, "enrich_client", None)
    tags_adjudicated = 0
    chunks_used = 0
    for i in range(0, len(unknown), chunk_size):
        chunk = unknown[i:i + chunk_size]
        results = adjudicate_tags(chunk, model=model, client=injected_client)
        chunks_used += 1
        for norm, decision in results.items():
            classification = decision.get("classification")
            if classification and classification != "review_only":
                store.cache_adjudication(
                    normalized_tag=norm,
                    classification=classification,
                    confidence=float(decision.get("confidence", 0.0)),
                    classifier="ai",
                )
                tags_adjudicated += 1

    # Per-release: re-classify (cache hits now), rebuild signatures, fuse, materialize.
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.hybrid_evidence import fuse_release_evidence

    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)

    pages_by_release: dict[str, List[int]] = {}
    for rk, pid in page_pairs:
        pages_by_release.setdefault(rk, []).append(pid)

    prog = (
        ProgressLogger(
            logger, total=len(pending_keys), label="enrich", unit="releases",
            interval_s=getattr(args, "progress_interval", 15.0),
            every_n=getattr(args, "progress_every", 500),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
        if getattr(args, "progress", True) else None
    )

    enriched = 0
    assignments = 0
    for rk in pending_keys:
        release = by_key.get(rk)
        if release is None:
            continue
        if prog:
            prog.update(detail=rk)
        for pid in pages_by_release.get(rk, []):
            store.classify_source_tags(pid, adjudicate=False, model=None)
        store.rebuild_enriched_genres_for_release(rk)
        fused = fuse_release_evidence(store, release)
        summary = materialize_layered_assignments(
            store, release_id=rk, artist=release.normalized_artist,
            album=release.normalized_album, report=fused, taxonomy=taxonomy,
        )
        assignments += summary.genre_assignment_count
        enriched += 1

    if prog:
        prog.finish(detail=f"enriched {enriched:,} releases")
    logger.info("enrich stage: releases=%d tags_adjudicated=%d chunks=%d assignments=%d",
                enriched, tags_adjudicated, chunks_used, assignments)
    return {"skipped": False, "releases_enriched": enriched,
            "tags_adjudicated": tags_adjudicated, "chunks_used": chunks_used,
            "genre_assignments": assignments, "total": enriched}
```

Add `resolve_enrichment_model` to the imports (it is also used by Task 4's stage; import once near the Task-2 imports):

```python
from src.ai_genre_enrichment.provider import resolve_enrichment_model
```

- [ ] **Step 4: Add the fingerprint and estimate branches**

In `compute_stage_fingerprint`, add after the `lastfm` branch:

```python
    if stage == "enrich":
        source_pages = _sidecar_count("SELECT COUNT(*) FROM ai_genre_source_pages")
        signatures = _sidecar_count("SELECT COUNT(*) FROM enriched_genre_signatures")
        assignments = _sidecar_count(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments")
        key = {"stage": stage, "source_pages": source_pages,
               "signatures": signatures, "assignments": assignments}
        return _hash_obj(key)
```

In `estimate_stage_units`, add:

```python
        if stage == "enrich":
            source_pages = _sidecar_count("SELECT COUNT(DISTINCT release_key) "
                                          "FROM ai_genre_source_pages")
            return source_pages, "releases with source pages to enrich"
```

- [ ] **Step 5: Register the stage**

Add to `STAGE_FUNCS`, after the `lastfm` entry:

```python
    "enrich": stage_enrich,
```

Insert `enrich` into `STAGE_ORDER_DEFAULT` after `sonic`:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "enrich", "genre-sim", "artifacts", "genre-embedding", "verify"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -k enrich -v`
Expected: the three `enrich` tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_graph_stages.py
git commit -m "feat(analyze): enrich stage adjudicates unknown tags and materializes graph genres"
```

---

### Task 4: `publish` stage

**Files:**
- Modify: `scripts/analyze_library.py` (`stage_publish`, fingerprint branch, estimate branch, registration, order)
- Test: `tests/unit/test_analyze_graph_stages.py` (append)

**Behavior:** publish authoritative genres into `metadata.db`. If `release_effective_genres` does not yet exist (first publish ever), copy `metadata.db` to a timestamped `.bak` first. Then call `genre_publish.publish(metadata_db, sidecar_db, dry_run=args.dry_run)` and run the read-only `validate()` checks. Reports publish stats + validation status.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_analyze_graph_stages.py`:

```python
def _published_table_exists(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='release_effective_genres'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def test_stage_publish_first_run_backs_up_and_publishes(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path, sidecar)
    result = al.stage_publish(ctx)
    ctx["conn"].close()

    assert result["skipped"] is False
    assert result["backed_up"] is True
    # a timestamped backup was created next to metadata.db
    backups = list(Path(db_path).parent.glob("metadata.db.bak.*"))
    assert backups, "expected a first-publish backup"
    # release_effective_genres now exists
    assert _published_table_exists(db_path)
    assert result["validation_ok"] is True


def test_stage_publish_second_run_no_backup(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    al.stage_publish(_ctx(tmp_path, db_path, sidecar))  # first publish (backs up)
    before = set(Path(db_path).parent.glob("metadata.db.bak.*"))
    ctx2 = _ctx(tmp_path, db_path, sidecar)
    result = al.stage_publish(ctx2)
    ctx2["conn"].close()
    after = set(Path(db_path).parent.glob("metadata.db.bak.*"))

    assert result["backed_up"] is False
    assert before == after, "second publish must not create a new backup"


def test_stage_publish_dry_run_rolls_back(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))
    SidecarStore(sidecar).initialize()

    ctx = _ctx(tmp_path, db_path, sidecar, dry_run=True)
    result = al.stage_publish(ctx)
    ctx["conn"].close()
    assert result["dry_run"] is True
    # dry-run rolls back the publish transaction → no published table persists
    assert not _published_table_exists(db_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -k publish -v`
Expected: FAIL with `AttributeError: ... has no attribute 'stage_publish'`

- [ ] **Step 3: Implement `stage_publish`**

Add `stage_publish` after `stage_enrich`. It imports `shutil`/`publish`/`validate` lazily (keeps them off the warm path):

```python
def _release_effective_genres_exists(db_path: str) -> bool:
    """True if metadata.db already has the published release_effective_genres table."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='release_effective_genres'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def stage_publish(ctx: Dict) -> Dict:
    """Publish authoritative genres into metadata.db (release_effective_genres).

    First publish (table absent) takes a timestamped metadata.db backup; later
    runs write directly (publish is atomic + idempotent and scoped to derived
    genre tables — it never touches tracks/sonic/track_genres). Dry-run rolls back.
    """
    import shutil
    from src.genre.genre_publish import publish as publish_genres
    from scripts.validate_published_genres import validate as validate_published

    args = ctx["args"]
    db_path = ctx["db_path"]
    sidecar = str(ENRICHMENT_DB_PATH)
    if not ENRICHMENT_DB_PATH.exists():
        logger.info("Skipping publish stage (no enrichment sidecar at %s)", sidecar)
        return {"skipped": True, "reason": "no_sidecar"}

    dry_run = bool(getattr(args, "dry_run", False))
    backed_up = False
    if not dry_run and not _release_effective_genres_exists(db_path):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.bak.{ts}"
        shutil.copy2(db_path, backup_path)
        backed_up = True
        logger.info("First publish — backed up metadata.db to %s", backup_path)

    stats = publish_genres(db_path, sidecar, dry_run=dry_run)
    stats_dict = stats.as_dict()

    validation_ok = True
    if not dry_run:
        rc = validate_published(db_path)
        validation_ok = (rc == 0)
        if not validation_ok:
            logger.warning("publish stage: validation reported problems (rc=%d)", rc)

    logger.info("publish stage: graph_albums=%d legacy_albums=%d total=%d backed_up=%s dry_run=%s",
                stats_dict.get("graph_albums", 0), stats_dict.get("legacy_albums", 0),
                stats_dict.get("total_albums", 0), backed_up, dry_run)
    return {"skipped": False, "backed_up": backed_up, "dry_run": dry_run,
            "validation_ok": validation_ok, "stats": stats_dict,
            "total": stats_dict.get("total_albums", 0),
            "errors": 0 if validation_ok else 1}
```

- [ ] **Step 4: Add the fingerprint and estimate branches**

In `compute_stage_fingerprint`, add after the `enrich` branch. The fingerprint reflects the sidecar graph state (publish input) and whether the published table is already current:

```python
    if stage == "publish":
        side_assignments = _sidecar_count(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments")
        published_rows = _safe_count(conn, "SELECT COUNT(*) FROM release_effective_genres")
        key = {"stage": stage, "side_assignments": side_assignments,
               "published_rows": published_rows}
        return _hash_obj(key)
```

In `estimate_stage_units`, add:

```python
        if stage == "publish":
            total_albums = _safe_count(conn, "SELECT COUNT(*) FROM albums "
                                            "WHERE album_id IS NOT NULL AND album_id != ''")
            return total_albums, "albums to resolve into release_effective_genres"
```

**Note for the implementer:** `_safe_count` on `release_effective_genres` returns 0 when the table is absent (its `except` clause), which is correct — a missing published table yields a distinct fingerprint from a populated one, so the first publish always runs.

- [ ] **Step 5: Register and order the stage**

Add to `STAGE_FUNCS`, after `enrich`:

```python
    "publish": stage_publish,
```

Insert `publish` into `STAGE_ORDER_DEFAULT` after `enrich`:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "enrich", "publish", "genre-sim", "artifacts", "genre-embedding", "verify"]
```

- [ ] **Step 6: Note the SP4 follow-up in the docstring**

Per design decision 2, add this comment immediately above the `return` in `stage_publish` (so the future SP4 author finds it):

```python
    # SP4 follow-up: when the artifact builder consumes the graph (genre_source
    # != "legacy"), set ctx["genres_dirty"] = True here so genre-sim/artifacts
    # rebuild after a publish. In Phase 2 artifacts read legacy genres, which
    # publish does not change, so we deliberately do not trigger a rebuild.
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -k publish -v`
Expected: the three `publish` tests PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_graph_stages.py
git commit -m "feat(analyze): publish stage writes release_effective_genres with first-run backup"
```

---

### Task 5: Full-order integration test + skip/chaining verification

**Files:**
- Test: `tests/unit/test_analyze_graph_stages.py` (append)

**Goal:** prove (a) the new stages appear in the default order in the right positions, (b) `run_pipeline` runs the chain end-to-end against temp DBs with fakes, and (c) re-running with unchanged inputs skips the new stages via fingerprints.

- [ ] **Step 1: Write the integration test**

Append to `tests/unit/test_analyze_graph_stages.py`:

```python
def test_default_stage_order_has_new_stages_positioned():
    order = al.STAGE_ORDER_DEFAULT
    for name in ("lastfm", "enrich", "publish"):
        assert name in order, f"{name} missing from STAGE_ORDER_DEFAULT"
        assert name in al.STAGE_FUNCS
    # lastfm after discogs; enrich after sonic; publish after enrich
    assert order.index("lastfm") > order.index("discogs")
    assert order.index("enrich") > order.index("sonic")
    assert order.index("publish") == order.index("enrich") + 1
    # publish precedes genre-sim/artifacts
    assert order.index("publish") < order.index("genre-sim")


def test_run_pipeline_runs_new_stages_then_skips_on_rerun(tmp_path, monkeypatch):
    db_path = _metadata_db(tmp_path)
    sidecar = str(tmp_path / "side.db")
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", Path(sidecar))

    monkeypatch.setattr(al, "fetch_lastfm_tags",
                        lambda artist, album, api_key, limit=20: ["shoegaze", "zzz unknown thing"])
    monkeypatch.setattr(al, "adjudicate_tags", _RecordingAdjudicator())

    args = Namespace(
        config="config.yaml", db_path=db_path, stages="lastfm,enrich,publish",
        workers="auto", limit=None, max_tracks=0, force=False,
        force_no_match=False, force_error=False, force_reject=False, force_all=False,
        out_dir=str(tmp_path), beat_sync=False, dry_run=False,
        progress=False, progress_interval=15.0, progress_every=500, verbose=False,
        lastfm_api_key="FAKEKEY", model=None, enrich_chunk_size=50,
        log_level="INFO", log_file=str(tmp_path / "log.txt"),
        quiet=False, debug=False, show_run_id=False,
    )
    rc = al.run_pipeline(args, console_logging=False)
    assert rc == 0
    assert _published_table_exists(db_path)

    # Re-run: inputs unchanged → all three stages skip via fingerprint.
    import json
    rc2 = al.run_pipeline(args, console_logging=False)
    assert rc2 == 0
    report = json.loads((Path(tmp_path) / "analyze_run_report.json").read_text())
    for name in ("lastfm", "enrich", "publish"):
        assert report["stages"][name]["decision"] == "skipped", \
            f"{name} should skip on unchanged rerun"
```

**Implementer note:** if `run_pipeline` raises because the `Namespace` is missing an attribute it reads (e.g. a logging arg), add that attribute to the test `Namespace` with the same default `parse_args` uses — do **not** change `run_pipeline` to be lenient. The set above mirrors `parse_args`; adjust only if a genuinely new read appears.

- [ ] **Step 2: Run the integration tests**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -v`
Expected: all tests PASS (lastfm + enrich + publish + the two integration tests).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_analyze_graph_stages.py
git commit -m "test(analyze): end-to-end order + fingerprint-skip for the graph stages"
```

---

### Task 6: Docs + full verification

**Files:**
- Modify: `docs/GOLDEN_COMMANDS.md` (or the analyze section it points to)
- Modify: `docs/AI_GENRE_ENRICHMENT.md` (note the analyze stages)

- [ ] **Step 1: Document the new stages**

In `docs/GOLDEN_COMMANDS.md`, find the `analyze_library.py` section and update the default stage list and add a short note. Add (adapt to the file's existing format):

```markdown
### Analyze Library stages

Default order: `scan → genres → discogs → lastfm → sonic → enrich → publish → genre-sim → artifacts → genre-embedding → verify`

- **lastfm** — fetch Last.fm top tags into the enrichment sidecar (needs LASTFM_API_KEY / config; no LLM).
- **enrich** — adjudicate unknown tags via Claude (provider factory; de-duped library-wide, chunked) and materialize layered graph genres into `ai_genre_enrichment.db`.
- **publish** — resolve graph-where-present-else-legacy into `release_effective_genres` in metadata.db. First publish backs up metadata.db (timestamped); idempotent thereafter.

Run a subset: `python scripts/analyze_library.py --stages lastfm,enrich,publish`
Dry-run publish (compute + roll back): `python scripts/analyze_library.py --stages publish --dry-run`
```

In `docs/AI_GENRE_ENRICHMENT.md`, under the provider section added in Phase 1, add one line:

```markdown
The analyze pipeline runs enrichment automatically: the `enrich` stage of
`scripts/analyze_library.py` adjudicates unknown tags and materializes graph
genres; the `publish` stage writes `release_effective_genres`.
```

- [ ] **Step 2: Lint + types on the changed files**

Run: `ruff check scripts/analyze_library.py src/ai_genre_enrichment/hybrid_evidence.py tests/unit/test_analyze_graph_stages.py`
Expected: clean. (`ruff check --fix` for trivial import-order/whitespace.)

Run: `python -m mypy scripts/analyze_library.py`
Expected: no NEW errors attributable to the new stages. `analyze_library.py` may have pre-existing mypy noise; compare against `git stash` baseline if unsure and only fix what this task introduced.

- [ ] **Step 3: Full enrichment + analyze suites**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py tests/unit/test_analyze_orchestration.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_ai_genre_enrichment.py -v`
Expected: all PASS except the documented pre-existing failures (the Phase-1 deselect list). No NEW failures.

- [ ] **Step 4: Full fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: pass modulo the documented pre-existing deselect list. Any NEW failure must be fixed before commit.

- [ ] **Step 5: Manual smoke (optional, real backend — not CI)**

On the real library, dry-run the publish stage (no writes, rolls back):

```bash
python scripts/analyze_library.py --stages publish --dry-run
```

Expected: prints publish stats JSON; `release_effective_genres` is NOT persisted (dry-run rollback). This verifies the wiring against real data without touching metadata.db.

- [ ] **Step 6: Commit**

```bash
git add docs/GOLDEN_COMMANDS.md docs/AI_GENRE_ENRICHMENT.md
git commit -m "docs(analyze): document lastfm/enrich/publish stages"
```

---

## Out of scope (later plans)

- **Phase 3** — web `/api/tools/analyze` + `/api/tools/enrich` endpoints and the React Tools panel. The worker handlers (`handle_analyze_library`, `handle_enrich_genres`) already invoke `run_pipeline`, so the new stages flow through automatically once Phase 3 wires the UI.
- **SP4** — flipping the artifact builder to consume the graph (`genre_source != "legacy"`). That is where `stage_publish` should begin setting `genres_dirty` (Task 4 Step 6 marks the spot).
- **`call_structured_batch` single-session adjudication** — a future optimization if enrich chunk counts grow large enough that cross-chunk prefix caching matters.
- Bandcamp: dropped (Phase 1 decision).

---

## Self-review notes

- **Spec coverage:** lastfm stage (Task 2) ✓; enrich stage with chunked classification + hybrid fusion + graph assignments (Task 3) ✓; publish stage into release_effective_genres + validate (Task 4) ✓; stage ordering `scan→genres→discogs→lastfm→sonic→enrich→publish→genre-sim→artifacts→genre-embedding→verify` (Tasks 2-4 + Task 5 assertion) ✓; loud failure on missing Last.fm key (Task 2) ✓; sidecar-only writes for enrich + sanctioned metadata.db write for publish (Tasks 3-4) ✓; no live metadata.db writes in tests — temp DBs only (all tasks) ✓.
- **Deliberate spec deviation:** enrich/publish do not set `genres_dirty` in Phase 2 (design decision 2, documented in Task 4 Step 6). Rationale: artifacts read legacy genres; forcing a rebuild burns the warm path for identical inputs.
- **Type consistency:** stage functions return `dict` with a `skipped` key and stage-specific counts; `_sidecar_count`/`_safe_count`/`_resolve_lastfm_api_key`/`_release_effective_genres_exists` helper names are used consistently across tasks; `fuse_release_evidence(store, release)` signature is identical in Task 1 (definition) and Task 3 (call).
