# Artist-mode Popular Seeds — Data Path (plan 2a of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch each artist's Last.fm top tracks offline, resolve them to the *canonical* local track per song, and write a `popularity_sidecar.npz` (track_id → per-artist popularity) plus a runtime loader — so generation can later prefer recognizable hits without ever touching the network.

**Architecture:** Mirror the energy sidecar end-to-end. A new `LastFMClient.get_artist_top_tracks` fetches `artist.gettoptracks`; a new offline analyze stage (`popularity`) fetches qualifying artists (caching raw ranked lists in `ai_genre_enrichment.db` so re-runs skip), resolves each Last.fm track to one local `track_id` (mbid-first, then loose-title + version-preference so a remaster keeps the hit and a live take loses it), and writes `popularity_sidecar.npz`. A runtime `popularity_loader` reads it. **This plan builds the data path only** — consuming it in medoid selection + the GUI controls is plan 2b.

**Tech Stack:** Python 3.11+, NumPy, SQLite, the existing `LastFMClient` + `RateLimiter`, `src/title_dedupe.py`.

**Spec:** `docs/superpowers/specs/2026-06-23-artist-energy-spread-popular-seeds-design.md` (Component 2). **Read it** — especially the canonical-version resolution.

## Global Constraints

- **Python 3.11+.**
- **Local-first (hard):** Last.fm is called ONLY in the offline `popularity` analyze stage. Generation/runtime reads only `popularity_sidecar.npz`. No network at generation time, ever.
- **`metadata.db` is irreplaceable — READ ONLY.** This plan reads `tracks` (track_id, title, norm_title, musicbrainz_id, artist_key) but NEVER writes it. The raw fetch cache goes in `data/ai_genre_enrichment.db` (gitignored enrichment store, safe to write). The sidecar is a new npz.
- **No remaster penalty (Dylan's hard requirement).** Resolution prefers studio/remaster over live/demo/alt via `title_dedupe.calculate_version_preference_score` (live −30 / demo −25 / remix −20, remaster only −5). A live cut wins only if it is the song's sole local version. The "penalty" is version-preference rank among real matches, never a match failure.
- **Popularity basis = per-artist RANK** (not raw playcount): for an artist's N ranked top tracks, the track at 0-based rank `r` gets `1.0 − r/N`. Rank is robust across artists of wildly different global play counts. Unmatched local tracks → `NaN` (neutral).
- **Sidecar lives beside the artifact:** `data/artifacts/beat3tower_32k/popularity/popularity_sidecar.npz`, with `track_ids` taken verbatim (same order) from `data/artifacts/beat3tower_32k/data_matrices_step1.npz["track_ids"]`, mirroring `scripts/extract_energy_sidecar.py::_merge()`.
- **Tests:** `python -m pytest -q ...` DIRECTLY — never pipe through tail/head. On Windows, if many `tmp_path` tests error with `PermissionError` on `pytest-of-Dylan`, pass `--basetemp=<scratchpad>/pt`.
- **A configured knob that can't act warns loudly** (runtime loader: weight>0 but sidecar missing → WARN, not silent).

## File Structure

- **Modify** `src/lastfm_client.py` — add `get_artist_top_tracks` (mirrors `get_similar_artists`).
- **Create** `src/analyze/popularity_runner.py` — the build side: cache table helpers (in `ai_genre_enrichment.db`), the resolver (`resolve_top_tracks_to_popularity`), and the sidecar builder (`build_popularity_sidecar`). Mirrors `src/analyze/energy_runner.py` + `scripts/extract_energy_sidecar.py`.
- **Modify** `scripts/analyze_library.py` — add `stage_popularity` + register in `STAGE_FUNCS`.
- **Modify** `src/playlist/request_models.py` — add `"popularity"` to `AnalyzeLibraryStage` + `ANALYZE_LIBRARY_STAGE_ORDER`.
- **Create** `src/playlist/popularity_loader.py` — runtime read (mirrors `src/playlist/energy_loader.py`).
- **Modify** `config.example.yaml` — `lastfm.artist_top_tracks_limit`, `playlists.ds_pipeline.artist_style.toptracks_min_artist_tracks`.
- **Test** `tests/unit/test_lastfm_top_tracks.py`, `tests/unit/test_popularity_resolver.py`, `tests/unit/test_popularity_sidecar_build.py`, `tests/unit/test_popularity_loader.py`.

---

### Task 1: `LastFMClient.get_artist_top_tracks`

**Files:**
- Modify: `src/lastfm_client.py` (add method near `get_similar_artists`, ~line 651)
- Test: `tests/unit/test_lastfm_top_tracks.py`

**Interfaces:**
- Produces: `LastFMClient.get_artist_top_tracks(self, artist_name: str, limit: int = 50) -> List[Dict[str, Any]]` — returns ranked list, each `{"name": str, "playcount": int, "listeners": int, "mbid": str, "rank": int}` (rank is 0-based position in the returned order).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_lastfm_top_tracks.py`:

```python
from unittest.mock import patch
from src.lastfm_client import LastFMClient


def _client():
    return LastFMClient(api_key="k", username="u")


def test_get_artist_top_tracks_parses_ranked_list():
    fake = {
        "toptracks": {
            "track": [
                {"name": "Smells Like Teen Spirit", "playcount": "9000000",
                 "listeners": "2000000", "mbid": "mbid-slts"},
                {"name": "Come as You Are", "playcount": "7000000",
                 "listeners": "1800000", "mbid": ""},
            ]
        }
    }
    with patch.object(LastFMClient, "_make_request", return_value=fake):
        out = _client().get_artist_top_tracks("Nirvana", limit=50)
    assert [t["name"] for t in out] == ["Smells Like Teen Spirit", "Come as You Are"]
    assert out[0]["playcount"] == 9000000 and out[0]["mbid"] == "mbid-slts"
    assert out[0]["rank"] == 0 and out[1]["rank"] == 1


def test_get_artist_top_tracks_handles_single_and_empty():
    single = {"toptracks": {"track": {"name": "X", "playcount": "5", "mbid": ""}}}
    with patch.object(LastFMClient, "_make_request", return_value=single):
        out = _client().get_artist_top_tracks("A")
    assert len(out) == 1 and out[0]["name"] == "X" and out[0]["rank"] == 0
    with patch.object(LastFMClient, "_make_request", return_value=None):
        assert _client().get_artist_top_tracks("A") == []
    with patch.object(LastFMClient, "_make_request", return_value={"toptracks": {}}):
        assert _client().get_artist_top_tracks("A") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_lastfm_top_tracks.py`
Expected: FAIL — `AttributeError: 'LastFMClient' object has no attribute 'get_artist_top_tracks'`

- [ ] **Step 3: Implement the method**

In `src/lastfm_client.py`, after `get_similar_artists` (~line 682), add:

```python
    def get_artist_top_tracks(self, artist_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch an artist's most-popular tracks (artist.gettoptracks), ranked.

        Returns a list of {name, playcount, listeners, mbid, rank}; rank is the
        0-based position (0 = most popular). [] on missing/empty response.
        """
        data = self._make_request('artist.gettoptracks', {
            'artist': artist_name,
            'limit': limit,
            'autocorrect': 1,
        })
        if not data or 'toptracks' not in data:
            return []
        tracks = data['toptracks'].get('track', [])
        if not isinstance(tracks, list):
            tracks = [tracks]
        out: List[Dict[str, Any]] = []
        for rank, t in enumerate(tracks):
            name = str(t.get('name', '')).strip()
            if not name:
                continue
            out.append({
                'name': name,
                'playcount': int(t.get('playcount', 0) or 0),
                'listeners': int(t.get('listeners', 0) or 0),
                'mbid': str(t.get('mbid', '') or ''),
                'rank': rank,
            })
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_lastfm_top_tracks.py`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/lastfm_client.py tests/unit/test_lastfm_top_tracks.py
git commit -m "feat(lastfm): get_artist_top_tracks (artist.gettoptracks, ranked)"
```

---

### Task 2: Artist top-tracks cache (new table in the enrichment DB)

**Files:**
- Create: `src/analyze/popularity_runner.py`
- Test: `tests/unit/test_popularity_sidecar_build.py` (cache portion)

**Interfaces:**
- Produces:
  - `ENRICHMENT_DB_DEFAULT = "data/ai_genre_enrichment.db"`
  - `init_top_tracks_cache(db_path: str) -> None` — creates `artist_top_tracks_cache(artist_key TEXT PRIMARY KEY, fetched_at TEXT, track_count INTEGER, payload_json TEXT)`.
  - `cached_artist_keys(db_path: str) -> set[str]` — keys already fetched.
  - `upsert_artist_top_tracks(db_path: str, artist_key: str, fetched_at: str, top_tracks: list[dict]) -> None` — stores `json.dumps(top_tracks)`.
  - `get_artist_top_tracks_cached(db_path: str, artist_key: str) -> list[dict]` — `[]` if absent.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_popularity_sidecar_build.py`:

```python
from src.analyze.popularity_runner import (
    init_top_tracks_cache, cached_artist_keys,
    upsert_artist_top_tracks, get_artist_top_tracks_cached,
)


def test_cache_roundtrip_and_skip_set(tmp_path):
    db = str(tmp_path / "enrich.db")
    init_top_tracks_cache(db)
    assert cached_artist_keys(db) == set()
    rows = [{"name": "X", "playcount": 5, "mbid": "", "rank": 0}]
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00Z", rows)
    assert cached_artist_keys(db) == {"nirvana"}
    assert get_artist_top_tracks_cached(db, "nirvana") == rows
    # upsert replaces
    upsert_artist_top_tracks(db, "nirvana", "later", [])
    assert get_artist_top_tracks_cached(db, "nirvana") == []
    assert get_artist_top_tracks_cached(db, "absent") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_popularity_sidecar_build.py::test_cache_roundtrip_and_skip_set`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analyze.popularity_runner'`

- [ ] **Step 3: Implement the cache helpers**

Create `src/analyze/popularity_runner.py`:

```python
"""Build side of the Last.fm popularity sidecar.

Fetches each artist's top tracks (cached in ai_genre_enrichment.db so re-runs
skip), resolves each Last.fm track to the *canonical* local track per song
(mbid-first, then loose-title + version-preference), and writes
data/artifacts/beat3tower_32k/popularity/popularity_sidecar.npz aligned to the
artifact's track_ids. Mirrors the energy sidecar. Reads metadata.db read-only.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ENRICHMENT_DB_DEFAULT = "data/ai_genre_enrichment.db"


def init_top_tracks_cache(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS artist_top_tracks_cache ("
            "artist_key TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, "
            "track_count INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL)"
        )


def cached_artist_keys(db_path: str) -> set:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT artist_key FROM artist_top_tracks_cache")
        return {r[0] for r in rows}


def upsert_artist_top_tracks(
    db_path: str, artist_key: str, fetched_at: str, top_tracks: List[dict]
) -> None:
    payload = json.dumps(top_tracks)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO artist_top_tracks_cache (artist_key, fetched_at, track_count, payload_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(artist_key) DO UPDATE SET "
            "fetched_at=excluded.fetched_at, track_count=excluded.track_count, "
            "payload_json=excluded.payload_json",
            (artist_key, fetched_at, len(top_tracks), payload),
        )


def get_artist_top_tracks_cached(db_path: str, artist_key: str) -> List[dict]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM artist_top_tracks_cache WHERE artist_key = ?",
            (artist_key,),
        ).fetchone()
    return json.loads(row[0]) if row else []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_popularity_sidecar_build.py::test_cache_roundtrip_and_skip_set`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analyze/popularity_runner.py tests/unit/test_popularity_sidecar_build.py
git commit -m "feat(popularity): artist top-tracks cache table in enrichment db"
```

---

### Task 3: The resolver — Last.fm names → canonical local track_id (THE crux)

**Files:**
- Modify: `src/analyze/popularity_runner.py` (add `resolve_top_tracks_to_popularity`)
- Test: `tests/unit/test_popularity_resolver.py`

**Interfaces:**
- Consumes: `top_tracks` from the cache (Task 2 shape), `local_tracks` rows (see below).
- Produces: `resolve_top_tracks_to_popularity(top_tracks: list[dict], local_tracks: list[dict]) -> dict[str, float]` — maps `track_id → popularity in [0,1]`. `local_tracks` items: `{"track_id": str, "title": str, "musicbrainz_id": str}`. mbid-first match, else loose-title group + highest version-preference. Rank score `1.0 − rank/N`; if two top tracks resolve to the same track_id, keep the higher score.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_popularity_resolver.py`:

```python
from src.analyze.popularity_runner import resolve_top_tracks_to_popularity


def test_resolver_prefers_studio_over_live_and_honors_remaster_and_mbid():
    # Last.fm top tracks (canonical names), ranked
    top = [
        {"name": "Smells Like Teen Spirit", "mbid": "mbid-slts", "rank": 0},
        {"name": "In Bloom", "mbid": "", "rank": 1},
        {"name": "Come as You Are", "mbid": "", "rank": 2},
    ]
    local = [
        # SLTS only as a remaster -> must still match + carry top popularity
        {"track_id": "t_slts_rem", "title": "Smells Like Teen Spirit (2021 Remaster)", "musicbrainz_id": "mbid-slts"},
        # In Bloom: studio + live -> studio must win
        {"track_id": "t_inbloom_studio", "title": "In Bloom", "musicbrainz_id": ""},
        {"track_id": "t_inbloom_live", "title": "In Bloom (Live In Seattle)", "musicbrainz_id": ""},
        # Come as You Are: only studio
        {"track_id": "t_caya", "title": "Come as You Are", "musicbrainz_id": ""},
        # an unrelated deep cut -> no popularity
        {"track_id": "t_deep", "title": "Endless, Nameless", "musicbrainz_id": ""},
    ]
    pop = resolve_top_tracks_to_popularity(top, local)
    assert pop["t_slts_rem"] == 1.0                  # rank 0, matched via mbid; remaster NOT penalized
    assert "t_inbloom_studio" in pop                  # studio In Bloom got the popularity
    assert "t_inbloom_live" not in pop                # live lost to studio
    assert pop["t_inbloom_studio"] > pop["t_caya"]    # rank 1 > rank 2
    assert "t_deep" not in pop                         # unmatched deep cut neutral


def test_resolver_keeps_higher_score_on_collision_and_handles_empty():
    top = [{"name": "Song", "mbid": "", "rank": 0}, {"name": "song", "mbid": "", "rank": 1}]
    local = [{"track_id": "t1", "title": "Song", "musicbrainz_id": ""}]
    pop = resolve_top_tracks_to_popularity(top, local)
    assert pop["t1"] == 1.0   # both ranks map to t1; keep the higher (rank 0)
    assert resolve_top_tracks_to_popularity([], local) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_popularity_resolver.py`
Expected: FAIL — `ImportError: cannot import name 'resolve_top_tracks_to_popularity'`

- [ ] **Step 3: Implement the resolver**

Append to `src/analyze/popularity_runner.py`:

```python
def resolve_top_tracks_to_popularity(
    top_tracks: List[dict], local_tracks: List[dict]
) -> Dict[str, float]:
    """Map one artist's ranked Last.fm top tracks to local track_ids + popularity.

    mbid-first, else loose-normalized-title grouping with version-preference
    (studio/remaster beat live/demo/alt). Score = 1 - rank/N. On collision keep
    the higher score. Returns {track_id: popularity in [0,1]}.
    """
    if not top_tracks or not local_tracks:
        return {}
    from src.title_dedupe import (
        calculate_version_preference_score,
        normalize_title_for_dedupe,
    )

    by_mbid: Dict[str, str] = {}
    by_norm: Dict[str, List[dict]] = {}
    for lt in local_tracks:
        mbid = str(lt.get("musicbrainz_id") or "")
        if mbid:
            by_mbid.setdefault(mbid, str(lt["track_id"]))
        norm = normalize_title_for_dedupe(str(lt.get("title") or ""), mode="loose")
        if norm:
            by_norm.setdefault(norm, []).append(lt)

    n = len(top_tracks)
    out: Dict[str, float] = {}
    for t in top_tracks:
        rank = int(t.get("rank", 0))
        score = 1.0 - rank / n
        tid: Optional[str] = None
        mbid = str(t.get("mbid") or "")
        if mbid and mbid in by_mbid:
            tid = by_mbid[mbid]
        else:
            norm = normalize_title_for_dedupe(str(t.get("name") or ""), mode="loose")
            cands = by_norm.get(norm, [])
            if cands:
                best = max(
                    cands,
                    key=lambda lt: (
                        calculate_version_preference_score(str(lt.get("title") or "")),
                        str(lt["track_id"]),
                    ),
                )
                tid = str(best["track_id"])
        if tid is not None and score > out.get(tid, -1.0):
            out[tid] = score
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_popularity_resolver.py`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/analyze/popularity_runner.py tests/unit/test_popularity_resolver.py
git commit -m "feat(popularity): resolve Last.fm top tracks to canonical local track_id"
```

---

### Task 4: Sidecar builder + `stage_popularity` (offline fetch → resolve → npz)

**Files:**
- Modify: `src/analyze/popularity_runner.py` (add `build_popularity_sidecar`)
- Modify: `scripts/analyze_library.py` (add `stage_popularity`, register in `STAGE_FUNCS`)
- Modify: `src/playlist/request_models.py` (add `"popularity"` to the Literal + the order tuple)
- Modify: `config.example.yaml`
- Test: `tests/unit/test_popularity_sidecar_build.py` (build portion)

**Interfaces:**
- Consumes: `resolve_top_tracks_to_popularity`, `get_artist_top_tracks_cached`, the cache helpers.
- Produces: `build_popularity_sidecar(*, artifact_npz: str, metadata_db: str, enrichment_db: str, out_path: str, min_artist_tracks: int) -> dict` — reads `track_ids` from `artifact_npz`, groups local tracks by `artist_key`, resolves each cached artist, writes `out_path` (`track_ids` + `popularity` float32, NaN default), returns `{"tracks": int, "matched": int, "artists_resolved": int}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_popularity_sidecar_build.py`:

```python
import numpy as np
import sqlite3
from src.analyze.popularity_runner import build_popularity_sidecar, init_top_tracks_cache, upsert_artist_top_tracks


def _make_artifact(tmp_path, track_ids):
    p = tmp_path / "data_matrices_step1.npz"
    np.savez(p, track_ids=np.array(track_ids, dtype=object))
    return str(p)


def _make_metadata(tmp_path, rows):
    # rows: (track_id, title, musicbrainz_id, artist_key)
    db = str(tmp_path / "metadata.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tracks (track_id TEXT, title TEXT, musicbrainz_id TEXT, artist_key TEXT)")
        conn.executemany("INSERT INTO tracks VALUES (?,?,?,?)", rows)
    return db


def test_build_popularity_sidecar_aligns_and_resolves(tmp_path):
    track_ids = ["a_studio", "a_live", "b_only"]
    artifact = _make_artifact(tmp_path, track_ids)
    meta = _make_metadata(tmp_path, [
        ("a_studio", "In Bloom", "", "nirvana"),
        ("a_live", "In Bloom (Live)", "", "nirvana"),
        ("b_only", "Tom Courtenay", "", "yo la tengo"),
    ])
    enrich = str(tmp_path / "enrich.db")
    init_top_tracks_cache(enrich)
    upsert_artist_top_tracks(enrich, "nirvana", "t", [{"name": "In Bloom", "mbid": "", "rank": 0}])
    # yo la tengo NOT cached -> b_only stays NaN
    out = str(tmp_path / "popularity" / "popularity_sidecar.npz")
    stats = build_popularity_sidecar(
        artifact_npz=artifact, metadata_db=meta, enrichment_db=enrich,
        out_path=out, min_artist_tracks=1,
    )
    z = np.load(out, allow_pickle=True)
    ids = [str(t) for t in z["track_ids"]]
    pop = z["popularity"]
    assert ids == track_ids                       # aligned to artifact order
    assert pop[ids.index("a_studio")] == 1.0      # studio In Bloom got popularity
    assert np.isnan(pop[ids.index("a_live")])     # live did not
    assert np.isnan(pop[ids.index("b_only")])     # uncached artist -> neutral
    assert stats["matched"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_popularity_sidecar_build.py::test_build_popularity_sidecar_aligns_and_resolves`
Expected: FAIL — `ImportError: cannot import name 'build_popularity_sidecar'`

- [ ] **Step 3: Implement the builder**

Append to `src/analyze/popularity_runner.py`:

```python
def _local_tracks_by_artist(metadata_db: str, min_artist_tracks: int) -> Dict[str, List[dict]]:
    by_artist: Dict[str, List[dict]] = {}
    with sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT track_id, title, musicbrainz_id, artist_key FROM tracks "
            "WHERE artist_key IS NOT NULL AND artist_key <> ''"
        ):
            by_artist.setdefault(str(r["artist_key"]), []).append({
                "track_id": str(r["track_id"]),
                "title": str(r["title"] or ""),
                "musicbrainz_id": str(r["musicbrainz_id"] or ""),
            })
    return {k: v for k, v in by_artist.items() if len(v) >= min_artist_tracks}


def build_popularity_sidecar(
    *, artifact_npz: str, metadata_db: str, enrichment_db: str,
    out_path: str, min_artist_tracks: int,
) -> dict:
    """Resolve cached Last.fm top tracks to local track_ids and write the sidecar."""
    tids = [str(t) for t in np.load(artifact_npz, allow_pickle=True)["track_ids"]]
    pos = {t: i for i, t in enumerate(tids)}
    popularity = np.full(len(tids), np.nan, dtype=np.float32)

    by_artist = _local_tracks_by_artist(metadata_db, min_artist_tracks)
    cached = cached_artist_keys(enrichment_db)
    matched = artists_resolved = 0
    for artist_key, local_tracks in by_artist.items():
        if artist_key not in cached:
            continue
        top = get_artist_top_tracks_cached(enrichment_db, artist_key)
        if not top:
            continue
        artists_resolved += 1
        for tid, score in resolve_top_tracks_to_popularity(top, local_tracks).items():
            j = pos.get(tid)
            if j is not None:
                popularity[j] = score
                matched += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, track_ids=np.array(tids, dtype=object), popularity=popularity,
    )
    logger.info("popularity sidecar: %d tracks, %d matched, %d artists resolved -> %s",
                len(tids), matched, artists_resolved, out_path)
    return {"tracks": len(tids), "matched": matched, "artists_resolved": artists_resolved}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_popularity_sidecar_build.py`
Expected: PASS (both cache + build tests)

- [ ] **Step 5: Register the analyze stage**

In `src/playlist/request_models.py`, add `"popularity"` to BOTH the `AnalyzeLibraryStage` Literal (after `"energy"`) and the `ANALYZE_LIBRARY_STAGE_ORDER` tuple (after `"energy"`).

In `scripts/analyze_library.py`, add the stage handler (place near `stage_energy`, ~line 2337) — it fetches qualifying artists not already cached, then builds the sidecar:

```python
def stage_popularity(ctx: Dict) -> Dict:
    """Fetch each qualifying artist's Last.fm top tracks (cached, resumable) and
    build the popularity sidecar. Offline only; never touched at generation."""
    from src.analyze.popularity_runner import (
        ENRICHMENT_DB_DEFAULT, init_top_tracks_cache, cached_artist_keys,
        upsert_artist_top_tracks, build_popularity_sidecar,
    )
    from src.lastfm_client import LastFMClient
    import sqlite3 as _sqlite
    import time as _time
    import yaml
    from datetime import datetime, timezone

    args = ctx["args"]
    out_dir = Path(ctx["out_dir"])
    artifact_npz = out_dir / "data_matrices_step1.npz"
    if not artifact_npz.exists():
        logger.info("stage_popularity: artifact missing; skipping (build artifacts first)")
        return {"skipped": True, "reason": "no_artifact"}
    api_key = _resolve_lastfm_api_key(ctx)
    if not api_key:
        raise RuntimeError("stage_popularity requires a Last.fm API key (config lastfm.api_key / LASTFM_API_KEY)")

    with open(ctx["config_path"], "r", encoding="utf-8") as _fh:
        cfg = yaml.safe_load(_fh) or {}
    limit = int(((cfg.get("lastfm") or {}).get("artist_top_tracks_limit", 50)))
    min_tracks = int((((cfg.get("playlists") or {}).get("ds_pipeline") or {}).get("artist_style") or {})
                     .get("toptracks_min_artist_tracks", 8))
    username = (cfg.get("lastfm") or {}).get("username", "")

    enrich_db = str(ENRICHMENT_DB_DEFAULT)
    init_top_tracks_cache(enrich_db)
    # qualifying artists: >= min_tracks local tracks, not already cached
    with _sqlite.connect(f"file:{ctx['db_path']}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT artist_key, MIN(artist) AS name, COUNT(*) c FROM tracks "
            "WHERE artist_key IS NOT NULL AND artist_key <> '' "
            "GROUP BY artist_key HAVING c >= ?", (min_tracks,),
        ).fetchall()
    already = cached_artist_keys(enrich_db)
    pending = [(r[0], r[1]) for r in rows if args.force or r[0] not in already]

    client = LastFMClient(api_key=api_key, username=username)
    fetched = failed = 0
    for artist_key, name in pending:
        try:
            top = client.get_artist_top_tracks(name, limit=limit)
            upsert_artist_top_tracks(
                enrich_db, artist_key,
                datetime.now(timezone.utc).isoformat(), top,
            )
            fetched += 1
        except Exception as exc:  # network/parse — log and continue
            failed += 1
            logger.warning("popularity fetch failed for %s: %s", name, exc)
        _time.sleep(0.2)  # ~5 req/s courtesy

    stats = build_popularity_sidecar(
        artifact_npz=str(artifact_npz), metadata_db=str(ctx["db_path"]),
        enrichment_db=enrich_db, out_path=str(out_dir / "popularity" / "popularity_sidecar.npz"),
        min_artist_tracks=min_tracks,
    )
    return {"skipped": False, "fetched": fetched, "failed": failed,
            "pending": len(pending), **stats}
```

Then register it in `STAGE_FUNCS` (after `"energy": stage_energy,`):

```python
    "popularity": stage_popularity,
```

(The stage reads `ctx["config_path"]` directly with `yaml.safe_load` — see the
handler above — so it needs no new config helper. The ctx keys used
(`args`, `out_dir`, `db_path`, `config_path`) and `_resolve_lastfm_api_key(ctx)`
are exactly those `stage_lastfm` / `stage_energy` already rely on.)

- [ ] **Step 6: Document config**

In `config.example.yaml`, under the `lastfm:` block add:

```yaml
  artist_top_tracks_limit: 50      # how many top tracks to fetch per artist (popularity stage)
```

and under `playlists: ds_pipeline: artist_style:` add:

```yaml
      toptracks_min_artist_tracks: 8   # only fetch Last.fm top-tracks for artists with >= N local tracks
```

- [ ] **Step 7: Run tests + stage import smoke**

Run: `python -m pytest -q tests/unit/test_popularity_sidecar_build.py` (expect PASS)
Run: `python -c "import scripts.analyze_library as a; assert 'popularity' in a.STAGE_FUNCS"` (expect no error)
Run: `python -c "from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER as o; assert 'popularity' in o"`

- [ ] **Step 8: Commit**

```bash
git add src/analyze/popularity_runner.py scripts/analyze_library.py src/playlist/request_models.py config.example.yaml tests/unit/test_popularity_sidecar_build.py
git commit -m "feat(popularity): sidecar builder + offline 'popularity' analyze stage"
```

---

### Task 5: Runtime loader `popularity_loader.py`

**Files:**
- Create: `src/playlist/popularity_loader.py`
- Test: `tests/unit/test_popularity_loader.py`

**Interfaces:**
- Produces: `load_popularity_vector(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray` — shape `(len(track_ids),)`, the per-track popularity aligned to `track_ids`, `NaN` for gaps/missing-file. Mirrors `energy_loader.load_energy_matrix` (no z-scoring — popularity is already [0,1]).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_popularity_loader.py`:

```python
import numpy as np
from src.playlist.popularity_loader import load_popularity_vector


def test_load_popularity_aligns_and_nan_for_gaps(tmp_path):
    side = tmp_path / "popularity_sidecar.npz"
    np.savez(side, track_ids=np.array(["a", "b", "c"], dtype=object),
             popularity=np.array([1.0, np.nan, 0.5], dtype=np.float32))
    out = load_popularity_vector(["c", "a", "zzz"], sidecar_path=str(side))
    assert out.shape == (3,)
    assert out[0] == 0.5 and out[1] == 1.0
    assert np.isnan(out[2])            # not in sidecar -> NaN


def test_load_popularity_missing_file_is_all_nan(tmp_path):
    out = load_popularity_vector(["a", "b"], sidecar_path=str(tmp_path / "nope.npz"))
    assert out.shape == (2,) and np.all(np.isnan(out))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_popularity_loader.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.playlist.popularity_loader'`

- [ ] **Step 3: Implement the loader**

Create `src/playlist/popularity_loader.py`:

```python
"""Load a per-track popularity vector from the popularity sidecar.

Runtime-only consumer (no Last.fm import). Mirrors energy_loader: returns values
aligned to the requested track_ids, NaN for gaps. Popularity is already a
per-artist rank score in [0,1] — no z-scoring.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_popularity_vector(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray:
    """Return (len(track_ids),) popularity aligned to track_ids; NaN for gaps."""
    n = len(track_ids)
    out = np.full(n, np.nan, dtype=float)
    if not Path(sidecar_path).exists():
        logger.info("popularity_loader: sidecar missing at %s; all-NaN (neutral)", sidecar_path)
        return out
    z = np.load(sidecar_path, allow_pickle=True)
    if "popularity" not in z or "track_ids" not in z:
        logger.warning("popularity_loader: sidecar missing expected keys; all-NaN")
        return out
    pos = {str(t): i for i, t in enumerate(z["track_ids"])}
    col = np.asarray(z["popularity"], dtype=float)
    for ti, tid in enumerate(track_ids):
        j = pos.get(str(tid))
        if j is not None and np.isfinite(col[j]):
            out[ti] = col[j]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_popularity_loader.py`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/popularity_loader.py tests/unit/test_popularity_loader.py
git commit -m "feat(popularity): runtime popularity_loader (mirrors energy_loader)"
```

---

## Final verification

- [ ] Full focused suite (fresh basetemp): `python -m pytest -q -p no:cacheprovider --basetemp=<scratchpad>/pt tests/unit/test_lastfm_top_tracks.py tests/unit/test_popularity_resolver.py tests/unit/test_popularity_sidecar_build.py tests/unit/test_popularity_loader.py`
- [ ] `ruff check src/analyze/popularity_runner.py src/playlist/popularity_loader.py src/lastfm_client.py`
- [ ] `mypy src/analyze/popularity_runner.py src/playlist/popularity_loader.py`
- [ ] Stage registered: `'popularity'` in both `ANALYZE_LIBRARY_STAGE_ORDER` and `STAGE_FUNCS`.

## Operational step (after the code lands — gates plan 2b)

Run the offline fetch + build over the real library (one-time, resumable, rate-limited; needs a Last.fm API key in `config.yaml`):

```bash
python scripts/analyze_library.py --stages popularity
```

Then **validate the resolution with the probe lens** — confirm Nirvana's resolved popular tracks are the studio hits (Smells Like Teen Spirit, Come as You Are, In Bloom, Heart-Shaped Box), not live takes; check the sidecar's `matched` count is reasonable; spot-check that remasters carried popularity. Only once the sidecar resolves cleanly does plan 2b (the `w_pop` medoid term + "Popular Seeds" checkbox + "New Seeds" button) become meaningful.

## Two consumption paths (decided 2026-06-24) — why the eager sidecar is retained

This data path feeds **two** features with different needs:

- **"Popular Seeds"** (plan 2b) needs popularity for only the **seed artist's own tracks**
  (the piers). The seed artist is known at generation → fetch it **lazily, cache-first**
  (read the per-artist cache; on miss/stale fetch one artist + cache; on failure →
  neutral). This mirrors how `get_recent_tracks` history already works
  (`playlist_generator.py:1211`). It needs **no batch run** — generate Nirvana, fetch
  Nirvana, done. So Popular Seeds does NOT depend on the `popularity` stage or the
  library-wide sidecar.

- **"Oops, All Bangers"** (future plan) optimizes **every** track — including the bridge
  tracks, which come from *other* artists not known in advance — toward each artist's
  top-N hits. You cannot lazily fetch the top-tracks of every candidate's artist
  mid-generation, so this mode needs popularity **precomputed library-wide** = exactly
  what the eager `popularity` stage + `popularity_sidecar.npz` + `popularity_loader`
  produce (every track → its rank among its own artist's hits). **This is why the eager
  stage/sidecar/loader are retained** even though Popular Seeds doesn't use them.

Both paths SHARE the per-artist cache + the resolver, and both use per-artist rank, so
values are consistent; the only difference is freshness (seed artist always fresh via
lazy; All-Bangers bridges as-of-last-batch via the sidecar). Plan 2b adds the lazy
`load_artist_popularity(seed_artist)` path; it does not remove anything here.

## Out of scope (plan 2b)

- `medoid_popularity_weight` config + `load_artist_popularity_values` + the popularity term in `_medoids_for_cluster` (mirrors the energy term; `w_pop < w_energy` so energy-spread wins, popularity picks the recognizable track within each slot — which also resolves the energy-pulls-in-a-rehearsal residue found in plan 1).
- `popular_seeds: bool` + `seed_epoch: int` through `GenerateRequestBody` → `GeneratePlaylistRequest` → worker → `create_playlist_for_artist` (which already has both params; worker just needs to pass them).
- React: `web/src/lib/types.ts` fields + "Popular Seeds" checkbox + "New Seeds" reroll button in `GenerateControls.tsx`.
