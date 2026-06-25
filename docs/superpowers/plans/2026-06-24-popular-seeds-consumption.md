# Artist-mode "Popular Seeds" — Consumption + GUI (plan 2b of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The React task additionally REQUIRES the `web-gui` skill (stale-dist / worker-restart / end-to-end-wiring traps).

**Goal:** When "Popular Seeds" is on, bias each artist-mode pier toward the artist's recognizable hits — by fetching the seed artist's Last.fm top tracks *lazily at generation* (cache-first), resolving them to canonical local tracks (the plan-2a resolver), and adding a `w_pop` term to the medoid score (with `w_energy > w_pop` so energy-spread keeps the structure and popularity picks the recognizable track within each slot). Plus a "New Seeds" reroll button.

**Architecture:** Lazy, not batch (see plan 2a's "Two consumption paths" note). At generation, `create_playlist_for_artist` fetches the seed artist's top tracks cache-first (mirrors `get_recent_tracks` history at `playlist_generator.py:1211`), resolves to that artist's local tracks, and injects a `popularity_values` vector into `cluster_artist_tracks` — exactly parallel to how `energy_values` is injected. The network client stays in the orchestrator (which has `self.lastfm` + `self.config`); `cluster_artist_tracks`/`_medoids_for_cluster` stay client-free. Never gates generation: cache hit = instant, miss = one rate-limited call then cached, failure = neutral popularity (playlist still generates).

**Tech Stack:** Python 3.11+, NumPy, SQLite, the plan-2a `popularity_runner` (cache + resolver), `LastFMClient`; React + TypeScript + Vite (web GUI).

**Spec:** `docs/superpowers/specs/2026-06-23-artist-energy-spread-popular-seeds-design.md` (Component 2 `w_pop` + Component 3 GUI). **Reuses plan 2a:** `get_artist_top_tracks`, `get_artist_top_tracks_cached`/`upsert_artist_top_tracks`, `resolve_top_tracks_to_popularity`.

## Global Constraints

- **Python 3.11+.**
- **Local-first / never gates generation:** the Last.fm fetch is cache-first with a freshness window and a graceful fallback. A miss → one fetch (cached); offline/API failure → neutral popularity (NaN→0), playlist still generates. No batch dependency — works whether or not the `popularity` analyze stage ever ran.
- **Opt-in, backward-compatible:** `medoid_popularity_weight: 0.0` default + the "Popular Seeds" checkbox default OFF ⇒ byte-identical to today. `popularity_values=None`/all-NaN ⇒ the term is inert.
- **`metadata.db` READ-ONLY.** `load_artist_popularity_values` builds the seed artist's local tracks from the **artifact bundle** (`track_ids` + `track_titles`), not a metadata write. (mbid matching is a future precision add; title matching via the resolver is sufficient for one artist's own tracks.)
- **Energy-spread wins:** keep `medoid_energy_weight > medoid_popularity_weight` so popularity picks *within* each energy slot, never reorders the slots.
- **Config path:** `playlists.ds_pipeline.artist_style.*` (the two `ArtistStyleConfig` construction sites are `playlist_generator.py:1625` and `:2554`).
- **Tests:** `python -m pytest -q ...` DIRECTLY (never pipe). On Windows tmp_path `PermissionError`, pass `--basetemp=<scratchpad>/pt`. Last.fm is MOCKED in all unit tests — no live calls during implementation. The one live call is the operational validation (Task 0), run explicitly.
- **React:** follow the `web-gui` skill — rebuild `web/dist` and restart `serve_web.py` after edits; verify the value reaches the worker (end-to-end), don't trust the UI alone.

## File Structure

- **Modify** `src/analyze/popularity_runner.py` — add `get_artist_top_tracks_cached_or_fetch` (lazy cache-first) + `load_artist_popularity_values` (bundle → vector).
- **Modify** `src/playlist/artist_style.py` — `medoid_popularity_weight` config field; `popularity_weight`/`popularity_values` in `_medoids_for_cluster`; `popularity_values` kwarg + use in `cluster_artist_tracks`.
- **Modify** `src/playlist_generator.py` — wire both `ArtistStyleConfig` sites; in `create_playlist_for_artist`, accept `popular_seeds`, load popularity when active, inject into `cluster_artist_tracks`; pass `seed_epoch` through.
- **Modify** `src/playlist_web/schemas.py`, `src/playlist/request_models.py`, `src/playlist_gui/worker.py` — `popular_seeds: bool`, `seed_epoch: int` plumbing.
- **Modify** `config.example.yaml` — `medoid_popularity_weight`, `popular_seeds_weight`, `popularity_max_age_days`.
- **Modify** `web/src/lib/types.ts`, `web/src/components/GenerateControls.tsx` — checkbox + reroll button.
- **Test** `tests/unit/test_popularity_lazy.py`, additions to `tests/test_artist_style.py`.

---

### Task 0 (operational gate): live-validate the resolver on a real artist

**Not a code task — a validation checkpoint. Run before Task 3 builds on the resolver.** Needs the user's Last.fm key + one live API call. Confirms Last.fm's real top-track names resolve to the studio hits in the real library (the one thing unit tests can't prove).

- [ ] **Step 1: One-artist live resolve probe**

Write a throwaway probe (scratchpad, not committed) that: constructs `LastFMClient` from config, calls `get_artist_top_tracks("Nirvana", limit=50)`, reads Nirvana's local tracks from the REAL metadata.db (read-only, absolute path) as `{track_id, title, musicbrainz_id}`, runs `resolve_top_tracks_to_popularity`, and prints the top ~10 resolved `(title, score)` pairs.

- [ ] **Step 2: Eyeball the result**

Expected: the resolved high-popularity tracks are studio hits (*Smells Like Teen Spirit, Come as You Are, In Bloom, Heart-Shaped Box, …*), NOT live takes; remasters carry popularity; the match count is a sane fraction of the top-50. If resolution is poor (titles don't match), STOP and report — the version-keyword/normalization may need tuning before proceeding. If good, proceed to code.

---

### Task 1: Lazy cache-first fetch `get_artist_top_tracks_cached_or_fetch`

**Files:**
- Modify: `src/analyze/popularity_runner.py`
- Test: `tests/unit/test_popularity_lazy.py`

**Interfaces:**
- Consumes: `get_artist_top_tracks_cached`, `upsert_artist_top_tracks`, `init_top_tracks_cache` (plan 2a); a `client` with `.get_artist_top_tracks(name, limit)`.
- Produces: `get_artist_top_tracks_cached_or_fetch(artist_key, artist_name, *, client, db_path, limit=50, max_age_days=30, now_iso) -> list[dict]` — returns cached top tracks if present and fresher than `max_age_days`; else fetches (one call), caches, returns. On fetch failure: returns the stale cache if any, else `[]`. `now_iso` is injected (ISO-8601 string) so tests are deterministic.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_popularity_lazy.py`:

```python
from unittest.mock import MagicMock
from src.analyze.popularity_runner import (
    init_top_tracks_cache, upsert_artist_top_tracks,
    get_artist_top_tracks_cached_or_fetch,
)

ROWS = [{"name": "Hit", "playcount": 9, "mbid": "", "rank": 0}]


def test_cache_hit_skips_fetch(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-20T00:00:00+00:00", ROWS)
    client = MagicMock()
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00",
    )
    assert out == ROWS
    client.get_artist_top_tracks.assert_not_called()   # fresh cache -> no network


def test_miss_fetches_and_caches(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    client = MagicMock(); client.get_artist_top_tracks.return_value = ROWS
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db, now_iso="2026-06-24T00:00:00+00:00",
    )
    assert out == ROWS
    client.get_artist_top_tracks.assert_called_once()
    # now cached -> second call doesn't fetch
    client2 = MagicMock()
    again = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client2, db_path=db, now_iso="2026-06-24T00:00:00+00:00")
    client2.get_artist_top_tracks.assert_not_called()
    assert again == ROWS


def test_stale_refetches(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", [])  # old + empty
    fresh = [{"name": "New", "playcount": 1, "mbid": "", "rank": 0}]
    client = MagicMock(); client.get_artist_top_tracks.return_value = fresh
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    client.get_artist_top_tracks.assert_called_once()
    assert out == fresh


def test_fetch_failure_falls_back_to_stale_then_empty(tmp_path):
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-01-01T00:00:00+00:00", ROWS)  # stale
    client = MagicMock(); client.get_artist_top_tracks.side_effect = RuntimeError("net")
    out = get_artist_top_tracks_cached_or_fetch(
        "nirvana", "Nirvana", client=client, db_path=db,
        max_age_days=30, now_iso="2026-06-24T00:00:00+00:00")
    assert out == ROWS    # stale cache used on failure (graceful)
    # no cache at all + failure -> []
    out2 = get_artist_top_tracks_cached_or_fetch(
        "absent", "Absent", client=client, db_path=db, now_iso="2026-06-24T00:00:00+00:00")
    assert out2 == []
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest -q tests/unit/test_popularity_lazy.py`
Expected: FAIL — `ImportError: cannot import name 'get_artist_top_tracks_cached_or_fetch'`

- [ ] **Step 3: Implement**

Append to `src/analyze/popularity_runner.py` (add `from datetime import datetime, timezone` and `import logging`/`logger` if not present):

```python
def _fetched_at_iso(db_path: str, artist_key: str) -> Optional[str]:
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at FROM artist_top_tracks_cache WHERE artist_key = ?",
            (artist_key,),
        ).fetchone()
    return row[0] if row else None


def get_artist_top_tracks_cached_or_fetch(
    artist_key: str, artist_name: str, *, client, db_path: str,
    limit: int = 50, max_age_days: int = 30, now_iso: str,
) -> List[dict]:
    """Cache-first per-artist top tracks. Fresh cache -> no network. Miss/stale ->
    one fetch + cache. Fetch failure -> stale cache if any, else []. Never raises."""
    init_top_tracks_cache(db_path)
    cached = get_artist_top_tracks_cached(db_path, artist_key)
    fetched_at = _fetched_at_iso(db_path, artist_key)
    fresh = False
    if fetched_at is not None:
        try:
            age = datetime.fromisoformat(now_iso) - datetime.fromisoformat(fetched_at)
            fresh = age.total_seconds() <= max_age_days * 86400
        except ValueError:
            fresh = False
    if fetched_at is not None and fresh:
        return cached
    try:
        rows = client.get_artist_top_tracks(artist_name, limit=limit)
        upsert_artist_top_tracks(db_path, artist_key, now_iso, rows)
        return rows
    except Exception as exc:  # network/parse — never gate generation
        logger.warning("popularity lazy fetch failed for %s: %s; using stale/empty", artist_name, exc)
        return cached  # stale cache if present, else []
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest -q tests/unit/test_popularity_lazy.py` (expect 4 passed) + `ruff check src/analyze/popularity_runner.py`

- [ ] **Step 5: Commit**

```bash
git add src/analyze/popularity_runner.py tests/unit/test_popularity_lazy.py
git commit -m "feat(popularity): lazy cache-first per-artist fetch (TTL + graceful fallback)"
```

---

### Task 2: `load_artist_popularity_values` (bundle → per-track popularity vector)

**Files:**
- Modify: `src/analyze/popularity_runner.py`
- Test: `tests/unit/test_popularity_lazy.py`

**Interfaces:**
- Consumes: `get_artist_top_tracks_cached_or_fetch`, `resolve_top_tracks_to_popularity`, `src.playlist.artist_style._artist_indices_in_bundle`.
- Produces: `load_artist_popularity_values(bundle, artist_name, *, client, db_path, limit, max_age_days, now_iso, include_collaborations=False) -> Optional[np.ndarray]` — a vector aligned to `bundle.track_ids`, popularity in [0,1] for the seed artist's matched tracks, `NaN` elsewhere. Returns `None` if no client (popularity inert). Builds the artist's `local_tracks` from `bundle.track_ids` + `bundle.track_titles` (mbid `""`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_popularity_lazy.py`:

```python
import numpy as np
import types
from src.analyze.popularity_runner import load_artist_popularity_values, init_top_tracks_cache, upsert_artist_top_tracks


def _bundle(track_ids, titles, artist_keys):
    return types.SimpleNamespace(
        track_ids=np.array(track_ids, dtype=object),
        track_titles=np.array(titles, dtype=object),
        artist_keys=np.array(artist_keys, dtype=object),
        track_artists=None,
    )


def test_load_artist_popularity_aligns_seed_artist(tmp_path):
    b = _bundle(
        ["n_studio", "n_live", "other"],
        ["In Bloom", "In Bloom (Live)", "Some Song"],
        ["nirvana", "nirvana", "other"],
    )
    db = str(tmp_path / "e.db"); init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    client = MagicMock()
    vec = load_artist_popularity_values(
        b, "Nirvana", client=client, db_path=db, limit=50, max_age_days=30,
        now_iso="2026-06-24T00:00:00+00:00")
    assert vec is not None and vec.shape == (3,)
    assert vec[0] == 1.0            # studio In Bloom got the hit popularity
    assert np.isnan(vec[1])         # live lost
    assert np.isnan(vec[2])         # other artist untouched
    client.get_artist_top_tracks.assert_not_called()   # used the fresh cache


def test_load_artist_popularity_none_without_client(tmp_path):
    b = _bundle(["a"], ["T"], ["x"])
    assert load_artist_popularity_values(
        b, "X", client=None, db_path=str(tmp_path/"e.db"),
        limit=50, max_age_days=30, now_iso="2026-06-24T00:00:00+00:00") is None
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest -q tests/unit/test_popularity_lazy.py -k load_artist_popularity`
Expected: FAIL — `ImportError: cannot import name 'load_artist_popularity_values'`

- [ ] **Step 3: Implement**

Append to `src/analyze/popularity_runner.py`:

```python
def load_artist_popularity_values(
    bundle, artist_name: str, *, client, db_path: str, limit: int,
    max_age_days: int, now_iso: str, include_collaborations: bool = False,
) -> Optional[np.ndarray]:
    """Per-track popularity for the seed artist, aligned to bundle.track_ids.

    Lazy cache-first fetch of the seed artist's top tracks + resolve to the
    artist's local tracks (title-based; mbid blank from the bundle). None if no
    client. NaN for non-matched / other-artist tracks (neutral)."""
    if client is None:
        return None
    from src.playlist.artist_style import _artist_indices_in_bundle
    from src.string_utils import normalize_artist_key

    indices = _artist_indices_in_bundle(
        bundle, artist_name, include_collaborations=include_collaborations)
    if not indices:
        return None
    titles = getattr(bundle, "track_titles", None)
    local_tracks = [{
        "track_id": str(bundle.track_ids[i]),
        "title": str(titles[i]) if titles is not None else "",
        "musicbrainz_id": "",
    } for i in indices]
    artist_key = normalize_artist_key(artist_name)
    top = get_artist_top_tracks_cached_or_fetch(
        artist_key, artist_name, client=client, db_path=db_path,
        limit=limit, max_age_days=max_age_days, now_iso=now_iso)
    pop = resolve_top_tracks_to_popularity(top, local_tracks)
    if not pop:
        return None
    out = np.full(len(bundle.track_ids), np.nan, dtype=float)
    pos = {str(t): i for i, t in enumerate(bundle.track_ids)}
    for tid, score in pop.items():
        j = pos.get(tid)
        if j is not None:
            out[j] = score
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest -q tests/unit/test_popularity_lazy.py` (expect 6 passed) + `ruff check src/analyze/popularity_runner.py`

- [ ] **Step 5: Commit**

```bash
git add src/analyze/popularity_runner.py tests/unit/test_popularity_lazy.py
git commit -m "feat(popularity): load_artist_popularity_values (lazy seed-artist vector)"
```

---

### Task 3: `w_pop` term in the medoid score

**Files:**
- Modify: `src/playlist/artist_style.py` (config field; `_medoids_for_cluster`; `cluster_artist_tracks`)
- Modify: `src/playlist_generator.py` (both `ArtistStyleConfig` sites)
- Modify: `config.example.yaml`
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Produces: `ArtistStyleConfig.medoid_popularity_weight: float = 0.0`; `_medoids_for_cluster(..., popularity_weight=0.0, popularity_values=None)` (additive `scores += popularity_values[indices]·popularity_weight`, NaN→0); `cluster_artist_tracks(..., popularity_values=None)` passes per-cluster slices. Energy-wins invariant: callers keep `medoid_energy_weight > medoid_popularity_weight`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artist_style.py`:

```python
def test_medoid_popularity_term_breaks_tie_toward_popular():
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]; centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3)
    pop = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3, 0.0, None,
        5.0, np.array([0.0, 1.0, 0.0]))   # popularity_weight, popularity_values
    assert pop == [1]                       # strong popularity on index 1 wins the pick
    del base                                # baseline computed only to mirror the call shape


def test_medoid_popularity_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]; centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3)
    z = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3, 0.0, None,
        0.0, np.array([1.0, 0.0, 0.0]))   # popularity weight 0 -> ignored
    assert z == base
```

(`_centroid_for` already exists in this file from the energy tests.)

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest -q tests/test_artist_style.py -k medoid_popularity`
Expected: FAIL — `_medoids_for_cluster() takes ... positional arguments but ... were given`

- [ ] **Step 3: Add config field + extend `_medoids_for_cluster`**

In `ArtistStyleConfig` (after `dedupe_versions`):

```python
    # Popularity (Last.fm) bias on the within-slot medoid pick. Activated by the
    # "Popular Seeds" checkbox (overrides this to popular_seeds_weight). Keep
    # below medoid_energy_weight so energy-spread keeps the slot structure.
    medoid_popularity_weight: float = 0.0
```

In `_medoids_for_cluster`, add two trailing params after `energy_proximity`:

```python
    popularity_weight: float = 0.0,
    popularity_values: Optional[np.ndarray] = None,
```

and after the energy block (right after `scores = scores + prox * energy_weight` logic), add:

```python
    # Popularity bias: prefer the recognizable hit WITHIN this cluster's slot.
    if popularity_values is not None and popularity_weight > 0:
        pv = np.asarray(popularity_values, dtype=float)
        if pv.shape[0] == len(indices):
            pv = np.where(np.isfinite(pv), pv, 0.0)   # unknown -> neutral, no bonus
            scores = scores + pv * popularity_weight
        else:
            logger.warning(
                "artist_style: popularity_values len %d != cluster size %d; skipping",
                pv.shape[0], len(indices))
```

- [ ] **Step 4: Thread through `cluster_artist_tracks`**

Add `popularity_values: Optional[np.ndarray] = None` to the `cluster_artist_tracks` signature (after `energy_values`). In the medoid loop, alongside `energy_prox`, compute the per-cluster popularity slice and pass it:

```python
        pop_slice = None
        if popularity_values is not None and cfg.medoid_popularity_weight > 0:
            pop_slice = np.asarray(popularity_values, dtype=float)[members_local]
        medoid_list = _medoids_for_cluster(
            X_norm, members_local, centroids[c], track_ids,
            medoid_top_k, rng, medoid_top_k,
            artist_duration_stats, bundle.durations_ms,
            cfg.medoid_similarity_weight, cfg.medoid_duration_weight,
            cfg.medoid_energy_weight, energy_prox,
            cfg.medoid_popularity_weight, pop_slice,
        )
```

- [ ] **Step 5: Wire both generator construction sites + config**

In `src/playlist_generator.py`, both `ArtistStyleConfig(...)` sites (after `dedupe_versions=...`):

```python
            medoid_popularity_weight=float(style_cfg_raw.get("medoid_popularity_weight", 0.0)),
```

In `config.example.yaml` under `playlists: ds_pipeline: artist_style:`:

```yaml
      medoid_popularity_weight: 0.0    # within-slot bias toward Last.fm-popular tracks; keep < medoid_energy_weight
      popular_seeds_weight: 0.5        # weight applied when the "Popular Seeds" checkbox is ON
      popularity_max_age_days: 30      # refetch an artist's top tracks if the cache entry is older
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest -q tests/test_artist_style.py -k "popularity or energy or cluster_artist"` (expect pass, incl. regression) then the whole file `python -m pytest -q tests/test_artist_style.py` + `ruff check src/playlist/artist_style.py` + `mypy src/playlist/artist_style.py`.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/artist_style.py src/playlist_generator.py config.example.yaml tests/test_artist_style.py
git commit -m "feat(artist-style): w_pop medoid term + popularity_values plumbing (default off)"
```

---

### Task 4: Wire "Popular Seeds" + "New Seeds" through the request stack

**Files:**
- Modify: `src/playlist_web/schemas.py`, `src/playlist/request_models.py`, `src/playlist_gui/worker.py`, `src/playlist_generator.py`
- Test: `tests/unit/test_request_models.py` (or wherever request-model tests live; create if absent)

**Interfaces:**
- Produces: `GenerateRequestBody.popular_seeds: bool = False`, `.seed_epoch: int = 0`; same on `GeneratePlaylistRequest` (+ `from_worker_args`/`to_worker_args`); `create_playlist_for_artist(..., popular_seeds: bool = False)`. When `popular_seeds` is True, the orchestrator sets the effective `medoid_popularity_weight = popular_seeds_weight`, loads `load_artist_popularity_values(...)`, and injects it into `cluster_artist_tracks`. `seed_epoch` is already a `create_playlist_for_artist` param — the worker just passes it.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_request_models.py` (add/extend):

```python
from src.playlist.request_models import GeneratePlaylistRequest


def test_popular_seeds_and_seed_epoch_roundtrip_worker_args():
    req = GeneratePlaylistRequest(mode="artist", artist="Nirvana", popular_seeds=True, seed_epoch=3)
    args = req.to_worker_args()
    assert args.get("popular_seeds") is True and args.get("seed_epoch") == 3
    back = GeneratePlaylistRequest.from_worker_args(args)
    assert back.popular_seeds is True and back.seed_epoch == 3
    # defaults sparse
    base = GeneratePlaylistRequest(mode="artist", artist="X")
    a2 = base.to_worker_args()
    assert "popular_seeds" not in a2 and "seed_epoch" not in a2
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest -q tests/unit/test_request_models.py -k popular_seeds`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'popular_seeds'`

- [ ] **Step 3: Add the fields**

`src/playlist/request_models.py` — add to `GeneratePlaylistRequest`:

```python
    popular_seeds: bool = False
    seed_epoch: int = 0
```

In `from_worker_args(...)` return: `popular_seeds=bool(args.get("popular_seeds", False)), seed_epoch=int(args.get("seed_epoch", 0)),`

In `to_worker_args(...)` (sparse): `if self.popular_seeds: args["popular_seeds"] = True` and `if self.seed_epoch: args["seed_epoch"] = int(self.seed_epoch)`

`src/playlist_web/schemas.py` — add to `GenerateRequestBody`: `popular_seeds: bool = False` and `seed_epoch: int = 0`; pass both through `to_request()`.

- [ ] **Step 4: Worker passes them to the generator**

`src/playlist_gui/worker.py` `handle_generate_playlist`, the `create_playlist_for_artist(...)` call — add:

```python
                popular_seeds=request.popular_seeds,
                seed_epoch=request.seed_epoch,
```

- [ ] **Step 5: Orchestrator activates popularity**

`src/playlist_generator.py` `create_playlist_for_artist` — add `popular_seeds: bool = False` to the signature. Where the `ArtistStyleConfig` is built (site 1, ~1625) and BEFORE `cluster_artist_tracks` is called, when `popular_seeds` and a Last.fm client exists, override the weight and load the vector. Concretely, after `style_cfg = ArtistStyleConfig(...)`:

```python
        popularity_values = None
        if popular_seeds and getattr(self, "lastfm", None) is not None:
            from dataclasses import replace
            from datetime import datetime, timezone
            from src.analyze.popularity_runner import (
                enrichment_db_path, load_artist_popularity_values,
            )
            pop_w = float(style_cfg_raw.get("popular_seeds_weight", 0.5))
            style_cfg = replace(style_cfg, medoid_popularity_weight=pop_w)
            popularity_values = load_artist_popularity_values(
                bundle, artist_name, client=self.lastfm,
                db_path=enrichment_db_path(),
                limit=int((self.config.config.get("lastfm", {}) or {}).get("artist_top_tracks_limit", 50)),
                max_age_days=int(style_cfg_raw.get("popularity_max_age_days", 30)),
                now_iso=datetime.now(timezone.utc).isoformat(),
                include_collaborations=include_collaborations,
            )
```

Then pass `popularity_values=popularity_values` into the `cluster_artist_tracks(...)` call.

First add this DRY, ROOT-anchored path helper to `src/analyze/popularity_runner.py` (so the lazy generation path and the analyze stage agree on one absolute location — `src/analyze/popularity_runner.py` is at `<root>/src/analyze/`, so `parents[2]` is the repo root):

```python
def enrichment_db_path() -> str:
    """ROOT-anchored absolute path to the enrichment DB (the per-artist cache)."""
    return str(Path(__file__).resolve().parents[2] / "data" / "ai_genre_enrichment.db")
```

(`scripts/analyze_library.py::stage_popularity` already resolves to the same absolute `data/ai_genre_enrichment.db` via its `ENRICHMENT_DB_PATH` constant — same file, so the lazy cache and the batch cache are one and the same. Optionally switch the stage to import `enrichment_db_path()` too for a single source of truth; not required since the resolved path is identical.)

- [ ] **Step 6: Run tests + a fidelity check**

Run: `python -m pytest -q tests/unit/test_request_models.py` + the artist_style suite. Then a `playlist-testing`-skill multi-pier generation with `popular_seeds=True` on an artist (Last.fm mocked or a real key) confirming it runs and the picked piers shift toward popular tracks vs `popular_seeds=False`. Confirm `popular_seeds=False` is byte-identical to today.

- [ ] **Step 7: Commit**

```bash
git add src/playlist_web/schemas.py src/playlist/request_models.py src/playlist_gui/worker.py src/playlist_generator.py tests/unit/test_request_models.py
git commit -m "feat(popular-seeds): wire popular_seeds + seed_epoch request -> generator"
```

---

### Task 5: GUI — "Popular Seeds" checkbox + "New Seeds" button

**REQUIRED: use the `web-gui` skill first.** Traps: rebuild `web/dist` after edits, restart `serve_web.py`, verify the value reaches the worker (don't trust the UI).

**Files:**
- Modify: `web/src/lib/types.ts` (`GenerateRequestBody`)
- Modify: `web/src/components/GenerateControls.tsx` (state, checkbox, button, submit body)

**Interfaces:**
- Consumes: the API fields `popular_seeds`, `seed_epoch` (Task 4).
- Produces: a "Popular Seeds" checkbox (mirrors `includeCollabs`) and a "New Seeds" button that bumps a `seedEpoch` state and re-submits the same body.

- [ ] **Step 1: Add the TS fields**

`web/src/lib/types.ts`, in `GenerateRequestBody`:

```typescript
  popular_seeds?: boolean;
  seed_epoch?: number;
```

- [ ] **Step 2: Checkbox state + render (mirror includeCollabs)**

`GenerateControls.tsx`: near line 87,

```tsx
  const [popularSeeds, setPopularSeeds] = useLocalStorage("pg_popular_seeds", false);
  const [seedEpoch, setSeedEpoch] = useState(0);
```

Render a checkbox cell mirroring the `include collaborations` block (~line 285):

```tsx
            <Cell>
              <label className="flex items-center gap-1.5 cursor-pointer select-none"
                title="Bias the seed tracks toward this artist's most popular (Last.fm) songs.">
                <input type="checkbox" checked={popularSeeds}
                  onChange={(e) => setPopularSeeds(e.target.checked)}
                  className="accent-[#5eead4] cursor-pointer" />
                <Lbl>popular seeds</Lbl>
              </label>
            </Cell>
```

- [ ] **Step 3: "New Seeds" button (mirror Generate)**

Next to the Generate button (~line 303), only meaningful in artist mode:

```tsx
        {mode === "artist" && (
          <Cell>
            <button onClick={() => { setSeedEpoch((e) => e + 1); submit(seedEpoch + 1); }}
              disabled={busy}
              className="border border-[#5eead4] text-[#5eead4] text-[11px] px-3 py-[4px] rounded disabled:opacity-50 whitespace-nowrap"
              title="Re-roll: same settings, fresh seed tracks.">
              ↻ New Seeds
            </button>
          </Cell>
        )}
```

- [ ] **Step 4: Include in the submit body**

Change `submit()` to accept an optional epoch and include both fields:

```tsx
  function submit(epoch: number = seedEpoch) {
    const body: GenerateRequestBody = {
      ...,                         // existing fields
      include_collaborations: includeCollabs,
      popular_seeds: popularSeeds,
      seed_epoch: epoch,
    };
    onSubmit(body);
  }
```

(Update the Generate button's `onClick={submit}` → `onClick={() => submit()}`.)

- [ ] **Step 5: Build + verify end-to-end (web-gui skill)**

```bash
cd web && npm run build    # rebuild dist — REQUIRED or the GUI serves stale JS
```
Restart `python tools/serve_web.py`. In the browser: check "popular seeds", generate an artist playlist, and confirm via the worker/generation log that `popular_seeds=True` reached generation and the piers shifted toward hits. Click "New Seeds" and confirm a different-but-still-good set (the `seed_epoch` bumped). Per the web-gui skill, verify the value arrives at the worker — don't trust the checkbox alone.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/types.ts web/src/components/GenerateControls.tsx
git commit -m "feat(web): Popular Seeds checkbox + New Seeds reroll button"
```

---

## Final verification

- [ ] Full focused suite (fresh basetemp): the new `tests/unit/test_popularity_lazy.py`, `tests/unit/test_request_models.py`, `tests/test_artist_style.py`.
- [ ] `ruff check` + `mypy` on `src/analyze/popularity_runner.py`, `src/playlist/artist_style.py`.
- [ ] Full `python -m pytest -q -m "not slow"` (fresh `--basetemp`) — no regressions from the request-model/generator changes.
- [ ] Opt-in invariant by eye: `medoid_popularity_weight: 0.0` + checkbox OFF ⇒ `popularity_values` never loaded, `_medoids_for_cluster` gets `popularity_weight=0` ⇒ identical to today.
- [ ] End-to-end (web-gui): real generation with Popular Seeds ON vs OFF; eyeball that an artist's piers become recognizable hits and "New Seeds" rerolls.

## Calibration follow-up (NOT this plan)

`popular_seeds_weight` ships at `0.5` and `medoid_popularity_weight` at `0.0` (checkbox-gated). After the GUI works, tune `popular_seeds_weight` relative to `medoid_energy_weight` (energy must stay larger so spread holds) on the artist panel, and confirm popularity cleans up the energy-pulls-in-a-rehearsal residue. The future "Oops, All Bangers" mode (whole-pool popularity via the eager sidecar) is its own later plan.
