# "Oops, All Bangers" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-stop popularity control (`OFF → ON → OOPS, ALL BANGERS`) that biases a playlist's bridge tracks toward each artist's Last.fm hits, via a graded soft penalty in the pier-bridge beam.

**Architecture:** Consumption-only on the shipped popularity substrate (cache + resolver + loaders). The orchestrator does a one-shot cache-first pool-scan to build a bundle-aligned popularity vector, passes it + a strength into `build_pier_bridge_playlist`, and the beam multiplies each candidate's score by `1 − strength·(1−popularity)` (NaN→max demotion). Default OFF is byte-identical.

**Tech Stack:** Python 3.11, numpy, SQLite (cache in `ai_genre_enrichment.db`), existing `src/analyze/popularity_runner.py`, `src/playlist/pier_bridge/beam.py`, FastAPI + React/TS GUI.

## Global Constraints

- **Default OFF = byte-identical to today.** Guarded by BOTH a mode/strength check (`strength == 0.0`) AND a `popularity_values is None` check. No scan, no penalty when OFF.
- **Soft, never a hard gate.** The penalty only lowers `combined_score`; it never excludes a candidate and never changes the candidate pool / beam universe.
- **Never fail / never exceed ~90s because of this.** Pool-scan is cache-first; a fetch failure yields stale/neutral popularity, never an exception. Respects any active generation deadline.
- **Penalty form (exact):** per candidate bundle index `cand`, with `p = popularity_values[int(cand)]`:
  `d = 1.0 - (p if p == p else 0.0)` (the `p == p` test is the NaN check → NaN gives `d = 1.0`);
  `combined_score *= (1.0 - popularity_penalty_strength * d)`. Mirrors the genre penalty at `beam.py:1368` / `:1485`, but graded in `p`.
- **`metadata.db` read-only.** Only the Last.fm cache table is written (on a miss), by existing code.
- **Strengths are calibration placeholders** (`ON=0.10`, `OOPS=0.30`); ship them as config defaults, calibrate later against a diverse artist panel (legacy/active, niche/popular) per the spec.

## File Structure

- `src/analyze/popularity_runner.py` — **add** `load_pool_popularity_values` (multi-artist generalization of `load_artist_popularity_values`).
- `src/playlist/pier_bridge/config.py` — **add** `popularity_penalty_strength: float = 0.0` to `PierBridgeConfig`.
- `src/playlist/pier_bridge/beam.py` — **add** `popularity_values` kwarg to `_beam_search_segment`; apply the graded penalty at the two genre-penalty sites.
- `src/playlist/pier_bridge_builder.py` — thread `popularity_values` from `build_pier_bridge_playlist` into the `_beam_search_segment` call (`:1476`).
- `src/playlist/pier_bridge/micro_pier.py` — thread the same kwarg into its `_beam_search_segment` calls (`:270`, `:305`).
- `src/playlist/request_models.py`, `src/playlist_web/schemas.py`, `src/playlist_gui/worker.py` — thread `popularity_mode: str = "off"` (mirror the shipped `popular_seeds` plumbing).
- `src/playlist_generator.py` (+ the shared bridge-invoking path) — resolve `popularity_mode → strength`, run the pool-scan, pass `popularity_values` + strength down.
- `web/src/lib/types.ts`, `web/src/components/GenerateControls.tsx` — three-way segmented control.
- Tests under `tests/unit/` and the existing `scripts/research/slider_differentiation_eval.py` for calibration.

---

### Task 1: `load_pool_popularity_values` — bundle-aligned popularity for a whole pool

**Files:**
- Modify: `src/analyze/popularity_runner.py`
- Test: `tests/unit/test_popularity_pool_loader.py`

**Interfaces:**
- Consumes: existing `get_artist_top_tracks_cached_or_fetch`, `resolve_top_tracks_to_rank`, `normalize_artist_key`; `bundle.track_ids`, `bundle.track_titles`, `bundle.artist_keys`.
- Produces: `load_pool_popularity_values(bundle, artist_name_by_key: dict[str,str], *, client, db_path, limit=50, max_age_days=30, now_iso=None) -> Optional[np.ndarray]` — a vector aligned to `bundle.track_ids` (NaN where unknown), `score = 1 − rank/N` per artist. Returns `None` if `client is None`. Never raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_popularity_pool_loader.py
import numpy as np, types
from unittest.mock import MagicMock
from src.analyze.popularity_runner import load_pool_popularity_values, init_top_tracks_cache, upsert_artist_top_tracks


def _bundle(ids, titles, keys):
    return types.SimpleNamespace(
        track_ids=np.array(ids, dtype=object),
        track_titles=np.array(titles, dtype=object),
        artist_keys=np.array(keys, dtype=object),
    )


def test_pool_loader_scores_multiple_artists_aligned(tmp_path):
    db = str(tmp_path / "e.db")
    init_top_tracks_cache(db)
    upsert_artist_top_tracks(db, "nirvana", "2026-06-25T00:00:00+00:00",
                             [{"name": "In Bloom", "mbid": "", "rank": 0}])
    upsert_artist_top_tracks(db, "the smiths", "2026-06-25T00:00:00+00:00",
                             [{"name": "This Charming Man", "mbid": "", "rank": 0}])
    b = _bundle(
        ["n_hit", "n_deep", "s_hit", "other"],
        ["In Bloom", "Endless Nameless", "This Charming Man", "Song"],
        ["nirvana", "nirvana", "the smiths", "unknownband"],
    )
    client = MagicMock()  # everything cached -> no fetch
    vec = load_pool_popularity_values(
        b, {"nirvana": "Nirvana", "the smiths": "The Smiths", "unknownband": "Unknown"},
        client=client, db_path=db, now_iso="2026-06-25T00:00:00+00:00")
    assert vec.shape == (4,)
    assert vec[0] == 1.0          # Nirvana hit
    assert np.isnan(vec[1])       # Nirvana deep cut, not on top list
    assert vec[2] == 1.0          # Smiths hit (different artist, aligned)
    assert np.isnan(vec[3])       # unknown artist, no cache
    client.get_artist_top_tracks.assert_not_called()


def test_pool_loader_none_without_client(tmp_path):
    b = _bundle(["a"], ["T"], ["x"])
    assert load_pool_popularity_values(b, {"x": "X"}, client=None, db_path=str(tmp_path / "e.db")) is None
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest -q --basetemp=<scratch> tests/unit/test_popularity_pool_loader.py` → FAIL (`load_pool_popularity_values` undefined).

- [ ] **Step 3: Implement.** Reuse the per-artist machinery already in `load_artist_popularity_values` — group bundle indices by `artist_key`, and for each pool artist do the cache-first fetch + resolve, writing into one shared vector.

```python
def load_pool_popularity_values(
    bundle,
    artist_name_by_key,          # dict[str, str]: normalized artist_key -> display name
    *,
    client,
    db_path: str,
    limit: int = 50,
    max_age_days: int = 30,
    now_iso: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Per-track popularity (1 - rank/N) for every artist in the candidate pool,
    aligned to bundle.track_ids. NaN where unknown. Cache-first + TTL; never raises."""
    if client is None:
        return None
    track_ids = bundle.track_ids
    titles = getattr(bundle, "track_titles", None)
    keys = getattr(bundle, "artist_keys", None)
    if keys is None:
        return None
    out = np.full(len(track_ids), np.nan, dtype=np.float32)
    # group bundle rows by artist_key, restricted to the requested pool artists
    rows_by_key: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        k = str(k)
        if k in artist_name_by_key:
            rows_by_key.setdefault(k, []).append(i)
    for key, idxs in rows_by_key.items():
        try:
            top = get_artist_top_tracks_cached_or_fetch(
                key, artist_name_by_key[key], client=client, db_path=db_path,
                limit=limit, max_age_days=max_age_days, now_iso=now_iso)
        except Exception:        # never gate generation
            top = []
        if not top:
            continue
        local = [{"track_id": str(track_ids[i]),
                  "title": str(titles[i]) if titles is not None else "",
                  "musicbrainz_id": ""} for i in idxs]
        ranks = resolve_top_tracks_to_rank(top, local)   # track_id -> 0-based rank
        n = len(top)
        pos = {str(track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            j = pos.get(tid)
            if j is not None:
                out[j] = 1.0 - rank / n
    return out
```

- [ ] **Step 4: Run tests** → PASS. Then `ruff check src/analyze/popularity_runner.py tests/unit/test_popularity_pool_loader.py` and `mypy src/analyze/popularity_runner.py`.

- [ ] **Step 5: Commit** — `git commit -m "feat(popularity): load_pool_popularity_values — bundle-aligned multi-artist popularity"`

---

### Task 2: `PierBridgeConfig.popularity_penalty_strength`

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (the `genre_penalty_threshold`/`genre_penalty_strength` block, ~line 91)
- Test: `tests/unit/test_pier_bridge_config.py` (or wherever PierBridgeConfig defaults are asserted; create if absent)

**Interfaces:**
- Produces: `PierBridgeConfig.popularity_penalty_strength: float = 0.0`. Default 0.0 ⇒ inert. Flows through the relaxation cascade automatically (the cascade builds attempt configs via `replace(cfg, ...)`).

- [ ] **Step 1: Failing test** — assert `PierBridgeConfig().popularity_penalty_strength == 0.0` and that `dataclasses.replace(cfg, popularity_penalty_strength=0.3).popularity_penalty_strength == 0.3`.
- [ ] **Step 2: Run → fail** (attribute missing).
- [ ] **Step 3: Add the field** beside the genre-penalty fields:
```python
    genre_penalty_threshold: float = 0.20
    genre_penalty_strength: float = 0.10
    # Oops, All Bangers: graded popularity demotion in the beam (0.0 = off / today).
    popularity_penalty_strength: float = 0.0
```
- [ ] **Step 4: Run → pass.** `ruff` + `mypy` on `config.py`.
- [ ] **Step 5: Commit** — `git commit -m "feat(pier-bridge): popularity_penalty_strength config field (default 0 = off)"`

---

### Task 3: Beam penalty + threading

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (signature ~`:216`; entry-extraction ~`:277`; penalty sites `:1368–1371` and `:1485–1494`)
- Modify: `src/playlist/pier_bridge_builder.py` (`build_pier_bridge_playlist` → `_beam_search_segment` call at `:1476`)
- Modify: `src/playlist/pier_bridge/micro_pier.py` (calls at `:270`, `:305`)
- Test: `tests/unit/test_beam_popularity_penalty.py`

**Interfaces:**
- Consumes: the `popularity_values` vector from Task 1; `cfg.popularity_penalty_strength` from Task 2.
- Produces: `_beam_search_segment(..., *, popularity_values: Optional[np.ndarray] = None, ...)`. With `popularity_penalty_strength == 0.0` OR `popularity_values is None`, scoring is unchanged.

- [ ] **Step 1: Failing unit test** for the penalty math (a tiny helper or a focused beam run). Prefer extracting the math into a pure helper so it's directly testable:
```python
# in beam.py, near _compute_duration_penalty:
def _popularity_factor(p: float, strength: float) -> float:
    """Graded multiplicative demotion: NaN -> max. Returns a factor in (0, 1]."""
    if strength <= 0.0:
        return 1.0
    d = 1.0 - (p if p == p else 0.0)   # p == p is False for NaN
    return 1.0 - strength * d
```
```python
# tests/unit/test_beam_popularity_penalty.py
import math
from src.playlist.pier_bridge.beam import _popularity_factor

def test_popularity_factor_grades_and_handles_nan():
    assert _popularity_factor(1.0, 0.3) == 1.0          # banger: no demotion
    assert _popularity_factor(0.0, 0.3) == 0.7          # bottom of chart: full strength
    assert _popularity_factor(float("nan"), 0.3) == 0.7 # unknown -> max (ruthless)
    assert abs(_popularity_factor(0.5, 0.2) - 0.9) < 1e-9
    assert _popularity_factor(0.2, 0.0) == 1.0          # strength 0 -> inert
```
- [ ] **Step 2: Run → fail** (`_popularity_factor` undefined).
- [ ] **Step 3: Implement.**
  1. Add `_popularity_factor` (above).
  2. Add the kwarg to `_beam_search_segment` after `durations_ms`: `popularity_values: Optional[np.ndarray] = None,`.
  3. At entry (near `:277`): `popularity_penalty_strength = float(getattr(cfg, "popularity_penalty_strength", 0.0))`.
  4. At BOTH genre-penalty sites (`:1368` path A and `:1485` path B), immediately after the genre-penalty block, add:
```python
     if popularity_penalty_strength > 0.0 and popularity_values is not None:
         combined_score *= _popularity_factor(float(popularity_values[int(cand)]), popularity_penalty_strength)
```
  5. In `pier_bridge_builder.py` at the `_beam_search_segment` call (`:1476`), pass `popularity_values=popularity_values,` (the array threaded into `build_pier_bridge_playlist` — see Task 5). Add `popularity_values: Optional[np.ndarray] = None` to `build_pier_bridge_playlist`'s signature.
  6. In `micro_pier.py` (`:270`, `:305`), pass `popularity_values=popularity_values,` (thread it into the micro-pier helper's signature too).
- [ ] **Step 4: Run → pass.** Add a focused integration assertion (optional): a small multi-pier beam run with a crafted `popularity_values` prefers the higher-scored candidate. `ruff` + `mypy` on the three files.
- [ ] **Step 5: Commit** — `git commit -m "feat(beam): graded popularity penalty on bridge candidates (inert at strength 0)"`

---

### Task 4: Request / schema / worker plumbing for `popularity_mode`

**Files (mirror the shipped `popular_seeds` / `seed_epoch` threading exactly):**
- Modify: `src/playlist/request_models.py` — add `popularity_mode: str = "off"` to `GeneratePlaylistRequest`; include in `from_worker_args` / `to_worker_args` (sparse, like `seed_epoch`).
- Modify: `src/playlist_web/schemas.py` — add `popularity_mode: str = "off"` to `GenerateRequestBody` and `to_request()`.
- Modify: `src/playlist_gui/worker.py` — pass `popularity_mode=request.popularity_mode` into the generation call(s) (the same dispatch block that already passes `popular_seeds`).
- Test: extend `tests/unit/test_request_models.py` with a round-trip for `popularity_mode`.

**Interfaces:**
- Produces: `popularity_mode` carried end-to-end; values `"off" | "on" | "oops"`; default `"off"`.

- [ ] **Step 1: Failing round-trip test** — `to_worker_args`/`from_worker_args` preserves `popularity_mode="oops"`, and defaults to `"off"` when absent.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Add the field** at each site, copying the `popular_seeds` lines verbatim and renaming (bool → str, default `"off"`). Validate against `{"off","on","oops"}` where `popular_seeds` was validated, defaulting unknown values to `"off"`.
- [ ] **Step 4: Run → pass.** `ruff` + `mypy` on the four files.
- [ ] **Step 5: Commit** — `git commit -m "feat(api): popularity_mode request/schema/worker plumbing (default off)"`

---

### Task 5: Orchestrator activation — pool-scan + wire into bridge building

**Files:**
- Modify: `src/playlist_generator.py` — at the bridge-building invocation path (where `build_pier_bridge_playlist` is reached for artist / seeds / genre modes). Resolve `popularity_mode → strength`, run the pool-scan, pass `popularity_values` + set `cfg.popularity_penalty_strength`.
- Possibly modify: the pipeline layer that calls `build_pier_bridge_playlist`, to accept + forward `popularity_values`.
- Test: `tests/unit/test_all_bangers_activation.py` (opt-in invariant + activation).

**Interfaces:**
- Consumes: `load_pool_popularity_values` (Task 1), `PierBridgeConfig.popularity_penalty_strength` (Task 2), the threaded `popularity_values` param (Task 3), `popularity_mode` (Task 4), `self.lastfm`, `enrichment_db_path()`.
- Produces: when `popularity_mode in {"on","oops"}` AND `self.lastfm`: a bundle-aligned `popularity_values` for the candidate pool's artists, and `cfg.popularity_penalty_strength` set to the resolved strength, both flowing into the beam. OFF ⇒ `None` + 0.0.

**Decision — strength map** (module constant, calibration placeholders):
```python
_POPULARITY_STRENGTH = {"off": 0.0, "on": 0.10, "oops": 0.30}
```

- [ ] **Step 1: Failing tests.**
  - `popularity_mode="off"` ⇒ `popularity_values` is None, strength 0.0, and a generation is byte-identical to baseline (assert the same track list as a no-popularity run on a fixed seed).
  - `popularity_mode="oops"` with a stubbed `self.lastfm` + warm cache ⇒ `load_pool_popularity_values` is invoked with the pool's artists and the resolved strength is `0.30`.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** at the bridge-building site:
```python
popularity_values = None
strength = _POPULARITY_STRENGTH.get(popularity_mode, 0.0)
if strength > 0.0 and getattr(self, "lastfm", None) is not None:
    from src.analyze.popularity_runner import enrichment_db_path, load_pool_popularity_values
    # distinct pool artists -> display names (key -> name), from the candidate pool tracks
    name_by_key = {str(k): str(nm) for k, nm in _pool_artist_names(bundle, candidate_pool_indices)}
    popularity_values = load_pool_popularity_values(
        bundle, name_by_key, client=self.lastfm, db_path=enrichment_db_path())
    cfg = replace(cfg, popularity_penalty_strength=strength)
# ... pass popularity_values into build_pier_bridge_playlist(...)
```
  `_pool_artist_names` derives `{artist_key: display_name}` for the **admitted candidate pool** indices (use `bundle.artist_keys` + `bundle.track_artists`, or look up names once; restrict to the pool's distinct keys — the bounded universe, per the spec). If the orchestrator can't see the lastfm client at this layer, thread it the same way other generator services reach the pipeline.
- [ ] **Step 4: Run → pass.** Then exercise the real path: a multi-pier generation at `off` vs `oops` with the warm cache, confirm `off` is unchanged and `oops` shifts bridges toward higher-popularity tracks (read the generation log). `ruff` + `mypy`.
- [ ] **Step 5: Commit** — `git commit -m "feat(all-bangers): orchestrator pool-scan + beam activation (off = byte-identical)"`

---

### Task 6: GUI three-way control

**Files:**
- Modify: `web/src/lib/types.ts` — add `popularity_mode?: "off" | "on" | "oops"` to `GenerateRequestBody`.
- Modify: `web/src/components/GenerateControls.tsx` — a three-way segmented control (Off / On / Oops, All Bangers), `useLocalStorage("pg_popularity_mode", "off")`, shown in **all** modes, sent in `submit()`.
- Test: `web/tests/` Playwright (optional) or rely on the build + manual e2e.

**Interfaces:**
- Consumes: `GenerateRequestBody.popularity_mode` (Task 4 backend).
- Produces: the control sends `popularity_mode` on generate; default `"off"`.

- [ ] **Step 1:** Add the type field.
- [ ] **Step 2:** Add `const [popularityMode, setPopularityMode] = useLocalStorage<"off"|"on"|"oops">("pg_popularity_mode", "off")`; render a 3-button segmented control; include `popularity_mode: popularityMode` in the `submit()` body (next to `popular_seeds`).
- [ ] **Step 3:** `cd web && npm run build` → 0 TS errors.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): Oops All Bangers three-way popularity control"`

---

## Final review & calibration

- Run the full not-slow suite (fresh `--basetemp` on Windows) — expect green; the 5 pre-existing `test_dense_genre_integration` failures are unrelated (stale dense genre sidecar).
- Dispatch the whole-branch code review (opus) per subagent-driven-development.
- **Calibration (post-merge, mandatory):** via `scripts/research/slider_differentiation_eval.py` across the diverse artist panel (legacy/active, niche/popular, high/low coverage). Tune `_POPULARITY_STRENGTH["on"|"oops"]`. Watch for the niche-artist over-demotion failure mode the spec flags.

## Self-review notes (author)

- Spec coverage: signal (T1/T3), soft graded penalty (T3), three-stop control (T2/T4/T5/T6), pool-scan/coverage (T1/T5), all-modes scope (T5 at the shared bridge path), OFF byte-identical (T5 invariant test), never-fail (T1 broad-except, T3 inert guards). ✔
- Type consistency: `popularity_mode: str` everywhere; `popularity_values: Optional[np.ndarray]`; `popularity_penalty_strength: float`. ✔
- Open implementation detail flagged in T5: exact layer where the lastfm client + pool indices co-exist (artist path has it like Popular Seeds; seeds/genre modes route through the pipeline — confirm the client is reachable there, else thread it).
