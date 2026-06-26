# Oops, All Bangers — Pool-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-architect "Oops, All Bangers" from a soft beam re-ranker into a **popularity admission gate** on the candidate pool, so the playlist is actually all bangers (cross-genre hits included), with a `sonic → pace → genre → popularity` relax-to-fill cascade and OOPS-implies-popular-seed-piers in artist mode.

**Architecture:** A new cache-only **rank** loader feeds a popularity **admission filter** added to `build_candidate_pool` (final eligibility step). The gate is activated by a new `PierBridgeConfig.popularity_rank_cutoff` resolved from `popularity_mode` in `create_playlist_for_artist`. `core.generate_playlist_ds` loads ranks once, threads them into the pool builder, and runs a **fixed-order relaxation cascade** (mirroring the existing One-Each relaxation loop) when the banger pool is too small. The policy layer loosens OOPS's sonic/pace baseline; the existing beam penalty stays as a secondary refinement.

**Tech Stack:** Python 3.11, numpy, dataclasses, pytest. SQLite cache (`ai_genre_enrichment.db`). No new dependencies.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-26-oops-all-bangers-poolgate-design.md` — this plan implements it. Read it first.
- **Branch / worktree:** work **inline** on branch `worktree-oops-bangers-poolgate` (worktree `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.claude\worktrees\oops-bangers-poolgate`). **Do NOT dispatch implementation subagents** — they launch in the MAIN checkout and leak commits to master. Read-only Explore subagents are fine.
- **Never** touch `data/metadata.db`, the MERT shards/sidecar, or audio files. `metadata.db` opens `?mode=ro` only.
- **Activate, don't legacy-default:** every knob ships as a **live default**, configurable for rollback. `off` mode stays byte-identical to today.
- **Never silently no-op a configured gate** — the missing-data path (uncached artist) excludes the track (treated as non-banger); that is intended and logged, not a silent failure.
- **Pytest discipline:** run directly, `-q`, bounded by the tool timeout — **never** pipe through `tail`/`head`. Always pass a fresh basetemp to dodge the Windows `pytest-of-Dylan WinError 5` trap:
  `python -m pytest <paths> -q --basetemp="C:/Users/Dylan/AppData/Local/Temp/claude/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3--claude-worktrees-oops-all-bangers/9557faa1-c0b6-4b3a-9fc0-25d9e1d30f43/scratchpad/pt"`
  (referred to below as `--basetemp=$PT`).
- **No real data in this worktree** → unit tests only. **Live generation verification is the user's** via the GUI (Task 9). Do not claim end-to-end success from unit tests.
- **Commit after every task.** End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do NOT push to origin.
- **reuse-first:** before adding any helper, check the reuse ladder (existing repo code → stdlib → installed dep). The pure helpers below are new because no equivalent exists; the rank loader deliberately mirrors `load_pool_popularity_values_cached`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/analyze/popularity_runner.py` | +`load_pool_popularity_ranks_cached` | cache-only per-track **rank** vector for the gate |
| `src/playlist/candidate_pool.py` | +gate kwargs, +`_apply_popularity_gate`, backstop guard | the admission gate (final eligibility filter) |
| `src/playlist/pier_bridge/config.py` | +`popularity_rank_cutoff` field | carries the resolved cutoff to core |
| `src/playlist_generator.py` | +`_resolve_popularity_rank_cutoff`, +`_resolve_popular_seeds`, wire both | resolve cutoff + force popular-seed piers (artist mode) |
| `src/playlist/pipeline/core.py` | load ranks, thread into `_build_pool`, +`_banger_relaxation_steps` + cascade loop | activate gate + relax-to-fill cascade |
| `src/playlist_gui/ui_state.py` | +`popularity_mode` field | carry mode to the policy layer |
| `src/playlist_web/app.py` | populate `ui.popularity_mode` | wire request → policy |
| `src/playlist_gui/policy.py` | OOPS sonic/pace baseline override | OOPS owns sonic/pace |
| `config.example.yaml` | `playlists.bangers.*` keys | tunable defaults |
| `tests/unit/goldens/pipeline/*.json` | regen (4 files) | absorb the new `PierBridgeConfig` field |

---

## Task 1: Popularity **rank** loader

**Files:**
- Modify: `src/analyze/popularity_runner.py` (add after `load_pool_popularity_values_cached`, ~line 360)
- Test: `tests/unit/test_popularity_pool_loader.py` (add to existing file)

**Interfaces:**
- Consumes: existing `get_artist_top_tracks_cached(db_path, key)`, `resolve_top_tracks_to_rank(top, local)`.
- Produces: `load_pool_popularity_ranks_cached(bundle, pool_indices, *, db_path: str) -> np.ndarray` — int array aligned to `bundle.track_ids`; value = 0-based Last.fm rank, `-1` where uncached / not in the artist's top-N.

- [ ] **Step 1: Write the failing test**

Look at the existing tests in `tests/unit/test_popularity_pool_loader.py` for how they build a fake bundle and monkeypatch the cache. Add:

```python
def test_load_pool_popularity_ranks_cached_returns_rank_not_score(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import src.analyze.popularity_runner as pr

    bundle = SimpleNamespace(
        track_ids=np.array(["t0", "t1", "t2", "t3"], dtype=object),
        track_titles=np.array(["Hit A", "Hit B", "Deep Cut", "Other"], dtype=object),
        artist_keys=np.array(["nirvana", "nirvana", "nirvana", "uncached"], dtype=object),
    )
    # nirvana top tracks: Hit A rank 0, Hit B rank 1 (Deep Cut absent)
    def fake_cached(db_path, key):
        if key == "nirvana":
            return [{"name": "Hit A", "rank": 0, "mbid": ""},
                    {"name": "Hit B", "rank": 1, "mbid": ""}]
        return []
    monkeypatch.setattr(pr, "get_artist_top_tracks_cached", fake_cached)

    ranks = pr.load_pool_popularity_ranks_cached(bundle, [0, 1, 2, 3], db_path=":memory:")
    assert ranks.tolist() == [0, 1, -1, -1]   # rank, rank, not-in-top-N, uncached
    assert ranks.dtype.kind == "i"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_popularity_pool_loader.py::test_load_pool_popularity_ranks_cached_returns_rank_not_score -q --basetemp=$PT`
Expected: FAIL with `AttributeError: ... has no attribute 'load_pool_popularity_ranks_cached'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/analyze/popularity_runner.py` (mirror `load_pool_popularity_values_cached` exactly; store the rank, not `1 - rank/n`):

```python
def load_pool_popularity_ranks_cached(
    bundle, pool_indices, *, db_path: str
) -> np.ndarray:
    """Cache-ONLY per-track Last.fm rank (0-based) for the given bundle pool indices.

    Sibling of load_pool_popularity_values_cached, but stores the rank itself — the
    popularity admission gate compares against a rank cutoff (top-10 / top-50), and
    the score 1 - rank/n is not a fixed-rank threshold (n varies per artist). Artists
    not in the warm cache and tracks not in the artist's top-N stay -1. Aligned to
    bundle.track_ids. Never raises."""
    track_ids = bundle.track_ids
    out = np.full(len(track_ids), -1, dtype=int)
    keys = getattr(bundle, "artist_keys", None)
    titles = getattr(bundle, "track_titles", None)
    if keys is None:
        return out
    rows_by_key: Dict[str, List[int]] = {}
    for i in pool_indices:
        i = int(i)
        rows_by_key.setdefault(str(keys[i]), []).append(i)
    for key, idxs in rows_by_key.items():
        try:
            top = get_artist_top_tracks_cached(db_path, key)
        except Exception:  # never gate generation
            top = []
        if not top:
            continue
        local = [{
            "track_id": str(track_ids[i]),
            "title": str(titles[i]) if titles is not None else "",
            "musicbrainz_id": "",
        } for i in idxs]
        ranks = resolve_top_tracks_to_rank(top, local)
        pos = {str(track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            j = pos.get(tid)
            if j is not None:
                out[j] = int(rank)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_popularity_pool_loader.py -q --basetemp=$PT`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/analyze/popularity_runner.py tests/unit/test_popularity_pool_loader.py
git commit -m "feat(bangers): cache-only popularity rank loader for the pool gate"
```

---

## Task 2: Popularity admission gate in `build_candidate_pool`

**Files:**
- Modify: `src/playlist/candidate_pool.py` (signature ~506; gate insertion after energy rescue ~1043; backstop guard ~1138)
- Test: `tests/unit/test_candidate_pool_popularity_gate.py` (new)

**Interfaces:**
- Consumes: rank vector from Task 1 (aligned to bundle indices).
- Produces:
  - new kwargs on `build_candidate_pool(...)`: `popularity_ranks: Optional[np.ndarray] = None`, `popularity_rank_cutoff: Optional[int] = None`.
  - module helper `_apply_popularity_gate(eligible: list[int], popularity_ranks: np.ndarray, rank_cutoff: int) -> tuple[list[int], int]` returning `(kept, excluded_count)`.

- [ ] **Step 1: Write the failing test (pure helper)**

Create `tests/unit/test_candidate_pool_popularity_gate.py`:

```python
import numpy as np
from src.playlist.candidate_pool import _apply_popularity_gate


def test_gate_keeps_only_ranks_below_cutoff():
    ranks = np.array([0, 4, 9, 10, 49, -1], dtype=int)
    eligible = [0, 1, 2, 3, 4, 5]
    kept, excluded = _apply_popularity_gate(eligible, ranks, rank_cutoff=10)
    assert kept == [0, 1, 2]          # ranks 0,4,9 < 10
    assert excluded == 3              # rank 10, rank 49, and -1 (uncached) dropped


def test_gate_excludes_uncached_minus_one():
    ranks = np.array([-1, -1], dtype=int)
    kept, excluded = _apply_popularity_gate([0, 1], ranks, rank_cutoff=50)
    assert kept == []
    assert excluded == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_candidate_pool_popularity_gate.py -q --basetemp=$PT`
Expected: FAIL with `ImportError: cannot import name '_apply_popularity_gate'`

- [ ] **Step 3: Write the helper**

Add near the other module-level helpers in `src/playlist/candidate_pool.py`:

```python
def _apply_popularity_gate(
    eligible: list[int],
    popularity_ranks: np.ndarray,
    rank_cutoff: int,
) -> tuple[list[int], int]:
    """Oops, All Bangers admission gate: keep only candidates whose 0-based Last.fm
    rank is in [0, rank_cutoff). -1 (uncached / not in the artist's top-N) and any
    rank >= cutoff are non-bangers and excluded. Returns (kept, excluded_count)."""
    kept = [i for i in eligible if 0 <= int(popularity_ranks[i]) < rank_cutoff]
    return kept, len(eligible) - len(kept)
```

- [ ] **Step 4: Add the kwargs + apply the gate in `build_candidate_pool`**

Add to the keyword-only signature (after `genre_graph_source: str = "legacy",` ~line 547):

```python
    popularity_ranks: Optional[np.ndarray] = None,
    popularity_rank_cutoff: Optional[int] = None,
```

Apply the gate as the **final eligibility filter — immediately after the energy-rescue block and before `grouped: Dict[str, list[int]] = {}`** (currently ~line 1094):

```python
    # ── Oops, All Bangers: popularity admission gate ─────────────────────────
    # Final eligibility filter so EVERY pooled track is a banger, including any
    # energy-rescued tracks. NaN/-1 (uncached / not in the artist's top-N) excluded.
    if popularity_ranks is not None and popularity_rank_cutoff is not None:
        _before_pop = len(eligible)
        eligible, _pop_excluded = _apply_popularity_gate(
            eligible, np.asarray(popularity_ranks), int(popularity_rank_cutoff)
        )
        logger.info(
            "Popularity gate applied: cutoff=top-%d before=%d after=%d excluded=%d",
            int(popularity_rank_cutoff), _before_pop, len(eligible), _pop_excluded,
        )
```

- [ ] **Step 5: Guard the never-starve backstop**

The `min_pool_size` backfill (~line 1138) pulls highest-sonic-sim candidates regardless of popularity. Constrain it so it never smuggles a non-banger in. In the `_ranked` comprehension (~line 1147), add a popularity predicate when the gate is active:

```python
        _gate_on = popularity_ranks is not None and popularity_rank_cutoff is not None
        _ranked = sorted(
            (i for i in range(len(track_ids))
             if i not in _already and i not in _seed_set
             and (not _gate_on or 0 <= int(popularity_ranks[i]) < int(popularity_rank_cutoff))),
            key=lambda i: float(sonic_seed_sim[i]) if sonic_seed_sim is not None else 0.0,
            reverse=True,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_candidate_pool_popularity_gate.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_candidate_pool_popularity_gate.py
git commit -m "feat(bangers): popularity admission gate in build_candidate_pool"
```

---

## Task 3: Resolve cutoff + force popular-seed piers (artist mode)

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (add `popularity_rank_cutoff` field to `PierBridgeConfig`)
- Modify: `src/playlist_generator.py` (helpers + wire in `create_playlist_for_artist`)
- Test: `tests/unit/test_bangers_resolve.py` (new); `tests/unit/test_pier_bridge_config_popularity.py` (extend)

**Interfaces:**
- Consumes: `popularity_mode` ("off"|"on"|"oops"), `playlists.bangers` config dict.
- Produces:
  - `PierBridgeConfig.popularity_rank_cutoff: Optional[int] = None`.
  - `_resolve_popularity_rank_cutoff(popularity_mode: str, bangers_cfg: dict) -> Optional[int]`.
  - `_resolve_popular_seeds(popular_seeds: bool, popularity_mode: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_bangers_resolve.py`:

```python
from src.playlist_generator import (
    _resolve_popularity_rank_cutoff,
    _resolve_popular_seeds,
)


def test_cutoff_off_is_none():
    assert _resolve_popularity_rank_cutoff("off", {}) is None


def test_cutoff_on_defaults_50_oops_defaults_10():
    assert _resolve_popularity_rank_cutoff("on", {}) == 50
    assert _resolve_popularity_rank_cutoff("oops", {}) == 10


def test_cutoff_reads_config_overrides():
    cfg = {"rank_cutoff_on": 40, "rank_cutoff_oops": 15}
    assert _resolve_popularity_rank_cutoff("on", cfg) == 40
    assert _resolve_popularity_rank_cutoff("oops", cfg) == 15


def test_popular_seeds_forced_only_by_oops():
    assert _resolve_popular_seeds(False, "oops") is True     # OOPS forces it
    assert _resolve_popular_seeds(False, "on") is False      # ON does not
    assert _resolve_popular_seeds(False, "off") is False
    assert _resolve_popular_seeds(True, "on") is True        # user's own choice respected
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_bangers_resolve.py -q --basetemp=$PT`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add the helpers**

Add at module scope in `src/playlist_generator.py` (near the top-level helpers, not inside the class):

```python
def _resolve_popularity_rank_cutoff(popularity_mode: str, bangers_cfg: dict) -> Optional[int]:
    """Oops, All Bangers admission-gate cutoff. off -> None (gate disabled);
    on -> rank_cutoff_on (default 50); oops -> rank_cutoff_oops (default 10)."""
    m = str(popularity_mode or "off").lower()
    if m == "on":
        return int((bangers_cfg or {}).get("rank_cutoff_on", 50))
    if m == "oops":
        return int((bangers_cfg or {}).get("rank_cutoff_oops", 10))
    return None


def _resolve_popular_seeds(popular_seeds: bool, popularity_mode: str) -> bool:
    """OOPS forces popular-seed pier selection (piers -> the artist's hits), so the
    whole playlist is bangers, not just the bridges. Artist-mode-only by construction:
    this is called on the artist-mode entry point; seed mode never reaches here."""
    return bool(popular_seeds) or str(popularity_mode or "off").lower() == "oops"
```

- [ ] **Step 4: Add the `PierBridgeConfig` field**

In `src/playlist/pier_bridge/config.py`, next to `popularity_penalty_strength` (~line 95), add:

```python
    # Oops, All Bangers admission gate: resolved per popularity_mode (off->None,
    # on->50, oops->10). None = gate disabled. Consumed by core.generate_playlist_ds.
    popularity_rank_cutoff: Optional[int] = None
```

- [ ] **Step 5: Wire both into `create_playlist_for_artist`**

(a) Force popular-seed piers — near the top of the method, right after the `popularity_mode`/`popular_seeds` params are in scope (after ~line 1305):

```python
        popular_seeds = _resolve_popular_seeds(popular_seeds, popularity_mode)
```

(b) Resolve the cutoff in the existing `_bangers_cfg` block (~line 1924) and pass it on the `PierBridgeConfig(...)` (~line 1931, next to `popularity_penalty_strength=_pop_strength,`):

```python
                _pop_rank_cutoff = _resolve_popularity_rank_cutoff(popularity_mode, _bangers_cfg)
```
```python
                    popularity_rank_cutoff=_pop_rank_cutoff,
```

- [ ] **Step 6: Extend the config test for the new field**

Add to `tests/unit/test_pier_bridge_config_popularity.py`:

```python
def test_pier_bridge_config_popularity_rank_cutoff_defaults_none():
    from src.playlist.pier_bridge.config import PierBridgeConfig
    cfg = PierBridgeConfig()
    assert cfg.popularity_rank_cutoff is None
```

(If `PierBridgeConfig()` requires args, mirror however the existing tests in that file construct it.)

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/unit/test_bangers_resolve.py tests/unit/test_pier_bridge_config_popularity.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/playlist_generator.py src/playlist/pier_bridge/config.py tests/unit/test_bangers_resolve.py tests/unit/test_pier_bridge_config_popularity.py
git commit -m "feat(bangers): resolve rank cutoff + force popular-seed piers (artist mode)"
```

---

## Task 4: Wire the gate into `core.generate_playlist_ds`

**Files:**
- Modify: `src/playlist/pipeline/core.py` (load ranks before `_build_pool`; forward kwargs through the `_build_pool` closure ~463; initial build ~512)
- Test: `tests/unit/test_core_popularity_gate_wiring.py` (new — spy on `build_candidate_pool`)

**Interfaces:**
- Consumes: `pb_cfg.popularity_rank_cutoff` (Task 3), `load_pool_popularity_ranks_cached` (Task 1), `_build_pool` closure + `build_candidate_pool` gate kwargs (Task 2).
- Produces: ranks loaded once into `_banger_ranks`; `_build_pool` forwards `popularity_ranks=_banger_ranks, popularity_rank_cutoff=<cutoff>` to `build_candidate_pool`.

- [ ] **Step 1: Write the failing test (spy)**

Create `tests/unit/test_core_popularity_gate_wiring.py`. This asserts that when a cutoff is set, the gate kwargs reach `build_candidate_pool`. Use the existing pipeline test fixtures as a reference for a minimal `generate_playlist_ds` invocation (see `tests/support/gui_fidelity.py` and the golden pipeline tests for how a bundle/artifact is faked or skipped). If a full invocation needs artifacts unavailable in the worktree, mark the test `@pytest.mark.integration` and instead unit-test the **forwarding logic** by extracting it; otherwise spy:

```python
import numpy as np
import pytest
import src.playlist.pipeline.core as core


def test_build_pool_forwards_gate_kwargs(monkeypatch):
    captured = {}

    def fake_build_candidate_pool(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")   # we only care about the kwargs

    monkeypatch.setattr(core, "build_candidate_pool", fake_build_candidate_pool)
    monkeypatch.setattr(
        core, "load_pool_popularity_ranks_cached",
        lambda bundle, idx, *, db_path: np.array([0, 5, -1]),
        raising=False,
    )
    # ... build the minimal generate_playlist_ds(...) call used by the golden tests,
    # with pier_bridge_config.popularity_rank_cutoff=10, wrapped in pytest.raises:
    with pytest.raises(RuntimeError, match="stop after capture"):
        core.generate_playlist_ds(...)   # fill args per the golden pipeline test
    assert captured.get("popularity_rank_cutoff") == 10
    assert captured.get("popularity_ranks") is not None
```

> Note for the implementer: if wiring a full `generate_playlist_ds(...)` call is impractical without artifacts, split the rank-load + forwarding into a tiny pure helper in `core.py` (e.g. `_banger_gate_kwargs(bundle, pb_cfg, pool_indices) -> dict`) and unit-test THAT directly. Prefer the pure-helper route — it is the cleaner test.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_core_popularity_gate_wiring.py -q --basetemp=$PT`
Expected: FAIL

- [ ] **Step 3: Load ranks once, before the initial build**

In `generate_playlist_ds`, just before `_candidate_cfg = replace(cfg.candidate, **_candidate_cfg_kwargs)` (~line 511), add:

```python
    # Oops, All Bangers: cache-only popularity RANK for the gate (separate from the
    # beam's score load). Loaded once over the full bundle; the gate reads per-index.
    _banger_cutoff: Optional[int] = (
        int(pb_cfg.popularity_rank_cutoff)
        if getattr(pb_cfg, "popularity_rank_cutoff", None) is not None
        else None
    )
    _banger_ranks = None
    if _banger_cutoff is not None:
        from src.analyze.popularity_runner import (
            enrichment_db_path,
            load_pool_popularity_ranks_cached,
        )
        _banger_ranks = load_pool_popularity_ranks_cached(
            bundle, list(range(len(bundle.track_ids))), db_path=enrichment_db_path()
        )
```

- [ ] **Step 4: Forward the kwargs through `_build_pool`**

Add two parameters to the `_build_pool` closure and forward them to `build_candidate_pool`. Change the signature (line 463) to:

```python
    def _build_pool(candidate_cfg: Any, genre_gate: Optional[float],
                    popularity_rank_cutoff: Optional[int] = _banger_cutoff):
```

and add to the `build_candidate_pool(...)` call (after `genre_graph_source=genre_graph_source,` ~line 501):

```python
            popularity_ranks=_banger_ranks,
            popularity_rank_cutoff=popularity_rank_cutoff,
```

> `_build_pool` is defined after `_banger_cutoff`/`_banger_ranks` only if Step 3 is placed ABOVE the `def _build_pool`. The closure currently sits at line 463, before line 511. **Move the Step-3 block to immediately before `def _build_pool` (line 463)** so the closure captures `_banger_ranks`/`_banger_cutoff`. Keep the existing `_candidate_cfg`/`pool = _build_pool(...)` lines where they are.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/unit/test_core_popularity_gate_wiring.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pipeline/core.py tests/unit/test_core_popularity_gate_wiring.py
git commit -m "feat(bangers): load popularity ranks and wire the gate into the pool build"
```

---

## Task 5: Relax-to-fill cascade (`sonic → pace → genre → popularity`)

**Files:**
- Modify: `src/playlist/pipeline/core.py` (cascade generator + loop after the initial build ~512)
- Test: `tests/unit/test_banger_cascade.py` (new)

**Interfaces:**
- Consumes: `CandidatePoolConfig` (fields `min_sonic_similarity`, `sonic_admission_percentile`, `bpm_admission_max_log_distance`, `onset_admission_max_log_distance`), the genre gate float, the cutoff int; `_build_pool` (Task 4).
- Produces: `_banger_relaxation_steps(base_cfg, base_genre_gate, base_cutoff) -> Iterator[_BangerRelaxStep]` yielding the spec's §3.5 ladder in order; a cascade loop that rebuilds until the pool reaches `_min_banger_pool` (= `max(2 * num_tracks, 40)`) or steps are exhausted, logging each notch.

- [ ] **Step 1: Write the failing test (ladder order is the invariant)**

Create `tests/unit/test_banger_cascade.py`:

```python
from dataclasses import replace
from src.playlist.pipeline.core import _banger_relaxation_steps
from src.playlist.config import CandidatePoolConfig


def _cfg():
    return CandidatePoolConfig(
        similarity_floor=0.0, min_sonic_similarity=0.3, max_pool_size=200,
        target_artists=20, candidates_per_artist=6, seed_artist_bonus=4,
        max_artist_fraction_final=0.2, sonic_admission_percentile=0.6,
    )


def test_ladder_order_sonic_pace_genre_then_popularity_last():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    labels = [s.label for s in steps]
    # sonic/pace appear before genre; popularity rungs are strictly last
    pop_idx = [i for i, l in enumerate(labels) if l.startswith("popularity")]
    genre_idx = [i for i, l in enumerate(labels) if l.startswith("genre")]
    sonic_pace_idx = [i for i, l in enumerate(labels) if l.startswith(("sonic", "pace"))]
    assert max(sonic_pace_idx) < min(genre_idx)         # sonic/pace before genre
    assert max(genre_idx) < min(pop_idx)                # genre before popularity
    assert pop_idx == list(range(min(pop_idx), len(labels)))  # popularity is the tail


def test_popularity_rungs_loosen_cutoff_then_disable():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    pop_cutoffs = [s.rank_cutoff for s in steps if s.label.startswith("popularity")]
    assert pop_cutoffs == [25, 50, None]   # top-25, top-50, gate off (last resort)


def test_final_step_disables_all_gates():
    steps = list(_banger_relaxation_steps(_cfg(), base_genre_gate=0.2, base_cutoff=10))
    last = steps[-1]
    assert last.rank_cutoff is None
    assert last.genre_gate is None
    assert last.candidate_cfg.sonic_admission_percentile in (0.0, None)
    assert last.candidate_cfg.min_sonic_similarity in (0.0, None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_banger_cascade.py -q --basetemp=$PT`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement the cascade generator**

Add to `src/playlist/pipeline/core.py` (module scope, near `_relaxed_one_each_candidate_attempts` ~line 97). The ladder follows spec §3.5 exactly:

```python
@dataclass
class _BangerRelaxStep:
    candidate_cfg: Any
    genre_gate: Optional[float]
    rank_cutoff: Optional[int]
    label: str


def _loosen_sonic(cfg, level: float):
    """level in (0,1] scales the sonic floors toward open; 0.0 disables sonic gating."""
    sap = cfg.sonic_admission_percentile
    mss = cfg.min_sonic_similarity
    if level <= 0.0:
        return replace(cfg, sonic_admission_percentile=0.0, min_sonic_similarity=None)
    return replace(
        cfg,
        sonic_admission_percentile=(float(sap) * level) if sap else sap,
        min_sonic_similarity=(float(mss) * level) if mss else mss,
    )


def _loosen_pace(cfg, *, off: bool):
    """Widen the BPM/onset admission bands; off=True disables pace gating entirely."""
    if off:
        return replace(cfg, bpm_admission_max_log_distance=float("inf"),
                       onset_admission_max_log_distance=float("inf"))
    def _wider(x):
        return float("inf") if x == float("inf") else float(x) * 2.0
    return replace(cfg,
                   bpm_admission_max_log_distance=_wider(cfg.bpm_admission_max_log_distance),
                   onset_admission_max_log_distance=_wider(cfg.onset_admission_max_log_distance))


def _banger_relaxation_steps(base_cfg, base_genre_gate, base_cutoff):
    """Progressively looser banger-pool admission, in the fixed priority order
    sonic -> pace -> genre -> popularity (spec §3.5). Sonic gets the most/earliest
    relaxation; popularity is the LAST rung and the ONLY one that admits a non-banger.
    Mirrors _relaxed_one_each_candidate_attempts (a deterministic generator)."""
    cfg, gate, cutoff = base_cfg, base_genre_gate, base_cutoff
    # 1 sonic notch 1, 2 pace notch 1, 3 sonic notch 2, 4 pace off, 5 sonic off
    cfg = _loosen_sonic(cfg, 0.66);            yield _BangerRelaxStep(cfg, gate, cutoff, "sonic notch1")
    cfg = _loosen_pace(cfg, off=False);        yield _BangerRelaxStep(cfg, gate, cutoff, "pace notch1")
    cfg = _loosen_sonic(cfg, 0.33);            yield _BangerRelaxStep(cfg, gate, cutoff, "sonic notch2")
    cfg = _loosen_pace(cfg, off=True);         yield _BangerRelaxStep(cfg, gate, cutoff, "pace off")
    cfg = _loosen_sonic(cfg, 0.0);             yield _BangerRelaxStep(cfg, gate, cutoff, "sonic off")
    # 6 genre notch (one past the user), 7 genre off
    gate = (float(base_genre_gate) * 0.5) if base_genre_gate is not None else None
    yield _BangerRelaxStep(cfg, gate, cutoff, "genre notch1")
    gate = None
    yield _BangerRelaxStep(cfg, gate, cutoff, "genre off")
    # 8-10 popularity: the only purity-breaking rungs (logged loudly by the caller)
    for new_cutoff in (25, 50, None):
        cutoff = new_cutoff
        yield _BangerRelaxStep(cfg, gate, cutoff, f"popularity top-{new_cutoff}" if new_cutoff else "popularity off")
```

(Confirm `from dataclasses import dataclass, replace` and `from typing import Any, Optional, Iterator` are imported at the top of `core.py`; `replace` is already used in the file.)

- [ ] **Step 4: Run the generator tests**

Run: `python -m pytest tests/unit/test_banger_cascade.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 5: Add the cascade loop after the initial build**

Right after `pool = _build_pool(_candidate_cfg, min_genre_similarity)` and `pool.stats["target_length"] = num_tracks` (~line 512-513), add:

```python
    # Oops, All Bangers: relax-to-fill cascade. If the banger-gated pool is too
    # small to build a coherent playlist, relax sonic -> pace -> genre -> popularity
    # (popularity LAST — the only purity-breaking rung), rebuilding and stopping the
    # instant the pool fills. Only runs when the gate is active.
    if _banger_cutoff is not None:
        _min_banger_pool = max(2 * int(num_tracks), 40)
        _pool_n = len(getattr(pool, "eligible_indices", pool.pool_indices))
        if _pool_n < _min_banger_pool:
            for _step in _banger_relaxation_steps(_candidate_cfg, min_genre_similarity, _banger_cutoff):
                logger.info(
                    "Bangers relax-to-fill: pool=%d < target=%d -> relaxing [%s]%s",
                    _pool_n, _min_banger_pool, _step.label,
                    "  (ADMITTING NON-BANGERS)" if _step.label.startswith("popularity") else "",
                )
                pool = _build_pool(_step.candidate_cfg, _step.genre_gate,
                                   popularity_rank_cutoff=_step.rank_cutoff)
                pool.stats["target_length"] = num_tracks
                _pool_n = len(getattr(pool, "eligible_indices", pool.pool_indices))
                if _pool_n >= _min_banger_pool:
                    logger.info("Bangers relax-to-fill: filled at [%s] pool=%d", _step.label, _pool_n)
                    break
```

> This relaxes the *initial* `_candidate_cfg`/`min_genre_similarity`/`_banger_cutoff`. It runs before the pier-bridge build, so the One-Each loop and infeasible-handling still operate downstream on the (now adequately sized) banger pool.

- [ ] **Step 6: Run the cascade tests again + the wiring test**

Run: `python -m pytest tests/unit/test_banger_cascade.py tests/unit/test_core_popularity_gate_wiring.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist/pipeline/core.py tests/unit/test_banger_cascade.py
git commit -m "feat(bangers): sonic->pace->genre->popularity relax-to-fill cascade"
```

---

## Task 6: Policy-layer OOPS sonic/pace baseline override

**Files:**
- Modify: `src/playlist_gui/ui_state.py` (add `popularity_mode` field to `UIStateModel`)
- Modify: `src/playlist_web/app.py` (populate `ui.popularity_mode` from the request, ~line 180-194)
- Modify: `src/playlist_gui/policy.py` (override sonic/pace mode when oops, ~after line 257)
- Test: `tests/unit/test_gui_policy.py` (extend)

**Interfaces:**
- Consumes: `ui.popularity_mode`.
- Produces: when `popularity_mode == "oops"`, `overrides["playlists"]["sonic_mode"]` and `["pace_mode"]` are set to the OOPS baseline (`_OOPS_SONIC_MODE`, `_OOPS_PACE_MODE`, both `"dynamic"`); `genre_mode` is untouched.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_gui_policy.py` (mirror how the file builds `UIStateModel` and reads overrides):

```python
def test_oops_overrides_sonic_and_pace_baseline_not_genre():
    from src.playlist_gui.policy import derive_runtime_config
    from src.playlist_gui.ui_state import UIStateModel
    ui = UIStateModel(   # fill required fields as the other tests in this file do
        genre_mode="strict", sonic_mode="strict", pace_mode="strict",
        popularity_mode="oops",
    )
    pol = derive_runtime_config(ui)
    pl = pol.overrides["playlists"]
    assert pl["sonic_mode"] == "dynamic"   # OOPS owns sonic
    assert pl["pace_mode"] == "dynamic"    # OOPS owns pace
    assert pl["genre_mode"] == "strict"    # user still owns genre


def test_on_and_off_do_not_override_modes():
    from src.playlist_gui.policy import derive_runtime_config
    from src.playlist_gui.ui_state import UIStateModel
    for m in ("on", "off"):
        ui = UIStateModel(genre_mode="strict", sonic_mode="strict",
                          pace_mode="strict", popularity_mode=m)
        pl = derive_runtime_config(ui).overrides["playlists"]
        assert pl["sonic_mode"] == "strict"
        assert pl["pace_mode"] == "strict"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_gui_policy.py -q --basetemp=$PT`
Expected: FAIL (`UIStateModel` has no `popularity_mode`, or override absent)

- [ ] **Step 3: Add `popularity_mode` to `UIStateModel`**

In `src/playlist_gui/ui_state.py`, add to the dataclass (default keeps existing call sites valid):

```python
    popularity_mode: str = "off"   # Oops All Bangers: off / on / oops
```

- [ ] **Step 4: Add the override in `derive_runtime_config`**

In `src/playlist_gui/policy.py`, add module constants near the top:

```python
# Oops, All Bangers OOPS baseline: OOPS owns sonic + pace (loosened "radio" tier),
# the user still owns genre. Tunable; calibrate by ear (see spec §10).
_OOPS_SONIC_MODE = "dynamic"
_OOPS_PACE_MODE = "dynamic"
```

Then immediately after the block that sets `playlists.sonic_mode` / `playlists.pace_mode` (after line 257), add:

```python
    # Oops, All Bangers: OOPS rests looser on sonic/pace ("FM radio" baseline);
    # genre_mode is left as the user's umbrella width.
    if getattr(ui, "popularity_mode", "off") == "oops":
        _set_nested(overrides, "playlists.sonic_mode", _OOPS_SONIC_MODE)
        _set_nested(overrides, "playlists.pace_mode", _OOPS_PACE_MODE)
        notes.append(f"OOPS baseline: sonic_mode->{_OOPS_SONIC_MODE}, pace_mode->{_OOPS_PACE_MODE} (genre untouched)")
```

- [ ] **Step 5: Populate `ui.popularity_mode` in the web app**

In `src/playlist_web/app.py`, where the `UIStateModel` is constructed before `derive_runtime_config(ui, ...)` (~line 180-194), add `popularity_mode=<request>.popularity_mode` to the constructor (match the request object name used there; `GenerateRequestBody` carries `popularity_mode`, schemas.py:37).

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/unit/test_gui_policy.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist_gui/ui_state.py src/playlist_gui/policy.py src/playlist_web/app.py tests/unit/test_gui_policy.py
git commit -m "feat(bangers): OOPS owns sonic/pace baseline via policy layer"
```

---

## Task 7: Config defaults + regenerate pipeline goldens

**Files:**
- Modify: `config.example.yaml` (add `playlists.bangers.*`)
- Regen: `tests/unit/goldens/pipeline/*.json` (4 files — `PierBridgeConfig` gained a field)

**Interfaces:**
- Consumes: Task 3's `popularity_rank_cutoff` field (the golden drift source).
- Produces: documented config defaults; green goldens whose only diff is the new key.

- [ ] **Step 1: Add config defaults**

In `config.example.yaml`, under `playlists:`, add (or extend the existing `bangers:` block from the prior session, which already has `strength_on`/`strength_oops`):

```yaml
  bangers:
    # Oops, All Bangers (popularity_mode dropdown: off / on / oops)
    rank_cutoff_on: 50      # ON admits Last.fm rank < 50 (trims deep cuts/deluxe/live)
    rank_cutoff_oops: 10    # OOPS admits rank < 10 (real hits only)
    strength_on: 0.25       # secondary beam penalty (existing)
    strength_oops: 0.60     # secondary beam penalty (existing)
    # OOPS sonic/pace baseline lives in policy.py (_OOPS_SONIC_MODE/_OOPS_PACE_MODE);
    # min_banger_pool + relax ladder are core defaults (max(2*tracks,40) / spec §3.5).
```

- [ ] **Step 2: Confirm the goldens drift (expected failure)**

Run the pipeline config golden test (per the handoff this is `test_pipeline_smoke_golden`; find it under `tests/unit/`):
Run: `python -m pytest tests/unit -k "golden and pipeline" -q --basetemp=$PT`
Expected: FAIL — the 4 goldens now mismatch on the new `popularity_rank_cutoff` key.

- [ ] **Step 3: Regenerate the goldens**

Delete the 4 baseline files, then run the golden test twice (first run writes + skips, second passes):

```bash
rm tests/unit/goldens/pipeline/dynamic_default.json \
   tests/unit/goldens/pipeline/narrow_with_pier_bridge_overrides.json \
   tests/unit/goldens/pipeline/narrow_progress_arc_dry_run.json \
   tests/unit/goldens/pipeline/discover_with_dj_bridging.json
python -m pytest tests/unit -k "golden and pipeline" -q --basetemp=$PT   # writes baselines, skips
python -m pytest tests/unit -k "golden and pipeline" -q --basetemp=$PT   # passes
```

- [ ] **Step 4: Verify the diff is ONLY the new key**

Run: `git diff tests/unit/goldens/pipeline/`
Expected: every change is an added `popularity_rank_cutoff` (value `null`) line — nothing else. If anything else drifted, STOP and investigate (do not commit a noisy golden).

- [ ] **Step 5: Commit**

```bash
git add config.example.yaml tests/unit/goldens/pipeline/
git commit -m "chore(bangers): config defaults + regen pipeline goldens for popularity_rank_cutoff"
```

---

## Task 8: Full unit sweep

**Files:** none (verification task)

- [ ] **Step 1: Run the bangers-related unit tests together**

Run:
```bash
python -m pytest tests/unit/test_popularity_pool_loader.py tests/unit/test_candidate_pool_popularity_gate.py tests/unit/test_bangers_resolve.py tests/unit/test_pier_bridge_config_popularity.py tests/unit/test_core_popularity_gate_wiring.py tests/unit/test_banger_cascade.py tests/unit/test_gui_policy.py -q --basetemp=$PT
```
Expected: PASS (quote the real count).

- [ ] **Step 2: Run the broader fast suite to catch regressions**

Run: `python -m pytest tests/unit -q -m "not slow" --basetemp=$PT`
Expected: PASS. If anything adjacent breaks (config parsing, policy, pier-bridge config), fix root cause — do not loosen assertions. Quote real pass/fail counts from output you actually saw.

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "test(bangers): full unit sweep green"
```

---

## Task 9: Live GUI verification (USER-DRIVEN — hand off)

**This worktree has no real data; generation must be verified by the user from the MAIN checkout.** Do not claim success without this.

- [ ] **Step 1: Build the front end and prepare**

Provide the user these steps (run from the MAIN checkout `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3`, which has real data):
1. Merge or check out this branch (see `superpowers:finishing-a-development-branch` at the end).
2. `npm --prefix web run build` (rebuild `web/dist`).
3. Restart `python tools/serve_web.py` (kill the old process first — the worker must reload).

- [ ] **Step 2: Verification script (the user runs these, reading the logs)**

For a Pearl Jam (or Nirvana) artist-mode seed, generate at each mode and read the generation logs (INFO), not just the track list:
- **OFF:** baseline; unchanged from today.
- **ON:** no deep cuts / live / deluxe; `Popularity gate applied: cutoff=top-50 ...` in logs; piers unchanged (popular_seeds NOT forced).
- **OOPS:** cross-genre hits present (Float On / Kool Thing-type), piers are the artist's hits (popular_seeds forced; `OOPS baseline: sonic_mode->dynamic ...`), and the per-track Last.fm rank annotation shows mostly top-10. If the cascade fired, `Bangers relax-to-fill: ...` lines show the order and where it filled; any `ADMITTING NON-BANGERS` line is the only place a non-hit entered.
- **Budget:** every generation completes < 90s.
- **Seed mode (regression):** a multi-seed seed-mode playlist with OOPS still respects the user's chosen seeds as piers (popular_seeds NOT forced) while the bridge gate applies.

- [ ] **Step 3: Calibrate (post-verify, with the user)**

Per spec §10, tune by ear: OOPS sonic/pace baseline (policy constants), cutoffs (`rank_cutoff_*`), `min_banger_pool`, and `popular_seeds_weight`. These are the "see behavior first" knobs.

---

## Self-Review (completed)

- **Spec coverage:** rank loader (§3.1→T1), gate placement + backstop reconciliation (§3.2/§3.6→T2), threading + `PierBridgeConfig` field + popular-seed piers (§3.3/§3.8→T3), gate activation in core (§3.3→T4), cascade (§3.5→T5), OOPS owns sonic/pace (§3.4→T6), config + goldens (§5→T7), secondary beam penalty (§3.7→unchanged, untouched by design), observability (§6→logging in T2/T5), testing (§8→T1-T8), live verify (§8→T9). Deferred items (version-preference, single-pass relax, config-wiring of baselines/ladder/min_pool) are called out in the spec §9/§10 and not tasked — intentional.
- **Placeholder scan:** the only `...` is in Task 4's spy test where the implementer fills the `generate_playlist_ds(...)` args from the golden pipeline test, with an explicit pure-helper fallback — not a silent gap.
- **Type consistency:** `popularity_rank_cutoff` (Optional[int]) and `popularity_ranks` (np.ndarray) names are identical across T2 (gate), T3 (config field), T4 (core wiring), T5 (cascade). `_resolve_popularity_rank_cutoff` / `_resolve_popular_seeds` names match between T3 definition and T3 wiring. `_banger_relaxation_steps` / `_BangerRelaxStep` match between T5 definition, test, and loop.
