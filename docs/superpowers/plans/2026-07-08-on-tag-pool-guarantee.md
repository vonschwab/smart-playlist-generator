# On-Tag Pool Guarantee Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Force-admit a capped, per-artist-limited set of eligible authority on-tag tracks into the DS candidate pool (past the per-artist rank walk), so on-tag bridge tracks (e.g. Ghost Box for BoC+hauntology) reach the beam.

**Architecture:** A pure ranking/cap helper selects which on-tag eligible indices to guarantee. `build_candidate_pool` gains guarantee params and force-admits them after its rank walk + backstop. `pipeline/core.py` derives the guarantee ids from the on-tag rows it already resolves (`_rows`, seed-artist-excluded) and passes them through. Segment pools inherit (segment_pool_max ≫ pool size).

**Tech Stack:** Python 3.11, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-on-tag-pool-guarantee.md`. **Prior art:** genre-mode notes F3/F5; the tag-first pier plan (same session).

## Global Constraints

- **One Rule:** on-tag ids come from `_rows` (already resolved via `resolve_tag_sonic_prototype_rows`, which reads `release_effective_genres`). No new genre read.
- **Only guarantee ELIGIBLE tracks** (passed the pool's sonic/genre/BPM gates) — never inject sub-floor jarring bridges (#25).
- **Live default when steering active; `max=0` = byte-identical rollback** (#22). No-op when no tag / no on-tag rows.
- **Diversity:** per-artist cap on the guaranteed set; final-playlist diversity constraints unchanged → no flooding (#11).
- **Tests mirror production:** generation validation through a real `PlaylistGenerator` (artist mode) as in the tag-first pier plan — `generate_like_gui` only reaches seeds mode.
- **Sub-agent models:** Task 1 haiku (pure, mechanical); Tasks 2, 3 sonnet (integration/validation). Never inherit the session model.
- **Shared checkout:** commit explicit paths only; never `git add -A/-u/.`; verify `git diff --cached --name-only`.

---

## File Structure

- `src/playlist/candidate_pool.py` — add pure `select_pool_guarantee(...)`; add 3 params to `build_candidate_pool` + force-admit block after the backstop (~line 1308).
- `src/playlist/pipeline/core.py` — capture `_on_tag_guarantee_ids` from `_rows` (~623); pass the 3 params in `_build_pool`'s `build_candidate_pool(...)` call (~673-716).
- `config.example.yaml` — document the 2 knobs.
- `tests/unit/test_pool_guarantee.py` — new unit tests (Task 1).
- `tests/integration/test_gui_fidelity_regressions.py` — integration cases (Task 3).

---

### Task 1: Pure `select_pool_guarantee` helper

**Files:**
- Modify: `src/playlist/candidate_pool.py` (add module-level function near other pool helpers, above `build_candidate_pool` ~line 518)
- Test: `tests/unit/test_pool_guarantee.py`

**Interfaces:**
- Produces: `select_pool_guarantee(candidate_indices, guarantee_ids, track_ids, artist_keys, sonic_seed_sim, already_admitted, max_total, per_artist) -> list[int]` — indices to append to the pool, ranked by `sonic_seed_sim` desc, capped per-artist and total, skipping non-guarantee / already-admitted.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pool_guarantee.py
import numpy as np
from src.playlist.candidate_pool import select_pool_guarantee


def test_guarantee_ranks_by_sim_and_caps_total():
    track_ids = np.array([f"t{i}" for i in range(8)])
    artist_keys = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    sim = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    got = select_pool_guarantee(
        candidate_indices=range(8),
        guarantee_ids={"t0", "t2", "t4", "t6"},   # one per artist a,b,c,d
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted=set(), max_total=3, per_artist=2,
    )
    assert got == [2, 4, 6]        # highest-sim guarantee ids first (t2=.8,t4=.7,t6=.6), capped at 3


def test_guarantee_per_artist_cap():
    track_ids = np.array([f"t{i}" for i in range(4)])
    artist_keys = np.array(["a", "a", "a", "b"])
    sim = np.array([0.9, 0.8, 0.7, 0.6])
    got = select_pool_guarantee(
        candidate_indices=range(4), guarantee_ids={"t0", "t1", "t2", "t3"},
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted=set(), max_total=10, per_artist=2,
    )
    assert got == [0, 1, 3]        # artist 'a' capped at 2 (t0,t1); t2 dropped; t3 (artist b) kept


def test_guarantee_skips_already_admitted_and_non_guarantee():
    track_ids = np.array([f"t{i}" for i in range(4)])
    artist_keys = np.array(["a", "b", "c", "d"])
    sim = np.array([0.9, 0.8, 0.7, 0.6])
    got = select_pool_guarantee(
        candidate_indices=range(4), guarantee_ids={"t0", "t2"},
        track_ids=track_ids, artist_keys=artist_keys, sonic_seed_sim=sim,
        already_admitted={0}, max_total=10, per_artist=5,
    )
    assert got == [2]              # t0 already admitted; t1/t3 not in guarantee_ids


def test_guarantee_empty_inputs():
    tids = np.array(["t0"]); aks = np.array(["a"]); sim = np.array([0.5])
    assert select_pool_guarantee(range(1), set(), tids, aks, sim, set(), 10, 5) == []
    assert select_pool_guarantee(range(1), {"t0"}, tids, aks, sim, set(), 0, 5) == []


def test_guarantee_none_sim_falls_back_to_index_order():
    tids = np.array([f"t{i}" for i in range(3)]); aks = np.array(["a", "b", "c"])
    got = select_pool_guarantee(range(3), {"t0", "t1", "t2"}, tids, aks, None, set(), 10, 5)
    assert got == [0, 1, 2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pool_guarantee.py -q`
Expected: FAIL (ImportError: `select_pool_guarantee`).

- [ ] **Step 3: Implement**

```python
# src/playlist/candidate_pool.py  (module-level, above build_candidate_pool)
def select_pool_guarantee(
    candidate_indices,
    guarantee_ids,
    track_ids,
    artist_keys,
    sonic_seed_sim,
    already_admitted,
    max_total,
    per_artist,
) -> list[int]:
    """Indices to force-admit into the pool: those in ``candidate_indices`` (the
    eligible/gate-passing set) whose track_id is in ``guarantee_ids`` and not in
    ``already_admitted``, ranked by ``sonic_seed_sim`` desc (index tiebreak; index
    order when sim is None), capped at ``per_artist`` per normalized artist key and
    ``max_total`` overall. Pure. [] when guarantee_ids empty or max_total<=0."""
    from collections import Counter
    if not guarantee_ids or int(max_total) <= 0:
        return []
    gids = {str(g) for g in guarantee_ids}
    adm = {int(i) for i in already_admitted}
    cands = [
        int(i) for i in candidate_indices
        if str(track_ids[int(i)]) in gids and int(i) not in adm
    ]
    cands.sort(key=lambda i: (
        -(float(sonic_seed_sim[i]) if sonic_seed_sim is not None else 0.0), int(i)))
    per: Counter = Counter()
    out: list[int] = []
    for i in cands:
        if len(out) >= int(max_total):
            break
        ak = str(artist_keys[i])
        if per[ak] >= int(per_artist):
            continue
        out.append(i)
        per[ak] += 1
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_pool_guarantee.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_pool_guarantee.py
git commit --only -- src/playlist/candidate_pool.py tests/unit/test_pool_guarantee.py -m "feat(candidate-pool): pure select_pool_guarantee (rank + per-artist/total cap)"
```

---

### Task 2: Wire the guarantee into `build_candidate_pool` + `pipeline/core.py` + config

**Files:**
- Modify: `src/playlist/candidate_pool.py` (`build_candidate_pool` signature + force-admit block ~after line 1308, before `seed_sim_pool` is built ~1310)
- Modify: `src/playlist/pipeline/core.py` (~623 capture ids; ~673-716 pass params)
- Modify: `config.example.yaml` (~beside `pier_tag_skew` / the tag_steering block ~333)
- Test: covered by Task 3.

**Interfaces:**
- Consumes: `select_pool_guarantee` (Task 1).
- `build_candidate_pool` gains: `on_tag_guarantee_ids: Optional[set] = None`, `on_tag_guarantee_max: int = 0`, `on_tag_guarantee_per_artist: int = 0`.

- [ ] **Step 1: Config knobs** (`config.example.yaml`, under `pier_bridge:`, beside `pier_tag_skew`):

```yaml
      tag_steering_pool_guarantee_max: 30        # force-admit up to N eligible on-tag tracks into the DS pool past the per-artist rank walk (0 = rollback/off)
      tag_steering_pool_guarantee_per_artist: 3  # per-artist cap within the guaranteed set (keeps the roster diverse)
```

- [ ] **Step 2: Add params to `build_candidate_pool`** — in the signature (near the other tag-steering params `steering_target`/`sonic_pool_affinity`), add:

```python
    on_tag_guarantee_ids: Optional[set] = None,
    on_tag_guarantee_max: int = 0,
    on_tag_guarantee_per_artist: int = 0,
```

- [ ] **Step 3: Force-admit block** — insert immediately AFTER the min-pool backstop block (after the `if _min_pool_size > 0 ...` block ends, ~line 1308) and BEFORE `seed_sim_pool = np.array(...)` (~1310):

```python
    # Tag steering pool guarantee: force-admit eligible on-tag tracks past the
    # per-artist rank walk so on-tag bridges (which rank low sonically and get
    # walked out) reach the beam. Keyed on authority membership (genre-dense
    # discriminator); only ELIGIBLE tracks (passed the gates) are guaranteed.
    if on_tag_guarantee_ids and int(on_tag_guarantee_max) > 0:
        _already = {int(i) for i in pool_indices}
        _guar = select_pool_guarantee(
            candidate_indices=eligible,
            guarantee_ids=on_tag_guarantee_ids,
            track_ids=track_ids,
            artist_keys=artist_keys,
            sonic_seed_sim=sonic_seed_sim,
            already_admitted=_already,
            max_total=int(on_tag_guarantee_max),
            per_artist=int(on_tag_guarantee_per_artist),
        )
        if _guar:
            pool_indices.extend(_guar)
            pool_indices = list(dict.fromkeys(pool_indices))
            for _i in _guar:
                pool_artists.add(str(artist_keys[_i]))
            logger.info(
                "Tag steering pool guarantee: force-admitted %d on-tag track(s) across "
                "%d artist(s) past the rank walk (cap total=%d per_artist=%d)",
                len(_guar), len({str(artist_keys[_i]) for _i in _guar}),
                int(on_tag_guarantee_max), int(on_tag_guarantee_per_artist),
            )
        else:
            logger.info(
                "Tag steering pool guarantee: 0 eligible on-tag tracks to force-admit "
                "(all already pooled or gate-rejected).",
            )
```

Confirm `eligible`, `pool_indices`, `pool_artists`, `sonic_seed_sim`, `artist_keys`, `track_ids` are all in scope at that point (they are — `eligible` is used at the `artist_cap_excluded` line just below). Note: `sonic_seed_sim` may be `None` (no genre gate / no sonic) — the helper handles it.

- [ ] **Step 4: Capture guarantee ids in `pipeline/core.py`** — at ~line 619-623, `_rows` is the seed-artist-excluded on-tag rows. Add a local (visible to the `_build_pool` closure) BEFORE `_build_pool` is defined:

```python
    # (near the top of the tag block, initialize)
    _on_tag_guarantee_ids = None
    ...
            _rows, _n, _ = resolve_tag_sonic_prototype_rows(...)
            if _rows is not None:
                _on_tag_guarantee_ids = {str(bundle.track_ids[r]) for r in _rows}  # ADD: independent of the cohesion gate below
                _xs = np.asarray(_xsonic, dtype=np.float64)
                ... (existing cohesion-gated prototype code unchanged) ...
```

Place the `_on_tag_guarantee_ids = None` init alongside `_tag_sonic_affinity = None` (~593) so it exists even when tags are absent. Read `max`/`per_artist` from `pb_overrides` near the other tag knobs:

```python
    _guar_max = int(pb_overrides.get("tag_steering_pool_guarantee_max", 30))
    _guar_per_artist = int(pb_overrides.get("tag_steering_pool_guarantee_per_artist", 3))
```

- [ ] **Step 5: Pass params in `_build_pool`** — add to the `build_candidate_pool(...)` call (after `sonic_pool_affinity=...`):

```python
            on_tag_guarantee_ids=_on_tag_guarantee_ids,
            on_tag_guarantee_max=_guar_max,
            on_tag_guarantee_per_artist=_guar_per_artist,
```

- [ ] **Step 6: Verify off-path + imports**

Run: `python -c "import src.playlist.candidate_pool, src.playlist.pipeline.core"` — clean import.
Run: `python -m pytest tests/unit/test_pool_guarantee.py tests/test_gui_fidelity.py -q` — PASS.
Run: `ruff check src/playlist/candidate_pool.py src/playlist/pipeline/core.py` — fix only E/F issues you introduce (both files have pre-existing warnings; leave them).
With `tag_steering_pool_guarantee_max: 0` OR no tags, `on_tag_guarantee_ids` is None / max 0 → block is skipped → byte-identical.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py config.example.yaml
git commit --only -- src/playlist/candidate_pool.py src/playlist/pipeline/core.py config.example.yaml -m "feat(tag-steering): on-tag pool guarantee wired live-default (bridge-side surfacing)"
```

---

### Task 3: Integration validation + manual verify + Part-2 decision

**Files:**
- Modify: `tests/integration/test_gui_fidelity_regressions.py`

**Interfaces:** end-to-end through a real `PlaylistGenerator` (artist mode), same construction as the tag-first pier plan's Task 6 (drives `create_playlist_for_artist` via the production config chain; do NOT use `generate_like_gui` — it only reaches seeds mode).

- [ ] **Step 1: Integration cases** (mark `@pytest.mark.integration @pytest.mark.slow`, skip if artifact absent):
  - BoC + ["hauntology"], off: assert the generation log emits the "pool guarantee: force-admitted N" line with N≥1, AND the realized pool contains ≥1 non-BoC authority-hauntology track. (Read authority membership for realized pool/playlist track_ids as in Task 6's helper.)
  - BoC + ["hauntology"], off: assert the **playlist** contains ≥1 non-BoC authority-hauntology bridge. **If 0**, do NOT fail silently — assert-xfail with a message pointing to Part 2 (the beam still prefers closer neighbors; engage `tag_steering_sonic_beam_weight`), and record it for the controller.
  - Rollback (`tag_steering_pool_guarantee_max: 0`): pool contains 0 forced on-tag tracks (guarantee line absent) — guards the knob.
  - Real Estate + ["jangle pop"], off: distinct-artist count + worst-edge min-T within one notch of the pre-guarantee baseline (guarantee ≈ no-op — few/no force-admits because RE-neighborhood on-tag tracks are already walked in).

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_gui_fidelity_regressions.py -q -k "guarantee or hauntology"` (bounded; NO head/tail pipe; artifact present on canonical checkout). `ruff check` the file.

- [ ] **Step 3: MANUAL VERIFICATION (record real numbers, do not skip):**
  - Regenerate BoC + hauntology through the worker path (or `scratchpad/verify_hauntology.py`); read the log for the "pool guarantee" line; **count Ghost Box / non-BoC hauntology tracks in the realized playlist**; report worst-edge min-T vs the pre-guarantee run (`logs/playlists/2026-07-08_223912...` had none).
  - **Part-2 decision:** if the playlist now has ≥2 Ghost Box → report success, Part 2 not needed. If it has the tracks in the POOL but 0-1 in the playlist → set `tag_steering_sonic_beam_weight` to a few trial values (e.g. 0.5, 1.0, 2.0) via the override, re-run, and report the weight that surfaces ≥2 without worst-edge dropping below ~one notch of baseline. Recommend a default; do NOT commit the config default change — report the recommended value to the controller for a follow-up decision.
  - RE + jangle: confirm unchanged (quote distinct-artist + worst-edge with vs without the guarantee).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_gui_fidelity_regressions.py
git commit --only -- tests/integration/test_gui_fidelity_regressions.py -m "test(tag-steering): on-tag pool guarantee integration (BoC/hauntology surfaces, RE no-regression, rollback)"
```

---

## Self-Review (completed)

- **Spec coverage:** guarantee selection (T1), pool force-admit + caller + config (T2), validation + Part-2 gate (T3). Covered.
- **Placeholders:** none — the only read-and-confirm note (scope of `eligible` in T2) is bounded and named.
- **Type consistency:** `select_pool_guarantee(...) -> list[int]` (T1) called in T2 with `eligible` + `on_tag_guarantee_ids`; `_on_tag_guarantee_ids: set|None` (T2 core.py) → `on_tag_guarantee_ids` param (T2 candidate_pool.py). Consistent.
