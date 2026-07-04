# Pier Bridgeability Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Veto sonic-outlier tracks from pier (medoid) candidacy in artist mode so unbridgeable piers like Torrey "First Jam" (2026-07-03 incident, T=0.002 edges) can never be seated; failed clusters' slots reallocate to passing clusters.

**Architecture:** A pure bridgeability function in `src/playlist/artist_style.py` computes each artist track's k-th-best calibrated-T neighbor against the non-seed-artist library. `cluster_artist_tracks` applies it as a member filter before medoid scoring (cluster membership untouched), bumps per-cluster medoid production when clusters fail so `target_pier_count` holds, and falls back to unchecked medoids if everything fails. `playlist_generator.py` parses three new config keys at its two `ArtistStyleConfig` sites and passes `target_pier_count` at its two `cluster_artist_tracks` call sites.

**Tech Stack:** Python 3.11, numpy, pytest. Spec: `docs/superpowers/specs/2026-07-03-pier-bridgeability-design.md`.

## Global Constraints

- Defaults (verbatim from spec): `pier_bridgeability_enabled: true`, `pier_bridgeability_floor_t: 0.30`, `pier_bridgeability_k: 10`. Live default ON (activate-fixes rule) — the dataclass defaults ARE the live values.
- Calibration params come from `src.playlist.transition_metrics.resolve_transition_calib(bundle.sonic_variant)` — never hardcoded. It already raises on unknown variants (startup-error discipline satisfied).
- Fire mode (`popular_seeds_mode == "fire"`) piers are exempt automatically: `select_popular_piers` draws from `clusters` members, and the veto must NOT mutate `clusters` — it filters medoid candidacy only.
- Same-artist library rows never count as bridging neighbors; the exclusion mask uses `_artist_indices_in_bundle(bundle, artist_name, include_collaborations=True)` (all the artist's rows, collabs included, pre-dedupe — an alternate version of the same song must not fake bridgeability).
- Never-fail: if the veto would leave zero eligible members across ALL clusters, WARN and run unchecked.
- **Shared checkout warning:** another session has uncommitted work in this tree (notably `config.example.yaml`, `src/playlist/mode_presets.py`, `src/playlist/pier_bridge/beam.py`, `var_bridge.py`). Stage ONLY the explicit paths named in each commit step. Before every commit, run `git status --porcelain` and `git diff -- <file>` on each file you intend to stage; if a file contains hunks you did not write, do not stage that file — finish the commit without it and report. Target files for this plan (`src/playlist/artist_style.py`, `src/playlist_generator.py`, `tests/test_artist_style.py`) were clean at planning time.
- Run pytest directly with a bounded timeout, never piped through `head`/`tail` (project rule).

---

### Task 1: Bridgeability signal function

**Files:**
- Modify: `src/playlist/artist_style.py` (new top-level function, near `_medoids_for_cluster` ~line 400)
- Test: `tests/test_artist_style.py` (append)

**Interfaces:**
- Consumes: `src.playlist.pier_bridge.vec._calibrate_transition_cos(value, *, center, scale, gain)` (existing single source of truth for the T sigmoid).
- Produces: `compute_pier_bridgeability(X_norm, member_indices, excluded_indices, k, calib_center, calib_scale, calib_gain) -> np.ndarray` — float array aligned to `member_indices`, each value the calibrated T of that member's k-th best non-excluded neighbor. Task 2 calls this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artist_style.py` (the file already imports `numpy as np`; extend the `from src.playlist.artist_style import (...)` block with `compute_pier_bridgeability`):

```python
def _unit_rows(rows):
    X = np.array(rows, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def test_bridgeability_kth_rank_same_artist_exclusion():
    # 5-dim basis geometry. Artist 'a' rows: 0 (e1), 1 (e2), 2 (e3), 9 (e2 twin of 1).
    # Library rows: 3,4 (e1 — two neighbors for member 0), 5 (e3 — ONE neighbor for
    # member 2), 6,7 (e4), 8 (e5) filler.
    e = np.eye(5)
    X = _unit_rows([e[0], e[1], e[2],          # members 0,1,2 (artist 'a')
                    e[0], e[0], e[2],          # lib 3,4,5
                    e[3], e[3], e[4],          # lib 6,7,8
                    e[1]])                     # 9 = same-artist twin of member 1
    from src.playlist.artist_style import compute_pier_bridgeability
    # muq calibration band (what resolve_transition_calib(None) returns)
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)

    t = compute_pier_bridgeability(X, [0, 1, 2], [0, 1, 2, 9], k=2,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    assert t.shape == (3,)
    assert t[0] > 0.9        # member 0: two e1 library neighbors -> kth cos = 1.0
    assert t[1] < 0.05       # member 1: only close row is same-artist twin 9 -> excluded
    assert t[2] < 0.05       # member 2: ONE library neighbor but k=2 -> kth cos = 0.0

    # Same geometry, k=1, excluding ONLY self (callers must always exclude the
    # member's own row — production passes all artist rows): the same-artist twin
    # at row 9 now counts and rescues member 1 -> proves the artist mask (not the
    # geometry) drove the failure above.
    t_selfonly = compute_pier_bridgeability(X, [1], [1], k=1,
                                            calib_center=c, calib_scale=s, calib_gain=g)
    assert t_selfonly[0] > 0.9


def test_bridgeability_empty_members_and_k_clamp():
    from src.playlist.artist_style import compute_pier_bridgeability
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)
    e = np.eye(3)
    X = _unit_rows([e[0], e[0], e[1]])
    assert compute_pier_bridgeability(X, [], [0], k=10,
                                      calib_center=c, calib_scale=s, calib_gain=g).shape == (0,)
    # k larger than available non-excluded columns clamps to what's there (2 columns).
    t = compute_pier_bridgeability(X, [0], [0], k=10,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    # kth clamps to 2nd best = cos(e0, e1) = 0 -> low T, but finite (no crash/inf)
    assert np.isfinite(t[0]) and t[0] < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -q -k bridgeability` (timeout 120s)
Expected: FAIL — `ImportError: cannot import name 'compute_pier_bridgeability'`

- [ ] **Step 3: Implement the function**

In `src/playlist/artist_style.py`, add after `_medoids_for_cluster` (imports: the module already imports numpy and `logging`; add `from src.playlist.pier_bridge.vec import _calibrate_transition_cos` at the top with the other imports):

```python
def compute_pier_bridgeability(
    X_norm: np.ndarray,
    member_indices: Sequence[int],
    excluded_indices: Sequence[int],
    k: int,
    *,
    calib_center: float,
    calib_scale: float,
    calib_gain: float,
) -> np.ndarray:
    """Calibrated T of each member's k-th best library neighbor (pier bridgeability).

    A pier must have at least k library tracks it could plausibly sit next to;
    same-artist rows (``excluded_indices``, collabs and alternate versions
    included) never count — interiors can't be seed-artist tracks. Returns a
    float array aligned to ``member_indices``. See
    docs/superpowers/specs/2026-07-03-pier-bridgeability-design.md.
    """
    members = np.asarray(list(member_indices), dtype=int)
    if members.size == 0:
        return np.zeros(0, dtype=float)
    excl = np.asarray(sorted({int(i) for i in excluded_indices}), dtype=int)
    n_avail = int(X_norm.shape[0]) - int(excl.size)
    if n_avail <= 0:
        return np.zeros(members.size, dtype=float)
    sims = X_norm[members] @ X_norm.T  # (m, N) cosines; rows are already L2-normalized
    if excl.size:
        sims[:, excl] = -np.inf
    kk = max(1, min(int(k), n_avail))
    kth = np.partition(sims, sims.shape[1] - kk, axis=1)[:, sims.shape[1] - kk]
    return np.asarray(
        [
            _calibrate_transition_cos(
                float(v), center=calib_center, scale=calib_scale, gain=calib_gain
            )
            for v in kth
        ],
        dtype=float,
    )
```

Note `Sequence` is already imported in this module (used by `_medoids_for_cluster`'s `bundle_track_ids: Sequence[str]`); verify, and add to the `typing` import if absent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -q -k bridgeability` (timeout 120s)
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): pier bridgeability signal (kth-neighbor calibrated T)"
```

---

### Task 2: Veto + reallocation inside cluster_artist_tracks

**Files:**
- Modify: `src/playlist/artist_style.py` — `ArtistStyleConfig` (~line 55), `cluster_artist_tracks` (~lines 597–761)
- Test: `tests/test_artist_style.py` (append)

**Interfaces:**
- Consumes: `compute_pier_bridgeability` (Task 1); `src.playlist.transition_metrics.resolve_transition_calib(variant)` -> `(center, scale, gain)`; `_artist_indices_in_bundle(bundle, artist_name, include_collaborations=True)` (existing).
- Produces: `cluster_artist_tracks(..., target_pier_count: Optional[int] = None)` — new keyword-only-by-position optional param, default `None` preserves every existing call site (scripts included). New `ArtistStyleConfig` fields: `pier_bridgeability_enabled: bool = True`, `pier_bridgeability_floor_t: float = 0.30`, `pier_bridgeability_k: int = 10`. Task 3 relies on these exact names.

**Behavior contract (from spec):**
1. Filter applies to medoid candidacy only; `clusters` lists are never mutated.
2. A cluster with zero eligible members appends `[]` to `medoids_by_cluster` and contributes no medoids (fire's member pool, tag affinities, `slot_targets[ci]` alignment all preserved — the cluster is still appended to `clusters`).
3. When some (but not all) clusters fail and `target_pier_count` is provided, per-cluster medoid production bumps to `max(medoid_top_k, ceil(target_pier_count / n_eligible_clusters))` so the caller's existing cap-to-target keeps pier count at target. `top_k` (the first-pick randomization band) stays `medoid_top_k` — unchanged semantics.
4. If NO cluster has an eligible member: WARN and run unchecked (never-fail).
5. When nothing is vetoed, output is identical to a disabled run (guard test below).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artist_style.py`:

```python
def _bridgeability_fixture():
    """Artist 'a' with 2 sonic clusters; only cluster A has library mass near it.

    Rows 0-2: artist cluster A (~e1, tiny jitter so kmeans is stable).
    Rows 3-5: artist cluster B (~e2) — isolated, no library neighbors.
    Rows 6-11: library tracks near e1 (cluster A's bridges).
    Rows 12-13: library filler on e3.
    """
    e = np.eye(4)
    jit = [0.00, 0.01, -0.01]
    a_rows = [e[0] + j * e[3] for j in jit] + [e[1] + j * e[3] for j in jit]
    lib_rows = [e[0] + j * e[3] for j in (0.02, -0.02, 0.03, -0.03, 0.04, -0.04)]
    fill = [e[2], e[2] + 0.01 * e[3]]
    X = _unit_rows(a_rows + lib_rows + fill)
    artist_keys = np.array(["a"] * 6 + ["lib"] * 8)
    track_ids = np.array([f"t{i}" for i in range(14)])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def _cfg_bridge(**kw):
    base = dict(cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
                dedupe_versions=False, pier_bridgeability_k=3,
                pier_bridgeability_floor_t=0.30)
    base.update(kw)
    return ArtistStyleConfig(**base)


def test_outlier_cluster_contributes_no_piers_and_slots_reallocate():
    bundle = _bridgeability_fixture()
    clusters, medoids, by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0,
        medoid_top_k=2, target_pier_count=4,
    )
    assert len(clusters) == 2                      # cluster membership untouched
    assert sorted(len(m) for m in by_cluster) == [0, 3]   # B empty; A bumped to all 3
    assert all(m in {0, 1, 2} for m in medoids)    # every pier from cluster A
    assert len(medoids) == 3                       # ceil(4/1)=4 capped by cluster size


def test_bridgeability_all_fail_falls_back_unchecked():
    # Shrink the library to filler only: no artist track has 3 neighbors anywhere.
    e = np.eye(4)
    X = _unit_rows([e[0], e[0] + 0.01 * e[3], e[1], e[1] + 0.01 * e[3], e[2], e[3]])
    bundle = DummyBundle(X_sonic=X, artist_keys=np.array(["a"] * 4 + ["lib"] * 2),
                         track_ids=np.array([f"t{i}" for i in range(6)]))
    checked = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0, medoid_top_k=1)
    unchecked = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0, medoid_top_k=1)
    assert checked[1] == unchecked[1]              # same medoids: never-fail fallback


def test_bridgeability_no_veto_is_byte_identical():
    bundle = _bridgeability_fixture()
    # Floor 0.0 => nothing vetoed => identical to disabled.
    on = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_floor_t=0.0), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    off = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    assert on[1] == off[1] and on[2] == off[2]


def test_artist_style_config_has_bridgeability_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.pier_bridgeability_enabled is True   # live default (activate-fixes rule)
    assert cfg.pier_bridgeability_floor_t == 0.30
    assert cfg.pier_bridgeability_k == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -q -k "bridgeability or outlier_cluster"` (timeout 120s)
Expected: FAIL — `TypeError: ArtistStyleConfig.__init__() got an unexpected keyword argument 'pier_bridgeability_k'` (and the two Task 1 tests still pass).

- [ ] **Step 3: Implement**

3a. Add fields to `ArtistStyleConfig` (after `medoid_tag_weight`, keeping the dataclass frozen):

```python
    # Pier bridgeability veto (spec 2026-07-03): a medoid candidate needs >= k
    # non-seed-artist library neighbors at calibrated T >= floor_t to seat as a
    # pier. Vetoes medoid candidacy only; cluster membership is untouched.
    pier_bridgeability_enabled: bool = True
    pier_bridgeability_floor_t: float = 0.30
    pier_bridgeability_k: int = 10
```

3b. Add `target_pier_count: Optional[int] = None` to the `cluster_artist_tracks` keyword params (it is a `*`-only signature already).

3c. In `cluster_artist_tracks`, after the `if len(artist_indices) < max(3, cfg.cluster_k_min)` guard (~line 657) and before the energy block, compute eligibility:

```python
    # Pier bridgeability veto (medoid candidacy only — never mutates clusters).
    bridgeable_set: Optional[set] = None
    if cfg.pier_bridgeability_enabled:
        from src.playlist.transition_metrics import resolve_transition_calib
        _cal_c, _cal_s, _cal_g = resolve_transition_calib(
            getattr(bundle, "sonic_variant", None)
        )
        _excl_cols = _artist_indices_in_bundle(
            bundle, artist_name, include_collaborations=True
        )
        _bt = compute_pier_bridgeability(
            X_norm, artist_indices, _excl_cols, cfg.pier_bridgeability_k,
            calib_center=_cal_c, calib_scale=_cal_s, calib_gain=_cal_g,
        )
        _floor = float(cfg.pier_bridgeability_floor_t)
        bridgeable_set = {
            idx for idx, t in zip(artist_indices, _bt) if float(t) >= _floor
        }
        _vetoed = [
            (idx, float(t)) for idx, t in zip(artist_indices, _bt) if float(t) < _floor
        ]
        if _vetoed:
            logger.info(
                "Pier bridgeability: vetoed %d/%d member(s) (floor_t=%.2f k=%d): %s",
                len(_vetoed), len(artist_indices), _floor,
                int(cfg.pier_bridgeability_k),
                [
                    f"{bundle.track_ids[i]} kth-T={t:.3f}"
                    for i, t in sorted(_vetoed, key=lambda p: p[1])[:10]
                ],
            )
        if not bridgeable_set:
            logger.warning(
                "Pier bridgeability: ALL %d members failed floor_t=%.2f — running "
                "unchecked for this generation (a playlist never fails on a soft axis)",
                len(artist_indices), _floor,
            )
            bridgeable_set = None
```

3d. Before the `for ci, (c, members_local) in enumerate(nonempty):` medoid loop (~line 726), compute the reallocation bump (add `import math` to the module imports if not present):

```python
    eff_top_k = medoid_top_k
    if bridgeable_set is not None:
        n_eligible_clusters = sum(
            1 for _c, _ml in nonempty if any(m in bridgeable_set for m in _ml)
        )
        if target_pier_count and 0 < n_eligible_clusters < len(nonempty):
            eff_top_k = max(
                medoid_top_k, math.ceil(int(target_pier_count) / n_eligible_clusters)
            )
            logger.warning(
                "Pier bridgeability: %d/%d cluster(s) have no eligible member — "
                "bumping per-cluster medoids to %d to reallocate slots",
                len(nonempty) - n_eligible_clusters, len(nonempty), eff_top_k,
            )
```

3e. Inside the loop, right after `clusters.append(members_local)`, resolve eligible members and skip empty clusters; then use `members_eligible` (NOT `members_local`) for the energy/popularity/tag slices and the `_medoids_for_cluster` call, and `eff_top_k` as the `per_cluster` argument (the 5th positional). The `top_k` argument (7th positional) stays `medoid_top_k`:

```python
        members_eligible = (
            members_local if bridgeable_set is None
            else [m for m in members_local if m in bridgeable_set]
        )
        if not members_eligible:
            logger.warning(
                "Pier bridgeability: cluster %d has 0/%d eligible members — "
                "contributes no piers; slots reallocate to passing clusters",
                ci, len(members_local),
            )
            medoids_by_cluster.append([])
            continue
```

and change the existing slice/call lines from `members_local` to `members_eligible`:

```python
        if slot_targets is not None:
            member_energy = np.asarray(energy_values, dtype=float)[members_eligible]
            energy_prox = _slot_proximity(member_energy, slot_targets[ci], span_width)
        pop_slice = None
        if popularity_values is not None and cfg.medoid_popularity_weight > 0:
            pop_slice = np.asarray(popularity_values, dtype=float)[members_eligible]
        tag_slice: Optional[np.ndarray] = None
        _xgd = getattr(bundle, "X_genre_dense", None)
        if steering_target is not None and _xgd is not None and cfg.medoid_tag_weight > 0:
            tag_slice = np.asarray(_xgd, dtype=float)[members_eligible] @ np.asarray(
                steering_target, dtype=float
            )
        medoid_list = _medoids_for_cluster(
            X_norm,
            members_eligible,
            centroids[c],
            track_ids,
            eff_top_k,
            rng,
            medoid_top_k,
            ...  # remaining args unchanged
        )
```

(Keep the full existing argument list; only the two names shown change. `centroids[c]` stays the full-cluster centroid.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -q` (timeout 300s)
Expected: full file passes, including all pre-existing tests (the no-veto guard proves determinism; if any pre-existing test fails, the veto changed behavior it must not — fix before proceeding, do not adjust the old test).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): bridgeability veto at medoid selection with slot reallocation"
```

---

### Task 3: Caller wiring + config keys

**Files:**
- Modify: `src/playlist_generator.py:1712` (ArtistStyleConfig parse #1), `src/playlist_generator.py:1883` (call site #1), `src/playlist_generator.py:2786` (parse #2), `src/playlist_generator.py:2859` (call site #2)
- Modify: `config.example.yaml` (artist_style block) — **subject to the shared-checkout guard in Global Constraints**
- Test: `tests/test_artist_style.py` (append)

**Interfaces:**
- Consumes: Task 2's config fields and `target_pier_count` param.
- Produces: config keys `playlists.ds_pipeline.artist_style.pier_bridgeability_{enabled,floor_t,k}` resolvable from `config.yaml`.

- [ ] **Step 1: Write the failing test**

The parse blocks are inline in the 4.1k-LOC orchestrator, so guard the contract at the seam we control — the raw-dict key names. Append to `tests/test_artist_style.py`:

```python
def test_bridgeability_config_keys_parse():
    """Guards the exact key names playlist_generator.py parses (both sites)."""
    raw = {"pier_bridgeability_enabled": False,
           "pier_bridgeability_floor_t": 0.42,
           "pier_bridgeability_k": 7}
    cfg = ArtistStyleConfig(
        pier_bridgeability_enabled=bool(raw.get("pier_bridgeability_enabled", True)),
        pier_bridgeability_floor_t=float(raw.get("pier_bridgeability_floor_t", 0.30)),
        pier_bridgeability_k=int(raw.get("pier_bridgeability_k", 10)),
    )
    assert (cfg.pier_bridgeability_enabled, cfg.pier_bridgeability_floor_t,
            cfg.pier_bridgeability_k) == (False, 0.42, 7)
```

Run: `python -m pytest tests/test_artist_style.py -q -k config_keys_parse` (timeout 120s)
Expected: PASS immediately (it exercises Task 2's fields). It exists to break loudly if anyone renames the fields out from under the parse blocks below.

- [ ] **Step 2: Wire both ArtistStyleConfig parse sites**

At `src/playlist_generator.py:1712` (inside `_maybe_generate_ds_playlist`) and `:2786` (inside the artist-mode entry), append to the `ArtistStyleConfig(...)` kwargs — identical three lines at both sites:

```python
            pier_bridgeability_enabled=bool(style_cfg_raw.get("pier_bridgeability_enabled", True)),
            pier_bridgeability_floor_t=float(style_cfg_raw.get("pier_bridgeability_floor_t", 0.30)),
            pier_bridgeability_k=int(style_cfg_raw.get("pier_bridgeability_k", 10)),
```

- [ ] **Step 3: Pass target_pier_count at both call sites**

Both call sites already have a local `target_pier_count` in scope (line ~1794 and ~2842). Add to the `cluster_artist_tracks(...)` kwargs at `:1883` and `:2859`:

```python
                    target_pier_count=target_pier_count,
```

- [ ] **Step 4: Document the knobs in config.example.yaml (shared-checkout guard applies)**

Run `git diff -- config.example.yaml` first. If it shows foreign hunks, SKIP this step, leave a `[deferred]` note in the final report, and continue — the dataclass defaults keep the feature live without it. Otherwise add under the `artist_style:` block:

```yaml
    # Pier bridgeability veto (artist mode; fire piers exempt). A track can
    # only seat as a pier if it has >= pier_bridgeability_k non-seed-artist
    # library neighbors at calibrated T >= pier_bridgeability_floor_t (same
    # units as tail_dp_floor / repair t_floor). Clusters whose members all
    # fail contribute no piers; their slots reallocate. Motivating incident:
    # docs/superpowers/specs/2026-07-03-pier-bridgeability-design.md.
    pier_bridgeability_enabled: true
    pier_bridgeability_floor_t: 0.30
    pier_bridgeability_k: 10
```

- [ ] **Step 5: Run the full artist-style file + lint/type gates**

Run (each bounded, no pipes):
- `python -m pytest tests/test_artist_style.py -q` (timeout 300s) — expected: all pass
- `ruff check src/playlist/artist_style.py src/playlist_generator.py` — expected: clean
- `mypy src/playlist/artist_style.py` — expected: no new errors vs master (run `git stash && mypy src/playlist/artist_style.py && git stash pop` if a baseline is needed)

- [ ] **Step 6: Commit**

```bash
git add src/playlist_generator.py tests/test_artist_style.py
# plus config.example.yaml ONLY if Step 4 ran and git diff shows solely our hunk
git commit -m "feat(artist-style): wire pier bridgeability knobs through both artist-mode entries"
```

---

### Task 4: Full-suite gate + live verify (Torrey)

**Files:**
- None created; verification only. (Fix-forward commits allowed if the suite or live run exposes a defect.)

**Interfaces:**
- Consumes: everything above, plus the live artifact `data/artifacts/beat3tower_32k/data_matrices_step1.npz` and `config.yaml`.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q -m "not slow"` (timeout 600s)
Expected: same pass/fail profile as master baseline plus the new tests. Quote real counts. If unrelated failures exist from the other session's in-flight work, list them explicitly — do not claim they're pre-existing without checking `git stash` is NOT involved (never stash others' work; instead run `git stash list` only to confirm you created nothing).

- [ ] **Step 2: Live verify — regenerate Torrey in artist mode**

This is the incident's exact path (artist mode piers come from DB-clustering, which the gui_fidelity harness does NOT cover — worker/CLI required):

Run: `python main_app.py --artist "Torrey" --tracks 50` (timeout 600s)

Then inspect the newest `logs/playlists/*_Torrey_*.log`:
- `grep "Pier bridgeability" <log>` — expect veto lines naming `26504`-era outliers; specifically First Jam (`5f13a0dd70178e952b2cde223f8ff1c9`) and likely Garage Intermission (`29638484771c5fb80a3466a4ceccb3e6`) with kth-T well under 0.30.
- `grep "seed order" <log>` — neither track id appears as a pier.
- Weakest-transitions block: min T materially above 0.002 (the two catastrophic edges were pier-adjacent; with the outlier piers gone, expect min ≳ 0.2; report the actual number, don't assert a hard bar).
- Pier count: still `target_piers` Torrey tracks (reallocation held presence) — compare "Artist presence pier calculation: target_piers=N" vs seed-artist track count in the report.

If generating through the GUI instead: restart `serve_web.py` first (worker caches the module).

- [ ] **Step 3: Report**

State plainly: suite counts, veto lines observed, old vs new min-T, pier count held or not. If the live run contradicts the unit tests (e.g. veto fires but min-T stays ~0), STOP and diagnose from the per-playlist log before any tuning — do not adjust `floor_t`/`k` on speculation.

---

## Self-review notes (completed at planning time)

- Spec coverage: signal (T1), placement/gating + candidacy-only + never-fail (T2), reallocation (T2 3d/3e + caller cap), fire exemption (structural: `clusters` untouched, verified by T2 test 1 asserting `len(clusters) == 2`), config + both parse sites + example yaml (T3), logging (T2 3c/3d/3e), tests incl. live verify + golden note (T4; goldens shift by design — the golden replay gate guards the perf work, not this feature).
- Type consistency: `compute_pier_bridgeability` keyword calib args match between T1 def and T2 3c call; `target_pier_count: Optional[int]` matches T3 call sites; config field names match T3 parse lines and the defaults test.
- Determinism: no-veto path proven byte-identical by `test_bridgeability_no_veto_is_byte_identical`; rng draw order only changes when a cluster is skipped (already output-changing by intent).
