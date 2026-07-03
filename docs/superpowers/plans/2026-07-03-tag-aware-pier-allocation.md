# Tag-aware Pier Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make artist-mode piers honor the selected genre tags by distributing pier slots across sonic clusters weighted by each cluster's tag affinity (soft skew), instead of allocating them evenly.

**Architecture:** A pure allocator `allocate_piers_by_tag_affinity` (in `artist_style.py`) apportions `target_pier_count` slots across clusters — blend of uniform and normalized affinity, floor 1 per non-empty cluster, cap = cluster's medoid count. `create_playlist_for_artist` computes per-cluster affinity from `X_genre_dense @ steering_target`, over-produces medoids when steering, calls the allocator, then orders as today. When there are no tags, the existing even-allocation path runs unchanged (byte-identical).

**Tech Stack:** Python 3.11 (numpy, pytest). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-03-tag-aware-pier-allocation-design.md` (approved 2026-07-03).

## Global Constraints

- **Shared working tree, concurrent sessions live.** Commit with EXPLICIT PATHSPEC only: `git commit -m "..." -- <exact paths>`. For a NEW (untracked) file, `git add <path>` first, THEN `git commit -m "..." -- <path>`. NEVER `git add -A`/`-u`/`.`, NEVER a bare `git commit`. Commit ONLY the files a task changed.
- **`config.yaml` is GITIGNORED** — edit it (mirror any new key) but never stage/commit it. Only `config.example.yaml` is committed.
- **Inert when off, byte-identical when unused.** With no tags (`steering_target is None`), the pier-allocation path must not run — the existing even-allocation + truncate at `playlist_generator.py:1900-1909` stays exactly as-is. Guard every new branch on `steering_target is not None`.
- **No hard gates / never-fail.** The allocator only redistributes existing pier slots; it never excludes candidates from generation and never raises on normal inputs. Floor-1-per-cluster preserves the arc; cap-at-size preserves feasibility.
- **Pytest:** run `python -m pytest -q <path>` directly with the Bash tool's timeout parameter. NEVER pipe pytest through `tail`/`head` (a hook blocks it and it has hung sessions).
- **`data/metadata.db` is production — read-only.** Any test that clusters a real artist opens the DB read-only; never write/migrate it.
- **Commit message prefix:** `feat(tag-steering): ...` / `test(tag-steering): ...`.

## File Structure

| File | Change |
|---|---|
| `src/playlist/artist_style.py` | NEW pure fn `allocate_piers_by_tag_affinity(...)` |
| `src/playlist_generator.py` | over-produce `medoid_top_k` when steering (`:1803-1809`); replace even-allocate/truncate (`:1900-1909`) with affinity-computation + allocator call + log; read `pier_tag_skew` from `ds_cfg["pier_bridge"]` |
| `config.example.yaml` + `config.yaml` (gitignored) | `pier_tag_skew: 0.6` under `playlists.ds_pipeline.pier_bridge` |
| `tests/unit/test_pier_tag_allocation.py` | NEW — allocator unit tests |
| `tests/integration/test_pier_tag_allocation_live.py` | NEW — cluster a real artist + allocator, assert skew |

---

### Task 1: Pure allocator `allocate_piers_by_tag_affinity`

**Files:**
- Modify: `src/playlist/artist_style.py` (add the function; place it near `select_popular_piers`, ~`:513`)
- Test: `tests/unit/test_pier_tag_allocation.py` (new)

**Interfaces:**
- Consumes: nothing from earlier tasks (pure function).
- Produces: `allocate_piers_by_tag_affinity(medoids_by_cluster: list[list[int]], cluster_affinities: list[float], target_pier_count: int, skew: float) -> list[int]` — returns selected bundle indices (unordered). Task 2 calls this exact signature.

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for tag-weighted pier allocation across clusters."""
from src.playlist.artist_style import allocate_piers_by_tag_affinity


def _counts(selected, medoids_by_cluster):
    """How many selected piers came from each cluster (clusters are disjoint index sets)."""
    sets = [set(m) for m in medoids_by_cluster]
    return [sum(1 for i in selected if i in s) for s in sets]


def test_skew_zero_is_balanced_ignoring_affinity():
    # 3 clusters, distinct indices, high/mid/low affinity. skew=0 => even split, affinity ignored.
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=0.0)
    assert len(sel) == 6
    assert _counts(sel, mbc) == [2, 2, 2]


def test_skew_one_favors_high_affinity_cluster():
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=1.0)
    # Floor 1 each, remaining 3 flow to the highest-affinity cluster.
    assert _counts(sel, mbc) == [4, 1, 1]
    # High-affinity cluster contributes its TOP tag-ranked medoids (list order preserved).
    assert sel[:4] == [0, 1, 2, 3]


def test_soft_skew_between():
    mbc = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.5, 0.1], target_pier_count=6, skew=0.6)
    c = _counts(sel, mbc)
    assert sum(c) == 6
    assert c[0] > c[2]           # skews toward high affinity
    assert min(c) >= 1           # floor preserved (arc holds)


def test_cap_at_cluster_size():
    # High-affinity cluster is tiny (size 1); it cannot exceed its size even at skew=1.
    mbc = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.4, 0.1], target_pier_count=6, skew=1.0)
    assert len(sel) == 6
    assert _counts(sel, mbc)[0] == 1      # capped at size


def test_floor_per_nonempty_cluster_when_p_ge_k():
    mbc = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.1, 0.1, 0.1], target_pier_count=5, skew=1.0)
    assert min(_counts(sel, mbc)) >= 1    # every cluster represented


def test_p_less_than_k_gives_floor_to_top_weight_clusters():
    mbc = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.8, 0.1, 0.05], target_pier_count=2, skew=1.0)
    assert len(sel) == 2
    c = _counts(sel, mbc)
    assert c[0] == 1 and c[1] == 1 and c[2] == 0 and c[3] == 0


def test_few_tracks_takes_all():
    mbc = [[0, 1], [2]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.9, 0.1], target_pier_count=10, skew=0.6)
    assert sorted(sel) == [0, 1, 2]       # total_available <= P => everything


def test_empty_input():
    assert allocate_piers_by_tag_affinity([], [], target_pier_count=10, skew=0.6) == []
    assert allocate_piers_by_tag_affinity([[], []], [0.0, 0.0], target_pier_count=10, skew=0.6) == []


def test_all_equal_affinity_is_uniform():
    mbc = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    sel = allocate_piers_by_tag_affinity(mbc, [0.5, 0.5, 0.5], target_pier_count=6, skew=1.0)
    assert _counts(sel, mbc) == [2, 2, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_pier_tag_allocation.py`
Expected: FAIL — `ImportError: cannot import name 'allocate_piers_by_tag_affinity'`

- [ ] **Step 3: Implement** (add to `src/playlist/artist_style.py`, near `select_popular_piers`)

```python
def allocate_piers_by_tag_affinity(
    medoids_by_cluster: list[list[int]],
    cluster_affinities: list[float],
    target_pier_count: int,
    skew: float,
) -> list[int]:
    """Distribute ``target_pier_count`` pier slots across clusters, skewed toward
    high tag-affinity clusters (soft). Each cluster's medoid list is already
    tag-ranked (``_medoids_for_cluster``), so we take its top ``n_c``.

    ``skew=0`` -> uniform across clusters (affinity ignored); ``skew=1`` -> pure
    affinity weighting. A floor of 1 pier per non-empty cluster preserves the sonic
    arc; the cap is each cluster's available medoid count. Returns the selected
    bundle indices (unordered; the caller orders them).
    """
    sizes = [len(m) for m in medoids_by_cluster]
    nonempty = [c for c, s in enumerate(sizes) if s > 0]
    total_available = sum(sizes)
    P = int(target_pier_count)
    if P <= 0 or not nonempty:
        return []
    # Few tracks: everything becomes a pier (matches legacy under-target behavior).
    if total_available <= P:
        return [i for cluster in medoids_by_cluster for i in cluster]

    K = len(nonempty)
    affs = [float(cluster_affinities[c]) for c in nonempty]
    amin, amax = min(affs), max(affs)
    if amax > amin:
        norm = {c: (float(cluster_affinities[c]) - amin) / (amax - amin) for c in nonempty}
    else:
        norm = {c: 0.5 for c in nonempty}
    nsum = sum(norm.values()) or 1.0
    s = float(skew)
    weight = {c: (1.0 - s) * (1.0 / K) + s * (norm[c] / nsum) for c in nonempty}

    alloc = {c: 0 for c in nonempty}
    # Floor: 1 per non-empty cluster, highest-weight first (handles P < K gracefully).
    for c in sorted(nonempty, key=lambda c: (weight[c], -c), reverse=True):
        if sum(alloc.values()) >= P:
            break
        alloc[c] = 1
    # Fill: each remaining slot goes to the cluster furthest below its weighted
    # target (weight[c] * P), respecting each cluster's available-medoid cap.
    while sum(alloc.values()) < P:
        cands = [c for c in nonempty if alloc[c] < sizes[c]]
        if not cands:
            break
        c = max(cands, key=lambda c: (weight[c] * P - alloc[c], weight[c], -c))
        alloc[c] += 1

    selected: list[int] = []
    for c in nonempty:
        selected.extend(medoids_by_cluster[c][: alloc[c]])
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_pier_tag_allocation.py`
Expected: 9 passed. If `test_skew_one_favors_high_affinity_cluster` disagrees on the exact split, DO NOT weaken the assertion — re-derive the apportionment by hand from the code and, if the code is correct but the expected counts were mis-derived, fix the EXPECTED value in the test to match the code's deterministic output (document the hand-derivation in the commit message). The algorithm is the source of truth; the specific `[4,1,1]` is a hand-computed expectation of it.

- [ ] **Step 5: Run ruff on the changed file**

Run: `python -m ruff check src/playlist/artist_style.py tests/unit/test_pier_tag_allocation.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/playlist/artist_style.py tests/unit/test_pier_tag_allocation.py
git commit -m "feat(tag-steering): pure tag-weighted pier allocator" -- src/playlist/artist_style.py tests/unit/test_pier_tag_allocation.py
```

---

### Task 2: Wire the allocator into artist-mode pier selection + config

**Files:**
- Modify: `src/playlist_generator.py` (`create_playlist_for_artist`: `medoid_top_k` override ~`:1803-1809`; replace even-allocate/truncate at `:1900-1909`)
- Modify: `config.example.yaml` (add `pier_tag_skew: 0.6`) and `config.yaml` (gitignored — mirror the key, do NOT stage)
- Test: `tests/integration/test_pier_tag_allocation_live.py` (new)

**Interfaces:**
- Consumes: `allocate_piers_by_tag_affinity(medoids_by_cluster, cluster_affinities, target_pier_count, skew)` (Task 1); `resolve_tag_steering_target` (Stage 1); `cluster_artist_tracks` returning `(clusters, medoids, medoids_by_cluster, X_norm)`; `steering_target` already in scope in `create_playlist_for_artist` (resolved after `load_artifact_bundle`, Stage 1 Task 5).
- Produces: config key `playlists.ds_pipeline.pier_bridge.pier_tag_skew` (float, default 0.6); the INFO log line `Tag steering pier allocation: ...`.

- [ ] **Step 1: Over-produce medoids when steering.** In `create_playlist_for_artist`, immediately AFTER the `medoid_top_k` if/else block (ends `:1809`), add:

```python
                # Tag steering: over-produce medoids per cluster so the tag-weighted
                # allocator (below) has enough tag-ranked candidates to reallocate.
                if steering_target is not None:
                    medoid_top_k = max(medoid_top_k, target_pier_count)
```

- [ ] **Step 2: Replace the even-allocate/truncate block.** Replace the block at `:1900-1909` (currently `ordered_medoids = order_clusters(medoids, X_norm)` followed by the `if len(ordered_medoids) > target_pier_count:` cap) with:

```python
                _xgd = getattr(bundle, "X_genre_dense", None)
                if (
                    steering_target is not None
                    and popular_seeds_mode != "fire"
                    and _xgd is not None
                ):
                    # Tag-weighted pier allocation: skew slots toward on-tag clusters
                    # (soft; floor 1 per cluster keeps the arc). Off-tag clusters
                    # (e.g. an artist's interludes) contribute fewer piers.
                    _xgd = np.asarray(_xgd, dtype=float)
                    _tgt = np.asarray(steering_target, dtype=float)
                    cluster_affinities = [
                        float(np.mean(_xgd[members] @ _tgt)) if len(members) else 0.0
                        for members in clusters
                    ]
                    pier_tag_skew = float(
                        (ds_cfg.get("pier_bridge", {}) or {}).get("pier_tag_skew", 0.6)
                    )
                    selected = allocate_piers_by_tag_affinity(
                        medoids_by_cluster, cluster_affinities, target_pier_count, pier_tag_skew,
                    )
                    ordered_medoids = order_clusters(selected, X_norm)
                    logger.info(
                        "Tag steering pier allocation: skew=%.2f cluster_affinities=%s selected=%d/%d",
                        pier_tag_skew,
                        [round(a, 3) for a in cluster_affinities],
                        len(selected), len(medoids),
                    )
                else:
                    ordered_medoids = order_clusters(medoids, X_norm)
                    # Cap medoids to target_pier_count to avoid ceiling overshoot
                    # (e.g., 5 clusters × ceil(6/5)=2 per cluster = 10, but we want 6)
                    if len(ordered_medoids) > target_pier_count:
                        logger.info(
                            "Capping medoids from %d to target_pier_count=%d",
                            len(ordered_medoids), target_pier_count,
                        )
                        ordered_medoids = ordered_medoids[:target_pier_count]
```

Then update the import at the top of `playlist_generator.py` where `artist_style` symbols are imported (`select_popular_piers`, `cluster_artist_tracks`, `order_clusters`, `build_balanced_candidate_pool`, ...) to also import `allocate_piers_by_tag_affinity`. Grep `from src.playlist.artist_style import` (or `from .playlist.artist_style import`) to find the existing import line and add the name.

- [ ] **Step 3: Add the config key.** In `config.example.yaml`, under `playlists: → ds_pipeline: → pier_bridge:` (next to `tag_steering_pier_weight`), add:

```yaml
      # Tag steering: how hard artist-mode pier selection skews toward on-tag sonic
      # clusters. 0 = even across clusters (legacy); 1 = pure affinity weighting.
      # Soft skew keeps every cluster represented (floor 1 pier). Inert without tags.
      pier_tag_skew: 0.6
```

Mirror the same key in `config.yaml` (gitignored — edit it so live runs pick it up, but do NOT `git add` it).

- [ ] **Step 4: Write the integration test** (`tests/integration/test_pier_tag_allocation_live.py`)

This clusters a REAL artist through the actual `cluster_artist_tracks` (the faithful mechanism) and asserts the allocator's skew raises the piers' mean tag affinity. It does NOT drive full `create_playlist_for_artist` (that needs the worker-tier harness, which is not built — the config-plumbing + `None`-guard are covered by Step 6's live acceptance).

```python
"""Behavioral: tag-weighted pier allocation raises the anchors' tag affinity.

Clusters a real artist via the production cluster_artist_tracks, then compares
the allocator at skew=0.6 vs skew=0.0 on the real medoids/affinities.
"""
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path  # noqa: E402

from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.artist_style import (  # noqa: E402
    ArtistStyleConfig,
    allocate_piers_by_tag_affinity,
    cluster_artist_tracks,
)
from src.playlist.tag_steering import resolve_tag_steering_target  # noqa: E402

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB = "data/metadata.db"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    resolved_artifact_path()  # publishes sonic_variant_override (muq) as a side effect
    return load_artifact_bundle(str(ARTIFACT))


def test_skew_raises_pier_affinity(bundle):
    if getattr(bundle, "genre_emb", None) is None:
        pytest.skip("dense genre sidecar absent")
    target, mapped, _ = resolve_tag_steering_target(
        ["jangle pop", "indie rock"],
        genre_vocab=[str(v) for v in bundle.genre_vocab],
        genre_emb=bundle.genre_emb,
    )
    if target is None or len(mapped) == 0:
        pytest.skip("steering tags did not map in this artifact vocab")

    # Minimal-but-real ArtistStyleConfig: enable clustering + the tag lever. Read the
    # dataclass definition in artist_style.py and set only the fields without defaults
    # plus medoid_tag_weight=0.3; leave the rest at their dataclass defaults.
    style_cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)

    clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle,
        artist_name="Real Estate",
        cfg=style_cfg,
        random_seed=0,
        medoid_top_k=10,                 # over-produce, mirrors the steering path
        steering_target=target,
        metadata_db_path=DB,
    )
    xgd = np.asarray(bundle.X_genre_dense, dtype=float)
    tgt = np.asarray(target, dtype=float)
    affs = [float(np.mean(xgd[m] @ tgt)) if len(m) else 0.0 for m in clusters]

    sel_skew = allocate_piers_by_tag_affinity(medoids_by_cluster, affs, 10, 0.6)
    sel_flat = allocate_piers_by_tag_affinity(medoids_by_cluster, affs, 10, 0.0)

    def mean_aff(ids):
        return float(np.mean([xgd[i] @ tgt for i in ids])) if ids else 0.0

    assert mean_aff(sel_skew) >= mean_aff(sel_flat), (
        f"skew mean affinity {mean_aff(sel_skew):.3f} < flat {mean_aff(sel_flat):.3f} — "
        "read the per-cluster affinities before concluding the skew is inert"
    )
    # The lowest-affinity cluster should give up slots under skew vs flat.
    def counts(sel):
        sets = [set(m) for m in medoids_by_cluster]
        return [sum(1 for i in sel if i in s) for s in sets]
    low = int(np.argmin(affs))
    assert counts(sel_skew)[low] <= counts(sel_flat)[low]
```

If `ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)` raises for missing required fields, read the dataclass and add just those fields with sensible values — do not hand-tune sonic params, the test only needs clustering to run.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q tests/integration/test_pier_tag_allocation_live.py` (Bash timeout 600000)
Expected: 1 passed (or `skipped` if the artifact/sidecar is absent — but on this checkout the sidecar was rebuilt 2026-07-02, so it should RUN).
Run: `python -m pytest -q tests/unit/test_pier_tag_allocation.py`
Expected: 9 passed (Task 1 still green).
Run: `python -m ruff check src/playlist_generator.py tests/integration/test_pier_tag_allocation_live.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit** (config.yaml is gitignored — do NOT stage it)

```bash
git add src/playlist_generator.py config.example.yaml tests/integration/test_pier_tag_allocation_live.py
git commit -m "feat(tag-steering): tag-weighted pier allocation in artist mode + pier_tag_skew knob" -- src/playlist_generator.py config.example.yaml tests/integration/test_pier_tag_allocation_live.py
```

- [ ] **Step 7: Full fast-suite regression gate**

Run: `python -m pytest -q -m "not slow" -p no:cacheprovider` (Bash timeout 600000; NO pipe)
Expected: quote the real pass/fail counts. Pre-existing failures unrelated to this change (synthetic `ds_pipeline_smoke` / `playlist_golden_files` track-count goldens from a concurrent session's `edge_delete` work, per the ledger) are NOT ours — verify any failure is pre-existing (present without this task's diff) before attributing it here; do not attribute or "fix" them.

- [ ] **Step 8: Live acceptance (hand to Dylan — needs the browser).** The automated tests cover the allocator (Task 1) and the clustering+allocator mechanism (Step 4); the config-plumbing + `None`-guard are verified live. Restart the worker first (`@lru_cache` holds the bundle), then re-run **Real Estate + `jangle pop`/`indie rock`** in artist mode and confirm in the newest `logs/playlists/*.log`: the `Tag steering pier allocation: skew=0.60 ...` line appears, and the `Tag steering piers: affinity per selected pier` values are higher than the 2026-07-03 baseline (which had four ~0.385 piers). Also generate the SAME artist with NO tags and confirm the allocation line is ABSENT (byte-identical legacy path).

---

## Self-Review

- **Spec coverage:** §1 architecture → Task 2 Steps 1-2. §2 algorithm → Task 1 (allocator) + its unit tests. §3 inertness/arc/never-fail → Task 2 Step 2 `None`/`fire` guards + Task 1 floor/cap tests + Task 2 Step 8 no-tag check. §4 config/composition → Task 2 Steps 1-3 (`pier_tag_skew`, fire skip, popular-on composition via unchanged within-cluster weighting). §5 logging → Task 2 Step 2 log line. §6 testing → Task 1 unit + Task 2 Step 4 integration + Step 8 live. All spec sections map to a task.
- **Placeholder scan:** none — all code blocks are complete; the one judgment call (exact `[4,1,1]` split) has an explicit hand-derivation fallback.
- **Type consistency:** `allocate_piers_by_tag_affinity(medoids_by_cluster, cluster_affinities, target_pier_count, skew) -> list[int]` is identical in Task 1 (def), Task 2 (call), and the integration test.
