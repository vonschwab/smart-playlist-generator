# Bridge-side Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Guarantee representative on-tag tracks appear by injecting up to K of them as **piers** (anchors), selected to be bridgeable to a seed pier, tag-central, and cross-artist-diverse.

**Architecture:** One pure selection function + one injection point in the artist-mode pier block. Anchors are appended to `ordered_medoids` (the pier set); the existing pier resolution/ordering treats them as ungated, un-droppable piers (multi-artist playlists already do this). All gated on tag steering active.

**Tech Stack:** Python 3.11, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-bridge-side-phase-b.md`. **Background:** `docs/superpowers/specs/2026-07-08-tag-steering-architecture.md`; Phase A spec/result.

## Global Constraints

- **Gated on tag steering active** (`_on_tag_track_ids` non-empty). Non-steered → byte-identical.
- **On-tag = authority membership** (`_on_tag_track_ids`, already computed in the artist block ~1807-1828). No new genre read.
- **Live default with rollback** (#22): `tag_steering_anchor_max` default 3; `0` = Phase-A-only.
- **No silent no-op:** if anchors requested but none clear the bridge floor, log INFO and inject none (graceful).
- **Tests through the real `PlaylistGenerator`** for generation (pier-fix precedent).
- **Sub-agent models:** Task 1 haiku (pure fn, complete code), Tasks 2-3 sonnet (integration + validation). Never inherit the session model.
- **Shared checkout:** explicit-path commits only; verify `git diff --cached --name-only`.

## File Structure

- `src/playlist/tag_steering.py` — add pure `select_on_tag_anchors(...)`.
- `src/playlist_generator.py` — inject anchors into `ordered_medoids` in the artist-mode block (~2113); read config knobs.
- `config.example.yaml` — 3 new knobs.
- `tests/unit/test_on_tag_anchors.py` — new unit tests (Task 1).
- `tests/integration/test_gui_fidelity_regressions.py` — integration cases (Task 3).

---

### Task 1: Pure `select_on_tag_anchors`

**Files:** Modify `src/playlist/tag_steering.py`. Test: `tests/unit/test_on_tag_anchors.py`.

**Interfaces:** `select_on_tag_anchors(on_tag_indices, pier_indices, X_sonic, tag_centrality, artist_keys, track_ids, *, max_anchors, min_bridge, per_artist) -> list[int]` — bundle rows to inject as anchors. `tag_centrality` is a bundle-aligned array (the caller passes `sonic_tag_affinity`) or `None` (fall back to bridge strength).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_on_tag_anchors.py
import numpy as np
from src.playlist.tag_steering import select_on_tag_anchors


def test_selects_bridgeable_tag_central_diverse():
    # 6 tracks. piers = [0]. on-tag = [1,2,3,4,5].
    # track 5 is an ISLAND (far from pier 0) -> excluded by min_bridge.
    X = np.array([
        [1.0, 0.0, 0.0],   # 0 pier
        [0.9, 0.1, 0.0],   # 1 bridgeable, artist A
        [0.85, 0.2, 0.0],  # 2 bridgeable, artist A
        [0.8, 0.3, 0.0],   # 3 bridgeable, artist B
        [0.75, 0.4, 0.0],  # 4 bridgeable, artist C
        [0.0, 0.0, 1.0],   # 5 ISLAND, artist D
    ], dtype=np.float64)
    tag_centrality = np.array([0.0, 0.9, 0.8, 0.7, 0.6, 0.99])  # 5 most central but island
    artist_keys = np.array(["p", "a", "a", "b", "c", "d"])
    track_ids = np.array([f"t{i}" for i in range(6)])
    got = select_on_tag_anchors(
        on_tag_indices=[1, 2, 3, 4, 5], pier_indices=[0], X_sonic=X,
        tag_centrality=tag_centrality, artist_keys=artist_keys, track_ids=track_ids,
        max_anchors=3, min_bridge=0.5, per_artist=1,
    )
    assert 5 not in got                 # island excluded (max sim to pier 0 = 0.0 < 0.5)
    assert got == [1, 3, 4]             # per-artist cap 1: A->only track1 (higher centrality), then B(3), C(4); capped at 3


def test_empty_when_no_bridgeable():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert select_on_tag_anchors(
        on_tag_indices=[1], pier_indices=[0], X_sonic=X, tag_centrality=None,
        artist_keys=np.array(["p", "a"]), track_ids=np.array(["t0", "t1"]),
        max_anchors=3, min_bridge=0.5, per_artist=1,
    ) == []


def test_max_zero_returns_empty():
    X = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64)
    assert select_on_tag_anchors(
        on_tag_indices=[1], pier_indices=[0], X_sonic=X, tag_centrality=None,
        artist_keys=np.array(["p", "a"]), track_ids=np.array(["t0", "t1"]),
        max_anchors=0, min_bridge=0.5, per_artist=1,
    ) == []
```

- [ ] **Step 2: Run to verify it fails**
Run: `python -m pytest tests/unit/test_on_tag_anchors.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/playlist/tag_steering.py
def select_on_tag_anchors(
    on_tag_indices,
    pier_indices,
    X_sonic,
    tag_centrality,
    artist_keys,
    track_ids,
    *,
    max_anchors: int,
    min_bridge: float,
    per_artist: int,
) -> list:
    """Pick up to ``max_anchors`` on-tag tracks to inject as piers: each must be bridgeable
    to a seed pier (max L2-normalized sonic cosine to any ``pier_indices`` >= ``min_bridge``),
    ranked by ``tag_centrality`` desc (or bridge strength when None), capped at ``per_artist``
    per normalized artist key. Pure. [] when max_anchors<=0 or nothing is bridgeable."""
    from collections import Counter
    if int(max_anchors) <= 0 or not on_tag_indices or not pier_indices:
        return []
    X = np.asarray(X_sonic, dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    pier_mat = Xn[[int(p) for p in pier_indices]]
    scored = []
    for i in on_tag_indices:
        i = int(i)
        bridge = float(np.max(pier_mat @ Xn[i]))
        if bridge < float(min_bridge):
            continue
        rank = float(tag_centrality[i]) if tag_centrality is not None else bridge
        scored.append((rank, bridge, i))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    per: Counter = Counter()
    out = []
    for _rank, _bridge, i in scored:
        if len(out) >= int(max_anchors):
            break
        ak = str(artist_keys[i])
        if per[ak] >= int(per_artist):
            continue
        out.append(i)
        per[ak] += 1
    return out
```

- [ ] **Step 4: Run to verify it passes** → PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add src/playlist/tag_steering.py tests/unit/test_on_tag_anchors.py
git commit --only -- src/playlist/tag_steering.py tests/unit/test_on_tag_anchors.py -m "feat(tag-steering): select_on_tag_anchors (bridgeable + tag-central + diverse)"
```

---

### Task 2: Inject anchors into the artist-mode pier set + config

**Files:** `src/playlist_generator.py`, `config.example.yaml`. Test: covered by Task 3.

**Interfaces:** Consumes `select_on_tag_anchors` (Task 1). Injects into `ordered_medoids` (bundle indices) after it's finalized (~2113, after `_filter_title_excluded_bundle_indices`) and BEFORE `pier_indices=ordered_medoids` is used for pools (~2140) and `pier_ids` (~2153).

- [ ] **Step 1: Config knobs** (`config.example.yaml`, under `pier_bridge:`, near the other `tag_steering_*`):
```yaml
      tag_steering_anchor_max: 3          # on-tag tracks injected as piers (0 = Phase-A-only / off)
      tag_steering_anchor_min_bridge: 0.35  # min sonic cosine to a seed pier for an anchor to qualify (no islands)
      tag_steering_anchor_per_artist: 1   # per-artist cap within the anchor set
```

- [ ] **Step 2: Inject** — immediately after the `if not ordered_medoids: raise ...` guard (~2113) and before `cluster_piers`/pool building (~2140). Read the surrounding scope to confirm the names (`_on_tag_track_ids`, `sonic_tag_affinity`, `steering_target`, `bundle`, `_pb`/`ds_cfg`, `target_pier_count`, a track_id→row map). Insert:

```python
                # Tag steering on-tag ANCHORS (Phase B): inject representative on-tag tracks as
                # piers so a sonically-peripheral clique is GUARANTEED to appear (bridges alone
                # can't place them — see the bridge-side Phase A result). Selection is bridgeable
                # + tag-central + diverse; gated on steering; capped so interiors aren't starved.
                _anchor_max = int((ds_cfg.get("pier_bridge", {}) or {}).get("tag_steering_anchor_max", 3))
                if steering_target is not None and _on_tag_track_ids and _anchor_max > 0:
                    from src.playlist.tag_steering import select_on_tag_anchors
                    _t2r_full = {str(t): i for i, t in enumerate(bundle.track_ids)}
                    _on_tag_rows = [_t2r_full[t] for t in _on_tag_track_ids if t in _t2r_full]
                    _existing = set(int(m) for m in ordered_medoids)
                    _on_tag_rows = [r for r in _on_tag_rows if r not in _existing]
                    _pbc = (ds_cfg.get("pier_bridge", {}) or {})
                    _anchors = select_on_tag_anchors(
                        on_tag_indices=_on_tag_rows,
                        pier_indices=list(ordered_medoids),
                        X_sonic=getattr(bundle, "X_sonic", None),
                        tag_centrality=sonic_tag_affinity,   # centered sonic affinity to the tag prototype (or None)
                        artist_keys=bundle.track_artists,
                        track_ids=bundle.track_ids,
                        max_anchors=_anchor_max,
                        min_bridge=float(_pbc.get("tag_steering_anchor_min_bridge", 0.35)),
                        per_artist=int(_pbc.get("tag_steering_anchor_per_artist", 1)),
                    )
                    if _anchors:
                        _cap = int(target_pier_count) + _anchor_max
                        ordered_medoids = (list(ordered_medoids) + _anchors)[:_cap]
                        logger.info(
                            "Tag steering on-tag anchors: injected %d on-tag pier(s) across %d artist(s): %s",
                            len(_anchors), len({str(bundle.track_artists[a]) for a in _anchors}),
                            [str(bundle.track_ids[a]) for a in _anchors],
                        )
                    else:
                        logger.info(
                            "Tag steering on-tag anchors: 0 bridgeable on-tag tracks (min_bridge=%.2f) — "
                            "no anchors injected (Phase-A bridges only).",
                            float(_pbc.get("tag_steering_anchor_min_bridge", 0.35)),
                        )
```
Confirm by reading: `sonic_tag_affinity` may be `None` (cohesion gate) — the fn handles it (falls back to bridge strength). `bundle.track_artists` is the artist-key source used elsewhere in this block. If a track_id→row map already exists in scope (e.g. `_t2r`), reuse it instead of rebuilding `_t2r_full`.

- [ ] **Step 3: Verify off-path + imports**
Run: `python -c "import src.playlist_generator, src.playlist.tag_steering"` → clean.
Run: `python -m pytest tests/unit/test_on_tag_anchors.py tests/unit/test_tag_first_piers.py tests/test_gui_fidelity.py -q` → PASS.
`ruff check src/playlist_generator.py src/playlist/tag_steering.py` (fix only new E/F). With no tag (`_on_tag_track_ids` empty) OR `tag_steering_anchor_max: 0`, the block is skipped → `ordered_medoids` unchanged → byte-identical.

- [ ] **Step 4: Commit**
```bash
git add src/playlist_generator.py config.example.yaml
git commit --only -- src/playlist_generator.py config.example.yaml -m "feat(tag-steering): inject on-tag anchors as piers (Phase B), gated + capped"
```

---

### Task 3: Integration validation (the payoff)

**Files:** `tests/integration/test_gui_fidelity_regressions.py`.

- [ ] **Step 1: Integration cases** (real `PlaylistGenerator`, `@pytest.mark.integration @pytest.mark.slow`, skip if artifact absent; re-read authority membership for realized pier track_ids as in the pier-fix Task-6 helpers):
  - **BoC + ["hauntology"]**: the playlist contains **≥2 non-BoC authority-hauntology tracks as piers** (the long-standing goal Phase A couldn't hit). Assert the "on-tag anchors: injected N" log fired with N≥2. Worst-edge min-T ≥ Phase-A baseline − ~0.1.
  - **Bowie + ["krautrock"]**: ≥2 authority-krautrock anchor piers present.
  - **Eno + ["neoclassical"]** (or Real Estate + ["jangle pop"]): no-regression — still on-genre, worst-edge not worse by >~1 notch; anchors are close neighbors (assert each injected anchor's max sonic cos to a seed pier ≥ the 0.35 floor, i.e. not jarring).
  - **`tag_steering_anchor_max: 0`**: realized piers = seed-only (byte-identical to Phase-A-only) — rollback guard.

- [ ] **Step 2: Run + manual verify (record real numbers, do NOT skip).**
Run: `python -m pytest tests/integration/test_gui_fidelity_regressions.py -q -k "anchor or hauntology or krautrock"` (bounded; NO head/tail pipe).
Manually regenerate BoC/hauntology + Bowie/krautrock + Eno via the worker path; count on-genre anchor piers + worst-edge; quote vs the Phase-A runs. **Report the BoC Ghost Box count explicitly** — it's the whole point.

- [ ] **Step 3: Commit**
```bash
git add tests/integration/test_gui_fidelity_regressions.py
git commit --only -- tests/integration/test_gui_fidelity_regressions.py -m "test(tag-steering): Phase B on-tag anchor integration (BoC Ghost Box, krautrock, Eno no-regression, rollback)"
```

---

## Self-Review (completed)

- **Spec coverage:** selection fn (T1), injection + config (T2), validation (T3). Covered.
- **Placeholders:** none — real code; the injection's in-scope-name confirmation is a bounded read, not a TBD.
- **Type consistency:** `select_on_tag_anchors(...) -> list[int]` (T1) ← called in T2 with `_on_tag_rows` (bundle rows), `ordered_medoids` (bundle rows), `sonic_tag_affinity` (bundle-aligned or None). Injected back into `ordered_medoids` (bundle rows). Consistent.
- **Gating:** requires `steering_target` + `_on_tag_track_ids` + `anchor_max>0` → non-steered/rollback byte-identical (asserted T3).
