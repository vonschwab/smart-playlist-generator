# Popular Seeds Three-Way (OFF / ON / 🔥) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Popular Seeds into a three-way control (OFF / ON / 🔥), where 🔥 selects piers by pure top-N popularity, ON raises the popularity weight to 1.0, and Bangers OOPS forces 🔥.

**Architecture:** A new `select_popular_piers` selector overrides the cluster-medoid piers with the artist's top-N hits in 🔥 mode (clustering still seeds the bridge pool; pier *ordering* stays). The boolean `popular_seeds` becomes a `popular_seeds_mode` string threaded through the request stack, and the pool-gate's `_resolve_popular_seeds` bool resolver is replaced by a `_resolve_popular_seeds_mode` string resolver that forces `"fire"` under OOPS.

**Tech Stack:** Python 3.11, numpy, dataclasses, pytest; React + TS (Vite) for the GUI dropdown.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-26-popular-seeds-three-way-design.md` — implements it; read first.
- **Branch / worktree:** work **inline** on `worktree-oops-bangers-poolgate` (worktree `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.claude\worktrees\oops-bangers-poolgate`). Subagents must `cd` into the worktree first, edit via absolute paths under it, and verify `git rev-parse --abbrev-ref HEAD` == `worktree-oops-bangers-poolgate` before every commit. **Do NOT** edit/commit in the main checkout or any other worktree.
- **Builds on the pool-gate work already on this branch** (commits through `e36910c`): `_resolve_popular_seeds` (bool) and its call site in `create_playlist_for_artist` exist and are **replaced** by this plan (Task 2).
- **Modes:** `"off" | "on" | "fire"`. OFF stays byte-identical to today. 🔥 constrains *selection* only — the pier-bridge still orders piers + beam-searches bridges for cohesion.
- **Recency/freshness unchanged** — applies to all three modes (it's already inside `cluster_artist_tracks`; do not touch it).
- **Never reach past hits in 🔥** — count-only fallback; zero hits → OFF clustering, logged.
- **Activate, don't legacy-default:** single-user app — *replace* the `popular_seeds` bool, don't keep it alongside.
- **Pytest:** run directly, `-q`, bounded by the tool timeout; never pipe through `tail`/`head`. Always pass a fresh basetemp:
  `--basetemp="C:/Users/Dylan/AppData/Local/Temp/claude/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3--claude-worktrees-oops-all-bangers/9557faa1-c0b6-4b3a-9fc0-25d9e1d30f43/scratchpad/pt"` (referred to as `--basetemp=$PT`).
- **No real data in the worktree** → unit tests only; **live generation verification is the user's** via the GUI.
- **Commit after every task.** End messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do not push to origin.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/playlist/artist_style.py` | +`select_popular_piers` | top-N popularity pier selection (🔥) |
| `src/playlist_generator.py` | replace `_resolve_popular_seeds`→`_resolve_popular_seeds_mode`; rename param; mode-based popularity load; 🔥 override; config default | wiring + OOPS→🔥 coupling |
| `src/playlist/request_models.py` | `popular_seeds: bool`→`popular_seeds_mode: str` | request threading |
| `src/playlist_web/schemas.py` | `popular_seeds`→`popular_seeds_mode` | API body |
| `src/playlist_gui/worker.py` | pass `popular_seeds_mode` | worker dispatch |
| `web/src/lib/types.ts` | `popular_seeds_mode` | TS type |
| `web/src/components/GenerateControls.tsx` | checkbox → dropdown | GUI |
| `config.example.yaml` | `popular_seeds_weight: 1.0` | default |

---

## Task 1: `select_popular_piers` selector

**Files:**
- Modify: `src/playlist/artist_style.py` (add near `_medoids_for_cluster`, ~line 510)
- Test: `tests/unit/test_select_popular_piers.py` (new)

**Interfaces:**
- Produces: `select_popular_piers(member_indices: list[int], popularity_values: np.ndarray, target_pier_count: int) -> list[int]` — the up-to-`target_pier_count` member indices with the highest popularity score (`load_artist_popularity_values` returns `1 - rank/n`; higher = more popular). Excludes non-finite (non-hit) scores. Ties broken by index. Returns `[]` when no member has a finite score.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_select_popular_piers.py`:

```python
import numpy as np
from src.playlist.artist_style import select_popular_piers


def test_picks_top_n_by_popularity_descending():
    # indices 0..4; popularity score = 1 - rank/n (higher = more popular)
    pv = np.array([0.10, 0.95, np.nan, 0.80, 0.50])
    piers = select_popular_piers([0, 1, 2, 3, 4], pv, target_pier_count=3)
    assert piers == [1, 3, 4]          # 0.95, 0.80, 0.50; index 2 (NaN) excluded


def test_returns_fewer_when_hits_scarce():
    pv = np.array([np.nan, 0.90, np.nan, np.nan])
    piers = select_popular_piers([0, 1, 2, 3], pv, target_pier_count=3)
    assert piers == [1]                # only one hit; never pads with non-hits


def test_returns_empty_when_no_finite_scores():
    pv = np.array([np.nan, np.nan])
    assert select_popular_piers([0, 1], pv, target_pier_count=3) == []


def test_tie_broken_by_index():
    pv = np.array([0.5, 0.5, 0.5])
    assert select_popular_piers([2, 0, 1], pv, target_pier_count=2) == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_select_popular_piers.py -q --basetemp=$PT`
Expected: FAIL with `ImportError: cannot import name 'select_popular_piers'`

- [ ] **Step 3: Implement**

Add to `src/playlist/artist_style.py`:

```python
def select_popular_piers(
    member_indices: list[int],
    popularity_values: np.ndarray,
    target_pier_count: int,
) -> list[int]:
    """🔥 pier selection: the up-to-target_pier_count member indices with the highest
    Last.fm popularity score (1 - rank/n; higher = more popular). Pure top-N — no
    sonic-diversity constraint. Non-finite scores (non-hits) are excluded; ties break
    by index. Returns [] when no member has a finite score (caller falls back to
    medoid piers). The pier-bridge still orders these for cohesion downstream."""
    pv = np.asarray(popularity_values, dtype=float)
    scored = [(int(i), float(pv[int(i)])) for i in member_indices if np.isfinite(pv[int(i)])]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [i for i, _ in scored[: max(0, int(target_pier_count))]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_select_popular_piers.py -q --basetemp=$PT`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/unit/test_select_popular_piers.py
git commit -m "feat(popular-seeds): select_popular_piers (top-N popularity pier selection)"
```

---

## Task 2: `_resolve_popular_seeds_mode` (replaces the bool resolver)

**Files:**
- Modify: `src/playlist_generator.py` (replace `_resolve_popular_seeds`, ~the module-scope helper added by the pool-gate)
- Test: `tests/unit/test_bangers_resolve.py` (update the existing `_resolve_popular_seeds` test)

**Interfaces:**
- Produces: `_resolve_popular_seeds_mode(popular_seeds_mode: str, popularity_mode: str) -> str` — returns `"fire"` when `popularity_mode == "oops"` (OOPS forces 🔥); otherwise the normalized user mode (`"off"|"on"|"fire"`, default `"off"`).
- Replaces: `_resolve_popular_seeds(popular_seeds: bool, popularity_mode: str) -> bool` (delete it).

- [ ] **Step 1: Update the test**

In `tests/unit/test_bangers_resolve.py`, **replace** the existing `test_popular_seeds_forced_only_by_oops` (which imports/asserts `_resolve_popular_seeds`) with:

```python
from src.playlist_generator import _resolve_popular_seeds_mode


def test_oops_forces_fire_regardless_of_user_mode():
    assert _resolve_popular_seeds_mode("off", "oops") == "fire"
    assert _resolve_popular_seeds_mode("on", "oops") == "fire"
    assert _resolve_popular_seeds_mode("fire", "oops") == "fire"


def test_non_oops_passes_through_user_mode():
    assert _resolve_popular_seeds_mode("off", "off") == "off"
    assert _resolve_popular_seeds_mode("on", "on") == "on"
    assert _resolve_popular_seeds_mode("fire", "on") == "fire"
    assert _resolve_popular_seeds_mode("", "off") == "off"
```

Also remove the now-stale `_resolve_popular_seeds` import if it's in the file's import line.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_bangers_resolve.py -q --basetemp=$PT`
Expected: FAIL with `ImportError: cannot import name '_resolve_popular_seeds_mode'`

- [ ] **Step 3: Replace the resolver**

In `src/playlist_generator.py`, **delete** `_resolve_popular_seeds(...)` and add:

```python
def _resolve_popular_seeds_mode(popular_seeds_mode: str, popularity_mode: str) -> str:
    """Popular-seed pier mode: off / on / fire. OOPS (the all-bangers bridge gate) forces
    'fire' so the piers are unambiguous hits too. Artist-mode-only by construction (this is
    called on the artist-mode entry point; seed mode never reaches here)."""
    if str(popularity_mode or "off").lower() == "oops":
        return "fire"
    m = str(popular_seeds_mode or "off").lower()
    return m if m in {"off", "on", "fire"} else "off"
```

(The call site in `create_playlist_for_artist` is updated in Task 4.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_bangers_resolve.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/playlist_generator.py tests/unit/test_bangers_resolve.py
git commit -m "feat(popular-seeds): _resolve_popular_seeds_mode (OOPS forces fire)"
```

---

## Task 3: Thread `popular_seeds_mode` through the request stack

**Files:**
- Modify: `src/playlist/request_models.py` (`:104`, `:156`, `:232-233`)
- Modify: `src/playlist_web/schemas.py` (`:36`, `:54`)
- Modify: `src/playlist_gui/worker.py` (`:1312`)
- Modify: `web/src/lib/types.ts`
- Test: `tests/unit/test_bangers_resolve.py` (add a round-trip test) — or wherever request-model tests live; create `tests/unit/test_popular_seeds_mode_threading.py` if no home exists.

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `GeneratePlaylistRequest.popular_seeds_mode: str = "off"` survives `from_worker_args` / `to_worker_args` round-trip; `GenerateRequestBody.popular_seeds_mode: str = "off"`; worker passes `popular_seeds_mode=request.popular_seeds_mode`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_popular_seeds_mode_threading.py`:

```python
from src.playlist.request_models import GeneratePlaylistRequest


def test_popular_seeds_mode_round_trips_through_worker_args():
    req = GeneratePlaylistRequest.from_worker_args(
        {"mode": "artist", "artist": "Stereolab", "popular_seeds_mode": "fire"}
    )
    assert req.popular_seeds_mode == "fire"
    args = req.to_worker_args()
    assert args.get("popular_seeds_mode") == "fire"


def test_popular_seeds_mode_defaults_off():
    req = GeneratePlaylistRequest.from_worker_args({"mode": "artist", "artist": "X"})
    assert req.popular_seeds_mode == "off"
```

(If `to_worker_args` is named differently, mirror the existing `popularity_mode` round-trip in the file. Confirm the method name from `request_models.py` ~line 232.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_popular_seeds_mode_threading.py -q --basetemp=$PT`
Expected: FAIL (`popular_seeds_mode` attribute missing or stays "off")

- [ ] **Step 3: Apply the threading edits**

`src/playlist/request_models.py`:
- Line ~104: replace `popular_seeds: bool = False` with `popular_seeds_mode: str = "off"`
- Line ~156 (`from_worker_args`): replace `popular_seeds=bool(args.get("popular_seeds", False)),` with `popular_seeds_mode=str(args.get("popular_seeds_mode") or "off"),`
- Lines ~232-233 (`to_worker_args`): replace
  ```python
  if self.popular_seeds:
      args["popular_seeds"] = True
  ```
  with
  ```python
  if self.popular_seeds_mode and self.popular_seeds_mode != "off":
      args["popular_seeds_mode"] = str(self.popular_seeds_mode)
  ```

`src/playlist_web/schemas.py`:
- Line ~36: replace `popular_seeds: bool = False` with `popular_seeds_mode: str = "off"`
- Line ~54 (`to_request`): replace `popular_seeds=self.popular_seeds,` with `popular_seeds_mode=self.popular_seeds_mode,`

`src/playlist_gui/worker.py`:
- Line ~1312: replace `popular_seeds=request.popular_seeds,` with `popular_seeds_mode=request.popular_seeds_mode,`

`web/src/lib/types.ts`:
- Replace the `popular_seeds` field (a `boolean`) with `popular_seeds_mode: "off" | "on" | "fire"` (match the existing `popularity_mode` field's style in the same interface).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_popular_seeds_mode_threading.py -q --basetemp=$PT`
Expected: PASS

- [ ] **Step 5: Verify no stale `popular_seeds` references remain in Python**

Run: `git grep -n "popular_seeds\b" -- "src/**/*.py"` (from the worktree)
Expected: only `popular_seeds_mode` / `popular_seeds_weight` hits — NO bare `popular_seeds` boolean reads. Fix any stragglers (e.g. `from_ui_state` if it sets it). The `create_playlist_for_artist` param is renamed in Task 4 — it may still show here; that's expected until Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/request_models.py src/playlist_web/schemas.py src/playlist_gui/worker.py web/src/lib/types.ts tests/unit/test_popular_seeds_mode_threading.py
git commit -m "feat(popular-seeds): thread popular_seeds_mode (off/on/fire) through the stack"
```

---

## Task 4: Wire into `create_playlist_for_artist` + 🔥 override + config default

**Files:**
- Modify: `src/playlist_generator.py` (`create_playlist_for_artist`: param `:1281`, resolver call `:1320`, popularity load `:1747-1764`, 🔥 override after `:1777`, weight default `:1755`)
- Modify: `config.example.yaml`
- Test: covered by Tasks 1–2 unit tests + the live run (this is integration wiring; no new unit test unless a pure helper is extracted).

**Interfaces:**
- Consumes: `select_popular_piers` (Task 1), `_resolve_popular_seeds_mode` (Task 2), `popular_seeds_mode` param (Task 3).
- Produces: artist-mode generation honoring off/on/🔥; OOPS → 🔥.

- [ ] **Step 1: Rename the param + resolve the mode**

In `create_playlist_for_artist` (`playlist_generator.py`):
- Param (~line 1281): replace `popular_seeds: bool = False,` with `popular_seeds_mode: str = "off",`
- The resolver call added by the pool-gate (~line 1320, currently `popular_seeds = _resolve_popular_seeds(popular_seeds, popularity_mode)`): replace with
  ```python
  popular_seeds_mode = _resolve_popular_seeds_mode(popular_seeds_mode, popularity_mode)
  ```

- [ ] **Step 2: Mode-gate the popularity load + raise the weight default**

At the popularity-load block (~lines 1747-1764), replace the activation condition and weight default:
- `if popular_seeds and getattr(self, "lastfm", None) is not None:` → `if popular_seeds_mode in {"on", "fire"} and getattr(self, "lastfm", None) is not None:`
- `pop_w = float(style_cfg_raw.get("popular_seeds_weight", 0.5))` → `pop_w = float(style_cfg_raw.get("popular_seeds_weight", 1.0))`
- Also update the per-pier rank logging block (the `if popular_seeds:` guard at ~line 1816, found via `git grep "if popular_seeds" -- src/playlist_generator.py`) to `if popular_seeds_mode in {"on", "fire"}:`.

- [ ] **Step 3: Add the 🔥 pier override**

Immediately AFTER the `cluster_artist_tracks(...)` call returns and BEFORE `if not medoids:` (i.e. right after line ~1777), insert:

```python
                # 🔥 Pure-hits piers: override cluster medoids with the artist's top-N
                # most-popular tracks (selection only — order_clusters still sequences them).
                if popular_seeds_mode == "fire" and popularity_values is not None:
                    _all_members = [i for _cluster in clusters for i in _cluster]
                    _fire_piers = select_popular_piers(_all_members, popularity_values, target_pier_count)
                    if _fire_piers:
                        logger.info(
                            "Popular Seeds 🔥: overriding %d cluster-medoid piers with top-%d popular tracks",
                            len(medoids), len(_fire_piers),
                        )
                        medoids = _fire_piers
                    else:
                        logger.warning(
                            "Popular Seeds 🔥: no popular piers resolved (uncached artist?) — "
                            "falling back to cluster-medoid piers",
                        )
```

Ensure `select_popular_piers` is imported from `src.playlist.artist_style` at the top of the file (add to the existing `from src.playlist.artist_style import (...)` block at line ~15).

- [ ] **Step 4: Config default**

In `config.example.yaml`, under `playlists.ds_pipeline.artist_style`, set/add:
```yaml
        popular_seeds_weight: 1.0   # ON-mode medoid popularity weight (was 0.5)
```
(If the key already exists at 0.5, change it to 1.0.)

- [ ] **Step 5: Import-sanity + targeted tests**

Run:
```bash
python -c "import src.playlist_generator" && echo IMPORT_OK
python -m pytest tests/unit/test_bangers_resolve.py tests/unit/test_select_popular_piers.py tests/unit/test_popular_seeds_mode_threading.py -q --basetemp=$PT
```
Expected: `IMPORT_OK` and all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/playlist_generator.py config.example.yaml
git commit -m "feat(popular-seeds): wire off/on/fire into create_playlist_for_artist + weight default 1.0"
```

---

## Task 5: GUI dropdown (checkbox → Off / On / 🔥)

**Files:**
- Modify: `web/src/components/GenerateControls.tsx` (the Popular Seeds control)
- (types.ts already updated in Task 3)

**Interfaces:**
- Consumes: `popular_seeds_mode: "off" | "on" | "fire"` (Task 3).
- Produces: a three-option dropdown that sets `popular_seeds_mode`; shows forced-🔥 (disabled) when Bangers = OOPS.

- [ ] **Step 1: Read the existing Bangers dropdown for the pattern**

In `web/src/components/GenerateControls.tsx`, find the **Bangers** dropdown (the `popularity_mode` Off / On / "Oops, All Bangers" control the prior session added). The Popular Seeds control is currently a checkbox bound to `popular_seeds`. Mirror the Bangers dropdown exactly for Popular Seeds.

- [ ] **Step 2: Replace the checkbox with a dropdown**

Replace the `popular_seeds` checkbox with a `popular_seeds_mode` dropdown with options:
- `off` → "Popular Seeds: Off"
- `on` → "Popular Seeds: On"
- `fire` → "🔥 Pure Hits"

Bind it to the request field `popular_seeds_mode` (mirror how the Bangers dropdown binds `popularity_mode`). When `popularity_mode === "oops"`, render the dropdown disabled and showing 🔥 (mirror any analogous disable behavior; if none exists, set `disabled={popularity_mode === "oops"}` and force the displayed value to `fire`).

- [ ] **Step 3: Build the front end**

Run (from the worktree root): `npm --prefix web run build`
Expected: build succeeds, no TS errors (the `popular_seeds_mode` union type from Task 3 must line up).

- [ ] **Step 4: Commit**

```bash
git add web/src/components/GenerateControls.tsx web/src/lib/types.ts
git commit -m "feat(popular-seeds): GUI three-way dropdown (Off/On/Fire), forced Fire under OOPS"
```

---

## Task 6: Full unit sweep + live verification handoff

**Files:** none (verification).

- [ ] **Step 1: Bangers + popular-seeds unit tests**

Run:
```bash
python -m pytest tests/unit/test_select_popular_piers.py tests/unit/test_bangers_resolve.py tests/unit/test_popular_seeds_mode_threading.py -q --basetemp=$PT
```
Expected: PASS (quote real counts).

- [ ] **Step 2: Broader fast suite for regressions**

Run: `python -m pytest tests/unit -q -m "not slow" --basetemp=$PT`
Expected: PASS. Fix root causes of any breakage (likely stale `popular_seeds` references); quote real pass/fail counts you saw.

- [ ] **Step 3: Live verification (USER-driven)**

Hand to the user (run from the MAIN checkout, which has data; rebuild `web/dist`, restart `serve_web`):
- **OFF** — unchanged from today.
- **ON** — Stereolab piers shallower than the #20/#28/#29 baseline (`popular_seeds_weight=1.0`); `Popular Seeds: N/N piers on top-50` log shows shallower ranks.
- **🔥** — `Popular Seeds 🔥: overriding … with top-N popular tracks` in the log; piers = the top ~6 hits; arc may flatten but the order is still optimized.
- **OOPS** — auto-runs 🔥 (piers are hits) + the banger bridge gate; budget < 90s.
- **Seed mode** — unaffected (user-chosen seeds).

- [ ] **Step 4: Commit (only if Step 2 needed fixes)**

```bash
git add -A
git commit -m "test(popular-seeds): full unit sweep green"
```

---

## Self-Review (completed)

- **Spec coverage:** 🔥 selector (§3.4 → T1), OOPS→🔥 resolver (§3.2 → T2), threading bool→mode (§3.1 → T3), ON weight 1.0 + 🔥 override wiring + config (§3.3/§3.4/§5 → T4), GUI dropdown (§3.5 → T5), testing (§6 → T1–T6). Recency-unchanged and 🔥 ordering-preserved are constraints honored by *not* touching `cluster_artist_tracks`' filtering or the pier-bridge.
- **Placeholder scan:** GUI task (T5) references the existing Bangers dropdown as the concrete pattern rather than reproducing JSX — acceptable for a mirror-an-existing-component front-end task verified by build + live. No TBD/TODO.
- **Type consistency:** `popular_seeds_mode` (str "off"/"on"/"fire") and `select_popular_piers(member_indices, popularity_values, target_pier_count)` and `_resolve_popular_seeds_mode(popular_seeds_mode, popularity_mode)` names/signatures are identical across T1–T5 definitions, wiring, and tests. The pool-gate's `_resolve_popular_seeds` (bool) is explicitly deleted in T2 (no dangling references — T3 Step 5 + T4 Step 5 grep/import-sanity catch stragglers).
