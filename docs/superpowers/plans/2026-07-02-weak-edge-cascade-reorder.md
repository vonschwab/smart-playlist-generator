# Weak-edge Cascade Reorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the pier-bridge weak-edge fixers into a least→most destructive escalation — add-only variable bridge first, deletion of a track as the true last resort.

**Architecture:** Split variable-bridge length into an early **add-only** pass (never shortens) and a new **remove-only** post-assembly pass (`edge_delete`) that runs after break-glass repair and deletes an interior track only when the merge strictly lifts a still-broken edge (never a pier). Track count is not a target, so all length-budget bookkeeping is deleted.

**Tech Stack:** Python 3.11, pytest, numpy. Pier-bridge builder (`src/playlist/pier_bridge_builder.py`, ~5.3k LOC hotspot), `src/playlist/repair/` package, `src/playlist/pier_bridge/config.py`.

## Global Constraints

- **Cascade order (runtime):** `beam → var-bridge ADD-only → tail-DP → assemble → break-glass repair → remove-only`. Passes 2 (tail-DP) and 3 (repair) are UNCHANGED.
- **Never delete a pier/seed.** Deletion only ever removes interior bridge tracks whose bundle index is NOT in the protected set.
- **Never-worse deletion:** accept a deletion only if the merged edge strictly exceeds the broken edge's T. Otherwise leave the edge (the true "nothing worked" outcome).
- **No length budget / target-count tracking anywhere.** Track count is arbitrary; a playlist ending shorter or longer than requested is fine.
- **Shared floor = 0.30** across var-bridge (`variable_bridge_min_edge`), tail-DP (`tail_dp_floor`), repair (`edge_repair_t_floor`), delete (`edge_delete_floor`).
- **Activate fixes:** `edge_delete_enabled` defaults **True** (live), rollback via `false`. With `edge_delete_enabled=false`, output is byte-identical to pre-feature.
- **Shared `master`, concurrent sessions.** Stage explicit pathspecs, never `git add -A`, never bare `git commit`. `git status --short -- <file>` before each edit; STOP on foreign uncommitted changes. Co-author trailer on every commit: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **pytest never piped through head/tail** (a hook blocks it). Run bounded/direct.

---

### Task 1: `edge_delete.py` — the deletion algorithm (pure)

**Files:**
- Create: `src/playlist/repair/edge_delete.py`
- Test: `tests/unit/test_edge_delete.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class DeleteResult: indices: list[int]; delete_log: list[dict]`
  - `delete_broken_edges(indices, *, edge_score, floor, protected_indices, max_deletions=4) -> DeleteResult`
    - `indices: Sequence[int]` — the assembled playlist (bundle indices).
    - `edge_score: Callable[[int, int], float]` — pairwise transition T (injected; the builder wraps `score_transition_edge`).
    - `floor: float` — an edge below this is "broken".
    - `protected_indices: set[int]` — bundle indices never deletable (piers/seeds).
    - `max_deletions: int` — deterministic cap.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_edge_delete.py
from src.playlist.repair.edge_delete import delete_broken_edges, DeleteResult


def _score(pairs, default=0.9):
    """edge_score backed by a symmetric dict; default for unspecified pairs."""
    def score(a, b):
        return pairs.get((a, b), pairs.get((b, a), default))
    return score


def test_deletes_worse_interior_endpoint_and_merges():
    # piers 10,13 protected; edge 11->12 broken (0.05). Deleting 11 merges 10->12=0.7;
    # deleting 12 merges 11->13=0.6. Best = delete 11.
    score = _score({(11, 12): 0.05, (10, 12): 0.70, (11, 13): 0.60, (10, 11): 0.8, (12, 13): 0.8})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 12, 13]
    assert len(r.delete_log) == 1 and r.delete_log[0]["deleted_idx"] == 11


def test_leaves_edge_when_no_deletion_improves():
    # broken 11->12=0.05; both merges also below the broken value -> never-worse blocks it.
    score = _score({(11, 12): 0.05, (10, 12): 0.02, (11, 13): 0.01})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_never_deletes_between_two_piers():
    # broken edge 10->13 is directly between two protected piers -> cannot delete either.
    score = _score({(10, 13): 0.05})
    r = delete_broken_edges([10, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 13]
    assert r.delete_log == []


def test_noop_when_nothing_broken():
    score = _score({}, default=0.8)  # all edges 0.8 >= floor
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_respects_max_deletions():
    # two broken interior edges; cap at 1 deletion.
    score = _score({(11, 12): 0.05, (12, 13): 0.05, (10, 12): 0.7, (11, 13): 0.7,
                    (10, 11): 0.8, (13, 14): 0.8}, default=0.8)
    r = delete_broken_edges([10, 11, 12, 13, 14], edge_score=score, floor=0.30,
                            protected_indices={10, 14}, max_deletions=1)
    assert len(r.delete_log) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_edge_delete.py -q`
Expected: FAIL (ModuleNotFoundError: src.playlist.repair.edge_delete).

- [ ] **Step 3: Implement `edge_delete.py`**

```python
# src/playlist/repair/edge_delete.py
"""Remove-only weak-edge fixer (repair-by-deletion) — the last resort.

Runs AFTER break-glass repair. For an edge still below floor, delete the interior
endpoint whose removal best merges the two edges, but ONLY if the merged edge
strictly beats the broken edge (never-worse). Never deletes a pier/seed. See
docs/superpowers/specs/2026-07-02-weak-edge-cascade-reorder-design.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeleteResult:
    indices: list[int]
    delete_log: list[dict] = field(default_factory=list)


def delete_broken_edges(
    indices: Sequence[int],
    *,
    edge_score: Callable[[int, int], float],
    floor: float,
    protected_indices: set[int],
    max_deletions: int = 4,
) -> DeleteResult:
    idx = [int(x) for x in indices]
    delete_log: list[dict] = []
    try:
        for _ in range(max(0, int(max_deletions))):
            if len(idx) < 3:
                break
            # worst adjacent edge
            worst_pos, worst_t = 0, float("inf")
            for i in range(len(idx) - 1):
                t = float(edge_score(idx[i], idx[i + 1]))
                if t < worst_t:
                    worst_pos, worst_t = i, t
            if worst_t >= float(floor):
                break  # nothing broken
            # candidate deletions: the two endpoints of the worst edge
            best = None  # (merged_t, del_pos)
            for del_pos in (worst_pos, worst_pos + 1):
                if idx[del_pos] in protected_indices:
                    continue
                prev, nxt = del_pos - 1, del_pos + 1
                if prev < 0 or nxt >= len(idx):
                    continue  # boundary track has no neighbor to merge across
                merged = float(edge_score(idx[prev], idx[nxt]))
                if best is None or merged > best[0]:
                    best = (merged, del_pos)  # ties -> lower del_pos (worst_pos first)
            if best is None or best[0] <= worst_t:
                break  # nothing improves the broken edge -> leave it
            merged_t, del_pos = best
            delete_log.append({
                "position": del_pos,
                "deleted_idx": idx[del_pos],
                "old_worst_T": worst_t,
                "new_merged_T": merged_t,
            })
            logger.info(
                "Edge-delete: removed interior idx=%s at pos=%d, worst-T %.3f -> merged %.3f",
                idx[del_pos], del_pos, worst_t, merged_t,
            )
            del idx[del_pos]
    except Exception:  # never break a generation
        logger.warning("edge_delete failed; leaving playlist unchanged", exc_info=True)
        return DeleteResult(indices=[int(x) for x in indices], delete_log=[])
    return DeleteResult(indices=idx, delete_log=delete_log)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_edge_delete.py -q`
Expected: PASS (5 passed). Then `python -m mypy src/playlist/repair/edge_delete.py` (clean) and `ruff check src/playlist/repair/edge_delete.py tests/unit/test_edge_delete.py` (clean).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/repair/edge_delete.py tests/unit/test_edge_delete.py
git commit -m "feat(edge-delete): repair-by-deletion pure module (never-worse, never a pier)" -- src/playlist/repair/edge_delete.py tests/unit/test_edge_delete.py
```

---

### Task 2: Wire `edge_delete` into the builder + config knobs

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (add 3 knobs near `edge_repair_*` ~line 323)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (thread the knobs — mirror the `edge_repair` override idiom)
- Modify: `src/playlist/pier_bridge_builder.py` (call after the repair block, ~after line 2916)

**Interfaces:**
- Consumes: `delete_broken_edges` / `DeleteResult` (Task 1); the builder's existing `transition_metric_context`, `final_indices`, `seed_indices`; `score_transition_edge` (the SAME function `repair_playlist_edges` uses internally — confirm its exact signature `score_transition_edge(metric_context, a, b) -> float` by reading `src/playlist/repair/edge_repair.py`).

- [ ] **Step 1: Add config knobs.** In `src/playlist/pier_bridge/config.py`, next to the `edge_repair_*` fields (~line 323):

```python
    edge_delete_enabled: bool = True        # remove-only last resort (repair-by-deletion)
    edge_delete_floor: float = 0.30         # an edge below this may trigger deletion
    edge_delete_max_deletions: int = 4       # deterministic cap. Spec 2026-07-02.
```

- [ ] **Step 2: Thread overrides.** In `src/playlist/pipeline/pier_bridge_overrides.py`, find the `edge_repair` override block and add a sibling that maps a nested `edge_delete: {enabled, floor, max_deletions}` dict onto the config (byte-for-byte structural copy of the `edge_repair` idiom). Add a knob-threading test to the file's existing override test if present, else to `tests/unit/test_edge_delete.py`:

```python
def test_edge_delete_knobs_default_and_override():
    from src.playlist.pier_bridge.config import PierBridgeConfig
    c = PierBridgeConfig()
    assert c.edge_delete_enabled is True
    assert c.edge_delete_floor == 0.30
    assert c.edge_delete_max_deletions == 4
```

- [ ] **Step 3: Write the failing integration test.** In `tests/unit/test_edge_delete.py`, add a builder-path test that a still-broken edge (unrepairable — empty candidate pool for repair) gets deleted when `edge_delete_enabled=True` and is left when `False`. Use the `gui_fidelity`/multi-pier harness the other builder tests use (read `tests/unit/test_builder_pair_floor_wiring.py` for the pattern — do NOT hand-build single-seed topology). Assert the playlist is shorter by exactly the number of `delete_log` entries and that no pier index was removed.

- [ ] **Step 4: Run it to verify it fails.**

Run: `python -m pytest tests/unit/test_edge_delete.py -q -k builder`
Expected: FAIL (builder does not call `delete_broken_edges` yet).

- [ ] **Step 5: Integrate in the builder.** In `src/playlist/pier_bridge_builder.py`, immediately AFTER the `edge_repair` block (after ~line 2916, before "Convert to track IDs" at ~2918):

```python
    edge_delete_log: list[dict[str, Any]] = []
    if bool(getattr(cfg, "edge_delete_enabled", True)) and len(final_indices) >= 3:
        from src.playlist.repair.edge_delete import delete_broken_edges
        from src.playlist.transition_metrics import score_transition_edge  # confirm import path

        _protected = {int(s) for s in seed_indices}

        def _del_edge_score(a: int, b: int) -> float:
            return float(score_transition_edge(transition_metric_context, int(a), int(b)))

        _del_result = delete_broken_edges(
            final_indices,
            edge_score=_del_edge_score,
            floor=float(getattr(cfg, "edge_delete_floor", 0.30)),
            protected_indices=_protected,
            max_deletions=int(getattr(cfg, "edge_delete_max_deletions", 4)),
        )
        edge_delete_log = list(_del_result.delete_log)
        if list(_del_result.indices) != list(final_indices):
            final_indices = list(_del_result.indices)
            all_beam_components = []
```

Then add `edge_delete_log` (and an `edge_delete_applied` bool) to the diagnostics dict near the `edge_repair_swap_log` entry (~line 3052), mirroring its shape.

- [ ] **Step 6: Run tests to verify they pass.**

Run: `python -m pytest tests/unit/test_edge_delete.py -q`
Expected: PASS (all). Then `python -m mypy src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py` (no NEW errors vs baseline) and `python -c "import src.playlist.pier_bridge_builder"` (OK) and `ruff check` on the 3 edited files.

- [ ] **Step 7: Commit** (pathspec, all four files).

```bash
git commit -m "feat(edge-delete): wire remove-only pass after break-glass repair" -- src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/test_edge_delete.py
```

---

### Task 3: Convert variable-bridge to ADD-only (drop the length budget)

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` var-bridge init (~lines 1730-1736) and the length-selection block (~lines 1986-2019)

**Interfaces:**
- Consumes: `choose_segment_length(nominal, lo, hi, build_and_score, *, good_enough, eps)` — UNCHANGED (already range-general). Only the caller's `lo`/`hi` and the removed `_vbl_band`/`_vbl_total_dev` bookkeeping change.

- [ ] **Step 1: Write the failing test.** Add to `tests/unit/test_edge_delete.py` (or a new `tests/unit/test_var_bridge_add_only.py`) a builder-path test through the multi-pier `gui_fidelity` harness asserting **no segment is shorter than its nominal even-split length** when variable bridge is ON (i.e. var-bridge only ever adds). Capture per-segment `nominal`/`chosen` from the `Var-bridge seg …` log or the segment diagnostics. (Mirror the harness in `tests/unit/test_layered_bridge_overrides.py`.)

- [ ] **Step 2: Run it to verify it fails** (today var-bridge can choose `chosen < nominal`).

Run: `python -m pytest tests/unit/test_var_bridge_add_only.py -q`
Expected: FAIL (a segment shrank below nominal, OR the assertion has no data because nothing shrank in this fixture — if so, strengthen the fixture with a close pier-pair that currently shortens; confirm RED before proceeding).

- [ ] **Step 3: Convert to add-only.** In `src/playlist/pier_bridge_builder.py`:
  - In the var-bridge init (~1730-1736), DELETE the `_vbl_band = ...` and `_vbl_total_dev = 0` lines. Keep `_vbl_k`, `_vbl_good`, `_vbl_eps`, `_vbl_flexed`, `_vbl_max_flex`.
  - In the selection block (~1986-2015), replace the band-clamp with add-only bounds and drop the running-total update:

```python
        else:
            # ADD-only: never shorten a segment. lo == nominal; may lengthen up to
            # +variable_bridge_flex. No length budget (track count is not a target).
            lo = nominal
            hi = nominal + _vbl_k
            # Deterministic cap: once max-flex-segments have flexed, force nominal.
            if _vbl_flexed >= _vbl_max_flex:
                lo = hi = nominal

            def _build_and_score(length: int) -> tuple[dict[str, Any], float]:
                r = _build_segment_at(length)
                ar = r["attempt_result"]
                p = ar["segment_path"] if ar is not None else None
                if p:
                    nodes = [int(pier_a), *[int(x) for x in p], int(pier_b)]
                    b = segment_bottleneck(nodes, _edge_T)[0]
                else:
                    b = float("-inf")
                return (r, b)

            chosen_len, _seg_build, _vbl_seg_flexed = choose_segment_length(
                nominal, lo, hi, _build_and_score,
                good_enough=_vbl_good, eps=_vbl_eps,
            )
            if _vbl_seg_flexed:
                _vbl_flexed += 1
            interior_len = int(chosen_len)
            logger.info(
                "Var-bridge seg %d: nominal=%d chosen=%d flexed=%s (%d/%d) [add-only]",
                seg_idx, nominal, chosen_len, _vbl_seg_flexed, _vbl_flexed, _vbl_max_flex,
            )
```

  Grep the function for any remaining `_vbl_total_dev` / `_vbl_band` reference and remove it. `variable_bridge_band` in config becomes unused — leave the field defined (harmless) but add a comment `# unused since add-only reorder (2026-07-02)`.

- [ ] **Step 4: Run tests to verify they pass.**

Run: `python -m pytest tests/unit/test_var_bridge_add_only.py tests/unit/test_edge_delete.py -q`
Expected: PASS. Then `python -m mypy src/playlist/pier_bridge_builder.py` (no new errors) and `python -c "import src.playlist.pier_bridge_builder"` (OK) and `ruff check src/playlist/pier_bridge_builder.py`.

- [ ] **Step 5: Commit** (pathspec).

```bash
git commit -m "feat(var-bridge): add-only length selection (never shorten; drop length budget)" -- src/playlist/pier_bridge_builder.py tests/unit/test_var_bridge_add_only.py
```

---

### Task 4: Config surface, goldens, and live composition validation

**Files:**
- Modify: `config.example.yaml`, `config.yaml` (gitignored — edit, won't stage)
- Modify: `tests/unit/goldens/pipeline/*.json` (regenerate)

- [ ] **Step 1: Add the config block.** In `config.example.yaml` (and `config.yaml` if present/untracked), under the pier-bridge config section, add:

```yaml
      edge_delete_enabled: true
      edge_delete_floor: 0.30
      edge_delete_max_deletions: 4
```
Place it near the existing `edge_repair_*` keys. Confirm the exact nesting by reading how `edge_repair_*` appears in `config.example.yaml`.

- [ ] **Step 2: Identify ALL failing goldens, then regenerate.** Run `python -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q` to a file and list every failing golden. Two golden SETS shift from this feature:
  - **Config-snapshot** (`tests/unit/goldens/pipeline/*.json`, 4 files): gain the 3 `edge_delete_*` keys.
  - **Behavioral** (`tests/unit/goldens/pier_bridge/*.json`, at least `two_seeds_default.json` + `three_seeds_centered.json`): the *playlist itself* changes because `edge_delete` (live default) now deletes a broken-edge outlier AND add-only var-bridge no longer shortens. (Task 2 already confirmed `edge_delete_enabled=false` reproduces the old behavioral goldens byte-for-byte.)

```bash
# regenerate whichever files the run above reported as failing (delete then rebuild)
rm <each-failing-golden.json>
python -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q   # rebuild baselines (skips)
python -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q   # verify (passes)
```

- [ ] **Step 3: Diff-audit — legitimacy, not just shape.** For **config-snapshot** goldens, every changed line must be an `edge_delete_*` key addition. For **behavioral** goldens, every removed track must be a genuine `edge_delete` deletion of a **broken edge** — verify by re-running that fixture with `edge_delete_enabled=false` and confirming (a) the old output returns AND (b) the deleted track sat on an edge below 0.30 whose merge lifted the worst edge (the feature working, not a misfire). Any track removed from an already-healthy edge (worst edge was ≥ 0.30) is a BUG — STOP and investigate. Likewise any segment that got *longer* with no weak-edge justification. Commit only after the audit passes, with explicit pathspec for every regenerated file:

```bash
git commit -m "test(golden): regen for edge_delete + add-only var-bridge (diff-audited, deletions verified legit)" -- <each-regenerated-golden.json>
```

- [ ] **Step 4: Full fast suite.**

Run: `python -m pytest tests/unit -q -m "not slow"` (redirect to a file; do NOT pipe through head/tail).
Expected: green except the known pre-existing foreign `test_layered_genre_taxonomy.py::test_reviewed_taxonomy_conditional_aliases_require_context` failure (concurrent session). Quote the real pass/fail counts.

- [ ] **Step 5: Live composition validation (orchestrator-run, not a subagent).** Run a real multi-pier generation through the `gui_fidelity` harness (per the `playlist-testing` skill — never single-seed). At INFO, confirm: (a) `Var-bridge seg …` lines show `chosen >= nominal` on every segment (add-only holds); (b) when an edge is genuinely unfixable, an `Edge-delete: removed interior …` line fires and the final `min_transition` improves vs the same run with `edge_delete_enabled=false`; (c) on a healthy pool, edge-delete does NOT fire (delete_log empty). Record the numbers in the ledger.

---

## Self-Review

**Spec coverage:** add-only (Task 3) ✅; tail-DP/repair unchanged (untouched) ✅; remove-only new pass (Tasks 1-2) ✅; never-delete-pier (Task 1 `protected_indices` + test) ✅; never-worse (Task 1 `best[0] <= worst_t` guard + test) ✅; no length budget (Task 3 deletes `_vbl_band`/`_vbl_total_dev`) ✅; config knobs default-on (Task 2) ✅; goldens regen (Task 4) ✅; consequence — no marginal shortening — is the *point* of Task 3 (`lo = nominal`) ✅.

**Placeholder scan:** the one place an implementer must confirm-not-guess is the exact import path/signature of `score_transition_edge` (Task 2, flagged explicitly to read `edge_repair.py`) and the `edge_repair` override idiom (Task 2). Both are "read this existing code and mirror it," not placeholders.

**Type consistency:** `delete_broken_edges` / `DeleteResult` / `edge_score: Callable[[int,int],float]` / `protected_indices: set[int]` are consistent across Tasks 1-2. `interior_len`/`_vbl_flexed` match the existing builder names (Task 3).
