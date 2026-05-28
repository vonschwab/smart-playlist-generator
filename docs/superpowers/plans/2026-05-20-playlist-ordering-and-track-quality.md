# Playlist Ordering & Track Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve playlist sequencing and track-level quality without regressing the recently-fixed candidate pool. Diagnose why high-T edges still feel jarring, why below-floor edges leak into final output, and why track artifacts (live/demo/medley) survive selection — then implement targeted, opt-in fixes informed by the diagnostics.

**Architecture:** Three-phase plan. Phase A adds per-edge diagnostics for the final emitted playlist (no behavior change) so we can see exactly what the beam selected and why. Phase B introduces three opt-in scoring/filtering knobs (title-artifact penalty, scaled local-sonic-edge penalty, worst-edge lexicographic objective) — all off by default. Phase C lands tests and a tuning recipe.

**Tech Stack:** Python 3.11+, NumPy, existing pier-bridge beam search, pytest.

---

## Investigation Findings (already gathered, anchors the plan)

From the bad-playlist log at `C:\Users\Dylan\Desktop\tmp\log.txt` and source inspection of `src/playlist/pier_bridge/beam.py`, `src/playlist/segment_pool_builder.py`, `src/playlist/pier_bridge/config.py`:

1. **A T=0.092 edge (below `transition_floor=0.20`) made it into the final 30-track playlist**, but `beam.py:821` has a `if trans_score < cfg.transition_floor: continue` hard gate. `Pier+Bridge complete: ... 4 successful` — no fallback was triggered. This means either (a) the beam's `trans_score` formulation differs from the final-emitted-playlist's T (different rescaling/centering), or (b) the failing edge is at a segment boundary not covered by the gate. **Phase A must root-cause this before any remediation.**
2. **Beam combined score is `0.6 * bridge_score + 0.4 * trans_score - penalties`** (`beam.py:836-839`). Bridge-dominant: candidates that are midpoint-balanced toward both piers outrank candidates with smoother local handoffs. The user-reported jarring transitions (Arctic Monkeys → Pavement → Destroyer → Yo La Tengo, each at T≥0.95) are consistent with bridge-dominance — locally jarring but globally midpoint-balanced.
3. **`local_sonic_edge_penalty` is essentially decorative**. Penalty math at `beam.py:236-238` is `strength * (threshold - edge_sonic)`. With `strength=0.30, threshold=0.10`, the **maximum possible penalty is 0.03** on a combined_score of ~0.6–1.0. Cannot meaningfully deter a bad-local-edge candidate when bridge_score is high.
4. **Final emitted playlist reports `T_centered_cos min=-0.817` (anti-correlated) but T=0.092** — the rescaling collapses catastrophic mismatches into the "just below floor" range, hiding severity in the headline metric.
5. **Title artifacts (e.g., "Sonic Youth - Lee #2 (8 Track Demo)") enter the pool.** `title_exclusion_words` in `config.yaml:96` currently only excludes `["interlude", "skit", "acapella", "a cappella", "a capella"]` — no demo/medley/live/remix/take/version/remaster/instrumental.
6. **Per-edge diagnostics for the final playlist are sparse.** `reporter.py` emits T per track and T/S/G for the bottom 3 by T. No per-edge breakdown of bridge_score, trans_score (raw cosine), progress_t, progress_jump, local_sonic_penalty applied, genre_penalty applied, was-clipped-by-floor, was-near-pier-boundary. Without these, we cannot tell **why** a specific bad edge was chosen.
7. **Provenance counters show `baseline_only=all`** across all segments even though `winner_changed=2/3` for waypoints — the provenance categorization is tracking pool membership, not actual ranking influence. Misleading diagnostic.

These findings inform task design. Phase A produces the data needed to confirm Phase B fixes are correctly scoped.

---

## File Structure

- **Diagnostics (Phase A)**
  - Modify `src/playlist/pier_bridge/beam.py` — record per-edge scoring breakdown into a new optional dict
  - Modify `src/playlist/reporter.py` — emit a "Selected-edge audit" table when `dj_diagnostics_pool_verbose` is true
  - Modify `src/playlist/constructor.py` — surface beam_trans_score vs final_T mismatch
  - Create `src/playlist/title_quality.py` — flag detection (live/demo/medley/etc.); pure function, no I/O
  - Modify `src/playlist_generator.py` — call title_quality flagging during edge audit emission
  - Tests: `tests/unit/test_title_quality.py`, `tests/unit/test_selected_edge_audit.py`

- **Configurable scoring/filtering (Phase B)**
  - Modify `src/playlist/pier_bridge/config.py` — new fields: `title_artifact_penalty_enabled`, `title_artifact_penalty_strength`, `title_artifact_terms`, `local_sonic_edge_penalty_mode`, `local_sonic_edge_penalty_scale`, `min_edge_objective_enabled`, `min_edge_objective_floor`
  - Modify `src/playlist/pier_bridge/beam.py` — wire title-artifact penalty into the per-candidate scoring loop; add scaled local-sonic mode; implement worst-edge lexicographic ranking when enabled
  - Modify `src/playlist/pipeline/pier_bridge_overrides.py` — parse the new config keys from the `pier_bridge:` override block
  - Modify `config.yaml` — leave behavior unchanged by default; add commented documentation block for the new knobs
  - Tests: `tests/unit/test_title_artifact_penalty.py`, `tests/unit/test_local_sonic_scaled_mode.py`, `tests/unit/test_min_edge_objective.py`

- **Tuning recipe (Phase C)**
  - Create `docs/PLAYLIST_ORDERING_TUNING.md` — recipe for tuning the new knobs after running Phase A diagnostics on representative playlists

---

## Phase A: Diagnostics (no default behavior change)

### Task 1: Per-edge audit table for final emitted playlist

**Goal:** For every edge in the final emitted playlist, log a row with all scoring components so we can root-cause bad transitions.

**Files:**
- Modify: `src/playlist/reporter.py`
- Modify: `src/playlist/pier_bridge/beam.py` (capture `chosen_edge_components` into edge_scores)
- Test: `tests/unit/test_selected_edge_audit.py`

- [ ] **Step 1: Write failing test for edge audit emission**

Create `tests/unit/test_selected_edge_audit.py`:

```python
import logging

from src.playlist.reporter import emit_selected_edge_audit


def test_emit_selected_edge_audit_writes_one_row_per_edge(caplog):
    caplog.set_level(logging.INFO)
    edges = [
        {
            "from_idx": 1, "to_idx": 2,
            "from_artist": "Geese", "from_title": "Cowboy Nudes",
            "to_artist": "Arctic Monkeys", "to_title": "Library Pictures",
            "T": 0.989, "T_centered_cos": 0.978, "S": 0.491, "G": 0.890,
            "bridge_score": 0.61, "trans_score_in_beam": 0.95,
            "progress_t": 0.14, "progress_jump": 0.14,
            "local_sonic_raw_cos": 0.42, "local_sonic_penalty_applied": 0.0,
            "genre_penalty_applied": 0.0,
            "below_transition_floor": False,
        },
        {
            "from_idx": 14, "to_idx": 15,
            "from_artist": "Hideous Sun Demon", "from_title": "Gimmicks",
            "to_artist": "Stove", "to_title": "Nightwalk",
            "T": 0.092, "T_centered_cos": -0.817, "S": 0.306, "G": 1.000,
            "bridge_score": 0.55, "trans_score_in_beam": 0.25,
            "progress_t": 0.85, "progress_jump": 0.10,
            "local_sonic_raw_cos": 0.03, "local_sonic_penalty_applied": 0.021,
            "genre_penalty_applied": 0.0,
            "below_transition_floor": True,
        },
    ]
    emit_selected_edge_audit(edges)
    text = caplog.text
    assert "Selected-edge audit" in text
    assert "Stove" in text and "Nightwalk" in text
    assert "T=0.092" in text
    assert "T_centered_cos=-0.817" in text
    assert "below_floor=True" in text


def test_emit_selected_edge_audit_handles_missing_fields(caplog):
    caplog.set_level(logging.INFO)
    edges = [{"from_idx": 0, "to_idx": 1, "T": 0.5}]
    emit_selected_edge_audit(edges)
    assert "Selected-edge audit" in caplog.text
```

- [ ] **Step 2: Run the failing test**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_selected_edge_audit.py -q --basetemp .pytest-tmp-edge-audit -o cache_dir=.pytest-tmp-cache-edge-audit
```
Expected: ImportError for `emit_selected_edge_audit` from `src.playlist.reporter`.

- [ ] **Step 3: Implement `emit_selected_edge_audit` in reporter.py**

In `src/playlist/reporter.py`, add near the top-level emission functions:

```python
def emit_selected_edge_audit(edge_rows: list[dict]) -> None:
    """Emit one log row per selected edge with full scoring breakdown.

    Diagnostic only; no behavior change. Each row contains the fields
    populated in beam scoring (bridge_score, trans_score_in_beam,
    progress_t, progress_jump, local_sonic_raw_cos, local_sonic_penalty_applied,
    genre_penalty_applied, below_transition_floor) plus the final-emitted
    edge metrics (T, T_centered_cos, S, G). Missing fields render as 'n/a'.
    """
    if not edge_rows:
        return
    logger.info("=" * 80)
    logger.info("Selected-edge audit (%d edges)", len(edge_rows))
    logger.info("=" * 80)
    for i, row in enumerate(edge_rows):
        def _f(key, fmt="%.3f"):
            v = row.get(key)
            if v is None:
                return "n/a"
            try:
                return fmt % float(v)
            except Exception:
                return str(v)

        from_label = "%s - %s" % (
            row.get("from_artist", "?"),
            row.get("from_title", "?"),
        )
        to_label = "%s - %s" % (
            row.get("to_artist", "?"),
            row.get("to_title", "?"),
        )
        logger.info(
            "Edge #%02d: %s -> %s", i + 1, from_label, to_label,
        )
        logger.info(
            "  T=%s T_centered_cos=%s S=%s G=%s | bridge=%s trans_beam=%s",
            _f("T"), _f("T_centered_cos"), _f("S"), _f("G"),
            _f("bridge_score"), _f("trans_score_in_beam"),
        )
        logger.info(
            "  progress_t=%s progress_jump=%s local_sonic_cos=%s local_pen=%s genre_pen=%s below_floor=%s",
            _f("progress_t"), _f("progress_jump"),
            _f("local_sonic_raw_cos"),
            _f("local_sonic_penalty_applied"),
            _f("genre_penalty_applied"),
            bool(row.get("below_transition_floor", False)),
        )
```

- [ ] **Step 4: Run the test and verify pass**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_selected_edge_audit.py -q --basetemp .pytest-tmp-edge-audit -o cache_dir=.pytest-tmp-cache-edge-audit
```
Expected: 2 passed.

- [ ] **Step 5: Populate per-edge components from beam search**

In `src/playlist/pier_bridge/beam.py`, locate the `next_beam.append(BeamState(...))` call inside the main scoring loop (~line 843). Just before it, build a per-candidate component dict and attach to the BeamState via a new optional field. First extend `BeamState`:

```python
@dataclass
class BeamState:
    path: List[int]
    score: float
    used: Set[int]
    used_artists: Set[str] = field(default_factory=set)
    last_progress: float = 0.0
    edge_components: List[dict] = field(default_factory=list)
```

Then where the chosen candidate is appended, also append a component dict:

```python
edge_component = {
    "from_idx": int(current),
    "to_idx": int(cand),
    "bridge_score": float(bridge_score),
    "trans_score_in_beam": float(trans_score),
    "progress_t": float(cand_t) if progress_active else None,
    "progress_jump": (float(cand_t) - float(state.last_progress)) if progress_active else None,
    "local_sonic_raw_cos": float(np.dot(X_full_norm[int(current)], X_full_norm[int(cand)])),
    "local_sonic_penalty_applied": float(local_sonic_penalty_applied_value),
    "genre_penalty_applied": float(genre_penalty_applied_value),
    "below_transition_floor": False,
}
new_edge_components = list(state.edge_components) + [edge_component]
```

Pass `edge_components=new_edge_components` to the new BeamState. Capture `local_sonic_penalty_applied_value` and `genre_penalty_applied_value` from the existing penalty branches (set to 0.0 if penalty did not fire on this candidate).

The winning beam state's `edge_components` is returned alongside the path. Wire it through `_beam_search_segment`'s return tuple and up into the per-segment result, then merge across segments into the playlist-level `edge_scores` list that `reporter.py` already consumes.

- [ ] **Step 6: Call the emitter from the playlist generator**

In `src/playlist_generator.py`, where the final playlist stats are emitted (search for `PLAYLIST STATISTICS`), add after the existing weakest-edges block:

```python
if logger.isEnabledFor(logging.INFO) and bool(
    overrides.get("pier_bridge", {}).get("emit_selected_edge_audit", False)
):
    from src.playlist.reporter import emit_selected_edge_audit
    emit_selected_edge_audit(edge_audit_rows)
```

Where `edge_audit_rows` is assembled by zipping `playlist.stats["edge_scores"]` with the per-edge components captured in Step 5 and the track titles/artists from the bundle.

- [ ] **Step 7: Add the opt-in flag to PierBridgeConfig**

In `src/playlist/pier_bridge/config.py`, add near the existing diagnostic flags:

```python
emit_selected_edge_audit: bool = False
"""Diagnostic-only: when True, log a per-edge audit table for the final
emitted playlist showing T, T_centered_cos, S, G, bridge_score,
trans_score_in_beam, progress_t/jump, local_sonic_raw_cos,
local_sonic_penalty_applied, genre_penalty_applied, and below_transition_floor.
No behavior change. Use this to root-cause specific bad transitions."""
```

In `src/playlist/pipeline/pier_bridge_overrides.py`, parse the override (add adjacent to the existing `collapse_segment_pool_by_artist` block):

```python
emit_audit = pb_overrides.get("emit_selected_edge_audit")
if isinstance(emit_audit, bool):
    pb_cfg = replace(pb_cfg, emit_selected_edge_audit=bool(emit_audit))
```

- [ ] **Step 8: Run focused tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_selected_edge_audit.py -q --basetemp .pytest-tmp-edge-audit -o cache_dir=.pytest-tmp-cache-edge-audit
```
Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add src/playlist/reporter.py src/playlist/pier_bridge/beam.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py src/playlist_generator.py tests/unit/test_selected_edge_audit.py
git commit -m "feat: per-edge audit table for final emitted playlist"
```

---

### Task 2: Title-quality flag detection

**Goal:** Pure-function detection of "low-priority" title patterns (live, demo, medley, remix, instrumental, remaster, version, take, mono, stereo, edit, outtake, alternate). Returns flags only; no filtering/penalty in this task — that lands in Task 4.

**Files:**
- Create: `src/playlist/title_quality.py`
- Test: `tests/unit/test_title_quality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_title_quality.py`:

```python
from src.playlist.title_quality import detect_title_artifacts


def test_detect_demo_in_parenthetical():
    flags = detect_title_artifacts("Lee #2 (8 Track Demo)")
    assert flags == {"demo"}


def test_detect_live_at_venue():
    flags = detect_title_artifacts("All Of You (Take 2 / Live At The Village Vanguard / 1961)")
    assert flags == {"live", "take"}


def test_detect_medley():
    flags = detect_title_artifacts("Rubber Ring/What She Said (Medley)")
    assert flags == {"medley"}


def test_detect_remaster_variants():
    assert detect_title_artifacts("Witchcraft (Remastered Stereo 2025)") == {"remaster", "stereo"}
    assert detect_title_artifacts("Some Day My Prince Will Come - Remastered") == {"remaster"}


def test_detect_instrumental():
    flags = detect_title_artifacts("Song Title (Instrumental)")
    assert flags == {"instrumental"}


def test_clean_title_returns_empty_set():
    assert detect_title_artifacts("Library Pictures") == set()
    assert detect_title_artifacts("Soiled Little Filly") == set()


def test_word_boundaries_only():
    # 'demolish' should NOT match 'demo'
    assert detect_title_artifacts("Demolish the Building") == set()
    # 'alternative' should NOT match 'alternate' substring; only 'alternate take' / parenthetical
    assert detect_title_artifacts("Alternative Rock Anthem") == set()


def test_empty_and_none_inputs():
    assert detect_title_artifacts("") == set()
    assert detect_title_artifacts(None) == set()
```

- [ ] **Step 2: Run the failing tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_quality.py -q --basetemp .pytest-tmp-title-quality -o cache_dir=.pytest-tmp-cache-title-quality
```
Expected: ImportError for `src.playlist.title_quality`.

- [ ] **Step 3: Implement `detect_title_artifacts`**

Create `src/playlist/title_quality.py`:

```python
"""Title-artifact flag detection (pure functions, no I/O).

Used to flag tracks whose titles indicate non-canonical recordings:
live, demo, medley, remix, instrumental, remaster, alternate take, etc.

This module ONLY detects flags. Soft penalties and hard exclusions are
applied in the beam/scoring layer based on these flags plus config knobs.
"""
from __future__ import annotations

import re
from typing import Set

# Each entry maps a canonical flag to a list of case-insensitive patterns.
# Patterns use word boundaries to avoid false positives ("demolish" != "demo").
# Patterns target parenthetical/bracketed annotations and stand-alone tokens.
_FLAG_PATTERNS: dict[str, list[str]] = {
    "live": [
        r"\blive\s+(?:at|in|from)\b",
        r"\(live\b",
        r"-\s*live\b",
        r"\[live\b",
    ],
    "demo": [
        r"\bdemo\b",
    ],
    "medley": [
        r"\bmedley\b",
    ],
    "remix": [
        r"\bremix\b",
        r"\bmix\)\s*$",
        r"-\s*[\w\s]+\s+remix\b",
    ],
    "instrumental": [
        r"\binstrumental\b",
    ],
    "remaster": [
        r"\bremaster(?:ed)?\b",
    ],
    "version": [
        r"\bversion\b",
        r"\balternate\s+version\b",
        r"\balt\.\s+version\b",
    ],
    "take": [
        r"\btake\s+\d+\b",
        r"\(take\s+\d+",
    ],
    "mono": [
        r"\bmono\b",
    ],
    "stereo": [
        r"\bstereo\b",
    ],
    "edit": [
        r"\bradio\s+edit\b",
        r"\bsingle\s+edit\b",
        r"\(edit\)",
        r"-\s*edit\b",
    ],
    "outtake": [
        r"\bouttake\b",
    ],
    "alternate": [
        r"\balternate\s+take\b",
        r"\balt\.\s+take\b",
    ],
}


def detect_title_artifacts(title: str | None) -> Set[str]:
    """Return the set of artifact flags present in `title`.

    Detection is case-insensitive with word-boundary matching.
    Returns an empty set for None or empty strings.
    """
    if not title:
        return set()
    text = str(title)
    flags: set[str] = set()
    for flag, patterns in _FLAG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                flags.add(flag)
                break
    return flags
```

- [ ] **Step 4: Run the tests and verify pass**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_quality.py -q --basetemp .pytest-tmp-title-quality -o cache_dir=.pytest-tmp-cache-title-quality
```
Expected: all 8 tests pass.

- [ ] **Step 5: Include flags in the edge audit row (diagnostic only)**

In `src/playlist_generator.py`, when assembling `edge_audit_rows` for Task 1's emitter, populate `to_title_flags` from `detect_title_artifacts(bundle.track_titles[to_idx])`. In `reporter.py`'s `emit_selected_edge_audit`, render flags on the second log line per edge:

```python
flags = row.get("to_title_flags") or set()
flag_str = ",".join(sorted(flags)) if flags else "-"
logger.info(
    "  ... title_flags=%s",
    flag_str,
)
```

(Merge into the existing second log line; do not add a third.)

- [ ] **Step 6: Commit**

```bash
git add src/playlist/title_quality.py tests/unit/test_title_quality.py src/playlist_generator.py src/playlist/reporter.py
git commit -m "feat: title-artifact flag detection + audit integration"
```

---

### Task 3: Root-cause the below-floor T leak

**Goal:** Determine whether the T=0.092 edge that leaked into the final playlist (despite `transition_floor=0.20` and the beam's hard gate at `beam.py:821`) is caused by (a) different transition formulations between beam and reporter, (b) a segment-boundary edge not covered by the gate, or (c) fallback relaxation that wasn't visible in the log.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (add diagnostic log when an edge "would-fail" today's reporter T)
- Modify: `src/playlist/reporter.py` (cross-check beam_trans_score vs final T at emission time)
- Test: `tests/unit/test_beam_vs_final_t_diagnostic.py`

- [ ] **Step 1: Write failing test for the cross-check function**

Create `tests/unit/test_beam_vs_final_t_diagnostic.py`:

```python
import logging

from src.playlist.reporter import diagnose_t_mismatch


def test_diagnose_t_mismatch_flags_disagreement(caplog):
    caplog.set_level(logging.WARNING)
    edges = [
        {"from_idx": 14, "to_idx": 15,
         "T": 0.092, "trans_score_in_beam": 0.25,
         "below_transition_floor": True},
        {"from_idx": 0, "to_idx": 1,
         "T": 0.95, "trans_score_in_beam": 0.94,
         "below_transition_floor": False},
    ]
    issues = diagnose_t_mismatch(edges, transition_floor=0.20, tolerance=0.05)
    assert len(issues) == 1
    assert issues[0]["from_idx"] == 14
    assert "beam_trans=0.250" in caplog.text
    assert "final_T=0.092" in caplog.text


def test_diagnose_t_mismatch_quiet_on_agreement(caplog):
    caplog.set_level(logging.WARNING)
    edges = [
        {"from_idx": 0, "to_idx": 1,
         "T": 0.95, "trans_score_in_beam": 0.94,
         "below_transition_floor": False},
    ]
    issues = diagnose_t_mismatch(edges, transition_floor=0.20, tolerance=0.05)
    assert issues == []
    assert "diagnose_t_mismatch" not in caplog.text.lower() or "beam_trans" not in caplog.text
```

- [ ] **Step 2: Run the failing test**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_beam_vs_final_t_diagnostic.py -q --basetemp .pytest-tmp-t-diag -o cache_dir=.pytest-tmp-cache-t-diag
```
Expected: ImportError for `diagnose_t_mismatch`.

- [ ] **Step 3: Implement `diagnose_t_mismatch` in reporter.py**

Append to `src/playlist/reporter.py`:

```python
def diagnose_t_mismatch(
    edges: list[dict],
    *,
    transition_floor: float,
    tolerance: float = 0.05,
) -> list[dict]:
    """Cross-check beam-scored trans_score vs final-emitted T per edge.

    Returns a list of edges where the disagreement exceeds `tolerance`
    AND the final T is below `transition_floor` (i.e., the beam thought
    the edge was acceptable, but the final reporter scored it as broken).
    Logs a WARNING for each such edge with both scores side by side.
    """
    issues: list[dict] = []
    for e in edges:
        final_t = e.get("T")
        beam_t = e.get("trans_score_in_beam")
        if final_t is None or beam_t is None:
            continue
        try:
            ft = float(final_t)
            bt = float(beam_t)
        except Exception:
            continue
        if ft < float(transition_floor) and (bt - ft) > float(tolerance):
            logger.warning(
                "T-mismatch edge %s->%s: beam_trans=%.3f final_T=%.3f (floor=%.2f)",
                e.get("from_idx"), e.get("to_idx"),
                bt, ft, float(transition_floor),
            )
            issues.append(dict(e))
    return issues
```

- [ ] **Step 4: Wire `diagnose_t_mismatch` into the audit emitter**

In `src/playlist/reporter.py`, inside `emit_selected_edge_audit`, after iterating the rows, call:

```python
diagnose_t_mismatch(
    edge_rows,
    transition_floor=float(transition_floor),
    tolerance=0.05,
)
```

Extend the function signature: `def emit_selected_edge_audit(edge_rows, *, transition_floor: float = 0.20):`. Update the Task 1 test if needed to pass `transition_floor=0.20` explicitly.

- [ ] **Step 5: Run the tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_beam_vs_final_t_diagnostic.py tests/unit/test_selected_edge_audit.py -q --basetemp .pytest-tmp-t-diag -o cache_dir=.pytest-tmp-cache-t-diag
```
Expected: all pass.

- [ ] **Step 6: Run the actual bad playlist with the audit enabled**

Set the GUI's pier_bridge override (or edit `config.yaml` temporarily) to include `emit_selected_edge_audit: true`. Generate the same seeded indie/noise playlist from the user's report. Capture the audit table output. Compare beam_trans vs final T for the Hideous Sun Demon → Stove edge.

Expected outcomes (one of):
- **(a) Formulation mismatch:** beam_trans ≈ 0.50, final T = 0.092 → next plan iteration must align the scoring; either use the same formula in both places, or treat the reporter's T as the gating signal during beam search.
- **(b) Boundary edge:** the bad edge is between a pier and its preceding interior track (pier-adjacent), where the beam's pier-closing path uses `final_trans` but skips the standard gate. The "fix" path then becomes wiring the gate at the pier closure.
- **(c) Fallback relaxation:** an earlier attempt failed and an unlogged relaxation lowered the floor. The fix is to make relaxation visible in logs.

Document the finding in a short markdown note at `docs/run_audits/t-mismatch-2026-05-20.md` (no template required; freeform observations).

- [ ] **Step 7: Commit**

```bash
git add src/playlist/reporter.py tests/unit/test_beam_vs_final_t_diagnostic.py docs/run_audits/t-mismatch-2026-05-20.md
git commit -m "diag: beam vs final T mismatch detector + audit run notes"
```

---

## Phase A Decision Gate

Before continuing to Phase B, the implementer SHOULD run the audit on at least three representative playlists from the user's complaint set (the seeded indie/noise example, the Pains of Being Pure at Heart artist mode, and one new artist-mode run for a narrow-style band). For each, capture:

1. Are bad-feeling transitions correlated with low `local_sonic_raw_cos`, low `bridge_score`, low `trans_score_in_beam`, or none of the above?
2. How often does `local_sonic_penalty_applied` fire, and what is its magnitude vs the combined_score?
3. Are title-artifact flags present on bad-feeling tracks (Sonic Youth demo, take, etc.)?
4. Does `diagnose_t_mismatch` flag the T-leak edge? Which of (a/b/c) is the root cause?

If the data confirms the investigation hypotheses, proceed to Phase B. If it diverges, update this plan before implementing Phase B tasks.

---

## Phase B: Opt-in scoring & filtering (default off, backward compatible)

### Task 4: Soft title-artifact penalty (opt-in)

**Goal:** When enabled, demote candidates whose title contains artifact flags. Hard exclusion stays restricted to the existing `title_exclusion_words` list (`interlude`, `skit`, `acapella`). The new penalty is per-flag-configurable and defaults to off.

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`
- Modify: `src/playlist/pier_bridge/beam.py`
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py`
- Modify: `config.yaml` (documented example, default off)
- Test: `tests/unit/test_title_artifact_penalty.py`

- [ ] **Step 1: Write failing tests for the penalty function**

Create `tests/unit/test_title_artifact_penalty.py`:

```python
from src.playlist.title_quality import compute_title_artifact_penalty


def test_no_flags_no_penalty():
    assert compute_title_artifact_penalty(
        title="Library Pictures",
        weights={"demo": 0.10, "live": 0.05, "remix": 0.08},
    ) == 0.0


def test_single_flag_applies_weight():
    p = compute_title_artifact_penalty(
        title="Lee #2 (8 Track Demo)",
        weights={"demo": 0.10},
    )
    assert abs(p - 0.10) < 1e-9


def test_multiple_flags_sum():
    # 'Live At Venue (Take 2)' triggers both live and take
    p = compute_title_artifact_penalty(
        title="Live At The Village Vanguard (Take 2)",
        weights={"live": 0.05, "take": 0.07},
    )
    assert abs(p - 0.12) < 1e-9


def test_unmapped_flags_ignored():
    # 'remaster' flag fires, but no weight provided → no penalty
    p = compute_title_artifact_penalty(
        title="Some Song (Remastered 2025)",
        weights={"demo": 0.10},
    )
    assert p == 0.0
```

- [ ] **Step 2: Run the failing tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_artifact_penalty.py -q --basetemp .pytest-tmp-title-pen -o cache_dir=.pytest-tmp-cache-title-pen
```
Expected: ImportError for `compute_title_artifact_penalty`.

- [ ] **Step 3: Implement `compute_title_artifact_penalty`**

Append to `src/playlist/title_quality.py`:

```python
def compute_title_artifact_penalty(
    *,
    title: str | None,
    weights: dict[str, float],
) -> float:
    """Sum of weights for each flag detected in the title.

    `weights` maps flag name (e.g., 'demo') to per-flag penalty magnitude.
    Flags detected but not present in `weights` contribute nothing.
    Returns 0.0 for empty/None titles.
    """
    if not title or not weights:
        return 0.0
    flags = detect_title_artifacts(title)
    if not flags:
        return 0.0
    total = 0.0
    for flag in flags:
        w = weights.get(flag)
        if w is None:
            continue
        try:
            total += float(w)
        except Exception:
            continue
    return float(max(0.0, total))
```

- [ ] **Step 4: Add config fields**

In `src/playlist/pier_bridge/config.py`, near the existing penalty fields:

```python
title_artifact_penalty_enabled: bool = False
"""When True, candidates whose title matches artifact flags
(demo/live/medley/remix/instrumental/remaster/version/take/mono/stereo/edit
/outtake/alternate) are demoted by sum of their flag weights. Detection
is from src.playlist.title_quality.detect_title_artifacts."""

title_artifact_penalty_weights: Optional[Dict[str, float]] = None
"""Per-flag penalty magnitudes. None or empty = no penalty. Recommended
starting point for narrow/dynamic modes:
{"demo": 0.10, "live": 0.05, "medley": 0.20, "remix": 0.10,
 "instrumental": 0.08, "remaster": 0.0, "version": 0.05,
 "take": 0.10, "mono": 0.0, "stereo": 0.0, "edit": 0.0,
 "outtake": 0.15, "alternate": 0.10}.
Tune after Phase A diagnostics."""
```

Add `from typing import Dict` if not already imported.

- [ ] **Step 5: Wire the penalty into beam scoring**

In `src/playlist/pier_bridge/beam.py`, inside the scoring loop where `combined_score` is computed (around line 836), after the existing penalty subtractions add:

```python
title_artifact_penalty_value = 0.0
if bool(cfg.title_artifact_penalty_enabled) and cfg.title_artifact_penalty_weights:
    cand_title = ""
    try:
        if bundle is not None and bundle.track_titles is not None:
            cand_title = str(bundle.track_titles[int(cand)] or "")
    except Exception:
        cand_title = ""
    if cand_title:
        from src.playlist.title_quality import compute_title_artifact_penalty
        title_artifact_penalty_value = compute_title_artifact_penalty(
            title=cand_title,
            weights=cfg.title_artifact_penalty_weights,
        )
        combined_score -= title_artifact_penalty_value
```

Record the penalty in `edge_component` (Task 1):
```python
"title_artifact_penalty_applied": float(title_artifact_penalty_value),
```

- [ ] **Step 6: Parse the new override block**

In `src/playlist/pipeline/pier_bridge_overrides.py`, add after the existing penalty parsing:

```python
title_artifact = pb_overrides.get("title_artifact_penalty")
if isinstance(title_artifact, dict):
    enabled = title_artifact.get("enabled")
    weights = title_artifact.get("weights")
    if isinstance(enabled, bool):
        pb_cfg = replace(pb_cfg, title_artifact_penalty_enabled=enabled)
    if isinstance(weights, dict):
        normalized = {
            str(k): float(v) for k, v in weights.items()
            if isinstance(v, (int, float))
        }
        pb_cfg = replace(pb_cfg, title_artifact_penalty_weights=normalized or None)
```

- [ ] **Step 7: Document in config.yaml (commented, default off)**

In `config.yaml`, under the `pier_bridge:` block, add a documented commented example:

```yaml
      # Soft title-artifact penalty (default OFF, opt-in).
      # Demotes candidates whose title matches artifact flags. Hard exclusion
      # is still controlled by candidate_pool.title_exclusion_words.
      # Tune after running Phase A audit (emit_selected_edge_audit: true).
      # title_artifact_penalty:
      #   enabled: true
      #   weights:
      #     demo: 0.10
      #     live: 0.05
      #     medley: 0.20
      #     remix: 0.10
      #     instrumental: 0.08
      #     version: 0.05
      #     take: 0.10
      #     outtake: 0.15
      #     alternate: 0.10
```

- [ ] **Step 8: Run tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_artifact_penalty.py tests/unit/test_title_quality.py -q --basetemp .pytest-tmp-title-pen -o cache_dir=.pytest-tmp-cache-title-pen
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/playlist/title_quality.py src/playlist/pier_bridge/config.py src/playlist/pier_bridge/beam.py src/playlist/pipeline/pier_bridge_overrides.py config.yaml tests/unit/test_title_artifact_penalty.py
git commit -m "feat: opt-in soft title-artifact penalty in beam scoring"
```

---

### Task 5: Scaled local-sonic-edge penalty (opt-in mode, default legacy)

**Goal:** Current penalty math is `strength * (threshold - edge_sonic)` → max possible = `0.30 * 0.10 = 0.03`. This is too small to influence beam choice. Add a `mode: legacy | scaled` flag. In `scaled` mode the penalty becomes `scale * max(0, threshold - edge_sonic)`, so a `scale` of 1.0–3.0 produces 0.05–0.30 penalties — comparable to bridge/transition score deltas.

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`
- Modify: `src/playlist/pier_bridge/beam.py` (the `_apply_local_sonic_edge_policy` helper)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py`
- Test: `tests/unit/test_local_sonic_scaled_mode.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_local_sonic_scaled_mode.py`:

```python
from src.playlist.pier_bridge.beam import _local_sonic_penalty_value


def test_legacy_mode_matches_existing_math():
    # strength=0.30, threshold=0.10, edge_cos=0.03 → penalty = 0.30 * (0.10 - 0.03) = 0.021
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=1.0, mode="legacy",
    )
    assert abs(p - 0.021) < 1e-9


def test_scaled_mode_applies_scale():
    # mode=scaled, scale=2.0 → penalty = 2.0 * (0.10 - 0.03) = 0.14
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=2.0, mode="scaled",
    )
    assert abs(p - 0.14) < 1e-9


def test_above_threshold_no_penalty_both_modes():
    for mode in ("legacy", "scaled"):
        p = _local_sonic_penalty_value(
            edge_cos=0.5, threshold=0.10, strength=0.30, scale=2.0, mode=mode,
        )
        assert p == 0.0


def test_unknown_mode_falls_back_to_legacy():
    p = _local_sonic_penalty_value(
        edge_cos=0.03, threshold=0.10, strength=0.30, scale=5.0, mode="bogus",
    )
    assert abs(p - 0.021) < 1e-9
```

- [ ] **Step 2: Run the failing tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_local_sonic_scaled_mode.py -q --basetemp .pytest-tmp-local-sonic -o cache_dir=.pytest-tmp-cache-local-sonic
```
Expected: ImportError for `_local_sonic_penalty_value`.

- [ ] **Step 3: Extract penalty math into a pure helper**

In `src/playlist/pier_bridge/beam.py`, near the top of the module, add (at module scope, not inside the search function):

```python
def _local_sonic_penalty_value(
    *,
    edge_cos: float,
    threshold: float,
    strength: float,
    scale: float,
    mode: str,
) -> float:
    """Compute the per-edge local-sonic penalty.

    legacy: penalty = strength * max(0, threshold - edge_cos)  (current behavior)
    scaled: penalty = scale    * max(0, threshold - edge_cos)
    Any other mode value falls back to legacy.
    """
    if not (edge_cos < threshold):
        return 0.0
    gap = float(threshold) - float(edge_cos)
    if str(mode).strip().lower() == "scaled":
        return float(max(0.0, float(scale) * gap))
    return float(max(0.0, float(strength) * gap))
```

In `_apply_local_sonic_edge_policy` (already in `beam.py`), replace the inline penalty math with a call to `_local_sonic_penalty_value`, passing `mode=cfg.local_sonic_edge_penalty_mode` and `scale=cfg.local_sonic_edge_penalty_scale`.

- [ ] **Step 4: Add config fields**

In `src/playlist/pier_bridge/config.py`, near the local-sonic fields:

```python
local_sonic_edge_penalty_mode: str = "legacy"
"""'legacy' (default) preserves the existing strength*(threshold-edge_cos)
math, which is small in practice. 'scaled' uses scale*(threshold-edge_cos)
producing penalties large enough to actually influence beam selection.
Tune `local_sonic_edge_penalty_scale` if mode=scaled."""

local_sonic_edge_penalty_scale: float = 1.0
"""Multiplier used in 'scaled' mode. Typical values 1.0-3.0. Ignored in
'legacy' mode."""
```

- [ ] **Step 5: Parse overrides**

In `src/playlist/pipeline/pier_bridge_overrides.py`, near the existing `local_sonic_edge_*` parsing:

```python
local_sonic_mode = pb_overrides.get("local_sonic_edge_penalty_mode")
if isinstance(local_sonic_mode, str) and local_sonic_mode.strip():
    pb_cfg = replace(pb_cfg, local_sonic_edge_penalty_mode=local_sonic_mode.strip())

local_sonic_scale = pb_overrides.get("local_sonic_edge_penalty_scale")
if isinstance(local_sonic_scale, (int, float)):
    pb_cfg = replace(pb_cfg, local_sonic_edge_penalty_scale=float(local_sonic_scale))
```

- [ ] **Step 6: Run the tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_local_sonic_scaled_mode.py -q --basetemp .pytest-tmp-local-sonic -o cache_dir=.pytest-tmp-cache-local-sonic
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/pier_bridge/beam.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/test_local_sonic_scaled_mode.py
git commit -m "feat: opt-in scaled mode for local-sonic-edge penalty"
```

---

### Task 6: Worst-edge lexicographic beam objective (opt-in)

**Goal:** Today's beam picks the path with the highest cumulative score. Two paths can have similar totals but very different minimum-edge quality. When enabled, this task changes the final selection to lexicographic: prefer the path whose worst edge is highest, breaking ties by total score. Implements Layer 1 principle 5 ("the worst edge defines the experience"). Off by default.

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`
- Modify: `src/playlist/pier_bridge/beam.py`
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py`
- Test: `tests/unit/test_min_edge_objective.py`

- [ ] **Step 1: Write failing test for the path comparator**

Create `tests/unit/test_min_edge_objective.py`:

```python
from src.playlist.pier_bridge.beam import _select_best_beam_state


class FakeState:
    def __init__(self, score, edge_scores):
        self.score = score
        self.edge_components = [{"trans_score_in_beam": v} for v in edge_scores]


def test_total_score_objective_picks_higher_total():
    a = FakeState(score=4.0, edge_scores=[1.0, 0.1, 0.9, 2.0])  # min=0.1
    b = FakeState(score=3.5, edge_scores=[0.7, 0.8, 0.9, 1.1])  # min=0.7
    chosen = _select_best_beam_state([a, b], objective="total_score")
    assert chosen is a


def test_min_edge_objective_picks_higher_min():
    a = FakeState(score=4.0, edge_scores=[1.0, 0.1, 0.9, 2.0])  # min=0.1
    b = FakeState(score=3.5, edge_scores=[0.7, 0.8, 0.9, 1.1])  # min=0.7
    chosen = _select_best_beam_state([a, b], objective="min_edge")
    assert chosen is b


def test_min_edge_ties_broken_by_total_score():
    a = FakeState(score=4.0, edge_scores=[0.5, 0.7, 0.9, 1.9])  # min=0.5 total=4.0
    b = FakeState(score=3.0, edge_scores=[0.5, 0.6, 0.8, 1.1])  # min=0.5 total=3.0
    chosen = _select_best_beam_state([a, b], objective="min_edge")
    assert chosen is a


def test_empty_beam_returns_none():
    assert _select_best_beam_state([], objective="min_edge") is None
```

- [ ] **Step 2: Run the failing tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_min_edge_objective.py -q --basetemp .pytest-tmp-min-edge -o cache_dir=.pytest-tmp-cache-min-edge
```
Expected: ImportError for `_select_best_beam_state`.

- [ ] **Step 3: Implement the selector**

In `src/playlist/pier_bridge/beam.py`, add at module scope:

```python
def _select_best_beam_state(states, *, objective: str = "total_score"):
    """Pick the winning beam state from a non-empty list.

    objective='total_score' (default): return max by state.score.
    objective='min_edge': lexicographic (min_edge_score, state.score), preferring
        the state whose worst edge is highest. Ties broken by total score.
    """
    if not states:
        return None
    if str(objective).strip().lower() == "min_edge":
        def _key(s):
            edges = getattr(s, "edge_components", None) or []
            if not edges:
                return (-1e18, float(getattr(s, "score", 0.0)))
            vals = [
                float(e.get("trans_score_in_beam", -1e18))
                for e in edges
                if e is not None
            ]
            min_v = min(vals) if vals else -1e18
            return (min_v, float(getattr(s, "score", 0.0)))
        return max(states, key=_key)
    return max(states, key=lambda s: float(getattr(s, "score", 0.0)))
```

In the existing beam search, replace the final selection step (where the winning beam state is chosen) with:

```python
best = _select_best_beam_state(
    beam,
    objective=str(getattr(cfg, "min_edge_objective", "total_score") or "total_score"),
)
```

- [ ] **Step 4: Add config field**

In `src/playlist/pier_bridge/config.py`:

```python
min_edge_objective: str = "total_score"
"""Beam selection objective:
  'total_score' (default) — pick highest cumulative score (current behavior)
  'min_edge'             — lexicographic (highest min-edge, ties by total)
The 'min_edge' objective optimizes for 'no broken moments' rather than
'good on average' — see Layer 1 principle 5."""
```

- [ ] **Step 5: Parse override**

In `src/playlist/pipeline/pier_bridge_overrides.py`:

```python
min_edge_obj = pb_overrides.get("min_edge_objective")
if isinstance(min_edge_obj, str) and min_edge_obj.strip():
    pb_cfg = replace(pb_cfg, min_edge_objective=min_edge_obj.strip())
```

- [ ] **Step 6: Run tests**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_min_edge_objective.py -q --basetemp .pytest-tmp-min-edge -o cache_dir=.pytest-tmp-cache-min-edge
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/pier_bridge/beam.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/test_min_edge_objective.py
git commit -m "feat: opt-in min-edge lexicographic beam objective"
```

---

## Phase C: Tuning recipe & full validation

### Task 7: Tuning recipe documentation + full test suite

**Files:**
- Create: `docs/PLAYLIST_ORDERING_TUNING.md`
- No code changes; full pytest run

- [ ] **Step 1: Write the tuning recipe**

Create `docs/PLAYLIST_ORDERING_TUNING.md`:

```markdown
# Playlist Ordering Tuning Recipe (2026-05-20)

This document captures recommended starting values for the opt-in ordering
& track-quality knobs added in plan
`2026-05-20-playlist-ordering-and-track-quality.md`.

## When to enable

Symptoms suggesting these knobs may help:
- High-T transitions still feel jarring (texture/density mismatch)
- Tracks like "...(8 Track Demo)", "(Live At ...)", "(Medley)" appear unprompted
- A few catastrophically bad edges (T<0.20) per playlist despite high mean T

## Diagnostic step (always first)

Enable the Phase A audit on one or two representative bad playlists:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true
```

Generate a playlist. The log will include a "Selected-edge audit" section
with per-edge T, T_centered_cos, bridge_score, trans_score_in_beam,
progress_jump, local_sonic_raw_cos, local_sonic_penalty_applied,
genre_penalty_applied, title_flags, and below_floor. Use this to confirm
which knob is most likely to help.

## Knob recipes

### 1. Title-artifact penalty (Task 4)

If the audit shows demos/live/medleys in the playlist:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      title_artifact_penalty:
        enabled: true
        weights:
          demo: 0.10
          live: 0.05
          medley: 0.20
          remix: 0.10
          instrumental: 0.08
          take: 0.10
          outtake: 0.15
          alternate: 0.10
```

Tune individual weights based on observation. Higher = stronger demotion.

### 2. Scaled local-sonic-edge penalty (Task 5)

If the audit shows bad edges with `local_sonic_raw_cos` below 0.10 and
`local_sonic_penalty_applied` is tiny (< 0.03):

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      local_sonic_edge_penalty_enabled: true
      local_sonic_edge_penalty_threshold: 0.15
      local_sonic_edge_penalty_mode: scaled
      local_sonic_edge_penalty_scale: 2.0
```

Verify penalty magnitudes in the audit (`local_sonic_penalty_applied` should
now be 0.05–0.30 on triggering edges).

### 3. Worst-edge objective (Task 6)

If a few catastrophic transitions are dragging down playlists despite high
mean T:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      min_edge_objective: min_edge
```

Compare the resulting playlist's min_transition vs the previous run.
Expected: min_transition rises noticeably; mean_transition may drop
slightly.

## When NOT to enable

- One Each / long narrow-style artist segments (these knobs may starve
  already-tight pools). Re-run after enabling and confirm generation
  still completes. If it doesn't, dial weights down by 50%.
```

- [ ] **Step 2: Run the full test suite**

Run:
```bash
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-full -o cache_dir=.pytest-tmp-cache-full
```
Expected: all tests pass. Investigate any failures before committing.

- [ ] **Step 3: Commit**

```bash
git add docs/PLAYLIST_ORDERING_TUNING.md
git commit -m "docs: playlist ordering & track quality tuning recipe"
```

---

## Self-Review

**Spec coverage:**
- Better selected-edge diagnostics → Task 1 (full per-edge audit), Task 2 (title flags integrated), Task 3 (T-mismatch diagnostic).
- Worst-edge / lexicographic ranking → Task 6.
- Stronger local sonic penalty option → Task 5 (scaled mode).
- Track title quality penalties separate from hard exclusions → Task 4 (soft penalty, hard list unchanged).
- Tests for new behavior → every task includes failing/passing tests; Task 7 adds the full-suite run.
- Backward compatible by default → every new knob defaults to legacy behavior.
- Avoid hard gates that starve segments → Task 5 keeps the existing `local_sonic_edge_floor` (null by default); Task 4 uses soft demotion only.
- Diagnostics-first → Phase A is independently mergeable and produces the data Phase B tuning needs.
- Tunable / configurable → all new behavior is config-driven.

**Placeholder scan:** No "TBD", "TODO", "implement later", or "add appropriate error handling" present. Every step has either exact code blocks or exact commands.

**Type consistency:**
- `detect_title_artifacts(title: str | None) -> Set[str]` is called from `compute_title_artifact_penalty` (Task 4) and from `_compute_title_flags` in Task 1 wiring — same signature both places.
- `_local_sonic_penalty_value(...)` keyword-only signature used identically in test (Task 5 Step 1) and implementation (Step 3).
- `_select_best_beam_state(states, *, objective)` signature consistent between test and implementation.
- `emit_selected_edge_audit(edge_rows, *, transition_floor=0.20)` — final signature includes `transition_floor` after Task 3 Step 4; the Task 1 test should be updated when Task 3 lands (noted in Task 3 Step 4).

**Sequencing check:**
- Phase A tasks can land independently and provide value alone.
- Phase B tasks each touch the same `beam.py` scoring loop — implementer should land them in order (4 → 5 → 6) to avoid merge conflicts on adjacent lines.
- Phase B tasks each depend on Phase A's edge_components plumbing for diagnostics; verify Task 1 Step 5 is merged before Phase B begins.
