# Local genre smoothness recalibration — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate the existing `soft_genre_penalty_threshold` / `_strength` knobs per mode so the in-beam local genre penalty actually fires on edges that drop below typical neighborhood values — suppressing single-track genre detours (e.g., the Violent Femmes case in narrow mode) without changing any production code.

**Architecture:** The mechanism already exists in `src/playlist/pier_bridge/beam.py:1030` — it compares candidate-to-previous-track genre similarity against `genre_penalty_threshold` and applies `combined_score *= (1 - strength)` on miss. The per-mode resolver `_resolve_mode_number_with_source` (`src/playlist/config.py:90-126`) already supports `<key>_<mode>` overrides. This plan adds per-mode YAML keys to `config.example.yaml`, mirrors them into the user's local `config.yaml`, documents the new role of the knob, and runs an iterative calibration loop using existing diagnostics (`soft_genre_penalty_hits`, `soft_genre_penalty_edges_scored`, and the post-generation `G genre` quantile stats).

**Tech Stack:** YAML config, `pytest`, manual playlist generation via `main_app.py` CLI.

---

## File map

- **Modify:** `config.example.yaml` (lines 226-227) — replace flat keys with per-mode variants
- **Modify:** `config.yaml` (gitignored, on user's machine) — same per-mode keys, applied manually
- **Modify:** `docs/PLAYLIST_ORDERING_TUNING.md` — add "Local genre continuity" section
- **Modify:** `tests/test_artist_style.py` — add per-mode resolution smoke test
- **No production code changes.** All wiring exists.

The reference spec lives at `docs/superpowers/specs/2026-05-23-local-genre-smoothness-recalibration-design.md`.

---

### Task 1: Add per-mode keys to `config.example.yaml`

**Files:**
- Modify: `config.example.yaml:226-227`

- [ ] **Step 1: Read the current pier_bridge block**

Run: open `config.example.yaml` and locate lines 222-230 (the `# Small genre tie-breaker` comment through the `genre: tie_break_band: null` block).

- [ ] **Step 2: Replace the flat soft_genre_penalty keys with per-mode variants**

Replace:

```yaml
      # Soft genre whiplash penalty (does NOT gate): if edge_genre < threshold,
      # final_edge_score *= (1 - strength)
      soft_genre_penalty_threshold: 0.20
      soft_genre_penalty_strength: 0.10
```

With:

```yaml
      # Soft genre whiplash penalty (does NOT gate): if edge_genre < threshold,
      # final_edge_score *= (1 - strength). Per-mode so stricter modes can
      # enforce local genre continuity while discover/off keep safety-net
      # behavior only. See docs/PLAYLIST_ORDERING_TUNING.md for tuning.
      soft_genre_penalty_threshold_strict: 0.82
      soft_genre_penalty_threshold_narrow: 0.78
      soft_genre_penalty_threshold_dynamic: 0.55
      soft_genre_penalty_threshold_discover: 0.20
      soft_genre_penalty_threshold_off: 0.20
      soft_genre_penalty_strength_strict: 0.40
      soft_genre_penalty_strength_narrow: 0.30
      soft_genre_penalty_strength_dynamic: 0.15
      soft_genre_penalty_strength_discover: 0.10
      soft_genre_penalty_strength_off: 0.10
```

- [ ] **Step 3: Verify YAML is still valid**

Run: `python -c "import yaml; yaml.safe_load(open('config.example.yaml'))"`
Expected: no output (silent success).

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml
git commit -m "config: per-mode soft_genre_penalty for local genre continuity"
```

---

### Task 2: Smoke test — per-mode resolution flows through

**Files:**
- Modify: `tests/test_artist_style.py`

This test confirms the wiring: when a YAML config sets `soft_genre_penalty_threshold_narrow: 0.78`, the resolved `PierBridgeTuning` for `mode="narrow"` has `genre_penalty_threshold == 0.78`. It does NOT validate playlist quality — that's the manual calibration loop in Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artist_style.py` (or place near the existing `test_soft_genre_penalty_*` tests):

```python
def test_soft_genre_penalty_per_mode_resolution():
    """Per-mode soft_genre_penalty keys override the legacy flat defaults."""
    from src.playlist.config import resolve_pier_bridge_tuning

    overrides = {
        "pier_bridge": {
            "soft_genre_penalty_threshold_strict": 0.82,
            "soft_genre_penalty_threshold_narrow": 0.78,
            "soft_genre_penalty_threshold_dynamic": 0.55,
            "soft_genre_penalty_strength_strict": 0.40,
            "soft_genre_penalty_strength_narrow": 0.30,
            "soft_genre_penalty_strength_dynamic": 0.15,
        }
    }

    for mode, expected_threshold, expected_strength in [
        ("strict", 0.82, 0.40),
        ("narrow", 0.78, 0.30),
        ("dynamic", 0.55, 0.15),
    ]:
        tuning, sources = resolve_pier_bridge_tuning(
            mode=mode, similarity_floor=0.20, overrides=overrides
        )
        assert tuning.genre_penalty_threshold == expected_threshold, (
            f"mode={mode}: expected threshold {expected_threshold}, "
            f"got {tuning.genre_penalty_threshold}"
        )
        assert tuning.genre_penalty_strength == expected_strength, (
            f"mode={mode}: expected strength {expected_strength}, "
            f"got {tuning.genre_penalty_strength}"
        )
        assert sources["genre_penalty_threshold"] == (
            f"pier_bridge.soft_genre_penalty_threshold_{mode}"
        )
        assert sources["genre_penalty_strength"] == (
            f"pier_bridge.soft_genre_penalty_strength_{mode}"
        )

    # Modes without per-mode keys fall back to the flat default (0.20 / 0.10)
    tuning_off, sources_off = resolve_pier_bridge_tuning(
        mode="off", similarity_floor=0.20, overrides=overrides
    )
    assert tuning_off.genre_penalty_threshold == 0.20
    assert tuning_off.genre_penalty_strength == 0.10
    assert sources_off["genre_penalty_threshold"] == "default"
    assert sources_off["genre_penalty_strength"] == "default"
```

- [ ] **Step 2: Run the test, expect it to PASS immediately**

Run: `pytest tests/test_artist_style.py::test_soft_genre_penalty_per_mode_resolution -v`
Expected: PASS. The wiring already exists in `_resolve_mode_number_with_source`; the test confirms it. If it fails, the resolver does not actually support `<key>_<mode>` for these keys and the spec assumption is wrong — STOP and investigate `src/playlist/config.py:90-126` before continuing.

- [ ] **Step 3: Run the full pier-bridge config test module to catch regressions**

Run: `pytest tests/test_artist_style.py -v -k "soft_genre_penalty or pier_bridge"`
Expected: all PASS, including the existing `test_soft_genre_penalty_changes_ranking_without_gating` and `test_soft_genre_penalty_*_clamp` tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_artist_style.py
git commit -m "test: smoke test for per-mode soft_genre_penalty resolution"
```

---

### Task 3: Document the new knob role in `PLAYLIST_ORDERING_TUNING.md`

**Files:**
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md`

- [ ] **Step 1: Locate the right insertion point**

Run: `grep -n "^## " docs/PLAYLIST_ORDERING_TUNING.md` (use the Grep tool, not bash) to list section headers. Insert the new section after the existing "Knob 5" (pace_mode) section, or wherever the user's existing knob numbering ends. Use the next available "Knob N" number.

- [ ] **Step 2: Add the new section**

Insert this block (substitute the actual next knob number for `N`):

```markdown
## Knob N — Local genre continuity (`soft_genre_penalty_*`)

**What it does.** Penalizes any beam edge whose candidate-to-previous-track
genre similarity drops below a per-mode threshold. The penalty multiplies the
edge's combined beam score by `(1 - strength)`, demoting (but not gating)
genre-jarring transitions. This is what suppresses single-track genre detours
like a one-off folk-punk track in the middle of a dream-pop run.

**Where it lives.** `src/playlist/pier_bridge/beam.py:1030` (penalty
application); `src/playlist/config.py:268-276` (per-mode resolution); flat
default `0.20 / 0.10` if no per-mode key is set.

**Per-mode defaults (post-recalibration).** Adjust in `config.yaml`:

| Mode      | threshold | strength | Role                                    |
|-----------|-----------|----------|-----------------------------------------|
| strict    | 0.82      | 0.40     | Hard enforcement of local continuity    |
| narrow    | 0.78      | 0.30     | Suppress single-track detours           |
| dynamic   | 0.55      | 0.15     | Light continuity nudge                  |
| discover  | 0.20      | 0.10     | Safety net only — allow variety         |
| off       | 0.20      | 0.10     | Safety net only — allow variety         |

**How to diagnose.** Run with `--log-level DEBUG` and look for per-segment
`Segment N: soft_genre_penalty_hits=H edges_scored=E threshold=T strength=S`
lines. The post-generation summary also reports total `soft_genre_penalty_hits`.

- If `hits == 0` across all segments in a non-discover mode, the threshold
  is too low to be doing anything — raise it toward the observed `G genre`
  median (look at the `G genre: mean=... p50=...` line in the summary).
- If `hits > 50%` of `edges_scored` in narrow or strict mode, the threshold
  is too high — you're penalizing the median edge, not just outliers. Lower
  toward the `G genre` p25-p33 range.
- If you see bridge relaxation warnings (`Segment N attempt 2: widened=True`)
  appearing in narrow mode after recalibration, the penalty plus the gate
  is starving segments — lower `strength` first, then `threshold`.

**Caveat.** This knob was originally designed as a safety net against
genuine genre conflicts (raw overlap near zero). The recalibration extends
it to continuity enforcement. If you ever need both behaviors at different
thresholds, that's the signal to split into a separate
`local_genre_edge_penalty` mechanism (see brainstorm 2026-05-23 Strategy B).

**Relationship to `genre_tiebreak_weight`.** The tiebreaker (default 0.05)
nudges near-tied edges; the penalty actively demotes below-threshold edges.
They are independent — leave tiebreaker at 0.05 unless you have a specific
reason.
```

- [ ] **Step 3: Spot-check the file renders cleanly**

Run: `grep -n "^## Knob" docs/PLAYLIST_ORDERING_TUNING.md` (use Grep tool) to confirm sequential numbering and no duplicated headers.

- [ ] **Step 4: Commit**

```bash
git add docs/PLAYLIST_ORDERING_TUNING.md
git commit -m "docs: tuning guide for local genre continuity"
```

---

### Task 4: Apply initial values to local `config.yaml` and run calibration baseline

**Files:**
- Modify: `config.yaml` (gitignored, do NOT commit)

This task is interactive — generate playlists, inspect, adjust. The artifact at the end is a set of per-mode values that suppress single-track detours without starving bridges. Those values then get reflected back into `config.example.yaml` in Task 5 if they differ from the initial guesses.

**CLI flag reference.** Mode flag names below (`--genre-mode`, `--sonic-mode`, `--pace-mode`, `--artist`, `--tracks`) match the invocation observed in the user's bug-report log. If a flag is rejected, consult `docs/GOLDEN_COMMANDS.md` for the authoritative current syntax and substitute.

- [ ] **Step 1: Record a pre-recalibration baseline FIRST (before changing any config)**

With your existing `config.yaml` unchanged, run the exact case from the bug report:

```bash
python main_app.py --artist "The Sundays" --tracks 30 --genre-mode narrow --sonic-mode narrow --pace-mode dynamic --log-level INFO > /tmp/baseline_sundays_narrow.log 2>&1
```

On Windows PowerShell, use `$env:TEMP\baseline_sundays_narrow.log` instead of `/tmp/...`.

From the log, extract and write down:
- `G genre: mean=... p10=... p50=... p90=... min=...`
- `min_transition=... mean_transition=...`
- The full TRACKLIST block (eyeball: where are the genre-jarring tracks?)

This is the "before" snapshot. Keep the log file.

- [ ] **Step 2: Add the per-mode keys to local `config.yaml`**

Open `config.yaml` and locate the `playlists.ds_pipeline.pier_bridge` block. If keys `soft_genre_penalty_threshold` or `soft_genre_penalty_strength` exist as flat scalars, replace them with the same per-mode block added to `config.example.yaml` in Task 1. If they don't exist, just append the per-mode block.

Sanity check after editing:

```bash
python -c "import yaml; cfg = yaml.safe_load(open('config.yaml')); pb = cfg['playlists']['ds_pipeline']['pier_bridge']; print({k: pb[k] for k in pb if 'soft_genre_penalty' in k})"
```

Expected: prints the 10 per-mode keys you added with their values.

- [ ] **Step 3: Run the same command with recalibrated values**

Re-run with the recalibrated `config.yaml`:

```bash
python main_app.py --artist "The Sundays" --tracks 30 --genre-mode narrow --sonic-mode narrow --pace-mode dynamic --log-level DEBUG > /tmp/recalibrated_sundays_narrow.log 2>&1
```

From the log, extract:
- `G genre` quantile line (compare to baseline)
- `min_transition` and `mean_transition` (compare to baseline)
- Per-segment `soft_genre_penalty_hits=H edges_scored=E` lines (must include some `H > 0` in narrow mode now)
- Whether any `Segment N attempt 2: widened=True` lines appeared (bridge relaxation — bad)
- Final TRACKLIST — eyeball for single-track genre detours

Record the values.

- [ ] **Step 4: Decide whether to adjust narrow-mode values**

Apply the decision matrix from the tuning guide (Task 3, Step 2):

- `narrow` `soft_genre_penalty_hits` is 0 across all 5 segments → threshold too low → raise `soft_genre_penalty_threshold_narrow` by 0.03, re-run Step 3.
- `narrow` hits exceed 50% of edges_scored OR any segment widened → strength or threshold too aggressive → lower `soft_genre_penalty_strength_narrow` by 0.05 first, re-run. If still bad, also lower threshold by 0.03.
- Tracklist still shows a clear one-track genre detour (subjective) → raise `soft_genre_penalty_strength_narrow` by 0.05, re-run.
- Tracklist looks coherent, no relaxation, hits in single digits per segment → narrow is calibrated. Move on.

Iterate until narrow is calibrated. Document the final narrow values.

- [ ] **Step 5: Repeat for strict mode**

```bash
python main_app.py --artist "The Sundays" --tracks 30 --genre-mode strict --sonic-mode strict --pace-mode strict --log-level DEBUG > /tmp/recalibrated_sundays_strict.log 2>&1
```

Apply the same decision matrix to `soft_genre_penalty_*_strict`. Strict tolerates more aggressive penalties because the playlist is meant to be ultra-cohesive — accept slightly more relaxation risk in exchange for stronger continuity.

- [ ] **Step 6: Sanity-check dynamic mode (should be near-unchanged)**

```bash
python main_app.py --artist "The Sundays" --tracks 30 --genre-mode dynamic --sonic-mode dynamic --pace-mode dynamic --log-level DEBUG > /tmp/recalibrated_sundays_dynamic.log 2>&1
```

Expected: tracklist should look similar to the baseline (light nudge only). If dynamic mode now refuses interesting variety or feels over-pruned, lower `soft_genre_penalty_strength_dynamic` toward 0.10.

- [ ] **Step 7: Spot-check at least one non-artist-mode seed**

Pick a single-track seed (e.g., `--track-title "Here's Where the Story Ends" --artist "The Sundays"`) or a multi-seed call and re-run narrow mode. Confirm the recalibration doesn't break non-artist flows.

```bash
python main_app.py --track-title "Here's Where the Story Ends" --artist "The Sundays" --tracks 30 --genre-mode narrow --sonic-mode narrow --pace-mode dynamic --log-level DEBUG > /tmp/recalibrated_track_seed_narrow.log 2>&1
```

Confirm: playlist completes, no relaxation cascade, tracklist looks coherent.

- [ ] **Step 8: Write down the final per-mode values**

Record the calibrated `soft_genre_penalty_threshold_*` and `soft_genre_penalty_strength_*` values you settled on. These go into Task 5 if they differ from the initial guesses committed in Task 1.

---

### Task 5: Reflect final values back to `config.example.yaml` (if they changed)

**Files:**
- Modify: `config.example.yaml` (only if Task 4 values differ from Task 1 initial guesses)

- [ ] **Step 1: Diff your local `config.yaml` against `config.example.yaml`**

```bash
git diff --no-index config.example.yaml config.yaml | grep -A 2 -B 2 soft_genre_penalty
```

(Will exit non-zero because of other expected diffs; that's fine — read the genre-penalty rows.)

- [ ] **Step 2: If values differ, update `config.example.yaml`**

Edit `config.example.yaml` so the per-mode genre penalty values match your calibrated local values.

- [ ] **Step 3: If you updated PLAYLIST_ORDERING_TUNING.md's defaults table during calibration, update it now too**

The "Per-mode defaults (post-recalibration)" table in `docs/PLAYLIST_ORDERING_TUNING.md` should match `config.example.yaml`. Make them consistent.

- [ ] **Step 4: Re-run the smoke test to ensure no drift**

Run: `pytest tests/test_artist_style.py::test_soft_genre_penalty_per_mode_resolution -v`
Expected: PASS (the test hardcodes 0.82/0.78/0.55 and 0.40/0.30/0.15 — if you changed those during calibration, update the test's `expected_threshold` / `expected_strength` values to match the new defaults).

- [ ] **Step 5: Commit (only if anything actually changed)**

```bash
git status
# If config.example.yaml, PLAYLIST_ORDERING_TUNING.md, or test_artist_style.py changed:
git add config.example.yaml docs/PLAYLIST_ORDERING_TUNING.md tests/test_artist_style.py
git commit -m "config: lock in calibrated soft_genre_penalty values"
```

If `git status` is clean (Task 1 values held up), skip the commit — nothing to do.

---

## Verification checklist

After Task 5, confirm:

- [ ] `git log --oneline` shows commits from Task 1, Task 2, Task 3, and (if Task 5 made changes) Task 5.
- [ ] `pytest tests/test_artist_style.py -k "soft_genre_penalty or pier_bridge" -v` passes.
- [ ] `python tools/doctor.py` is clean.
- [ ] Re-running the bug-report case (`--artist "The Sundays" --tracks 30 --genre-mode narrow --sonic-mode narrow --pace-mode dynamic`) produces a tracklist without the single-track genre detour pattern.
- [ ] `G genre min` in the bug-report case has improved by ≥0.05 compared to the baseline log captured in Task 4 Step 2.
- [ ] No new bridge-relaxation warnings (`Segment N attempt 2: widened=True`) in narrow mode.
- [ ] `min_transition` and `mean_transition` have not dropped by more than 0.02 from baseline.

If any of these fail, return to Task 4 and iterate. If iteration doesn't produce a clean result, the spec's "Risks" section anticipates this: the fallback is Strategy B (a separate `local_genre_edge_penalty` mechanism), which is out of scope for this plan but documented in the spec.
