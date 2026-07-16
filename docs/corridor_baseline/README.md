# Corridor Phase 0a baseline capture

Captures the pre-corridor behavior of the playlist engine on current `master`
so `docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md` has something concrete to
diff against once the corridor rework starts touching pooling code. Everything
here runs faithful, production-policy generations via
`scripts/corridor_baseline/runner.py` — never a hand-built override or a
reimplementation of the slider→config mapping (`tests/support/gui_fidelity.py`
design rule).

## Files in this directory

| File | Produced by | Committed? | Contract category |
|---|---|---|---|
| `transforms_summary.json` | `capture_transforms.py` | yes | A (transforms/identity) |
| `tables/*.json.gz` | `capture_transforms.py` | yes | A — the actual golden input→output tables (`artist_identity`, `title_keys`, `alias_links`, `genre_authority`, `dedup_collapse`); `transforms_summary.json` is their index (name/path/sha256/rows) |
| `corpus_baseline.json` | `capture_corpus.py` | yes | D (topology) + F (reporting presence), 12 cells (6-artist × {home, open}) |
| `knob_sweep.json` | `capture_knob_sweep.py --merge` | yes | B/C/E net — the "no knob goes inert" sweep, plus `dead_outlets` |
| `feature_baseline.json` | `assemble_baseline.py` | yes | Everything above, joined under one `meta` provenance block — **the artifact the contract gate reads** |

Checkpoints and raw generation logs (`logs/corridor_baseline/`,
`logs/corridor_baseline/sweep/`) are **not** committed — `logs/` is
blanket-gitignored. They're the resumable working state behind
`knob_sweep.json`; `corpus_baseline.json`/`transforms_summary.json` point at
their own log paths for provenance but those logs are local-only.

## Re-run commands

All commands run from the repo root with the project's normal Python
environment (`pip install -e .[web,dev]`). Each capture script writes its own
output under `docs/corridor_baseline/` by default; none of them touch
`data/metadata.db` or the sonic artifact (read-only, `file:...?mode=ro` for
the DB).

### 1. Determinism gate (run first — a nondeterministic engine makes the sweep meaningless)

```
python scripts/corridor_baseline/determinism_check.py --artist "Bill Evans Trio" --detent open
```

Runs the same cell twice and diffs `track_ids` + the DS-success `effective`
config blob. Must print `DETERMINISM GATE: PASS` (exit 0) before trusting any
other capture.

### 2. Category A — transforms/identity golden tables

```
python scripts/corridor_baseline/capture_transforms.py
```

Writes `tables/*.json.gz` + the `transforms_summary.json` index. Not
resumable — it's a single pass over the DB and one artifact-bundle load
(~seconds to low minutes depending on library size). Re-run wholesale to
refresh; there is no partial/incremental mode.

### 3. Categories D + F — 12-cell corpus baseline

```
python scripts/corridor_baseline/capture_corpus.py
```

Runs all 6 `CORPUS` artists × `{home, open}` (`runner.DETENTS`) as real
faithful generations and writes `corpus_baseline.json`. Not resumable — it
always runs all 12 cells and overwrites the file. Exits nonzero if any cell
errored or if an always-on `F_PATTERNS` entry was never observed in any cell
(harness bug, not a feature regression — see the module docstring in
`patterns.py`). Per-cell DEBUG logs land under `logs/corridor_baseline/`
(gitignored, kept for later re-extraction without re-generating).

### 4. Categories B/C/E — the "no knob goes inert" sweep

```
# Full sweep, both SWEEP_CELLS ("Bill Evans Trio"/open, "Swirlies"/home):
python scripts/corridor_baseline/capture_knob_sweep.py

# Smoke test — first 5 fields considered, first cell only:
python scripts/corridor_baseline/capture_knob_sweep.py --limit 5

# One field only (debugging a specific knob):
python scripts/corridor_baseline/capture_knob_sweep.py --only-field candidate_pool.similarity_floor

# One cell only:
python scripts/corridor_baseline/capture_knob_sweep.py --cell "Bill Evans Trio:open"

# Merge-only: re-merge whatever checkpoints already exist on disk into
# knob_sweep.json without running any new generations (use after a sweep was
# interrupted mid-run, or to regenerate the merged file after hand-editing a
# checkpoint):
python scripts/corridor_baseline/capture_knob_sweep.py --merge
```

**Resume semantics.** Each `(cell, field)` gets its own checkpoint file under
`logs/corridor_baseline/sweep/<cell_tag>/<sanitized_field>.json` (plus one
`_reference.json` per cell). Re-running the same command skips any field that
already has a checkpoint — killing the process and re-invoking the identical
command resumes exactly where it left off, no flag needed. `--limit N` counts
every field *considered* (whether newly run or skipped because cached)
against the budget, so a resumed run with the same `--limit` and the same
checkpoints on disk performs **zero** new generations — it just re-consumes
the same budget against existing files. This sweep is the long pole (up to
~2,500 single-cell generations if it ever covered all 12 corpus cells instead
of the current 2 `SWEEP_CELLS` — see the Task 6 self-review note on that cost
tradeoff) — expect it to run for hours and to need resuming.

The final step of any sweep invocation (or a bare `--merge`) writes the
committed `knob_sweep.json`: `records` (one per perturbed field per cell,
`status` ∈ `changed`/`inert`/`unmapped`/`skipped_type`/`override_failed`/
`did_not_resolve`/`error`), `status_counts`, `dead_outlets` (fields recorded
`inert` whose `_enabled` sibling flag is `False` in the reference blob), and
`n_checkpoints`.

### 5. Assemble the committed baseline

```
python scripts/corridor_baseline/assemble_baseline.py
```

Reads the three files above (`transforms_summary.json`, `corpus_baseline.json`,
`knob_sweep.json` — all from `docs/corridor_baseline/` by default) plus
`config.yaml`, and writes `docs/corridor_baseline/feature_baseline.json`.
**Fails loudly** (naming the missing file) if any of the three inputs is
absent — it never writes a partial baseline. Every input/output path can be
overridden (`--transforms`, `--corpus`, `--knob-sweep`, `--out`, `--config`)
— used by the unit tests and by anyone re-running against a non-default
capture location; never re-point these at the live `docs/corridor_baseline/`
or `logs/corridor_baseline/` paths from a test.

`feature_baseline.json`'s `meta` block records:
- `captured_on_commit` — `git merge-base HEAD origin/master`: the `master`
  state the baseline actually describes (the branch's own harness commits
  don't affect engine behavior).
- `branch_tip` — `git rev-parse HEAD`: honesty about where the harness that
  produced this baseline actually lived.
- `captured_date` — `--captured-date YYYY-MM-DD` if passed, else today's local
  date at run time (`time.strftime("%Y-%m-%d")`). Not derived from any input
  file's mtime — those can be stale across a resumed multi-day sweep.
- `artifact` — `{path, sha256, size_bytes}` of `playlists.ds_pipeline.artifact_path`
  read directly from `config.yaml` (equivalent to the policy-merged config for
  this key — mode presets never touch `artifact_path`/`database_path` — so
  `assemble_baseline.py` reads the YAML directly rather than importing the
  engine stack just to resolve two path strings), hashed streaming (1 MiB
  chunks) so the ~507MB file is never loaded into memory — hashing takes
  seconds.
- `db` — `{path, track_count}` of `library.database_path` (same direct-read
  rationale), read via a read-only SQLite connection.
- `corpus`, `detents`, `sweep_cells` — the `runner.py` constants, so the
  baseline is self-describing without cross-referencing source.
- `perturbation_rules` — the module docstring of `perturb.py`, read via `ast`
  (no import), so the exact rule text travels with the data it produced.

All paths this script emits use POSIX separators (`Path.as_posix()`), even on
Windows — a Task 3 review flagged the backslashes baked into
`transforms_summary.json`'s table paths as a portability wart; this script
does not repeat it for anything it newly writes (the embedded
`transforms`/`corpus`/`knob_sweep` JSON is passed through verbatim, backslashes
and all, since rewriting committed data out from under other tooling is out of
scope here).

## Dead-outlets triage — `inert` ≠ proven dead

`knob_sweep.json`'s `dead_outlets` (212 entries = every `inert` field) splits in
two: **119 entries carry an `enabling_parent_flag`** (a `*_enabled` sibling is
`False` in the reference blob — e.g. the whole `dj_*` family behind
`dj_bridging_enabled: false`; inert-ness is *explained*), and **93 entries have
`enabling_parent_flag: null`** — no automated explanation. The null group
includes knobs that should obviously bind (`candidate_pool.similarity_floor`,
`max_pool_size`, `initial_beam_width`/`max_beam_width`,
`genre_admission_percentile`, `genre_arc_floor`): for these, `inert` may only
mean the single fixed-magnitude perturbation didn't cross a threshold in these
2 cells (floor below the adaptive percentile that actually binds, pool never
hitting the cap, beam winner unchanged at width+1). **Rule: a `dead_outlets`
entry with `enabling_parent_flag: null` is NOT safe to delete without human
triage** (widen the perturbation, test in a stressed cell, or trace the read
site). Only flag-explained entries — and knobs independently root-caused as
unreachable (see `perturb.py`'s None-map comments) — are demolition-ready.

## Live-DB drift — comparison protocol

`data/metadata.db` is a LIVE database (canonical keeps analyzing/publishing
while baselines exist): it grew 43,036 → 43,088 tracks during the 2026-07-16
capture day alone. The generation universe is pinned by the **artifact** (fixed
`.npz`, sha256 in `meta.artifact`), so corpus/sweep runs are stable against DB
growth — but the Category A tables (`artist_identity`, `genre_authority`, …)
snapshot DB content and WILL drift as tracks are added or genres published.
Therefore: a GREEN check against this baseline must not naively diff a fresh
capture against the committed tables. The protocol is a **paired capture** —
re-run `capture_transforms.py` on pre-change code (current `master`) and on the
corridor branch **the same day**, and diff those two against each other; the
committed 2026-07-16 snapshot is the historical anchor, not the comparison arm.
(The corpus/sweep comparisons need no pairing — artifact-pinned — but re-baseline
them per phase anyway, per the contract.)

## Retirement clause

This harness and baseline exist to serve
`docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md`. When corridor Phase 2 is
merged GREEN and the contract closed, delete `scripts/corridor_baseline/` and
`tests/unit/test_corridor_baseline_*.py`; keep this directory's JSON as the
historical record. Do not wire anything else into this harness.
