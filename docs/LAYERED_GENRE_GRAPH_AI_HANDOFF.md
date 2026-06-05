# Layered Genre Graph AI Handoff

This document is written for the next AI coding session. Continue from the current worktree; do not restart the design or fall back to the old flat genre system.

## Current Workspace

- Main repo: `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3`
- Active worktree: `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.worktrees\layered-genre-graph`
- Branch: `codex/layered-genre-graph`
- Last committed layered substrate commit: `df53796 layered-genre-graph-substrate`
- Production metadata DB: `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db`
- Production AI genre sidecar: `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\ai_genre_enrichment.db`

Important safety rule: do not mutate `data\metadata.db`. Use the production metadata DB read-only. Do not mutate the production sidecar unless the user explicitly asks; use a copied disposable sidecar for smoke tests.

## Non-Negotiable Architecture Decision

When `genre_graph.source=layered`, the Layered Genre Graph is the genre substrate.

That means:

- Do not use the old flat genre pool as an additional genre gate.
- Do not route or score through legacy flat genre vectors in layered mode.
- Broad/family, leaf/style, bridge, and facet vectors must drive genre admission and transition explanation.
- Legacy flat behavior may remain as `legacy` fallback or `layered_shadow` comparison only.

The user explicitly rejected “flat genre plus graph bonus.” Do not reintroduce that.

## What Has Been Accomplished

### Design and Documentation

The following docs already exist and should be used as context:

- `docs\LAYERED_GENRE_GRAPH_SPEC.md`
- `docs\LAYERED_GENRE_TAXONOMY_IMPORT_HARDENING.md`
- `docs\LAYERED_GENRE_FIXTURE_DIAGNOSIS.md`
- `docs\SIDECAR_UNIQUE_GENRE_TERMS.md`

The spec defines the intended model:

- Family/broad genre layer
- Genre/style/leaf layer
- Facet layer
- Bridge layer
- Separate layered vectors
- Explicit parent edges only
- No substring parent inference
- No `pop/rock`
- Standalone `indie` rejected/context-only
- Human rejects override automation

### Reviewed Taxonomy Import

The reviewed layered taxonomy YAML has been integrated as the machine-actionable seed.

Implemented behavior includes:

- Structured taxonomy loading and validation.
- Canonical genres/families/facets/aliases/rejects.
- Explicit parent edges.
- Conditional aliases.
- Machine-readable reject reasons.
- Layered assignment materialization from hybrid genre reports.
- Fixture diagnostics and exact release inspection.

Key commands/features already present:

- `graph-show-release` supports exact `--release-key`.
- `graph-fixture-report` supports fixture evaluation.
- `graph-fixture-report --build-assignments` can materialize layered assignments into a selected sidecar.
- `graph-build-assignments` can materialize layered assignments for discovered releases.

### Artifact Support

Layered artifact support has been added.

Artifacts can now include:

- `X_genre_leaf_idf`
- `X_genre_family`
- `X_genre_bridge`
- `X_facet`
- `genre_leaf_vocab`
- `genre_family_vocab`
- `genre_bridge_vocab`
- `facet_vocab`
- `genre_graph_taxonomy_version`
- `genre_graph_sidecar_fingerprint`

Important: the production sidecar currently has taxonomy tables but empty layered assignment tables. If you build a layered artifact directly from the production sidecar without materializing assignments first, the layered matrices will be empty.

Observed production sidecar state before disposable materialization:

- `genre_graph_canonical_genres`: 30
- `genre_graph_canonical_facets`: 14
- `genre_graph_edges`: 18
- `genre_graph_aliases`: 6
- `genre_graph_release_genre_assignments`: 0
- `genre_graph_release_facet_assignments`: 0

### Layered Mode Wiring

Commit `df53796` made layered mode exclusive rather than additive.

In `genre_graph.source=layered`:

- Candidate pool disables old flat genre gate, dense genre gate, smoothed genre admission, compatibility penalty, and old genre tie-breaks.
- Candidate pool applies layered admission when matrices are valid.
- Candidate diagnostics report `layered_genre_admission`.
- `legacy_flat_genre_gate_applied` is explicitly `False`.
- Pier bridge disables legacy DJ genre routing, genre steering, `weight_genre`, and `dj_pooling_k_genre`.
- Beam scoring disables old flat genre soft penalties/tie-break/waypoints/coverage when layered transition scoring is active and layered matrices exist.
- Layered transition diagnostics are emitted.

### Tests Already Added

There are focused tests for:

- Taxonomy validation.
- Noise policy.
- Hybrid policy.
- Assignment materialization.
- Artifact emission.
- Candidate admission.
- Candidate-pool diagnostics.
- Layered scoring.
- Bridge scoring.
- Bridge diagnostics.
- CLI graph inspection/reporting.
- Pipeline/config preservation of `genre_graph.source`.

Before the latest uncommitted fixes, the following had passed:

- 59 focused layered tests.
- 4 pipeline smoke/golden tests.

## Current Uncommitted Changes

As of this handoff, `git status --short` shows:

```text
 M scripts/ai_genre_enrich.py
 M src/ai_genre_enrichment/layered_vectors.py
 M tests/unit/test_layered_artifact_builder.py
 M tests/unit/test_layered_genre_cli.py
```

These changes are intentional and should be kept.

### 1. Console-Safe `graph-build-assignments` JSON

File: `scripts\ai_genre_enrich.py`

Problem found during a real temp-sidecar materialization run:

- `graph-build-assignments` successfully wrote assignments to the disposable sidecar.
- It then crashed on Windows final JSON printing with `UnicodeEncodeError` because the console was using `cp1252`.

Fix:

- Final JSON print now uses `ensure_ascii=True`.

Test added:

- `tests\unit\test_layered_genre_cli.py::test_graph_build_assignments_output_is_console_encoding_safe`
- It verifies non-ASCII terms are escaped and still parse as JSON.

Verification already run:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_genre_cli.py -q
```

Result:

```text
8 passed in 4.91s
```

### 2. Bridge Vector Shape Fix

File: `src\ai_genre_enrichment\layered_vectors.py`

Problem found during real artifact smoke setup:

- The artifact builder produced:
  - `X_genre_leaf_idf (39957, 34)`
  - `X_genre_bridge (39957, 13)`
- Candidate-pool validation requires bridge and leaf dimensions to align.
- The scorer also expects this alignment because it compares:
  - `seed_bridge` against `candidate_leaf`
  - `candidate_bridge` against `seed_leaf`

Root cause:

- `X_genre_bridge` used only the set of observed bridge affordance targets as its vocabulary.
- That makes bridge a separate dimension space, but the scoring contract treats it as leaf-coordinate affordance.

Fix:

- `X_genre_bridge` now uses the same vocabulary as `X_genre_leaf_idf`.
- `genre_bridge_vocab == genre_leaf_vocab`.
- Bridge values mark which leaf genres a release can validly bridge toward.

Tests updated:

- `tests\unit\test_layered_artifact_builder.py`
- Now asserts:
  - `X_genre_bridge.shape == X_genre_leaf_idf.shape`
  - `genre_bridge_vocab == genre_leaf_vocab`

Verification already run:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_artifact_builder.py tests\unit\test_layered_candidate_admission.py tests\unit\test_layered_genre_scoring.py -q --basetemp=C:\tmp\pytest-layered-bridge-shape
```

Result:

```text
20 passed in 2.46s
```

## Disposable Smoke Artifacts

A disposable sidecar was created from the production sidecar:

```cmd
copy /Y C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\ai_genre_enrichment.db C:\tmp\layered-genre-smoke-sidecar.db
```

Layered assignments were materialized into that temp sidecar. The command wrote rows but previously failed only on final JSON print; that crash is now fixed in the uncommitted CLI change.

After materialization, the temp sidecar contained:

- `genre_graph_canonical_genres`: 76
- `genre_graph_canonical_facets`: 25
- `genre_graph_edges`: 197
- `genre_graph_aliases`: 25
- `genre_graph_release_genre_assignments`: 1604
- `genre_graph_release_facet_assignments`: 51
- `genre_graph_rejected_terms`: 13

Layered shadow artifact path:

```text
C:\tmp\layered-genre-artifacts\shadow\77d30c5f8906f881\data_matrices_step1.npz
```

After the bridge-vector fix and rebuild, inspected matrix shapes were:

```text
X_genre_leaf_idf (39957, 34)
X_genre_family (39957, 17)
X_genre_bridge (39957, 34)
X_facet (39957, 3)
genre_leaf_vocab (34,)
genre_family_vocab (17,)
genre_bridge_vocab (34,)
facet_vocab (3,)
```

That means the artifact is now structurally valid for layered admission.

## Commands That Worked

Use the worktree as cwd:

```cmd
cd C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\.worktrees\layered-genre-graph
```

Use this Python:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe
```

Rebuild the temp layered artifact:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe scripts\ai_genre_enrich.py --metadata-db C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db --sidecar-db C:\tmp\layered-genre-smoke-sidecar.db rebuild-artifacts --artifacts-dir C:\tmp\layered-genre-artifacts --config config.yaml --genre-source layered_shadow --overwrite-shadow
```

Run focused tests for current uncommitted changes:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_artifact_builder.py tests\unit\test_layered_candidate_admission.py tests\unit\test_layered_genre_scoring.py tests\unit\test_layered_genre_cli.py -q --basetemp=C:\tmp\pytest-layered-current
```

Run broader focused layered suite before committing:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_artifact_builder.py tests\unit\test_layered_bridge_diagnostics.py tests\unit\test_layered_bridge_overrides.py tests\unit\test_layered_bridge_scoring.py tests\unit\test_layered_candidate_admission.py tests\unit\test_layered_candidate_pool_diagnostics.py tests\unit\test_layered_diagnostics.py tests\unit\test_layered_genre_assignments.py tests\unit\test_layered_genre_cli.py tests\unit\test_layered_genre_scoring.py tests\unit\test_layered_genre_taxonomy.py tests\unit\test_layered_hybrid_policy.py tests\unit\test_layered_noise_policy.py -q --basetemp=C:\tmp\pytest-layered-full
```

## Current Incomplete Smoke Test

A temporary DS smoke runner was created at:

```text
C:\tmp\run_layered_ds_smoke.py
```

It tried to run:

- Artifact: `C:\tmp\layered-genre-artifacts\shadow\77d30c5f8906f881\data_matrices_step1.npz`
- Seed: Duster - “Me and the Birds”
- Seed track id: `9483fdaee0daab8b6f5f0ce40eb23b1a`
- Mode: `dynamic`
- Length: 20
- `genre_graph.source=layered`
- Audit dir: `C:\tmp\layered-genre-audit`

Status:

- The first attempt failed because the temp script was outside the repo and could not import `src`.
- That was fixed by adding the worktree to `sys.path`.
- The second attempt started but ran for over a minute with no output.
- The user interrupted because usage was nearly exhausted.
- Attempting to send Ctrl-C to the exec session failed because stdin was closed.
- `tasklist` access was denied in sandbox, so this document cannot confirm whether the process is still running.

Next session should not start with another full 20-track pier-bridge smoke. Start smaller and more targeted.

## Update: Direct Candidate-Pool Smoke Passed

After this handoff was first written, a direct candidate-pool smoke was run successfully.

Temp helper:

```text
C:\tmp\run_layered_candidate_pool_smoke.py
```

Command:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe C:\tmp\run_layered_candidate_pool_smoke.py
```

Input:

- Artifact: `C:\tmp\layered-genre-artifacts\shadow\77d30c5f8906f881\data_matrices_step1.npz`
- Seed: Duster - “Me and the Birds”
- Seed track id: `9483fdaee0daab8b6f5f0ce40eb23b1a`
- Mode: `dynamic`
- Candidate pool used `embedding=bundle.X_sonic`
- `genre_graph_source="layered"`
- Legacy genre matrices were not passed as active genre gates.

Result:

```text
X_genre_leaf_idf: (39957, 34)
X_genre_family: (39957, 17)
X_genre_bridge: (39957, 34)
X_facet: (39957, 3)
layered_genre_admission.applied: true
layered_genre_admission.source: layered
legacy_flat_genre_gate_applied: false
input_eligible_count: 24149
admitted_count: 683
rejected_count: 23466
pool_size: 53
distinct_artists: 28
```

Rejection reasons:

```text
unexplained_family_jump: 21782
below_layered_score_threshold: 1684
```

Meaning:

- The corrected layered artifact is structurally valid.
- `genre_graph.source=layered` does apply graph admission on real data.
- The old flat genre gate is not participating in layered candidate admission.
- Diagnostics show readable seed/candidate leaf, family, facet, and bridge terms.
- The next issue is not candidate-pool graph wiring; it is DS/pier-bridge construction/runtime.

Observed smoke sample:

- Seed terms included leaf `indie rock`, `shoegaze`, `slowcore`; family `indie/alternative`, `rock`; bridge affordance `dream pop`.
- First admitted tracks included Duster, Ada Lea, and Bachelor.
- Many rejected samples had no layered assignments, explaining `unexplained_family_jump`.

## Update: Small DS Smoke Still Hung

A bounded small DS smoke was attempted after candidate-pool smoke passed.

Temp helper:

```text
C:\tmp\run_layered_ds_small_smoke.py
```

Intent:

- Length 8, not 20.
- Audit disabled.
- Tiny pier-bridge search settings:
  - `initial_neighbors_m=30`
  - `initial_bridge_helpers=20`
  - `initial_beam_width=8`
  - `segment_pool_max=60`
  - `max_expansion_attempts=1`
  - progress disabled
  - edge repair disabled
- `genre_graph.source=layered`
- Direct `PierBridgeConfig` passed to avoid expensive defaults.

Result:

- The wrapper still hung and did not return after the intended 60-second child timeout.
- The smoke processes were manually stopped.

Interpretation:

- Candidate-pool graph admission is confirmed working.
- This first interpretation was incomplete. A later stack-probe run showed the DS/pier-bridge path itself can complete with the same reduced settings. The hang was likely caused by the multiprocessing smoke wrapper blocking on a large `Queue` payload before the parent drained it.
- Do not use `C:\tmp\run_layered_ds_small_smoke.py` as evidence of DS failure.

## Update: Small DS Stack Probe Completed

A direct stack-probe smoke was run successfully.

Temp helper:

```text
C:\tmp\run_layered_ds_stack_probe.py
```

Command:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe C:\tmp\run_layered_ds_stack_probe.py
```

Result:

```text
ok: true
track_count: 8
```

Important log facts from that run:

```text
Layered genre admission applied: before=3338 after=184 rejected=3154 mode=dynamic
Candidate pool: admitted=36
Pier bridge candidate pool deduped: 184 -> 163 tracks
Pier+Bridge: 1 seeds, target 8 tracks
Pier+Bridge: single-seed arc mode
Segment 0: bridge_floor=0.00 pool_before=60 pool_after=60
Pier+Bridge complete: 8 tracks, 1 segments, 1 successful
Pier bridge result: 8 tracks, 1 segments, success=True
```

Meaning:

- A small real DS generation can complete with `genre_graph.source=layered`.
- Candidate admission used the layered graph.
- Pier-bridge construction completed when search settings were reduced.
- The next step is to turn this into a stable unit/integration smoke or documented manual command, not to keep using the multiprocessing helper.

One later no-timeout summary helper, `C:\tmp\run_layered_ds_summary_smoke.py`, behaved inconsistently and was stopped. Do not treat it as a product failure; prefer the direct stack-probe pattern or build a proper test with small synthetic artifacts.

## Update: Repo-Level Synthetic DS Smoke Test Added

A proper synthetic regression test was added after the stack-probe success.

File:

```text
tests\unit\test_layered_ds_pipeline_smoke.py
```

What it verifies:

- `generate_playlist_ds(...)` can return a playlist with `genre_graph.source=layered`.
- Layered admission is applied.
- `legacy_flat_genre_gate_applied` is false.
- Effective candidate-pool params preserve `genre_graph_source="layered"`.
- Pier-bridge returns success.
- Layered transition diagnostics are enabled and report expected edge count.

The test uses a tiny synthetic artifact and `pace_mode="off"` so it does not depend on production metadata, BPM lookup, or the production sidecar.

Verification:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_ds_pipeline_smoke.py -q --basetemp=C:\tmp\pytest-layered-ds-smoke
```

Result:

```text
1 passed in 1.62s
```

Expanded focused layered suite including this test:

```text
61 passed in 11.62s
```

## Recommended Next Steps

### Step 1: Preserve and Verify Current Uncommitted Fixes

Do not discard the four modified files.

Run:

```cmd
C:\Users\Dylan\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\unit\test_layered_artifact_builder.py tests\unit\test_layered_candidate_admission.py tests\unit\test_layered_genre_scoring.py tests\unit\test_layered_genre_cli.py -q --basetemp=C:\tmp\pytest-layered-current
```

If this passes, run the broader focused layered suite listed above.

Then commit the current fixes with a message like:

```text
fix-layered-artifact-bridge-vectors
```

### Step 2: Do a Direct Candidate-Pool Smoke Before Full Generation

The next useful validation is not a full playlist. It is:

- Load the corrected artifact.
- Pick a seed index for Duster, Stereolab, Mount Eerie, or The Clientele.
- Call `build_candidate_pool(...)` directly with:
  - `genre_graph_source="layered"`
  - all layered matrices and vocabs
  - no legacy genre matrices
- Assert/report:
  - `layered_genre_admission.applied == True`
  - `legacy_flat_genre_gate_applied == False`
  - `admitted_count > 0`
  - rejected samples include readable leaf/family/facet/bridge terms
  - `layered_genre_shadow.enabled == True`

This isolates graph admission from pier-bridge runtime.

### Step 3: Then Do a Very Small DS Smoke

Only after direct candidate-pool smoke works, run a small DS smoke:

- Length 8 or 10, not 20.
- Disable or reduce audit if runtime is too high.
- Use `mode=dynamic`.
- Keep `genre_graph.source=layered`.

Expected checks:

- Playlist generation returns a result.
- Candidate-pool stats include `layered_genre_admission.applied=True`.
- Candidate-pool stats include `legacy_flat_genre_gate_applied=False`.
- Playlist stats include `layered_transition_diagnostics`.
- Beam edge components include layered transition components when applicable.

If this hangs, diagnose pier-bridge runtime separately. Do not assume the graph is wrong until candidate-pool smoke has been confirmed.

### Step 4: If Candidate Coverage Is Too Sparse

The temp sidecar only had 1604 release genre assignments across roughly 39957 tracks. A sparse graph artifact can make layered admission too restrictive.

If candidate-pool smoke admits too few candidates:

- Inspect seed release assignments.
- Inspect which tracks have nonzero leaf/family vectors.
- Confirm release-key mapping from tracks to sidecar assignments.
- Decide whether to materialize more assignments or relax dynamic-mode layered thresholds.
- Do not silently fall back to flat genres.

### Step 5: Only Then Wire User-Facing Testing

Once direct candidate-pool and small DS smoke pass:

- Make sure config can select `genre_graph.source=layered`.
- Make sure artifact rebuild instructions are documented.
- Make sure GUI/worker path passes the source setting through, if needed.
- Run one or two actual playlist generation tests with audit.

## Key Files To Inspect First

- `src\ai_genre_enrichment\layered_vectors.py`
- `src\playlist\candidate_pool.py`
- `src\playlist\layered_genre_scoring.py`
- `src\playlist\pier_bridge\beam.py`
- `src\playlist\layered_bridge_diagnostics.py`
- `src\analyze\artifact_builder.py`
- `scripts\ai_genre_enrich.py`
- `tests\unit\test_layered_artifact_builder.py`
- `tests\unit\test_layered_candidate_admission.py`
- `tests\unit\test_layered_genre_cli.py`

## Known Pitfalls

- The worktree-local `data\metadata.db` may be empty or invalid. Use the main repo DB path explicitly:
  `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3\data\metadata.db`
- Production sidecar currently has no layered assignment rows unless materialized.
- `graph-build-assignments` wrote assignment rows before the old Unicode print crash; the crash was not evidence of DB failure.
- `X_genre_bridge` must stay aligned to leaf vocabulary.
- Do not infer parents from substrings. `dream pop` must not imply broad `pop` unless an explicit curated taxonomy edge says so.
- `pop/rock` must remain rejected and must not canonicalize to `pop rock`, `pop`, or `rock`.
- Standalone `indie` remains rejected/context-only.
- `lo-fi`, `instrumental`, regions, languages, formats, joke tags, and user-list tags must not become leaf genres.

## What Is Still Needed Before Real Playlist Testing

Minimum remaining work:

1. Commit the current four-file fix after tests.
2. Run direct candidate-pool smoke on the corrected layered artifact.
3. Run a small DS smoke with `genre_graph.source=layered`.
4. Diagnose any runtime or sparse-coverage issue from that smoke.
5. If the small smoke passes, wire/document the exact user command path for rebuilding layered artifacts and generating with layered mode.

Do not spend time expanding architecture until the direct smoke and small DS smoke answer the basic question: “Does layered genre admission and transition scoring participate in real playlist generation?”
