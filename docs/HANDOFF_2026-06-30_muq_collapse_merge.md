# HANDOFF — MuQ + collapse-prevention merge to master (code issues)

**Date:** 2026-06-30
**From:** documentation-audit session (read-only reconnaissance)
**To:** the session merging `worktree-sp1-collapse-harness` → `master`
**Purpose:** surface the code issues found while auditing the codebase for a docs rewrite that
are *relevant to landing MuQ + the collapse-prevention system as the new default*. Fold what
applies into the merge so the new default ships fully wired.

---

## ⚠️ Scope & confidence caveat (read first)

This recon read **`master` (`d48bae9`) + the live gitignored `config.yaml`**, **not** the full
`worktree-sp1-collapse-harness` branch (50 commits ahead). Some issues below **may already be
fixed on the branch** — e.g. the branch is known to carry a `fold_muq` script
(`0aed7f3 "ready-to-run fold script for muq_sidecar -> X_sonic_muq"`). Each item is therefore
framed as **"verify against the branch"**, not "this is broken." Treat this as a checklist of
things that must be true *after* the merge, with pointers to where they live on master today.

Current divergence being resolved by the merge:

| | `master` today (old default) | Target (new default) |
|---|---|---|
| Sonic space | MERT (768-d), artifact-declared `X_sonic_variant="mert"` | **MuQ** (512-d, `MuQ-MuLan-large`) |
| Bridge selection | var-bridge / anti_center / roam / edge-repair **off** (dataclass defaults) | **all on** |
| Gen budget | 60 s | **0 = disabled** |

The consumption/calibration side of MuQ is **already on master** (see item 1); the *producing*
side (scan + fold) and the collapse system (SP1/SP2/SP3) are on the branch.

---

## 1. 🔴 MuQ auto-fold footgun (highest priority)

**What must be true after the merge:** a full `analyze` / artifacts rebuild must (re)produce a
**fresh `X_sonic_muq`** whenever MuQ is the active variant — not silently leave it stale.

**Evidence on master (the gap):**
- `scripts/analyze_library.py:2044-2070` auto-folds at the end of the `artifacts` stage but only
  calls `fold_mert()`. `_mert_fold_settings` (`analyze_library.py:128-148`) reads
  `artifacts.sonic_variant_override` (default `'mert'`) purely to decide **what string to stamp
  into `X_sonic_variant`** after that fold.
- master's `scripts/` has `fold_mert_into_artifact.py` and `fold_2dftm_into_artifact.py` but
  **no `fold_muq_into_artifact.py`**.
- Net failure mode with `sonic_variant_override: muq`: a rebuild re-folds **MERT**, stamps
  `X_sonic_variant="muq"`, and leaves `X_sonic_muq` **stale** (from the last hand-run fold).
  The `verify` stage (`analyze_library.py` ~`:2114`) checks `X_sonic_variant` == configured
  variant — which now **matches** (`muq`==`muq`), so it **passes silently**. The staleness is
  invisible.

**Action:** ensure the `artifacts` stage's auto-fold branch on the correct variant and calls
the MuQ fold when `sonic_variant_override: muq` (wire the branch's `fold_muq` into
`analyze_library.py`, mirroring the MERT auto-fold). Consider hardening `verify` to also assert
the *active* variant's matrix was rebuilt this cycle (fingerprint), not just that the stamp
matches — the current stamp-only check is what makes this silent.

**This is the exact "a configured knob that can't act is a startup error, not a silent no-op"
anti-pattern from `CLAUDE.md`, applied to the new default.** Verify the branch already closes it.

## 2. 🟠 Analyze stage-list triple-drift

There are **three** divergent stage lists. Pick one authority and reconcile the others as part
of the merge (the MuQ/collapse work adds stages, so this is the moment):

- **Canonical (use this):** `src/playlist/request_models.py:44-60`
  `ANALYZE_LIBRARY_STAGE_ORDER` — **15 stages**:
  `scan, genres, discogs, lastfm, sonic, mert, adjudicate, apply, publish, genre-sim,
  artifacts, energy, popularity, genre-embedding, verify`. (`enrich` is the *legacy tag-grain*
  path, registered but deliberately excluded — comment at `request_models.py:38-43` warns "a
  past divergence silently left the GUI on `enrich`.")
- `scripts/analyze_library.py` `STAGE_FUNCS` (`:2487-2505`) additionally registers `mbid` and
  `enrich`.
- **Stale — fix:** GUI `web/src/components/ToolsPanel.tsx:7-11` `ALL_STAGES` (13) still lists
  `enrich` and is **missing** `adjudicate`, `apply`, `popularity`. `_clean_stages` drops unknown
  names, then **runs ALL stages if the result is empty** — a silent footgun.

**Action:** align `ToolsPanel` `ALL_STAGES` to `ANALYZE_LIBRARY_STAGE_ORDER`; if MuQ adds a
`muq`-scan stage, add it in all three places. (Note: MuQ embeddings currently ride the `mert`
stage / are folded separately — confirm how the branch stages MuQ extraction and reflect it.)

## 3. 🟠 `config.example.yaml` should ship the new defaults

Per `CLAUDE.md` "activate fixes, never default to legacy," the shipped template should match the
validated live config once MuQ + collapse are the default. Today `config.example.yaml` leaves
`sonic_variant_override` **commented** (`:801-808`) and the selection levers at their **off**
dataclass defaults; your live `config.yaml` runs them on:

| Key | live `config.yaml` | ship in `config.example.yaml`? |
|---|---|---|
| `artifacts.sonic_variant_override` | `muq` (`:456`) | set to `muq` (once fresh clones can build a MuQ artifact — gated on item 1) |
| `pier_bridge.variable_bridge_length` | `true` (`:123`) | `true` |
| `pier_bridge.seed_character_mode` / `_strength` | `anti_center` / `2.0` (`:118-119`) | `anti_center` / `2.0` |
| `pier_bridge.roam.enabled` | `true` (`:129`) | `true` |
| `pier_bridge` edge-repair enabled | `true` (`:194-199`) | `true` |
| `pier_bridge.generation_budget_s` | `0` (`:112`) | decide: `0` (disabled, "quality-first") vs a positive re-arm |

**Caveat:** don't flip `sonic_variant_override: muq` in the *shipped* template until a fresh
clone can actually **produce** a MuQ artifact (item 1) — otherwise a new install hard-errors at
load (missing `X_sonic_muq` key raises: `features/artifacts.py:200-208`).

## 4. 🟡 Minor / independent (fix opportunistically, not merge-blocking)

- **`tools/doctor.py:75-88`** checks Python **≥3.8**; `pyproject.toml:10` requires **≥3.11**. Align.
- **`pace_admission_floor`** — defined (`pier_bridge/config.py:61`), threaded
  (`pipeline/core.py:374`), **never read** in `candidate_pool.py`. Dead knob (harmless at 0.0).
- **`CLAUDE.md` stale gotchas** (fix separately from the merge if you prefer):
  - Genre gotcha names `genre_conflict_min_confidence` / `genre_conflict_penalty_strength` —
    **not shipped keys** (only in CLAUDE.md + a 2026-05-20 design doc). Real keys:
    `candidate_pool.genre_compatibility_enabled/compatible_threshold/conflict_threshold/penalty_strength`
    (`genre_compatibility.py`, `candidate_pool.py:914-931`).
  - "tower blend 162-d (9+57+96)" → live artifact is **163-d** (rhythm PCA dim = **10**,
    `tower_dims=[10,57,96]`); rhythm dim drifts 9↔10 by rebuild (PCA 95% var). Expected, not a bug.
  - "~455 genres" → **465 active canonical nodes / 1010 taxonomy records** (`layered_genre_taxonomy.yaml`
    `taxonomy_version: 0.20.0-gui-20260630`).
  - `--merge-only` is a flag on `scripts/extract_mert_sidecar.py`, **not** `analyze_library.py`.
- Energy sidecar covers **41,043** tracks (memory/older docs say "40,572" — stale, library grew).

---

## 5. Post-merge verification checklist

Prove the new default is *live*, not just configured (read logs, not config):

1. `X_sonic_variant` resolves to `muq` at load and generation runs on the 512-d MuQ matrix
   (log: `Using precomputed sonic variant 'muq' …`; `X_sonic (N, 512)`).
2. Transition calibration picks the MuQ band `(0.594, 0.092)` — already wired at
   `transition_metrics.py:23-59` / `pier_bridge_builder.py:481-496` (keyed off `bundle.sonic_variant`).
   Confirm an unrecognized variant still **raises** (no silent MERT fallback).
3. A full `analyze` rebuild leaves `X_sonic_muq` **fresh** (item 1) and `verify` passes for the
   right reason.
4. Collapse levers fire in a real generation log: variable-bridge flex, `anti_center` penalty,
   roam corridor, edge-repair swaps — each visible, not just present in config.
5. Full suite green; regenerate any sonic/transition goldens against the MuQ space (goldens were
   MERT-calibrated — watch for false-red on the sonic tip, the trap from the 2026-06-26 handoff).

---

## Reference material

Full code-verified current-state digest (all subsystems, `file:line`-cited) is in the audit
session scratchpad `GROUND_TRUTH_DIGEST.md`, and the durable summary is in the
`project_documentation_rewrite` auto-memory. The **documentation rewrite is deliberately paused
until this merge lands** — docs will describe merged `master` as canonical (MuQ + collapse
default; MERT/towers as documented rollbacks).
